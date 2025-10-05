"""
Base prompt loader

The idea is to be able to specify prompts for the various agents in a way that is
changeable and extensible.
"""

import asyncio
import dataclasses
import fnmatch
import json
import traceback
import yaml
import os
import random
import re
import uuid
from contextvars import ContextVar
from typing import Any, Tuple

import jinja2
import nest_asyncio
import structlog

import talemate.instance as instance
import talemate.thematic_generators as thematic_generators
from talemate.config import get_config
from talemate.context import regeneration_context, active_scene
from talemate.emit import emit
from talemate.exceptions import LLMAccuracyError, RenderPromptError
from talemate.util import (
    count_tokens,
    dedupe_string,
    extract_json,
    extract_list,
    fix_faulty_json,
    remove_extra_linebreaks,
    iso8601_diff_to_human,
)
from talemate.util.prompt import condensed, no_chapters
from talemate.agents.context import active_agent

__all__ = [
    "Prompt",
    "LoopedPrompt",
    "register_sectioning_handler",
    "SECTIONING_HANDLERS",
    "DEFAULT_SECTIONING_HANDLER",
    "set_default_sectioning_handler",
]

log = structlog.get_logger("talemate")

prepended_template_dirs = ContextVar("prepended_template_dirs", default=[])


class PydanticJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        """Return the model dump of the object if available, else call the superclass method."""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return super().default(obj)


class PrependTemplateDirectories:
    def __init__(self, prepend_dir: list):
        if isinstance(prepend_dir, str):
            prepend_dir = [prepend_dir]

        self.prepend_dir = prepend_dir

    def __enter__(self):
        self.token = prepended_template_dirs.set(self.prepend_dir)

    def __exit__(self, *args):
        prepended_template_dirs.reset(self.token)


nest_asyncio.apply()

SECTIONING_HANDLERS = {}
DEFAULT_SECTIONING_HANDLER = "titles"


class register_sectioning_handler:
    def __init__(self, name):
        self.name = name

    def __call__(self, func):
        SECTIONING_HANDLERS[self.name] = func
        return func


def set_default_sectioning_handler(name):
    """Sets the default sectioning handler if it exists."""
    if name not in SECTIONING_HANDLERS:
        raise ValueError(
            f"Sectioning handler {name} does not exist. Possible values are {list(SECTIONING_HANDLERS.keys())}"
        )

    global DEFAULT_SECTIONING_HANDLER
    DEFAULT_SECTIONING_HANDLER = name


def validate_line(line):
    """Check if a line is not a comment or end marker."""
    return (
        not line.strip().startswith("//")
        and not line.strip().startswith("/*")
        and not line.strip().startswith("[end of")
    )


def clean_response(response):
    # remove invalid lines
    cleaned = "\n".join(
        [line.rstrip() for line in response.split("\n") if validate_line(line)]
    )

    # find lines containing [end of .*] and remove the match within  the line

    cleaned = re.sub(r"\[end of .*?\]", "", cleaned, flags=re.IGNORECASE)

    return cleaned.strip()


@dataclasses.dataclass
class LoopedPrompt:
    limit: int = 200
    items: list = dataclasses.field(default_factory=list)
    generated: dict = dataclasses.field(default_factory=dict)
    _current_item: str = None
    _current_loop: int = 0
    _initialized: bool = False
    validate_value: callable = lambda k, v: v
    on_update: callable = None

    def __call__(self, item: str):
        if item not in self.items and item not in self.generated:
            self.items.append(item)
        return self.generated.get(item) or ""

    @property
    def render_items(self):
        """Return a string representation of generated items."""
        return "\n".join([f"{key}: {value}" for key, value in self.generated.items()])

    @property
    def next_item(self):
        """Retrieve the next item from the list, skipping generated items."""
        item = self.items.pop(0)
        while self.generated.get(item):
            try:
                item = self.items.pop(0)
            except IndexError:
                return None
        return item

    @property
    def current_item(self):
        try:
            if not self._current_item:
                self._current_item = self.next_item
            elif self.generated.get(self._current_item):
                self._current_item = self.next_item
            return self._current_item
        except IndexError:
            return None

    @property
    def done(self):
        """
        Check if the current loop is complete.
        
        This property checks whether the loop has been initialized and  increments the
        current loop count. If the loop count exceeds  the specified limit, a
        ValueError is raised. It also logs the  current state of the loop, including
        the current item and  generated keys. The function returns True if there are no
        remaining items and the current item is in the generated  dictionary;
        otherwise, it returns False.
        """
        if not self._initialized:
            self._initialized = True
            return False
        self._current_loop += 1
        if self._current_loop > self.limit:
            raise ValueError(f"LoopedPrompt limit reached: {self.limit}")
        log.debug(
            "looped_prompt.done",
            current_item=self.current_item,
            items=self.items,
            keys=list(self.generated.keys()),
        )
        if self.current_item:
            return len(self.items) == 0 and self.generated.get(self.current_item)
        return len(self.items) == 0

    def q(self, item: str):
        log.debug(
            "looped_prompt.q",
            item=item,
            current_item=self.current_item,
            q=self.current_item == item,
        )

        if item not in self.items and item not in self.generated:
            self.items.append(item)
        return item == self.current_item

    def update(self, value):
        """
        Update the current item with a new value.
        
        This method checks if the provided value is valid and updates the current item
        in the generated dictionary. If the value is None or empty, or if there is no
        current item, the method returns early. After updating, it attempts to remove
        the current item from the items list and triggers the on_update callback if it
        is set. Finally, it resets the current item to None.
        """
        if value is None or not value.strip() or self._current_item is None:
            return
        self.generated[self._current_item] = self.validate_value(
            self._current_item, value
        )
        try:
            self.items.remove(self._current_item)
        except ValueError:
            pass

        if self.on_update:
            self.on_update(self._current_item, self.generated[self._current_item])

        self._current_item = None


class JoinableList(list):
    def join(self, separator: str = "\n"):
        """Join elements with a specified separator."""
        return separator.join(self)


@dataclasses.dataclass
class Prompt:
    """
    Base prompt class.
    """

    # unique prompt id {agent_type}-{prompt_name}
    uid: str

    # agent type
    agent_type: str

    # prompt name
    name: str

    # prompt text
    prompt: str = None

    # template text
    template: str | None = None

    # prompt variables
    vars: dict = dataclasses.field(default_factory=dict)

    # pad prepared response and ai response with a white-space
    pad_prepended_response: bool = True

    prepared_response: str = ""

    eval_response: bool = False
    eval_context: dict = dataclasses.field(default_factory=dict)

    # Replace json_response with data_response and data_format_type
    data_response: bool = False
    data_format_type: str = "json"

    client: Any = None

    sectioning_hander: str = dataclasses.field(
        default_factory=lambda: DEFAULT_SECTIONING_HANDLER
    )

    dedupe_enabled: bool = True

    @classmethod
    def get(cls, uid: str, vars: dict = None):
        # split uid into agent_type and prompt_name

        """Retrieve a prompt instance based on the given uid."""
        try:
            agent_type, prompt_name = uid.split(".")
        except ValueError as exc:
            log.debug("prompt.get", uid=uid, error=exc)
            agent_type = ""
            prompt_name = uid

        prompt = cls(
            uid=uid,
            agent_type=agent_type,
            name=prompt_name,
            vars=vars or {},
        )

        return prompt

    @classmethod
    def from_text(cls, text: str, vars: dict = None):
        """Create an instance from the given text and optional variables."""
        return cls(
            uid="",
            agent_type="",
            name="",
            template=text,
            vars=vars or {},
        )

    @classmethod
    async def request(
        cls, uid: str, client: Any, kind: str, vars: dict = None, **kwargs
    ):
        if "decensor" not in vars:
            vars.update(decensor=client.decensor_enabled)
        prompt = cls.get(uid, vars)

        # kwargs update prompt class attributes
        for key, value in kwargs.items():
            setattr(prompt, key, value)

        return await prompt.send(client, kind)

    @property
    def as_list(self):
        """Return the prompt as a list of strings split by newlines."""
        if not self.prompt:
            return ""
        return self.prompt.split("\n")

    @property
    def config(self):
        """Get the configuration, initializing it if necessary."""
        if not hasattr(self, "_config"):
            self._config = get_config()
        return self._config

    def __str__(self):
        return self.render()

    def template_env(self):
        # Get the directory of this file
        """Create a Jinja2 environment with template directories."""
        dir_path = os.path.dirname(os.path.realpath(__file__))

        _prepended_template_dirs = prepended_template_dirs.get() or []

        _fixed_template_dirs = [
            os.path.join(
                dir_path, "..", "..", "..", "templates", "prompts", self.agent_type
            ),
            os.path.join(dir_path, "..", "..", "..", "templates", "prompts", "common"),
            os.path.join(dir_path, "..", "..", "..", "templates", "modules"),
            os.path.join(dir_path, "templates", self.agent_type),
            os.path.join(dir_path, "templates", "common"),
        ]

        template_dirs = _prepended_template_dirs + _fixed_template_dirs

        # Create a jinja2 environment with the appropriate template paths
        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dirs),
        )

    def list_templates(self, search_pattern: str):
        env = self.template_env()
        found = []

        # Ensure the loader is FileSystemLoader
        if isinstance(env.loader, jinja2.FileSystemLoader):
            for search_path in env.loader.searchpath:
                for root, dirs, files in os.walk(search_path):
                    for filename in fnmatch.filter(files, search_pattern):
                        # Compute the relative path to the template directory
                        relpath = os.path.relpath(root, search_path)
                        found.append(os.path.join(relpath, filename))

        return found

    def render(self):
        """
        Render the prompt using jinja2.
        
        This method utilizes the jinja2 library to render a prompt by first creating a
        jinja2 environment  with the necessary template paths. It prepares a context
        dictionary with various global variables  and functions, then loads the
        appropriate template based on the prompt name. Finally, it renders  the
        template with the context variables and handles any potential errors during the
        rendering process.  If a sectioning handler is specified, it applies that
        handler to the rendered prompt. The method  also includes error logging and
        emits system messages in case of rendering failures.
        """
        env = self.template_env()

        ctx = {
            "bot_token": "<|BOT|>",
            "thematic_generator": thematic_generators.ThematicGenerator(),
            "regeneration_context": regeneration_context.get(),
            "active_agent": active_agent.get(),
            "agent_context_state": active_agent.get().state
            if active_agent.get()
            else {},
        }

        env.globals["render_template"] = self.render_template
        env.globals["render_and_request"] = self.render_and_request
        env.globals["debug"] = lambda *a, **kw: log.debug(*a, **kw)
        env.globals["set_prepared_response"] = self.set_prepared_response
        env.globals["set_prepared_response_random"] = self.set_prepared_response_random
        env.globals["set_eval_response"] = self.set_eval_response
        env.globals["set_json_response"] = self.set_json_response
        env.globals["set_data_response"] = self.set_data_response
        env.globals["set_question_eval"] = self.set_question_eval
        env.globals["disable_dedupe"] = self.disable_dedupe
        env.globals["random"] = self.random
        env.globals["random_as_str"] = lambda x, y: str(random.randint(x, y))
        env.globals["random_choice"] = lambda x: random.choice(x)
        env.globals["query_scene"] = self.query_scene
        env.globals["query_memory"] = self.query_memory
        env.globals["query_text"] = self.query_text
        env.globals["query_text_eval"] = self.query_text_eval
        env.globals["instruct_text"] = self.instruct_text
        env.globals["agent_action"] = self.agent_action
        env.globals["agent_config"] = self.agent_config
        env.globals["retrieve_memories"] = self.retrieve_memories
        env.globals["time_diff"] = self.time_diff
        env.globals["uuidgen"] = lambda: str(uuid.uuid4())
        env.globals["to_int"] = lambda x: int(x)
        env.globals["to_str"] = lambda x: str(x)
        env.globals["config"] = self.config
        env.globals["li"] = self.get_bullet_num
        env.globals["len"] = lambda x: len(x)
        env.globals["max"] = lambda x, y: max(x, y)
        env.globals["min"] = lambda x, y: min(x, y)
        env.globals["join"] = lambda x, y: y.join(x)
        env.globals["make_list"] = lambda: JoinableList()
        env.globals["make_dict"] = lambda: {}
        env.globals["join"] = lambda x, y: y.join(x)
        env.globals["data_format_type"] = (
            lambda: getattr(self.client, "data_format", None) or self.data_format_type
        )
        env.globals["count_tokens"] = lambda x: count_tokens(
            dedupe_string(x, debug=False)
        )
        env.globals["print"] = lambda x: print(x)
        env.globals["json"] = lambda x: json.dumps(x, indent=2, cls=PydanticJsonEncoder)
        env.globals["emit_status"] = self.emit_status
        env.globals["emit_system"] = lambda status, message: emit(
            "system", status=status, message=message
        )
        env.globals["llm_can_be_coerced"] = lambda: (
            self.client.can_be_coerced if self.client else False
        )
        env.globals["text_to_chunks"] = self.text_to_chunks
        env.globals["emit_narrator"] = lambda message: emit("system", message=message)
        env.filters["condensed"] = condensed
        env.filters["no_chapters"] = no_chapters
        ctx.update(self.vars)

        if "decensor" not in ctx:
            ctx["decensor"] = False

        # Load the template corresponding to the prompt name
        if not self.template:
            # no template text specified, load from file
            template = env.get_template("{}.jinja2".format(self.name))
        else:
            template = env.from_string(self.template)

        sectioning_handler = SECTIONING_HANDLERS.get(self.sectioning_hander)

        # Render the template with the prompt variables
        self.eval_context = {}
        # self.dedupe_enabled = True
        try:
            self.prompt = template.render(ctx)
            if not sectioning_handler:
                log.warning(
                    "prompt.render",
                    prompt=self.name,
                    warning=f"Sectioning handler `{self.sectioning_hander}` not found",
                )
            else:
                self.prompt = sectioning_handler(self)
        except jinja2.exceptions.TemplateError as e:
            log.error("prompt.render", prompt=self.name, error=traceback.format_exc())
            emit(
                "system",
                status="error",
                message=f"Error rendering prompt `{self.name}`: {e}",
            )
            raise RenderPromptError(f"Error rendering prompt: {e}")

        self.prompt = self.render_second_pass(self.prompt)

        return self.prompt

    def render_second_pass(self, prompt_text: str):

        # replace any {{ and }} as they are not from the scenario content
        # and not meant to be rendered
        """
        Will find all {!{ and }!} occurances replace them with {{ and }} and
        then render the prompt again.
        """
        prompt_text = prompt_text.replace("{{", "__").replace("}}", "__")

        prompt_text = prompt_text.replace("{!{", "{{").replace("}!}", "}}")

        env = self.template_env()
        env.globals["random"] = self.random
        parsed_text = env.from_string(prompt_text).render(self.vars)

        if self.dedupe_enabled:
            parsed_text = dedupe_string(parsed_text, debug=False)

        parsed_text = remove_extra_linebreaks(parsed_text)

        return parsed_text

    def render_template(self, uid, **kwargs) -> "Prompt":
        # copy self.vars and update with kwargs
        """Render a template with updated variables."""
        vars = self.vars.copy()
        vars.update(kwargs)
        return Prompt.get(uid, vars=vars)

    def render_and_request(
        self, prompt: "Prompt", kind: str = "create", dedupe_enabled: bool = True
    ) -> str:
        if not self.client:
            raise ValueError("Prompt has no client set.")

        prompt.dedupe_enabled = dedupe_enabled

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(prompt.send(self.client, kind=kind))

    async def loop(self, client: any, loop_name: str, kind: str = "create"):
        loop = self.vars.get(loop_name)

        while not loop.done:
            result = await self.send(client, kind=kind)
            loop.update(result)

    def get_bullet_num(self):
        """Retrieve and increment the bullet number."""
        _bullet_num = self.vars.get("bullet_num", 1)
        self.vars["bullet_num"] = _bullet_num + 1
        return _bullet_num

    def query_scene(
        self,
        query: str,
        at_the_end: bool = True,
        as_narrative: bool = False,
        as_question_answer: bool = True,
    ):
        """Processes a query and returns a narration or a question-answer format."""
        loop = asyncio.get_event_loop()
        narrator = instance.get_agent("narrator")
        query = query.format(**self.vars)

        if not as_question_answer:
            return loop.run_until_complete(
                narrator.narrate_query(
                    query, at_the_end=at_the_end, as_narrative=as_narrative
                )
            )

        return "\n".join(
            [
                f"Question: {query}",
                "Answer: "
                + loop.run_until_complete(
                    narrator.narrate_query(
                        query, at_the_end=at_the_end, as_narrative=as_narrative
                    )
                ),
            ]
        )

    def query_text(
        self,
        query: str,
        text: str,
        as_question_answer: bool = True,
        short: bool = False,
    ):
        """
        Process a query against the provided text and return the result.
        
        This function formats the query using the instance variables and  processes the
        text to either return a direct answer or a formatted  question-answer pair. It
        utilizes the `world_state` agent to analyze  the text and generate responses
        based on the specified parameters.  The response length can be adjusted based
        on the `short` flag,  allowing for flexibility in the output.
        
        Args:
            query (str): The query string to be processed.
            text (str): The text to analyze against the query.
            as_question_answer (bool?): Flag to determine output format.
                Defaults to True.
            short (bool?): Flag to specify response length. Defaults to False.
        """
        loop = asyncio.get_event_loop()
        world_state = instance.get_agent("world_state")
        query = query.format(**self.vars)

        if isinstance(text, list):
            text = "\n".join(text)

        if not as_question_answer:
            return loop.run_until_complete(
                world_state.analyze_text_and_answer_question(
                    text, query, response_length=10 if short else 512
                )
            )

        return "\n".join(
            [
                f"Question: {query}",
                "Answer: "
                + loop.run_until_complete(
                    world_state.analyze_text_and_answer_question(
                        text, query, response_length=10 if short else 512
                    )
                ),
            ]
        )

    def query_text_eval(self, query: str, text: str):
        """Evaluates a query against the provided text for a yes or no answer."""
        query = f"{query} Answer with a yes or no."
        response = self.query_text(query, text, as_question_answer=False, short=True)
        return response.strip().lower().startswith("y")

    def query_memory(self, query: str, as_question_answer: bool = True, **kwargs):
        """
        Query the memory and return the result.
        
        This function formats the provided query using the instance variables  and
        interacts with the memory agent to retrieve answers. It can return  the result
        as a question-answer format or as a direct answer based on  the
        `as_question_answer` flag. If the `iterate` keyword argument is  set, it will
        perform a multi-query operation for each line in the  query string.
        
        Args:
            query (str): The query string to be processed.
            as_question_answer (bool?): Flag to determine the output
                format. Defaults to True.
            **kwargs: Additional keyword arguments passed to the memory query.
        """
        loop = asyncio.get_event_loop()
        memory = instance.get_agent("memory")
        query = query.format(**self.vars)

        if not kwargs.get("iterate"):
            if not as_question_answer:
                return loop.run_until_complete(memory.query(query, **kwargs))

            answer = loop.run_until_complete(memory.query(query, **kwargs))

            return "\n".join(
                [
                    f"Question: {query}",
                    f"Answer: {answer if answer else 'Unknown'}",
                ]
            )
        else:
            return loop.run_until_complete(
                memory.multi_query(
                    [q for q in query.split("\n") if q.strip()], **kwargs
                )
            )

    def instruct_text(self, instruction: str, text: str, as_list: bool = False):
        """
        Processes an instruction with the given text and returns the response.
        
        Args:
            instruction (str): The instruction to be processed.
            text (str): The text to analyze.
            as_list (bool?): If True, return the response as a list. Defaults to False.
        """
        loop = asyncio.get_event_loop()
        world_state = instance.get_agent("world_state")
        instruction = instruction.format(**self.vars)

        if isinstance(text, list):
            text = "\n".join(text)

        response = loop.run_until_complete(
            world_state.analyze_and_follow_instruction(text, instruction)
        )

        if as_list:
            return extract_list(response)
        else:
            return response

    def retrieve_memories(self, lines: list[str], goal: str = None):
        """Retrieve memories by analyzing provided text lines."""
        loop = asyncio.get_event_loop()
        world_state = instance.get_agent("world_state")

        lines = [str(line) for line in lines]

        return loop.run_until_complete(
            world_state.analyze_text_and_extract_context("\n".join(lines), goal=goal)
        )

    def agent_config(self, config_path: str):
        try:
            agent_name, action_name, config_name = config_path.split(".")
            agent = instance.get_agent(agent_name)
            return agent.actions[action_name].config.get(config_name).value
        except Exception as e:
            log.error("agent_config", config_path=config_path, error=e)
            return ""

    def agent_action(self, agent_name: str, _action_name: str, **kwargs):
        """Executes an action for a specified agent."""
        loop = asyncio.get_event_loop()
        agent = instance.get_agent(agent_name)
        action = getattr(agent, _action_name)
        return loop.run_until_complete(action(**kwargs))

    def emit_status(self, status: str, message: str, **kwargs):
        if kwargs:
            emit("status", status=status, message=message, data=kwargs)
        else:
            emit("status", status=status, message=message)

    def time_diff(self, iso8601_time: str):
        """Calculate the time difference from the given ISO 8601 time."""
        scene = active_scene.get()
        if not iso8601_time:
            return ""
        return iso8601_diff_to_human(iso8601_time, scene.ts)

    def text_to_chunks(self, text: str, chunk_size: int = 512) -> list[str]:
        """
        Takes a text string and splits it into chunks based length of the text.

        Arguments:

        - text: The text to split into chunks.
        - chunk_size: number of characters in each chunk.
        """

        chunks = []

        for i, line in enumerate(text.split("\n")):
            # dont push empty lines into empty chunks
            if not line.strip() and (not chunks or not chunks[-1]):
                continue

            if not chunks:
                chunks.append([line])
                continue

            if len("\n".join(chunks[-1])) + len(line) < chunk_size:
                chunks[-1].append(line)
            else:
                chunks.append([line])

        return ["\n\n".join(chunk) for chunk in chunks]

    def set_prepared_response(self, response: str, prepend: str = ""):
        """
        Set the prepared response.

        Args:
            response (str): The prepared response.
        """
        self.prepared_response = response
        return f"<|BOT|>{prepend}{response}"

    def set_prepared_response_random(self, responses: list[str], prefix: str = ""):
        """
        Set the prepared response from a list of responses using random.choice

        Args:

            responses (list[str]): A list of responses.
        """
        response = random.choice(responses)
        return self.set_prepared_response(f"{prefix}{response}")

    def set_eval_response(self, empty: str = None):
        """
        Set the evaluation response and update counters if provided.
        
        Args:
            empty (str?): The key to update in the counters.
        """
        if empty:
            self.eval_context.setdefault("counters", {})[empty] = 0

        self.eval_response = True
        return self.set_json_response(
            {"answers": [""]},
            instruction='schema: {"answers": [ {"question": "question?", "answer": "yes", "reasoning": "your reasoning"}, ...]}',
        )

    def set_data_response(
        self, initial_object: dict, instruction: str = "", cutoff: int = 3
    ):
        # Always use client data format if available
        """Prepare a data response in the client's preferred format (YAML or JSON).
        
        This function determines the appropriate data format based on the client's
        settings and serializes the provided `initial_object` accordingly. If the
        format is YAML, it handles specific cases for lists and nested dictionaries to
        ensure the response is concise. For JSON, it applies a similar trimming logic
        and includes optional instructions as comments.
        
        Args:
            initial_object (dict): The data structure to serialize.
            instruction (str): Optional instruction/schema comment.
            cutoff (int): Number of lines to trim from the end.
        
        Returns:
            str: The prepared data response in the specified format.
        
        Raises:
            ImportError: If PyYAML is required for YAML support but not installed.
        """
        data_format_type = (
            getattr(self.client, "data_format", None) or self.data_format_type
        )

        self.data_format_type = data_format_type
        self.data_response = True

        if data_format_type == "yaml":
            if yaml is None:
                raise ImportError(
                    "PyYAML is required for YAML support. Please install it with 'pip install pyyaml'."
                )

            # Serialize to YAML
            prepared_response = yaml.safe_dump(initial_object, sort_keys=False).split(
                "\n"
            )

            # For list structures, ensure we stop after the key with a colon
            if isinstance(initial_object, dict) and any(
                isinstance(v, list) for v in initial_object.values()
            ):
                # Find the first key that has a list value and stop there
                for i, line in enumerate(prepared_response):
                    if line.strip().endswith(":"):  # Found a key that might have a list
                        # Look ahead to see if next line has a dash (indicating it's a list)
                        if i + 1 < len(prepared_response) and prepared_response[
                            i + 1
                        ].strip().startswith("- "):
                            # Keep only up to the key with colon, drop the list items
                            prepared_response = prepared_response[: i + 1]
                            break
            # For nested dictionary structures, keep only the top-level keys
            elif isinstance(initial_object, dict) and any(
                isinstance(v, dict) for v in initial_object.values()
            ):
                # Find keys that have dictionary values
                for i, line in enumerate(prepared_response):
                    if line.strip().endswith(
                        ":"
                    ):  # Found a key that might have a nested dict
                        # Look ahead to see if next line is indented (indicating nested structure)
                        if i + 1 < len(prepared_response) and prepared_response[
                            i + 1
                        ].startswith("  "):
                            # Keep only up to the key with colon, drop the nested content
                            prepared_response = prepared_response[: i + 1]
                            break
            elif cutoff > 0:
                # For other structures, just remove last lines
                prepared_response = prepared_response[:-cutoff]

            if instruction:
                prepared_response.insert(0, f"# {instruction}")

            cleaned = "\n".join(prepared_response)
            # Wrap in markdown code block for YAML, but do not close the code block
            # Add an extra newline to ensure the model's response starts on a new line
            return self.set_prepared_response(f"```yaml\n{cleaned}")
        else:
            # Use existing JSON logic
            prepared_response = json.dumps(initial_object, indent=2).split("\n")
            prepared_response = ["".join(prepared_response[:-cutoff])]
            if instruction:
                prepared_response.insert(0, f"// {instruction}")
            cleaned = "\n".join(prepared_response)
            # remove all duplicate whitespace
            cleaned = re.sub(r"\s+", " ", cleaned)
            return self.set_prepared_response(cleaned)

    def set_json_response(
        self, initial_object: dict, instruction: str = "", cutoff: int = 3
    ):
        """
        Prepares for a json response
        """
        self.data_format_type = "json"
        return self.set_data_response(
            initial_object, instruction=instruction, cutoff=cutoff
        )

    def set_question_eval(
        self, question: str, trigger: str, counter: str, weight: float = 1.0
    ):
        """Sets a question evaluation in the context."""
        self.eval_context.setdefault("questions", [])
        self.eval_context.setdefault("counters", {})[counter] = 0
        self.eval_context["questions"].append((question, trigger, counter, weight))

        num_questions = len(self.eval_context["questions"])
        return f"{num_questions}. {question}"

    def disable_dedupe(self):
        """Disable deduplication functionality."""
        self.dedupe_enabled = False
        return ""

    def random(self, min: int, max: int):
        return random.randint(min, max)

    async def parse_yaml_response(self, response):
        """
        Parse a YAML response from the LLM.
        
        This function extracts YAML content from a given response string, which may
        contain markdown code blocks. It checks for the presence of YAML code blocks
        and handles various scenarios, including incomplete blocks. If the YAML parsing
        fails, it logs the error and raises an LLMAccuracyError with relevant details.
        """
        if yaml is None:
            raise ImportError(
                "PyYAML is required for YAML support. Please install it with 'pip install pyyaml'."
            )

        # Extract YAML from markdown code blocks
        if "```yaml" in response and "```" in response.split("```yaml", 1)[1]:
            yaml_block = response.split("```yaml", 1)[1].split("```", 1)[0]
        # Starts with ```yaml but has not ``` at the end
        elif "```yaml" in response and "```" not in response.split("```yaml", 1)[1]:
            yaml_block = response.split("```yaml", 1)[1]
        elif "```" in response:
            # Try any code block as fallback
            yaml_block = response.split("```", 1)[1].split("```", 1)[0]
        else:
            yaml_block = response

        try:
            return yaml.safe_load(yaml_block)
        except Exception as e:
            log.error("parse_yaml_response", response=response, error=e)
            raise LLMAccuracyError(
                f"{self.name} - Error parsing YAML response: {e}",
                model_name=self.client.model_name if self.client else "unknown",
            )

    async def parse_data_response(self, response):
        # If json_response is True for backward compatibility, default to JSON
        """Parse response based on the configured data format."""
        if self.data_format_type == "json":
            return await self.parse_json_response(response)
        elif self.data_format_type == "yaml":
            return await self.parse_yaml_response(response)
        else:
            raise ValueError(f"Unsupported data format: {self.data_format_type}")

    async def parse_json_response(self, response, ai_fix: bool = True):
        # strip comments
        """
        Parse a JSON response, attempting to fix errors if necessary.
        
        This function processes a JSON response by first stripping comments and
        handling specific formatting cases. It attempts to decode the JSON and, if that
        fails, it sanitizes the response and tries to fix any faulty JSON structure. If
        the initial parsing fails and AI fixing is enabled, it sends the response to an
        AI client for correction. The function logs various stages of processing and
        raises an LLMAccuracyError if parsing ultimately fails.
        
        Args:
            response (str): The JSON response string to be parsed.
            ai_fix (bool): A flag indicating whether to use AI to fix parsing errors.
        
        Returns:
            dict: The parsed JSON response.
        
        Raises:
            LLMAccuracyError: If the JSON response cannot be parsed after attempts to fix it.
        """
        try:
            # if response starts with ```json and ends with ```
            # then remove those
            if response.startswith("```json") and response.endswith("```"):
                response = response[7:-3]

            try:
                response = json.loads(response)
                return response
            except json.decoder.JSONDecodeError:
                pass
            response = response.replace("True", "true").replace("False", "false")
            response = "\n".join(
                [line for line in response.split("\n") if validate_line(line)]
            ).strip()

            response = fix_faulty_json(response)
            response, json_response = extract_json(response)
            log.debug(
                "parse_json_response ", response=response, json_response=json_response
            )
            return json_response
        except Exception as e:
            # JSON parsing failed, try to fix it via AI

            if self.client and ai_fix:
                log.warning(
                    "parse_json_response error on first attempt - sending to AI to fix",
                    response=response,
                    error=e,
                )
                fixed_response = await self.client.send_prompt(
                    f"fix the syntax errors in this JSON string, but keep the structure as is. Remove any comments.\n\nError:{e}\n\n```json\n{response}\n```<|BOT|>"
                    + "{",
                    kind="analyze_long",
                )
                log.debug(
                    "parse_json_response error on first attempt - sent to AI to fix",
                    fixed_response=fixed_response,
                )
                try:
                    fixed_response = "{" + fixed_response
                    return json.loads(fixed_response)
                except Exception as e:
                    log.error(
                        "parse_json_response error on second attempt",
                        response=fixed_response,
                        error=e,
                    )
                    raise LLMAccuracyError(
                        f"{self.name} - Error parsing JSON response: {e}",
                        model_name=self.client.model_name,
                    )

            else:
                log.error("parse_json_response", response=response, error=e)
                raise LLMAccuracyError(
                    f"{self.name} - Error parsing JSON response: {e}",
                    model_name=self.client.model_name,
                )

    async def evaluate(self, response: str) -> Tuple[str, dict]:
        """
        async def evaluate(self, response: str) -> Tuple[str, dict]:
        Evaluate the response against predefined questions.  This function processes
        the provided response by parsing it as JSON  and comparing the answers to the
        expected questions. It checks for  consistency in the number of questions and
        answers, collects the  answers while handling potential errors, and evaluates
        them based  on specific criteria defined in the evaluation context. The results
        are logged and returned along with updated counters.
        
        Args:
            response (str): The JSON response string to evaluate."""
        questions = self.eval_context["questions"]
        log.debug("evaluate", response=response)

        try:
            parsed_response = await self.parse_json_response(response)
            answers = parsed_response["answers"]
        except Exception as e:
            log.error("evaluate", response=response, error=e)
            raise LLMAccuracyError(
                f"{self.name} - Error parsing JSON response: {e}",
                model_name=self.client.model_name,
            )

        # if questions and answers are not the same length, raise an error
        if len(questions) != len(answers):
            log.error(
                "evaluate", response=response, questions=questions, answers=answers
            )
            raise LLMAccuracyError(
                f"{self.name} - Number of questions ({len(questions)}) does not match number of answers ({len(answers)})",
                model_name=self.client.model_name,
            )

        # collect answers
        try:
            answers = [
                (answer["answer"] + ", " + answer.get("reasoning", ""))
                .strip("")
                .strip(",")
                for answer in answers
            ]
        except KeyError as e:
            log.error("evaluate", response=response, error=e)
            raise LLMAccuracyError(
                f"{self.name} - expected `answer` key missing: {e}",
                model_name=self.client.model_name,
            )

        # evaluate answers against questions and tally up the counts for each counter
        # by checking if the lowercase string starts with the trigger word

        questions_and_answers = zip(self.eval_context["questions"], answers)
        response = []
        for (question, trigger, counter, weight), answer in questions_and_answers:
            log.debug(
                "evaluating",
                question=question,
                trigger=trigger,
                counter=counter,
                weight=weight,
                answer=answer,
            )
            if answer.lower().startswith(trigger):
                self.eval_context["counters"][counter] += weight
            response.append(
                f"Question: {question}\nAnswer: {answer}",
            )

        log.debug("eval_context", **self.eval_context)

        return "\n".join(response), self.eval_context.get("counters")

    async def send(self, client: Any, kind: str = "create"):
        """
        Send the prompt to the client and handle the response.
        
        This asynchronous function sends a prompt to the specified client and processes
        the response based on the expected data format. It checks if the response needs
        to be prepended with a prepared response and handles different formats such as
        JSON and YAML. The function also evaluates the response if required and logs
        the data response for debugging purposes.
        
        Args:
            client (Any): The client to send the prompt to.
            kind (str): The kind of prompt to send.
        
        Returns:
            Any: The processed response from the client, which may vary based on the evaluation
                and data response settings.
        """
        self.client = client

        response = await client.send_prompt(
            str(self), kind=kind, data_expected=self.data_response
        )

        # Handle prepared response prepending based on response format
        if not self.data_response:
            # not awaiting a structured response
            if not response.lower().startswith(self.prepared_response.lower()):
                pad = " " if self.pad_prepended_response else ""
                response = self.prepared_response.rstrip() + pad + response.strip()
        else:
            format_type = (
                getattr(self.client, "data_format", None) or self.data_format_type
            )

            json_start = response.lstrip().startswith("{")
            yaml_block = "```yaml" in response
            json_block = "```json" in response

            if format_type == "json" and json_block:
                response = response.split("```json", 1)[1].split("```", 1)[0]
            elif format_type == "yaml" and yaml_block:
                response = response.split("```yaml", 1)[1].split("```", 1)[0].strip()
            else:
                # If response doesn't start with expected format markers, prepend the prepared response
                if (format_type == "json" and not json_start) or (
                    format_type == "yaml" and not yaml_block
                ):
                    pad = " " if self.pad_prepended_response else ""
                    if format_type == "yaml":
                        if self.client.can_be_coerced:
                            response = self.prepared_response + response.rstrip()
                        else:
                            response = (
                                self.prepared_response.rstrip()
                                + "\n  "
                                + response.rstrip()
                            )
                    else:
                        response = (
                            self.prepared_response.rstrip() + pad + response.strip()
                        )

        if self.eval_response:
            return await self.evaluate(response)

        if self.data_response:
            log.debug(
                "data_response", format_type=self.data_format_type, response=response
            )
            return response, await self.parse_data_response(response)

        response = clean_response(response)

        return response

    def poplines(self, num):
        """Pop the first n lines from the prompt.
        
        Args:
            num (int): The number of lines to pop.
        """
        lines = self.as_list[:-num]
        self.prompt = "\n".join(lines)

    def cleaned(self, as_list: bool = False):
        """
        Clean the prompt.
        """
        cleaned = []

        for line in self.as_list:
            if "<|BOT|>" in line:
                cleaned.append(line.split("<|BOT|>")[0])
                break
            cleaned.append(line)

        if as_list:
            return cleaned
        return "\n".join(cleaned)


def _prompt_sectioning(
    prompt: Prompt,
    handle_open: callable,
    handle_close: callable,
    strip_empty_lines: bool = False,
) -> str:
    """
    Will loop through the prompt lines and find <|SECTION:{NAME}|> and <|CLOSE_SECTION|> tags
    and replace them with section tags according to the handle_open and handle_close functions.

    Arguments:
        prompt (Prompt): The prompt to section.
        handle_open (callable): A function that takes the section name as an argument and returns the opening tag.
        handle_close (callable): A function that takes the section name as an argument and returns the closing tag.
        strip_empty_lines (bool): Whether to strip empty lines after opening and before closing tags.
    """

    # loop through the prompt lines and find <|SECTION:{NAME}|> tags
    # keep track of currently open sections and close them when a new one is found
    #
    # sections are either closed by a <|CLOSE_SECTION|> tag or a new <|SECTION:{NAME}|> tag

    lines = prompt.as_list

    section_name = None

    new_lines = []
    at_beginning_of_section = False

    def _handle_strip_empty_lines_on_close():
        if not strip_empty_lines:
            return
        while new_lines[-1] == "":
            new_lines.pop()

    for line in lines:
        if "<|SECTION:" in line:
            if not handle_open:
                continue

            if section_name and handle_close:
                if at_beginning_of_section:
                    new_lines.pop()
                else:
                    _handle_strip_empty_lines_on_close()
                    new_lines.append(handle_close(section_name))
                    new_lines.append("")

            section_name = line.split("<|SECTION:")[1].split("|>")[0].lower()
            new_lines.append(handle_open(section_name))
            at_beginning_of_section = True
            continue

        if "<|CLOSE_SECTION|>" in line and section_name:
            if at_beginning_of_section:
                section_name = None
                new_lines.pop()
                continue

            if not handle_close:
                section_name = None
                continue
            _handle_strip_empty_lines_on_close()
            new_lines.append(handle_close(section_name))
            section_name = None
            continue
        elif "<|CLOSE_SECTION|>" in line and not section_name:
            continue

        if line == "" and strip_empty_lines and at_beginning_of_section:
            continue

        at_beginning_of_section = False

        new_lines.append(line)

    return "\n".join(new_lines)


@register_sectioning_handler("bracket")
def bracket_prompt_sectioning(prompt: Prompt) -> str:

    """Replaces <|SECTION:{NAME}|> and <|CLOSE_SECTION|> tags with bracketed sections."""
    return _prompt_sectioning(
        prompt,
        lambda section_name: f"[{section_name}]",
        lambda section_name: f"[end of {section_name}]",
        strip_empty_lines=True,
    )


@register_sectioning_handler("none")
def none_prompt_sectioning(prompt: Prompt) -> str:
    """Handles sectioning for a 'none' prompt."""
    return _prompt_sectioning(
        prompt,
        None,
        None,
    )


@register_sectioning_handler("titles")
def titles_prompt_sectioning(prompt: Prompt) -> str:
    """Formats the section titles in a prompt."""
    return _prompt_sectioning(
        prompt,
        lambda section_name: f"\n## {section_name.capitalize()}",
        None,
    )


@register_sectioning_handler("html")
def html_prompt_sectioning(prompt: Prompt) -> str:
    """Wraps the prompt in HTML section tags."""
    return _prompt_sectioning(
        prompt,
        lambda section_name: f"<{section_name.capitalize().replace(' ', '')}>",
        lambda section_name: f"</{section_name.capitalize().replace(' ', '')}>",
        strip_empty_lines=True,
    )
