from __future__ import annotations

import re
import dataclasses

import structlog
from typing import TYPE_CHECKING, Literal
import talemate.emit.async_signals
import talemate.util as util
from talemate.events import GameLoopEvent
from talemate.prompts import Prompt
from talemate.scene_message import (
    DirectorMessage,
    TimePassageMessage,
    ContextInvestigationMessage,
    ReinforcementMessage,
)
from talemate.world_state.templates import GenerationOptions
from talemate.client import ClientBase
from talemate.agents.base import (
    Agent,
    AgentAction,
    AgentActionConfig,
    set_processing,
    AgentEmission,
    AgentTemplateEmission,
    RagBuildSubInstructionEmission,
)
from talemate.agents.registry import register
from talemate.agents.memory.rag import MemoryRAGMixin

from talemate.history import ArchiveEntry

from .analyze_scene import SceneAnalyzationMixin
from .context_investigation import ContextInvestigationMixin
from .layered_history import LayeredHistoryMixin
from .tts_utils import TTSUtilsMixin

if TYPE_CHECKING:
    from talemate.tale_mate import Character

log = structlog.get_logger("talemate.agents.summarize")

talemate.emit.async_signals.register(
    "agent.summarization.before_build_archive",
    "agent.summarization.after_build_archive",
    "agent.summarization.rag_build_sub_instruction",
    "agent.summarization.summarize.before",
    "agent.summarization.summarize.after",
)


@dataclasses.dataclass
class BuildArchiveEmission(AgentEmission):
    generation_options: GenerationOptions | None = None


@dataclasses.dataclass
class SummarizeEmission(AgentTemplateEmission):
    text: str = ""
    extra_context: str | None = None
    extra_instructions: str | None = None
    generation_options: GenerationOptions | None = None
    summarization_history: list[str] | None = None
    summarization_type: Literal["dialogue", "events"] = "dialogue"


@register()
class SummarizeAgent(
    MemoryRAGMixin,
    LayeredHistoryMixin,
    ContextInvestigationMixin,
    # Needs to be after ContextInvestigationMixin so signals are connected in the right order
    SceneAnalyzationMixin,
    TTSUtilsMixin,
    Agent,
):
    """
    An agent that can be used to summarize text
    """

    agent_type = "summarizer"
    verbose_name = "Summarizer"
    auto_squish = False

    @classmethod
    def init_actions(cls) -> dict[str, AgentAction]:
        """Initialize and return a dictionary of agent actions."""
        actions = {
            "archive": AgentAction(
                enabled=True,
                label="Summarize to long-term memory archive",
                description="Automatically summarize scene dialogue when the number of tokens in the history exceeds a threshold. This helps keep the context history from growing too large.",
                config={
                    "threshold": AgentActionConfig(
                        type="number",
                        label="Token Threshold",
                        description="Will summarize when the number of tokens in the history exceeds this threshold",
                        min=512,
                        max=8192,
                        step=256,
                        value=1536,
                    ),
                    "method": AgentActionConfig(
                        type="text",
                        label="Summarization Method",
                        description="Which method to use for summarization",
                        value="balanced",
                        choices=[
                            {"label": "Short & Concise", "value": "short"},
                            {"label": "Balanced", "value": "balanced"},
                            {"label": "Lengthy & Detailed", "value": "long"},
                            {"label": "Factual List", "value": "facts"},
                        ],
                    ),
                    "include_previous": AgentActionConfig(
                        type="number",
                        label="Use preceeding summaries to strengthen context",
                        description="Number of entries",
                        note="Help the AI summarize by including the last few summaries as additional context. Some models may incorporate this context into the new summary directly, so if you find yourself with a bunch of similar history entries, try setting this to 0.",
                        value=6,
                        min=0,
                        max=24,
                        step=1,
                    ),
                },
            ),
        }
        LayeredHistoryMixin.add_actions(actions)
        MemoryRAGMixin.add_actions(actions)
        SceneAnalyzationMixin.add_actions(actions)
        ContextInvestigationMixin.add_actions(actions)
        return actions

    def __init__(self, client: ClientBase | None = None, **kwargs):
        self.client = client

        self.actions = SummarizeAgent.init_actions()

    @property
    def threshold(self):
        """Get the threshold value from the archive action configuration."""
        return self.actions["archive"].config["threshold"].value

    @property
    def estimated_entry_count(self):
        all_tokens = sum([util.count_tokens(entry) for entry in self.scene.history])
        return all_tokens // self.threshold

    @property
    def archive_threshold(self):
        """Gets the archive threshold value from the configuration."""
        return self.actions["archive"].config["threshold"].value

    @property
    def archive_method(self):
        """Gets the archive method configuration value."""
        return self.actions["archive"].config["method"].value

    @property
    def archive_include_previous(self):
        return self.actions["archive"].config["include_previous"].value

    def connect(self, scene):
        """Connects the scene and sets up the game loop signal."""
        super().connect(scene)
        talemate.emit.async_signals.get("game_loop").connect(self.on_game_loop)

    async def on_game_loop(self, emission: GameLoopEvent):

        """Handles the game loop event."""
        await self.build_archive(self.scene)

    def clean_result(self, result):
        """Cleans the result by removing comments and partial sentences."""
        if "#" in result:
            result = result.split("#")[0]

        # Removes partial sentence at the end
        result = util.strip_partial_sentences(result)
        result = result.strip()

        return result

    # RAG HELPERS

    async def rag_build_sub_instruction(self):
        # Fire event to get the sub instruction from mixins
        """Builds and sends a sub instruction emission."""
        emission = RagBuildSubInstructionEmission(
            agent=self,
        )
        await talemate.emit.async_signals.get(
            "agent.summarization.rag_build_sub_instruction"
        ).send(emission)

        return emission.sub_instruction

    # SUMMARIZATION HELPERS

    async def previous_summaries(self, entry: ArchiveEntry) -> list[str]:
        """Retrieve previous summaries from the archived history.
        
        This asynchronous function searches for a specific entry in the  archived
        history using its ID. If found, it compiles a list of  previous summaries based
        on the specified number of entries to  include. The function checks if layered
        history is available to  determine the method of compilation, either using the
        compile_layered_history method or directly accessing the archived  history
        list.
        
        Args:
            entry (ArchiveEntry): The entry for which previous summaries
        """
        num_previous = self.archive_include_previous

        # find entry by .id
        entry_index = next(
            (
                i
                for i, e in enumerate(self.scene.archived_history)
                if e["id"] == entry.id
            ),
            None,
        )
        if entry_index is None:
            raise ValueError("Entry not found")
        end = entry_index - 1

        previous_summaries = []

        if entry and num_previous > 0:
            if self.layered_history_available:
                previous_summaries = self.compile_layered_history(
                    include_base_layer=True, base_layer_end_id=entry.id
                )[-num_previous:]
            else:
                previous_summaries = [
                    entry.text
                    for entry in self.scene.archived_history[end - num_previous : end]
                ]

        return previous_summaries

    # SUMMARIZE

    @set_processing
    async def build_archive(
        self, scene, generation_options: GenerationOptions | None = None
    ):
        end = None

        emission = BuildArchiveEmission(
            agent=self,
            generation_options=generation_options,
        )

        await talemate.emit.async_signals.get(
            "agent.summarization.before_build_archive"
        ).send(emission)

        if not self.actions["archive"].enabled:
            return

        if not scene.archived_history:
            start = 0
            recent_entry = None
        else:
            recent_entry = scene.archived_history[-1]
            if "end" not in recent_entry:
                # permanent historical archive entry, not tied to any specific history entry
                # meaning we are still at the beginning of the scene
                start = 0
            else:
                start = recent_entry.get("end", 0) + 1

        # if there is a recent entry we also collect the 3 most recentries
        # as extra context

        num_previous = self.actions["archive"].config["include_previous"].value
        if recent_entry and num_previous > 0:
            if self.layered_history_available:
                extra_context = self.compile_layered_history(include_base_layer=True)
            else:
                extra_context = [
                    entry["text"] for entry in scene.archived_history[-num_previous:]
                ]

        else:
            extra_context = None

        tokens = 0
        dialogue_entries = []
        ts = "PT0S"
        time_passage_termination = False

        token_threshold = self.actions["archive"].config["threshold"].value

        log.debug("build_archive", start=start, recent_entry=recent_entry)

        if recent_entry:
            ts = recent_entry.get("ts", ts)

        # we ignore the most recent entry, as the user may still chose to
        # regenerate it
        for i in range(start, max(start, len(scene.history) - 1)):
            dialogue = scene.history[i]

            # log.debug("build_archive", idx=i, content=str(dialogue)[:64]+"...")

            if isinstance(
                dialogue,
                (DirectorMessage, ContextInvestigationMessage, ReinforcementMessage),
            ):
                # these messages are not part of the dialogue and should not be summarized
                if i == start:
                    start += 1
                continue

            if isinstance(dialogue, TimePassageMessage):
                log.debug("build_archive", time_passage_message=dialogue)
                ts = util.iso8601_add(ts, dialogue.ts)

                if i == start:
                    log.debug(
                        "build_archive",
                        time_passage_message=dialogue,
                        start=start,
                        i=i,
                        ts=ts,
                    )
                    start += 1
                    continue
                log.debug("build_archive", time_passage_message_termination=dialogue)
                time_passage_termination = True
                end = i - 1
                break

            tokens += util.count_tokens(dialogue)
            dialogue_entries.append(dialogue)
            if tokens > token_threshold:  #
                end = i
                break

        if end is None:
            # nothing to archive yet
            return

        log.debug(
            "build_archive",
            start=start,
            end=end,
            ts=ts,
            time_passage_termination=time_passage_termination,
        )

        # in order to summarize coherently, we need to determine if there is a favorable
        # cutoff point (e.g., the scene naturally ends or shifts meaninfully in the middle
        # of the  dialogue)
        #
        # One way to do this is to check if the last line is a TimePassageMessage, which
        # indicates a scene change or a significant pause.
        #
        # If not, we can ask the AI to find a good point of
        # termination.

        if not time_passage_termination:
            # No TimePassageMessage, so we need to ask the AI to find a good point of termination

            terminating_line = await self.analyze_dialoge(dialogue_entries)

            if terminating_line:
                adjusted_dialogue = []
                for line in dialogue_entries:
                    if str(line) in terminating_line:
                        break
                    adjusted_dialogue.append(line)

                # if difference start and end is less than 4, ignore the termination
                if len(adjusted_dialogue) > 4:
                    dialogue_entries = adjusted_dialogue
                    end = start + len(dialogue_entries) - 1
                else:
                    log.debug(
                        "build_archive",
                        message="Ignoring termination",
                        start=start,
                        end=end,
                        adjusted_dialogue=adjusted_dialogue,
                    )

        if dialogue_entries:
            if not extra_context:
                # prepend scene intro to dialogue
                dialogue_entries.insert(0, scene.intro)

            summarized = None
            retries = 5

            while not summarized and retries > 0:
                summarized = await self.summarize(
                    "\n".join(map(str, dialogue_entries)),
                    extra_context=extra_context,
                    generation_options=generation_options,
                )
                retries -= 1

            if not summarized:
                raise IOError("Failed to summarize dialogue", dialogue=dialogue_entries)

        else:
            # AI has likely identified the first line as a scene change, so we can't summarize
            # just use the first line
            summarized = str(scene.history[start])

        # determine the appropariate timestamp for the summarization

        await scene.push_archive(
            ArchiveEntry(text=summarized, start=start, end=end, ts=ts)
        )

        scene.ts = ts
        scene.emit_status()

        await talemate.emit.async_signals.get(
            "agent.summarization.after_build_archive"
        ).send(emission)

        return True

    @set_processing
    async def analyze_dialoge(self, dialogue):
        response = await Prompt.request(
            "summarizer.analyze-dialogue",
            self.client,
            "analyze_freeform",
            vars={
                "dialogue": "\n".join(map(str, dialogue)),
                "scene": self.scene,
                "max_tokens": self.client.max_token_length,
            },
        )

        response = self.clean_result(response)
        return response

    @set_processing
    async def find_natural_scene_termination(
        self, event_chunks: list[str]
    ) -> list[list[str]]:

        # scan through event chunks and split into paragraphs
        """Analyzes a list of events and separates them at natural scene termination
        points.
        
        This function processes the provided event_chunks by scanning through them and
        splitting them into paragraphs. It then sends the cleaned event data to a
        summarization service to identify natural scene termination events. The results
        are parsed to extract unique and  sorted event indices, which are used to
        segment the original event_chunks into separate  lists based on these indices.
        """
        rebuilt_chunks = []

        for chunk in event_chunks:
            paragraphs = [p.strip() for p in chunk.split("\n") if p.strip()]
            rebuilt_chunks.extend(paragraphs)

        event_chunks = rebuilt_chunks

        response = await Prompt.request(
            "summarizer.find-natural-scene-termination-events",
            self.client,
            "analyze_short2",
            vars={
                "scene": self.scene,
                "max_tokens": self.client.max_token_length,
                "events": event_chunks,
            },
        )
        response = response.strip()

        items = util.extract_list(response)

        # will be a list of
        # ["Progress 1", "Progress 12", "Progress 323", ...]
        # convert to a list of just numbers

        numbers = []

        for item in items:
            match = re.match(r"Progress (\d+)", item.strip())
            if match:
                numbers.append(int(match.group(1)))

        # make sure its unique and sorted
        numbers = sorted(list(set(numbers)))

        result = []
        prev_number = 0
        for number in numbers:
            result.append(event_chunks[prev_number : number + 1])
            prev_number = number + 1

        # result = {
        #    "selected": event_chunks[:number+1],
        #    "remaining": event_chunks[number+1:]
        # }

        log.debug(
            "find_natural_scene_termination",
            response=response,
            result=result,
            numbers=numbers,
        )

        return result

    @set_processing
    async def summarize(
        self,
        text: str,
        extra_context: str = None,
        method: str = None,
        extra_instructions: str = None,
        generation_options: GenerationOptions | None = None,
    ):

        """Summarize the given text.
        
        This asynchronous function generates a summary of the provided text using a
        specified summarization method.  It prepares the necessary template variables,
        emits pre-summarization signals, and requests the summary  from a prompt. After
        receiving the response, it processes the summary, ensuring proper formatting,
        and  emits post-summarization signals before returning the cleaned result.
        
        Args:
            text (str): The text to be summarized.
            extra_context (str?): Additional context to enhance the summary.
            method (str?): The method to be used for summarization.
            extra_instructions (str?): Any extra instructions for the summarization process.
            generation_options (GenerationOptions | None?): Options for the generation process.
        """
        response_length = 1024

        template_vars = {
            "dialogue": text,
            "scene": self.scene,
            "max_tokens": self.client.max_token_length,
            "summarization_method": (
                self.actions["archive"].config["method"].value
                if method is None
                else method
            ),
            "extra_context": extra_context or "",
            "num_extra_context": len(extra_context) if extra_context else 0,
            "extra_instructions": extra_instructions or "",
            "generation_options": generation_options,
            "analyze_chunks": self.layered_history_analyze_chunks,
            "response_length": response_length,
        }

        emission = SummarizeEmission(
            agent=self,
            text=text,
            extra_context=extra_context,
            extra_instructions=extra_instructions,
            generation_options=generation_options,
            template_vars=template_vars,
            summarization_history=extra_context or [],
            summarization_type="dialogue",
        )

        await talemate.emit.async_signals.get(
            "agent.summarization.summarize.before"
        ).send(emission)

        template_vars["dynamic_instructions"] = emission.dynamic_instructions

        response = await Prompt.request(
            "summarizer.summarize-dialogue",
            self.client,
            f"summarize_{response_length}",
            vars=template_vars,
            dedupe_enabled=False,
        )

        log.debug(
            "summarize", dialogue_length=len(text), summarized_length=len(response)
        )

        try:
            summary = response.split("SUMMARY:")[1].strip()
        except Exception as e:
            log.error("summarize failed", response=response, exc=e)
            return ""

        # capitalize first letter
        try:
            summary = summary[0].upper() + summary[1:]
        except IndexError:
            pass

        emission.response = self.clean_result(summary)

        await talemate.emit.async_signals.get(
            "agent.summarization.summarize.after"
        ).send(emission)

        summary = emission.response

        return self.clean_result(summary)

    @set_processing
    async def summarize_events(
        self,
        text: str,
        extra_context: str = None,
        extra_instructions: str = None,
        generation_options: GenerationOptions | None = None,
        analyze_chunks: bool = False,
        chunk_size: int = 1280,
        response_length: int = 2048,
    ):

        """Summarize the given text.
        
        This asynchronous function processes the input text along with optional extra
        context and instructions to generate a summarized version of the events
        described. It first parses any mentioned characters from the text, constructs a
        set of template variables, and emits a pre-summarization signal. After
        receiving a response from the summarization prompt, it cleans up the response
        by removing any analysis text and formatting it appropriately before emitting a
        post-summarization signal.
        
        Args:
            text (str): The text to be summarized.
            extra_context (str?): Additional context to include in the summary. Defaults to None.
            extra_instructions (str?): Specific instructions for the summarization process. Defaults to None.
            generation_options (GenerationOptions | None?): Options for the generation process. Defaults to None.
            analyze_chunks (bool?): Flag to indicate if the text should be analyzed in chunks. Defaults to False.
            chunk_size (int?): The size of each chunk for analysis. Defaults to 1280.
            response_length (int?): The desired length of the summary response. Defaults to 2048.
        """
        if not extra_context:
            extra_context = ""

        mentioned_characters: list["Character"] = self.scene.parse_characters_from_text(
            text + extra_context, exclude_active=True
        )

        template_vars = {
            "dialogue": text,
            "scene": self.scene,
            "max_tokens": self.client.max_token_length,
            "extra_context": extra_context,
            "num_extra_context": len(extra_context),
            "extra_instructions": extra_instructions or "",
            "generation_options": generation_options,
            "analyze_chunks": analyze_chunks,
            "chunk_size": chunk_size,
            "response_length": response_length,
            "mentioned_characters": mentioned_characters,
        }

        emission = SummarizeEmission(
            agent=self,
            text=text,
            extra_context=extra_context,
            extra_instructions=extra_instructions,
            generation_options=generation_options,
            template_vars=template_vars,
            summarization_history=[extra_context] if extra_context else [],
            summarization_type="events",
        )

        await talemate.emit.async_signals.get(
            "agent.summarization.summarize.before"
        ).send(emission)

        template_vars["dynamic_instructions"] = emission.dynamic_instructions

        response = await Prompt.request(
            "summarizer.summarize-events",
            self.client,
            f"summarize_{response_length}",
            vars=template_vars,
            dedupe_enabled=False,
        )

        response = response.strip()
        response = response.replace('"', "")

        log.debug(
            "layered_history_summarize",
            original_length=len(text),
            summarized_length=len(response),
        )

        # clean up analyzation (remove analyzation text)
        if self.layered_history_analyze_chunks:
            # remove all lines that begin with "ANALYSIS OF CHUNK \d+:"
            response = "\n".join(
                [
                    line
                    for line in response.split("\n")
                    if not line.startswith("ANALYSIS OF CHUNK")
                ]
            )

        # strip all occurences of "CHUNK \d+: " from the summary
        response = re.sub(r"(CHUNK|CHAPTER) \d+:\s+", "", response)

        # capitalize first letter
        try:
            response = response[0].upper() + response[1:]
        except IndexError:
            pass

        emission.response = self.clean_result(response)

        await talemate.emit.async_signals.get(
            "agent.summarization.summarize.after"
        ).send(emission)

        response = emission.response

        log.debug(
            "summarize_events",
            original_length=len(text),
            summarized_length=len(response),
        )

        return self.clean_result(response)
