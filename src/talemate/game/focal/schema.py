from typing import Callable, Any, Literal
import pydantic
import uuid
import json
import yaml

from talemate.prompts.base import Prompt

__all__ = [
    "Argument",
    "Call",
    "Callback",
    "State",
    "InvalidCallbackArguments",
    "ExampleCallbackArguments",
]

YAML_OPTIONS = {
    "default_flow_style": False,
    "allow_unicode": True,
    "indent": 2,
    "sort_keys": False,
    "width": 100,
}

YAML_PRESERVE_NEWLINES = (
    "If there are newlines, they should be preserved by using | style."
)


class InvalidCallbackArguments(ValueError):
    pass


class ExampleCallbackArguments(InvalidCallbackArguments):
    pass


class State(pydantic.BaseModel):
    calls: list["Call"] = pydantic.Field(default_factory=list)
    schema_format: Literal["json", "yaml"] = "json"


class Argument(pydantic.BaseModel):
    name: str
    type: str
    preserve_newlines: bool = False

    def extra_instructions(self, state: State) -> str:
        """Returns extra instructions based on the state schema format."""
        if state.schema_format == "yaml" and self.preserve_newlines:
            return f" {YAML_PRESERVE_NEWLINES}"
        return ""


class Call(pydantic.BaseModel):
    name: str = pydantic.Field(
        validation_alias=pydantic.AliasChoices("name", "function")
    )
    arguments: dict[str, Any] = pydantic.Field(default_factory=dict)
    result: str | int | float | bool | dict | list | None = None
    uid: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()))
    called: bool = False

    @pydantic.field_validator("arguments")
    def check_for_schema_examples(cls, v: dict[str, Any]) -> dict[str, str]:
        """Check for schema examples in the provided arguments.
        
        This function validates the values in the 'arguments' field of a Pydantic
        model.  It checks if any string value starts with a valid type prefix (e.g.,
        'str - ', 'int - ').  If such a prefix is found, it raises an
        ExampleCallbackArguments exception to indicate  that the argument contains a
        schema example, which is not allowed.
        """
        valid_types = ["str", "int", "float", "bool", "dict", "list"]
        for key, value in v.items():
            if isinstance(value, str):
                for type_name in valid_types:
                    if value.startswith(f"{type_name} - "):
                        raise ExampleCallbackArguments(
                            f"Argument '{key}' contains schema example: '{value}'. AI repeated the schema format."
                        )
        return v

    @pydantic.field_validator("arguments")
    def join_string_lists(cls, v: dict[str, Any]) -> dict[str, str]:
        """Join lists of strings in a dictionary into single strings."""
        return {
            key: "\n".join(str(item) for item in value)
            if isinstance(value, list)
            else str(value)
            for key, value in v.items()
        }


class Callback(pydantic.BaseModel):
    name: str
    arguments: list[Argument] = pydantic.Field(default_factory=list)
    fn: Callable
    state: State = State()
    multiple: bool = True

    @property
    def pretty_name(self) -> str:
        """Return the pretty version of the name with spaces and title case."""
        return self.name.replace("_", " ").title()

    def render(self, usage: str, examples: list[dict] = None, **argument_usage) -> str:
        """Renders a prompt based on the provided usage and arguments."""
        prompt = Prompt.get(
            "focal.callback",
            {
                "callback": self,
                "name": self.name,
                "usage": usage,
                "argument_usage": argument_usage or {},
                "arguments": self.arguments,
                "state": self.state,
                "examples": examples or [],
            },
        )

        return prompt.render()

    ## schema

    def _usage(self, argument_usage) -> dict:
        """Returns a dictionary of function usage and argument details."""
        return {
            "function": self.name,
            "arguments": {
                argument.name: f"{argument.type} - {argument_usage.get(argument.name, '')}{argument.extra_instructions(self.state)}"
                for argument in self.arguments
            },
        }

    def _example(self, example: dict) -> dict:
        """Return a dictionary with function name and filtered arguments."""
        return {
            "function": self.name,
            "arguments": {k: v for k, v in example.items() if not k.startswith("_")},
        }

    def usage(self, argument_usage) -> str:
        """Return formatted usage information for the given argument."""
        fmt: str = self.state.schema_format
        text = getattr(self, f"{fmt}_usage")(argument_usage)
        text = text.rstrip()
        return f"```{fmt}\n{text}\n```"

    def example(self, example: dict) -> str:
        """Generate a formatted example string from the given dictionary."""
        fmt: str = self.state.schema_format
        text = getattr(self, f"{fmt}_example")(example)
        text = text.rstrip()
        return f"```{fmt}\n{text}\n```"

    ## JSON

    def json_usage(self, argument_usage) -> str:
        """Convert usage data to a JSON string."""
        return json.dumps(self._usage(argument_usage), indent=2)

    def json_example(self, example: dict) -> str:
        """Convert a dictionary example to a formatted JSON string."""
        return json.dumps(self._example(example), indent=2)

    ## YAML

    def yaml_usage(self, argument_usage) -> str:
        """Convert usage information to YAML format."""
        return yaml.dump(self._usage(argument_usage), **YAML_OPTIONS)

    def yaml_example(self, example: dict) -> str:
        """Convert a dictionary to a YAML string."""
        return yaml.dump(self._example(example), **YAML_OPTIONS)
