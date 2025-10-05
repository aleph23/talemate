import enum
import re
import structlog
from dataclasses import dataclass, field

log = structlog.get_logger("talemate.scene_message")

__all__ = [
    "SceneMessage",
    "CharacterMessage",
    "NarratorMessage",
    "DirectorMessage",
    "TimePassageMessage",
    "ReinforcementMessage",
    "ContextInvestigationMessage",
    "Flags",
    "MESSAGES",
]

_message_id = 0


def get_message_id():
    """Increments and returns the global message ID."""
    global _message_id
    _message_id += 1
    return _message_id


def reset_message_id():
    """Reset the global message ID to zero."""
    global _message_id
    _message_id = 0


class Flags(enum.IntFlag):
    """
    Flags for messages
    """

    NONE = 0x0
    HIDDEN = 0x1


@dataclass
class SceneMessage:
    """
    Base class for all messages that are sent to the scene.
    """

    # the mesage itself
    message: str

    # the id of the message
    id: int = field(default_factory=get_message_id)

    # the source of the message (e.g. "ai", "progress_story", "director")
    source: str = ""

    meta: dict | None = None

    flags: Flags = Flags.NONE

    typ = "scene"

    def __str__(self):
        return self.message

    def __int__(self):
        return self.id

    def __len__(self):
        return len(self.message)

    def __in__(self, other):
        return other in self.message

    def __contains__(self, other):
        return self.message in other

    def __dict__(self) -> dict:
        rv = {
            "message": self.message,
            "id": self.id,
            "typ": self.typ,
            "source": self.source,
            "flags": int(self.flags),
        }

        if self.meta:
            rv["meta"] = self.meta

        return rv

    def __iter__(self):
        return iter(self.message)

    def split(self, *args, **kwargs):
        """Splits the message using the specified arguments."""
        return self.message.split(*args, **kwargs)

    def startswith(self, *args, **kwargs):
        """Check if the message starts with the given prefix."""
        return self.message.startswith(*args, **kwargs)

    def endswith(self, *args, **kwargs):
        """Check if the message ends with the given suffix."""
        return self.message.endswith(*args, **kwargs)

    @property
    def secondary_source(self):
        """Returns the source property."""
        return self.source

    @property
    def raw(self):
        """Return the string representation of the message."""
        return str(self.message)

    @property
    def hidden(self):
        return self.flags & Flags.HIDDEN

    @property
    def fingerprint(self) -> str:
        """Returns a unique hash fingerprint for the message."""
        return str(hash(self.message))[:16]

    @property
    def source_agent(self) -> str | None:
        """Return the source agent from meta or None if not present."""
        return (self.meta or {}).get("agent", None)

    @property
    def source_function(self) -> str | None:
        """Return the function name from meta or None if not present."""
        return (self.meta or {}).get("function", None)

    @property
    def source_arguments(self) -> dict:
        """Return the source arguments as a dictionary."""
        return (self.meta or {}).get("arguments", {})

    @property
    def meta_hash(self) -> int:
        """Return the hash of the meta attribute as an integer."""
        return hash(str(self.meta))

    def hide(self):
        """Set the HIDDEN flag for the instance."""
        self.flags |= Flags.HIDDEN

    def unhide(self):
        self.flags &= ~Flags.HIDDEN

    def as_format(self, format: str, **kwargs) -> str:
        """Formats the message based on the specified format type."""
        if format == "movie_script":
            return self.message.rstrip("\n") + "\n"
        elif format == "narrative":
            return self.message.strip()
        return self.message

    def set_source(self, agent: str, function: str, **kwargs):
        """Sets the source metadata for the agent and function."""
        if not self.meta:
            self.meta = {}
        self.meta["agent"] = agent
        self.meta["function"] = function
        self.meta["arguments"] = kwargs

    def set_meta(self, **kwargs):
        if not self.meta:
            self.meta = {}
        self.meta.update(kwargs)


@dataclass
class CharacterMessage(SceneMessage):
    typ = "character"
    source: str = "ai"
    from_choice: str | None = None

    def __str__(self):
        return self.message

    @property
    def character_name(self):
        """Get the character name from the message."""
        return self.message.split(":", 1)[0]

    @property
    def secondary_source(self):
        """Return the character name."""
        return self.character_name

    @property
    def raw(self):
        """Returns the raw message content after processing."""
        return self.message.split(":", 1)[1].replace('"', "").replace("*", "").strip()

    @property
    def without_name(self) -> str:
        """Returns the part of the message after the first colon."""
        return self.message.split(":", 1)[1]

    @property
    def as_movie_script(self):
        """
        Returns the dialogue line as a script dialogue line.

        Example:
        {CHARACTER_NAME}
        {dialogue}
        """

        try:
            message = self.message.split(":", 1)[1].strip()
        except IndexError:
            log.warning(
                "character_message_as_movie_script failed to parse correct format",
                msg=self.message,
            )
            message = self.message

        return f"\n{self.character_name.upper()}\n{message}\nEND-OF-LINE\n"

    def __dict__(self) -> dict:
        rv = super().__dict__()

        if self.from_choice:
            rv["from_choice"] = self.from_choice

        return rv

    def as_format(self, format: str, **kwargs) -> str:
        """Returns a formatted string based on the specified format type."""
        if format == "movie_script":
            return self.as_movie_script
        elif format == "narrative":
            return self.without_name.strip()
        return self.message


@dataclass
class NarratorMessage(SceneMessage):
    source: str = "ai"
    typ = "narrator"

    def source_to_meta(self) -> dict:
        """Converts a source string into a metadata dictionary.
        
        This function takes the source string, splits it into an action name and  its
        associated arguments, and constructs a parameters dictionary based on  the
        action type. It supports various actions such as "paraphrase",
        "narrate_character_entry", and "progress_story", each requiring different
        arguments. The resulting dictionary includes the agent type, function name,
        and the constructed parameters.
        
        Returns:
            dict: A dictionary containing the agent, function name, and arguments.
        """
        source = self.source
        action_name, *args = source.split(":")
        parameters = {}

        if action_name == "paraphrase":
            parameters["narration"] = args[0]
        elif action_name == "narrate_character_entry":
            parameters["character"] = args[0]
        elif action_name == "narrate_character_exit":
            parameters["character"] = args[0]
        elif action_name == "narrate_character":
            parameters["character"] = args[0]
        elif action_name == "narrate_query":
            parameters["query"] = args[0]
        elif action_name == "narrate_time_passage":
            parameters["duration"] = args[0]
            parameters["time_passed"] = args[1]
            parameters["narrative"] = args[2]
        elif action_name == "progress_story":
            parameters["narrative_direction"] = args[0]
        elif action_name == "narrate_after_dialogue":
            parameters["character"] = args[0]

        return {"agent": "narrator", "function": action_name, "arguments": parameters}

    def migrate_source_to_meta(self):
        """Migrate data from source to meta if meta is not set."""
        if self.source and not self.meta:
            try:
                self.meta = self.source_to_meta()
            except Exception as e:
                log.warning("migrate_narrator_source_to_meta", error=e, msg=self.id)

        return self


@dataclass
class DirectorMessage(SceneMessage):
    action: str = "actor_instruction"
    source: str = "ai"
    typ = "director"
    subtype: str | None = None

    @property
    def character_name(self) -> str:
        return self.meta.get("character") if self.meta else None

    @property
    def instructions(self) -> str:
        """Return the message as instructions."""
        return self.message

    @property
    def as_inner_monologue(self):
        # instructions may be written referencing the character as you, your etc.,
        # so we need to replace those to fit a first person perspective

        # first we lowercase
        """Converts instructions to a first-person perspective for the character."""
        instructions = self.instructions.lower()

        if not self.character_name:
            return instructions

        # then we replace yourself with myself using regex, taking care of word boundaries
        instructions = re.sub(r"\byourself\b", "myself", instructions)

        # then we replace your with my using regex, taking care of word boundaries
        instructions = re.sub(r"\byour\b", "my", instructions)

        # then we replace you with i using regex, taking care of word boundaries
        instructions = re.sub(r"\byou\b", "i", instructions)

        return f"{self.character_name} thinks: I should {instructions}"

    @property
    def as_story_progression(self):
        """Return the story progression for the character."""
        return f"{self.character_name}'s next action: {self.instructions}"

    @property
    def as_director_action(self) -> str:
        """Return the action message for the director."""
        if not self.character_name:
            return f"{self.message}\n{self.action}"

    # Become aggressive towards Elmer as you no longer recognize the man.
    def migrate_message_to_meta(self):
        """Migrates the message to meta format if it starts with 'Director instructs'."""
        if self.message.startswith("Director instructs"):
            parts = self.message.split(":", 1)
            character_name = parts[0].replace("Director instructs ", "").strip()
            instructions = parts[1].strip()

            self.set_source(
                "director",
                "actor_instruction",
                character=character_name,
            )
            self.message = instructions
            self.source = "player"

        return self

    def __dict__(self) -> dict:
        rv = super().__dict__()

        if self.action:
            rv["action"] = self.action

        return rv

    def __str__(self):
        """
        The director message is a special case and needs to be transformed
        """
        return self.as_format("chat")

    def as_format(self, format: str, **kwargs) -> str:
        """Format the output based on the specified format type.
        
        This method checks the `instructions` attribute and returns an  appropriate
        formatted string based on the provided `format` and  `mode` arguments. If the
        `format` is either "movie_script" or  "narrative", it will return a different
        format for internal  monologues compared to story progression. For other
        formats,  it uses a different prefix while still considering the `mode`.
        
        Args:
            format (str): The format type to apply.
            **kwargs: Additional keyword arguments, including mode.
        """
        if not self.instructions.strip():
            return ""

        mode = kwargs.get("mode", "direction")
        if format in ["movie_script", "narrative"]:
            if mode == "internal_monologue":
                return f"\n({self.as_inner_monologue})\n"
            else:
                return f"\n({self.as_story_progression})\n"
        else:
            if mode == "internal_monologue":
                return f"# {self.as_inner_monologue}"
            else:
                return f"# {self.as_story_progression}"


@dataclass
class TimePassageMessage(SceneMessage):
    ts: str = "PT0S"
    source: str = "manual"
    typ = "time"

    def __dict__(self) -> dict:
        rv = super().__dict__()
        rv["ts"] = self.ts
        return rv


@dataclass
class ReinforcementMessage(SceneMessage):
    typ = "reinforcement"
    source: str = "ai"

    @property
    def character_name(self):
        return self.source_arguments.get("character", "character")

    @property
    def question(self):
        """Return the question from source_arguments or a default value."""
        return self.source_arguments.get("question", "question")

    def __str__(self):
        return f"# Internal note for {self.character_name} - {self.question}\n{self.message}"

    def as_format(self, format: str, **kwargs) -> str:
        """Formats the message based on the specified format type."""
        if format in ["movie_script", "narrative"]:
            message = str(self)[2:]
            return f"\n({message})\n"
        return f"\n{self.message}\n"

    def migrate_source_to_meta(self):
        """Migrate data from source to meta if conditions are met."""
        if self.source and not self.meta:
            try:
                self.source_to_meta()
            except Exception as e:
                log.warning(
                    "migrate_reinforcement_source_to_meta", error=e, msg=self.id
                )

        return self

    def source_to_meta(self):
        """Sets the source parameters from the source string."""
        source = self.source
        args = source.split(":")
        parameters = {"character": args[1], "question": args[0]}
        self.set_source("world_state", "update_reinforcement", **parameters)


@dataclass
class ContextInvestigationMessage(SceneMessage):
    typ = "context_investigation"
    source: str = "ai"
    sub_type: str | None = None

    @property
    def character(self) -> str:
        """Return the character from source_arguments or a default value."""
        return self.source_arguments.get("character", "character")

    @property
    def query(self) -> str:
        return self.source_arguments.get("query", "query")

    @property
    def title(self) -> str:
        """
        The title will differ based on sub_type

        Current sub_types:

        - visual-character
        - visual-scene
        - query

        A natural language title will be generated based on the sub_type
        """

        if self.sub_type == "visual-character":
            return f"Visual description of {self.character} in the current moment"
        elif self.sub_type == "visual-scene":
            return "Visual description of the current moment"
        elif self.sub_type == "query":
            return f"Query: {self.query}"
        return "Internal note"

    def __str__(self):
        return f"# {self.title}: {self.message}"

    def __dict__(self) -> dict:
        rv = super().__dict__()
        rv["sub_type"] = self.sub_type
        return rv

    def as_format(self, format: str, **kwargs) -> str:
        """Formats the message based on the specified format type."""
        if format in ["movie_script", "narrative"]:
            message = str(self)[2:]
            return f"\n({message})\n".replace("*", "")
        return f"\n{self.message}\n".replace("*", "")


MESSAGES = {
    "scene": SceneMessage,
    "character": CharacterMessage,
    "narrator": NarratorMessage,
    "director": DirectorMessage,
    "time": TimePassageMessage,
    "reinforcement": ReinforcementMessage,
    "context_investigation": ContextInvestigationMessage,
}
