import asyncio
import json
import os
import re
import traceback
import uuid
from typing import Generator, Callable

import isodate
import structlog
from blinker import signal

import talemate.agents as agents
import talemate.client as client
import talemate.commands as commands
import talemate.emit.async_signals as async_signals
import talemate.events as events
import talemate.save as save
import talemate.util as util
import talemate.world_state.templates as world_state_templates
from talemate.agents.context import active_agent
from talemate.config import Config, get_config
from talemate.context import interaction
from talemate.emit import Emitter, emit
from talemate.exceptions import (
    ExitScene,
    LLMAccuracyError,
    ResetScene,
    RestartSceneLoop,
    TalemateError,
    TalemateInterrupt,
    GenerationCancelled,
)
from talemate.game.state import GameState
from talemate.scene_assets import SceneAssets
from talemate.scene_message import (
    CharacterMessage,
    DirectorMessage,
    ReinforcementMessage,
    SceneMessage,
    TimePassageMessage,
    ContextInvestigationMessage,
    MESSAGES as MESSAGE_TYPES,
)
from talemate.util import count_tokens
from talemate.util.prompt import condensed
from talemate.world_state import WorldState
from talemate.world_state.manager import WorldStateManager
from talemate.game.engine.nodes.core import GraphState
from talemate.game.engine.nodes.layout import load_graph
from talemate.game.engine.nodes.packaging import initialize_packages
from talemate.scene.intent import SceneIntent
from talemate.history import emit_archive_add, ArchiveEntry
from talemate.character import Character
from talemate.agents.tts.schema import VoiceLibrary
from talemate.instance import get_agent

__all__ = [
    "Character",
    "Actor",
    "Scene",
    "Player",
]


log = structlog.get_logger("talemate")

async_signals.register(
    "scene_init",
    "scene_init_after",
    "game_loop_start",
    "game_loop",
    "game_loop_actor_iter",
    "game_loop_new_message",
    "player_turn_start",
)


class Actor:
    """
    links a character to an agent
    """

    def __init__(self, character: Character, agent: agents.Agent):
        self.character = character
        self.agent = agent
        self.scene = None

        if agent:
            agent.character = character

        character.agent = agent
        character.actor = self

    @property
    def history(self):
        return self.scene.history


class Player(Actor):
    muted = 0
    ai_controlled = 0


class Scene(Emitter):
    """
    A scene containing one ore more AI driven actors to interact with.
    """

    ExitScene = ExitScene

    @classmethod
    def scenes_dir(cls):
        """Return the absolute path to the scenes directory."""
        relative_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "scenes",
        )
        return os.path.abspath(relative_path)

    def __init__(self):
        """
        Initializes a new Scene instance with default values and configuration.
        This constructor sets up the actors, assets, world state, and other properties required for managing a scene.
        """
        self.actors = []
        self.helpers = []
        self.history = []
        self.archived_history = []
        self.inactive_characters = {}
        self.layered_history = []
        self.assets = SceneAssets(scene=self)
        self.voice_library: VoiceLibrary = VoiceLibrary()
        self.description = ""
        self.intro = ""
        self.outline = ""
        self.title = ""
        self.writing_style_template = None

        self.experimental = False
        self.help = ""

        self.name = ""
        self.filename = ""
        self._nodes_filename = ""
        self._creative_nodes_filename = ""
        self.memory_id = str(uuid.uuid4())[:10]
        self.saved_memory_session_id = None
        self.memory_session_id = str(uuid.uuid4())[:10]
        self.restore_from = None

        # has scene been saved before?
        self.saved = False

        # if immutable_save is True, save will always
        # happen as save-as and not overwrite the original
        self.immutable_save = False

        self.context = ""
        self.commands = commands.Manager(self)
        self.environment = "scene"
        self.world_state = WorldState()
        self.game_state = GameState()
        self.agent_state = {}
        self.intent_state = SceneIntent()
        self.ts = "PT0S"
        self.active = False
        self.Actor = Actor
        self.Player = Player
        self.Character = Character

        self.narrator_character_object = Character(name="__narrator__")

        self.active_pins = []
        # Add an attribute to store the most recent AI Actor
        self.most_recent_ai_actor = None

        # if the user has requested to cancel the current action
        # or series of agent actions this will be true
        #
        # A check to self.continue_actions() will be made
        #
        # if self.cancel_requested is True self.continue_actions() will raise
        # a GenerationCancelled exception
        self.cancel_requested = False

        self.signals = {
            "ai_message": signal("ai_message"),
            "player_message": signal("player_message"),
            "history_add": signal("history_add"),
            "game_loop": async_signals.get("game_loop"),
            "game_loop_start": async_signals.get("game_loop_start"),
            "game_loop_actor_iter": async_signals.get("game_loop_actor_iter"),
            "game_loop_new_message": async_signals.get("game_loop_new_message"),
            "scene_init": async_signals.get("scene_init"),
            "player_turn_start": async_signals.get("player_turn_start"),
            "config.changed": async_signals.get("config.changed"),
        }

        self.setup_emitter(scene=self)

        self.world_state.emit()

    @property
    def config(self) -> Config:
        """Returns the current configuration for the scene."""
        return get_config()

    @property
    def main_character(self) -> Actor | None:
        """Returns the main character's actor or None if not available."""
        try:
            return self.get_player_character().actor
        except AttributeError:
            return None

    @property
    def player_character_exists(self) -> bool:
        """Check if a player character exists."""
        try:
            character = self.get_player_character()
            return character is not None and character.is_player
        except AttributeError:
            return False

    @property
    def characters(self):
        """Yield the characters of each actor."""
        for actor in self.actors:
            yield actor.character

    @property
    def all_characters(self) -> Generator[Character, None, None]:
        """
        Returns all characters in the scene, including inactive characters
        """

        for actor in self.actors:
            yield actor.character

        for character in self.inactive_characters.values():
            yield character

    @property
    def all_character_names(self):
        """Return a list of all character names."""
        return [character.name for character in self.all_characters]

    @property
    def npcs(self):
        for actor in self.actors:
            if not actor.character.is_player:
                yield actor.character

    @property
    def character_names(self):
        return [character.name for character in self.characters]

    @property
    def npc_character_names(self):
        """Returns a list of NPC character names."""
        return [character.name for character in self.get_npc_characters()]

    @property
    def has_active_npcs(self):
        """Check if there are active NPC characters."""
        return bool(list(self.get_npc_characters()))

    @property
    def log(self):
        return log

    @property
    def project_name(self) -> str:
        """Return the project name formatted as a lowercase string with hyphens."""
        return self.name.replace(" ", "-").replace("'", "").lower()

    @property
    def save_files(self) -> list[str]:
        """
        Returns list of save files for the current scene (*.json files
        in the save_dir)
        """
        if hasattr(self, "_save_files"):
            return self._save_files

        save_files = []

        for file in os.listdir(self.save_dir):
            if file.endswith(".json"):
                save_files.append(file)

        self._save_files = sorted(save_files)

        return self._save_files

    @property
    def num_history_entries(self):
        """Returns the number of entries in the history."""
        return len(self.history)

    @property
    def prev_actor(self) -> str:
        # will find the first CharacterMessage in history going from the end
        # and return the character name attached to it to determine the actor
        # that most recently spoke

        """Return the name of the most recent actor from history."""
        for idx in range(len(self.history) - 1, -1, -1):
            if isinstance(self.history[idx], CharacterMessage):
                return self.history[idx].character_name

    @property
    def save_dir(self):
        """Return the path to the save directory, creating it if it doesn't exist."""
        saves_dir = os.path.join(
            self.scenes_dir(),
            self.project_name,
        )

        if not os.path.exists(saves_dir):
            os.makedirs(saves_dir)

        return saves_dir

    @property
    def full_path(self):
        """Get the full path of the file."""
        if not self.filename:
            return None

        return os.path.join(self.save_dir, self.filename)

    @property
    def template_dir(self):
        """Returns the path to the templates directory."""
        return os.path.join(self.save_dir, "templates")

    @property
    def nodes_dir(self):
        return os.path.join(self.save_dir, "nodes")

    @property
    def info_dir(self):
        """Return the path to the info directory."""
        return os.path.join(self.save_dir, "info")

    @property
    def auto_save(self):
        """Gets the auto_save configuration setting."""
        return self.config.game.general.auto_save

    @property
    def auto_progress(self):
        """Gets the auto_progress setting from the game configuration."""
        return self.config.game.general.auto_progress

    @property
    def world_state_manager(self) -> WorldStateManager:
        return WorldStateManager(self)

    @property
    def conversation_format(self):
        """Get the conversation format from the agent."""
        return get_agent("conversation").conversation_format

    @property
    def writing_style(self) -> world_state_templates.WritingStyle | None:
        """Returns the writing style template or None if not available."""
        if not self.writing_style_template:
            return None

        try:
            group_uid, template_uid = self.writing_style_template.split("__", 1)
            return self._world_state_templates.find_template(group_uid, template_uid)
        except ValueError:
            return None

    @property
    def max_backscroll(self):
        """Get the maximum backscroll value from the game configuration."""
        return self.config.game.general.max_backscroll

    @property
    def nodes_filename(self):
        """Get the filename for the nodes."""
        return self._nodes_filename or "scene-loop.json"

    @nodes_filename.setter
    def nodes_filename(self, value: str):
        """Set the nodes filename, defaulting to an empty string if None."""
        self._nodes_filename = value or ""

    @property
    def nodes_filepath(self) -> str:
        return os.path.join(self.nodes_dir, self.nodes_filename)

    @property
    def creative_nodes_filename(self):
        return self._creative_nodes_filename or "creative-loop.json"

    @creative_nodes_filename.setter
    def creative_nodes_filename(self, value: str):
        """Set the creative nodes filename."""
        self._creative_nodes_filename = value or ""

    @property
    def creative_nodes_filepath(self) -> str:
        """Get the file path for creative nodes."""
        return os.path.join(self.nodes_dir, self.creative_nodes_filename)

    @property
    def intent(self) -> dict:
        """Returns the intent as a dictionary based on the current phase."""
        phase = self.intent_state.phase
        if not phase:
            return {}

        return {
            "name": self.intent_state.current_scene_type.name,
            "intent": phase.intent,
        }

    @property
    def active_node_graph(self):
        return getattr(self, "node_graph", getattr(self, "creative_node_graph", None))

    def set_intro(self, intro: str):
        """Sets the introduction string."""
        self.intro = intro

    def set_name(self, name: str):
        self.name = name

    def set_title(self, title: str):
        """Set the title of the object."""
        self.title = title

    def connect(self):
        """
        connect scenes to signals
        """
        self.signals["config.changed"].connect(self.on_config_changed)

    def disconnect(self):
        """Disconnects the on_config_changed signal from config.changed."""
        self.signals["config.changed"].disconnect(self.on_config_changed)

    def __del__(self):
        self.disconnect()

    async def on_config_changed(self, event):
        self.emit_status()

    def recent_history(self, max_tokens: int = 2048):
        """Retrieve recent history entries up to a specified token limit."""
        scene = self
        history_legnth = len(scene.history)
        num = 0
        idx = history_legnth - 1
        recent_history = []
        total_tokens = 0

        while idx > -1:
            recent_history.insert(0, scene.history[idx])

            total_tokens += util.count_tokens(scene.history[idx])

            num += 1
            idx -= 1

            if total_tokens >= max_tokens:
                break

        return recent_history

    def push_history(self, messages: list[SceneMessage]):

        """Adds one or more messages to the scene history.
        
        This function processes a list of messages, ensuring that only one
        DirectorMessage from a specific source exists in the history. If a  new
        DirectorMessage is added, any existing one from the same source  is removed.
        Additionally, if a TimePassageMessage is encountered,  the time is advanced
        accordingly. The messages are then added to  the history, and relevant signals
        are sent to update the game loop  with the new messages.
        
        Args:
            messages (list[SceneMessage]): A list of messages to be added
        """
        if isinstance(messages, SceneMessage):
            messages = [messages]

        # there can only ever be one director message in the history
        # so if there is a new DirectorMessage in messages we remove the old one
        # from the history

        for message in messages:
            if isinstance(message, DirectorMessage):
                for idx in range(len(self.history) - 1, -1, -1):
                    if (
                        isinstance(self.history[idx], DirectorMessage)
                        and self.history[idx].source == message.source
                    ):
                        self.history.pop(idx)
                        break

            elif isinstance(message, TimePassageMessage):
                self.advance_time(message.ts)

        self.history.extend(messages)
        self.signals["history_add"].send(
            events.HistoryEvent(
                scene=self,
                event_type="history_add",
                messages=messages,
            )
        )

        loop = asyncio.get_event_loop()
        for message in messages:
            loop.run_until_complete(
                self.signals["game_loop_new_message"].send(
                    events.GameLoopNewMessageEvent(
                        scene=self, event_type="game_loop_new_message", message=message
                    )
                )
            )

    def pop_message(self, message: SceneMessage | int) -> bool:
        """Removes the last message from the history that matches the given message.
        
        This function checks the type of the provided message. If it is an instance of
        SceneMessage, it attempts to remove it from the history. If the message is an
        integer, it first finds the corresponding SceneMessage and then removes it.  If
        the message cannot be found, it returns False. An invalid message type raises
        a ValueError.
        """
        if isinstance(message, SceneMessage):
            try:
                self.history.remove(message)
            except ValueError:
                return False
            return True
        elif isinstance(message, int):
            message = self.find_message(message)
            if message:
                self.history.remove(message)
                return True
            return False
        else:
            raise ValueError("Invalid message type")

    def pop_history(
        self,
        typ: str,
        source: str = None,
        all: bool = False,
        max_iterations: int = None,
        reverse: bool = False,
        meta_hash: int = None,
        **filters,
    ):
        """
        Removes the last message from the history that matches the given typ and source
        """
        iterations = 0

        if not reverse:
            iter_range = range(len(self.history) - 1, -1, -1)
        else:
            iter_range = range(len(self.history))

        to_remove = []

        for idx in iter_range:
            message = self.history[idx]

            if message.typ != typ:
                iterations += 1
                continue

            if source is not None and message.source != source:
                iterations += 1
                continue

            if meta_hash is not None and message.meta_hash != meta_hash:
                iterations += 1
                continue

            # Apply additional filters
            valid = True
            for filter_name, filter_value in filters.items():
                if getattr(message, filter_name, None) != filter_value:
                    valid = False
                    break

            if valid:
                to_remove.append(message)
                if not all:
                    break

            iterations += 1
            if max_iterations and iterations >= max_iterations:
                break

        for message in to_remove:
            self.history.remove(message)

    def find_message(self, typ: str, max_iterations: int = 100, **filters):
        """def find_message(self, typ: str, max_iterations: int = 100, **filters):
        
        Finds the last message in the history that matches the given typ and source.
        This function iterates through the message history in reverse order,  checking
        each message's type against the specified typ. It also applies  any additional
        filters provided as keyword arguments. The iteration stops  if the maximum
        number of iterations is reached, returning None if no  matching message is
        found.
        """
        iterations = 0
        for idx in range(len(self.history) - 1, -1, -1):
            message: SceneMessage = self.history[idx]

            iterations += 1
            if iterations >= max_iterations:
                return None

            if message.typ != typ:
                continue

            for filter_name, filter_value in filters.items():
                if getattr(message, filter_name, None) != filter_value:
                    continue

            return self.history[idx]

    def message_index(self, message_id: int) -> int:
        """Returns the index of the given message in the history."""
        for idx in range(len(self.history) - 1, -1, -1):
            if self.history[idx].id == message_id:
                return idx
        return -1

    def get_message(self, message_id: int) -> SceneMessage:
        """Retrieve the message from history by its ID."""
        for idx in range(len(self.history) - 1, -1, -1):
            if self.history[idx].id == message_id:
                return self.history[idx]

    def last_player_message(self) -> str:
        """Returns the last message from the player."""
        for idx in range(len(self.history) - 1, -1, -1):
            if isinstance(self.history[idx], CharacterMessage):
                if self.history[idx].source == "player":
                    return self.history[idx]

    def last_message_of_type(
        self,
        typ: str | list[str],
        source: str = None,
        max_iterations: int = None,
        stop_on_time_passage: bool = False,
        on_iterate: Callable = None,
        **filters,
    ) -> SceneMessage | None:

        """Return the last message of the specified type and source from the history.
        
        This function iterates through the message history in reverse order, checking
        each message against the provided type and source. It allows for a maximum
        number of iterations and can stop early if a TimePassageMessage is encountered.
        Additional filters can be applied to refine the search based on message
        attributes.
        
        Args:
            typ (str | list[str]): The type of message to find.
            source (str?): The source of the message.
            max_iterations (int?): The maximum number of iterations to search for the message.
            stop_on_time_passage (bool?): If True, the search will stop when a TimePassageMessage is found.
            on_iterate (Callable?): A function to call on each iteration of the search.
            **filters: Any additional keyword arguments will be used to filter the messages against
                their attributes.
        
        Returns:
            SceneMessage | None: The last message matching the criteria, or None if not
                found.
        """
        if not isinstance(typ, list):
            typ = [typ]

        num_iterations = 0

        for idx in range(len(self.history) - 1, -1, -1):
            if max_iterations is not None and num_iterations >= max_iterations:
                return None

            message = self.history[idx]

            if on_iterate:
                on_iterate(message)

            if isinstance(message, TimePassageMessage) and stop_on_time_passage:
                return None

            num_iterations += 1

            if message.typ not in typ or (source and message.source != source):
                continue

            valid = True

            for filter_name, filter_value in filters.items():
                message_value = getattr(message, filter_name, None)
                if message_value != filter_value:
                    valid = False
                    break

            if valid:
                return message

    def collect_messages(
        self,
        typ: str | list[str] = None,
        source: str = None,
        max_iterations: int = 100,
        max_messages: int | None = None,
        stop_on_time_passage: bool = False,
        start_idx: int | None = None,
    ):

        """Collect messages from history that match specified criteria.
        
        This function retrieves messages from the history based on the provided `typ`
        and `source` parameters. It iterates through the history in reverse order,
        collecting messages that meet the criteria until it reaches the specified
        limits of `max_iterations` and `max_messages`. The function also has the option
        to stop collecting messages upon encountering a `TimePassageMessage` if
        `stop_on_time_passage` is set to True.
        
        Args:
            typ (str | list[str]?): The type(s) of messages to collect. Defaults to None.
            source (str?): The source of the messages to collect. Defaults to None.
            max_iterations (int?): The maximum number of iterations to perform. Defaults to 100.
            max_messages (int | None?): The maximum number of messages to collect. Defaults to None.
            stop_on_time_passage (bool?): Whether to stop collecting on time passage messages. Defaults to False.
            start_idx (int | None?): The index to start collecting messages from. Defaults to None.
        
        Returns:
            list: A list of messages that match the specified criteria.
        """
        if typ and not isinstance(typ, list):
            typ = [typ]

        messages = []
        iterations = 0
        collected = 0

        if start_idx is None:
            start_idx = len(self.history) - 1

        for idx in range(start_idx, -1, -1):
            message = self.history[idx]
            if (not typ or message.typ in typ) and (
                not source or message.source == source
            ):
                messages.append(message)
                collected += 1
                if max_messages is not None and collected >= max_messages:
                    break
            if isinstance(message, TimePassageMessage) and stop_on_time_passage:
                break

            iterations += 1
            if iterations >= max_iterations:
                break

        return messages

    def snapshot(
        self,
        lines: int = 3,
        ignore: list[str | SceneMessage] = None,
        start: int = None,
        as_format: str = "movie_script",
    ) -> str:

        """Return a snapshot of the scene history.
        
        This function retrieves a specified number of recent messages from the scene
        history, optionally ignoring certain message types. It processes the `ignore`
        list to ensure it contains valid message types, and then collects the relevant
        messages from the history based on the provided `lines` and `start` parameters.
        The collected messages are formatted according to the specified `as_format`.
        
        Args:
            lines (int?): The number of recent messages to retrieve. Defaults to 3.
            ignore (list[str | SceneMessage]?): A list of message types to ignore. Defaults to None.
            start (int?): The index from which to start collecting messages. Defaults to None.
            as_format (str?): The format in which to return the messages. Defaults to "movie_script".
        
        Returns:
            str: A formatted string containing the collected messages.
        
        Raises:
            ValueError: If the `ignore` list contains invalid types.
        """
        if not ignore:
            ignore = [
                ReinforcementMessage,
                DirectorMessage,
                ContextInvestigationMessage,
            ]
        else:
            # ignore me also be a list of message type strings (e.g. 'director')
            # convert to class types
            _ignore = []
            for item in ignore:
                if isinstance(item, str):
                    _ignore.append(MESSAGE_TYPES.get(item))
                elif isinstance(item, SceneMessage):
                    _ignore.append(item)
                else:
                    raise ValueError(
                        "ignore must be a list of strings or SceneMessage types"
                    )
            ignore = _ignore

        collected = []

        segment = (
            self.history[-lines:] if not start else self.history[: start + 1][-lines:]
        )

        for idx in range(len(segment) - 1, -1, -1):
            if isinstance(segment[idx], tuple(ignore)):
                continue
            collected.insert(0, segment[idx])
            if len(collected) >= lines:
                break

        return "\n".join([message.as_format(as_format) for message in collected])

    async def push_archive(self, entry: ArchiveEntry):

        """Adds an entry to the archive history."""
        self.archived_history.append(entry.model_dump(exclude_none=True))
        await emit_archive_add(self, entry)
        emit(
            "archived_history",
            data={
                "history": [
                    archived_history for archived_history in self.archived_history
                ]
            },
        )

    def edit_message(self, message_id: int, message: str):
        """
        Finds the message in `history` by its id and will update its contents
        """

        for i, _message in enumerate(self.history):
            if _message.id == message_id:
                self.history[i].message = message
                emit("message_edited", self.history[i], id=message_id)
                self.log.info("Message edited", message=message, id=message_id)
                return

    async def add_actor(self, actor: Actor):
        """
        Add an actor to the scene
        """
        self.actors.append(actor)
        actor.scene = self

        if isinstance(actor, Player):
            actor.character.is_player = True

        for actor in self.actors:
            if (
                not isinstance(actor, Player)
                and self.main_character
                and actor.character.introduce_main_character
            ):
                actor.character.introduce_main_character(self.main_character.character)

        if not isinstance(actor, Player):
            if not self.context and actor.character.base_attributes.get(
                "scenario_context"
            ):
                self.context = actor.character.base_attributes["scenario_context"]

            if actor.character.greeting_text and not self.intro:
                self.intro = actor.character.greeting_text
            if not self.name or self.name.lower() == "new scenario":
                self.name = actor.character.name
                self.emit_status()
            if actor.character.base_attributes.get("scenario overview"):
                self.description = actor.character.base_attributes["scenario overview"]

        memory = get_agent("memory")
        await actor.character.commit_to_memory(memory)

    async def remove_character(
        self, character: Character, purge_from_memory: bool = True
    ):

        """Remove a character from the scene.
        
        This function checks if the specified character is active among the  current
        actors. If found, it removes the actor associated with that  character. If the
        character is inactive, it deletes the character  from the inactive_characters
        dictionary. Additionally, if  purge_from_memory is set to True, it purges the
        character from  memory.
        """
        for actor in self.actors:
            if actor.character == character:
                await self.remove_actor(actor)

        if character.name in self.inactive_characters:
            del self.inactive_characters[character.name]

        if purge_from_memory:
            await character.purge_from_memory()

    async def remove_actor(self, actor: Actor):
        """
        Remove an actor from the scene
        """

        for _actor in self.actors:
            if _actor == actor:
                self.actors.remove(_actor)

        actor.character = None

    def get_character(self, character_name: str, partial: bool = False):

        """Retrieve a character object by name, with support for partial matches.
        
        This function checks if the provided character_name is valid and then attempts
        to return the corresponding character object. It first checks for a special
        case of the narrator, then looks in the inactive characters. If not found, it
        iterates through the active actors, returning a match based on exact or partial
        name comparisons.
        
        Args:
            character_name (str): The name of the character to retrieve.
            partial (bool): A flag indicating whether to allow partial name matches.
        
        Returns:
            Character: The character object that matches the given name or None if not found.
        """
        if not character_name:
            return

        if character_name == "__narrator__":
            return self.narrator_character_object

        if character_name in self.inactive_characters:
            return self.inactive_characters[character_name]

        for actor in self.actors:
            if not partial and actor.character.name.lower() == character_name.lower():
                return actor.character
            elif partial and character_name.lower() in actor.character.name.lower():
                return actor.character
            elif partial and actor.character.name.lower() in character_name.lower():
                return actor.character

    def get_player_character(self):
        for actor in self.actors:
            if isinstance(actor, Player):
                return actor.character

        # No active player found, return the first NPC
        for actor in self.actors:
            return actor.character

    def get_npc_characters(self):
        for actor in self.actors:
            if not isinstance(actor, Player):
                yield actor.character

    def num_npc_characters(self) -> int:
        """Return the number of NPC characters."""
        return len(list(self.get_npc_characters()))

    def parse_character_from_line(self, line: str) -> Character:

        """Parse a character from a line of text."""
        for actor in self.actors:
            if actor.character.name.lower() in line.lower():
                return actor.character

    def parse_characters_from_text(
        self, text: str, exclude_active: bool = False
    ) -> list[Character]:

        """Parse characters from a block of text.
        
        This function processes a given text to identify and extract characters  based
        on their active or inactive status. It first converts the text to  lowercase
        and uses regular expressions to match whole words for active  characters if
        `exclude_active` is set to False. It then checks for  inactive characters and
        appends them to the list. Finally, the function  returns a sorted list of
        characters based on the length of their names.
        """
        characters = []
        text = condensed(text.lower())

        # active characters
        if not exclude_active:
            for actor in self.actors:
                # use regex with word boundaries to match whole words
                if re.search(rf"\b{actor.character.name.lower()}\b", text):
                    characters.append(actor.character)

        # inactive characters
        for character in self.inactive_characters.values():
            if re.search(rf"\b{character.name.lower()}\b", text):
                characters.append(character)

        return sorted(characters, key=lambda x: len(x.name))

    def get_characters(self) -> Generator[Character, None, None]:
        """
        Returns a list of all characters in the scene
        """

        for actor in self.actors:
            yield actor.character

    def set_description(self, description: str):
        """Sets the description of the scene."""
        self.description = description

    def get_intro(self, intro: str = None) -> str:
        """
        Returns the intro text of the scene
        """

        if not intro:
            intro = self.intro

        try:
            player_name = self.get_player_character().name
            intro = intro.replace("{{user}}", player_name).replace(
                "{{char}}", player_name
            )
        except AttributeError:
            intro = self.intro

        editor = get_agent("editor")

        if editor.fix_exposition_enabled and editor.fix_exposition_narrator:
            if '"' not in intro and "*" not in intro:
                intro = f"*{intro}*"
            intro = editor.fix_exposition_in_text(intro)

        return intro

    def history_length(self):
        """
        Calculate and return the length of all strings in the history added together.
        """
        return count_tokens(self.history)

    def count_messages(self, message_type: str = None, source: str = None) -> int:

        """Counts the number of messages in the history based on specified criteria.
        
        This function iterates through the `self.history` list and counts messages that
        match the given `message_type` and `source`. If no `message_type` or `source`
        is provided, it returns the total count of messages in the history. The
        function  utilizes conditional checks to filter messages based on the provided
        parameters.
        """
        count = 0

        for message in self.history:
            if message_type and message.typ != message_type:
                continue
            if (
                source
                and message.source != source
                and message.secondary_source != source
            ):
                continue
            count += 1

        return count

    def context_history(self, budget: int = 8192, **kwargs):
        """Generate context and dialogue history based on specified parameters.
        
        This function constructs a history of context and dialogue by processing
        archived and layered history entries. It manages the budget for context and
        dialogue separately, ensuring that the total token count does not exceed
        specified limits. The function also handles various conditions such as keeping
        director messages, context investigations, and chapter labeling, while logging
        warnings for potential issues with the history length.
        
        Args:
            budget (int): The budget for token count, default is 8192.
            **kwargs: Additional parameters to customize the behavior of the function, including:
                - keep_director (bool): Whether to retain director messages.
                - keep_context_investigation (bool): Whether to retain context investigation
                messages.
                - show_hidden (bool): Whether to include hidden messages.
                - include_reinforcements (bool): Whether to include reinforcement messages.
                - assured_dialogue_num (int): Minimum number of dialogue messages to assure.
                - chapter_labels (bool): Whether to include chapter labels in the output.
        
        Returns:
            list: A combined list of context and dialogue history as strings.
        """
        parts_context = []
        parts_dialogue = []

        budget_context = int(0.5 * budget)
        budget_dialogue = int(0.5 * budget)

        keep_director = kwargs.get("keep_director", False)
        keep_context_investigation = kwargs.get("keep_context_investigation", True)
        show_hidden = kwargs.get("show_hidden", False)

        conversation_format = self.conversation_format
        actor_direction_mode = get_agent("director").actor_direction_mode
        layered_history_enabled = get_agent("summarizer").layered_history_enabled
        include_reinforcements = kwargs.get("include_reinforcements", True)
        assured_dialogue_num = kwargs.get("assured_dialogue_num", 5)

        chapter_labels = kwargs.get("chapter_labels", False)
        chapter_numbers = []

        history_len = len(self.history)

        # CONTEXT
        # collect context, ignore where end > len(history) - count
        if (
            not self.layered_history
            or not layered_history_enabled
            or not self.layered_history[0]
        ):
            # no layered history available

            for i in range(len(self.archived_history) - 1, -1, -1):
                archive_history_entry = self.archived_history[i]
                end = archive_history_entry.get("end")

                if end is None:
                    continue

                try:
                    time_message = util.iso8601_diff_to_human(
                        archive_history_entry["ts"], self.ts
                    )
                    text = f"{time_message}: {archive_history_entry['text']}"
                except Exception as e:
                    log.error(
                        "context_history", error=e, traceback=traceback.format_exc()
                    )
                    text = archive_history_entry["text"]

                if count_tokens(parts_context) + count_tokens(text) > budget_context:
                    break

                text = condensed(text)

                parts_context.insert(0, text)

        else:
            # layered history available
            # start with the last layer and work backwards

            next_layer_start = None
            num_layers = len(self.layered_history)

            for i in range(len(self.layered_history) - 1, -1, -1):
                log.debug(
                    "context_history - layered history",
                    i=i,
                    next_layer_start=next_layer_start,
                )

                if not self.layered_history[i]:
                    continue

                k = next_layer_start if next_layer_start is not None else 0

                for layered_history_entry in self.layered_history[i][
                    next_layer_start if next_layer_start is not None else 0 :
                ]:
                    time_message_start = util.iso8601_diff_to_human(
                        layered_history_entry["ts_start"], self.ts
                    )
                    time_message_end = util.iso8601_diff_to_human(
                        layered_history_entry["ts_end"], self.ts
                    )

                    if time_message_start == time_message_end:
                        time_message = time_message_start
                    else:
                        time_message = (
                            f"Start:{time_message_start}, End:{time_message_end}"
                            if time_message_start != time_message_end
                            else time_message_start
                        )
                    text = f"{time_message} {layered_history_entry['text']}"

                    # prepend chapter labels
                    if chapter_labels:
                        chapter_number = f"{num_layers - i}.{k + 1}"
                        text = f"### Chapter {chapter_number}\n{text}"
                        chapter_numbers.append(chapter_number)

                    parts_context.append(text)

                    k += 1

                next_layer_start = layered_history_entry["end"] + 1

            # collect archived history entries that have not yet been
            # summarized to the layered history
            base_layer_start = (
                self.layered_history[0][-1]["end"] + 1
                if self.layered_history[0]
                else None
            )

            if base_layer_start is not None:
                i = 0

                # if chapter labels have been appanded, we need to
                # open a new section for the current scene

                if chapter_labels:
                    parts_context.append("### Current\n")

                for archive_history_entry in self.archived_history[base_layer_start:]:
                    time_message = util.iso8601_diff_to_human(
                        archive_history_entry["ts"], self.ts
                    )

                    text = f"{time_message}: {archive_history_entry['text']}"

                    text = condensed(text)

                    parts_context.append(text)

                    i += 1

        # log.warn if parts_context token count > budget_context
        if count_tokens(parts_context) > budget_context:
            # chop off the top until it fits
            while count_tokens(parts_context) > budget_context:
                parts_context.pop(0)

        # DIALOGUE
        try:
            summarized_to = (
                self.archived_history[-1]["end"] if self.archived_history else 0
            )
        except KeyError:
            # only static archived history entries exist (pre-entered history
            # that doesnt have start and end timestamps)
            summarized_to = 0

        # if summarized_to somehow is bigger than the length of the history
        # since we have no way to determine where they sync up just put as much of
        # the dialogue as possible
        if summarized_to and summarized_to >= history_len:
            log.warning(
                "context_history",
                message="summarized_to is greater than history length - may want to regenerate history",
            )
            summarized_to = 0

        log.debug(
            "context_history", summarized_to=summarized_to, history_len=history_len
        )

        dialogue_messages_collected = 0

        # for message in self.history[summarized_to if summarized_to is not None else 0:]:
        for i in range(len(self.history) - 1, -1, -1):
            message = self.history[i]

            if (
                i < summarized_to
                and dialogue_messages_collected >= assured_dialogue_num
            ):
                break

            if message.hidden and not show_hidden:
                continue

            if isinstance(message, ReinforcementMessage) and not include_reinforcements:
                continue

            elif isinstance(message, DirectorMessage):
                if not keep_director:
                    continue

                if not message.character_name:
                    # skip director messages that are not character specific
                    # TODO: we may want to include these in the future
                    continue

                elif (
                    isinstance(keep_director, str)
                    and message.character_name != keep_director
                ):
                    continue

            elif (
                isinstance(message, ContextInvestigationMessage)
                and not keep_context_investigation
            ):
                continue

            if count_tokens(parts_dialogue) + count_tokens(message) > budget_dialogue:
                break

            parts_dialogue.insert(
                0, message.as_format(conversation_format, mode=actor_direction_mode)
            )

            if isinstance(message, CharacterMessage):
                dialogue_messages_collected += 1

        if count_tokens(parts_context) < 128:
            intro = self.get_intro()
            if intro:
                parts_context.insert(0, intro)

        active_agent_ctx = active_agent.get()
        if active_agent_ctx:
            active_agent_ctx.state["chapter_numbers"] = chapter_numbers

        return list(map(str, parts_context)) + list(map(str, parts_dialogue))

    def delete_message(self, message_id: int):
        """
        Delete a message from the history
        """
        log.debug(f"Deleting message {message_id}")
        for i, message in enumerate(self.history):
            if message.id == message_id:
                self.history.pop(i)
                log.info(f"Deleted message {message_id}")
                emit("remove_message", "", id=message_id)

                if isinstance(message, TimePassageMessage):
                    self.sync_time()
                    self.emit_status()

                break

    def can_auto_save(self):

        """Check if the scene can be autosaved."""
        return self.filename and not self.immutable_save

    def emit_status(self, restored: bool = False):
        """Emit the current status of the scene.
        
        This function gathers various attributes related to the scene, including
        player character information, filenames, project details, and game state.  It
        then emits a structured status message containing this data. The  function also
        logs the scene status for debugging purposes, providing  insights into the
        current scene's time and saved state.
        
        Args:
            restored (bool): Indicates whether the scene has been restored.
        """
        player_character = self.get_player_character()
        emit(
            "scene_status",
            self.name,
            status="started",
            data={
                "path": self.full_path,
                "filename": self.filename,
                "project_name": self.project_name,
                "nodes_filename": self.nodes_filename,
                "creative_nodes_filename": self.creative_nodes_filename,
                "save_files": self.save_files,
                "restore_from": self.restore_from,
                "restored": restored,
                "title": self.title or self.name,
                "environment": self.environment,
                "player_character_name": (
                    player_character.name if player_character else None
                ),
                "inactive_characters": list(self.inactive_characters.keys()),
                "context": self.context,
                "assets": self.assets.dict(),
                "characters": [actor.character.model_dump() for actor in self.actors],
                "character_colors": {
                    character.name: character.color
                    for character in self.get_characters()
                },
                "scene_time": (
                    util.iso8601_duration_to_human(self.ts, suffix="")
                    if self.ts
                    else None
                ),
                "saved": self.saved,
                "auto_save": self.auto_save,
                "auto_progress": self.auto_progress,
                "can_auto_save": self.can_auto_save(),
                "game_state": self.game_state.model_dump(),
                "agent_state": self.agent_state,
                "active_pins": [pin.model_dump() for pin in self.active_pins],
                "experimental": self.experimental,
                "immutable_save": self.immutable_save,
                "description": self.description,
                "intro": self.intro,
                "help": self.help,
                "writing_style_template": self.writing_style_template,
                "intent": self.intent,
            },
        )

        self.log.debug(
            "scene_status",
            scene=self.name,
            scene_time=self.ts,
            human_ts=util.iso8601_duration_to_human(self.ts, suffix="")
            if self.ts
            else None,
            saved=self.saved,
        )

    def set_environment(self, environment: str):
        """Set the environment of the scene."""
        self.environment = environment
        self.emit_status()

    def set_content_context(self, context: str):
        """
        Updates the content context of the scene
        """
        self.context = context
        self.emit_status()

    def advance_time(self, ts: str):
        """Advances the scene's world state by the given iso6801 duration string."""
        log.debug(
            "advance_time",
            ts=ts,
            scene_ts=self.ts,
            duration=isodate.parse_duration(ts),
            scene_duration=isodate.parse_duration(self.ts),
        )

        self.ts = isodate.duration_isoformat(
            isodate.parse_duration(self.ts) + isodate.parse_duration(ts)
        )

    def sync_time(self):
        # reset time
        """Synchronizes the world state based on time passage messages.
        
        This function resets the time to "PT0S" and checks the  archived history for
        the most recent timestamp to establish  a baseline. It then iterates through
        the current history  starting from the end, advancing the world state for each
        TimePassageMessage encountered. Additionally, it logs the  current timestamp
        for debugging purposes. Future adjustments  to the archived history timestamps
        may be necessary.
        """
        self.ts = "PT0S"

        # archived history (if "ts" is set) should provide the base line
        # find the first archived_history entry from the back that has a ts
        # and set that as the base line

        if self.archived_history:
            for i in range(len(self.archived_history) - 1, -1, -1):
                if self.archived_history[i].get("ts"):
                    self.ts = self.archived_history[i]["ts"]
                    break

            end = self.archived_history[-1].get("end", 0)
        else:
            end = 0

        for message in self.history[end:]:
            if isinstance(message, TimePassageMessage):
                self.advance_time(message.ts)

        self.log.debug("sync_time", ts=self.ts)

        # TODO: need to adjust archived_history ts as well
        # but removal also probably means the history needs to be regenerated
        # anyway.

    def fix_time(self):
        """Fix time across the board using the base history as the source of truth."""
        try:
            ts = self.ts
            self._fix_time()
        except Exception:
            log.error("fix_time", exc=traceback.format_exc())
            self.ts = ts

    def _fix_time(self):
        starting_time = "PT0S"

        for archived_entry in self.archived_history:
            if "ts" in archived_entry and "end" not in archived_entry:
                starting_time = archived_entry["ts"]
            elif "end" in archived_entry:
                break

        # store time jumps by index
        time_jumps = []

        for idx, message in enumerate(self.history):
            if isinstance(message, TimePassageMessage):
                time_jumps.append((idx, message.ts))

        # now make the timejumps cumulative, meaning that each time jump
        # will be the sum of all time jumps up to that point
        cumulative_time_jumps = []
        ts = starting_time
        for idx, ts_jump in time_jumps:
            ts = util.iso8601_add(ts, ts_jump)
            cumulative_time_jumps.append((idx, ts))

        try:
            ending_time = cumulative_time_jumps[-1][1]
        except IndexError:
            # no time jumps found
            ending_time = starting_time
            self.ts = ending_time
            return

        # apply time jumps to the archived history
        ts = starting_time
        for _, entry in enumerate(self.archived_history):
            if "end" not in entry:
                continue

            # we need to find best_ts by comparing entry["end"]
            # index to time_jumps (find the closest time jump that is
            # smaller than entry["end"])

            best_ts = None
            for jump_idx, jump_ts in cumulative_time_jumps:
                if jump_idx < entry["end"]:
                    best_ts = jump_ts
                else:
                    break

            if best_ts:
                entry["ts"] = best_ts
                ts = entry["ts"]
            else:
                entry["ts"] = ts

        # finally set scene time to last entry in time_jumps
        log.debug("fix_time", ending_time=ending_time)
        self.ts = ending_time

    def calc_time(self, start_idx: int = 0, end_idx: int = None):

        """Calculates the total ISO 8601 duration from TimePassageMessage instances."""
        ts = "PT0S"
        found = False

        for message in self.history[start_idx:end_idx]:
            if isinstance(message, TimePassageMessage):
                ts = util.iso8601_add(ts, message.ts)
                found = True

        if not found:
            return None

        return ts

    async def load_active_pins(self):

        """Loads active pins from the world state manager."""
        _active_pins = await self.world_state_manager.get_pins(active=True)
        self.active_pins = list(_active_pins.pins.values())

    async def ensure_memory_db(self):
        """Ensure the memory database is set up if not already initialized."""
        memory = get_agent("memory")
        if not memory.db:
            await memory.set_db()

    async def emit_history(self):
        """Emit the game history and introduce the main character to NPCs.
        
        This function clears the screen and introduces the main character to  all non-
        player characters (NPCs) that require it. It retrieves the  introductory
        message using the `get_intro` method and sends it to  the narrator. Finally, it
        emits the history messages, processing  each message to determine if it is
        associated with a character,  and emits the appropriate type and content.
        """
        emit("clear_screen", "")
        # this is mostly to support character cards
        # we introduce the main character to all such characters, replacing
        # the {{ user }} placeholder
        for npc in self.npcs:
            if npc.introduce_main_character:
                npc.introduce_main_character(self.main_character.character)

        # emit intro
        intro: str = self.get_intro()
        self.narrator_message(intro)

        # emit history
        for message in self.history[-self.max_backscroll :]:
            if isinstance(message, CharacterMessage):
                character = self.get_character(message.character_name)
            else:
                character = None
            emit(message.typ, message, character=character)

    async def start(self):
        """Start the scene.
        
        This asynchronous function initializes the scene by ensuring the memory
        database is ready and loading active pins.  It then enters a loop where it
        emits the current world state and processes the scene based on the environment
        type.  Depending on whether the environment is "creative" or not, it loads the
        appropriate node graph and initializes the necessary packages,  subsequently
        running the corresponding loop until an exit or reset condition is met.
        """
        await self.ensure_memory_db()
        await self.load_active_pins()

        self.emit_status()

        first_loop = True

        while True:
            try:
                log.debug(f"Starting scene loop: {self.environment}")

                self.world_state.emit()

                if self.environment == "creative":
                    self.creative_node_graph, _ = load_graph(
                        self.creative_nodes_filename, [self.save_dir]
                    )
                    await initialize_packages(self, self.creative_node_graph)
                    await self._run_creative_loop(init=first_loop)
                else:
                    self.node_graph, _ = load_graph(
                        self.nodes_filename, [self.save_dir]
                    )
                    await initialize_packages(self, self.node_graph)
                    await self._run_game_loop(init=first_loop)
            except ExitScene:
                break
            except RestartSceneLoop:
                pass
            except ResetScene:
                continue

            first_loop = False

            await asyncio.sleep(0.01)

    async def _game_startup(self):
        self.commands = commands.Manager(self)

        await self.signals["scene_init"].send(
            events.SceneStateEvent(scene=self, event_type="scene_init")
        )

    async def _run_game_loop(self, init: bool = True, node_graph=None):
        if init:
            await self._game_startup()
            await self.emit_history()

        self.nodegraph_state = state = GraphState()
        state.data["continue_scene"] = True

        while state.data["continue_scene"] and self.active:
            try:
                await self.node_graph.execute(state)
            except GenerationCancelled:
                state.shared["signal_game_loop"] = False
                state.shared["skip_to_player"] = True
                self.cancel_requested = False
                self.log.warning("Generation cancelled, skipping to player")
            except TalemateInterrupt:
                raise
            except LLMAccuracyError as e:
                self.log.error("game_loop", error=e)
                emit(
                    "system",
                    status="error",
                    message=f"LLM Accuracy Error - The model returned an unexpected response, this may mean this specific model is not suitable for Talemate: {e}",
                )
            except TalemateError as e:
                self.log.error("game_loop", error=e)
            except client.ClientDisabledError as e:
                self.log.error("game_loop", error=e)
                emit(
                    "status",
                    status="error",
                    message=f"{e.client.name} is disabled and cannot be used.",
                )
                state.shared["signal_game_loop"] = False
                state.shared["skip_to_player"] = True
            except Exception as e:
                self.log.error(
                    "game_loop",
                    error=e,
                    unhandled=True,
                    traceback=traceback.format_exc(),
                )
                emit("system", status="error", message=f"Unhandled Error: {e}")

    async def _run_creative_loop(self, init: bool = True):
        """Runs the creative loop for the node graph.
        
        This function initializes the node graph state and enters a loop that
        continues executing the creative node graph as long as the scene is  set to
        continue and the loop is active. It handles various exceptions  that may arise
        during execution, logging errors and emitting system  messages as necessary.
        The loop will terminate when the scene is no  longer set to continue or if the
        active state changes.
        
        Args:
            init (bool): A flag indicating whether to initialize the loop.
        """
        await self.emit_history()

        self.nodegraph_state = state = GraphState()
        state.data["continue_scene"] = True

        while state.data["continue_scene"] and self.active:
            try:
                await self.creative_node_graph.execute(state)
            except GenerationCancelled:
                self.cancel_requested = False
                continue
            except TalemateInterrupt:
                raise
            except LLMAccuracyError as e:
                self.log.error("creative_loop", error=e)
                emit(
                    "system",
                    status="error",
                    message=f"LLM Accuracy Error - The model returned an unexpected response, this may mean this specific model is not suitable for Talemate: {e}",
                )
            except TalemateError as e:
                self.log.error("creative_loop", error=e)
            except Exception as e:
                self.log.error(
                    "creative_loop",
                    error=e,
                    unhandled=True,
                    traceback=traceback.format_exc(),
                )
                emit("system", status="error", message=f"Unhandled Error: {e}")

        return

    def set_new_memory_session_id(self):
        """Sets a new memory session ID and logs the change."""
        self.saved_memory_session_id = self.memory_session_id
        self.memory_session_id = str(uuid.uuid4())[:10]
        log.debug(
            "set_new_memory_session_id",
            saved_memory_session_id=self.saved_memory_session_id,
            memory_session_id=self.memory_session_id,
        )
        self.emit_status()

    async def save(
        self,
        save_as: bool = False,
        auto: bool = False,
        force: bool = False,
        copy_name: str = None,
    ):
        """
        Saves the scene data, conversation history, archived history, and characters to a json file.
        """

        if self.immutable_save and not save_as and not force:
            save_as = True

        if copy_name:
            save_as = True

        if save_as:
            self.filename = copy_name

        if not self.name and not auto:
            raise TalemateError("Scene has no name, cannot save")

        elif not self.filename and not auto:
            self.filename = str(uuid.uuid4())[:10]
            self.filename = self.filename.replace(" ", "-").lower() + ".json"

        if self.filename and not self.filename.endswith(".json"):
            self.filename = f"{self.filename}.json"

        elif not self.filename or not self.name and auto:
            # scene has never been saved, don't auto save
            return

        if save_as:
            self.immutable_save = False
            memory_agent = get_agent("memory")
            memory_agent.close_db(self)
            self.memory_id = str(uuid.uuid4())[:10]
            await self.commit_to_memory()

        self.set_new_memory_session_id()

        saves_dir = self.save_dir

        log.info("Saving", filename=self.filename, saves_dir=saves_dir, auto=auto)

        # Generate filename with date and normalized character name
        filepath = os.path.join(saves_dir, self.filename)

        # Create a dictionary to store the scene data
        scene_data = self.serialize

        if not auto:
            emit("status", status="success", message="Saved scene")

        with open(filepath, "w") as f:
            json.dump(scene_data, f, indent=2, cls=save.SceneEncoder)

        self.saved = True

        if hasattr(self, "_save_files"):
            delattr(self, "_save_files")

        self.emit_status()

        # add this scene to recent scenes in config
        await self.add_to_recent_scenes()

    async def save_restore(self, filename: str):

        """Serializes the scene to a specified file."""
        serialized = self.serialize
        serialized["immutable_save"] = True
        serialized["memory_session_id"] = str(uuid.uuid4())[:10]
        serialized["saved_memory_session_id"] = self.memory_session_id
        serialized["memory_id"] = str(uuid.uuid4())[:10]
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, "w") as f:
            json.dump(serialized, f, indent=2, cls=save.SceneEncoder)

    async def add_to_recent_scenes(self):
        """Add the current scene to the recent scenes list."""
        log.debug("add_to_recent_scenes", filename=self.filename)
        config = get_config()
        config.recent_scenes.push(self)
        await config.set_dirty()

    async def commit_to_memory(self):
        # will recommit scene to long term memory

        memory = get_agent("memory")
        memory.drop_db()
        await memory.set_db()

        for ah in self.archived_history:
            ts = ah.get("ts", "PT1S")

            if not ah.get("ts"):
                ah["ts"] = ts

            await emit_archive_add(self, ArchiveEntry(**ah))

        for character in self.characters:
            await character.commit_to_memory(memory)

        await self.world_state.commit_to_memory(memory)

    def reset(self):
        # remove messages
        """Resets the history and world state."""
        self.history = []

        # clear out archived history, but keep pre-established history
        self.archived_history = [
            ah for ah in self.archived_history if ah.get("end") is None
        ]

        self.world_state.reset()

        self.filename = ""

    async def remove_all_actors(self):
        """Remove all actors and reset their characters to None."""
        for actor in self.actors:
            actor.character = None

        self.actors = []

    async def reset_memory(self):
        """Resets the memory session and commits changes."""
        memory_agent = get_agent("memory")
        memory_agent.close_db(self)
        self.memory_id = str(uuid.uuid4())[:10]
        await self.commit_to_memory()

        self.set_new_memory_session_id()

    async def restore(self, save_as: str | None = None):
        try:
            self.log.info("Restoring", source=self.restore_from)

            restore_from = self.restore_from

            if not self.restore_from:
                self.log.error("No save file specified to restore from.")
                return

            self.reset()
            self.inactive_characters = {}
            await self.remove_all_actors()

            from talemate.load import load_scene

            await load_scene(
                self,
                os.path.join(self.save_dir, self.restore_from),
                get_agent("conversation").client,
            )

            await self.reset_memory()

            if save_as:
                self.restore_from = restore_from
                await self.save(save_as=True, copy_name=save_as)
            else:
                self.filename = None
            self.emit_status(restored=True)

            interaction_state = interaction.get()

            if interaction_state:
                # Break and restart the game loop
                interaction_state.reset_requested = True

        except Exception as e:
            self.log.error("restore", error=e, traceback=traceback.format_exc())

    def sync_restore(self, *args, **kwargs):
        """Synchronously restore using the event loop."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.restore())

    @property
    def serialize(self) -> dict:
        """Serialize the scene into a dictionary format."""
        scene = self
        return {
            "description": scene.description,
            "intro": scene.intro,
            "name": scene.name,
            "title": scene.title,
            "history": scene.history,
            "environment": scene.environment,
            "archived_history": scene.archived_history,
            "layered_history": scene.layered_history,
            "characters": [actor.character.model_dump() for actor in scene.actors],
            "inactive_characters": {
                name: character.model_dump()
                for name, character in scene.inactive_characters.items()
            },
            "context": scene.context,
            "world_state": scene.world_state.model_dump(),
            "game_state": scene.game_state.model_dump(),
            "agent_state": scene.agent_state,
            "intent_state": scene.intent_state.model_dump(),
            "assets": scene.assets.dict(),
            "memory_id": scene.memory_id,
            "memory_session_id": scene.memory_session_id,
            "saved_memory_session_id": scene.saved_memory_session_id,
            "immutable_save": scene.immutable_save,
            "ts": scene.ts,
            "help": scene.help,
            "experimental": scene.experimental,
            "writing_style_template": scene.writing_style_template,
            "restore_from": scene.restore_from,
            "nodes_filename": scene._nodes_filename,
            "creative_nodes_filename": scene._creative_nodes_filename,
        }

    @property
    def json(self):
        """Return the serialized JSON representation of the object."""
        return json.dumps(self.serialize, indent=2, cls=save.SceneEncoder)

    def interrupt(self):
        self.cancel_requested = True

    def continue_actions(self):
        """Handles cancellation of actions if requested."""
        if self.cancel_requested:
            self.cancel_requested = False
            raise GenerationCancelled("action cancelled")


Character.model_rebuild()
