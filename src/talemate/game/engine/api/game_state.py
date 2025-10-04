from typing import TYPE_CHECKING

import talemate.game.engine.api.schema as schema
from talemate.game.engine.api.base import ScopedAPI

if TYPE_CHECKING:
    from talemate.game.state import GameState

__all__ = ["create"]


def create(game_state: "GameState") -> "ScopedAPI":
    """Creates an API for managing game state."""
    class API(ScopedAPI):
        help_text = """Functions for game state management"""

        ### Variables

        def has_var(self, key: str) -> bool:

            """Check if a variable exists.
            
            Args:
                key (str): The name of the variable.
            
            Returns:
                bool: Whether the variable exists.
            """
            validated = schema.VariableSchema(key=key)
            return game_state.has_var(validated.key)

        def get_var(
            self, key: str, default: str | int | float | bool | None = None
        ) -> str | int | float | bool | None:

            """Returns the value of a variable or a default value if it doesn't exist.
            
            Args:
                key (str): The name of the variable.
                default (str | int | float | bool | None): The value to return if the variable doesn't exist.
            
            Returns:
                str | int | float | bool | None: The value of the variable.
            """
            validated = schema.VariableSchema(key=key, value=default)
            return game_state.get_var(validated.key, validated.value)

        def set_var(
            self, key: str, value: str | int | float | bool | None, commit: bool = False
        ):

            """Sets the value of a variable in the memory store."""
            validated = schema.VariableSchema(key=key, value=value)
            game_state.set_var(validated.key, validated.value, commit=commit)

        def get_or_set_var(
            self, key: str, value: str | int | float | bool | None, commit: bool = False
        ) -> str | int | float | bool | None:

            """Get the value of a variable or set it if it doesn't exist."""
            validated = schema.VariableSchema(key=key, value=value)
            return game_state.get_or_set_var(
                validated.key, validated.value, commit=commit
            )

        def unset_var(self, key: str):

            """Removes a variable from the game state."""
            validated = schema.VariableSchema(key=key)
            game_state.unset_var(validated.key)

    return API()
