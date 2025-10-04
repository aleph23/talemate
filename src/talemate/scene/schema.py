import pydantic

from talemate.world_state import WorldState
from talemate.game.state import GameState


__all__ = ["SceneType", "ScenePhase", "SceneIntent", "SceneState"]


def make_default_types() -> list["SceneType"]:
    """Create a default list of SceneType objects."""
    return {
        "roleplay": SceneType(
            id="roleplay",
            name="Roleplay",
            description="Freeform dialogue between one or more characters with occasional narration.",
        )
    }


def make_default_phase() -> "ScenePhase":
    """Creates a default ScenePhase with a roleplay type."""
    default_type = make_default_types().get("roleplay")
    return ScenePhase(scene_type=default_type.id)


class SceneType(pydantic.BaseModel):
    id: str
    name: str
    description: str
    instructions: str | None = None


class ScenePhase(pydantic.BaseModel):
    scene_type: str
    intent: str | None = None


class SceneIntent(pydantic.BaseModel):
    scene_types: dict[str, SceneType] | None = pydantic.Field(
        default_factory=make_default_types
    )
    intent: str | None = None
    phase: ScenePhase | None = pydantic.Field(default_factory=make_default_phase)
    start: int = 0

    @property
    def current_scene_type(self) -> SceneType:
        """Returns the current scene type based on the phase."""
        return self.scene_types[self.phase.scene_type]

    @property
    def active(self) -> bool:
        """Return True if either intent or phase is set."""
        return self.intent or self.phase

    def get_scene_type(self, scene_type_id: str) -> SceneType:
        """Retrieve the SceneType associated with the given scene_type_id."""
        return self.scene_types[scene_type_id]


class SceneState(pydantic.BaseModel):
    world_state: "WorldState | None" = None
    game_state: "GameState | None" = None
    agent_state: dict | None = None
    intent_state: SceneIntent | None = None

    def model_dump(self, **kwargs):
        """Return a model dump excluding None values."""
        return super().model_dump(exclude_none=True)
