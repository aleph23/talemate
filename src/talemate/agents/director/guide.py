from typing import TYPE_CHECKING
import structlog
from functools import wraps
import dataclasses
from talemate.agents.base import (
    set_processing as _set_processing,
    AgentAction,
    AgentActionConfig,
    AgentTemplateEmission,
    DynamicInstruction,
)
from talemate.agents.context import active_agent
from talemate.prompts import Prompt
import talemate.emit.async_signals
from talemate.util import strip_partial_sentences

if TYPE_CHECKING:
    from talemate.tale_mate import Character
    from talemate.agents.summarize.analyze_scene import SceneAnalysisEmission

log = structlog.get_logger()

talemate.emit.async_signals.register(
    "agent.director.guide.before_generate",
    "agent.director.guide.inject_instructions",
    "agent.director.guide.generated",
)


@dataclasses.dataclass
class DirectorGuidanceEmission(AgentTemplateEmission):
    pass


def set_processing(fn):
    """
    Custom decorator that emits the agent status as processing while the function
    is running and then emits the result of the function as a DirectorGuidanceEmission
    """

    @_set_processing
    @wraps(fn)
    async def wrapper(self, *args, **kwargs):
        """Wraps a function to emit signals before and after its execution."""
        emission: DirectorGuidanceEmission = DirectorGuidanceEmission(agent=self)

        await talemate.emit.async_signals.get(
            "agent.director.guide.before_generate"
        ).send(emission)
        await talemate.emit.async_signals.get(
            "agent.director.guide.inject_instructions"
        ).send(emission)

        agent_context = active_agent.get()
        agent_context.state["dynamic_instructions"] = emission.dynamic_instructions

        response = await fn(self, *args, **kwargs)
        emission.response = response
        await talemate.emit.async_signals.get("agent.director.guide.generated").send(
            emission
        )
        return emission.response

    return wrapper


class GuideSceneMixin:
    """
    Director agent mixin that provides functionality for automatically guiding
    the actors or the narrator during the scene progression.
    """

    @classmethod
    def add_actions(cls, actions: dict[str, AgentAction]):
        """Adds actions related to guiding scene progression to the provided actions
        dictionary."""
        actions["guide_scene"] = AgentAction(
            enabled=False,
            container=True,
            can_be_disabled=True,
            quick_toggle=True,
            experimental=True,
            label="Guide Scene",
            icon="mdi-lightbulb",
            description="Guide actors and the narrator during the scene progression. This uses the summarizer agent's scene analysis, which needs to be enabled for this to work.",
            config={
                "guide_actors": AgentActionConfig(
                    type="bool",
                    label="Guide actors",
                    description="Guide the actors in the scene. This happens during every actor turn.",
                    value=True,
                ),
                "guide_narrator": AgentActionConfig(
                    type="bool",
                    label="Guide narrator",
                    description="Guide the narrator during the scene. This happens during the narrator's turn.",
                    value=True,
                ),
                "guidance_length": AgentActionConfig(
                    type="text",
                    label="Max. Guidance Length",
                    description="The maximum length of the guidance to provide to the actors. This text will be inserted very close to end of the prompt. Selecting bigger values can have a detremental effect on the quality of generation.",
                    value="384",
                    choices=[
                        {"label": "Tiny (128)", "value": "128"},
                        {"label": "Short (256)", "value": "256"},
                        {"label": "Brief (384)", "value": "384"},
                        {"label": "Medium (512)", "value": "512"},
                        {"label": "Medium Long (768)", "value": "768"},
                        {"label": "Long (1024)", "value": "1024"},
                    ],
                ),
                "cache_guidance": AgentActionConfig(
                    type="bool",
                    label="Cache guidance",
                    description="Will not regenerate the guidance until the scene moves forward or the analysis changes.",
                    value=False,
                    quick_toggle=True,
                ),
            },
        )

    # config property helpers

    @property
    def guide_scene(self) -> bool:
        """Returns whether the guide scene action is enabled."""
        return self.actions["guide_scene"].enabled

    @property
    def guide_actors(self) -> bool:
        """Returns the value of guide_actors from the guide_scene configuration."""
        return self.actions["guide_scene"].config["guide_actors"].value

    @property
    def guide_narrator(self) -> bool:
        """Gets the value of guide_narrator from the guide_scene configuration."""
        return self.actions["guide_scene"].config["guide_narrator"].value

    @property
    def guide_scene_guidance_length(self) -> int:
        """Returns the guidance length for the guide scene."""
        return int(self.actions["guide_scene"].config["guidance_length"].value)

    @property
    def guide_scene_cache_guidance(self) -> bool:
        """Returns the cache guidance value for the guide scene."""
        return self.actions["guide_scene"].config["cache_guidance"].value

    # signal connect

    def connect(self, scene):
        super().connect(scene)
        talemate.emit.async_signals.get(
            "agent.summarization.scene_analysis.after"
        ).connect(self.on_summarization_scene_analysis_after)
        talemate.emit.async_signals.get(
            "agent.summarization.scene_analysis.cached"
        ).connect(self.on_summarization_scene_analysis_after)
        talemate.emit.async_signals.get(
            "agent.editor.revision-analysis.before"
        ).connect(self.on_editor_revision_analysis_before)

    async def on_summarization_scene_analysis_after(
        self, emission: "SceneAnalysisEmission"
    ):
        """Process scene analysis emissions to provide guidance based on analysis type.
        
        This asynchronous function evaluates the type of scene analysis received and
        determines the appropriate guidance to provide. It checks for cached guidance
        and, if not available, generates new guidance based on the analysis type,
        either for narration or conversation. The function also updates the context
        states and caches the guidance for future use.
        
        Args:
            emission (SceneAnalysisEmission): The emission object containing analysis
        """
        if not self.guide_scene:
            return

        guidance = None

        cached_guidance = await self.get_cached_guidance(emission.response)

        if emission.analysis_type == "narration" and self.guide_narrator:
            if cached_guidance:
                guidance = cached_guidance
            else:
                guidance = await self.guide_narrator_off_of_scene_analysis(
                    emission.response,
                    response_length=self.guide_scene_guidance_length,
                )

            if not guidance:
                log.warning("director.guide_scene.narration: Empty resonse")
                return

            self.set_context_states(narrator_guidance=guidance)

        elif emission.analysis_type == "conversation" and self.guide_actors:
            if cached_guidance:
                guidance = cached_guidance
            else:
                guidance = await self.guide_actor_off_of_scene_analysis(
                    emission.response,
                    emission.template_vars.get("character"),
                    response_length=self.guide_scene_guidance_length,
                )

            if not guidance:
                log.warning("director.guide_scene.conversation: Empty resonse")
                return

            self.set_context_states(actor_guidance=guidance)

        if guidance:
            await self.set_cached_guidance(
                emission.response,
                guidance,
                emission.analysis_type,
                emission.template_vars.get("character"),
            )

    async def on_editor_revision_analysis_before(self, emission: AgentTemplateEmission):
        """Handles analysis before editor revision by appending cached guidance to dynamic
        instructions."""
        cached_guidance = await self.get_cached_guidance(emission.response)
        if cached_guidance:
            emission.dynamic_instructions.append(
                DynamicInstruction(title="Guidance", content=cached_guidance)
            )

    # helpers

    def _cache_key(self) -> str:
        return "cached_guidance"

    async def get_cached_guidance(self, analysis: str | None = None) -> str | None:

        """Returns the cached guidance for the specified analysis.
        
        This function checks if there is any cached guidance available. If the cache is
        empty, it returns None.  It retrieves the cached guidance using a generated
        cache key. If an analysis is provided, it verifies  the fingerprint against the
        cached data. If the fingerprints match or no analysis is provided, it  returns
        the corresponding guidance from the cache.
        """
        if not self.guide_scene_cache_guidance:
            return None

        key = self._cache_key()
        cached_guidance = self.get_scene_state(key)

        if cached_guidance:
            if not analysis:
                return cached_guidance.get("guidance")
            elif cached_guidance.get("fp") == self.context_fingerprint(
                extra=[analysis]
            ):
                return cached_guidance.get("guidance")

        return None

    async def set_cached_guidance(
        self,
        analysis: str,
        guidance: str,
        analysis_type: str,
        character: "Character | None" = None,
    ):
        """Sets the cached guidance for the specified analysis."""
        key = self._cache_key()
        self.set_scene_states(
            **{
                key: {
                    "fp": self.context_fingerprint(extra=[analysis]),
                    "guidance": guidance,
                    "analysis_type": analysis_type,
                    "character": character.name if character else None,
                }
            }
        )

    async def get_cached_character_guidance(self, character_name: str) -> str | None:
        """Retrieve cached guidance for a specified character."""
        key = self._cache_key()
        cached_guidance = self.get_scene_state(key)

        if not cached_guidance:
            return None

        if (
            cached_guidance.get("character") == character_name
            and cached_guidance.get("analysis_type") == "conversation"
        ):
            return cached_guidance.get("guidance")

        return None

    # methods

    @set_processing
    async def guide_actor_off_of_scene_analysis(
        self, analysis: str, character: "Character", response_length: int = 256
    ):
        """
        Guides the actor based on the scene analysis.
        """

        log.debug(
            "director.guide_actor_off_of_scene_analysis",
            analysis=analysis,
            character=character,
        )
        response = await Prompt.request(
            "director.guide-conversation",
            self.client,
            f"direction_{response_length}",
            vars={
                "analysis": analysis,
                "scene": self.scene,
                "character": character,
                "response_length": response_length,
                "max_tokens": self.client.max_token_length,
            },
        )
        return strip_partial_sentences(response).strip()

    @set_processing
    async def guide_narrator_off_of_scene_analysis(
        self, analysis: str, response_length: int = 256
    ):

        """Guides the narrator based on the scene analysis."""
        log.debug("director.guide_narrator_off_of_scene_analysis", analysis=analysis)

        response = await Prompt.request(
            "director.guide-narration",
            self.client,
            f"direction_{response_length}",
            vars={
                "analysis": analysis,
                "scene": self.scene,
                "response_length": response_length,
                "max_tokens": self.client.max_token_length,
            },
        )
        return strip_partial_sentences(response).strip()
