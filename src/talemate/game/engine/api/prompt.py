from typing import TYPE_CHECKING

import pydantic

from talemate.client.base import ClientBase
from talemate.game.engine.api.base import ScopedAPI, run_async
from talemate.prompts.base import Prompt

if TYPE_CHECKING:
    from talemate.tale_mate import Scene


def create(scene: "Scene", client: "ClientBase") -> "ScopedAPI":
    """Creates an API instance for the given scene and client."""
    class API(ScopedAPI):
        def request(
            self,
            template_name: str,
            dedupe_enabled: bool = True,
            kind: str = "create",
            **kwargs,
        ) -> str:

            """Renders a prompt template and sends it to the LLM for generation."""
            class Arguments(pydantic.BaseModel):
                template_name: str
                dedupe_enabled: bool
                kind: str
                kwargs: dict

            validated = Arguments(
                template_name=template_name,
                dedupe_enabled=dedupe_enabled,
                kind=kind,
                kwargs=kwargs,
            )

            prompt = Prompt.get(validated.template_name, validated.kwargs)
            prompt.client = client
            prompt.dedupe_enabled = validated.dedupe_enabled
            return run_async(prompt.send(client, validated.kind))

    return API()
