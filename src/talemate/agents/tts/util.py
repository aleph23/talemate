from pathlib import Path
from typing import TYPE_CHECKING
import structlog

from .schema import TALEMATE_ROOT, Voice, VoiceProvider

from .voice_library import get_instance

if TYPE_CHECKING:
    from talemate.tale_mate import Scene

log = structlog.get_logger("talemate.agents.tts.util")

__all__ = [
    "voice_parameter",
    "voice_is_talemate_asset",
    "voice_is_scene_asset",
    "get_voice",
]


def voice_parameter(
    voice: Voice, provider: VoiceProvider, name: str
) -> str | float | int | bool | None:
    """Get a parameter from the voice or provider."""
    if name in voice.parameters:
        return voice.parameters[name]
    return provider.default_parameters.get(name)


def voice_is_talemate_asset(
    voice: Voice, provider: VoiceProvider
) -> tuple[bool, Path | None]:

    """Check if the voice is a Talemate asset.
    
    This function verifies whether the provided voice is associated with a
    Talemate asset by checking the file upload permissions of the provider  and
    resolving the path of the voice's provider ID. It ensures that the  resolved
    path is within the Talemate root directory. If any errors occur  during path
    resolution, they are logged, and the function returns  appropriate values
    indicating the asset's validity.
    """
    if not provider.allow_file_upload:
        return False, None

    path = Path(voice.provider_id)
    if not path.is_absolute():
        path = TALEMATE_ROOT / path
    try:
        resolved = path.resolve(strict=False)
    except Exception as e:
        log.error(
            "voice_is_talemate_asset - invalid path",
            error=e,
            voice_id=voice.provider_id,
        )
        return False, None

    root = TALEMATE_ROOT.resolve()
    log.debug(
        "voice_is_talemate_asset - resolved", resolved=str(resolved), root=str(root)
    )
    if not str(resolved).startswith(str(root)):
        return False, None

    return True, resolved


def voice_is_scene_asset(voice: Voice, provider: VoiceProvider) -> bool:

    """Check if the voice is a scene asset."""
    is_talemate_asset, resolved = voice_is_talemate_asset(voice, provider)
    if not is_talemate_asset:
        return False

    SCENES_DIR = TALEMATE_ROOT / "scenes"

    if str(resolved).startswith(str(SCENES_DIR.resolve())):
        return True

    return False


def get_voice(scene: "Scene", voice_id: str) -> Voice | None:

    """def get_voice(scene: "Scene", voice_id: str) -> Voice | None:
    Return a Voice by voice_id, preferring the scene's library.  The function first
    checks if a scene is provided and if it has a  voice_library attribute. If so,
    it attempts to retrieve the voice  using the given voice_id. If the voice is
    not found or an error  occurs, it falls back to the global voice library
    instance to  perform the lookup. Errors during the lookup process are logged
    for debugging purposes.
    
    Args:
        scene: Scene instance or ``None``.
        voice_id: The fully-qualified voice identifier (``provider:provider_id``)."""
    try:
        if scene and getattr(scene, "voice_library", None):
            voice = scene.voice_library.get_voice(voice_id)
            if voice:
                return voice
    except Exception as e:
        log.error("get_voice - scene lookup failed", error=e)

    try:
        return get_instance().get_voice(voice_id)
    except Exception as e:
        log.error("get_voice - global lookup failed", error=e)
        return None
