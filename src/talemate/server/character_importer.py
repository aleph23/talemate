import asyncio
import json

import pydantic
import structlog

from talemate.load import transfer_character

log = structlog.get_logger("talemate.server.character_importer")


class ListCharactersData(pydantic.BaseModel):
    scene_path: str


class ImportCharacterData(pydantic.BaseModel):
    scene_path: str
    character_name: str


class CharacterImporterServerPlugin:
    router = "character_importer"

    def __init__(self, websocket_handler):
        self.websocket_handler = websocket_handler

    @property
    def scene(self):
        """Get the current scene from the websocket handler."""
        return self.websocket_handler.scene

    async def handle(self, data: dict):
        """Handles character importer actions based on the provided data."""
        log.info("Character importer action", action=data.get("action"))

        fn = getattr(self, f"handle_{data.get('action')}", None)

        if fn is None:
            return

        await fn(data)

    async def handle_list_characters(self, data):
        """Handles and sorts character data from a scene file."""
        list_characters_data = ListCharactersData(**data)

        scene_path = list_characters_data.scene_path

        with open(scene_path, "r") as f:
            scene_data = json.load(f)

        sorted_characters = scene_data.get("characters", [])

        # sort by name

        sorted_characters = sorted(
            sorted_characters,
            key=lambda character: character["name"].lower(),
        )

        self.websocket_handler.queue_put(
            {
                "type": "character_importer",
                "action": "list_characters",
                "characters": [character["name"] for character in sorted_characters],
            }
        )

        await asyncio.sleep(0)

    async def handle_import(self, data):
        """Handles the import of character data."""
        import_character_data = ImportCharacterData(**data)

        scene = self.websocket_handler.scene

        await transfer_character(
            scene,
            import_character_data.scene_path,
            import_character_data.character_name,
        )

        scene.emit_status()

        self.websocket_handler.queue_put(
            {
                "type": "character_importer",
                "action": "import_character_done",
            }
        )
