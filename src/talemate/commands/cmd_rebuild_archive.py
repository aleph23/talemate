from talemate.commands.base import TalemateCommand
from talemate.commands.manager import register
from talemate.emit import emit
from talemate.instance import get_agent


@register
class CmdRebuildArchive(TalemateCommand):
    """
    Command class for the 'rebuild_archive' command
    """

    name = "rebuild_archive"
    description = "Rebuilds the archive of the scene"
    aliases = ["rebuild"]

    async def run(self):
        """Rebuilds the historical archive using the summarizer agent.
        
        This function retrieves the summarizer and memory agents, ensuring that  a
        summarizer is available before proceeding. It clears the archived history
        while preserving pre-established entries, then iteratively rebuilds the
        historical archive by calling the summarizer's build_archive method.  The
        progress is emitted as status updates until the archive is fully rebuilt,
        after which the scene's state is committed to memory.
        """
        summarizer = get_agent("summarizer")
        memory = get_agent("memory")

        if not summarizer:
            self.system_message("No summarizer found")
            return True

        # clear out archived history, but keep pre-established history
        self.scene.archived_history = [
            ah for ah in self.scene.archived_history if ah.get("end") is None
        ]

        self.scene.ts = "PT0S"

        memory.delete({"typ": "history"})

        entries = 0
        total_entries = summarizer.agent.estimated_entry_count
        while True:
            emit(
                "status",
                message=f"Rebuilding historical archive... {entries}/{total_entries}",
                status="busy",
            )
            more = await summarizer.agent.build_archive(self.scene)
            self.scene.sync_time()

            entries += 1

            if not more:
                break

        self.scene.sync_time()
        await self.scene.commit_to_memory()
        emit("status", message="Historical archive rebuilt", status="success")
