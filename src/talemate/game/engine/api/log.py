import structlog

from talemate.game.engine.api.base import ScopedAPI

__all__ = ["create"]


def create(log: structlog.BoundLogger) -> "ScopedAPI":
    """Create a LogAPI instance with logging methods."""
    class LogAPI(ScopedAPI):
        def info(self, event, *args, **kwargs):
            """Logs an informational message with the given event and arguments."""
            log.info(event, *args, **kwargs)

        def debug(self, event, *args, **kwargs):
            log.debug(event, *args, **kwargs)

        def error(self, event, *args, **kwargs):
            """Log an error event with optional arguments."""
            log.error(event, *args, **kwargs)

        def warning(self, event, *args, **kwargs):
            """Log a warning message with the given event and arguments."""
            log.warning(event, *args, **kwargs)

    return LogAPI()
