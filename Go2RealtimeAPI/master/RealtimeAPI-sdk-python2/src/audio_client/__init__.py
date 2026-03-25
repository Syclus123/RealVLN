"""Audio WebSocket Client for Air Gateway."""

from .sdk import AudioWebSocketClient, EventHandler, setup_logging
from .config import ClientConfig

__all__ = [
    "AudioWebSocketClient",
    "ClientConfig",
    "EventHandler",
    "setup_logging",
]
__version__ = "0.1.0"
