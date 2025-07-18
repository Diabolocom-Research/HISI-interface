"""Real-time Automatic Speech Recognition (ASR) interface with modular architecture."""

__version__ = "0.1.0"
__author__ = "ASR Interface Contributors"
__email__ = "contributors@example.com"

from .core.config import ASRConfig
from .core.protocols import ASRProcessor, ModelLoader
from .core.store import ASRComponentsStore

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "ASRConfig",
    "ASRProcessor",
    "ModelLoader",
    "ASRComponentsStore",
]
# Test comment
