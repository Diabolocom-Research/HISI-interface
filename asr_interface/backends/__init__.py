"""ASR backends and model loaders."""

from ..core.protocols import ASRBase
from .registry import MODEL_LOADERS
from .whisper_online_processor import SAMPLING_RATE, OnlineASRProcessor

__all__ = [
    "ASRBase",
    "OnlineASRProcessor",
    "SAMPLING_RATE",
    "MODEL_LOADERS",
]

# Try to import MLX Whisper loader (optional dependency)
try:
    from .mlx_whisper_loader import MLXWhisperLoader

    __all__.append("MLXWhisperLoader")
except ImportError:
    # MLX Whisper not available, skip it
    pass

# Try to import Whisper Timestamped loader (optional dependency)
try:
    from .whisper_timestamped_loader import WhisperTimestampedLoader

    __all__.append("WhisperTimestampedLoader")
except ImportError:
    # Whisper Timestamped not available, skip it
    pass
