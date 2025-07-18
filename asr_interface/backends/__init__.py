"""ASR backends and model loaders."""

from .whisper_online_processor import OnlineASRProcessor, asr_factory, SAMPLING_RATE
from .registry import MODEL_LOADERS

__all__ = [
    "OnlineASRProcessor",
    "asr_factory",
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