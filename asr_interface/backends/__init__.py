"""ASR backends and model loaders."""

from .whisper_loader import WhisperOnlineLoader
from .registry import MODEL_LOADERS

__all__ = [
    "WhisperOnlineLoader",
    "MODEL_LOADERS",
]

# Try to import MLX Whisper loader (optional dependency)
try:
    from .mlx_whisper_loader import MLXWhisperLoader
    __all__.append("MLXWhisperLoader")
except ImportError:
    # MLX Whisper not available, skip it
    pass 