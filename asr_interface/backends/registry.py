"""Registry for ASR model loaders."""

from typing import Dict

from ..core.protocols import ModelLoader

# Registry of available model loaders
MODEL_LOADERS: Dict[str, ModelLoader] = {}

# Try to import Whisper Timestamped loader (optional dependency)
try:
    from .whisper_timestamped_loader import WhisperTimestampedLoader
    MODEL_LOADERS["whisper_timestamped"] = WhisperTimestampedLoader()
except ImportError:
    # Whisper Timestamped not available, skip it
    pass

# Try to import MLX Whisper loader (optional dependency)
try:
    from .mlx_whisper_loader import MLXWhisperLoader
    MODEL_LOADERS["mlx_whisper"] = MLXWhisperLoader()
except ImportError:
    # MLX Whisper not available, skip it
    pass

def register_loader(name: str, loader: ModelLoader) -> None:
    """
    Register a new model loader.
    
    Args:
        name: The name/identifier for the loader
        loader: The model loader instance
    """
    MODEL_LOADERS[name] = loader

def get_loader(name: str) -> ModelLoader:
    """
    Get a model loader by name.
    
    Args:
        name: The name of the loader to retrieve
        
    Returns:
        The model loader instance
        
    Raises:
        KeyError: If the loader is not found
    """
    if name not in MODEL_LOADERS:
        available = list(MODEL_LOADERS.keys())
        raise KeyError(f"Unknown backend '{name}'. Available backends: {available}")
    return MODEL_LOADERS[name]

def list_loaders() -> list[str]:
    """
    List all available model loaders.
    
    Returns:
        List of loader names
    """
    return list(MODEL_LOADERS.keys()) 