"""Configuration models for ASR components."""

from typing import Optional
from pydantic import BaseModel, Field


class ASRConfig(BaseModel):
    """Configuration for ASR model loading and processing."""
    
    model: str = Field(..., description="Model name/size (e.g., 'tiny', 'base', 'small', 'medium', 'large')")
    lan: str = Field(default="auto", description="Language code or 'auto' for automatic detection")
    task: str = Field(default="transcribe", description="Task type: 'transcribe' or 'translate'")
    min_chunk_size: float = Field(default=1.0, description="Minimum chunk size in seconds")
    backend: str = Field(default="whisper_timestamped", description="ASR backend to use")
    buffer_trimming: str = Field(default="segment", description="Buffer trimming strategy")
    buffer_trimming_sec: float = Field(default=10.0, description="Buffer trimming duration in seconds")
    model_cache_dir: Optional[str] = Field(default=None, description="Directory to cache model files")
    model_dir: Optional[str] = Field(default=None, description="Directory containing model files")
    vac: bool = Field(default=False, description="Enable Voice Activity Control")
    vad: bool = Field(default=False, description="Enable Voice Activity Detection")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow additional fields for extensibility
        validate_assignment = True 