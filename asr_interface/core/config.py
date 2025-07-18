"""Configuration models for ASR components."""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class TURNConfig(BaseModel):
    """Configuration for TURN server settings."""
    
    provider: Literal["hf", "twilio", "cloudflare", "none"] = Field(
        default="none", 
        description="TURN server provider ('hf', 'twilio', 'cloudflare', or 'none')"
    )
    token: Optional[str] = Field(default=None, description="Provider-specific token/credentials")
    account_sid: Optional[str] = Field(default=None, description="Twilio Account SID")
    auth_token: Optional[str] = Field(default=None, description="Twilio Auth Token")
    key_id: Optional[str] = Field(default=None, description="Cloudflare Turn Token ID")
    api_token: Optional[str] = Field(default=None, description="Cloudflare API Token")
    ttl: int = Field(default=86400, description="Time-to-live for credentials in seconds")


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
    turn_config: Optional[TURNConfig] = Field(default=None, description="TURN server configuration")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow additional fields for extensibility
        validate_assignment = True 