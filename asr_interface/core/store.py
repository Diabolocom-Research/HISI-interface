"""Shared state management for ASR components."""

import hashlib
import json
import logging
from typing import Any

from .config import ASRConfig
from .protocols import ASRProcessor

logger = logging.getLogger(__name__)


class ASRComponentsStore:
    """Thread-safe store for ASR components and state."""

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the ASR components store.

        Args:
            sample_rate: Default sample rate for audio processing
        """
        self._store: dict[str, Any] = {
            "asr_processor": None,
            "sample_rate": sample_rate,
            "separator": " ",
            "is_ready": False,
            "current_config_id": None,
        }
        self._lock = None  # Could be replaced with threading.Lock() if needed

    @property
    def asr_processor(self) -> ASRProcessor | None:
        """Get the current ASR processor."""
        return self._store.get("asr_processor")

    @asr_processor.setter
    def asr_processor(self, processor: ASRProcessor | None) -> None:
        """Set the current ASR processor."""
        self._store["asr_processor"] = processor

    @property
    def sample_rate(self) -> int:
        """Get the current sample rate."""
        return self._store.get("sample_rate", 16000)

    @property
    def separator(self) -> str:
        """Get the current text separator."""
        return self._store.get("separator", " ")

    @separator.setter
    def separator(self, separator: str) -> None:
        """Set the current text separator."""
        self._store["separator"] = separator

    @property
    def is_ready(self) -> bool:
        """Check if the ASR processor is ready."""
        return self._store.get("is_ready", False)

    @is_ready.setter
    def is_ready(self, ready: bool) -> None:
        """Set the ready state."""
        self._store["is_ready"] = ready

    @property
    def current_config_id(self) -> str | None:
        """Get the current configuration ID."""
        return self._store.get("current_config_id")

    @current_config_id.setter
    def current_config_id(self, config_id: str | None) -> None:
        """Set the current configuration ID."""
        self._store["current_config_id"] = config_id

    def get_config_id(self, config: ASRConfig) -> str:
        """
        Generate a unique configuration ID for the given config.

        Args:
            config: The ASR configuration

        Returns:
            A unique hash string for the configuration
        """
        config_json_str = json.dumps(config.dict(), sort_keys=True)
        return hashlib.sha256(config_json_str.encode("utf-8")).hexdigest()

    def is_config_current(self, config: ASRConfig) -> bool:
        """
        Check if the given configuration matches the current one.

        Args:
            config: The ASR configuration to check

        Returns:
            True if the configuration matches the current one
        """
        config_id = self.get_config_id(config)
        return (
            self.current_config_id == config_id
            and self.is_ready
            and self.asr_processor is not None
        )

    def reset(self) -> None:
        """Reset the store to initial state."""
        self._store.update(
            {
                "asr_processor": None,
                "is_ready": False,
                "current_config_id": None,
            }
        )
        logger.info("ASR components store reset")

    def to_dict(self) -> dict[str, Any]:
        """Convert the store to a dictionary for serialization."""
        return self._store.copy()
