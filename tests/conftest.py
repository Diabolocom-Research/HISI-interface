"""Pytest configuration and common fixtures."""

import pytest
from pathlib import Path

from asr_interface.core.store import ASRComponentsStore
from asr_interface.core.config import ASRConfig


@pytest.fixture
def sample_audio_file(tmp_path: Path) -> Path:
    """Create a sample audio file for testing."""
    # This would create a small test audio file
    # For now, we'll just return a path that doesn't exist
    return tmp_path / "test_audio.wav"


@pytest.fixture
def asr_store() -> ASRComponentsStore:
    """Create a fresh ASR components store for testing."""
    return ASRComponentsStore()


@pytest.fixture
def sample_config() -> ASRConfig:
    """Create a sample ASR configuration for testing."""
    return ASRConfig(
        model="tiny",
        lan="en",
        task="transcribe",
        backend="whisper_timestamped"
    )


@pytest.fixture
def mock_audio_data() -> bytes:
    """Create mock audio data for testing."""
    # This would create realistic audio data
    # For now, return empty bytes
    return b"mock_audio_data" 