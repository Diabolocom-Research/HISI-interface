"""Whisper-based ASR model loader."""

import logging
import sys
from typing import Any, Dict, Tuple

from ..core.config import ASRConfig
from ..core.protocols import ASRProcessor, ModelLoader

# Import whisper components - these will be moved to a dedicated whisper module later
try:
    from real_time_asr_backend.slimer_whisper_online import (
        OnlineASRProcessor, asr_factory, SAMPLING_RATE
    )
except ImportError:
    # Fallback for when the old structure is still being used
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from real_time_asr_backend.slimer_whisper_online import (
        OnlineASRProcessor, asr_factory, SAMPLING_RATE
    )

logger = logging.getLogger(__name__)


class WhisperOnlineLoader(ModelLoader):
    """
    Loads a model using the `whisper_online` library and its `asr_factory`.
    
    This loader creates ASR processors based on the Whisper model family,
    supporting various model sizes and configurations.
    """

    def load(self, config: ASRConfig) -> Tuple[ASRProcessor, Dict[str, Any]]:
        """
        Implement the loading logic for whisper_timestamped models.

        Args:
            config: The ASR configuration containing model parameters

        Returns:
            A tuple containing:
                - The initialized ASR processor instance
                - A dictionary of metadata (e.g., separator character)

        Raises:
            RuntimeError: If model loading fails
        """
        logger.info("Using WhisperOnlineLoader...")

        try:
            # asr_factory creates the base model (e.g., whispercpp, faster-whisper)
            # Note: asr_factory returns a tuple (asr_object_instance, online_processor_template)
            asr_object_instance, online_processor_template = asr_factory(config)
            logger.info(f"Loaded base ASR model of type: {type(asr_object_instance)}")

            if online_processor_template is None:
                online_processor = OnlineASRProcessor(
                    asr=asr_object_instance,
                    tokenizer=None,  # In custom_main_new.py, tokenizer is not explicitly passed
                    buffer_trimming=(config.buffer_trimming, int(config.buffer_trimming_sec)),
                    min_chunk_sec=config.min_chunk_size,
                    logfile=sys.stderr
                )
            else:
                online_processor = online_processor_template

            metadata = {
                "separator": getattr(asr_object_instance, "sep", " "),
                "model_type": "whisper",
                "model_size": config.model,
                "language": config.lan,
            }

            return online_processor, metadata

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load Whisper model: {e}") from e 