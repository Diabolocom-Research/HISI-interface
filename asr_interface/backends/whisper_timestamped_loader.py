"""Whisper Timestamped ASR backend implementation."""

import logging
from typing import Any

from ..core.config import ASRConfig
from ..core.protocols import ASRBase, ASRProcessor, ModelLoader

logger = logging.getLogger(__name__)


class WhisperTimestampedASR(ASRBase):
    """
    Whisper Timestamped ASR backend implementation.

    Uses whisper_timestamped library as the backend.
    """

    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        import whisper
        from whisper_timestamped import transcribe_timestamped

        self.transcribe_timestamped = transcribe_timestamped
        if model_dir is not None:
            logger.debug("ignoring model_dir, not implemented")
        return whisper.load_model(modelsize, download_root=cache_dir)

    def transcribe(self, audio, init_prompt=""):
        result = self.transcribe_timestamped(
            self.model,
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            verbose=None,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        return result

    def ts_words(self, r):
        # return: transcribe result object to [(beg,end,"word1"), ...]
        o = []
        for s in r["segments"]:
            for w in s["words"]:
                t = (w["start"], w["end"], w["text"])
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s["end"] for s in res["segments"]]

    def use_vad(self):
        self.transcribe_kargs["vad"] = True


class WhisperTimestampedLoader(ModelLoader):
    """
    Loads Whisper Timestamped models for real-time ASR.

    This loader creates ASR backends based on the Whisper Timestamped library,
    which provides word-level timestamps for accurate real-time transcription.
    """

    def load(self, config: ASRConfig) -> tuple[ASRProcessor, dict[str, Any]]:
        """
        Implement the loading logic for Whisper Timestamped models.

        Args:
            config: The ASR configuration containing model parameters

        Returns:
            A tuple containing:
                - The initialized ASR processor instance (OnlineASRProcessor)
                - A dictionary of metadata (e.g., separator character)

        Raises:
            RuntimeError: If model loading fails
        """
        logger.info("Using WhisperTimestampedLoader...")

        try:
            # Create the ASR backend
            from .whisper_online_processor import OnlineASRProcessor

            asr_backend = WhisperTimestampedASR(
                lan=config.lan,
                modelsize=config.model,
                cache_dir=config.model_cache_dir,
                model_dir=config.model_dir,
            )

            # Create the OnlineASRProcessor that wraps the backend
            processor = OnlineASRProcessor(
                asr=asr_backend,
                buffer_trimming=(
                    config.buffer_trimming,
                    int(config.buffer_trimming_sec),
                ),
                min_chunk_sec=config.min_chunk_size,
            )

            metadata = {
                "separator": asr_backend.sep,
                "model_type": "whisper_timestamped",
                "model_size": config.model,
                "language": config.lan,
            }

            return processor, metadata

        except Exception as e:
            logger.error(
                f"Failed to load Whisper Timestamped model: {e}", exc_info=True
            )
            raise RuntimeError(f"Failed to load Whisper Timestamped model: {e}") from e
