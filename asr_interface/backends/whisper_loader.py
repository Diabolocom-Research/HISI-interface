"""Standard OpenAI Whisper ASR backend implementation (permissive license)."""

import logging
from typing import Any

from ..core.config import ASRConfig
from ..core.protocols import ASRBase, ASRProcessor, ModelLoader

logger = logging.getLogger(__name__)


class WhisperASR(ASRBase):
    """
    Standard OpenAI Whisper ASR backend implementation.

    Uses the official OpenAI Whisper library as the backend.
    Provides segment-level timestamps; word-level timestamps are approximated.
    """

    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        import whisper

        if model_dir is not None:
            logger.debug(
                f"Loading whisper model from model_dir {model_dir}. modelsize parameter is not used."
            )
            # OpenAI Whisper does not support loading from a directory directly, so fallback to modelsize
        return whisper.load_model(modelsize, download_root=cache_dir)

    def transcribe(self, audio, init_prompt=""):
        # Use the standard whisper transcribe method
        result = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            word_timestamps=True,
            verbose=None,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        return result

    def ts_words(self, r):
        # OpenAI Whisper does not provide word-level timestamps, only segment-level
        # We'll map each segment to a single word covering the whole segment
        o = []
        for segment in r["segments"]:
            if segment.get("no_speech_prob", 0) <= 0.9:
                for word in segment.get("words", []):
                    t = (word["start"], word["end"], word["word"])
                    o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s["end"] for s in res["segments"]]

    def use_vad(self):
        self.transcribe_kargs["vad"] = True


class WhisperLoader(ModelLoader):
    """
    Loads standard OpenAI Whisper models for real-time ASR.

    This loader creates ASR backends based on the official Whisper library.
    """

    def load(self, config: ASRConfig) -> tuple[ASRProcessor, dict[str, Any]]:
        """
        Implement the loading logic for Whisper models.

        Args:
            config: The ASR configuration containing model parameters

        Returns:
            A tuple containing:
                - The initialized ASR processor instance (OnlineASRProcessor)
                - A dictionary of metadata (e.g., separator character)

        Raises:
            RuntimeError: If model loading fails
        """
        logger.info("Using WhisperLoader...")

        try:
            from .whisper_online_processor import OnlineASRProcessor

            asr_backend = WhisperASR(
                lan=config.lan,
                modelsize=config.model,
                cache_dir=config.model_cache_dir,
                model_dir=config.model_dir,
            )

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
                "model_type": "whisper",
                "model_size": config.model,
                "language": config.lan,
            }

            return processor, metadata

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load Whisper model: {e}") from e
