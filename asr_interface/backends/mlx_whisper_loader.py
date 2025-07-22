"""MLX Whisper ASR backend implementation."""

import logging
from typing import Any

from ..core.config import ASRConfig
from ..core.protocols import ASRBase, ASRProcessor, ModelLoader

logger = logging.getLogger(__name__)


class MLXWhisper(ASRBase):
    """
    MLX Whisper ASR backend implementation.

    Uses MLX Whisper library as the backend, optimized for Apple Silicon.
    Models available: https://huggingface.co/collections/mlx-community/whisper-663256f9964fbb1177db93dc
    Significantly faster than faster-whisper (without CUDA) on Apple M1.
    """

    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        """
        Loads the MLX-compatible Whisper model.

        Args:
            modelsize (str, optional): The size or name of the Whisper model to load.
                If provided, it will be translated to an MLX-compatible model path using the `translate_model_name` method.
                Example: "large-v3-turbo" -> "mlx-community/whisper-large-v3-turbo".
            cache_dir (str, optional): Path to the directory for caching models.
                **Note**: This is not supported by MLX Whisper and will be ignored.
            model_dir (str, optional): Direct path to a custom model directory.
                If specified, it overrides the `modelsize` parameter.
        """
        import mlx.core as mx  # Is installed with mlx-whisper
        from mlx_whisper.transcribe import ModelHolder, transcribe

        if model_dir is not None:
            logger.debug(
                f"Loading whisper model from model_dir {model_dir}. modelsize parameter is not used."
            )
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = self.translate_model_name(modelsize)
            logger.debug(
                f"Loading whisper model {modelsize}. You use mlx whisper, so {model_size_or_path} will be used."
            )

        self.model_size_or_path = model_size_or_path

        dtype = mx.float16
        self.model = ModelHolder.get_model(
            model_size_or_path, dtype
        )  # Store the actual model object
        # Do NOT return transcribe

        return transcribe

    def translate_model_name(self, model_name):
        """
        Translates a given model name to its corresponding MLX-compatible model path.

        Args:
            model_name (str): The name of the model to translate.

        Returns:
            str: The MLX-compatible model path.
        """
        # Dictionary mapping model names to MLX-compatible paths
        model_mapping = {
            "tiny.en": "mlx-community/whisper-tiny.en-mlx",
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large-v1": "mlx-community/whisper-large-v1-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo-mlx",
        }

        return model_mapping.get(model_name, model_name)

    def transcribe(self, audio, init_prompt=""):
        result = self.model(
            audio,
            initial_prompt=init_prompt,
            word_timestamps=True,
            **self.transcribe_kargs,
        )
        return result.get("segments", [])

    def ts_words(self, segments):
        # return: transcribe result object to [(beg,end,"word1"), ...]
        return [
            (word["start"], word["end"], word["word"])
            for segment in segments
            for word in segment.get("words", [])
            if segment.get("no_speech_prob", 0) <= 0.9
        ]

    def segments_end_ts(self, res):
        return [s["end"] for s in res]


class MLXWhisperLoader(ModelLoader):
    """
    Loads MLX Whisper models optimized for Apple Silicon.

    This loader creates ASR processors based on the MLX Whisper model family,
    which provides significantly faster performance on Apple M1/M2 chips compared
    to other Whisper implementations.
    """

    def load(self, config: ASRConfig) -> tuple[ASRProcessor, dict[str, Any]]:
        """
        Implement the loading logic for MLX Whisper models.

        Args:
            config: The ASR configuration containing model parameters

        Returns:
            A tuple containing:
                - The initialized ASR processor instance
                - A dictionary of metadata (e.g., separator character)

        Raises:
            RuntimeError: If model loading fails
        """
        logger.info("Using MLXWhisperLoader...")

        try:
            # Create the ASR backend
            from .whisper_online_processor import OnlineASRProcessor

            asr_backend = MLXWhisper(
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

            logger.info(f"Loaded MLX Whisper model: {config.model}")

            # Define metadata for the MLX Whisper backend
            metadata = {
                "separator": asr_backend.sep,
                "model_type": "mlx_whisper",
                "model_size": config.model,
                "language": config.lan,
                "backend": "mlx_whisper",
                "optimized_for": "apple_silicon",
            }

            logger.info(
                f"MLX Whisper processor created successfully with model: {config.model}"
            )
            return processor, metadata

        except Exception as e:
            logger.error(f"Failed to load MLX Whisper model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load MLX Whisper model: {e}") from e
