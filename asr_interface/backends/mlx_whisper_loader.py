"""MLX Whisper ASR model loader."""

import logging
import sys
import numpy as np
from typing import Any, Dict, Optional, Tuple

from ..core.config import ASRConfig
from ..core.protocols import ASRProcessor, ModelLoader

logger = logging.getLogger(__name__)


class MLXWhisperProcessor(ASRProcessor):
    """
    MLX Whisper real-time ASR processor optimized for Apple Silicon.
    
    This processor implements the ASRProcessor protocol directly,
    providing complete control over the real-time processing pipeline
    for MLX Whisper models.
    
    Models available: https://huggingface.co/collections/mlx-community/whisper-663256f9964fbb1177db93dc
    """

    def __init__(self, model_size: str, language: str, min_chunk_sec: float = 1.0):
        """
        Initialize the MLX Whisper processor.
        
        Args:
            model_size: The size of the Whisper model (e.g., "tiny", "base", "small")
            language: Language code (e.g., "en", "auto")
            min_chunk_sec: Minimum chunk size in seconds
        """
        self.model_size = model_size
        self.language = language
        self.min_chunk_sec = min_chunk_sec
        self.sampling_rate = 16000
        
        # Internal state
        self._audio_buffer = np.array([], dtype=np.float32)
        self._offset = 0.0
        self._model = None
        self._transcribe_func = None
        
        # Load the model
        self._load_model()
        
        logger.info(f"MLX Whisper processor initialized with model: {model_size}")

    def _load_model(self):
        """Load the MLX Whisper model."""
        try:
            from mlx_whisper.transcribe import ModelHolder, transcribe
            import mlx.core as mx
            
            # Translate model name to MLX-compatible path
            model_path = self._translate_model_name(self.model_size)
            
            # Load the model (ModelHolder.get_model loads into static class variable)
            dtype = mx.float16
            ModelHolder.get_model(model_path, dtype)
            
            self._transcribe_func = transcribe
            self._model_path = model_path
            
            logger.info(f"MLX Whisper model loaded: {model_path}")
            
        except ImportError:
            raise RuntimeError(
                "MLX Whisper not installed. Install with: pip install mlx-whisper"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load MLX Whisper model: {e}")

    def _translate_model_name(self, model_name: str) -> str:
        """Translate Whisper model name to MLX-compatible path."""
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
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "large": "mlx-community/whisper-large-mlx"
        }
        
        mlx_model_path = model_mapping.get(model_name)
        if not mlx_model_path:
            raise ValueError(f"Model name '{model_name}' is not supported by MLX Whisper")
        
        return mlx_model_path

    def init(self, offset: float = 0.0) -> None:
        """Initialize the processor with an optional time offset."""
        self._audio_buffer = np.array([], dtype=np.float32)
        self._offset = offset
        logger.debug(f"MLX Whisper processor initialized with offset: {offset}")

    def insert_audio_chunk(self, audio_chunk: np.ndarray) -> None:
        """Insert an audio chunk for processing."""
        self._audio_buffer = np.append(self._audio_buffer, audio_chunk)

    def process_iter(self) -> Optional[Tuple[float, float, str]]:
        """
        Process the current audio buffer and return results if available.
        
        Returns:
            Optional[Tuple[float, float, str]]: A tuple of (start_time, end_time, text) 
            if a segment is ready, None otherwise.
        """
        # Check if we have enough audio to process
        min_samples = int(self.min_chunk_sec * self.sampling_rate)
        
        if len(self._audio_buffer) < min_samples:
            return None
        
        try:
            # Transcribe the current buffer
            segments = self._transcribe_func(
                self._audio_buffer,
                language=self.language,
                word_timestamps=True,
                condition_on_previous_text=True,
                path_or_hf_repo=self._model_path
            )
            
            # Extract text from segments
            if segments and "segments" in segments:
                text_parts = []
                for segment in segments["segments"]:
                    if segment.get("no_speech_prob", 0) <= 0.9:  # Filter out non-speech
                        text_parts.append(segment.get("text", "").strip())
                
                if text_parts:
                    # Calculate timing
                    start_time = self._offset
                    end_time = start_time + len(self._audio_buffer) / self.sampling_rate
                    text = " ".join(text_parts)
                    
                    # Clear the buffer
                    self._audio_buffer = np.array([], dtype=np.float32)
                    self._offset = end_time
                    
                    logger.debug(f"Processed segment: {start_time:.2f}s - {end_time:.2f}s: {text}")
                    return (start_time, end_time, text)
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None

    def finish(self) -> Optional[Tuple[float, float, str]]:
        """
        Finalize processing and return any remaining results.
        
        Returns:
            Optional[Tuple[float, float, str]]: Final segment if available, None otherwise.
        """
        if len(self._audio_buffer) > 0:
            logger.debug("Processing final audio buffer")
            return self.process_iter()
        return None


class MLXWhisperLoader(ModelLoader):
    """
    Loads MLX Whisper models optimized for Apple Silicon.
    
    This loader creates ASR processors based on the MLX Whisper model family,
    which provides significantly faster performance on Apple M1/M2 chips compared
    to other Whisper implementations.
    """

    def load(self, config: ASRConfig) -> Tuple[ASRProcessor, Dict[str, Any]]:
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
            # Create the MLX Whisper processor directly
            processor = MLXWhisperProcessor(
                model_size=config.model,
                language=config.lan,
                min_chunk_sec=config.min_chunk_size
            )
            
            logger.info(f"Loaded MLX Whisper model: {config.model}")

            # Define metadata for the MLX Whisper backend
            metadata = {
                "separator": " ",
                "model_type": "mlx_whisper",
                "model_size": config.model,
                "language": config.lan,
                "backend": "mlx_whisper",
                "optimized_for": "apple_silicon",
            }

            logger.info(f"MLX Whisper processor created successfully with model: {config.model}")
            return processor, metadata

        except Exception as e:
            logger.error(f"Failed to load MLX Whisper model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load MLX Whisper model: {e}") from e 