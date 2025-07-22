"""Protocol definitions for ASR components."""

import sys
from abc import ABC, abstractmethod
from typing import Any, Protocol

from pydantic import BaseModel


class ASRProcessor(Protocol):
    """
    A protocol defining the interface for a real-time ASR processor.

    Any object that implements these methods can be used by the RealTimeASRHandler.
    """

    def insert_audio_chunk(self, audio_chunk: Any) -> None:
        """Insert an audio chunk for processing."""
        ...

    def process_iter(self) -> tuple[float, float, str] | None:
        """
        Process the current audio buffer and return results if available.

        Returns:
            Optional[Tuple[float, float, str]]: A tuple of (start_time, end_time, text)
            if a segment is ready, None otherwise.
        """
        ...

    def init(self, offset: float = 0.0) -> None:
        """Initialize the processor with an optional time offset."""
        ...

    def finish(self) -> tuple[float, float, str] | None:
        """
        Finalize processing and return any remaining results.

        Returns:
            Optional[Tuple[float, float, str]]: Final segment if available, None otherwise.
        """
        ...


class ModelLoader(Protocol):
    """
    A protocol defining the interface for a model loader.

    Each loader is responsible for taking a configuration, loading all
    necessary components, and returning a fully initialized ASR processor
    and any associated metadata.
    """

    def load(self, config: BaseModel) -> tuple[ASRProcessor, dict[str, Any]]:
        """
        Loads a model and creates a processor.

        Args:
            config: A Pydantic model containing the configuration.

        Returns:
            A tuple containing:
                - The initialized ASR processor instance.
                - A dictionary of metadata (e.g., separator character).
        """
        ...


class ASRBase(ABC):
    """
    Abstract Base Class for an Automatic Speech Recognition (ASR) backend.

    This class defines a standard interface that the OnlineASRProcessor can use
    to interact with different ASR models. Any class inheriting from ASRBase must
    implement all its abstract methods.

    This is the same interface used by the legacy whisper-streaming system.

    Attributes:
        sep (str): The separator character used to join recognized words.
                   This can vary by backend (e.g., " " for some, "" for others).
    """

    sep = " "  # Default separator

    def __init__(
        self,
        lan: str,
        modelsize: str = None,
        cache_dir: str = None,
        model_dir: str = None,
        logfile=sys.stderr,
    ):
        """
        Initializes the ASR backend.

        Args:
            lan (str): Language code (e.g., "en", "de") or "auto" for detection.
            modelsize (str, optional): The size of the model to load (e.g., "large-v3").
            cache_dir (str, optional): Directory to cache downloaded models.
            model_dir (str, optional): Path to a directory containing a pre-downloaded model.
            logfile: A file-like object for logging.
        """
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.original_language = None if lan == "auto" else lan
        self.model = self.load_model(modelsize, cache_dir, model_dir)

    @abstractmethod
    def load_model(
        self, modelsize: str = None, cache_dir: str = None, model_dir: str = None
    ):
        """
        Loads the ASR model into memory.

        This method should handle the specifics of model loading for the backend,
        whether from a local directory, a cache, or by downloading.
        """
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def transcribe(self, audio, init_prompt: str = "") -> dict:
        """
        Performs transcription on a given audio buffer.

        Args:
            audio: The audio data to transcribe (e.g., a NumPy array).
            init_prompt (str, optional): A prompt to initialize the model's context.

        Returns:
            dict: The raw transcription result from the backend. The structure of this
                  result must be a dictionary containing a "segments" key, where the
                  value is a list of segment objects. This structure will be passed
                  to `ts_words` and `segments_end_ts` for parsing.
                  e.g. {"text": "...", "segments": [...], "language": "en"}
        """
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def ts_words(self, transcription_result: Any) -> list[tuple[float, float, str]]:
        """
        Parses the raw transcription result to extract word-level timestamps.

        This method acts as an adapter, converting the backend-specific output
        of `transcribe()` into a standardized format required by OnlineASRProcessor.

        Args:
            transcription_result (Any): The raw output from this class's `transcribe` method.

        Returns:
            List[Tuple[float, float, str]]: A list of tuples, where each tuple
            represents a word as (start_time, end_time, word_text).
        """
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def segments_end_ts(self, transcription_result: Any) -> list[float]:
        """
        Parses the raw transcription result to extract segment end timestamps.

        This is used by OnlineASRProcessor's buffer trimming logic to find safe
        points to cut the audio buffer.

        Args:
            transcription_result (Any): The raw output from this class's `transcribe` method.

        Returns:
            List[float]: A list of timestamps (in seconds) indicating the
                         end of each detected speech segment.
        """
        raise NotImplementedError("must be implemented in the child class")

    # @abstractmethod
    # def use_vad(self):
    #     """
    #     Enables Voice Activity Detection (VAD) for the ASR model, if supported.
    #     """
    #     raise NotImplementedError("must be implemented in the child class")
