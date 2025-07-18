"""Protocol definitions for ASR components."""

from typing import Any, Dict, Optional, Protocol, Tuple
from pydantic import BaseModel


class ASRProcessor(Protocol):
    """
    A protocol defining the interface for a real-time ASR processor.
    
    Any object that implements these methods can be used by the RealTimeASRHandler.
    """

    def insert_audio_chunk(self, audio_chunk: Any) -> None:
        """Insert an audio chunk for processing."""
        ...

    def process_iter(self) -> Optional[Tuple[float, float, str]]:
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

    def finish(self) -> Optional[Tuple[float, float, str]]:
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

    def load(self, config: BaseModel) -> Tuple[ASRProcessor, Dict[str, Any]]:
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