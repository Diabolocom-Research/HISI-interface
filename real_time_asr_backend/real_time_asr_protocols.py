from typing import Protocol
from pydantic import BaseModel
from typing import Any, Dict, Optional, Tuple


class ASRProcessor(Protocol):
    """
    A protocol defining the interface for a real-time ASR processor.
    Any object that implements these methods can be used by the RealTimeASRHandler.
    """

    def insert_audio_chunk(self, audio_chunk: Any) -> None:
        ...

    def process_iter(self) -> Optional[Tuple[float, float, str]]:
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
