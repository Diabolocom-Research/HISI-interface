"""Real-time ASR stream handler for WebRTC audio processing."""

import logging
from typing import Any

import librosa
import numpy as np
from fastrtc import AdditionalOutputs, StreamHandler

from ..core.protocols import ASRProcessor
from ..core.store import ASRComponentsStore

logger = logging.getLogger(__name__)


class RealTimeASRHandler(StreamHandler):
    """
    A generic, real-time audio processing handler for fastRTC.

    This class acts as a bridge between the fastRTC streaming server and a
    backend ASR (Automatic Speech Recognition) processor. It is designed to be
    decoupled from the specifics of the ASR implementation.

    It retrieves a pre-initialized ASR processor from a shared store
    and uses it to process incoming audio frames. For each new client connection,
    a new instance of this handler is created via the .copy() method.
    """

    def __init__(self, store: ASRComponentsStore, **kwargs: Any) -> None:
        """
        Initialize the handler instance for a new connection.

        Args:
            store: The shared ASR components store
            **kwargs: Additional arguments passed to StreamHandler
        """
        # Determine the sample rate to be used by the StreamHandler base class
        rate_to_use = kwargs.pop("input_sample_rate", store.sample_rate)
        super().__init__(input_sample_rate=rate_to_use, **kwargs)

        # Reference to the shared application state
        self.store = store

        # Local instance of the ASR processor, pulled from the store
        self.asr_processor: ASRProcessor | None = None

        # Per-connection state for managing transcription results
        self.accumulated_transcript = ""
        self.segments: list[dict[str, Any]] = []

        # State for managing audio buffering (optional, for potential future use like playback)
        self.full_audio = np.zeros((0,), dtype=np.float32)

        # State to track the configuration version and prevent re-initialization
        self.last_used_config_id: str | None = None

        # Unique identifier for this handler instance for clear logging
        self.handler_id = id(self)

        logger.info(
            f"Handler instance [{self.handler_id}] created. Waiting for processor."
        )
        self._ensure_processor()

    def _ensure_processor(self) -> None:
        """
        Synchronize the handler's local ASR processor with the shared store.

        This method is the key to the dynamic model loading. It checks if the
        globally available processor is newer than the one this handler instance
        is currently using and updates it if necessary.
        """
        # If we already have the correct processor, do nothing
        is_already_set = (
            self.asr_processor
            and self.last_used_config_id == self.store.current_config_id
        )
        if is_already_set:
            return

        # If the shared store has a ready processor, acquire it
        if self.store.is_ready and self.store.asr_processor:
            config_id = self.store.current_config_id or "unknown"
            logger.info(
                f"Handler [{self.handler_id}] - Acquiring new ASR processor for config '{config_id}'."
            )
            self.asr_processor = self.store.asr_processor
            self.last_used_config_id = config_id
            self._reset_instance_state()
            logger.info(
                f"Handler [{self.handler_id}] - Processor acquired successfully."
            )
        else:
            # If no processor is ready, ensure we don't hold onto an old one
            self.asr_processor = None

    def _reset_instance_state(self) -> None:
        """Reset the transcription state when a new processor is acquired."""
        logger.info(f"Handler [{self.handler_id}] - Resetting instance state.")
        self.accumulated_transcript = ""
        self.segments = []
        self.full_audio = np.zeros((0,), dtype=np.float32)

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        """
        Process an incoming audio frame from the WebRTC stream.

        This method is called by fastRTC for each audio chunk received. It handles
        audio format conversion, resampling, and passing the data to the ASR processor.

        Args:
            frame: A tuple containing sample rate and PCM audio data
        """
        self._ensure_processor()
        if not self.asr_processor:
            return

        sample_rate, pcm_data = frame

        # Convert audio to float32 format, required by Whisper
        audio_float32 = pcm_data.astype(np.float32) / 32768.0

        # Resample if the incoming audio's sample rate differs from the target
        target_sr = self.store.sample_rate
        if sample_rate != target_sr:
            audio_float32 = librosa.resample(
                audio_float32, orig_sr=sample_rate, target_sr=target_sr
            )

        self.asr_processor.insert_audio_chunk(audio_float32.flatten())

    def emit(self) -> AdditionalOutputs:
        """
        Poll the ASR processor for new transcription results to send to the client.

        This method is called periodically by fastRTC's event loop.

        Returns:
            AdditionalOutputs: A data structure containing the full transcript and segments
        """
        if not self.asr_processor:
            return AdditionalOutputs("", np.array([], dtype=np.float32), [])

        processed_output = self.asr_processor.process_iter()
        if processed_output is None:
            # No new segment, just return the current state
            return AdditionalOutputs(
                self.accumulated_transcript, self.full_audio, self.segments
            )

        beg, end, text_delta = processed_output
        if text_delta:
            self.segments.append({"start": beg, "end": end, "text": text_delta})

            # Correctly append the new text delta with a separator
            separator = self.store.separator
            if self.accumulated_transcript and text_delta.strip():
                self.accumulated_transcript += separator + text_delta.strip()
            elif text_delta.strip():
                self.accumulated_transcript = text_delta.strip()

        return AdditionalOutputs(
            self.accumulated_transcript, self.full_audio, self.segments
        )

    def copy(self) -> "RealTimeASRHandler":
        """
        Create a new instance of the handler for a new client connection.

        This is a factory method required by the fastRTC `Stream` object.

        Returns:
            A new RealTimeASRHandler instance
        """
        logger.info(
            f"RealTimeASRHandler.copy() called for master handler [{self.handler_id}]."
        )
        return RealTimeASRHandler(self.store)

    def shutdown(self) -> None:
        """Clean up resources when a client connection is closed."""
        logger.info(f"Handler [{self.handler_id}] shutting down.")
        self.asr_processor = None
