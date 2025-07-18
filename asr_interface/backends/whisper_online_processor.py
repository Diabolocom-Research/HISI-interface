"""Whisper Online ASR Processor for real-time streaming transcription."""

import logging
import sys
from typing import Any

import numpy as np

from ..core.config import ASRConfig
from ..core.protocols import ASRProcessor

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


class HypothesisBuffer:
    """
    A buffer for managing and stabilizing ASR output in real-time streaming.

    This class handles the complex logic of:
    1. Managing overlapping predictions from consecutive ASR calls
    2. Stabilizing transcripts by finding common prefixes
    3. Avoiding stuttering output by detecting and removing duplicates
    4. Maintaining timing information for each word
    """

    def __init__(self, logfile=sys.stderr):
        self.commited_in_buffer = []  # list which stores finalized words
        self.buffer = []  # stores the previous hypothesis from the ASR
        self.new = []  # holds the current incoming hypothesis

        self.last_commited_time = (
            0  # The end timestamp of the last word that was committed
        )
        self.last_commited_word = None

        self.logfile = logfile

    def insert(self, new, offset):
        """
        Insert new hypothesis with timing offset.

        The offset is the start time of the audio chunk that was processed.
        The ASR will give timestamps relative to this chunk (e.g., from 0.0 seconds).
        This converts those relative timestamps to absolute timestamps in the audio stream.
        """
        # Convert relative timestamps to absolute timestamps
        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if len(self.new) >= 1:
            # Handle overlapping predictions
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # Search for 1, 2, ..., 5 consecutive words (n-grams) that are identical
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):  # 5 is the maximum
                        c = " ".join(
                            [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][
                                ::-1
                            ]
                        )
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            words_msg = " ".join(words)
                            logger.debug(f"removing last {i} words: {words_msg}")
                            break

    def flush(self):
        """
        Returns committed chunk = the longest common prefix of 2 last inserts.
        """
        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        """Remove committed words that are older than the given time."""
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        """Return the current buffer as complete."""
        return self.buffer


class OnlineASRProcessor(ASRProcessor):
    """
    Manages the real-time, streaming processing of audio for an ASR model.

    This class acts as the main engine for a streaming ASR system. It is
    responsible for:
    1. Buffering incoming audio chunks.
    2. Calling an ASR backend to transcribe the audio.
    3. Using a HypothesisBuffer to stabilize the ASR's output.
    4. Intelligently managing the audio buffer's size to ensure low latency
       and memory usage in long-running sessions.
    5. Handling errors and ASR failures gracefully.
    """

    SAMPLING_RATE = 16000

    def __init__(
        self,
        asr: Any,  # ASR backend that conforms to the expected interface
        buffer_trimming: tuple[str, int] = ("segment", 15),
        min_chunk_sec: float = 1.0,
        logfile=sys.stderr,
    ):
        """
        Initialize the OnlineASRProcessor.

        Args:
            asr: An instance of an ASR backend that conforms to the expected interface.
                 This is the model that will perform the actual speech-to-text conversion.
            buffer_trimming: A tuple defining the strategy for trimming the audio buffer.
                            Defaults to ("segment", 15), meaning the buffer is trimmed
                            based on ASR segments when it exceeds 15 seconds.
            min_chunk_sec: The minimum amount of audio in seconds that must be in the
                          buffer before processing is attempted. Defaults to 1.0.
            logfile: A file-like object for logging output. Defaults to sys.stderr.
        """
        self.asr = asr
        self.logfile = logfile
        self.min_chunk_sec = min_chunk_sec
        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

        # Configuration for the fallback trimming mechanism
        self.MAX_CONSECUTIVE_ASR_FAILURES = 3
        self.FALLBACK_TRIM_THRESHOLD_SEC = 5

        self.init()

    def init(self, offset: float = 0.0):
        """
        Reset the processor to a clean initial state.

        This is useful for starting a new audio stream without creating a new
        processor instance.

        Args:
            offset: The initial time offset for the audio stream. Defaults to 0.0.
        """
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.buffer_time_offset = offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited = []
        self.consecutive_asr_failures = 0

    def insert_audio_chunk(self, audio: np.ndarray):
        """Append a new chunk of audio to the internal buffer."""
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def process_iter(self) -> tuple[float | None, float | None, str]:
        """
        Perform one complete iteration of the processing loop.

        This method orchestrates the transcription, stabilization, and buffer
        management steps.

        Returns:
            A tuple containing the start time, end time, and text of the newly
            committed transcript segment. Returns (None, None, "") if no new
            segment is committed.
        """
        # Stage 1: Check if there is enough audio to process.
        if len(self.audio_buffer) / self.SAMPLING_RATE < self.min_chunk_sec:
            return (None, None, "")

        # Stage 2: Transcribe the audio buffer.
        asr_result, asr_success = self._transcribe_audio()

        # Stage 3: Stabilize the transcript and get the newly committed part.
        committed_words = self._stabilize_transcript(asr_result)

        # Stage 4: Manage the audio buffer based on the ASR result.
        self._manage_audio_buffer(asr_result, asr_success)

        # Stage 5: Format and return the output.
        return self._format_output(committed_words)

    def _transcribe_audio(self) -> tuple[Any, bool]:
        """
        Transcribe the current audio buffer using the ASR backend.

        Returns:
            A tuple containing the ASR result and a boolean indicating success.
        """
        try:
            # Get the current prompt from the transcript buffer
            prompt = self._get_prompt()

            # Call the ASR backend
            result = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)

            self.consecutive_asr_failures = 0
            return result, True

        except Exception as e:
            self.consecutive_asr_failures += 1
            logger.warning(
                f"ASR transcription failed (attempt {self.consecutive_asr_failures}): {e}"
            )
            return None, False

    def _stabilize_transcript(self, asr_result: Any) -> list[tuple[float, float, str]]:
        """
        Stabilize the transcript using the hypothesis buffer.

        Args:
            asr_result: The raw ASR result.

        Returns:
            A list of committed word tuples (start_time, end_time, word).
        """
        if asr_result is None:
            return []

        # Extract word-level timestamps from the ASR result
        words = self.asr.ts_words(asr_result)

        # Insert the new hypothesis into the buffer
        self.transcript_buffer.insert(words, self.buffer_time_offset)

        # Flush the buffer to get committed words
        committed_words = self.transcript_buffer.flush()

        return committed_words

    def _manage_audio_buffer(self, asr_result: Any, asr_success: bool):
        """
        Manage the audio buffer based on the ASR result and success status.
        """
        if asr_success and asr_result is not None:
            # Use the ASR result to trim the buffer intelligently
            self._trim_buffer_by_segment(asr_result)
        else:
            # Apply fallback trimming if ASR failed
            self._apply_fallback_trim(asr_success)

    def _trim_buffer_by_segment(self, asr_result: Any):
        """
        Trim the audio buffer based on ASR segment boundaries.
        """
        try:
            # Get segment end timestamps from the ASR result
            segment_ends = self.asr.segments_end_ts(asr_result)

            if (
                segment_ends
                and len(self.audio_buffer) / self.SAMPLING_RATE
                > self.buffer_trimming_sec
            ):
                # Find the latest segment end that's within our trimming threshold
                for end_ts in reversed(segment_ends):
                    if end_ts <= self.buffer_trimming_sec:
                        self._chunk_at_timestamp(end_ts)
                        break

        except Exception as e:
            logger.warning(f"Failed to trim buffer by segment: {e}")
            # Fall back to simple trimming
            self._apply_fallback_trim(True)

    def _apply_fallback_trim(self, asr_success: bool):
        """
        Apply fallback buffer trimming when ASR-based trimming fails.
        """
        buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE

        if buffer_duration > self.FALLBACK_TRIM_THRESHOLD_SEC:
            # Trim to keep only the last portion of the buffer
            trim_samples = int(self.FALLBACK_TRIM_THRESHOLD_SEC * self.SAMPLING_RATE)
            if len(self.audio_buffer) > trim_samples:
                self.audio_buffer = self.audio_buffer[-trim_samples:]
                self.buffer_time_offset += (
                    buffer_duration - self.FALLBACK_TRIM_THRESHOLD_SEC
                )
                logger.debug(
                    f"Applied fallback buffer trimming, kept last {self.FALLBACK_TRIM_THRESHOLD_SEC}s"
                )

    def _chunk_at_timestamp(self, time: float):
        """
        Trim the audio buffer at a specific timestamp.
        """
        samples_to_keep = int(time * self.SAMPLING_RATE)
        if samples_to_keep < len(self.audio_buffer):
            self.audio_buffer = self.audio_buffer[samples_to_keep:]
            self.buffer_time_offset += time
            logger.debug(f"Trimmed buffer at timestamp {time}s")

    def _get_prompt(self) -> str:
        """
        Get the current prompt for the ASR model.
        """
        if self.commited:
            # Use the last few committed words as context
            last_words = [word[2] for word in self.commited[-10:]]  # Last 10 words
            return " ".join(last_words)
        return ""

    def _format_output(
        self, words: list[tuple[float, float, str]]
    ) -> tuple[float | None, float | None, str]:
        """
        Format the output from a list of word tuples.

        Args:
            words: List of word tuples (start_time, end_time, word).

        Returns:
            A tuple containing (start_time, end_time, text).
        """
        if not words:
            return (None, None, "")

        start_time = words[0][0]
        end_time = words[-1][1]
        text = " ".join(word[2] for word in words)

        # Update the committed words list
        self.commited.extend(words)

        return (start_time, end_time, text)

    def finish(self) -> tuple[float | None, float | None, str]:
        """
        Finalize processing and return any remaining results.

        Returns:
            A tuple containing (start_time, end_time, text) of the final segment.
        """
        if len(self.audio_buffer) > 0:
            # Process any remaining audio
            asr_result, _ = self._transcribe_audio()
            final_words = self._stabilize_transcript(asr_result)

            # Get any remaining uncommitted words
            remaining_words = self.transcript_buffer.complete()
            if remaining_words:
                final_words.extend(remaining_words)

            logger.debug(f"Final, uncommitted transcript: {final_words}")
            self.init()  # Reset for potential reuse.
            return self._format_output(final_words)

        return (None, None, "")


def asr_factory(config: ASRConfig) -> tuple[Any, OnlineASRProcessor]:
    """
    Create and configure ASR and OnlineASRProcessor instances.

    This factory creates the appropriate ASR backend based on the configuration
    and initializes the online processor.

    Args:
        config: The ASR configuration containing model parameters.

    Returns:
        A tuple containing the initialized ASR backend and the online ASR processor.

    Raises:
        ValueError: If an unsupported backend is specified.
        RuntimeError: If model loading fails.
    """
    logger.info(
        f"Creating ASR factory for model '{config.model}' with backend '{config.backend}'..."
    )

    # Import the appropriate backend based on configuration
    if config.backend == "whisper_timestamped":
        from .whisper_adapters import WhisperTimestampedAdapter
        from .whisper_timestamped_loader import WhisperTimestampedProcessor

        processor = WhisperTimestampedProcessor(
            model_size=config.model,
            language=config.lan,
            min_chunk_sec=config.min_chunk_size,
        )
        # Wrap the processor with the legacy-compatible adapter
        asr = WhisperTimestampedAdapter(processor, original_language=config.lan)

    elif config.backend == "mlx-whisper":
        from .mlx_whisper_loader import MLXWhisperProcessor
        from .whisper_adapters import MLXWhisperAdapter

        processor = MLXWhisperProcessor(
            model_size=config.model,
            language=config.lan,
            min_chunk_sec=config.min_chunk_size,
        )
        # Wrap the processor with the legacy-compatible adapter
        asr = MLXWhisperAdapter(processor, original_language=config.lan)

    else:
        raise ValueError(
            f"Unsupported ASR backend: '{config.backend}'. "
            f"Available backends are: whisper_timestamped, mlx-whisper"
        )

    # Create the online processor that wraps the ASR backend
    online = OnlineASRProcessor(
        asr=asr,
        buffer_trimming=(config.buffer_trimming, int(config.buffer_trimming_sec)),
        min_chunk_sec=config.min_chunk_size,
    )

    logger.info("ASR factory created successfully")
    return asr, online
