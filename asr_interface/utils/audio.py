"""Audio processing utilities."""

import io
import logging

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Default sampling rate for ASR processing
SAMPLING_RATE = 16000


def load_audio_from_bytes(
    audio_bytes: bytes, target_sr: int = SAMPLING_RATE
) -> np.ndarray:
    """
    Load audio from bytes, ensuring the specified sample rate and format.

    Args:
        audio_bytes: Raw audio data as bytes
        target_sr: Target sample rate (default: 16000 Hz)

    Returns:
        Audio data as numpy array with shape (samples,) and dtype float32

    Raises:
        ValueError: If audio cannot be loaded or processed
    """
    try:
        # Load audio using soundfile
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        # Convert stereo to mono if necessary
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if the sample rate doesn't match
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        return audio

    except Exception as e:
        logger.error(f"Error loading audio from bytes: {e}", exc_info=True)
        raise ValueError(f"Could not load audio: {e}") from e


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to prevent clipping and improve processing.

    Args:
        audio: Input audio array

    Returns:
        Normalized audio array
    """
    if audio.size == 0:
        return audio

    # Normalize to [-1, 1] range
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    return audio


def pad_audio(
    audio: np.ndarray, target_length: int, pad_value: float = 0.0
) -> np.ndarray:
    """
    Pad audio to a target length.

    Args:
        audio: Input audio array
        target_length: Target length in samples
        pad_value: Value to use for padding

    Returns:
        Padded audio array
    """
    if len(audio) >= target_length:
        return audio[:target_length]

    padding = np.full(target_length - len(audio), pad_value, dtype=audio.dtype)
    return np.concatenate([audio, padding])


def chunk_audio(
    audio: np.ndarray, chunk_size: int, overlap: int = 0
) -> list[np.ndarray]:
    """
    Split audio into overlapping chunks.

    Args:
        audio: Input audio array
        chunk_size: Size of each chunk in samples
        overlap: Overlap between chunks in samples

    Returns:
        List of audio chunks
    """
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(audio), step):
        chunk = audio[i : i + chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk)

    return chunks


def get_audio_duration(audio: np.ndarray, sample_rate: int = SAMPLING_RATE) -> float:
    """
    Calculate the duration of audio in seconds.

    Args:
        audio: Audio array
        sample_rate: Sample rate in Hz

    Returns:
        Duration in seconds
    """
    return len(audio) / sample_rate


def detect_silence(
    audio: np.ndarray,
    threshold: float = 0.01,
    min_silence_duration: float = 0.5,
    sample_rate: int = SAMPLING_RATE,
) -> list[tuple[float, float]]:
    """
    Detect silence periods in audio.

    Args:
        audio: Audio array
        threshold: Amplitude threshold for silence detection
        min_silence_duration: Minimum silence duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        List of (start_time, end_time) tuples for silence periods
    """
    # Calculate RMS energy
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    hop_length = int(0.010 * sample_rate)  # 10ms hop

    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]

    # Find silence frames
    silence_frames = rms < threshold

    # Convert to time segments
    silence_periods = []
    start_frame = None

    for i, is_silent in enumerate(silence_frames):
        if is_silent and start_frame is None:
            start_frame = i
        elif not is_silent and start_frame is not None:
            end_frame = i
            duration = (end_frame - start_frame) * hop_length / sample_rate

            if duration >= min_silence_duration:
                start_time = start_frame * hop_length / sample_rate
                end_time = end_frame * hop_length / sample_rate
                silence_periods.append((start_time, end_time))

            start_frame = None

    # Handle case where audio ends with silence
    if start_frame is not None:
        end_frame = len(silence_frames)
        duration = (end_frame - start_frame) * hop_length / sample_rate

        if duration >= min_silence_duration:
            start_time = start_frame * hop_length / sample_rate
            end_time = end_frame * hop_length / sample_rate
            silence_periods.append((start_time, end_time))

    return silence_periods
