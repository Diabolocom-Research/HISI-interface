#!/usr/bin/env python3
import argparse
import logging
import sys
import time
from functools import lru_cache
from typing import Tuple, List, Optional, Any

import librosa
import numpy as np

from real_time_asr_backend.backends_for_whisper_online import WhisperTimestampedASR, MLXWhisper, ASRBase

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


@lru_cache(10 ** 6)
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a


def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg * 16000)
    end_s = int(end * 16000)
    return audio[beg_s:end_s]


class HypothesisBuffer:

    def __init__(self, logfile=sys.stderr):
        self.commited_in_buffer = []  # list which stores finalized word
        self.buffer = []  # stores the previous hypothesis from the ASR
        self.new = []  # holds the current incoming hypothesis

        self.last_commited_time = 0  # The end timestamp of the last word that was committed. This is crucial for knowing where the stable part of the transcript ends.
        self.last_commited_word = None

        self.logfile = logfile

    def insert(self, new, offset):
        '''
        The offset is the start time of the audio chunk that was processed. The ASR will give timestamps relative to this chunk (e.g., from 0.0 seconds). This line converts those relative timestamps to absolute timestamps in the audio stream.
        '''
        # compare self.commited_in_buffer and new. It inserts only the words in new that extend the commited_in_buffer, it means they are roughly behind last_commited_time and new in content
        # the new tail is added to self.new

        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if len(self.new) >= 1:
            '''
            overlapping predictions.
                - It checks if the start of the new hypothesis is very close in time to the end of the last committed transcript (abs(a - self.last_commited_time) < 1).
                - If so, it looks for an overlap in the content of the words. It compares n-grams (sequences of 1 to 5 words) at the end of the commited_in_buffer with the beginning of the self.new hypothesis.
                - If it finds a matching n-gram (e.g., the last two words of the committed transcript are the same as the first two words of the new hypothesis), it removes those overlapping words from self.new. This is critical to avoid stuttering output like "this is a... this is a test".
            '''
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in commited and new. If they are, they're dropped.
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):  # 5 is the maximum
                        c = " ".join([self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1])
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            words_msg = " ".join(words)
                            logger.debug(f"removing last {i} words: {words_msg}")
                            break

    def flush(self):
        # returns commited chunk = the longest common prefix of 2 last inserts.

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
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer


class OnlineASRProcessor:
    """
    Manages the real-time, streaming processing of audio for an ASR model.

    This class acts as the main engine for a streaming ASR system. It is
    responsible for:
    1.  Buffering incoming audio chunks.
    2.  Calling an ASR backend to transcribe the audio.
    3.  Using a HypothesisBuffer to stabilize the ASR's output.
    4.  Intelligently managing the audio buffer's size to ensure low latency
        and memory usage in long-running sessions.
    5.  Handling errors and ASR failures gracefully.
    """
    SAMPLING_RATE = 16000

    def __init__(self,
                 asr: ASRBase,
                 buffer_trimming: Tuple[str, int] = ("segment", 15),
                 min_chunk_sec: float = 1.0,
                 logfile=sys.stderr):
        """
        Initializes the OnlineASRProcessor.

        Args:
            asr (ASRBase): An instance of an ASR backend that conforms to the
                           ASRBase interface. This is the model that will perform
                           the actual speech-to-text conversion.
            buffer_trimming (Tuple[str, int], optional): A tuple defining the
                strategy for trimming the audio buffer. Defaults to ("segment", 15),
                meaning the buffer is trimmed based on ASR segments when it
                exceeds 15 seconds.
            min_chunk_sec (float, optional): The minimum amount of audio in seconds
                that must be in the buffer before processing is attempted.
                Defaults to 1.0.
            logfile (file, optional): A file-like object for logging output.
                Defaults to sys.stderr.
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
        Resets the processor to a clean initial state.

        This is useful for starting a new audio stream without creating a new
        processor instance.

        Args:
            offset (float, optional): The initial time offset for the audio stream.
                Defaults to 0.0.
        """
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.buffer_time_offset = offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited = []
        self.consecutive_asr_failures = 0

    def insert_audio_chunk(self, audio: np.ndarray):
        """Appends a new chunk of audio to the internal buffer."""
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def process_iter(self) -> Tuple[Optional[float], Optional[float], str]:
        """
        Performs one complete iteration of the processing loop.

        This method orchestrates the transcription, stabilization, and buffer
        management steps.

        Returns:
            Tuple[Optional[float], Optional[float], str]: A tuple containing the
            start time, end time, and text of the newly committed transcript
            segment. Returns (None, None, "") if no new segment is committed.
        """
        # Stage 1: Check if there is enough audio to process.
        if len(self.audio_buffer) / self.SAMPLING_RATE < self.min_chunk_sec:
            return (None, None, "")

        # Stage 2: Transcribe the audio buffer.
        asr_result, asr_success = self._transcribe_audio()

        # Stage 3: Stabilize the transcript and get the newly committed part.
        committed_words = self._stabilize_transcript(asr_result)

        # Stage 4: Manage the audio buffer's size.
        self._manage_audio_buffer(asr_result, asr_success)

        # Stage 5: Format and return the committed segment.
        return self._format_output(committed_words)

    def _transcribe_audio(self) -> Tuple[Any, bool]:
        """
        Calls the ASR backend to transcribe the audio buffer and handles errors.

        Returns:
            Tuple[Any, bool]: A tuple containing the raw ASR result and a
                              boolean indicating if the transcription was successful.
        """
        prompt_str = self._get_prompt()
        logger.debug(f"PROMPT: {prompt_str}")

        buffered_sec = len(self.audio_buffer) / self.SAMPLING_RATE
        logger.debug(f"Transcribing {buffered_sec:.2f} seconds from {self.buffer_time_offset:.2f}")

        # This structure must be compatible with the ASR backend's expected result format,
        # even in failure cases.
        empty_result = {"text": "", "segments": [], "language": self.asr.original_language or "en"}

        try:
            # Attempt the transcription.
            result = self.asr.transcribe(self.audio_buffer, init_prompt=prompt_str)

            # Validate the result structure.
            if isinstance(result, dict) and "segments" in result:
                self.consecutive_asr_failures = 0
                return result, True
            else:
                logger.warning(f"ASR returned an unexpected structure: {type(result)}. Treating as failure.")
                self.consecutive_asr_failures += 1
                return empty_result, False
        except Exception as e:
            logger.error(f"Error during ASR transcription: {e}. Treating as failure.")
            self.consecutive_asr_failures += 1
            return empty_result, False

    def _stabilize_transcript(self, asr_result: Any) -> List[Tuple[float, float, str]]:
        """
        Uses the HypothesisBuffer to stabilize the transcript.

        Args:
            asr_result (Any): The raw result from the ASR backend.

        Returns:
            List[Tuple[float, float, str]]: A list of newly committed words.
        """
        timestamped_words = self.asr.ts_words(asr_result)
        self.transcript_buffer.insert(timestamped_words, self.buffer_time_offset)

        newly_committed_words = self.transcript_buffer.flush()
        if newly_committed_words:
            self.commited.extend(newly_committed_words)

        return newly_committed_words

    def _manage_audio_buffer(self, asr_result: Any, asr_success: bool):
        """
        Trims the audio buffer using intelligent or fallback strategies.

        Args:
            asr_result (Any): The raw result from the ASR backend.
            asr_success (bool): Flag indicating if the transcription was successful.
        """
        current_buffer_len_sec = len(self.audio_buffer) / self.SAMPLING_RATE

        # 1. Attempt intelligent trimming based on ASR segments.
        if current_buffer_len_sec > self.buffer_trimming_sec:
            self._trim_buffer_by_segment(asr_result)

        # 2. Apply fallback trimming if ASR is failing and buffer is too long.
        self._apply_fallback_trim(asr_success)

    def _trim_buffer_by_segment(self, asr_result: Any):
        """
        Finds a safe cut point based on ASR segments and trims the buffer.
        """
        if not self.commited:
            logger.debug("Cannot trim by segment yet: no text has been committed.")
            return

        segment_ends_relative = self.asr.segments_end_ts(asr_result)
        if not segment_ends_relative:
            return

        last_committed_time_abs = self.commited[-1][1]

        # Find the latest segment end that occurred before the last committed word.
        # This is a safe point to cut the audio.
        suitable_cut_point = None
        for seg_end in sorted(segment_ends_relative, reverse=True):
            seg_end_abs = seg_end + self.buffer_time_offset
            if seg_end_abs <= last_committed_time_abs and seg_end_abs > self.buffer_time_offset:
                suitable_cut_point = seg_end_abs
                break

        if suitable_cut_point:
            self._chunk_at_timestamp(suitable_cut_point)

    def _apply_fallback_trim(self, asr_success: bool):
        """
        Forcefully trims the buffer if ASR is failing and the buffer is too long.
        This prevents the process from getting stuck.
        """
        buffer_len_after_trim = len(self.audio_buffer) / self.SAMPLING_RATE
        is_failing = not asr_success and self.consecutive_asr_failures >= self.MAX_CONSECUTIVE_ASR_FAILURES
        is_too_long = buffer_len_after_trim > (self.buffer_trimming_sec + self.FALLBACK_TRIM_THRESHOLD_SEC)

        if is_failing and is_too_long:
            logger.warning(
                f"ASR failing and buffer is too long ({buffer_len_after_trim:.2f}s). "
                f"Applying fallback trim."
            )
            discard_seconds = self.min_chunk_sec
            self._chunk_at_timestamp(self.buffer_time_offset + discard_seconds)

    def _chunk_at_timestamp(self, time: float):
        """
        Cuts the audio buffer and all corresponding transcript buffers at a specific time.
        """
        cut_seconds = time - self.buffer_time_offset
        if cut_seconds <= 0 or cut_seconds > len(self.audio_buffer) / self.SAMPLING_RATE:
            return

        cut_samples = int(cut_seconds * self.SAMPLING_RATE)

        logger.debug(f"Trimming {cut_seconds:.2f}s from audio buffer. New offset: {time:.2f}")

        # Trim all relevant buffers
        self.audio_buffer = self.audio_buffer[cut_samples:]
        self.buffer_time_offset = time
        self.transcript_buffer.pop_commited(self.buffer_time_offset)
        self.commited = [item for item in self.commited if item[1] > self.buffer_time_offset]

    def _get_prompt(self) -> str:
        """
        Generates a context prompt from the recently committed text for the ASR.
        """
        # Find text that is older than the current audio buffer to use as a prompt.
        k = len(self.commited)
        while k > 0 and self.commited[k - 1][1] > self.buffer_time_offset:
            k -= 1

        prompt_words = [t for _, _, t in self.commited[:k]]

        # Extract the last ~200 characters.
        prompt_list = []
        char_count = 0
        while prompt_words and char_count < 200:
            word = prompt_words.pop(-1)
            char_count += len(word) + 1
            prompt_list.append(word)

        return self.asr.sep.join(prompt_list[::-1])

    def _format_output(self, words: List[Tuple[float, float, str]]) -> Tuple[Optional[float], Optional[float], str]:
        """
        Formats a list of word tuples into a single segment tuple.
        """
        if not words:
            return (None, None, "")

        text = self.asr.sep.join(w[2] for w in words)
        start_time = words[0][0]
        end_time = words[-1][1]
        return (start_time, end_time, text)

    def finish(self) -> Tuple[Optional[float], Optional[float], str]:
        """
        Processes any remaining audio and returns the final uncommitted transcript.

        Call this method at the very end of the stream to ensure no audio is lost.
        """
        # Get the final, uncommitted part of the transcript.
        final_words = self.transcript_buffer.complete()

        logger.debug(f"Final, uncommitted transcript: {final_words}")
        self.init()  # Reset for potential reuse.
        return self._format_output(final_words)


ASR_BACKENDS = {
    "whisper_timestamped": WhisperTimestampedASR,
    "mlx-whisper": MLXWhisper,
    # "faster-whisper": FasterWhisperASR, # Can be uncommented when implemented
}


def asr_factory(args: argparse.Namespace, logfile=sys.stderr) -> Tuple[ASRBase, OnlineASRProcessor]:
    """
    Creates and configures ASR and OnlineASRProcessor instances.

    This factory selects the appropriate ASR backend based on the provided
    arguments, loads the specified model, and initializes the online processor.

    Args:
        args (argparse.Namespace): The command-line arguments.
        logfile: A file-like object for logging.

    Returns:
        Tuple[ASRBase, OnlineASRProcessor]: A tuple containing the initialized
        ASR backend and the online ASR processor.

    Raises:
        ValueError: If an unsupported backend is specified in the arguments.
        NotImplementedError: If a backend is defined but not yet implemented.
    """
    # 1. Select the ASR backend class from the mapping.
    backend_name = args.backend
    asr_cls = ASR_BACKENDS.get(backend_name)

    if not asr_cls:
        # Handle unsupported or unimplemented backends cleanly.
        if backend_name in ["openai-api", "faster-whisper"]:
            raise NotImplementedError(f"The '{backend_name}' backend is not yet implemented.")
        raise ValueError(
            f"Unsupported ASR backend: '{backend_name}'. Available backends are: {list(ASR_BACKENDS.keys())}")

    # 2. Load the specified ASR model.
    t = time.time()
    logger.info(f"Loading Whisper model '{args.model}' for language '{args.lan}' using '{backend_name}' backend...")

    asr = asr_cls(
        modelsize=args.model,
        lan=args.lan,
        cache_dir=args.model_cache_dir,
        model_dir=args.model_dir
    )

    e = time.time()
    logger.info(f"Model loaded in {e - t:.2f} seconds.")

    # 3. Apply any common configurations to the ASR instance.
    if getattr(args, 'vad', False):
        logger.info("Voice Activity Detection (VAD) is not available.")
        raise NotImplementedError

    # 4. Create the online processor.
    if args.vac:
        raise NotImplementedError("Voice Activity Controller (VAC) is not yet implemented.")

    online = OnlineASRProcessor(
        asr=asr,
        logfile=logfile,
        buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec)
    )

    return asr, online


def add_shared_args(parser):
    """shared args for simulation (this entry point) and server
    parser: argparse.ArgumentParser object
    """
    parser.add_argument('--min-chunk-size', type=float, default=1.0,
                        help='Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.')
    parser.add_argument('--model', type=str, default='large-v2',
                        choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo".split(
                            ","),
                        help="Name size of the Whisper model to use (default: large-v2). The model is automatically downloaded from the model hub if not present in model cache dir.")
    parser.add_argument('--model_cache_dir', type=str, default=None,
                        help="Overriding the default model cache dir where models downloaded from the hub are saved")
    parser.add_argument('--model_dir', type=str, default=None,
                        help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.")
    parser.add_argument('--lan', '--language', type=str, default='auto',
                        help="Source language code, e.g. en,de,cs, or 'auto' for language detection.")
    parser.add_argument('--task', type=str, default='transcribe', choices=["transcribe", "translate"],
                        help="Transcribe or translate.")
    parser.add_argument('--backend', type=str, default="whisper_timestamped",
                        choices=["faster-whisper", "whisper_timestamped", "mlx-whisper", "openai-api"],
                        help='Load only this backend for Whisper processing.')
    parser.add_argument('--vac', action="store_true", default=False,
                        help='Use VAC = voice activity controller. Recommended. Requires torch.')
    parser.add_argument('--vac-chunk-size', type=float, default=0.04, help='VAC sample size in seconds.')
    parser.add_argument('--vad', action="store_true", default=False,
                        help='Use VAD = voice activity detection, with the default parameters.')
    parser.add_argument('--buffer_trimming', type=str, default="segment", choices=["sentence", "segment"],
                        help='Buffer trimming strategy -- trim completed sentences marked with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper. Sentence segmenter must be installed for "sentence" option.')
    parser.add_argument('--buffer_trimming_sec', type=float, default=10,
                        help='Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.')
    parser.add_argument("-l", "--log-level", dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the log level",
                        default='DEBUG')


def set_logging(args, logger, other="_server"):
    logging.basicConfig(  # format='%(name)s
        format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)
    logging.getLogger("whisper_online" + other).setLevel(args.log_level)


#    logging.getLogger("whisper_online_server").setLevel(args.log_level)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', type=str,
                        help="Filename of 16kHz mono channel wav, on which live streaming is simulated.")
    add_shared_args(parser)
    parser.add_argument('--start_at', type=float, default=0.0, help='Start processing audio at this time.')
    parser.add_argument('--offline', action="store_true", default=False, help='Offline mode.')
    parser.add_argument('--comp_unaware', action="store_true", default=False,
                        help='Computationally unaware simulation.')

    args = parser.parse_args()

    # reset to store stderr to different file stream, e.g. open(os.devnull,"w")
    logfile = sys.stderr

    if args.offline and args.comp_unaware:
        logger.error("No or one option from --offline and --comp_unaware are available, not both. Exiting.")
        sys.exit(1)

    #    if args.log_level:
    #        logging.basicConfig(format='whisper-%(levelname)s:%(name)s: %(message)s',
    #                            level=getattr(logging, args.log_level))

    set_logging(args, logger)

    audio_path = args.audio_path

    SAMPLING_RATE = 16000
    duration = len(load_audio(audio_path)) / SAMPLING_RATE
    logger.info("Audio duration is: %2.2f seconds" % duration)

    asr, online = asr_factory(args, logfile=logfile)
    if args.vac:
        min_chunk = args.vac_chunk_size
    else:
        min_chunk = args.min_chunk_size

    # load the audio into the LRU cache before we start the timer
    a = load_audio_chunk(audio_path, 0, 1)

    # warm up the ASR because the very first transcribe takes much more time than the other
    asr.transcribe(a)

    beg = args.start_at
    start = time.time() - beg


    def output_transcript(o, now=None):
        # output format in stdout is like:
        # 4186.3606 0 1720 Takhle to je
        # - the first three words are:
        #    - emission time from beginning of processing, in milliseconds
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript
        if now is None:
            now = time.time() - start
        if o[0] is not None:
            print("%1.4f %1.0f %1.0f %s" % (now * 1000, o[0] * 1000, o[1] * 1000, o[2]), file=logfile, flush=True)
            print("%1.4f %1.0f %1.0f %s" % (now * 1000, o[0] * 1000, o[1] * 1000, o[2]), flush=True)
        else:
            # No text, so no output
            pass


    if args.offline:  ## offline mode processing (for testing/debugging)
        a = load_audio(audio_path)
        online.insert_audio_chunk(a)
        try:
            o = online.process_iter()
        except AssertionError as e:
            logger.error(f"assertion error: {repr(e)}")
        else:
            output_transcript(o)
        now = None
    elif args.comp_unaware:  # computational unaware mode
        end = beg + min_chunk
        while True:
            a = load_audio_chunk(audio_path, beg, end)
            online.insert_audio_chunk(a)
            try:
                o = online.process_iter()
            except AssertionError as e:
                logger.error(f"assertion error: {repr(e)}")
                pass
            else:
                output_transcript(o, now=end)

            logger.debug(f"## last processed {end:.2f}s")

            if end >= duration:
                break

            beg = end

            if end + min_chunk > duration:
                end = duration
            else:
                end += min_chunk
        now = duration

    else:  # online = simultaneous mode
        end = 0
        while True:
            now = time.time() - start
            if now < end + min_chunk:
                time.sleep(min_chunk + end - now)
            end = time.time() - start
            a = load_audio_chunk(audio_path, beg, end)
            beg = end
            online.insert_audio_chunk(a)

            try:
                o = online.process_iter()
            except AssertionError as e:
                logger.error(f"assertion error: {e}")
                pass
            else:
                output_transcript(o)
            now = time.time() - start
            logger.debug(f"## last processed {end:.2f} s, now is {now:.2f}, the latency is {now - end:.2f}")

            if end >= duration:
                break
        now = None

    o = online.finish()
    output_transcript(o, now=now)
