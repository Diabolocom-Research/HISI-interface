#!/usr/bin/env python3
import sys
import numpy as np
import librosa
from functools import lru_cache
import time
import logging

import io
import soundfile as sf
import math

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


# Whisper backend

class ASRBase:
    sep = " "  # join transcribe words with this character (" " for whisper_timestamped,

    # "" for faster-whisper because it emits the spaces when neeeded)

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        self.logfile = logfile

        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan

        self.model = self.load_model(modelsize, cache_dir, model_dir)

    def load_model(self, modelsize, cache_dir):
        raise NotImplemented("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplemented("must be implemented in the child class")

    def use_vad(self):
        raise NotImplemented("must be implemented in the child class")


class WhisperTimestampedASR(ASRBase):
    """Uses whisper_timestamped library as the backend. Initially, we tested the code on this backend. It worked, but slower than faster-whisper.
    On the other hand, the installation for GPU could be easier.
    """

    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        import whisper
        from whisper_timestamped import transcribe_timestamped
        self.transcribe_timestamped = transcribe_timestamped
        if model_dir is not None:
            logger.debug("ignoring model_dir, not implemented")
        return whisper.load_model(modelsize, download_root=cache_dir)

    def transcribe(self, audio, init_prompt=""):
        result = self.transcribe_timestamped(self.model,
                                             audio, language=self.original_language,
                                             initial_prompt=init_prompt, verbose=None,
                                             condition_on_previous_text=True, **self.transcribe_kargs)
        return result

    def ts_words(self, r):
        # return: transcribe result object to [(beg,end,"word1"), ...]
        o = []
        for s in r["segments"]:
            for w in s["words"]:
                t = (w["start"], w["end"], w["text"])
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s["end"] for s in res["segments"]]

    def use_vad(self):
        self.transcribe_kargs["vad"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class MLXWhisper(ASRBase):
    """
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
        from mlx_whisper.transcribe import ModelHolder, transcribe
        import mlx.core as mx  # Is installed with mlx-whisper

        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. modelsize parameter is not used.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = self.translate_model_name(modelsize)
            logger.debug(
                f"Loading whisper model {modelsize}. You use mlx whisper, so {model_size_or_path} will be used.")

        self.model_size_or_path = model_size_or_path

        # Note: ModelHolder.get_model loads the model into a static class variable,
        # making it a global resource. This means:
        # - Only one model can be loaded at a time; switching models requires reloading.
        # - This approach may not be suitable for scenarios requiring multiple models simultaneously,
        #   such as using whisper-streaming as a module with varying model sizes.
        dtype = mx.float16  # Default to mx.float16. In mlx_whisper.transcribe: dtype = mx.float16 if decode_options.get("fp16", True) else mx.float32
        ModelHolder.get_model(model_size_or_path, dtype)  # Model is preloaded to avoid reloading during transcription

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
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "large": "mlx-community/whisper-large-mlx"
        }

        # Retrieve the corresponding MLX model path
        mlx_model_path = model_mapping.get(model_name)

        if mlx_model_path:
            return mlx_model_path
        else:
            raise ValueError(f"Model name '{model_name}' is not recognized or not supported.")

    def transcribe(self, audio, init_prompt=""):
        segments = self.model(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            word_timestamps=True,
            condition_on_previous_text=True,
            path_or_hf_repo=self.model_size_or_path,
            **self.transcribe_kargs
        )
        return segments.get("segments", [])

    def ts_words(self, segments):
        """
        Extract timestamped words from transcription segments and skips words with high no-speech probability.
        """
        return [
            (word["start"], word["end"], word["word"])
            for segment in segments
            for word in segment.get("words", [])
            if segment.get("no_speech_prob", 0) <= 0.9
        ]

    def segments_end_ts(self, res):
        return [s['end'] for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"



class HypothesisBuffer:

    def __init__(self, logfile=sys.stderr):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []

        self.last_commited_time = 0
        self.last_commited_word = None

        self.logfile = logfile

    def insert(self, new, offset):
        # compare self.commited_in_buffer and new. It inserts only the words in new that extend the commited_in_buffer, it means they are roughly behind last_commited_time and new in content
        # the new tail is added to self.new

        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if len(self.new) >= 1:
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
    SAMPLING_RATE = 16000

    def __init__(self, asr, tokenizer=None, buffer_trimming=("segment", 15), logfile=sys.stderr,
                 min_chunk_sec: float = 1.0):
        self.asr = asr
        self.tokenizer = tokenizer
        self.logfile = logfile
        self.min_chunk_sec = min_chunk_sec
        self.init()
        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming
        # For fallback trimming: track consecutive ASR failures
        self.consecutive_asr_failures = 0
        self.MAX_CONSECUTIVE_ASR_FAILURES_FOR_FALLBACK_TRIM = 3  # Configurable: after how many failures to force trim
        self.FALLBACK_TRIM_OVER_THRESHOLD_SEC = 5  # Configurable: how many seconds over buffer_trimming_sec to trigger fallback

    def init(self, offset=None):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.buffer_time_offset = 0
        if offset is not None:
            self.buffer_time_offset = offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited = []
        self.consecutive_asr_failures = 0  # Reset on init

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1][1] > self.buffer_time_offset:
            k -= 1
        p = self.commited[:k]
        p = [t for _, _, t in p]
        prompt_list = []
        l = 0
        while p and l < 200:
            x = p.pop(-1)
            l += len(x) + 1
            prompt_list.append(x)
        non_prompt = self.commited[k:]
        return self.asr.sep.join(prompt_list[::-1]), self.asr.sep.join(t for _, _, t in non_prompt)

    def process_iter(self):
        buffered_sec = len(self.audio_buffer) / self.SAMPLING_RATE
        if buffered_sec < self.min_chunk_sec:
            return (None, None, "")

        prompt_str, context_str = self.prompt()
        logger.debug(f"PROMPT: {prompt_str}")
        logger.debug(f"CONTEXT: {context_str}")
        logger.debug(
            f"transcribing {buffered_sec:.2f} seconds from {self.buffer_time_offset:.2f}")

        # Define a structure for an empty/failed ASR result
        # This structure should be compatible with self.asr.ts_words and self.asr.segments_end_ts
        # For WhisperTimestampedASR, an empty result usually has a "segments" key with an empty list.
        empty_asr_result = {"text": "", "segments": [], "language": self.asr.original_language or "en"}

        res = empty_asr_result  # Default to empty result
        tsw = []
        asr_success = False
        try:
            # Attempt transcription
            temp_res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt_str)

            # Basic validation of the result (depends on ASR backend)
            if isinstance(temp_res, dict) and "segments" in temp_res:
                res = temp_res  # Use the actual result
                tsw = self.asr.ts_words(
                    res)  # This might also fail if res structure is unexpected despite 'segments' key
                asr_success = True
                self.consecutive_asr_failures = 0  # Reset counter on success
            else:
                # Handle cases where transcribe doesn't error but returns unexpected structure
                logger.warning(
                    f"asr.transcribe returned an unexpected result structure: {type(temp_res)}. Treating as failure.")
                # res remains empty_asr_result, tsw remains []
                self.consecutive_asr_failures += 1
        except AssertionError as e:
            logger.error(f"AssertionError during asr.transcribe: {e}. Treating as failure.")
            # res remains empty_asr_result, tsw remains []
            self.consecutive_asr_failures += 1
        except Exception as e:  # Catch other potential errors from transcribe
            logger.error(f"Generic error during asr.transcribe: {e}. Treating as failure.")
            # res remains empty_asr_result, tsw remains []
            self.consecutive_asr_failures += 1

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)  # tsw is [] if ASR failed
        o = self.transcript_buffer.flush()  # o will be [] if ASR failed (no new words inserted)
        if o:  # Only extend commited if new words were flushed from transcript_buffer
            self.commited.extend(o)

        # Log current transcript state (optional, for debugging)
        # completed_log = self.to_flush(o)
        # logger.debug(f">>>>COMPLETE NOW: {completed_log}")
        # the_rest_log = self.to_flush(self.transcript_buffer.complete())
        # logger.debug(f"INCOMPLETE: {the_rest_log}")

        # --- Buffer Trimming Logic ---
        # Sentence-based trimming (if ASR was successful and new text was committed)
        if asr_success and o and self.buffer_trimming_way == "sentence":
            if len(self.audio_buffer) / self.SAMPLING_RATE > self.buffer_trimming_sec:
                self.chunk_completed_sentence()  # Relies on self.commited

        # Segment-based trimming (or default time-based if buffer too long)
        # This should be attempted even if ASR failed, using 'res' (which would be empty_asr_result)
        # or relying on self.commited for trimming points.
        current_buffer_len_sec = len(self.audio_buffer) / self.SAMPLING_RATE
        # Determine the primary threshold for attempting a trim
        trim_attempt_threshold_sec = self.buffer_trimming_sec if self.buffer_trimming_way == "segment" else 30

        if current_buffer_len_sec > trim_attempt_threshold_sec:
            trimmed_by_segment_logic = self.chunk_completed_segment(res)  # Pass 'res' (actual or empty)
            if trimmed_by_segment_logic:
                logger.debug(
                    f"Buffer trimmed by chunk_completed_segment. New length: {len(self.audio_buffer) / self.SAMPLING_RATE:.2f}s")
            else:
                logger.debug(f"chunk_completed_segment did not trim. Buffer length: {current_buffer_len_sec:.2f}s")

        # Fallback Trimming: If ASR has failed multiple times AND buffer is still too long
        current_buffer_len_sec_after_trim = len(self.audio_buffer) / self.SAMPLING_RATE
        fallback_trigger_threshold_sec = self.buffer_trimming_sec + self.FALLBACK_TRIM_OVER_THRESHOLD_SEC

        if (not asr_success and
                self.consecutive_asr_failures >= self.MAX_CONSECUTIVE_ASR_FAILURES_FOR_FALLBACK_TRIM and
                current_buffer_len_sec_after_trim > fallback_trigger_threshold_sec):

            # ASR is failing, and normal trimming (if any occurred) didn't bring buffer below a higher threshold.
            # Forcefully discard the oldest part of the audio buffer to ensure progress.
            samples_to_discard = int(self.min_chunk_sec * self.SAMPLING_RATE)  # Discard one min_chunk_sec

            if samples_to_discard > 0 and samples_to_discard <= len(self.audio_buffer):
                discard_sec = samples_to_discard / self.SAMPLING_RATE
                logger.warning(
                    f"ASR failed {self.consecutive_asr_failures} times & buffer ({current_buffer_len_sec_after_trim:.2f}s) > fallback threshold ({fallback_trigger_threshold_sec:.2f}s). "
                    f"Forcefully discarding oldest {discard_sec:.2f}s of audio."
                )
                self.audio_buffer = self.audio_buffer[samples_to_discard:]
                self.buffer_time_offset += discard_sec
                # Important: Adjust commited transcripts based on new buffer_time_offset
                self.transcript_buffer.pop_commited(self.buffer_time_offset)
                # Also, re-filter self.commited list itself if it contains items older than new offset
                self.commited = [item for item in self.commited if item[1] > self.buffer_time_offset]
            elif samples_to_discard == 0:
                logger.warning("Fallback trim: min_chunk_sec is zero, no audio discarded by fallback.")
            else:  # samples_to_discard > len(self.audio_buffer)
                logger.warning(
                    f"Fallback trim: Calculated discard ({samples_to_discard}) > buffer len ({len(self.audio_buffer)}). Not discarding.")

        logger.debug(f"End of process_iter. len of buffer now: {len(self.audio_buffer) / self.SAMPLING_RATE:.2f}")
        return self.to_flush(o)

    def chunk_completed_sentence(self):
        if not self.commited: return False  # Ensure self.commited is not empty
        logger.debug(f"Attempting sentence chunk. Commited: {self.commited}")
        sents = self.words_to_sentences(self.commited)
        # for s_idx, s_val in enumerate(sents): logger.debug(f"\t\tSENT {s_idx}: {s_val}")
        if len(sents) < 2:
            return False

        # Original logic: keeps last 2 sentences, trims before sents[-2][1]
        # Let's ensure we are trimming based on text actually older than the buffer_trimming_sec threshold
        # The decision to call this function is already based on buffer_trimming_sec.
        # Here, we find the split point.

        # We want to trim up to the end of the second-to-last sentence, *if* it makes sense.
        # The original logic:
        # while len(sents) > 2:
        #     sents.pop(0)
        # chunk_at_ts = sents[-2][1] # This assumes sents has at least 2 items after popping.
        # A simpler approach: find the end of the first sentence if multiple sentences exist.
        # Trim up to the end of (n-1)th sentence if n sentences exist.

        if len(sents) >= 2:  # Need at least two sentences to trim the earlier one(s)
            # Find the end timestamp of all sentences except the last one
            # Example: if 3 sents, trim up to end of sent[1]. if 2 sents, trim up to end of sent[0]
            chunk_at_ts = sents[-2][1]  # End of the second to last sentence
            if chunk_at_ts > self.buffer_time_offset:  # Ensure trimming point is valid
                logger.debug(f"--- sentence chunked at {chunk_at_ts:.2f}")
                self.chunk_at(chunk_at_ts)
                return True
            else:
                logger.debug(
                    f"--- sentence chunk: target trim point {chunk_at_ts:.2f} is not after buffer_time_offset {self.buffer_time_offset:.2f}")
        return False

    def chunk_completed_segment(self, res):  # res can be an actual ASR result or empty_asr_result
        if not self.commited:
            logger.debug(
                "--- chunk_completed_segment: No commited text yet. Cannot trim based on ASR segments alignment with committed text.")
            return False

        # Ensure 'res' is a dict and has 'segments' key
        if not (isinstance(res, dict) and "segments" in res):
            logger.warning(
                f"--- chunk_completed_segment: 'res' (type: {type(res)}) is not a valid ASR result structure or missing 'segments'. Cannot trim based on new ASR segments.")
            current_asr_segment_ends = []
        else:
            current_asr_segment_ends = self.asr.segments_end_ts(
                res)  # e.g., [0.5, 1.2, 2.0] relative to current audio_buffer start

        last_committed_word_end_time_abs = self.commited[-1][1]

        suitable_chunk_point_abs = None

        if current_asr_segment_ends:
            # Convert segment ends to absolute time and find the latest one that is <= last_committed_word_end_time_abs
            # These segments are from the *current* ASR processing attempt.
            for seg_end_relative in sorted(current_asr_segment_ends, reverse=True):
                seg_end_abs = seg_end_relative + self.buffer_time_offset
                if seg_end_abs <= last_committed_word_end_time_abs and seg_end_abs > self.buffer_time_offset:
                    suitable_chunk_point_abs = seg_end_abs
                    break

            if suitable_chunk_point_abs:
                logger.debug(
                    f"--- segment chunked at {suitable_chunk_point_abs:.2f} (based on current ASR segments ending before or at last committed word {last_committed_word_end_time_abs:.2f})")
                self.chunk_at(suitable_chunk_point_abs)
                return True
            else:
                logger.debug(
                    f"--- chunk_completed_segment: No current ASR segment end found suitable for trimming relative to last committed word {last_committed_word_end_time_abs:.2f}.")
                return False
        else:
            # No current ASR segments (e.g., ASR failed or produced no segments)
            # In this case, we can't use ASR segments to guide trimming.
            # The fallback trim in process_iter will handle persistent ASR failures if buffer grows too large.
            logger.debug(
                "--- chunk_completed_segment: No segment end timestamps from current ASR result. Cannot trim based on new ASR segments.")
            return False

    def chunk_at(self, time):
        cut_seconds = time - self.buffer_time_offset
        if cut_seconds <= 0:  # Cannot cut at or before the current start of the buffer
            logger.debug(
                f"chunk_at: Attempted to cut at or before buffer_time_offset ({time:.2f} <= {self.buffer_time_offset:.2f}). No cut made.")
            return

        # Ensure we don't cut more than available
        cut_samples = min(int(cut_seconds * self.SAMPLING_RATE), len(self.audio_buffer))

        if cut_samples <= 0:
            logger.debug(f"chunk_at: Calculated zero or negative samples to cut ({cut_samples}). No cut made.")
            return

        logger.debug(
            f"chunk_at: Trimming {cut_samples / self.SAMPLING_RATE:.2f}s from audio_buffer. Current offset: {self.buffer_time_offset:.2f}, new offset: {time:.2f}")
        self.audio_buffer = self.audio_buffer[cut_samples:]
        self.buffer_time_offset = time  # Update to the exact trim time

        # Clean up buffers based on the new offset
        self.transcript_buffer.pop_commited(self.buffer_time_offset)
        self.commited = [item for item in self.commited if item[1] > self.buffer_time_offset]

    def words_to_sentences(self, words):
        if not self.tokenizer:
            logger.warning("Tokenizer is None, cannot split words into sentences.")
            # Fallback: treat each word group (if any) as a sentence or return based on VAD segments if available
            # For simplicity, if no tokenizer, just return the words as a single "sentence"
            if not words: return []
            return [(words[0][0], words[-1][1], " ".join(w[2] for w in words))]

        # Original logic for words_to_sentences
        cwords = [w for w in words]  # Make a copy
        text_to_segment = " ".join(o[2] for o in cwords)
        try:
            segmented_sentences_text = self.tokenizer.split(text_to_segment)
        except Exception as e:
            logger.error(f"Error during sentence tokenization: {e}. Treating input as a single sentence.")
            if not words: return []
            return [(words[0][0], words[-1][1], text_to_segment)]

        out_sentences = []
        current_word_idx = 0
        for sent_text in segmented_sentences_text:
            sent_text_clean = sent_text.strip()
            if not sent_text_clean:
                continue

            sentence_start_time = None
            sentence_end_time = None
            accumulated_words_for_sentence = []

            temp_sent_text_matcher = sent_text_clean
            start_idx_for_this_sentence = current_word_idx

            for i in range(start_idx_for_this_sentence, len(words)):
                word_start_time, word_end_time, word_text = words[i]
                word_text_clean = word_text.strip()

                if not temp_sent_text_matcher:  # Should not happen if sent_text_clean was not empty
                    break

                if temp_sent_text_matcher.startswith(word_text_clean):
                    if sentence_start_time is None:
                        sentence_start_time = word_start_time

                    accumulated_words_for_sentence.append(word_text)  # Use original word text for rejoining
                    sentence_end_time = word_end_time  # Keep updating end time

                    # Remove matched part
                    temp_sent_text_matcher = temp_sent_text_matcher[len(word_text_clean):].strip()
                    current_word_idx = i + 1  # Advance current_word_idx past this word for next sentence

                    if not temp_sent_text_matcher:  # Full sentence matched
                        break
                else:
                    # Word mismatch, tokenizer might have altered words or split differently
                    # This indicates a potential issue with tokenizer vs. ASR word alignment
                    logger.warning(
                        f"Sentence segmentation mismatch: Expected start of '{temp_sent_text_matcher}' but found '{word_text_clean}'. Sentence: '{sent_text_clean}'")
                    # Attempt to recover: if we have accumulated some words, form a sentence.
                    # Or, break and let the outer loop try the next sentence from tokenizer.
                    # For now, if a mismatch occurs, we might lose this tokenized sentence.
                    # To be more robust, one might need a more complex alignment.
                    # Let's assume for now that if it starts matching, it continues.
                    # If first word doesn't match, this sentence from tokenizer might be skipped for this word list.
                    if not accumulated_words_for_sentence:  # First word of sentence didn't match
                        current_word_idx = start_idx_for_this_sentence  # Reset, try this word with next tokenized sentence
                    break  # Break from inner word loop, go to next tokenized sentence

            if sentence_start_time is not None and sentence_end_time is not None:
                # Rejoin words that formed this sentence to ensure exact text match with ASR output for this segment
                final_sentence_text = self.asr.sep.join(accumulated_words_for_sentence)
                out_sentences.append((sentence_start_time, sentence_end_time, final_sentence_text))
            elif sent_text_clean:  # Tokenizer gave a sentence, but we couldn't align it with words
                logger.warning(f"Could not align tokenized sentence '{sent_text_clean}' with ASR words.")

        return out_sentences

    def finish(self):
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        logger.debug(f"last, noncommited: {f}")
        # self.buffer_time_offset += len(self.audio_buffer) / self.SAMPLING_RATE # Already handled by chunk_at or fallback
        self.audio_buffer = np.array([], dtype=np.float32)  # Clear buffer on finish
        return f

    def to_flush(self, sents, sep=None, offset=0):  # offset is not used as sents have absolute timestamps
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if not sents:
            b = None
            e = None
        else:
            b = sents[0][0]
            e = sents[-1][1]
        return (b, e, t)



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


def asr_factory(args, logfile=sys.stderr):
    """
    Creates and configures an ASR and ASR Online instance based on the specified backend and arguments.
    """
    backend = args.backend
    if backend == "openai-api":
        logger.debug("Using OpenAI API.")
        # asr = OpenaiApiASR(lan=args.lan)
    else:
        if backend == "faster-whisper":
            asr_cls = FasterWhisperASR
        elif backend == "mlx-whisper":
            asr_cls = MLXWhisper
        else:
            asr_cls = WhisperTimestampedASR

        # Only for FasterWhisperASR and WhisperTimestampedASR
        size = args.model
        t = time.time()
        logger.info(f"Loading Whisper {size} model for {args.lan}...")
        asr = asr_cls(modelsize=size, lan=args.lan, cache_dir=args.model_cache_dir, model_dir=args.model_dir)
        e = time.time()
        logger.info(f"done. It took {round(e - t, 2)} seconds.")

    # Apply common configurations
    if getattr(args, 'vad', False):  # Checks if VAD argument is present and True
        logger.info("Setting VAD filter")
        asr.use_vad()

    language = args.lan

    tgt_language = language  # Whisper transcribes in this language
    tokenizer = None

    # Create the OnlineASRProcessor
    if args.vac:

        online = VACOnlineASRProcessor(args.min_chunk_size, asr, tokenizer, logfile=logfile,
                                       buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))
    else:
        online = OnlineASRProcessor(asr, tokenizer, logfile=logfile,
                                    buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))

    return asr, online


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