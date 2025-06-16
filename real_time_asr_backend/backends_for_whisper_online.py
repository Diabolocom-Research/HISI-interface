import sys
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)


class ASRBase(ABC):
    """
    Abstract Base Class for an Automatic Speech Recognition (ASR) backend.

    This class defines a standard interface that the OnlineASRProcessor can use
    to interact with different ASR models (e.g., whisper_timestamped, mlx-whisper).
    Any class inheriting from ASRBase must implement all its abstract methods.

    Attributes:
        sep (str): The separator character used to join recognized words.
                   This can vary by backend (e.g., " " for some, "" for others).
    """
    sep = " "  # Default separator

    def __init__(self, lan: str, modelsize: str = None, cache_dir: str = None, model_dir: str = None, logfile=sys.stderr):
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
    def load_model(self, modelsize: str = None, cache_dir: str = None, model_dir: str = None):
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
    def ts_words(self, transcription_result: Any) -> List[Tuple[float, float, str]]:
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
    def segments_end_ts(self, transcription_result: Any) -> List[float]:
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

    @abstractmethod
    def use_vad(self):
        """
        Enables Voice Activity Detection (VAD) for the ASR model, if supported.
        """
        raise NotImplementedError("must be implemented in the child class")




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
