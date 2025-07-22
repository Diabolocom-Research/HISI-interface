# Integrating Whisper Timestamped Backend

This guide provides a step-by-step walkthrough for integrating the **Whisper Timestamped** backend into the ASR Interface. Whisper Timestamped provides word-level timestamps for accurate real-time transcription.

---

## Overview

- **Backend Name:** `whisper_timestamped`
- **Features:** Word-level timestamps, real-time transcription
- **Integration Path:** ASRBase + OnlineASRProcessor (Recommended)

For general integration principles, see [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md).

---

## Step 1: Implement the Model Loader

Create a loader class that loads the Whisper Timestamped model and provides the processor and metadata.

```python
from asr_interface.core.protocols import ModelLoader, ASRProcessor
from asr_interface.core.config import ASRConfig
from asr_interface.backends.whisper_online_processor import OnlineASRProcessor

class WhisperTimestampedLoader(ModelLoader):
    """
    Loads Whisper Timestamped models for real-time ASR.
    """
    def load(self, config: ASRConfig) -> tuple[ASRProcessor, dict]:
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Using WhisperTimestampedLoader...")
        try:
            from .whisper_timestamped_asr import WhisperTimestampedASR
            asr_backend = WhisperTimestampedASR(
                lan=config.lan,
                modelsize=config.model,
                cache_dir=config.model_cache_dir,
                model_dir=config.model_dir,
            )
            processor = OnlineASRProcessor(
                asr=asr_backend,
                buffer_trimming=(config.buffer_trimming, int(config.buffer_trimming_sec)),
                min_chunk_sec=config.min_chunk_size,
            )
            metadata = {
                "separator": asr_backend.sep,
                "model_type": "whisper_timestamped",
                "model_size": config.model,
                "language": config.lan,
            }
            return processor, metadata
        except Exception as e:
            logger.error(f"Failed to load Whisper Timestamped model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load Whisper Timestamped model: {e}") from e
```

---

## Step 2: Implement the ASR Backend

Implement the `WhisperTimestampedASR` class, which must conform to the `ASRBase` protocol.

```python
import logging
from asr_interface.core.protocols import ASRBase

class WhisperTimestampedASR(ASRBase):
    """
    Whisper Timestamped ASR backend implementation.
    """
    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        import whisper
        from whisper_timestamped import transcribe_timestamped
        self.transcribe_timestamped = transcribe_timestamped
        if model_dir is not None:
            logging.debug("ignoring model_dir, not implemented")
        self.model = whisper.load_model(modelsize, download_root=cache_dir)
        return self.model

    def transcribe(self, audio, init_prompt=""):
        result = self.transcribe_timestamped(
            self.model,
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            verbose=None,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        return result

    def ts_words(self, r):
        # Convert transcribe result to [(start, end, word), ...]
        return [
            (w["start"], w["end"], w["text"])
            for s in r["segments"] for w in s["words"]
        ]

    def segments_end_ts(self, res):
        return [s["end"] for s in res["segments"]]

    def use_vad(self):
        self.transcribe_kargs["vad"] = True
```

---

## Step 3: Register the Loader

Register your loader in the backend registry so it can be discovered by the system.

```python
# Try to import Whisper Timestamped loader (optional dependency)
try:
    from .whisper_timestamped_loader import WhisperTimestampedLoader
    MODEL_LOADERS["whisper_timestamped"] = WhisperTimestampedLoader()
except ImportError:
    # Whisper Timestamped not available, skip it
    pass
```

---

## Usage Example

To use the Whisper Timestamped backend, send a POST request to `/load_model` with the following configuration:

```json
{
  "model": "base",           // or "small", "large", etc.
  "lan": "en",               // language code
  "task": "transcribe",
  "backend": "whisper_timestamped",
  "min_chunk_size": 1.0,
  "buffer_trimming": "segment",
  "buffer_trimming_sec": 10.0
}
```

---

## Troubleshooting

- **Import Errors:** Ensure `whisper_timestamped` and its dependencies are installed.
- **Missing Methods:** Make sure your backend implements all required methods from `ASRBase`.
- **Incorrect Return Format:** The `transcribe()` method must return a dict with a `segments` key.
- **Registration Issues:** Confirm your loader is registered before the server starts.

For more troubleshooting tips, see [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#troubleshooting).

---

## Further Reading

- [ASR Backend Integration Guide](INTEGRATION_GUIDE.md)
- [README.md](../README.md)
- [Whisper Timestamped Library](https://github.com/linto-ai/whisper-timestamped)

---

## Summary

The Whisper Timestamped backend enables accurate, word-level timestamped transcription in real time. By following this guide, you can quickly integrate and use this backend in the ASR Interface. For more advanced customization, refer to the main integration guide and existing backend examples.
