# ASR Backend Integration Guide

This comprehensive guide provides everything you need to integrate your own ASR models and engines into the ASR Interface. The system uses a modular, protocol-based architecture that makes it easy to add new backends without modifying the core server logic.

## Table of Contents

- [Quick Start - Choose Your Path](#quick-start---choose-your-path)
- [Architecture Overview](#architecture-overview)
- [Path 1: ASRBase + OnlineASRProcessor (Recommended)](#path-1-asrbase--onlineasrprocessor-recommended)
- [Path 2: Complete Custom ASRProcessor (Advanced)](#path-2-complete-custom-asrprocessor-advanced)
- [Testing Your Integration](#testing-your-integration)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## Quick Start - Choose Your Path

### 🟢 Path 1: ASRBase + OnlineASRProcessor (Recommended)
**Use this if:** You have an ASR model and want to leverage the proven real-time processing logic.

- ✅ **Easiest to implement** - just implement the ASRBase interface
- ✅ **Proven real-time logic** - uses the OnlineASRProcessor
- ✅ **Best for most use cases** - handles audio buffering, hypothesis stabilization, etc.
- ✅ **Legacy compatible** - same interface as whisper-streaming system


### 🔴 Path 2: Complete Custom ASRProcessor (Advanced)
**Use this if:** You need complete control over the real-time processing pipeline.

- ⚠️ **More complex** - you implement everything from scratch
- ⚠️ **More responsibility** - you handle buffering, stabilization, etc.
- ✅ **Maximum flexibility** - complete control over the processing logic
- ✅ **For advanced users** - when you have specific real-time requirements


---

## Architecture Overview

The ASR Interface uses a modular architecture based on the legacy whisper-streaming system:

### Core Components

1. **`ASRBase` Protocol** - Interface for ASR backends (transcription engines)
2. **`OnlineASRProcessor`** - Real-time processing engine (audio buffering, hypothesis stabilization)
3. **`ASRProcessor` Protocol** - Interface for real-time processors
4. **`ModelLoader` Protocol** - Factory for creating processors
5. **`RealTimeASRHandler`** - WebRTC audio stream handler

### Data Flow

```
WebRTC Audio → RealTimeASRHandler → ASRProcessor → Transcript Output
                                    ↓
                              OnlineASRProcessor
                                    ↓
                                ASRBase Backend
```

---

## Path 1: ASRBase + OnlineASRProcessor (Recommended)

This is the most common and straightforward approach. You provide your own ASR model that conforms to the `ASRBase` protocol, and the system handles all the real-time processing logic.

### Step 1: Implement ASRBase Interface

Your ASR backend must implement the `ASRBase` interface. This is the same interface used by the legacy whisper-streaming system.

**Required Methods:**
- `load_model(modelsize, cache_dir, model_dir)` - Load the ASR model
- `transcribe(audio, init_prompt)` - Run transcription on audio buffer
- `ts_words(transcription_result)` - Extract word-level timestamps
- `segments_end_ts(transcription_result)` - Extract segment end timestamps

**Complete Example: `my_custom_asr.py`**

```python
import numpy as np
from typing import Any, List, Tuple
from asr_interface.core.protocols import ASRBase

class MyCustomASR(ASRBase):
    """
    Your custom ASR backend implementation.
    
    This example shows how to integrate a hypothetical ASR model.
    Replace the placeholder methods with your actual model logic.
    """
    
    sep = " "  # Word separator for your model

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        """
        Load your custom ASR model.
        
        Args:
            modelsize: Model size identifier (e.g., "tiny", "base", "large")
            cache_dir: Directory for caching models
            model_dir: Path to pre-downloaded model directory
        """
        print(f"Loading custom model: {modelsize}")
        
        # Example: Load from Hugging Face
        # from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        # self.model = AutoModelForSpeechSeq2Seq.from_pretrained(modelsize)
        # self.processor = AutoProcessor.from_pretrained(modelsize)
        
        # Example: Load from local files
        # self.model = torch.load(f"{model_dir}/model.pth")
        
        # For this example, we'll just store the model size
        self.model_size = modelsize
        return self.model_size

    def transcribe(self, audio, init_prompt=""):
        """
        Transcribe audio using your model.
        
        Args:
            audio: Audio data as numpy array (16kHz, float32)
            init_prompt: Initial prompt for the model
            
        Returns:
            dict: Must contain "segments" key with list of segment objects
        """
        # Example: Using transformers
        # inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        # with torch.no_grad():
        #     generated_ids = self.model.generate(**inputs)
        # transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # For this example, return a mock result
        # In practice, replace this with your actual transcription logic
        mock_result = {
            "text": "This is a mock transcription",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "This is a",
                    "words": [
                        {"start": 0.0, "end": 0.5, "text": "This"},
                        {"start": 0.5, "end": 1.0, "text": "is"},
                        {"start": 1.0, "end": 1.5, "text": "a"}
                    ]
                },
                {
                    "start": 2.0,
                    "end": 3.0,
                    "text": "mock transcription",
                    "words": [
                        {"start": 2.0, "end": 2.5, "text": "mock"},
                        {"start": 2.5, "end": 3.0, "text": "transcription"}
                    ]
                }
            ],
            "language": self.original_language or "en"
        }
        
        return mock_result
    
    def ts_words(self, transcription_result):
        """
        Extract word-level timestamps from transcription result.
        
        Args:
            transcription_result: Result from transcribe() method
            
        Returns:
            List[Tuple[float, float, str]]: List of (start_time, end_time, word) tuples
        """
        words = []
        for segment in transcription_result.get("segments", []):
            for word in segment.get("words", []):
                words.append((
                    word.get("start", 0),
                    word.get("end", 0),
                    word.get("text", "")
                ))
        return words
    
    def segments_end_ts(self, transcription_result):
        """
        Extract segment end timestamps from transcription result.
        
        Args:
            transcription_result: Result from transcribe() method
            
        Returns:
            List[float]: List of segment end timestamps
        """
        return [segment.get("end", 0) for segment in transcription_result.get("segments", [])]

```

### Step 2: Create a ModelLoader

Create a loader that knows how to create your ASR backend and wrap it with OnlineASRProcessor.

**Example: `my_model_loader.py`**

```python
from asr_interface.core.protocols import ModelLoader, ASRProcessor
from asr_interface.core.config import ASRConfig
from asr_interface.backends.whisper_online_processor import OnlineASRProcessor
from .my_custom_asr import MyCustomASR

class MyCustomASRLoader(ModelLoader):
    """
    Loads your custom ASR backend and wraps it in the OnlineASRProcessor.
    """
    
    def load(self, config: ASRConfig) -> tuple[ASRProcessor, dict]:
        """
        Create and return your custom ASR processor.
        
        Args:
            config: Configuration containing model parameters
            
        Returns:
            Tuple of (processor, metadata)
        """
        print("Using MyCustomASRLoader...")

        # 1. Create your ASR backend
        my_asr_backend = MyCustomASR(
            lan=config.lan,
            modelsize=config.model,
            cache_dir=config.model_cache_dir,
            model_dir=config.model_dir
        )

        # 2. Wrap it with OnlineASRProcessor for real-time processing
        online_processor = OnlineASRProcessor(
            asr=my_asr_backend,
            buffer_trimming=(config.buffer_trimming, int(config.buffer_trimming_sec)),
            min_chunk_sec=config.min_chunk_size
        )

        # 3. Define metadata for your backend
        metadata = {
            "separator": my_asr_backend.sep,
            "model_type": "my_custom",
            "model_size": config.model,
            "language": config.lan,
            "backend": "my_custom_backend"
        }

        return online_processor, metadata
```

### Step 3: Register Your Loader

Register your loader so the system can find it.

```python
from asr_interface.backends.registry import register_loader
from .my_model_loader import MyCustomASRLoader

# Register with a unique backend name
register_loader("my_custom_backend", MyCustomASRLoader())
```

### Step 4: Use Your Custom Backend

Send a POST request to `/load_model` with your configuration:

```json
{
  "model": "my_model_size",
  "lan": "en",
  "task": "transcribe",
  "backend": "my_custom_backend",
  "min_chunk_size": 1.0,
  "buffer_trimming": "segment",
  "buffer_trimming_sec": 10.0
}
```

---

## Path 2: Complete Custom ASRProcessor (Advanced)

This approach gives you complete control over the real-time processing pipeline. You implement everything from scratch, including audio buffering, hypothesis stabilization, and output formatting.

### Step 1: Implement ASRProcessor Protocol

Create a class that implements the `ASRProcessor` protocol.

**Required Methods:**
- `init(offset)` - Reset state for new stream
- `insert_audio_chunk(audio)` - Add audio to buffer
- `process_iter()` - Process buffer and return results
- `finish()` - Process remaining audio

**Complete Example: `my_custom_engine.py`**

```python
import numpy as np
from typing import Optional, Tuple
from asr_interface.core.protocols import ASRProcessor

class MyCustomRealTimeEngine(ASRProcessor):
    """
    Complete custom real-time ASR engine.
    
    This example shows how to implement a custom real-time processor
    from scratch. Replace the placeholder logic with your actual implementation.
    """
    
    def __init__(self, model_path: str, language: str, min_chunk_sec: float = 1.0):
        """
        Initialize your custom engine.
        
        Args:
            model_path: Path to your model
            language: Language code
            min_chunk_sec: Minimum chunk size in seconds
        """
        print(f"Initializing MyCustomRealTimeEngine with model: {model_path}")
        
        # Load your model
        self.model = self._load_model(model_path)
        self.language = language
        self.min_chunk_sec = min_chunk_sec
        self.sampling_rate = 16000
        
        # Internal state
        self._audio_buffer = np.array([], dtype=np.float32)
        self._offset = 0.0
        self._last_processed_time = 0.0
        
    def init(self, offset: float = 0.0):
        """
        Reset state for a new audio stream.
        
        Args:
            offset: Time offset for the new stream
        """
        print("Resetting custom engine state.")
        self._audio_buffer = np.array([], dtype=np.float32)
        self._offset = offset
        self._last_processed_time = offset

    def insert_audio_chunk(self, audio: np.ndarray):
        """
        Add incoming audio to the buffer.
        
        Args:
            audio: Audio chunk as numpy array
        """
        self._audio_buffer = np.append(self._audio_buffer, audio)

    def process_iter(self) -> Optional[Tuple[float, float, str]]:
        """
        Process the audio buffer and return results if available.
        
        Returns:
            Optional[Tuple[float, float, str]]: (start_time, end_time, text) if ready, None otherwise
        """
        # Check if we have enough audio to process
        buffer_duration = len(self._audio_buffer) / self.sampling_rate
        if buffer_duration < self.min_chunk_sec:
            return None
        
        # Example: Process when we have enough audio
        # In practice, you might use VAD, silence detection, or other criteria
        if self._should_process_buffer():
            # Transcribe the current buffer
            transcript = self._transcribe_buffer()
            
            if transcript:
                # Calculate timing
                start_time = self._last_processed_time
                end_time = start_time + buffer_duration
                
                # Clear the buffer
                self._audio_buffer = np.array([], dtype=np.float32)
                self._last_processed_time = end_time
                
                return (start_time, end_time, transcript)
        
        return None

    def finish(self) -> Optional[Tuple[float, float, str]]:
        """
        Process any remaining audio at the end of the stream.
        
        Returns:
            Optional[Tuple[float, float, str]]: Final segment if available
        """
        print("Finishing up...")
        if len(self._audio_buffer) > 0:
            transcript = self._transcribe_buffer()
            if transcript:
                start_time = self._last_processed_time
                end_time = start_time + len(self._audio_buffer) / self.sampling_rate
                return (start_time, end_time, transcript)
        return None
    
    def _load_model(self, model_path: str):
        """
        Load your ASR model.
        
        Args:
            model_path: Path to your model
            
        Returns:
            Loaded model object
        """
        # Example: Load your model here
        # return torch.load(model_path)
        # return AutoModel.from_pretrained(model_path)
        return model_path  # Placeholder
    
    def _should_process_buffer(self) -> bool:
        """
        Decide whether to process the current buffer.
        
        Returns:
            bool: True if buffer should be processed
        """
        # Example: Process every 2 seconds
        buffer_duration = len(self._audio_buffer) / self.sampling_rate
        return buffer_duration >= 2.0
    
    def _transcribe_buffer(self) -> Optional[str]:
        """
        Transcribe the current audio buffer.
        
        Returns:
            Optional[str]: Transcription text or None
        """
        # Example: Mock transcription
        # In practice, call your actual model
        # result = self.model.transcribe(self._audio_buffer)
        # return result.text
        
        # Placeholder: return mock text
        return "Mock transcription from custom engine"
```

### Step 2: Create a ModelLoader

```python
from asr_interface.core.protocols import ModelLoader, ASRProcessor
from asr_interface.core.config import ASRConfig
from .my_custom_engine import MyCustomRealTimeEngine

class MyEngineLoader(ModelLoader):
    """
    Factory for creating instances of MyCustomRealTimeEngine.
    """
    
    def load(self, config: ASRConfig) -> tuple[ASRProcessor, dict]:
        """
        Create and return your custom real-time engine.
        
        Args:
            config: Configuration containing model parameters
            
        Returns:
            Tuple of (processor, metadata)
        """
        print("Using MyEngineLoader...")

        # Create your custom engine
        my_engine = MyCustomRealTimeEngine(
            model_path=config.model,
            language=config.lan,
            min_chunk_sec=config.min_chunk_size
        )

        # Define metadata
        metadata = {
            "separator": " ",  # Word separator
            "model_type": "custom_engine",
            "model_path": config.model,
            "language": config.lan,
            "backend": "my_custom_engine"
        }

        return my_engine, metadata
```

### Step 3: Register and Use

```python
# Register your loader
from asr_interface.backends.registry import register_loader
from .my_engine_loader import MyEngineLoader

register_loader("my_custom_engine", MyEngineLoader())

# Use via API
# POST /load_model
{
  "backend": "my_custom_engine",
  "model": "/path/to/your/model",
  "lan": "en",
  "min_chunk_size": 1.0
}
```

---

## Testing Your Integration

### Unit Testing

Create comprehensive tests for your components:

```python
import pytest
import numpy as np
from asr_interface.core.config import ASRConfig

# Test your ASR backend (Path 1)
def test_my_custom_asr():
    from .my_custom_asr import MyCustomASR
    
    # Test initialization
    asr = MyCustomASR("en", "tiny")
    assert asr.original_language == "en"
    
    # Test transcription
    audio = np.random.randn(16000)  # 1 second of audio
    result = asr.transcribe(audio)
    assert "segments" in result
    assert "text" in result
    
    # Test word extraction
    words = asr.ts_words(result)
    assert isinstance(words, list)
    if words:
        assert len(words[0]) == 3  # (start, end, word)
    
    # Test segment extraction
    segments = asr.segments_end_ts(result)
    assert isinstance(segments, list)

# Test your model loader
def test_my_model_loader():
    from .my_model_loader import MyCustomASRLoader
    
    loader = MyCustomASRLoader()
    config = ASRConfig(
        model="tiny",
        lan="en",
        backend="my_custom_backend"
    )
    
    processor, metadata = loader.load(config)
    assert processor is not None
    assert "separator" in metadata
    assert metadata["model_type"] == "my_custom"

# Test your custom engine (Path 2)
def test_my_custom_engine():
    from .my_custom_engine import MyCustomRealTimeEngine
    
    engine = MyCustomRealTimeEngine("/path/to/model", "en")
    engine.init()
    
    # Test audio insertion
    audio = np.random.randn(16000)
    engine.insert_audio_chunk(audio)
    
    # Test processing
    result = engine.process_iter()
    # Assert based on your expected behavior
```

### Integration Testing

Test your backend with the full system:

```python
import asyncio
from asr_interface.web.server import create_app
from fastapi.testclient import TestClient

def test_custom_backend_integration():
    app = create_app()
    client = TestClient(app)
    
    # Test model loading
    response = client.post("/load_model", json={
        "backend": "my_custom_backend",
        "model": "tiny",
        "lan": "en"
    })
    assert response.status_code == 200
    
    # Test transcription
    # Add more integration tests as needed
```

---

## Troubleshooting

### Common Issues

#### Import Errors
```python
# ❌ Wrong
from asr_interface.backends import ASRBase

# ✅ Correct
from asr_interface.core.protocols import ASRBase
```

#### Missing Methods
Make sure your ASR backend implements all required methods:
- `load_model()`
- `transcribe()`
- `ts_words()`
- `segments_end_ts()`
- `use_vad()`

#### Incorrect Return Format
Your `transcribe()` method must return a dict with a "segments" key:
```python
# ✅ Correct format
{
    "text": "transcription text",
    "segments": [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "segment text",
            "words": [
                {"start": 0.0, "end": 1.0, "text": "word"}
            ]
        }
    ],
    "language": "en"
}
```

#### Registration Issues
Make sure your loader is registered before the server starts:
```python
# Register in your module's __init__.py or during startup
from asr_interface.backends.registry import register_loader
register_loader("my_backend", MyLoader())
```

### Debug Tips

1. **Enable logging** to see what's happening:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Test your backend in isolation** before integrating:
```python
# Test your ASR backend directly
asr = MyCustomASR("en", "tiny")
audio = np.random.randn(16000)
result = asr.transcribe(audio)
print(result)
```

3. **Check the registry** to ensure your loader is registered:
```python
from asr_interface.backends.registry import list_loaders
print(list_loaders())  # Should include your backend
```

---

## Examples

### Real-World Examples

See the existing backends for complete, working examples:

- **Whisper Timestamped**: `asr_interface/backends/whisper_timestamped_loader.py`
- **MLX Whisper**: `asr_interface/backends/mlx_whisper_loader.py`

### Quick Templates

#### ASRBase Template (Path 1)
```python
from asr_interface.core.protocols import ASRBase

class MyASRBackend(ASRBase):
    sep = " "
    
    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        # Load your model
        pass
    
    def transcribe(self, audio, init_prompt=""):
        # Transcribe audio
        pass
    
    def ts_words(self, result):
        # Extract word timestamps
        pass
    
    def segments_end_ts(self, result):
        # Extract segment timestamps
        pass
    
    def use_vad(self):
        # Enable VAD
        pass
```

#### ASRProcessor Template (Path 2)
```python
from asr_interface.core.protocols import ASRProcessor

class MyCustomProcessor(ASRProcessor):
    def __init__(self, model_path, language):
        # Initialize
        pass
    
    def init(self, offset=0.0):
        # Reset state
        pass
    
    def insert_audio_chunk(self, audio):
        # Add audio
        pass
    
    def process_iter(self):
        # Process buffer
        pass
    
    def finish(self):
        # Finalize
        pass
```

---

## Need Help?

- 📖 **Documentation**: Check the main [README.md](README.md)
- 🐛 **Issues**: Report bugs on GitHub
- 💬 **Discussions**: Ask questions in GitHub Discussions
- 🔧 **Examples**: Look at existing backends for reference

The ASR Interface is designed to be modular and extensible. With this guide, you should be able to integrate any ASR model or engine quickly and efficiently! 