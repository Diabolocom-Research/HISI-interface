# Integrating a Custom ASR Model (Path 1: Using Existing Real-Time Engine)

This guide explains how to integrate your own Automatic Speech Recognition (ASR) model into the ASR Interface using the existing real-time processing engine. This is the **recommended approach** for most use cases as it requires minimal custom code while providing full real-time functionality.

## Architectural Overview

The ASR Interface uses a modular architecture based on the legacy whisper-streaming system with three key components:

1. **`ASRBase` Protocol (Backend Contract)**: This is the foundational interface that your underlying ASR model must implement. It defines methods like `transcribe()`, `ts_words()`, `segments_end_ts()`, etc. This is the same interface used by the legacy system.

2. **`OnlineASRProcessor` (Real-Time Engine)**: The main processor that handles audio buffering, hypothesis stabilization, and buffer management. It wraps your ASR backend and provides the real-time processing logic.

3. **`ModelLoader` Protocol (Factory)**: A dedicated class that knows how to create an `ASRBase` backend and wrap it with `OnlineASRProcessor`. The server maintains a registry to find the correct loader based on configuration.

## Integration Process

### Step 1: Implement ASRBase Interface

Your ASR backend must implement the `ASRBase` interface, which is the same interface used by the legacy whisper-streaming system:

- `__init__(lan, modelsize, cache_dir, model_dir, logfile)`: Initialize the backend
- `load_model(modelsize, cache_dir, model_dir)`: Load the ASR model
- `transcribe(audio, init_prompt)`: Run transcription on an audio buffer
- `ts_words(transcription_result)`: Extract word-level timestamps
- `segments_end_ts(transcription_result)`: Extract segment end times
- `use_vad()`: Enable Voice Activity Detection

**Example: `my_custom_asr.py`**

```python
import sys
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

class ASRBase(ABC):
    """Abstract base class for ASR backends."""
    sep = " "  # Default separator

    def __init__(self, lan: str, modelsize: str = None, cache_dir: str = None, model_dir: str = None, logfile=sys.stderr):
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.original_language = None if lan == "auto" else lan
        self.model = self.load_model(modelsize, cache_dir, model_dir)

    @abstractmethod
    def load_model(self, modelsize: str = None, cache_dir: str = None, model_dir: str = None):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def transcribe(self, audio, init_prompt: str = "") -> dict:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def ts_words(self, transcription_result: Any) -> List[Tuple[float, float, str]]:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def segments_end_ts(self, transcription_result: Any) -> List[float]:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def use_vad(self):
        raise NotImplementedError("must be implemented in the child class")

class MyCustomASR(ASRBase):
    """Your custom ASR backend implementation."""
    
    sep = " "  # Word separator

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        # Load your custom model
        print(f"Loading custom model: {modelsize}")
        return self._load_my_model(modelsize, cache_dir, model_dir)

    def transcribe(self, audio, init_prompt=""):
        # Your custom transcription logic
        # Must return a dictionary with a "segments" key
        result = self.model.process(audio, prompt=init_prompt, **self.transcribe_kargs)
        return self._adapt_result_to_standard_format(result)
    
    def ts_words(self, transcription_result):
        # Extract word-level timestamps from the result
        # Return list of (start_time, end_time, word) tuples
        return self._extract_word_timestamps(transcription_result)
    
    def segments_end_ts(self, transcription_result):
        # Extract segment end times
        # Return list of segment end timestamps
        return self._extract_segment_times(transcription_result)
    
    def use_vad(self):
        # Enable Voice Activity Detection
        self.transcribe_kargs["vad"] = True
    
    def _load_my_model(self, modelsize, cache_dir, model_dir):
        # Implement your model loading logic
        pass
    
    def _adapt_result_to_standard_format(self, raw_result):
        # Convert your model's output to the expected format
        # Should return a dict with "segments" key
        pass
```

### Step 2: Create a Custom ModelLoader

Create a new file, for example, `my_model_loader.py`:

```python
import sys
from asr_interface.core.protocols import ModelLoader, ASRProcessor
from asr_interface.core.config import ASRConfig
from asr_interface.backends.whisper_online_processor import OnlineASRProcessor
from .my_custom_asr import MyCustomASR

class MyCustomASRLoader(ModelLoader):
    """
    Loads the custom ASR model and wraps it in the OnlineASRProcessor.
    """
    def load(self, config: ASRConfig) -> tuple[ASRProcessor, dict]:
        """
        Contains the specific logic to initialize and return our custom ASR processor.
        """
        print("Using MyCustomASRLoader...")

        # 1. Instantiate your custom ASRBase-compliant backend
        my_asr_backend = MyCustomASR(
            lan=config.lan,
            modelsize=config.model,
            cache_dir=config.model_cache_dir,
            model_dir=config.model_dir
        )

        # 2. Wrap it with the standard OnlineASRProcessor
        online_processor = OnlineASRProcessor(
            asr=my_asr_backend,
            buffer_trimming=(config.buffer_trimming, int(config.buffer_trimming_sec)),
            min_chunk_sec=config.min_chunk_size
        )

        # 3. Define any specific metadata for your model
        metadata = {
            "separator": my_asr_backend.sep,
            "model_type": "custom",
            "model_size": config.model,
            "language": config.lan,
        }

        # 4. Return the processor and metadata
        return online_processor, metadata
```

### Step 3: Register Your Loader

```python
from asr_interface.backends.registry import register_loader
from .my_model_loader import MyCustomASRLoader

# Register your loader with a unique backend name
register_loader("my_custom_backend", MyCustomASRLoader())
```

### Step 4: Use Your Custom Backend

Send a POST request to `/load_model` with your backend configuration:

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

## Configuration Extensions

If your custom backend requires additional configuration parameters, you can extend the `ASRConfig` model:

```python
from asr_interface.core.config import ASRConfig
from pydantic import Field

class ExtendedASRConfig(ASRConfig):
    """Extended configuration for custom backends."""
    
    device: str = Field(default="cpu", description="Device to run model on")
    custom_param: str = Field(default="", description="Custom parameter for your backend")
    model_path: str = Field(default="", description="Path to model files")
    batch_size: int = Field(default=1, description="Batch size for processing")
    precision: str = Field(default="float32", description="Model precision")
```

## Testing Your Integration

### Unit Testing

Create tests for your custom components:

```python
import pytest
import numpy as np
from .my_custom_asr import MyCustomASR
from .my_model_loader import MyCustomASRLoader

def test_my_custom_asr():
    asr = MyCustomASR("en", "tiny")
    audio = np.random.randn(16000)  # 1 second of audio
    result = asr.transcribe(audio)
    assert "segments" in result

def test_my_model_loader():
    loader = MyCustomASRLoader()
    config = ASRConfig(model="tiny", lan="en", backend="my_custom_backend")
    processor, metadata = loader.load(config)
    assert processor is not None
    assert "separator" in metadata
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
    # ... add transcription test
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure your custom modules are in the Python path
2. **Protocol Compliance**: Ensure your classes implement all required methods
3. **Configuration**: Verify your configuration parameters match your loader's expectations
4. **Audio Format**: Ensure your model expects the same audio format (16kHz, float32)

### Debug Tips

1. **Enable Debug Logging**: Set log level to DEBUG to see detailed information
2. **Test Components Individually**: Test your model and loader separately before integration
3. **Check Registry**: Verify your loader is properly registered
4. **Validate Configuration**: Use Pydantic validation to catch configuration errors

## Example: MLX Whisper Integration

For a complete example, see the MLX Whisper integration in `asr_interface/backends/mlx_whisper_loader.py`. This demonstrates:

- Implementing a custom ASR model that conforms to ASRBase
- Creating a ModelLoader that wraps the model with OnlineASRProcessor
- Registering the backend in the registry
- Handling optional dependencies gracefully

## Best Practices

1. **Error Handling**: Implement robust error handling in your components
2. **Logging**: Add appropriate logging for debugging and monitoring
3. **Documentation**: Document your custom parameters and requirements
4. **Testing**: Write comprehensive tests for your integration
5. **Performance**: Optimize your model for real-time processing
6. **Compatibility**: Ensure your model works with the expected audio format