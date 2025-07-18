# Custom ASR Backend Integration Guide

This guide provides detailed instructions for integrating your own custom ASR models and engines into the ASR Interface. The system is designed with a modular, protocol-based architecture that makes it easy to add new backends without modifying the core server logic.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Integration Path 1: Using Existing Real-Time Engine](#integration-path-1-using-existing-real-time-engine)
- [Integration Path 2: Complete Custom Real-Time Engine](#integration-path-2-complete-custom-real-time-engine)
- [Configuration Extensions](#configuration-extensions)
- [Testing Your Integration](#testing-your-integration)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

The ASR Interface uses three key protocols to enable easy integration:

### 1. ASRBase Protocol (Model Contract)
This is the foundational interface that your underlying ASR model must implement. It defines methods like `transcribe()`, `ts_words()`, etc.

### 2. ASRProcessor Protocol (Real-Time Wrapper Contract)
The server communicates with a real-time "processor" that handles buffering, streaming logic, and state management.

### 3. ModelLoader Protocol (Factory)
A dedicated class that knows how to create an instance of your `ASRProcessor`. The server maintains a registry to find the correct loader based on configuration.

## Integration Path 1: Using Existing Real-Time Engine (Recommended)

This is the most common and straightforward approach. You provide your own ASR model that conforms to the `ASRBase` protocol, and the system handles all the real-time processing logic.

### Step 1: Ensure Your Model Conforms to ASRBase

Your ASR model must implement the following methods:

- `__init__(...)`: Initialize the model with parameters
- `transcribe(audio, init_prompt)`: Run transcription on an audio buffer
- `ts_words(transcription_result)`: Extract word-level timestamps
- `segments_end_ts(transcription_result)`: Extract segment end times
- `use_vad()`: Enable Voice Activity Detection

**Example: `my_custom_asr.py`**

```python
from asr_interface.core.protocols import ASRBase

class MyCustomASR(ASRBase):
    def __init__(self, model_size: str, language: str, **kwargs):
        # Your custom model loading logic
        print(f"Loading MY custom model: {model_size}")
        self.model = self._load_model(model_size, language)
        super().__init__(language, model_size, **kwargs)

    def transcribe(self, audio, init_prompt=""):
        # Your custom transcription logic
        # Must return a dictionary with a "segments" key
        raw_result = self.model.process(audio, prompt=init_prompt)
        return self._adapt_result_to_standard_format(raw_result)
    
    def ts_words(self, transcription_result):
        # Extract word-level timestamps from the result
        return self._extract_word_timestamps(transcription_result)
    
    def segments_end_ts(self, transcription_result):
        # Extract segment end times
        return self._extract_segment_times(transcription_result)
    
    def use_vad(self):
        # Enable Voice Activity Detection
        return True
    
    def _load_model(self, model_size: str, language: str):
        # Implement your model loading logic
        # This could load from Hugging Face, local files, etc.
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
from asr_interface.backends.whisper_loader import OnlineASRProcessor
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

        # 1. Instantiate your custom ASRBase-compliant model
        my_asr_model = MyCustomASR(
            model_size=config.model,
            language=config.lan,
            # Pass any other relevant config args here
        )

        # 2. Wrap it with the standard OnlineASRProcessor
        online_processor = OnlineASRProcessor(
            asr=my_asr_model,
            buffer_trimming=(config.buffer_trimming, int(config.buffer_trimming_sec)),
            min_chunk_sec=config.min_chunk_size,
            logfile=sys.stderr
        )

        # 3. Define any specific metadata for your model
        metadata = {
            "separator": getattr(my_asr_model, "sep", " "),
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

## Integration Path 2: Complete Custom Real-Time Engine (Advanced)

This approach is for developers who want to replace the entire real-time processing pipeline with their own custom logic.

### Step 1: Implement the ASRProcessor Protocol

Create a class that implements the `ASRProcessor` protocol:

```python
import numpy as np
from typing import Optional, Tuple
from asr_interface.core.protocols import ASRProcessor

class MyCustomRealTimeEngine(ASRProcessor):
    """
    A complete replacement for OnlineASRProcessor with custom logic.
    """
    def __init__(self, model_path: str, language: str):
        # 1. Load your own model, tokenizer, etc.
        print(f"Initializing MyCustomRealTimeEngine with model: {model_path}")
        self.model = self._load_model(model_path)
        self.language = language
        self._internal_buffer = np.array([], dtype=np.float32)
        self.sampling_rate = 16000
        self.offset = 0.0
        
    def init(self, offset: float = 0.0):
        # 2. Reset the internal state for a new stream
        print("Resetting custom engine state.")
        self._internal_buffer = np.array([], dtype=np.float32)
        self.offset = offset
        # Reset any other stateful variables (e.g., VAD state, conversation history)

    def insert_audio_chunk(self, audio: np.ndarray):
        # 3. Add incoming audio to your buffer
        self._internal_buffer = np.append(self._internal_buffer, audio)

    def process_iter(self) -> Optional[Tuple[float, float, str]]:
        # 4. Implement your core real-time logic
        # This is the most complex part. You need to decide when and how
        # to process the self._internal_buffer.
        
        # Example logic:
        if self._should_process_buffer():
            transcript = self.model.transcribe(self._internal_buffer)
            start_time = self.offset
            end_time = start_time + len(self._internal_buffer) / self.sampling_rate
            self._internal_buffer = np.array([], dtype=np.float32)  # Clear buffer
            self.offset = end_time
            return (start_time, end_time, transcript)

        # If no new segment is finalized in this iteration, return None
        return None

    def finish(self) -> Optional[Tuple[float, float, str]]:
        # 5. Process any leftover audio in the buffer at the end of the stream
        print("Finishing up...")
        if len(self._internal_buffer) > 0:
            transcript = self.model.transcribe(self._internal_buffer)
            start_time = self.offset
            end_time = start_time + len(self._internal_buffer) / self.sampling_rate
            return (start_time, end_time, transcript)
        return None
    
    def _load_model(self, model_path: str):
        # Implement your model loading logic
        pass
    
    def _should_process_buffer(self) -> bool:
        # Implement your logic to decide when to process the buffer
        # This could be based on VAD, buffer size, silence detection, etc.
        return len(self._internal_buffer) > self.sampling_rate * 2  # Example: 2 seconds
```

### Step 2: Create a ModelLoader

```python
from asr_interface.core.protocols import ModelLoader, ASRProcessor
from asr_interface.core.config import ASRConfig
from .processor import MyCustomRealTimeEngine

class MyEngineLoader(ModelLoader):
    """
    Factory for creating instances of MyCustomRealTimeEngine.
    """
    def load(self, config: ASRConfig) -> tuple[ASRProcessor, dict]:
        """
        Initializes and returns the custom real-time engine.
        """
        print("Using MyEngineLoader...")

        # 1. Instantiate your custom engine using parameters from the config
        my_engine_instance = MyCustomRealTimeEngine(
            model_path=config.model,  # 'model' field from the user's JSON
            language=config.lan
        )

        # 2. Define any metadata your system might need.
        # The separator is used by the server to join transcript segments.
        metadata = {
            "separator": "\n",  # Or " ", whatever your engine prefers
            "model_type": "custom_engine",
            "model_path": config.model,
            "language": config.lan,
        }

        # 3. Return the fully initialized processor and its metadata
        return my_engine_instance, metadata
```

### Step 3: Register Your Loader

```python
from asr_interface.backends.registry import register_loader
from .my_engine_loader import MyEngineLoader

# Register your loader with a unique backend name
register_loader("my_engine", MyEngineLoader())
```

### Step 4: Use Your Custom Engine

```json
{
  "backend": "my_engine",
  "model": "/path/to/your/model",
  "lan": "en",
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

### 1. Unit Testing

Create tests for your custom components:

```python
import pytest
import numpy as np
from .my_custom_asr import MyCustomASR
from .my_model_loader import MyCustomASRLoader

def test_my_custom_asr():
    asr = MyCustomASR("tiny", "en")
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

### 2. Integration Testing

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

### Getting Help

- Check the [main documentation](README.md)
- Review the [API reference](docs/README.md#api-reference)
- Open an issue on GitHub with detailed error information
- Join the community discussions

## Example Projects

For complete examples, see the following:

- [Whisper Integration](asr_interface/backends/whisper_loader.py) - Example of Path 1 integration
- [Custom Backend Template](examples/custom_backend_template/) - Template for new backends
- [Integration Tests](tests/integration/) - Examples of testing integrations

## Best Practices

1. **Error Handling**: Implement robust error handling in your components
2. **Logging**: Add appropriate logging for debugging and monitoring
3. **Documentation**: Document your custom parameters and requirements
4. **Testing**: Write comprehensive tests for your integration
5. **Performance**: Optimize your model for real-time processing
6. **Compatibility**: Ensure your model works with the expected audio format 