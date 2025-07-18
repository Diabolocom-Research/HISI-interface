# Advanced Integration: Complete Custom Real-Time ASR Engine (Path 2)

This guide is for developers who want to integrate a completely custom real-time ASR engine into the ASR Interface. This approach gives you full control over the real-time processing pipeline while leveraging the server's FastAPI boilerplate, WebRTC handling, and client communication logic.

## Architectural Overview

The key to this integration is understanding that the server only depends on two specific protocols:

1. **`ASRProcessor` Protocol**: This is the most important protocol. It defines the contract for any real-time processing engine. The server's `RealTimeASRHandler` interacts *only* with this interface to feed in audio and get back transcripts. You implement this to replace the `OnlineASRProcessor`.

2. **`ModelLoader` Protocol**: This is a factory protocol. Its job is to know how to create an instance of your custom `ASRProcessor`.

To integrate your system, you will need to provide your own concrete implementations of these two protocols, giving you complete control over the real-time processing pipeline.

## Integration Process

### Step 1: Implement the ASRProcessor Protocol

First, you must create a Python class that implements the `ASRProcessor` protocol. This class will contain your custom real-time logic (e.g., your own buffering, VAD, model inference, and text stabilization).

Your class must have the following methods:

| Method | Purpose |
|--------|---------|
| `init(offset: float = 0.0)` | Resets the processor's internal state. Called when a new audio stream begins. |
| `insert_audio_chunk(audio: np.ndarray)` | Receives a new chunk of raw audio data from the server. Your class should append this to its internal buffer. |
| `process_iter() -> Optional[Tuple[float, float, str]]` | The core processing method. It should run on your internal audio buffer and return any newly finalized transcript segment. **This is called in a loop.** |
| `finish() -> Optional[Tuple[float, float, str]]` | Called at the end of the stream to process any remaining audio in the buffer and return a final transcript segment. |

**Example: `my_engine/processor.py`**

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

Send a POST request to `/load_model` with your backend configuration:

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

If your custom engine requires additional configuration parameters, you can extend the `ASRConfig` model:

```python
from asr_interface.core.config import ASRConfig
from pydantic import Field

class ExtendedASRConfig(ASRConfig):
    """Extended configuration for custom engines."""
    
    device: str = Field(default="cpu", description="Device to run model on")
    custom_param: str = Field(default="", description="Custom parameter for your engine")
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
from .processor import MyCustomRealTimeEngine
from .loader import MyEngineLoader

def test_my_custom_engine():
    engine = MyCustomRealTimeEngine("/path/to/model", "en")
    engine.init()
    
    # Test audio insertion
    audio = np.random.randn(16000)
    engine.insert_audio_chunk(audio)
    
    # Test processing
    result = engine.process_iter()
    # Assert based on your expected behavior

def test_my_engine_loader():
    loader = MyEngineLoader()
    config = ASRConfig(model="/path/to/model", lan="en", backend="my_engine")
    processor, metadata = loader.load(config)
    assert processor is not None
    assert "separator" in metadata
```

### Integration Testing

Test your engine with the full system:

```python
import asyncio
from asr_interface.web.server import create_app
from fastapi.testclient import TestClient

def test_custom_engine_integration():
    app = create_app()
    client = TestClient(app)
    
    # Test model loading
    response = client.post("/load_model", json={
        "backend": "my_engine",
        "model": "/path/to/model",
        "lan": "en"
    })
    assert response.status_code == 200
    
    # Test transcription
    # ... add transcription test
```

## Example: MLX Whisper Integration

For a complete example, see the MLX Whisper integration in `asr_interface/backends/mlx_whisper_loader.py`. This demonstrates:

- Implementing a complete custom ASRProcessor
- Creating a ModelLoader that instantiates the processor
- Registering the backend in the registry
- Handling model loading and real-time processing

## Advanced Features

### Voice Activity Detection (VAD)

Implement VAD in your `process_iter` method:

```python
def process_iter(self) -> Optional[Tuple[float, float, str]]:
    # Check for voice activity
    if self.vad.is_speech_detected(self._internal_buffer):
        # Process speech segment
        transcript = self.model.transcribe(self._internal_buffer)
        # ... return result
    return None
```

### Buffer Management

Implement sophisticated buffer management:

```python
def _should_process_buffer(self) -> bool:
    # Process when buffer is full or silence is detected
    buffer_duration = len(self._internal_buffer) / self.sampling_rate
    
    if buffer_duration >= self.max_buffer_duration:
        return True
    
    if self._detect_silence(self._internal_buffer[-self.silence_samples:]):
        return True
    
    return False
```

### Real-time Optimization

Optimize for real-time performance:

```python
def process_iter(self) -> Optional[Tuple[float, float, str]]:
    # Use streaming inference if available
    if hasattr(self.model, 'stream_transcribe'):
        return self._stream_process()
    else:
        return self._batch_process()
```

## Troubleshooting

### Common Issues

1. **Timing Issues**: Ensure your timestamps are consistent with the audio stream
2. **Buffer Management**: Avoid memory leaks by properly clearing buffers
3. **Performance**: Profile your processing to ensure real-time performance
4. **Audio Format**: Ensure your engine expects 16kHz, float32 audio

### Debug Tips

1. **Enable Debug Logging**: Set log level to DEBUG to see detailed information
2. **Test Components Individually**: Test your engine separately before integration
3. **Monitor Performance**: Use profiling tools to identify bottlenecks
4. **Validate Audio**: Ensure audio format matches expectations

## Best Practices

1. **Error Handling**: Implement robust error handling in your components
2. **Logging**: Add appropriate logging for debugging and monitoring
3. **Performance**: Optimize your model for real-time processing
4. **Testing**: Write comprehensive tests for your integration
5. **Documentation**: Document your custom parameters and requirements
6. **Memory Management**: Properly manage audio buffers to avoid memory leaks
