# Advanced Integration: Bringing Your Own Real-Time ASR Engine

This guide is for developers who want to go beyond extending `whisper-online` and integrate a completely custom real-time ASR engine into the server. This allows you to leverage the server's FastAPI boilerplate, WebRTC handling (`fastrtc`), and client communication logic while swapping out the core audio processing component.

### Architectural Prerequisite

The key to this integration is understanding that the server does not depend on `whisper-online` itself. It only depends on two specific protocols (interfaces) defined in `real_time_asr_protocols.py`:

1.  **`ASRProcessor`**: This is the most important protocol. It defines the contract for any real-time processing engine. The server's `RealTimeASRHandler` interacts *only* with this interface to feed in audio and get back transcripts.
2.  **`ModelLoader`**: This is a factory protocol. Its job is to know how to create an instance of your custom `ASRProcessor`.

To integrate your system, you will need to provide your own concrete implementations of these two protocols.

---

### Step 1: Implement the `ASRProcessor` Protocol

First, you must create a Python class that implements the `ASRProcessor` protocol. This class will contain your custom real-time logic (e.g., your own buffering, VAD, model inference, and text stabilization).

Your class must have the following methods:

| Method                                      | Purpose                                                                                                                                              |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `init()`                                    | Resets the processor's internal state. Called when a new audio stream begins.                                                                        |
| `insert_audio_chunk(audio: np.ndarray)`     | Receives a new chunk of raw audio data from the server. Your class should append this to its internal buffer.                                      |
| `process_iter() -> Tuple[float, float, str]` | The core processing method. It should run on your internal audio buffer and return any newly finalized transcript segment. **This is called in a loop.** |
| `finish() -> Tuple[float, float, str]`      | Called at the end of the stream to process any remaining audio in the buffer and return a final transcript segment.                                  |

**Example: `my_engine/processor.py`**

Here is a skeleton for your custom processor. You will fill in the logic with your own implementation.

```python
import numpy as np
from typing import Tuple
from real_time_asr_backend.real_time_asr_protocols import ASRProcessor

class MyCustomRealTimeEngine(ASRProcessor):
    """
    A complete replacement for OnlineASRProcessor with custom logic.
    """
    def __init__(self, model_path: str, lang: str):
        # 1. Load your own model, tokenizer, etc.
        print(f"Initializing MyCustomRealTimeEngine with model: {model_path}")
        # self.model = YourModel.load(model_path)
        # self.tokenizer = YourTokenizer.load(...)
        self.lang = lang
        self._internal_buffer = np.array([], dtype=np.float32)
        self.SAMPLING_RATE = 16000
        
    def init(self):
        # 2. Reset the internal state for a new stream
        print("Resetting custom engine state.")
        self._internal_buffer = np.array([], dtype=np.float32)
        # Reset any other stateful variables (e.g., VAD state, conversation history)

    def insert_audio_chunk(self, audio: np.ndarray):
        # 3. Add incoming audio to your buffer
        self._internal_buffer = np.append(self._internal_buffer, audio)

    def process_iter(self) -> Tuple[float, float, str]:
        # 4. Implement your core real-time logic
        # This is the most complex part. You need to decide when and how
        # to process the self._internal_buffer.
        
        # Example logic:
        # if self.vad.is_speech_ended(self._internal_buffer):
        #     transcript = self.model.transcribe(self._internal_buffer)
        #     self._internal_buffer = np.array([], dtype=np.float32) # Clear buffer
        #     # The times should be absolute timestamps in the stream
        #     return (start_time, end_time, transcript) 

        # If no new segment is finalized in this iteration, return empty.
        return (None, None, "")

    def finish(self) -> Tuple[float, float, str]:
        # 5. Process any leftover audio in the buffer at the end of the stream
        print("Finishing up...")
        if len(self._internal_buffer) > 0:
            # transcript = self.model.transcribe(self._internal_buffer)
            # return (start_time, end_time, transcript)
            pass
        return (None, None, "")
```

### Step 2: Implement the `ModelLoader` Protocol

Next, you need to create a "factory" class that knows how to instantiate your `MyCustomRealTimeEngine`. This loader will read from the user's JSON configuration and pass the necessary parameters to your engine's constructor.

**Example: `my_engine/loader.py`**

```python
from real_time_asr_backend.real_time_asr_protocols import ModelLoader, ASRProcessor
from .processor import MyCustomRealTimeEngine # Import your engine from Step 1

class MyEngineLoader(ModelLoader):
    """
    Factory for creating instances of MyCustomRealTimeEngine.
    """
    def load(self, config: "ASRConfig") -> tuple[ASRProcessor, dict]:
        """
        Initializes and returns the custom real-time engine.
        """
        print("Using MyEngineLoader...")

        # 1. Instantiate your custom engine using parameters from the config
        my_engine_instance = MyCustomRealTimeEngine(
            model_path=config.model,  # 'model' field from the user's JSON
            lang=config.lan
        )

        # 2. Define any metadata your system might need.
        # The separator is used by the server to join transcript segments.
        metadata = {
            "separator": "\n" # Or " ", whatever your engine prefers
        }

        # 3. Return the fully initialized processor and its metadata
        return my_engine_instance, metadata
```

### Step 3: Register Your `ModelLoader` in the Server

The final step is to make the server aware of your new engine. Open `server.py` and add your loader to the `MODEL_LOADERS` registry.

**`server.py` (Modified)**

```python
# ... other imports ...
from real_time_asr_backend.slimer_whisper_online import WhisperOnlineLoader

# 1. Import your new loader class
from my_engine.loader import MyEngineLoader 

# ...

# 2. Add your loader to the registry with a unique 'backend' name.
# This name is what users will specify in their JSON config.
MODEL_LOADERS: dict[str, ModelLoader] = {
    "whisper_timestamped": WhisperOnlineLoader(),
    "my_engine": MyEngineLoader()  # <-- YOUR NEW ENTRY
}

# ... rest of the server code remains unchanged ...
```

### Step 4: Configure and Launch

You are now ready to run the server with your custom engine.

1.  **(Optional) Customize Config Model**: If your engine requires configuration parameters that are not in the default `ASRConfig` Pydantic model in `server.py`, you can add them. For example, if you need a `device` parameter:

    ```python
    class ASRConfig(BaseModel, extra=Extra.allow):
        model: str
        lan: str = "auto"
        device: str = "cpu"  # <-- Your new custom field
        backend: str = "my_engine"
        # ... other fields
    ```

2.  **Launch the Server**:

    ```bash
    python server.py
    ```

The server will now use your `MyEngineLoader` to create an instance of your `MyCustomRealTimeEngine`, and the `RealTimeASRHandler` will start feeding it audio data from connected clients. You have successfully integrated a completely custom backend!
