# Integrating a Custom Real-Time ASR Model

This guide explains how to integrate your own real-time Automatic Speech Recognition (ASR) model into the provided FastAPI server framework. The server is built on a modular, protocol-oriented design that makes it easy to add new ASR backends without modifying the core server logic.

### Architectural Overview

The server's design separates the "how" of model loading from the "what" of the server's operation. This is achieved through three key concepts:

1.  **`ASRBase` Protocol (The Model Contract):** This is the foundational interface that your underlying ASR model must adhere to. It defines methods like `transcribe()`, `ts_words()`, etc. The `whisper_online` library already provides this.

2.  **`ASRProcessor` Protocol (The Real-Time Wrapper Contract):** The server doesn't interact with your base model directly. It communicates with a real-time "processor" that handles buffering, streaming logic, and state management. The `OnlineASRProcessor` class is a ready-to-use implementation of this protocol.

3.  **`ModelLoader` Protocol & Registry (The Factory):** This is the main extension point. A `ModelLoader` is a dedicated class whose only job is to know how to create an instance of your `ASRProcessor`. The server maintains a registry (`MODEL_LOADERS` dictionary) to find the correct loader based on a configuration string.

The integration process involves creating your own components that fulfill these contracts and registering them with the server.

---

### Step 1: Ensure Your Model Conforms to `ASRBase`

Before you begin, you need a Python class for your ASR model that conforms to the `ASRBase` interface defined in `whisper_online`. This means it must have the following methods:

-   `__init__(...)`: To initialize the model with parameters like language, model size, etc.
-   `transcribe(audio, init_prompt)`: To run transcription on an audio buffer.
-   `ts_words(transcription_result)`: To extract word-level timestamps from the result.
-   `segments_end_ts(transcription_result)`: To extract segment end times.
-   `use_vad()`: To enable Voice Activity Detection.

If you have built your model by inheriting from `whisper_online.ASRBase`, you have already completed this step. If not, you will need to create an **adapter class** that wraps your model and exposes these methods.

**Example: `my_custom_asr.py`**
```python
from real_time_asr_backend.slimer_whisper_online import ASRBase

class MyCustomASR(ASRBase):
    def __init__(self, modelsize, lan, **kwargs):
        # Your custom model loading logic
        print(f"Loading MY custom model: {modelsize}")
        self.model = ... # Load your model weights/files here
        super().__init__(lan, modelsize, **kwargs)

    def transcribe(self, audio, init_prompt=""):
        # Your custom transcription logic
        # Must return a dictionary with a "segments" key
        raw_result = self.model.process(audio, prompt=init_prompt)
        return self._adapt_result_to_standard_format(raw_result)
    
    # ... implement ts_words, segments_end_ts, and use_vad ...
```

### Step 2: Reuse the `OnlineASRProcessor`

The `OnlineASRProcessor` is designed to work with *any* object that follows the `ASRBase` contract. It handles all the complex real-time logic like audio buffering, transcript stabilization (`HypothesisBuffer`), and buffer trimming.

For most use cases, **you do not need to write your own real-time processor.** You can simply wrap your `ASRBase`-compliant model from Step 1 with the existing `OnlineASRProcessor`.

### Step 3: Create a Custom `ModelLoader`

This is the primary piece of code you need to write. You will create a new class that inherits from `ModelLoader` and knows how to instantiate your specific ASR model and wrap it with the `OnlineASRProcessor`.

Create a new file, for example, `my_model_loader.py`:

**`my_model_loader.py`**
```python
import sys
from real_time_asr_backend.real_time_asr_protocols import ModelLoader, ASRProcessor
from real_time_asr_backend.slimer_whisper_online import OnlineASRProcessor
from .my_custom_asr import MyCustomASR  # Import your custom ASR class from Step 1

class MyCustomASRLoader(ModelLoader):
    """
    Loads the custom ASR model and wraps it in the OnlineASRProcessor.
    """
    def load(self, config: "ASRConfig") -> tuple[ASRProcessor, dict]:
        """
        Contains the specific logic to initialize and return our custom ASR processor.
        """
        print("Using MyCustomASRLoader...")

        # 1. Instantiate your custom ASRBase-compliant model
        my_asr_model = MyCustomASR(
            modelsize=config.model,
            lan=config.lan,
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
            "separator": getattr(my_asr_model, "sep", " ")
        }

        # 4. Return the processor and metadata
        return online_processor, metadata
```

### Step 4: Register Your New Loader in `server.py`

Now, you just need to tell the server about your new loader. Open `server.py` and make two small changes:

1.  **Import your new loader class.**
2.  **Add an entry to the `MODEL_LOADERS` dictionary.**

**`server.py` (Modified)**
```python
# ... other imports ...
from real_time_asr_backend.real_time_asr_protocols import ModelLoader
from real_time_asr_backend.slimer_whisper_online import WhisperOnlineLoader

# 1. Import your new loader
from my_model_loader import MyCustomASRLoader 

# ...

# 2. Add your loader to the registry with a unique backend name
MODEL_LOADERS: dict[str, ModelLoader] = {
    "whisper_timestamped": WhisperOnlineLoader(),
    "my_custom_backend": MyCustomASRLoader()  # Add your new entry here
}

# ... rest of the server code remains unchanged ...
```

### Step 5: Launch and Test

Your integration is now complete. To use your custom model, you need to send a POST request to the `/load_model` endpoint with the `backend` field set to the name you chose in the registry.

1.  **Run the server:** `python server.py`
2.  **Send the configuration request:** You can use a tool like `curl` or a Python script.

The server will now use your `MyCustomASRLoader` to initialize `MyCustomASR`, wrap it in the `OnlineASRProcessor`, and make it ready to handle real-time audio streams.