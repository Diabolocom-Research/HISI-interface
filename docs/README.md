# ASR Interface Documentation

Welcome to the ASR Interface documentation! This project provides a modular, real-time Automatic Speech Recognition (ASR) interface with support for various ASR backends.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Custom Backend Integration](#custom-backend-integration)
- [Development](#development)
- [Contributing](#contributing)

## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/asr-interface.git
cd asr-interface

# Install uv if you haven't already
pip install uv

# Create virtual environment and install dependencies
uv sync

# Install in development mode
uv pip install -e .
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/your-username/asr-interface.git
cd asr-interface

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Start the Web Server

```bash
# Using the CLI
asr-interface serve

# Or using uvicorn directly
uvicorn asr_interface.web.server:create_app --reload
```

### Transcribe an Audio File

```bash
# Using the CLI
asr-interface transcribe audio_file.wav --model tiny

# Or via the web interface
# 1. Open http://localhost:8000
# 2. Load a model
# 3. Upload and transcribe your audio file
```

## Architecture

The ASR Interface follows a modular architecture based on the legacy whisper-streaming system with clear separation of concerns:

### Core Components

- **`asr_interface.core`**: Core protocols, configuration, and state management
- **`asr_interface.backends`**: ASR backends and model loaders
- **`asr_interface.handlers`**: Real-time audio processing handlers
- **`asr_interface.web`**: FastAPI web server and API endpoints
- **`asr_interface.utils`**: Utility functions for audio processing
- **`asr_interface.cli`**: Command-line interface

### Key Protocols and Classes

- **`ASRBase`**: Abstract base class for ASR backends (Whisper, MLX Whisper, etc.)
- **`OnlineASRProcessor`**: Main processor that manages audio buffering and hypothesis stabilization
- **`ModelLoader`**: Interface for loading ASR models and creating processors
- **`ASRConfig`**: Configuration model for ASR settings

### Data Flow

1. **Model Loading**: `ModelLoader` creates an `ASRBase` backend and wraps it with `OnlineASRProcessor`
2. **Audio Processing**: `OnlineASRProcessor` manages audio buffering, calls the ASR backend, and stabilizes transcripts
3. **Real-time Streaming**: `RealTimeASRHandler` receives WebRTC audio and feeds it to the processor
4. **Output**: Stabilized transcripts are returned to the client

## API Reference

### Web API Endpoints

#### POST /load_model
Load an ASR model with the specified configuration.

**Request Body:**
```json
{
  "model": "tiny",
  "lan": "en",
  "task": "transcribe",
  "backend": "whisper_timestamped",
  "min_chunk_size": 1.0,
  "buffer_trimming": "segment",
  "buffer_trimming_sec": 10.0
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Processor created successfully."
}
```

#### POST /upload_and_transcribe
Upload and transcribe an audio file.

**Request:** Multipart form with audio file

**Response:** Server-sent events stream with transcription results

#### POST /evaluate_model
Evaluate model performance against a reference transcript.

**Request:** Multipart form with reference text and audio file

**Response:**
```json
{
  "reference": "reference text",
  "transcript": "generated transcript",
  "word_error_rate": 0.15,
  "char_error_rate": 0.08
}
```

### CLI Commands

#### serve
Start the web server.

```bash
asr-interface serve [OPTIONS]

Options:
  --host TEXT     Host to bind to [default: 127.0.0.1]
  --port INTEGER  Port to bind to [default: 8000]
  --reload        Enable auto-reload
  --verbose       Enable verbose logging
  --config PATH   Path to configuration file
```

#### transcribe
Transcribe an audio file directly.

```bash
asr-interface transcribe [OPTIONS] AUDIO_FILE

Options:
  --model TEXT    Whisper model size [default: tiny]
  --language TEXT Language code [default: auto]
  --output PATH   Output file for transcript
  --verbose       Enable verbose logging
```

## Custom Backend Integration

The ASR Interface is designed to make it easy to integrate your own custom ASR models and engines. There are two main integration paths:

### Path 1: Using the Existing Real-Time Engine (Recommended)

This is the most common approach. You provide your own ASR model that conforms to the `ASRBase` protocol, and the system handles all the real-time processing logic.

#### Step 1: Create Your ASR Model

Your model must implement the `ASRBase` protocol:

```python
from asr_interface.core.protocols import ASRBase

class MyCustomASR(ASRBase):
    def __init__(self, model_size: str, language: str, **kwargs):
        # Your custom model loading logic
        self.model = self._load_model(model_size, language)
        super().__init__(language, model_size, **kwargs)

    def transcribe(self, audio, init_prompt=""):
        # Your custom transcription logic
        # Must return a dictionary with a "segments" key
        result = self.model.process(audio, prompt=init_prompt)
        return self._adapt_result_to_standard_format(result)
    
    def ts_words(self, transcription_result):
        # Extract word-level timestamps
        return self._extract_word_timestamps(transcription_result)
    
    def segments_end_ts(self, transcription_result):
        # Extract segment end times
        return self._extract_segment_times(transcription_result)
    
    def use_vad(self):
        # Enable Voice Activity Detection
        return True
```

#### Step 2: Create a Model Loader

```python
from asr_interface.core.protocols import ModelLoader, ASRProcessor
from asr_interface.backends.whisper_loader import OnlineASRProcessor
from .my_custom_asr import MyCustomASR

class MyCustomASRLoader(ModelLoader):
    """Loads the custom ASR model and wraps it in the OnlineASRProcessor."""
    
    def load(self, config: "ASRConfig") -> tuple[ASRProcessor, dict]:
        # 1. Instantiate your custom ASR model
        my_asr_model = MyCustomASR(
            model_size=config.model,
            language=config.lan,
        )

        # 2. Wrap it with the standard OnlineASRProcessor
        online_processor = OnlineASRProcessor(
            asr=my_asr_model,
            buffer_trimming=(config.buffer_trimming, int(config.buffer_trimming_sec)),
            min_chunk_sec=config.min_chunk_size,
        )

        # 3. Define metadata
        metadata = {
            "separator": getattr(my_asr_model, "sep", " "),
            "model_type": "custom",
            "model_size": config.model,
            "language": config.lan,
        }

        return online_processor, metadata
```

#### Step 3: Register Your Loader

```python
from asr_interface.backends.registry import register_loader
from .my_loader import MyCustomASRLoader

# Register your loader
register_loader("my_custom_backend", MyCustomASRLoader())
```

### Path 2: Complete Custom Real-Time Engine (Advanced)

For developers who want complete control over the real-time processing pipeline.

#### Step 1: Implement ASRProcessor Protocol

```python
import numpy as np
from typing import Optional, Tuple
from asr_interface.core.protocols import ASRProcessor

class MyCustomRealTimeEngine(ASRProcessor):
    """A complete replacement for OnlineASRProcessor with custom logic."""
    
    def __init__(self, model_path: str, language: str):
        # Load your own model, tokenizer, etc.
        self.model = self._load_model(model_path)
        self.language = language
        self._internal_buffer = np.array([], dtype=np.float32)
        self.sampling_rate = 16000
        
    def init(self, offset: float = 0.0):
        """Reset the internal state for a new stream."""
        self._internal_buffer = np.array([], dtype=np.float32)
        self.offset = offset

    def insert_audio_chunk(self, audio: np.ndarray):
        """Add incoming audio to your buffer."""
        self._internal_buffer = np.append(self._internal_buffer, audio)

    def process_iter(self) -> Optional[Tuple[float, float, str]]:
        """Process the current audio buffer and return results if available."""
        # Implement your core real-time logic
        if self._should_process_buffer():
            transcript = self.model.transcribe(self._internal_buffer)
            start_time = self.offset
            end_time = start_time + len(self._internal_buffer) / self.sampling_rate
            self._internal_buffer = np.array([], dtype=np.float32)
            return (start_time, end_time, transcript)
        return None

    def finish(self) -> Optional[Tuple[float, float, str]]:
        """Process any remaining audio at the end of the stream."""
        if len(self._internal_buffer) > 0:
            transcript = self.model.transcribe(self._internal_buffer)
            start_time = self.offset
            end_time = start_time + len(self._internal_buffer) / self.sampling_rate
            return (start_time, end_time, transcript)
        return None
```

#### Step 2: Create Model Loader

```python
from asr_interface.core.protocols import ModelLoader
from .processor import MyCustomRealTimeEngine

class MyEngineLoader(ModelLoader):
    """Factory for creating instances of MyCustomRealTimeEngine."""
    
    def load(self, config: "ASRConfig") -> tuple[ASRProcessor, dict]:
        # Instantiate your custom engine
        my_engine = MyCustomRealTimeEngine(
            model_path=config.model,
            language=config.lan
        )

        metadata = {
            "separator": "\n",
            "model_type": "custom_engine",
            "model_path": config.model,
            "language": config.lan,
        }

        return my_engine, metadata
```

#### Step 3: Register and Use

```python
from asr_interface.backends.registry import register_loader
from .my_engine_loader import MyEngineLoader

# Register your loader
register_loader("my_engine", MyEngineLoader())

# Use it with configuration
config = {
    "backend": "my_engine",
    "model": "/path/to/your/model",
    "lan": "en"
}
```

### Configuration Extensions

If your custom backend requires additional configuration parameters, you can extend the `ASRConfig` model:

```python
from asr_interface.core.config import ASRConfig
from pydantic import Field

class ExtendedASRConfig(ASRConfig):
    """Extended configuration for custom backends."""
    
    device: str = Field(default="cpu", description="Device to run model on")
    custom_param: str = Field(default="", description="Custom parameter for your backend")
    model_path: str = Field(default="", description="Path to model files")
```

## Development

### Project Structure

```
asr-interface/
├── asr_interface/          # Main package
│   ├── core/              # Core protocols and configuration
│   ├── backends/          # ASR model loaders
│   ├── handlers/          # Real-time processing handlers
│   ├── web/               # Web server and API
│   ├── utils/             # Utility functions
│   └── cli/               # Command-line interface
├── tests/                 # Test suite
├── docs/                  # Documentation
├── pyproject.toml         # Project configuration
└── README.md             # Project overview
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=asr_interface

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### Code Quality

```bash
# Format code
black asr_interface tests

# Sort imports
isort asr_interface tests

# Lint code
ruff check asr_interface tests

# Type checking
mypy asr_interface
```

### Adding a Custom ASR Backend

1. **Create a Model Loader**:
   ```python
   from asr_interface.core.protocols import ModelLoader, ASRProcessor
   from asr_interface.core.config import ASRConfig
   
   class MyCustomLoader(ModelLoader):
       def load(self, config: ASRConfig) -> tuple[ASRProcessor, dict]:
           # Your loading logic here
           pass
   ```

2. **Register the Loader**:
   ```python
   from asr_interface.backends.registry import register_loader
   
   register_loader("my_backend", MyCustomLoader())
   ```

3. **Use the Backend**:
   ```json
   {
     "backend": "my_backend",
     "model": "my_model"
   }
   ```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `uv sync --group dev`
4. Make your changes
5. Run tests: `pytest`
6. Submit a pull request

### Code Style

- Follow [PEP 8](https://pep8.org/) with Black formatting
- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes
- Add tests for new functionality

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 