# ASR Interface

A modern, modular real-time Automatic Speech Recognition (ASR) interface with support for various ASR backends, built with FastAPI and WebRTC.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Features

- 🎤 **Real-time ASR**: Live transcription via WebRTC audio streaming
- 📁 **File Upload**: Upload and transcribe audio files
- 🔄 **Modular Architecture**: Easy integration of custom ASR backends
- 📊 **Model Evaluation**: Compare ASR performance with reference transcripts
- 🌐 **Web Interface**: Modern, responsive web UI for configuration and monitoring
- 🖥️ **CLI Interface**: Command-line tools for server management
- ⚡ **High Performance**: Optimized for low-latency real-time processing by relying on WebRTC

## Quick Start

### Environment Setup

Before installing the ASR Interface, you need to set up a Python environment:

#### Option 1: Using uv (Recommended)

uv is a fast Python package installer and resolver. It automatically creates virtual environments and manages dependencies.

```bash
# Install uv globally
pip install uv

# uv will automatically create a virtual environment when you run uv sync
```

#### Option 2: Using venv (Traditional)

If you prefer the traditional approach with venv:

```bash
# Create a virtual environment
python -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Or on Windows
# venv\Scripts\activate
```

### Installation

#### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

#### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/asr-interface.git
cd asr-interface

# Install uv if you haven't already
pip install uv

# Create virtual environment and install dependencies
uv sync

# Install the package in development mode
uv pip install -e .
```

#### Using pip

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

# Install the package in development mode
pip install -e .
```

### Start the Server

```bash
# Using the CLI
asr-interface serve

# Or with uvicorn
uvicorn asr_interface.web.server:create_app --reload
```

### Use the Web Interface

1. Open http://localhost:8000 in your browser
2. Configure your ASR model settings
3. Start recording or upload an audio file
4. View real-time transcriptions and segments

## Architecture

The ASR Interface follows a clean, modular architecture based on the legacy whisper-streaming system:

```
asr_interface/
├── core/           # Core protocols and configuration
├── backends/       # ASR backends and model loaders
├── handlers/       # Real-time audio processing
├── web/           # FastAPI web server
├── utils/         # Audio processing utilities
└── cli/           # Command-line interface
```

### Key Components

- **`ASRBase`**: Abstract base class for ASR backends (Whisper, MLX Whisper, etc.)
- **`OnlineASRProcessor`**: Main processor that manages audio buffering and hypothesis stabilization
- **`ModelLoader`**: Protocol for loading ASR models and creating processors
- **`RealTimeASRHandler`**: WebRTC audio stream handler
- **`ASRComponentsStore`**: Thread-safe state management

### Architecture Flow

1. **Model Loading**: `ModelLoader` creates an `ASRBase` backend and wraps it with `OnlineASRProcessor`
2. **Audio Processing**: `OnlineASRProcessor` manages audio buffering, calls the ASR backend, and stabilizes transcripts
3. **Real-time Streaming**: `RealTimeASRHandler` receives WebRTC audio and feeds it to the processor
4. **Output**: Stabilized transcripts are returned to the client

## API Reference

### Web API

- `POST /load_model` - Load ASR model with configuration
- `POST /upload_and_transcribe` - Upload and transcribe audio file
- `POST /evaluate_model` - Evaluate model performance
- `GET /` - Web interface
- `GET /transcript` - Get transcript for WebRTC session

### CLI Commands

```bash
# Start web server
asr-interface serve [--host HOST] [--port PORT] [--reload]

# Transcribe audio file
asr-interface transcribe audio.wav [--model MODEL] [--language LANG]

# Show project info
asr-interface info
```

## Adding Custom ASR Backends

The modular architecture makes it easy to add custom ASR backends. There are two integration paths:

### Path 1: Using Existing Real-Time Engine (Recommended)

For most use cases, you can provide your own ASR backend and reuse the existing `OnlineASRProcessor`:

```python
from asr_interface.core.protocols import ModelLoader, ASRProcessor
from asr_interface.core.config import ASRConfig
from asr_interface.backends.whisper_online_processor import OnlineASRProcessor

class MyCustomASR(ASRBase):
    """Your custom ASR backend implementing ASRBase."""
    
    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        # Load your model
        pass
    
    def transcribe(self, audio, init_prompt=""):
        # Transcribe audio and return result with segments
        pass
    
    def ts_words(self, result):
        # Extract word-level timestamps
        pass
    
    def segments_end_ts(self, result):
        # Extract segment end timestamps
        pass
    
    def use_vad(self):
        # Enable VAD if supported
        pass

class MyCustomLoader(ModelLoader):
    def load(self, config: ASRConfig) -> tuple[ASRProcessor, dict]:
        # Create your ASR backend
        asr_backend = MyCustomASR(
            lan=config.lan,
            modelsize=config.model,
            cache_dir=config.model_cache_dir,
            model_dir=config.model_dir
        )
        
        # Wrap with OnlineASRProcessor
        processor = OnlineASRProcessor(
            asr=asr_backend,
            buffer_trimming=(config.buffer_trimming, int(config.buffer_trimming_sec)),
            min_chunk_sec=config.min_chunk_size
        )
        
        metadata = {"separator": asr_backend.sep, "model_type": "my_custom"}
        return processor, metadata

# Register your loader
from asr_interface.backends.registry import register_loader
register_loader("my_backend", MyCustomLoader())
```

### Path 2: Complete Custom Real-Time Engine (Advanced)

For complete control over the real-time processing pipeline:

```python
from asr_interface.core.protocols import ASRProcessor

class MyCustomRealTimeEngine(ASRProcessor):
    def init(self, offset: float = 0.0):
        # Reset state for new stream
        pass
    
    def insert_audio_chunk(self, audio: np.ndarray):
        # Add audio to buffer
        pass
    
    def process_iter(self) -> Optional[Tuple[float, float, str]]:
        # Process buffer and return results
        pass
    
    def finish(self) -> Optional[Tuple[float, float, str]]:
        # Process remaining audio
        pass
```

For detailed integration guides, see [Custom Backend Integration](docs/INTEGRATION_GUIDE.md).

## Development

### Setup Development Environment

```bash
# Install development dependencies
uv sync --group dev

# Run tests
pytest

# Format code
black asr_interface tests

# Lint code
ruff check asr_interface tests

# Type checking
mypy asr_interface
```

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
└── README.md             # This file
```

## Documentation

For detailed documentation, see the [docs/](docs/) directory:

- [Installation Guide](docs/README.md#installation)
- [API Reference](docs/README.md#api-reference)
- [Architecture Overview](docs/README.md#architecture)
- [Development Guide](docs/README.md#development)

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/README.md#contributing) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- Uses [WebRTC](https://webrtc.org/) for real-time audio streaming
- Supports [Whisper](https://github.com/openai/whisper) and other ASR models
- Audio processing powered by [librosa](https://librosa.org/)

## Troubleshooting

### Common Issues

#### Python Version
Make sure you have Python 3.10 or higher:
```bash
python --version
```

#### Virtual Environment Issues
If you encounter permission errors or import issues:
```bash
# Deactivate any existing environment
deactivate

# Remove and recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### uv Issues
If uv sync fails:
```bash
# Clear uv cache
uv cache clean

# Try again
uv sync
```

#### Dependencies Issues
If you encounter dependency conflicts:
```bash
# With uv
uv sync --reinstall

# With pip
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Support

- 📖 [Documentation](docs/README.md)
- 🐛 [Issue Tracker](https://github.com/your-username/asr-interface/issues)
- 💬 [Discussions](https://github.com/your-username/asr-interface/discussions)