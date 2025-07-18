"""Main CLI entry point for the ASR interface."""

import logging
import sys
from pathlib import Path
from typing import Optional

import uvicorn
import typer
from rich.console import Console
from rich.logging import RichHandler

from ..core.store import ASRComponentsStore
from ..web.server import create_app

# Create CLI app
app = typer.Typer(
    name="asr-interface",
    help="Real-time Automatic Speech Recognition Interface",
    add_completion=False,
)

# Rich console for better output
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """
    Setup logging configuration.
    
    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
) -> None:
    """
    Start the ASR interface web server.
    
    This command starts the FastAPI web server that provides:
    - Real-time ASR transcription via WebRTC
    - File upload and transcription
    - Model evaluation capabilities
    - Web interface for configuration and monitoring
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Create the ASR components store
        store = ASRComponentsStore()
        
        # Load configuration if provided
        if config_file and config_file.exists():
            logger.info(f"Loading configuration from {config_file}")
            # TODO: Implement configuration loading
        
        # Create the FastAPI application
        fastapi_app = create_app(store)
        
        logger.info(f"Starting ASR interface server on {host}:{port}")
        logger.info("Press Ctrl+C to stop the server")
        
        # Start the server
        uvicorn.run(
            fastapi_app,
            host=host,
            port=port,
            reload=reload,
            log_level="debug" if verbose else "info",
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def transcribe(
    audio_file: Path = typer.Argument(..., help="Path to audio file to transcribe"),
    model: str = typer.Option("tiny", "--model", "-m", help="Whisper model size"),
    language: str = typer.Option("auto", "--language", "-l", help="Language code"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for transcript"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """
    Transcribe an audio file using the ASR interface.
    
    This command provides a simple way to transcribe audio files without
    starting the full web server.
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    if not audio_file.exists():
        console.print(f"[red]Error: Audio file '{audio_file}' does not exist[/red]")
        sys.exit(1)
    
    try:
        logger.info(f"Transcribing {audio_file} with model {model}")
        
        # TODO: Implement direct transcription without web server
        # This would require creating a minimal ASR processor instance
        
        console.print("[yellow]Direct transcription not yet implemented[/yellow]")
        console.print("Please use the web server for transcription")
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def info() -> None:
    """Display information about the ASR interface."""
    from .. import __version__, __author__
    
    console.print(f"[bold blue]ASR Interface[/bold blue] v{__version__}")
    console.print(f"Author: {__author__}")
    console.print()
    console.print("[bold]Features:[/bold]")
    console.print("• Real-time ASR transcription via WebRTC")
    console.print("• File upload and transcription")
    console.print("• Model evaluation and comparison")
    console.print("• Modular architecture for custom ASR backends")
    console.print()
    console.print("[bold]Usage:[/bold]")
    console.print("• Start server: asr-interface serve")
    console.print("• Transcribe file: asr-interface transcribe <file>")
    console.print("• Show help: asr-interface --help")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main() 