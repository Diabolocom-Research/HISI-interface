"""FastAPI web server for the ASR interface."""

import asyncio
import json
import logging
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastrtc import Stream
from jiwer import cer, wer

from ..backends.registry import get_loader
from ..core.config import ASRConfig, TURNConfig
from ..core.protocols import ASRProcessor
from ..core.store import ASRComponentsStore
from ..handlers.stream_handler import RealTimeASRHandler
from ..utils.audio import SAMPLING_RATE, load_audio_from_bytes
from ..utils.turn_server import get_rtc_credentials

logger = logging.getLogger(__name__)


class ASRServer:
    """Main ASR server application."""

    def __init__(self, store: ASRComponentsStore | None = None):
        """
        Initialize the ASR server.

        Args:
            store: Optional shared store for ASR components
        """
        self.store = store or ASRComponentsStore()
        self.app = FastAPI(
            title="ASR Interface",
            description="Real-time Automatic Speech Recognition Interface",
            version="0.1.0",
        )

        # Setup fastRTC stream
        self.master_handler = RealTimeASRHandler(store=self.store)
        self.stream = Stream(
            handler=self.master_handler, mode="send-receive", modality="audio"
        )

        # Default RTC configuration (STUN only)
        self.rtc_config = {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}

        self._setup_routes()
        self._setup_static_files()
        self._setup_events()

    def _get_rtc_configuration(
        self, turn_config: TURNConfig | None = None
    ) -> dict[str, Any]:
        """
        Get RTC configuration with optional TURN server support.

        Args:
            turn_config: Optional TURN server configuration

        Returns:
            Dictionary containing RTC configuration
        """
        if not turn_config or turn_config.provider == "none":
            return self.rtc_config

        try:
            if turn_config.provider == "hf":
                credentials = get_rtc_credentials(
                    provider="hf", token=turn_config.token
                )
            elif turn_config.provider == "twilio":
                credentials = get_rtc_credentials(
                    provider="twilio",
                    account_sid=turn_config.account_sid,
                    auth_token=turn_config.auth_token,
                )
            elif turn_config.provider == "cloudflare":
                credentials = get_rtc_credentials(
                    provider="cloudflare",
                    key_id=turn_config.key_id,
                    api_token=turn_config.api_token,
                    ttl=turn_config.ttl,
                )
            else:
                logger.warning(f"Unknown TURN provider: {turn_config.provider}")
                return self.rtc_config

            # Merge with default STUN configuration
            ice_servers = self.rtc_config["iceServers"] + credentials.get(
                "iceServers", []
            )
            return {"iceServers": ice_servers}

        except Exception as e:
            logger.error(f"Failed to get TURN credentials: {e}")
            return self.rtc_config

    def _setup_static_files(self) -> None:
        """Setup static file serving."""
        self.app.mount("/static", StaticFiles(directory="."), name="static")

    def _setup_routes(self) -> None:
        """Setup API routes."""

        @self.app.post("/load_model")
        async def load_model(config: ASRConfig):
            """Load an ASR model based on the provided configuration."""
            config_id = self.store.get_config_id(config)

            if self.store.is_config_current(config):
                logger.info(
                    f"Processor for config '{config_id}' is already loaded. Skipping."
                )
                return {"status": "success", "message": "Processor is already loaded."}

            logger.info(f"Request to create processor for new config: {config}")

            try:
                loader = get_loader(config.backend)
                online_processor, metadata = loader.load(config)

                self.store.asr_processor = online_processor
                self.store.separator = metadata.get("separator", " ")
                self.store.is_ready = True
                self.store.current_config_id = config_id

                # Update RTC configuration if TURN config is provided
                if config.turn_config:
                    self.rtc_config = self._get_rtc_configuration(config.turn_config)
                    logger.info(
                        f"Updated RTC configuration with TURN server: {config.turn_config.provider}"
                    )

                logger.info(f"Processor for config ID '{config_id}' is ready.")
                return {
                    "status": "success",
                    "message": "Processor created successfully.",
                }

            except Exception as e:
                logger.error(
                    f"Fatal error during ASR processor creation: {e}", exc_info=True
                )
                self.store.reset()
                raise HTTPException(
                    status_code=500,
                    detail="Failed to create processor. Check server logs.",
                )

        @self.app.post("/upload_and_transcribe")
        async def upload_and_transcribe(audio_file: UploadFile = File(...)):
            """Upload and transcribe an audio file."""
            logger.info(
                f"Received file upload for real-time transcription: {audio_file.filename}"
            )

            audio_bytes = await audio_file.read()

            processor_template = self.store.asr_processor
            if not processor_template:
                raise HTTPException(
                    status_code=400,
                    detail="ASR processor not loaded. Please load a model first.",
                )

            # Create a new processor instance for this file upload
            # This prevents interfering with the live ASR state
            upload_processor = self._create_upload_processor(processor_template)

            async def transcription_generator():
                """Generate transcription results."""
                current_full_transcript = []
                try:
                    async for result in self._transcribe_audio_in_chunks(
                        upload_processor, audio_bytes
                    ):
                        data = json.loads(result)
                        if "text" in data:
                            current_full_transcript.append(data["text"])
                            payload = {
                                "full_transcript": " ".join(current_full_transcript),
                                "segments": [
                                    {
                                        "start": data.get("start", 0),
                                        "end": data.get("end", 0),
                                        "text": data["text"],
                                    }
                                ],
                                "timestamp": data.get("end", 0),
                            }
                            yield f"event: output\ndata: {json.dumps(payload)}\n\n"
                except Exception as e:
                    logger.error(
                        f"Error during uploaded file transcription stream: {e}",
                        exc_info=True,
                    )
                    yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
                finally:
                    logger.info(
                        f"Finished processing uploaded file: {audio_file.filename}"
                    )

            return StreamingResponse(
                transcription_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        @self.app.post("/evaluate_model")
        async def evaluate_model(
            reference: str = Form(...), audio: UploadFile = File(...)
        ):
            """Evaluate model performance against a reference transcript."""
            logger.info(f"Received evaluation request for: {audio.filename}")

            audio_bytes = await audio.read()
            processor_template = self.store.asr_processor

            if not processor_template:
                raise HTTPException(
                    status_code=400,
                    detail="ASR processor not loaded. Please load a model first.",
                )

            upload_processor = self._create_upload_processor(processor_template)

            try:
                # Transcribe the audio
                transcript_parts = []
                async for result in self._transcribe_audio_in_chunks(
                    upload_processor, audio_bytes
                ):
                    # Parse the result and extract transcript
                    if isinstance(result, str) and result.startswith("data: "):
                        import json

                        data = json.loads(result[6:])
                        if "text" in data:
                            transcript_parts.append(data["text"])

                full_transcript = " ".join(transcript_parts)

                # Calculate metrics
                word_error_rate = wer(reference, full_transcript)
                char_error_rate = cer(reference, full_transcript)

                return {
                    "reference": reference,
                    "transcript": full_transcript,
                    "word_error_rate": word_error_rate,
                    "char_error_rate": char_error_rate,
                }

            except Exception as e:
                logger.error(f"Error during evaluation: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500, detail=f"Evaluation failed: {str(e)}"
                )

        @self.app.get("/")
        async def index():
            """Serve the main interface."""
            with open("index.html") as f:
                html_content = f.read()

            # Inject current RTC configuration
            html_content = html_content.replace(
                "##RTC_CONFIGURATION##", json.dumps(self.rtc_config)
            )

            return HTMLResponse(content=html_content)

        @self.app.get("/transcript")
        async def transcript_endpoint(webrtc_id: str):
            """Get transcript for a WebRTC session."""
            logger.debug(f"New transcript stream request for {webrtc_id}")

            async def output_stream_generator():
                try:
                    async for output in self.stream.output_stream(webrtc_id):
                        payload = {
                            "full_transcript": output.args[0],
                            "segments": output.args[2],
                        }
                        yield f"event: output\ndata: {json.dumps(payload)}\n\n"
                except asyncio.CancelledError:
                    logger.info(f"Transcript stream for {webrtc_id} disconnected.")
                except Exception as e:
                    logger.error(
                        f"Error in transcript stream for {webrtc_id}: {e}",
                        exc_info=True,
                    )

            return StreamingResponse(
                output_stream_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

    def _create_upload_processor(
        self, processor_template: ASRProcessor
    ) -> ASRProcessor:
        """Create a new processor instance for file uploads."""
        # This is a simplified version - in practice, you'd need to properly
        # clone the processor with its configuration
        return processor_template

    def _setup_events(self) -> None:
        """Setup FastAPI events."""

        @self.app.on_event("startup")
        async def startup_event():
            """Handle startup event."""
            logger.info(
                "FastAPI server started. Waiting for processor selection from a client."
            )

        # Mount the fastRTC stream
        self.stream.mount(self.app)

    async def _transcribe_audio_in_chunks(
        self, processor: ASRProcessor, audio_bytes: bytes
    ):
        """
        Transcribe audio in chunks using the processor and yield flushed segments.

        Args:
            processor: The ASR processor to use
            audio_bytes: The audio data to transcribe

        Yields:
            JSON strings containing transcription results
        """
        audio = load_audio_from_bytes(audio_bytes)

        # Reset processor's internal state for this new file
        processor.init(offset=0)

        chunk_size_seconds = getattr(processor, "min_chunk_sec", 1.0)
        samples_per_chunk = int(chunk_size_seconds * SAMPLING_RATE)

        for i in range(0, len(audio), samples_per_chunk):
            chunk = audio[i : i + samples_per_chunk]

            processor.insert_audio_chunk(chunk)

            processed_output = processor.process_iter()
            if processed_output and processed_output[2]:
                beg, end, text = processed_output
                result = {"start": beg, "end": end, "text": text, "segment": True}
                yield json.dumps(result)

            await asyncio.sleep(0.001)

        # Flush any remaining buffered text at the end
        final_flush_output = processor.finish()
        if final_flush_output and final_flush_output[2]:
            beg, end, text = final_flush_output
            result = {
                "start": beg,
                "end": end,
                "text": text,
                "segment": True,
                "final": True,
            }
            yield json.dumps(result)


def create_app(store: ASRComponentsStore | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        store: Optional shared store for ASR components

    Returns:
        The configured FastAPI application
    """
    server = ASRServer(store)
    return server.app
