import asyncio
import hashlib
import io
import json
import logging
import sys
from typing import Any

import librosa
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastrtc import Stream
from jiwer import cer, wer
from pydantic import BaseModel, Extra
from real_time_asr_backend.real_time_asr_protocols import ASRProcessor, ModelLoader
from real_time_asr_backend.real_time_stream_handler import RealTimeASRHandler
from real_time_asr_backend.slimer_whisper_online import (
    SAMPLING_RATE,
    OnlineASRProcessor,
    asr_factory,
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)

asr_components_store = {
    "asr_processor": None,
    "sample_rate": SAMPLING_RATE,
    "separator": " ",
    "is_ready": False,
    "current_config_id": None,
}


# Re-using this utility function for uploaded audio files
def load_audio_from_bytes(
    audio_bytes: bytes, target_sr: int = SAMPLING_RATE
) -> np.ndarray:
    """Loads audio from bytes, ensuring 16kHz mono float32."""
    try:
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio
    except Exception as e:
        logger.error(f"Error loading audio from bytes: {e}", exc_info=True)
        raise ValueError(f"Could not load audio: {e}")


# This function will now be used by both /evaluate_model and /upload_and_transcribe
async def transcribe_audio_in_chunks_and_flush(
    processor: OnlineASRProcessor, audio_bytes: bytes
):
    """
    Transcribes audio in chunks using the OnlineASRProcessor and yields flushed segments.
    Resets the processor at the beginning.
    """
    audio = load_audio_from_bytes(audio_bytes)

    # Reset processor's internal state for this new file
    processor.init(offset=0)

    chunk_size_seconds = processor.min_chunk_sec
    samples_per_chunk = int(chunk_size_seconds * SAMPLING_RATE)

    for i in range(0, len(audio), samples_per_chunk):
        chunk = audio[i : i + samples_per_chunk]

        processor.insert_audio_chunk(chunk)

        processed_output = processor.process_iter()
        if processed_output and processed_output[2]:
            yield processed_output

        await asyncio.sleep(0.001)

    # Flush any remaining buffered text at the end
    final_flush_output = processor.finish()
    if final_flush_output and final_flush_output[2]:
        yield final_flush_output


app = FastAPI()

app.mount("/static", StaticFiles(directory="."), name="static")

master_handler = RealTimeASRHandler(shared_store=asr_components_store)
stream = Stream(handler=master_handler, mode="send-receive", modality="audio")
stream.mount(app)


# --- 2. Create a Concrete Implementation for whisper-online (from custom_main_new.py) ---
class WhisperOnlineLoader(ModelLoader):
    """
    Loads a model using the `whisper_online` library and its `asr_factory`.
    """

    def load(self, config: "ASRConfig") -> tuple[ASRProcessor, dict[str, Any]]:
        """
        Implements the loading logic for whisper_timestamped models.
        """
        logging.info("Using WhisperOnlineLoader...")

        # asr_factory creates the base model (e.g., whispercpp, faster-whisper)
        # Note: asr_factory in custom_main.py returns a tuple (asr_object_instance, online_processor_template)
        # We need to adapt the logic slightly for the custom_main_new.py's asr_factory return
        asr_object_instance, online_processor_template = asr_factory(config)
        logging.info(f"Loaded base ASR model of type: {type(asr_object_instance)}")

        if online_processor_template is None:
            online_processor = OnlineASRProcessor(
                asr=asr_object_instance,
                tokenizer=None,  # In custom_main_new.py, tokenizer is not explicitly passed here, might need to check asr_factory
                buffer_trimming=(
                    config.buffer_trimming,
                    int(config.buffer_trimming_sec),
                ),
                min_chunk_sec=config.min_chunk_size,
                logfile=sys.stderr,
            )
        else:
            online_processor = online_processor_template

        metadata = {"separator": getattr(asr_object_instance, "sep", " ")}
        return online_processor, metadata


# --- 3. Build the Registry (from custom_main_new.py) ---
MODEL_LOADERS: dict[str, ModelLoader] = {"whisper_timestamped": WhisperOnlineLoader()}


class ASRConfig(BaseModel, extra=Extra.allow):
    model: str
    lan: str = "auto"
    task: str = "transcribe"
    min_chunk_size: float = 1.0
    backend: str = "whisper_timestamped"
    buffer_trimming: str = "segment"
    buffer_trimming_sec: float = 10.0
    model_cache_dir: str | None = None
    model_dir: str | None = None
    vac: bool = False
    vad: bool = False


# --- Refactored Endpoint: load_model (from custom_main_new.py) ---
@app.post("/load_model")
async def load_model(config: ASRConfig):
    """
    Loads an ASR model based on the provided configuration.
    """
    config_json_str = json.dumps(config.dict(), sort_keys=True)
    config_id = hashlib.sha256(config_json_str.encode("utf-8")).hexdigest()

    if asr_components_store.get(
        "current_config_id"
    ) == config_id and asr_components_store.get("is_ready"):
        logger.info(f"Processor for config '{config_id}' is already loaded. Skipping.")
        return {"status": "success", "message": "Processor is already loaded."}

    logger.info(f"Request to create processor for new config: {config_json_str}")

    loader = MODEL_LOADERS.get(config.backend)
    if not loader:
        logger.error(f"Unknown backend specified: {config.backend}")
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend: '{config.backend}'. Available backends are: {list(MODEL_LOADERS.keys())}",
        )

    asr_components_store["is_ready"] = False
    asr_components_store["asr_processor"] = None

    try:
        online_processor, metadata = loader.load(config)

        asr_components_store["asr_processor"] = online_processor
        asr_components_store["separator"] = metadata.get("separator", " ")
        asr_components_store["is_ready"] = True
        asr_components_store["current_config_id"] = config_id

        logger.info(f"Processor for config ID '{config_id}' is ready.")
        return {"status": "success", "message": "Processor created successfully."}

    except Exception as e:
        logger.error(f"Fatal error during ASR processor creation: {e}", exc_info=True)
        asr_components_store["is_ready"] = False
        asr_components_store["asr_processor"] = None
        raise HTTPException(
            status_code=500, detail="Failed to create processor. Check server logs."
        )


# --- Existing: Endpoint for Upload and Real-time Transcription ---
@app.post("/upload_and_transcribe")
async def upload_and_transcribe(audio_file: UploadFile = File(...)):
    logger.info(
        f"Received file upload for real-time transcription: {audio_file.filename}"
    )

    audio_bytes = await audio_file.read()

    processor_template = asr_components_store.get("asr_processor")
    if not processor_template:
        raise HTTPException(
            status_code=400,
            detail="ASR processor not loaded. Please load a model first.",
        )

    # Create a new OnlineASRProcessor instance for this file upload
    # This prevents interfering with the live ASR state if a live stream is active
    upload_processor = OnlineASRProcessor(
        asr=processor_template.asr,
        # tokenizer=processor_template.tokenizer,
        buffer_trimming=(
            getattr(processor_template, "buffer_trimming_way", "segment"),
            getattr(processor_template, "buffer_trimming_sec", 10.0),
        ),
        min_chunk_sec=processor_template.min_chunk_sec,
        logfile=sys.stderr,
    )

    async def transcription_generator():
        current_full_transcript = []
        try:
            async for segment_output in transcribe_audio_in_chunks_and_flush(
                upload_processor, audio_bytes
            ):
                current_full_transcript.append(segment_output[2])
                payload = {
                    "full_transcript": " ".join(current_full_transcript),
                    "segments": [
                        {
                            "start": segment_output[0],
                            "end": segment_output[1],
                            "text": segment_output[2],
                        }
                    ],
                    "timestamp": segment_output[1],
                }
                yield f"event: output\ndata: {json.dumps(payload)}\n\n"
        except Exception as e:
            logger.error(
                f"Error during uploaded file transcription stream: {e}", exc_info=True
            )
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            logger.info(f"Finished processing uploaded file: {audio_file.filename}")

    return StreamingResponse(transcription_generator(), media_type="text/event-stream")


@app.post("/evaluate_model")
async def evaluate_model(reference: str = Form(...), audio: UploadFile = File(...)):
    audio_bytes = await audio.read()

    processor_template = asr_components_store.get("asr_processor")
    if not processor_template:
        raise HTTPException(
            status_code=400,
            detail="ASR processor not loaded. Please load a model first.",
        )

    # Create a new OnlineASRProcessor instance specifically for this evaluation
    eval_processor = OnlineASRProcessor(
        asr=processor_template.asr,
        # tokenizer=processor_template.tokenizer,
        buffer_trimming=(
            getattr(processor_template, "buffer_trimming_way", "segment"),
            getattr(processor_template, "buffer_trimming_sec", 10.0),
        ),
        min_chunk_sec=processor_template.min_chunk_sec,
        logfile=sys.stderr,
    )

    try:
        full_segments_generator = transcribe_audio_in_chunks_and_flush(
            eval_processor, audio_bytes
        )
        hypothesis_parts = [segment[2] async for segment in full_segments_generator]
        hypothesis = " ".join(hypothesis_parts).strip()

    except Exception as e:
        logger.error(
            f"Error during full audio transcription for evaluation: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to transcribe audio for evaluation: {e}"
        )

    wer_score = wer(reference, hypothesis)
    cer_score = cer(reference, hypothesis)

    return {"wer": wer_score, "cer": cer_score, "hypothesis": hypothesis}


@app.get("/")
async def index():
    with open("index.html") as f:
        html_content = f.read()
    rtc_config = {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}
    return HTMLResponse(
        content=html_content.replace("##RTC_CONFIGURATION##", json.dumps(rtc_config))
    )


@app.get("/transcript")
async def transcript_endpoint(webrtc_id: str):
    logger.debug(f"New transcript stream request for {webrtc_id}")

    async def output_stream_generator():
        try:
            async for output in stream.output_stream(webrtc_id):
                payload = {
                    "full_transcript": output.args[0],
                    "segments": output.args[2],
                }
                yield f"event: output\ndata: {json.dumps(payload)}\n\n"
        except asyncio.CancelledError:
            logger.info(f"Transcript stream for {webrtc_id} disconnected.")
        except Exception as e:
            logger.error(
                f"Error in transcript stream for {webrtc_id}: {e}", exc_info=True
            )

    return StreamingResponse(output_stream_generator(), media_type="text/event-stream")


@app.on_event("startup")
async def startup_event():
    logger.info(
        "FastAPI server started. Waiting for processor selection from a client."
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
