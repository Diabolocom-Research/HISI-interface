import os
import sys
import logging
import json
import asyncio
import hashlib

import numpy as np
import librosa
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Extra
from typing import Optional

from whisper_online import (
    OnlineASRProcessor, asr_factory
)

from fastrtc import Stream, StreamHandler, AdditionalOutputs

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

asr_components_store = {
    "asr_processor": None, "sample_rate": 16000, "separator": " ",
    "is_ready": False, "current_config_id": None
}

class RealTimeASRHandler(StreamHandler):
    # This class remains exactly the same as the previous version.
    def __init__(self, shared_store, **kwargs):
        rate_to_use = kwargs.pop('input_sample_rate', shared_store.get("sample_rate", 16000))
        super().__init__(input_sample_rate=rate_to_use, **kwargs)
        self.store = shared_store
        self.asr_processor = None; self.full_audio = np.zeros((0,), dtype=np.float32)
        self.accumulated_transcript = ""; self.segments = []
        self.last_used_config_id = None; self.handler_id = id(self)
        logging.info(f"Handler instance [{self.handler_id}] created. Waiting for processor.")
        self._ensure_processor()
    def _ensure_processor(self):
        if self.asr_processor and self.last_used_config_id == self.store["current_config_id"]: return
        if self.store["is_ready"] and self.store["asr_processor"]:
            logging.info(f"Handler [{self.handler_id}] - Acquiring new ASR processor for config '{self.store['current_config_id']}'.")
            self.asr_processor = self.store["asr_processor"]
            self.last_used_config_id = self.store["current_config_id"]
            self._reset_instance_state()
            logging.info(f"Handler [{self.handler_id}] - Processor acquired successfully.")
        else: self.asr_processor = None
    def _reset_instance_state(self):
        self.full_audio = np.zeros((0,), dtype=np.float32); self.accumulated_transcript = ""; self.segments = []
    def receive(self, frame):
        self._ensure_processor()
        if not self.asr_processor: return
        sr, pcm = frame
        audio = pcm.astype(np.float32) / 32768.0 if pcm.dtype == np.int16 else pcm.astype(np.float32)
        target_sr = self.store["sample_rate"]
        if sr != target_sr: audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        self.asr_processor.insert_audio_chunk(audio.flatten())
    def emit(self):
        if not self.asr_processor: return AdditionalOutputs("", np.array([], dtype=np.float32), [])
        processed_output = self.asr_processor.process_iter()
        if processed_output is None: return AdditionalOutputs(self.accumulated_transcript, self.full_audio, self.segments)
        beg, end, delta = processed_output
        if delta:
            self.segments.append({"start": beg, "end": end, "text": delta})
            asr_separator = self.store.get("separator", " ")
            if self.accumulated_transcript and delta.strip(): self.accumulated_transcript += asr_separator + delta.strip()
            elif delta.strip(): self.accumulated_transcript = delta.strip()
        return AdditionalOutputs(self.accumulated_transcript, self.full_audio, self.segments)
    def copy(self):
        logging.info(f"RealTimeASRHandler copy() called for handler id: {self.handler_id}.")
        return RealTimeASRHandler(self.store)
    def shutdown(self):
        logging.info(f"Handler [{self.handler_id}] shutdown."); self.asr_processor = None

app = FastAPI()
master_handler = RealTimeASRHandler(shared_store=asr_components_store)
stream = Stream(handler=master_handler, mode="send-receive", modality="audio")
stream.mount(app)


# --- UPDATED: Pydantic model is now the single source of truth for all defaults ---
class ASRConfig(BaseModel, extra=Extra.allow):
    # Required field - will raise an error if not provided
    model: str

    # Optional fields with defaults. Pydantic fills these in if they're missing from the request.
    lan: str = "auto"
    task: str = "transcribe"
    min_chunk_size: float = 1.0
    backend: str = "whisper_timestamped"
    buffer_trimming: str = "segment"
    buffer_trimming_sec: float = 10.0
    model_cache_dir: Optional[str] = None
    model_dir: Optional[str] = None
    vac: bool = False
    vad: bool = False


# --- UPDATED: `load_model` endpoint is now simpler and more robust ---
@app.post("/load_model")
async def load_model(config: ASRConfig):
    # Pydantic has already validated the request and filled in default values.
    # The `config` object is guaranteed to have `model`, `lan`, `task`, etc.

    config_json_str = json.dumps(config.dict(), sort_keys=True)
    config_id = hashlib.sha256(config_json_str.encode('utf-8')).hexdigest()

    if asr_components_store.get("current_config_id") == config_id and asr_components_store["is_ready"]:
        logger.info(f"Processor for config '{config_id}' is already loaded. Skipping.")
        return {"status": "success", "message": f"Processor is already loaded."}

    logger.info(f"Received request to create processor for new config: {config_json_str}")
    asr_components_store["is_ready"] = False
    asr_components_store["asr_processor"] = None

    try:
        # The `config` object can be passed directly to the factory
        logger.info("Loading ASR model...")
        asr_object_instance, _ = asr_factory(config)
        logger.info("Model loaded.")

        logger.info("Instantiating OnlineASRProcessor...")
        online_processor = OnlineASRProcessor(
            asr=asr_object_instance,
            tokenizer=None, # whisper_timestamped backend handles tokenizer internally
            buffer_trimming=(config.buffer_trimming, config.buffer_trimming_sec),
            min_chunk_sec=config.min_chunk_size,
            logfile=sys.stderr
        )
        logger.info("OnlineASRProcessor instantiated.")

        # Update the shared store
        asr_components_store["asr_processor"] = online_processor
        asr_components_store["separator"] = getattr(asr_object_instance, "sep", " ")
        asr_components_store["is_ready"] = True
        asr_components_store["current_config_id"] = config_id

        logger.info(f"Processor for config ID '{config_id}' is ready.")
        return {"status": "success", "message": f"Processor created successfully."}

    except Exception as e:
        logger.error(f"Fatal error during ASR processor creation: {e}", exc_info=True)
        asr_components_store["is_ready"] = False
        asr_components_store["asr_processor"] = None
        raise HTTPException(status_code=500, detail=f"Failed to create processor. Check server logs.")


# --- Other endpoints remain the same ---

@app.get("/")
async def index():
    with open("custom_index.html") as f: html_content = f.read()
    rtc_config = {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}
    return HTMLResponse(content=html_content.replace("##RTC_CONFIGURATION##", json.dumps(rtc_config)))

@app.get("/transcript")
async def transcript_endpoint(webrtc_id: str):
    logger.debug(f"New transcript stream request for {webrtc_id}")
    async def output_stream_generator():
        try:
            async for output in stream.output_stream(webrtc_id):
                payload = {"full_transcript": output.args[0], "segments": output.args[2]}
                yield f"event: output\ndata: {json.dumps(payload)}\n\n"
        except asyncio.CancelledError: logger.info(f"Transcript stream for {webrtc_id} disconnected.")
        except Exception as e: logger.error(f"Error in transcript stream for {webrtc_id}: {e}", exc_info=True)
    return StreamingResponse(output_stream_generator(), media_type="text/event-stream")

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI server started. Waiting for processor selection from a client.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)