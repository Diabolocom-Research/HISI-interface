import os
import sys
import logging
import json
import asyncio

import numpy as np
import librosa
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

# Your core ASR logic from whisper_online.py
from whisper_online import (
    ASRBase, OnlineASRProcessor, add_shared_args, asr_factory, create_tokenizer
)

# fastRTC imports
from fastrtc import Stream, StreamHandler, AdditionalOutputs

# --- Basic Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Refactored ASR State Management ---
# The store now holds the fully initialized processor.
asr_components_store = {
    "asr_processor": None,  # Will hold the OnlineASRProcessor instance
    "language": "auto",
    "task": "transcribe",
    "model": None,
    "min_chunk_sec": 1.0,
    "buffer_trimming_config": ("segment", 10.0),
    "sample_rate": 16000,
    "separator": " ",  # Default separator, will be updated from the ASR object
    "is_ready": False,
    "current_config_id": None
}


# --- Refactored Generic RealTimeASRHandler ---
class RealTimeASRHandler(StreamHandler):
    """
    A generic handler that processes audio frames using an ASR processor
    provided via a shared store. It is completely decoupled from the specifics
    of how the ASR processor is created.
    """

    def __init__(self, shared_store, **kwargs):
        # We can default to a common sample rate. The `receive` method will resample if needed.
        rate_to_use = kwargs.pop('input_sample_rate', shared_store.get("sample_rate", 16000))
        super().__init__(input_sample_rate=rate_to_use, **kwargs)

        self.store = shared_store
        self.asr_processor = None  # The handler's local instance of the processor
        self.full_audio = np.zeros((0,), dtype=np.float32)
        self.accumulated_transcript = ""
        self.segments = []
        self.last_used_config_id = None
        self.handler_id = id(self)

        logging.info(f"Handler instance [{self.handler_id}] created. Waiting for processor.")
        self._ensure_processor()

    def _ensure_processor(self):
        """Pulls the ready-to-use processor from the shared store."""
        # Check if the store is ready and if we already have the correct processor
        if self.asr_processor and self.last_used_config_id == self.store["current_config_id"]:
            return

        if self.store["is_ready"] and self.store["asr_processor"]:
            logging.info(
                f"Handler [{self.handler_id}] - Acquiring new ASR processor for config '{self.store['current_config_id']}'.")
            self.asr_processor = self.store["asr_processor"]
            self.last_used_config_id = self.store["current_config_id"]
            self._reset_instance_state()
            logging.info(f"Handler [{self.handler_id}] - Processor acquired successfully.")
        else:
            # If not ready, ensure we don't hold onto an old processor
            self.asr_processor = None

    def _reset_instance_state(self):
        self.full_audio = np.zeros((0,), dtype=np.float32)
        self.accumulated_transcript = ""
        self.segments = []

    def receive(self, frame):
        self._ensure_processor()  # Check for processor on first frame
        if not self.asr_processor:
            return

        sr, pcm = frame
        audio = pcm.astype(np.float32) / 32768.0 if pcm.dtype == np.int16 else pcm.astype(np.float32)

        target_sr = self.store["sample_rate"]
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        self.asr_processor.insert_audio_chunk(audio.flatten())

    def emit(self):
        if not self.asr_processor:
            return AdditionalOutputs("", np.array([], dtype=np.float32), [])

        processed_output = self.asr_processor.process_iter()
        if processed_output is None:
            return AdditionalOutputs(self.accumulated_transcript, self.full_audio, self.segments)

        beg, end, delta = processed_output
        if delta:
            self.segments.append({"start": beg, "end": end, "text": delta})
            # The separator is now also pulled from the generic store
            asr_separator = self.store.get("separator", " ")
            if self.accumulated_transcript and delta.strip():
                self.accumulated_transcript += asr_separator + delta.strip()
            elif delta.strip():
                self.accumulated_transcript = delta.strip()

        return AdditionalOutputs(self.accumulated_transcript, self.full_audio, self.segments)

    def copy(self):
        logging.info(f"RealTimeASRHandler copy() called for handler id: {self.handler_id}.")
        return RealTimeASRHandler(self.store)

    def shutdown(self):
        logging.info(f"Handler [{self.handler_id}] shutdown.")
        self.asr_processor = None


# --- FastAPI Application ---
app = FastAPI()

# --- Main change: Initialize and configure fastRTC.Stream with the new handler ---
master_handler = RealTimeASRHandler(shared_store=asr_components_store)
stream = Stream(handler=master_handler, mode="send-receive", modality="audio")
stream.mount(app)


class ModelSelection(BaseModel):
    model: str


# --- Refactored `load_model` now acts as a Factory ---
@app.post("/load_model")
async def load_model(selection: ModelSelection):
    model_name = selection.model
    if model_name not in ["tiny", "small", "large-v3"]:
        raise HTTPException(status_code=400, detail="Invalid model name specified.")

    config_id = f"{model_name}_{asr_components_store['language']}_{asr_components_store['task']}"

    if asr_components_store["current_config_id"] == config_id and asr_components_store["is_ready"]:
        logger.info(f"Processor for '{model_name}' is already loaded. Skipping.")
        return {"status": "success", "message": f"Processor for '{model_name}' is already loaded."}

    logger.info(f"Received request to create processor for model: {model_name}.")
    # Reset state before loading
    asr_components_store["is_ready"] = False
    asr_components_store["asr_processor"] = None

    try:
        # Step 1: Load the base ASR model object and tokenizer
        logger.info("Loading ASR model and tokenizer...")

        class Args:
            model = model_name;
            lan = asr_components_store["language"];
            task = asr_components_store["task"]
            min_chunk_size = asr_components_store["min_chunk_sec"];
            buffer_trimming = asr_components_store["buffer_trimming_config"][0]
            buffer_trimming_sec = asr_components_store["buffer_trimming_config"][1];
            backend = "whisper_timestamped"
            model_cache_dir = None;
            model_dir = None;
            vac = False;
            vad = False

        args = Args()
        asr_object_instance, _ = asr_factory(args)
        logger.info("Model and tokenizer loaded.")

        # Step 2: Create the online processor instance
        logger.info("Instantiating OnlineASRProcessor...")
        online_processor = OnlineASRProcessor(
            asr=asr_object_instance,
            tokenizer=None,
            buffer_trimming=asr_components_store["buffer_trimming_config"],
            min_chunk_sec=asr_components_store["min_chunk_sec"],
            logfile=sys.stderr
        )
        logger.info("OnlineASRProcessor instantiated successfully.")

        # Step 3: Update the shared store with the ready-to-use processor
        asr_components_store["model"] = model_name
        asr_components_store["asr_processor"] = online_processor
        asr_components_store["separator"] = getattr(asr_object_instance, "sep", " ")
        asr_components_store["is_ready"] = True
        asr_components_store["current_config_id"] = config_id

        logger.info(f"Processor for ASR model '{model_name}' is ready.")
        return {"status": "success", "message": f"Processor for '{model_name}' created successfully."}

    except Exception as e:
        logger.error(f"Fatal error during ASR processor creation for '{model_name}': {e}", exc_info=True)
        asr_components_store["is_ready"] = False
        asr_components_store["asr_processor"] = None
        raise HTTPException(status_code=500, detail=f"Failed to create processor for model {model_name}.")


# --- UI and Other Endpoints (No changes needed) ---

@app.get("/")
async def index():
    with open("custom_index.html") as f: html_content = f.read()
    rtc_config = {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}
    return HTMLResponse(content=html_content.replace("##RTC_CONFIGURATION##", json.dumps(rtc_config)))


@app.get("/transcript")
async def transcript_endpoint(webrtc_id: str):
    logger.debug(f"New transcript stream request for webrtc_id: {webrtc_id}")

    async def output_stream_generator():
        try:
            async for output in stream.output_stream(webrtc_id):
                payload = {"full_transcript": output.args[0], "segments": output.args[2]}
                yield f"event: output\ndata: {json.dumps(payload)}\n\n"
        except asyncio.CancelledError:
            logger.info(f"Transcript stream for {webrtc_id} disconnected.")
        except Exception as e:
            logger.error(f"Error in transcript stream for {webrtc_id}: {e}", exc_info=True)

    return StreamingResponse(output_stream_generator(), media_type="text/event-stream")


@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI server started. Waiting for processor selection from a client.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)