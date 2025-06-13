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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Your core ASR logic from whisper_online.py
from whisper_online import (
    ASRBase, OnlineASRProcessor, add_shared_args, asr_factory, load_audio_chunk,
    set_logging as set_asr_logging, WHISPER_LANG_CODES, create_tokenizer
)

# fastRTC imports
from fastrtc import Stream, StreamHandler, AdditionalOutputs


from utils.turn_server import get_rtc_credentials

# For this example, we assume `utils` are local helper files
# from utils.turn_server import get_rtc_credentials # Assuming you have this helper

# --- Basic Setup ---
load_dotenv()
# setup_logging() # Assuming you have this helper
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Your ASR State Management (from your Gradio App) ---
# This global store will hold the loaded ASR model and configuration
asr_components_store = {
    "asr_object": None,
    "language": "auto",
    "task": "transcribe",
    "model": None,  # Will be set by the user
    "min_chunk_sec": 1.0,
    "buffer_trimming_config": ("segment", 10.0),
    "tokenizer_object": None,
    "use_vac": False,
    "sample_rate": 16000,
    "is_ready": False,
    "current_config_id": None
}


# --- Your Custom Handler (Integrated Here) ---
class SmartWhisperHandler(StreamHandler):
    # --- PASTE YOUR ENTIRE SmartWhisperHandler CLASS CODE HERE ---
    # The code you provided is perfect. I've included it below for completeness.
    def __init__(self, shared_store, **kwargs):
        rate_to_use = kwargs.pop('input_sample_rate', shared_store["sample_rate"])

        super().__init__(input_sample_rate=rate_to_use, **kwargs)
        self.store = shared_store
        self.online_proc = None
        self.full_audio = np.zeros((0,), dtype=np.float32)
        self.accumulated_transcript = ""
        self.segments = []
        self.last_used_config_id = None
        self.handler_id = id(self)

        logging.info(f"SmartWhisperHandler instance [{self.handler_id}] created.")
        self._ensure_online_processor()

    def _ensure_online_processor(self):
        if not self.store["is_ready"] or not self.store["asr_object"]:
            if self.online_proc:
                self.online_proc = None
            return

        if self.online_proc and self.last_used_config_id == self.store["current_config_id"]:
            return

        try:
            logging.info(
                f"Handler [{self.handler_id}] - (Re)initializing OnlineASRProcessor with config ID: {self.store['current_config_id']}")
            self.online_proc = OnlineASRProcessor(
                asr=self.store["asr_object"],
                tokenizer=self.store["tokenizer_object"],
                buffer_trimming=self.store["buffer_trimming_config"],
                min_chunk_sec=self.store["min_chunk_sec"],
                logfile=sys.stderr
            )
            self.last_used_config_id = self.store["current_config_id"]
            self._reset_instance_state()
            logging.info(
                f"Handler [{self.handler_id}] - OnlineASRProcessor (re)initialized successfully for config '{self.store['current_config_id']}'.")
        except Exception as e:
            logging.error(f"Handler [{self.handler_id}] - Failed to initialize OnlineASRProcessor: {e}", exc_info=True)
            self.online_proc = None

    def _reset_instance_state(self):
        self.full_audio = np.zeros((0,), dtype=np.float32)
        self.accumulated_transcript = ""
        self.segments = []

    def receive(self, frame):
        logger.info(f"Handler [{self.handler_id}] received an audio frame.")
        if not self.online_proc:
            return

        sr, pcm = frame
        audio = pcm.astype(np.float32) / 32768.0 if pcm.dtype == np.int16 else pcm.astype(np.float32)

        target_sr = self.store["sample_rate"]
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        self.online_proc.insert_audio_chunk(audio.flatten())
        # self.full_audio = np.concatenate([self.full_audio, audio.flatten()]) # Optional: for playback

    def emit(self):
        if not self.online_proc:
            return AdditionalOutputs("", np.array([], dtype=np.float32), [])

        processed_output = self.online_proc.process_iter()  # Your logic for real-time processing
        if processed_output is None:
            # If no new segment, just return the current state
            return AdditionalOutputs(self.accumulated_transcript, self.full_audio, self.segments)

        beg, end, delta = processed_output
        if delta:
            self.segments.append({"start": beg, "end": end, "text": delta})
            asr_separator = getattr(self.store.get("asr_object"), "sep", " ")
            if self.accumulated_transcript and delta.strip():
                self.accumulated_transcript += asr_separator + delta.strip()
            elif delta.strip():
                self.accumulated_transcript = delta.strip()

        # We wrap the results in AdditionalOutputs. fastRTC will make this available
        # in the stream.output_stream. The JavaScript client will receive the first argument.
        return AdditionalOutputs(self.accumulated_transcript, self.full_audio, self.segments)

    def copy(self):
        # fastRTC uses this to create a new handler for each connection
        logging.info(f"SmartWhisperHandler copy() called for handler id: {self.handler_id}.")
        return SmartWhisperHandler(self.store, input_sample_rate=self.store["sample_rate"])

    def shutdown(self):
        logging.info(f"SmartWhisperHandler shutdown() called for handler id: {self.handler_id}.")
        self.online_proc = None


# --- FastAPI Application ---
app = FastAPI()



class ModelSelection(BaseModel):
    model: str


@app.post("/load_model")
async def load_model(selection: ModelSelection):
    model_name = selection.model
    if model_name not in ["tiny", "small", "large-v3"]: # Basic validation
         raise HTTPException(status_code=400, detail="Invalid model name specified.")

    config_id = f"{model_name}_{asr_components_store['language']}_{asr_components_store['task']}"

    # Check if the requested model is already loaded and ready
    if asr_components_store["current_config_id"] == config_id and asr_components_store["is_ready"]:
        logger.info(f"Model '{model_name}' is already loaded. Skipping.")
        return {"status": "success", "message": f"Model '{model_name}' is already loaded."}

    logger.info(f"Received request to load model: {model_name}. Unloading any existing model.")
    # Reset state before loading a new model
    asr_components_store["is_ready"] = False
    asr_components_store["asr_object"] = None
    asr_components_store["tokenizer_object"] = None

    try:
        logger.info("Loading new ASR model...")
        asr_components_store["model"] = model_name

        # Create dummy args for asr_factory
        class Args:
            model = model_name
            lan = asr_components_store["language"]
            task = asr_components_store["task"]
            min_chunk_size = asr_components_store["min_chunk_sec"]
            buffer_trimming = asr_components_store["buffer_trimming_config"][0]
            buffer_trimming_sec = asr_components_store["buffer_trimming_config"][1]
            backend = "whisper_timestamped"
            model_cache_dir = None
            model_dir = None
            vac = False
            vad = False

        args = Args()
        asr_object_instance, _ = asr_factory(args)

        asr_components_store["asr_object"] = asr_object_instance
        # asr_components_store["tokenizer_object"] = create_tokenizer(model_name)
        asr_components_store["is_ready"] = True
        asr_components_store["current_config_id"] = config_id
        logger.info(f"ASR model '{model_name}' loaded and ready.")

        return {"status": "success", "message": f"Model '{model_name}' loaded successfully."}

    except Exception as e:
        logger.error(f"Fatal error during ASR model loading for '{model_name}': {e}", exc_info=True)
        # Reset state on failure
        asr_components_store["is_ready"] = False
        asr_components_store["current_config_id"] = None
        asr_components_store["model"] = None
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}. See server logs.")

# --- Main change: Initialize and configure fastRTC.Stream with YOUR handler ---
# Create one "master" handler instance. fastRTC will use its .copy() method for new clients.
master_handler = SmartWhisperHandler(shared_store=asr_components_store)

stream = Stream(
    handler=master_handler,

    mode="send-receive",
    modality="audio"
)

# This automatically creates the /webrtc/offer endpoint for WebRTC signaling
stream.mount(app)


# --- API Endpoints for UI ---
@app.get("/")
async def index():
    # Serve your custom_index.html file
    with open("custom_index.html") as f:
        html_content = f.read()

    # Inject RTC configuration if needed
    rtc_config = get_rtc_credentials(...)
    print(rtc_config)
    logger.info(json.dumps(rtc_config))
    config = {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}

    return HTMLResponse(content=html_content.replace("##RTC_CONFIGURATION##", json.dumps(config)))
    # return HTMLResponse(content=html_content)


# In main.py
@app.get("/transcript")
async def transcript_endpoint(webrtc_id: str):
    logger.debug(f"New transcript stream request for webrtc_id: {webrtc_id}")

    async def output_stream_generator():
        try:
            async for output in stream.output_stream(webrtc_id):
                # NEW WAY: Send a dictionary containing both the full transcript and the segments list

                # output.args[0] is self.accumulated_transcript
                # output.args[2] is self.segments (as defined in your handler's emit)
                payload = {
                    "full_transcript": output.args[0],
                    "segments": output.args[2]
                }

                # Send the entire payload as a single JSON object
                yield f"event: output\ndata: {json.dumps(payload)}\n\n"
        except asyncio.CancelledError:
            logger.info(f"Transcript stream for {webrtc_id} disconnected.")
        except Exception as e:
            logger.error(f"Error in transcript stream for {webrtc_id}: {e}", exc_info=True)

    return StreamingResponse(output_stream_generator(), media_type="text/event-stream")


# Mount static files directory to serve JS and CSS
# app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Application Startup Logic ---
@app.on_event("startup")
async def startup_event():
    # Model loading is now moved to the /load_model endpoint
    logger.info("FastAPI server started. Waiting for model selection from a client.")


if __name__ == "__main__":
    import uvicorn

    # To run this, save it as main.py and run: uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)