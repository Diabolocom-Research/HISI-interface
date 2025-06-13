import sys
import logging
import json
import asyncio
import hashlib

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Extra
from typing import Protocol
from whisper_online import (
    OnlineASRProcessor, asr_factory
)

from fastrtc import Stream, StreamHandler, AdditionalOutputs
from real_time_stream_handler import RealTimeASRHandler
from custom_protocol import ASRProcessor, ModelLoader

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

asr_components_store = {
    "asr_processor": None, "sample_rate": 16000, "separator": " ",
    "is_ready": False, "current_config_id": None
}

import logging
from typing import Any, Dict, List, Optional, Tuple



app = FastAPI()
master_handler = RealTimeASRHandler(shared_store=asr_components_store)
stream = Stream(handler=master_handler, mode="send-receive", modality="audio")
stream.mount(app)




# --- 2. Create a Concrete Implementation for whisper-online ---
class WhisperOnlineLoader(ModelLoader):
    """
    Loads a model using the `whisper_online` library and its `asr_factory`.
    """

    def load(self, config: "ASRConfig") -> Tuple[ASRProcessor, Dict[str, Any]]:
        """
        Implements the loading logic for whisper_timestamped models.
        """
        logging.info("Using WhisperOnlineLoader...")

        # asr_factory creates the base model (e.g., whispercpp, faster-whisper)
        asr_object_instance, _ = asr_factory(config)
        logging.info(f"Loaded base ASR model of type: {type(asr_object_instance)}")

        # OnlineASRProcessor wraps the base model with buffering and real-time logic
        online_processor = OnlineASRProcessor(
            asr=asr_object_instance,
            tokenizer=None,  # whisper_timestamped backend handles this internally
            buffer_trimming=(config.buffer_trimming, config.buffer_trimming_sec),
            min_chunk_sec=config.min_chunk_size,
            logfile=sys.stderr
        )

        metadata = {
            "separator": getattr(asr_object_instance, "sep", " ")
        }

        return online_processor, metadata


# --- 3. Build the Registry ---

# This registry maps a 'backend' name from the JSON config to a loader instance.
# To add a new model type, you just add an entry here.
MODEL_LOADERS: Dict[str, ModelLoader] = {
    "whisper_timestamped": WhisperOnlineLoader()
    # "hugging_face_streaming": HuggingFaceLoader(), # Example for the future
}


# --- Pydantic model for configuration remains the same ---
class ASRConfig(BaseModel, extra=Extra.allow):
    model: str
    lan: str = "auto"
    task: str = "transcribe"
    min_chunk_size: float = 1.0
    backend: str = "whisper_timestamped"  # This key is used for the registry lookup
    buffer_trimming: str = "segment"
    buffer_trimming_sec: float = 10.0
    model_cache_dir: Optional[str] = None
    model_dir: Optional[str] = None
    vac: bool = False
    vad: bool = False


# --- 4. Refactored Endpoint: Clean, Extensible, and SOLID ---

@app.post("/load_model")
async def load_model(config: ASRConfig):
    """
    Loads an ASR model based on the provided configuration.

    This endpoint acts as a controller that:
    1. Validates the incoming configuration using Pydantic.
    2. Uses the 'backend' field to find the appropriate model loader from a registry.
    3. Delegates the loading process to the selected loader.
    4. Updates the shared application state with the new processor.
    """
    # Create a unique ID for this specific configuration
    config_json_str = json.dumps(config.dict(), sort_keys=True)
    config_id = hashlib.sha256(config_json_str.encode('utf-8')).hexdigest()

    if asr_components_store.get("current_config_id") == config_id and asr_components_store.get("is_ready"):
        logger.info(f"Processor for config '{config_id}' is already loaded. Skipping.")
        return {"status": "success", "message": "Processor is already loaded."}

    logger.info(f"Request to create processor for new config: {config_json_str}")

    # --- The Core of the New Design ---
    # Look up the loader in our registry using the 'backend' from the JSON config
    loader = MODEL_LOADERS.get(config.backend)
    if not loader:
        logger.error(f"Unknown backend specified: {config.backend}")
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend: '{config.backend}'. Available backends are: {list(MODEL_LOADERS.keys())}"
        )

    # Reset state before loading
    asr_components_store["is_ready"] = False
    asr_components_store["asr_processor"] = None

    try:
        # Delegate the entire loading process to the selected loader
        online_processor, metadata = loader.load(config)

        # Update the shared store with the results from the loader
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