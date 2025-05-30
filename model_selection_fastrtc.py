#!/usr/bin/env python3
from whisper_online import (
    asr_factory, add_shared_args, set_logging, load_audio_chunk,
    OnlineASRProcessor, SAMPLING_RATE as WHISPER_SAMPLING_RATE,
    create_tokenizer, WHISPER_LANG_CODES, ASRBase  # Make sure this is importable if used directly
)
# If VACOnlineASRProcessor is used and defined in whisper_online, import it too.
# from whisper_online import VACOnlineASRProcessor

from fastrtc import StreamHandler, AdditionalOutputs, WebRTC
import argparse
import logging
import numpy as np
import gradio as gr
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from gradio_vistimeline import VisTimeline  # VisTimelineData not directly used in this file
import os
import sys  # For sys.stderr

# Ensure matplotlib uses a non-interactive backend for Gradio
import matplotlib

matplotlib.use('Agg')

TIMELINE_ID2 = "dateless_timeline2"
AUDIO_ID = "timeline-audio"

# --- JS and CSS Loading (copied from your original, ensure files exist or handle missing) ---
# Load JS
js_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'custom_time_control.js')
if os.path.exists(js_path):
    with open(js_path, 'r') as f:
        js_content = f.read()
    script = f"""<script>{js_content}</script>"""
else:
    script = ""
    logging.warning("custom_time_control.js not found, timeline audio sync might not work as expected.")

style_timeline_hack = f"""<style>.vis-custom-time.{TIMELINE_ID2} {{pointer-events: none !important;}}</style>"""  # Renamed to avoid conflict
head = script + style_timeline_hack

# Load the CSS
css_path_main = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'style.css')  # Renamed to avoid conflict
if os.path.exists(css_path_main):
    with open(css_path_main, 'r') as f:
        css_content = f.read()
    style_main_app = f"<style>{css_content}</style>"
else:
    style_main_app = ""
    logging.warning("style.css not found, custom styles will not be applied.")


# --- End JS and CSS Loading ---

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument("--warmup-file", type=str, default=None)
    add_shared_args(parser)  # Adds --lan, --model, --min-chunk-size etc.

    parsed_args = parser.parse_args()
    # Ensure defaults for args we rely on if not set by add_shared_args
    if not hasattr(parsed_args, 'min_chunk_size'): parsed_args.min_chunk_size = 1.0
    if not hasattr(parsed_args, 'lan'): parsed_args.lan = "auto"
    if not hasattr(parsed_args, 'task'): parsed_args.task = "transcribe"
    if not hasattr(parsed_args, 'buffer_trimming'): parsed_args.buffer_trimming = "segment"
    if not hasattr(parsed_args, 'buffer_trimming_sec'): parsed_args.buffer_trimming_sec = 10.0
    if not hasattr(parsed_args, 'vac'): parsed_args.vac = False  # Assuming default if not present

    return parsed_args


# --- Global Shared Configuration Store ---
initial_cli_args = get_args()
asr_components_store = {
    "asr_object": None,  # Will store the ASRBase subclass instance (e.g., WhisperTimestampedASR)
    "language": initial_cli_args.lan,
    "task": initial_cli_args.task,
    "min_chunk_sec": initial_cli_args.min_chunk_size,
    "buffer_trimming_config": (initial_cli_args.buffer_trimming, initial_cli_args.buffer_trimming_sec),
    "tokenizer_object": None,  # Stores the created tokenizer if applicable
    "use_vac": initial_cli_args.vac,  # Store whether VAC is requested

    "sample_rate": WHISPER_SAMPLING_RATE,
    "is_ready": False,
    "current_config_id": None  # To track if model/lang/task config changes
}


# --- SmartWhisperHandler using the Shared Store ---
class SmartWhisperHandler(StreamHandler):
    def __init__(self, shared_store, **kwargs):
        rate_to_use = kwargs.pop('input_sample_rate', shared_store["sample_rate"])

        super().__init__(input_sample_rate=rate_to_use, **kwargs)
        self.store = shared_store
        self.online_proc = None
        self.full_audio = np.zeros((0,), dtype=np.float32)
        self.accumulated_transcript = ""
        self.segments = []
        self.last_used_config_id = None
        self.handler_id = id(self)  # For logging

        logging.info(f"SmartWhisperHandler instance [{self.handler_id}] created.")
        self._ensure_online_processor()

    def _ensure_online_processor(self):
        if not self.store["is_ready"] or not self.store["asr_object"]:
            if self.online_proc:
                logging.info(f"Handler [{self.handler_id}] - Store not ready/no asr_object, clearing online_proc.")
                self.online_proc = None
                self.last_used_config_id = None
                self._reset_instance_state()
            return

        if self.online_proc and self.last_used_config_id == self.store["current_config_id"]:
            return

        try:
            logging.info(
                f"Handler [{self.handler_id}] - (Re)initializing OnlineASRProcessor with config ID: {self.store['current_config_id']}")

            # Decide which OnlineASRProcessor class to use (normal or VAC)
            # Assuming VACOnlineASRProcessor is available if self.store["use_vac"] is True
            # and has a compatible signature. For now, this part is illustrative.
            # If VACOnlineASRProcessor is not defined, this will need adjustment or VAC support removed.
            processor_class = OnlineASRProcessor  # Default
            if self.store["use_vac"]:
                # Check if VACOnlineASRProcessor is defined and use it
                if 'VACOnlineASRProcessor' in globals() or 'VACOnlineASRProcessor' in sys.modules.get('whisper_online',
                                                                                                      {}).__dict__:
                    processor_class = getattr(sys.modules.get('whisper_online', {}), 'VACOnlineASRProcessor',
                                              OnlineASRProcessor)
                    logging.info(f"Handler [{self.handler_id}] - Using VACOnlineASRProcessor.")
                else:
                    logging.warning(
                        f"Handler [{self.handler_id}] - VAC requested but VACOnlineASRProcessor not found. Using standard processor.")

            self.online_proc = processor_class(
                asr=self.store["asr_object"],
                tokenizer=self.store["tokenizer_object"],
                buffer_trimming=self.store["buffer_trimming_config"],
                min_chunk_sec=self.store["min_chunk_sec"],
                logfile=sys.stderr  # Or make this configurable
            )
            self.last_used_config_id = self.store["current_config_id"]
            self._reset_instance_state()
            logging.info(
                f"Handler [{self.handler_id}] - OnlineASRProcessor (re)initialized successfully for config '{self.store['current_config_id']}'.")
        except Exception as e:
            logging.error(f"Handler [{self.handler_id}] - Failed to initialize OnlineASRProcessor: {e}", exc_info=True)
            self.online_proc = None
            self.last_used_config_id = None

    def _reset_instance_state(self):
        self.full_audio = np.zeros((0,), dtype=np.float32)
        self.accumulated_transcript = ""
        self.segments = []
        logging.debug(f"Handler [{self.handler_id}] - Instance state reset.")

    def receive(self, frame):
        # self._ensure_online_processor()
        if not self.online_proc:
            return

        sr, pcm = frame
        if pcm.dtype == np.int16:
            audio = pcm.astype(np.float32) / 32768.0
        else:
            audio = pcm.astype(np.float32)

        target_sr = self.store["sample_rate"]
        if sr != target_sr: audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        audio = audio.flatten()

        try:
            self.online_proc.insert_audio_chunk(audio)
            self.full_audio = np.concatenate([self.full_audio, audio])
        except Exception as e:
            logging.error(f"Handler [{self.handler_id}] - Error in insert_audio_chunk: {e}", exc_info=True)

    def emit(self):
        # self._ensure_online_processor()
        if not self.online_proc:
            return None, AdditionalOutputs("", np.array([], dtype=np.float32), [])

        try:
            processed_output = self.online_proc.process_iter()
        except Exception as e:
            logging.error(f"Handler [{self.handler_id}] - Error in process_iter: {e}", exc_info=True)
            processed_output = None

        if processed_output is None:
            return None, AdditionalOutputs(self.accumulated_transcript, self.full_audio.copy(), list(self.segments))

        beg, end, delta = processed_output
        if delta:
            logging.info(f"Handler [{self.handler_id}] - New segment: '{delta}' (raw from process_iter)")
            # Accumulate based on the structure of delta from OnlineASRProcessor.to_flush
            # Assuming delta is the plain text string here.
            self.segments.append({"start": beg, "end": end, "text": delta})  # Store raw segments if needed

            # The OnlineASRProcessor.to_flush returns (b,e,t) where t is already a joined string.
            # So, self.accumulated_transcript should just become this new 't' if it represents the full current hypothesis,
            # or append if 't' is only the newest part. Based on whisper_online, process_iter's 'delta' (which is 't' from to_flush)
            # is the newly committed part.
            if self.accumulated_transcript and delta.strip():  # ensure delta is not just whitespace
                self.accumulated_transcript += self.store.get("asr_object",
                                                              ASRBase).sep + delta.strip()  # Use ASR specific separator
            elif delta.strip():
                self.accumulated_transcript = delta.strip()

        # Rebuild segments for output based on the current accumulated transcript logic might be needed
        # For now, let's assume the segments list is for the raw (beg,end,text) from process_iter.
        # The `accumulated_transcript` is what gets displayed in the main textbox.
        # The `segments_list` for plots/table should reflect meaningful chunks.
        # The current `self.segments` might grow very long.
        # Let's pass the direct output of `process_iter` to `AdditionalOutputs` for segments.
        current_segments_for_df = []
        if beg is not None and end is not None and delta:  # Only add valid new segments
            current_segments_for_df = [{"start": beg, "end": end, "text": delta}]

        return None, AdditionalOutputs(
            self.accumulated_transcript,
            self.full_audio.copy(),
            current_segments_for_df  # Pass only the newest segment(s) from this iteration for table/timeline
        )

    def copy(self):
        logging.info(f"SmartWhisperHandler copy() called for handler id: {self.handler_id}.")
        return SmartWhisperHandler(self.store, input_sample_rate=self.store["sample_rate"])

    def shutdown(self):
        logging.info(f"SmartWhisperHandler shutdown() called for handler id: {self.handler_id}.")
        self.online_proc = None


# --- End SmartWhisperHandler ---


def build_timeline_from_additional_outputs(segments_from_emit, existing_timeline_items):
    # This function now expects segments_from_emit to be the *new* segment(s) from the current emit.
    # It should append to existing_timeline_items if we want a persistent timeline.
    # However, the Gradio VisTimeline with preserve_old_content_on_value_change=True handles merging.
    # So, we can just rebuild the items list each time from a growing list of segments if needed,
    # or simpler, assume segments_from_emit from AdditionalOutputs IS the current list of all segments
    # if the handler's emit() method is changed to return all segments.
    # The current emit() returns only the newest segment in AdditionalOutputs.

    # For VisTimeline's merging, it's better if emit() returns ALL segments so far.
    # Let's adjust emit() slightly:
    # In emit(), change current_segments_for_df to list(self.segments)
    # Then this function becomes:

    # Assuming `segments_from_emit` is the full list of `{"start": b, "end": e, "text": t}` dicts
    # from the handler's `self.segments` list.

    timeline = {
        "groups": [
            {"id": "track-length", "content": ""},
            {"id": 0, "content": ""},
        ],
        "items": [
            {"content": "", "group": "track-length", "selectable": False, "type": "background", "start": 0,
             "end": 60000, "className": "color-primary-600"},
        ]
    }
    for idx, seg in enumerate(segments_from_emit):  # segments_from_emit is now the full list
        timeline["items"].append({
            "id": f"segment_{idx}_{seg['start']}",  # More unique ID
            "content": seg["text"],
            "group": 0,
            "selectable": False,
            "start": int(seg["start"] * 1000),  # Timestamps from OnlineASRProcessor are in seconds
            "end": int(seg["end"] * 1000),
        })
    return timeline


# --- Main Application ---
def main_gradio_app():
    # This function encapsulates the Gradio Blocks UI and logic
    # It uses the global asr_components_store and initial_cli_args

    master_handler_reference = SmartWhisperHandler(asr_components_store)

    def setup_and_start_asr(chosen_model_name: str, chosen_language: str):  # Added lang and task
        global asr_components_store

        logging.info(f"Setting up ASR with model: {chosen_model_name}, lang: {chosen_language}")

        current_processing_args = get_args()  # Get a base set of args
        current_processing_args.model = chosen_model_name
        current_processing_args.lan = chosen_language
        # current_processing_args.task = chosen_task
        # Respect other CLI args like buffer_trimming, min_chunk_size unless also made selectable in UI
        current_processing_args.min_chunk_size = initial_cli_args.min_chunk_size
        current_processing_args.buffer_trimming = initial_cli_args.buffer_trimming
        current_processing_args.buffer_trimming_sec = initial_cli_args.buffer_trimming_sec
        current_processing_args.vac = initial_cli_args.vac  # Respect VAC setting from CLI
        current_processing_args.backend = initial_cli_args.backend  # Ensure backend is consistent

        set_logging(current_processing_args, logging.getLogger(f"whisper_rtc_{chosen_model_name}"), other="")

        asr_components_store["is_ready"] = False
        asr_components_store["asr_object"] = None
        asr_components_store["current_config_id"] = None
        asr_components_store["tokenizer_object"] = None

        try:
            asr_object_instance, _initial_online_proc = asr_factory(current_processing_args, logfile=sys.stderr)
            # We don't use _initial_online_proc; SmartWhisperHandler instances create their own.
        except Exception as e:
            logging.error(
                f"Failed to load ASR model {chosen_model_name} (lang: {chosen_language}, task: {chosen_task}): {e}",
                exc_info=True)
            gr.Warning(f"Failed to load model {chosen_model_name} (lang: {chosen_language}, task: {chosen_task}): {e}")
            return (
                gr.update(visible=True), gr.update(visible=False),
                f"Error loading model: {e}", None, None, None, None,
                build_timeline_from_additional_outputs([], []), 0, gr.update(interactive=True)
            # Pass empty list for timeline
            )

        if current_processing_args.warmup_file and os.path.exists(current_processing_args.warmup_file):
            try:
                logging.info(f"Warming up model {chosen_model_name} with {current_processing_args.warmup_file}...")
                w = load_audio_chunk(current_processing_args.warmup_file, 0, 1)  # Load 1 sec
                asr_object_instance.transcribe(w)  # Transcribe on the base asr object
                logging.info("Warmup complete.")
            except Exception as e:
                logging.warning(f"Warmup failed for {chosen_model_name}: {e}", exc_info=True)
        elif current_processing_args.warmup_file:
            logging.warning(f"Warmup file not found: {current_processing_args.warmup_file}")

        # Create tokenizer (mirroring logic from asr_factory in whisper_online.py)
        created_tokenizer = None
        if current_processing_args.buffer_trimming == "sentence":
            effective_lang_for_tokenizer = asr_object_instance.original_language  # This should be the resolved language
            if not effective_lang_for_tokenizer:  # If 'auto' and not resolved yet, or None
                effective_lang_for_tokenizer = "en"  # Fallback for tokenizer if language still unknown
                logging.warning(
                    f"Language for tokenizer ambiguous (original_language: {asr_object_instance.original_language}, args.lan: {current_processing_args.lan}). Defaulting to '{effective_lang_for_tokenizer}' for tokenizer.")

            if effective_lang_for_tokenizer not in WHISPER_LANG_CODES:  # from whisper_online
                logging.warning(
                    f"Resolved language '{effective_lang_for_tokenizer}' not in WHISPER_LANG_CODES for tokenizer. Setting tokenizer to None.")
            else:
                try:
                    created_tokenizer = create_tokenizer(effective_lang_for_tokenizer)
                    logging.info(f"Tokenizer created for language: {effective_lang_for_tokenizer}")
                except Exception as e:  # Broader catch for create_tokenizer issues
                    logging.error(
                        f"Failed to create tokenizer for lang {effective_lang_for_tokenizer}: {e}. Proceeding without specific sentence tokenizer.",
                        exc_info=True)

        # Update the shared store
        asr_components_store["asr_object"] = asr_object_instance
        asr_components_store["language"] = current_processing_args.lan  # The --lan argument (can be "auto")
        asr_components_store["task"] = current_processing_args.task
        asr_components_store["min_chunk_sec"] = current_processing_args.min_chunk_size
        asr_components_store["buffer_trimming_config"] = (
        current_processing_args.buffer_trimming, current_processing_args.buffer_trimming_sec)
        asr_components_store["tokenizer_object"] = created_tokenizer
        asr_components_store["use_vac"] = current_processing_args.vac

        config_identifier = f"{chosen_model_name}_{current_processing_args.lan}_{current_processing_args.task}"
        asr_components_store["current_config_id"] = config_identifier

        asr_components_store["is_ready"] = True

        logging.info(f"Shared ASR store updated. Config ID: {config_identifier}")

        return (
            gr.update(visible=False), gr.update(visible=True),
            "Speak to see transcript...", None, None, None,
            pd.DataFrame(columns=["start", "end", "text"]),
            build_timeline_from_additional_outputs([], []), 0, gr.update(interactive=False)
        )

    with gr.Blocks(head=head, css=style_main_app, title="Whisper Real-Time Transcription") as demo:
        gr.Markdown("## 🎙️ Whisper Real-Time Transcription with Model Selection")
        audio_duration_state = gr.Number(value=0, visible=False)  # Used by JS for timeline sync

        with gr.Column(elem_id="model_selection_col", visible=True) as model_selection_col:
            gr.Markdown("### 1. Configure ASR Session")

            # Model Selection (from whisper_online add_shared_args)
            model_choices = "tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo".split(
                ",")
            default_model = initial_cli_args.model if initial_cli_args.model in model_choices else "small"
            model_dropdown = gr.Dropdown(
                model_choices, label="Whisper Model", value=default_model,
                info="Choose ASR model. CLI '--model' sets default."
            )

            # Language Selection (from whisper_online WHISPER_LANG_CODES + "auto")
            # WHISPER_LANG_CODES is defined in the provided whisper_online.py
            lang_choices = ["auto"] + WHISPER_LANG_CODES
            default_lang = initial_cli_args.lan if initial_cli_args.lan in lang_choices else "auto"
            language_dropdown = gr.Dropdown(
                lang_choices, label="Language", value=default_lang,
                info="Choose source language or 'auto'. CLI '--lan' sets default."
            )

            # Task Selection
            task_choices = ["transcribe", "translate"]
            default_task = initial_cli_args.task if initial_cli_args.task in task_choices else "transcribe"
            # task_dropdown = gr.Dropdown(
            #
            # task_choices, label="Task", value=default_task,
            #     info="Choose task. CLI '--task' sets default."
            # )

            start_button = gr.Button("🚀 Start Session", variant="primary")

        with gr.Column(elem_id="main_app_col", visible=False) as main_app_col:
            gr.Markdown("### 2. Live Audio Transcription")
            with gr.Row():
                webrtc = WebRTC(label="🎤 Click & Speak", mode="send", modality="audio")
            with gr.Row():
                transcript_text = gr.Textbox(label="📜 Transcript", lines=3, interactive=False, value="Initializing...")
            with gr.Row():
                playback_audio = gr.Audio(label="🗣️ Playback", interactive=False, elem_id=AUDIO_ID)
            with gr.Row():
                dateless_timeline = VisTimeline(
                    # build_timeline_from_additional_outputs will provide the full value dict
                    preserve_old_content_on_value_change=True,  # Important for merging
                    options={
                        "moment": "+00:00", "showCurrentTime": True,
                        "editable": False, "stack": True,
                        "itemsAlwaysDraggable": False, "showMajorLabels": False,
                        "format": {"minorLabels": {"millisecond": "mm:ss.SSS", "second": "mm:ss", "minute": "mm:ss",
                                                   "hour": "HH:mm:ss"}},
                        "rollingMode": {"follow": True, "offset": 0.5},
                        "start": 0, "min": 0, "max": 60000, "zoomMin": 1000, "zoomFriction": 6,
                    },
                    label="🕒 Transcript Timeline", interactive=False, elem_id=TIMELINE_ID2
                )
            with gr.Row():
                mel_plot = gr.Plot(label="📊 Mel Spectrogram")
                waveform_plot = gr.Plot(label="📈 Waveform")
            with gr.Row():
                segments_table = gr.Dataframe(
                    label="📄 Timestamped Segments", headers=["start", "end", "text"],
                    datatype=["number", "number", "str"], interactive=False
                )

            # --- Event Handlers for the main app ---
            webrtc.stream(
                master_handler_reference, inputs=[webrtc], outputs=[webrtc],
            )

            # This will now receive the full list of segments from SmartWhisperHandler's emit
            # if emit() is modified to return list(self.segments) in AdditionalOutputs
            # Let's modify SmartWhisperHandler.emit() for this:
            # Change the last line of SmartWhisperHandler.emit() to:
            # return None, AdditionalOutputs(self.accumulated_transcript, self.full_audio.copy(), list(self.segments))
            # This ensures `segments_list` here is the complete history for building timeline and table.

            def on_additional_outputs_update(
                    current_full_transcript: str,  # Renamed from transcript_str for clarity
                    audio_array: np.ndarray,
                    all_segments_list: list[dict]  # Renamed from segments_list
            ):
                if not asr_components_store["is_ready"]:
                    return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

                txt_out = current_full_transcript or "No speech detected yet..."
                spec_out, wave_out, aud_out, df_out = gr.update(), gr.update(), gr.update(), gr.update()
                timeline_data_out = gr.update()
                current_audio_duration_ms_for_js = gr.update()

                if audio_array is not None and len(audio_array) > 0:
                    current_sr = asr_components_store["sample_rate"]
                    try:  # Mel Spectrogram
                        n_fft = min(2048, len(audio_array))
                        hop_length = n_fft // 4
                        if len(audio_array) >= n_fft:
                            S = librosa.feature.melspectrogram(y=audio_array, sr=current_sr, n_mels=128, fmax=8000,
                                                               n_fft=n_fft, hop_length=hop_length)
                            S_db = librosa.power_to_db(S, ref=np.max)
                            fig1, ax1 = plt.subplots(figsize=(6, 2))
                            librosa.display.specshow(S_db, x_axis="time", y_axis="mel", sr=current_sr, ax=ax1)
                            ax1.set_title("Mel Spectrogram")
                            plt.tight_layout()
                            spec_out = fig1
                            plt.close(fig1)  # Close to free memory
                        else:
                            spec_out = gr.update()
                    except Exception as e:
                        logging.error(f"Error generating spectrogram: {e}", exc_info=True); spec_out = gr.update()

                    try:  # Waveform
                        fig2, ax2 = plt.subplots(figsize=(6, 1.5))
                        librosa.display.waveshow(audio_array, sr=current_sr, ax=ax2, axis='time', linewidth=0.5)
                        ax2.set_title("Waveform")
                        plt.tight_layout()
                        wave_out = fig2
                        plt.close(fig2)  # Close to free memory
                    except Exception as e:
                        logging.error(f"Error generating waveform: {e}", exc_info=True); wave_out = gr.update()

                    audio_int16 = (audio_array * 32767.0).astype(np.int16)
                    aud_out = (current_sr, audio_int16)

                    duration_sec = len(audio_array) / current_sr
                    current_audio_duration_ms_for_js = int(duration_sec * 1000)

                # Timeline data from all_segments_list
                # Ensure build_timeline_from_additional_outputs expects the full list
                timeline_data_out = build_timeline_from_additional_outputs(all_segments_list,
                                                                           [])  # Second arg not used if building fresh

                # Timestamp table from all_segments_list
                if all_segments_list:
                    try:
                        df = pd.DataFrame(all_segments_list)
                        if not df.empty:
                            df_out = df[["start", "end", "text"]]
                        else:
                            df_out = pd.DataFrame(columns=["start", "end", "text"])
                    except Exception as e:
                        logging.error(f"Error creating DataFrame: {e}", exc_info=True)
                else:
                    df_out = pd.DataFrame(columns=["start", "end", "text"])

                return txt_out, spec_out, wave_out, aud_out, df_out, timeline_data_out, current_audio_duration_ms_for_js

            # Ensure SmartWhisperHandler.emit() returns list(self.segments) as the 3rd item in AdditionalOutputs
            # Example modification in SmartWhisperHandler.emit():
            # At the end of emit():
            # return None, AdditionalOutputs(
            #     self.accumulated_transcript,
            #     self.full_audio.copy(),
            #     list(self.segments) # <--- ensure this is the full list
            # )
            webrtc.on_additional_outputs(
                on_additional_outputs_update,
                outputs=[
                    transcript_text, mel_plot, waveform_plot, playback_audio,
                    segments_table, dateless_timeline, audio_duration_state
                ]
            )

        # --- Connect Model Selection to ASR Setup ---
        start_button.click(
            fn=setup_and_start_asr,
            inputs=[model_dropdown, language_dropdown],  # Pass new inputs
            outputs=[
                model_selection_col, main_app_col,
                transcript_text, mel_plot, waveform_plot, playback_audio, segments_table,
                dateless_timeline, audio_duration_state, start_button
            ],
            api_name="start_transcription_session"
        )

        # --- Timeline JS interaction ---
        audio_duration_state.change(
            fn=None, inputs=[audio_duration_state], outputs=None,
            js=f"""(newAudioDurationMs) => {{
                if (newAudioDurationMs && newAudioDurationMs > 0) {{
                    try {{ setConfig("{TIMELINE_ID2}", newAudioDurationMs); }} 
                    catch (e) {{ console.error("Error in setConfig JS for timeline max:", e); }}
                }}
            }}"""
        )
        if os.path.exists(js_path):
            audio_duration_state.change(
                fn=None, inputs=[audio_duration_state], outputs=None,
                js=f'(AudioDurationMs) => {{ initAudioSync("{TIMELINE_ID2}", "{AUDIO_ID}", AudioDurationMs); }}',
            )
    return demo, initial_cli_args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)  # Quieten matplotlib
    logging.getLogger('PIL').setLevel(logging.WARNING)  # Quieten PIL/Pillow

    # Create dummy JS/CSS if not present
    for fname, content in [
        ('custom_time_control.js',
         "// Dummy JS\nfunction setConfig(id,d){console.log('JS:setConfig',id,d)} function initAudioSync(t,a,d){console.log('JS:initAudioSync',t,a,d)}"),
        ('style.css', "/* Dummy CSS */")
    ]:
        if not os.path.exists(fname):
            with open(fname, 'w') as f: f.write(content)
            logging.info(f"Created dummy '{fname}'")

    app_demo, launch_args = main_gradio_app()
    app_demo.launch(
        server_name=launch_args.host, server_port=launch_args.port, share=False,
        # debug=True # Enable for Gradio specific debugging if needed
    )