import gc
import json
import os
import tempfile
import uuid

import gradio as gr
import librosa
import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from gradio_vistimeline import VisTimeline
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchMultiTaskAED
from nemo.collections.asr.parts.utils.transcribe_utils import (
    get_buffered_pred_feat_multitaskAED,
)

SPECTOGRAM_ID = "melspec"
TIMELINE_ID = "dateless_timeline"
AUDIO_ID = "timeline-audio"
SAMPLE_RATE = 16000  # Hz
MAX_AUDIO_MINUTES = 10  # wont try to transcribe if longer than this

# Load JS
js_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scripts/custom_time_control.js"
)
with open(js_path) as f:
    js_content = f.read()
script = f"""<script>{js_content}</script>"""
style = f"""<style>.vis-custom-time.{TIMELINE_ID} {{pointer-events: none !important;}}</style>"""
head = script + style

# Load the CSS
css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
with open(css_path) as f:
    css_content = f.read()
style_2 = f"<style>{css_content}</style>"

# Global variables to track model state
current_model = None
frame_asr = None
model_stride_in_secs = None
amp_dtype = torch.float16
audio_duration_ms = 0


def generate_interactive_melspec(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    # Compute mel spectrogram
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        power=2.0,
    )

    melspec = mel_spectrogram(waveform)

    # Convert to dB
    melspec_db = torchaudio.functional.amplitude_to_DB(
        melspec, multiplier=10.0, amin=1e-10, db_multiplier=0.0
    )[
        0
    ]  # take first channel

    # Axes
    time_axis = np.arange(melspec_db.shape[1]) * 512 / sample_rate
    freq_axis = np.arange(melspec_db.shape[0])

    # Create Plotly heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=melspec_db.numpy(),
            x=time_axis,
            y=freq_axis,
            colorscale="Viridis",
            colorbar=dict(title="dB"),
        )
    )

    # Add initial vertical line (time cursor at 0)
    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=0,
        y1=melspec_db.shape[0],
        line=dict(color="red", width=2),
        name="time_cursor",
    )

    fig.update_layout(
        shapes=[
            dict(
                type="line",
                x0=0,
                x1=0,
                y0=0,
                y1=melspec_db.shape[0],
                line=dict(color="red", width=2),
                name="time_cursor",
            )
        ],
        # xaxis_title="Time (s)",
        # yaxis_title="Mel Frequency Bin",
        margin=dict(l=20, r=40, t=30, b=20),
        height=200,
    )

    return fig


def update_table(timeline):
    if hasattr(timeline, "model_dump"):
        data = timeline.model_dump(exclude_none=True)
    else:
        data = timeline

    # print("Timeline data:", data)

    items = data["items"][1::]

    if items is None:
        return []

    table_data = [
        [
            (
                "client" if item.get("group") == 0 else "agent"
            ),  # Map group 0 to "client", others to "agent"
            item.get("content"),
            item.get("start"),
            item.get("end"),
            item.get("end") - item.get("start"),  # Calculate duration
        ]
        for item in items
    ]

    print("Table data:", table_data)
    return table_data


def build_transcript_from_words(word_timestamps, duration):
    print(f"duration: {duration}")

    transcript = {
        "groups": [
            {"id": "track-length", "content": ""},
            {"id": 0, "content": ""},
            # {"id": 1, "content": "A"},
        ],
        "items": [
            {
                "content": "",
                "group": "track-length",
                "selectable": False,
                "type": "background",
                "start": 0,
                "end": int(duration * 1000),
                "className": "color-primary-600",
            },
        ],
    }

    # For the future if we have two speakers
    group_id = 0
    item_id = 0

    for word_data in word_timestamps:
        word = word_data["word"]
        start_time = word_data["start"]
        end_time = word_data["end"]

        transcript["items"].append(
            {
                "id": item_id,
                "content": word,
                "group": group_id,
                "selectable": False,
                "start": int(start_time * 1000),  # Convert to milliseconds
                "end": int(end_time * 1000),  # Convert to milliseconds
            }
        )

        item_id += 1

    return transcript


def load_audio_to_output(audio_file: str | tuple[int, np.ndarray] | None, timeline):
    global audio_duration_ms

    file_path = audio_file
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    audio_duration = librosa.get_duration(y=audio_data, sr=sample_rate)
    audio_duration_ms = audio_duration * 1000

    print(f"audio_duration_ms: {audio_duration_ms}")
    return audio_duration_ms


# Function to load a model
def load_model(model_name):
    global current_model, frame_asr, model_stride_in_secs

    # Clean up previous model if it exists
    if current_model is not None:
        del current_model
        del frame_asr
        gc.collect()
        torch.cuda.empty_cache()

    # Load new model
    current_model = ASRModel.from_pretrained(f"nvidia/{model_name}")
    current_model.eval()

    # make sure beam size always 1 for consistency
    current_model.change_decoding_strategy(None)
    decoding_cfg = current_model.cfg.decoding
    decoding_cfg.beam.beam_size = 1
    current_model.change_decoding_strategy(decoding_cfg)

    # setup for buffered inference
    current_model.cfg.preprocessor.dither = 0.0
    current_model.cfg.preprocessor.pad_to = 0

    feature_stride = current_model.cfg.preprocessor["window_stride"]
    model_stride_in_secs = (
        feature_stride * 8
    )  # 8 = model stride, which is 8 for FastConformer

    frame_asr = FrameBatchMultiTaskAED(
        asr_model=current_model,
        frame_len=40.0,
        total_buffer=40.0,
        batch_size=16,
    )

    return f"Model {model_name} loaded successfully!"


def convert_audio(audio_filepath, tmpdir, utt_id):
    """
    Convert all files to monochannel 16 kHz wav files.
    Do not convert and raise error if audio too long.
    Returns output filename and duration.
    """

    data, sr = librosa.load(audio_filepath, sr=None, mono=True)

    duration = librosa.get_duration(y=data, sr=sr)

    if duration / 60.0 > MAX_AUDIO_MINUTES:
        raise gr.Error(
            f"This demo can transcribe up to {MAX_AUDIO_MINUTES} minutes of audio. "
            "If you wish, you may trim the audio using the Audio viewer in Step 1 "
            "(click on the scissors icon to start trimming audio)."
        )

    if sr != SAMPLE_RATE:
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)

    out_filename = os.path.join(tmpdir, utt_id + ".wav")

    # save output audio
    sf.write(out_filename, data, SAMPLE_RATE)

    return out_filename, duration


def transcribe(audio_filepath, src_lang, tgt_lang, pnc):
    global current_model, frame_asr, model_stride_in_secs

    if current_model is None:
        raise gr.Error("Please select and load a model first")

    if audio_filepath is None:
        raise gr.Error(
            "Please provide some input audio: either upload an audio file or use the microphone"
        )

    utt_id = uuid.uuid4()
    with tempfile.TemporaryDirectory() as tmpdir:
        converted_audio_filepath, duration = convert_audio(
            audio_filepath, tmpdir, str(utt_id)
        )

        # map src_lang and tgt_lang from long versions to short
        LANG_LONG_TO_LANG_SHORT = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
        }
        if src_lang not in LANG_LONG_TO_LANG_SHORT.keys():
            raise ValueError(
                f"src_lang must be one of {LANG_LONG_TO_LANG_SHORT.keys()}"
            )
        else:
            src_lang = LANG_LONG_TO_LANG_SHORT[src_lang]

        if tgt_lang not in LANG_LONG_TO_LANG_SHORT.keys():
            raise ValueError(
                f"tgt_lang must be one of {LANG_LONG_TO_LANG_SHORT.keys()}"
            )
        else:
            tgt_lang = LANG_LONG_TO_LANG_SHORT[tgt_lang]

        # infer taskname from src_lang and tgt_lang
        if src_lang == tgt_lang:
            taskname = "asr"
        else:
            taskname = "s2t_translation"

        # update pnc variable to be "yes" or "no"
        pnc = "yes" if pnc else "no"

        # make manifest file and save
        manifest_data = {
            "audio_filepath": converted_audio_filepath,
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "taskname": taskname,
            "pnc": pnc,
            "answer": "predict",
            "duration": str(duration),
            "timestamp": "yes",
        }

        manifest_filepath = os.path.join(tmpdir, f"{utt_id}.json")

        with open(manifest_filepath, "w") as fout:
            line = json.dumps(manifest_data)
            fout.write(line + "\n")

        # call transcribe, passing in manifest filepath
        if duration < 40:
            hyp = current_model.transcribe(manifest_filepath)[0]
            # Check if result is already a string or a Hypothesis object
            if isinstance(hyp, str):
                output_text = hyp
                print(f"output_text: {output_text}")
            else:
                # It's a Hypothesis object, extract the text field
                output_text = hyp.text if hasattr(hyp, "text") else str(hyp)
                words_timestamps = [item for item in hyp.timestamp["word"]]
                transcript = build_transcript_from_words(words_timestamps, duration)

                print(words_timestamps)
                print(f"transcript: {transcript}")
        else:  # do buffered inference
            with torch.cuda.amp.autocast(
                dtype=amp_dtype
            ):  # TODO: make it work if no cuda
                with torch.no_grad():
                    hyps = get_buffered_pred_feat_multitaskAED(
                        frame_asr,
                        current_model.cfg.preprocessor,
                        model_stride_in_secs,
                        current_model.device,
                        manifest=manifest_filepath,
                        filepaths=None,
                    )

                    print(f"hyps: {hyps}")
                    output_text = (
                        hyps[0].text if hasattr(hyps[0], "text") else str(hyps[0])
                    )
                    words_timestamps = [item for item in hyps.timestamp["word"]]
                    transcript = build_transcript_from_words(words_timestamps, duration)
                    print(words_timestamps)
                    print(f"transcript: {transcript}")

    return output_text, transcript


# add logic to make sure dropdown menus only suggest valid combos
def on_src_or_tgt_lang_change(src_lang_value, tgt_lang_value, pnc_value):
    """Callback function for when src_lang or tgt_lang dropdown menus are changed.
    Args:
        src_lang_value(string), tgt_lang_value (string), pnc_value(bool) - the current
            chosen "values" of each Gradio component
    Returns:
        src_lang, tgt_lang, pnc - these are the new Gradio components that will be displayed

    Note: I found the required logic is easier to understand if you think about the possible src & tgt langs as
    a matrix, e.g. with English, Spanish, French, German as the langs, and only transcription in the same language,
    and X -> English and English -> X translation being allowed, the matrix looks like the diagram below ("Y" means it is
    allowed to go into that state).
    It is easier to understand the code if you think about which state you are in, given the current src_lang_value and
    tgt_lang_value, and then which states you can go to from there.
            tgt lang
            - |EN |ES |FR |DE
            ------------------
            EN| Y | Y | Y | Y
            ------------------
        src     ES| Y | Y |   |
        lang    ------------------
            FR| Y |   | Y |
            ------------------
            DE| Y |   |   | Y
    """

    if src_lang_value == "English" and tgt_lang_value == "English":
        # src_lang and tgt_lang can go anywhere
        src_lang = gr.Dropdown(
            choices=["English", "Spanish", "French", "German"],
            value=src_lang_value,
            label="Input audio is spoken in:",
        )
        tgt_lang = gr.Dropdown(
            choices=["English", "Spanish", "French", "German"],
            value=tgt_lang_value,
            label="Transcribe in language:",
        )
    elif src_lang_value == "English":
        # src is English & tgt is non-English
        # => src can only be English or current tgt_lang_values
        # & tgt can be anything
        src_lang = gr.Dropdown(
            choices=["English", tgt_lang_value],
            value=src_lang_value,
            label="Input audio is spoken in:",
        )
        tgt_lang = gr.Dropdown(
            choices=["English", "Spanish", "French", "German"],
            value=tgt_lang_value,
            label="Transcribe in language:",
        )
    elif tgt_lang_value == "English":
        # src is non-English & tgt is English
        # => src can be anything
        # & tgt can only be English or current src_lang_value
        src_lang = gr.Dropdown(
            choices=["English", "Spanish", "French", "German"],
            value=src_lang_value,
            label="Input audio is spoken in:",
        )
        tgt_lang = gr.Dropdown(
            choices=["English", src_lang_value],
            value=tgt_lang_value,
            label="Transcribe in language:",
        )
    else:
        # both src and tgt are non-English
        # => both src and tgt can only be switch to English or themselves
        src_lang = gr.Dropdown(
            choices=["English", src_lang_value],
            value=src_lang_value,
            label="Input audio is spoken in:",
        )
        tgt_lang = gr.Dropdown(
            choices=["English", tgt_lang_value],
            value=tgt_lang_value,
            label="Transcribe in language:",
        )
    # let pnc be anything if src_lang_value == tgt_lang_value, else fix to True
    if src_lang_value == tgt_lang_value:
        pnc = gr.Checkbox(
            value=pnc_value,
            label="Punctuation & Capitalization in transcript?",
            interactive=True,
        )
    else:
        pnc = gr.Checkbox(
            value=True,
            label="Punctuation & Capitalization in transcript?",
            interactive=False,
        )
    return src_lang, tgt_lang, pnc


with gr.Blocks() as demo1:
    gr.HTML(
        "<h1 style='text-align: center'>Diabolocom ASR interface: Transcribe & Translate audio</h1>"
    )

    audio_duration_box = gr.Number(visible=False)

    # Add model selection dropdown and load button
    with gr.Row(equal_height=True):

        with gr.Column():
            gr.HTML("<p><b>Step 1:</b> Select a model and load it.</p>")

            model_dropdown = gr.Dropdown(
                choices=["canary-180m-flash", "canary-1b-flash", "canary-1b"],
                value="canary-1b",
                label="Select model:",
            )

            load_model_button = gr.Button(
                value="Load selected model",
                variant="primary",
            )

            model_status = gr.Textbox(
                label="Model Status",
                value="No model loaded. Please select and load a model.",
                interactive=False,
            )

        with gr.Column(elem_id="lang-select"):
            gr.HTML(
                "<p><b>Step 2:</b> Choose the input and output language.</p>",
                show_label=False,
            )

            src_lang = gr.Dropdown(
                choices=["English", "Spanish", "French", "German"],
                value="English",
                label="Input audio is spoken in:",
            )
            tgt_lang = gr.Dropdown(
                choices=["English", "Spanish", "French", "German"],
                value="English",
                label="Transcribe in language:",
            )
            pnc = gr.Checkbox(
                value=True, label="Punctuation & Capitalization in transcript?"
            )

    with gr.Row():

        with gr.Column():
            gr.HTML(
                "<p><b>Step 3:</b> Upload an audio file or record with your microphone.</p>"
            )

            audio_file = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record or Upload Audio",
                interactive=True,
                elem_id=AUDIO_ID,
            )

    with gr.Row():
        go_button = gr.Button(
            value="Run model",
            variant="primary",  # make "primary" so it stands out (default is "secondary")
            interactive=False,  # Start with button disabled until model is loaded
        )

    with gr.Row():
        spectrogram_plot = gr.Plot(
            label="Mel Spectrogram", elem_id=SPECTOGRAM_ID, container=True
        )

    with gr.Row():
        dateless_timeline = VisTimeline(
            value={
                "groups": [
                    {"id": "track-length", "content": ""},
                    {"id": 1, "content": ""},
                ],
                "items": [
                    {
                        "content": "",
                        "group": "track-length",
                        "selectable": False,
                        "type": "background",
                        "start": 0,
                        "end": 60000,
                        "className": "color-primary-600",
                    },
                ],
            },
            preserve_old_content_on_value_change=True,  # True: Merges new data with existing content (updates existing items, adds new ones, removes missing ones)
            options={
                "moment": "+00:00",
                "showCurrentTime": True,
                "editable": {
                    "add": False,
                    "remove": False,
                    "updateGroup": True,
                    "updateTime": True,
                },
                "stack": True,  # Stack overlapping items (See transcript in different lines not just in one)
                "itemsAlwaysDraggable": {"item": False, "range": False},
                "showMajorLabels": False,
                "format": {
                    "minorLabels": {
                        "millisecond": "mm:ss.SSS",
                        "second": "mm:ss",
                        "minute": "mm:ss",
                        "hour": "HH:mm:ss",
                    }
                },
                "rollingMode": {"follow": True, "offset": 0.5},
                "start": 0,
                "min": 0,
                "max": 20000,  # (default) Restrict timeline navigation, timeline can not be scrolled further to the right than 20 seconds
                "zoomMin": 1000,
                "zoomFriction": 6,
            },
            label="Audio Transcript",
            interactive=True,
            elem_id=TIMELINE_ID,
        )

    with gr.Row():
        table = gr.DataFrame(
            value=[],
            headers=["Speaker", "Content", "Start Time", "End Time", "Duration"],
            label="Timeline Items",
            interactive=True,
            show_search="filter",
            show_row_numbers=True,
            max_height=200,
        )

    with gr.Row():
        model_output_text_box = gr.Textbox(
            show_copy_button=True,
            lines=2,
            label="Model Text Output",
            elem_id="model_output_text_box",
        )
    with gr.Row():

        gr.HTML(
            "<p style='text-align: center'>"
            "<a href='https://www.diabolocom.com/research/' target='_blank'>Diabolocom Research</a>"
            "</p>"
        )

    # Set timeline duration based on audio duration uploaded
    audio_duration_box.change(
        fn=None,
        inputs=[audio_duration_box],
        outputs=None,
        js=f"""(audioDurationMs) => {{
            console.log('Duration in JS:', audioDurationMs);
            setConfig("{TIMELINE_ID}", audioDurationMs);
        }}""",
    )

    dateless_timeline.change(
        fn=update_table, inputs=[dateless_timeline], outputs=[table]
    )

    dateless_timeline.load(fn=update_table, inputs=[dateless_timeline], outputs=[table])

    audio_file.stop_recording(
        fn=load_audio_to_output,
        inputs=[audio_file, dateless_timeline],
        outputs=[audio_duration_box],
    )

    audio_file.upload(
        fn=load_audio_to_output,
        inputs=[audio_file, dateless_timeline],
        outputs=[audio_duration_box],
    )

    # Load model button functionality
    load_model_button.click(
        fn=lambda: gr.update(interactive=False), inputs=None, outputs=load_model_button
    ).then(fn=load_model, inputs=[model_dropdown], outputs=[model_status]).then(
        fn=lambda: gr.update(interactive=True), inputs=None, outputs=load_model_button
    ).then(
        fn=lambda: gr.update(interactive=True), inputs=None, outputs=go_button
    )

    # Model dropdown change disables run button until model is loaded
    model_dropdown.change(
        fn=lambda: [
            gr.update(interactive=False),
            "No model loaded. Please load the selected model.",
        ],
        inputs=None,
        outputs=[go_button, model_status],
    )

    # Run model button functionality with disable during processing
    go_button.click(
        fn=lambda: gr.update(interactive=False), inputs=None, outputs=go_button
    ).then(
        fn=transcribe,
        inputs=[audio_file, src_lang, tgt_lang, pnc],
        outputs=[model_output_text_box, dateless_timeline],
    ).then(
        fn=lambda: gr.update(interactive=True), inputs=None, outputs=go_button
    ).then(
        fn=generate_interactive_melspec, inputs=[audio_file], outputs=[spectrogram_plot]
    ).then(
        fn=None,
        inputs=[audio_duration_box],
        outputs=None,
        js=f'(AudioDurationMs) => initAudioSync("{TIMELINE_ID}", "{AUDIO_ID}", AudioDurationMs, "{SPECTOGRAM_ID}")',
    )

    # Language dropdown change functionality remains the same
    src_lang.change(
        fn=on_src_or_tgt_lang_change,
        inputs=[src_lang, tgt_lang, pnc],
        outputs=[src_lang, tgt_lang, pnc],
    )

    tgt_lang.change(
        fn=on_src_or_tgt_lang_change,
        inputs=[src_lang, tgt_lang, pnc],
        outputs=[src_lang, tgt_lang, pnc],
    )

with gr.Blocks() as microphone:
    gr.HTML("<h1 style='text-align: center'>Work in Progress</h1>")

with gr.Blocks(
    title="ASR interface",
    theme=gr.themes.Default(
        text_size=gr.themes.sizes.text_lg
    ),  # make text slightly bigger (default is text_md )
    head=head,
    css=style_2,
) as demo:

    gr.TabbedInterface([demo1, microphone], ["Offline ASR", "Real-time ASR"])


demo.queue()
demo.launch()
