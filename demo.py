import gradio as gr
import json
import librosa
import os
import soundfile as sf
import tempfile
import uuid
import gc

import torch

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchMultiTaskAED
from nemo.collections.asr.parts.utils.transcribe_utils import get_buffered_pred_feat_multitaskAED

SAMPLE_RATE = 16000 # Hz
MAX_AUDIO_MINUTES = 10 # wont try to transcribe if longer than this

# Global variables to track model state
current_model = None
frame_asr = None
model_stride_in_secs = None
amp_dtype = torch.float16

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
    
    feature_stride = current_model.cfg.preprocessor['window_stride']
    model_stride_in_secs = feature_stride * 8 # 8 = model stride, which is 8 for FastConformer
    
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

    out_filename = os.path.join(tmpdir, utt_id + '.wav')

    # save output audio
    sf.write(out_filename, data, SAMPLE_RATE)

    return out_filename, duration


def transcribe(audio_filepath, src_lang, tgt_lang, pnc):
    global current_model, frame_asr, model_stride_in_secs
    
    if current_model is None:
        raise gr.Error("Please select and load a model first")
        
    if audio_filepath is None:
        raise gr.Error("Please provide some input audio: either upload an audio file or use the microphone")
    
    utt_id = uuid.uuid4()
    with tempfile.TemporaryDirectory() as tmpdir:
        converted_audio_filepath, duration = convert_audio(audio_filepath, tmpdir, str(utt_id))

        # map src_lang and tgt_lang from long versions to short
        LANG_LONG_TO_LANG_SHORT = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
        }
        if src_lang not in LANG_LONG_TO_LANG_SHORT.keys():
            raise ValueError(f"src_lang must be one of {LANG_LONG_TO_LANG_SHORT.keys()}")
        else:
            src_lang = LANG_LONG_TO_LANG_SHORT[src_lang]
        
        if tgt_lang not in LANG_LONG_TO_LANG_SHORT.keys():
            raise ValueError(f"tgt_lang must be one of {LANG_LONG_TO_LANG_SHORT.keys()}")
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
        }

        manifest_filepath = os.path.join(tmpdir, f'{utt_id}.json')

        with open(manifest_filepath, 'w') as fout:
            line = json.dumps(manifest_data)
            fout.write(line + '\n')

        # call transcribe, passing in manifest filepath
        if duration < 40:
            hyp = current_model.transcribe(manifest_filepath)[0]
            # Check if result is already a string or a Hypothesis object
            if isinstance(hyp, str):
                print("a: ", hyp)
                output_text = hyp
            else:
                # It's a Hypothesis object, extract the text field
                print("b: ", hyp)
                output_text = hyp.text if hasattr(hyp, 'text') else str(hyp)
                #words_timestamps = [item for item in hyp.timestamp['word']]
                #print("b: ", words_timestamps)
        else: # do buffered inference
            with torch.cuda.amp.autocast(dtype=amp_dtype): # TODO: make it work if no cuda
                with torch.no_grad():
                    hyps = get_buffered_pred_feat_multitaskAED(
                        frame_asr,
                        current_model.cfg.preprocessor,
                        model_stride_in_secs,
                        current_model.device,
                        manifest=manifest_filepath,
                        filepaths=None,
                    )
                    
                    print("c: ", hyps)
                    output_text = hyps[0].text if hasattr(hyps[0], 'text') else str(hyps[0])
                    #words_timestamps = [item for item in hyps.timestamp['word']]
                    #print(words_timestamps)

    return output_text

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
            label="Input audio is spoken in:"
        )
        tgt_lang = gr.Dropdown(
            choices=["English", "Spanish", "French", "German"],
            value=tgt_lang_value,
            label="Transcribe in language:"
        )
    elif src_lang_value == "English": 
        # src is English & tgt is non-English
        # => src can only be English or current tgt_lang_values
        # & tgt can be anything
        src_lang = gr.Dropdown(
            choices=["English", tgt_lang_value],
            value=src_lang_value,
            label="Input audio is spoken in:"
        )
        tgt_lang = gr.Dropdown(
            choices=["English", "Spanish", "French", "German"],
            value=tgt_lang_value,
            label="Transcribe in language:"
        )
    elif tgt_lang_value == "English": 
        # src is non-English & tgt is English
        # => src can be anything
        # & tgt can only be English or current src_lang_value
        src_lang = gr.Dropdown(
            choices=["English", "Spanish", "French", "German"],
            value=src_lang_value,
            label="Input audio is spoken in:"
        )
        tgt_lang = gr.Dropdown(
            choices=["English", src_lang_value],
            value=tgt_lang_value,
            label="Transcribe in language:"
        )
    else:
        # both src and tgt are non-English
        # => both src and tgt can only be switch to English or themselves
        src_lang = gr.Dropdown(
            choices=["English", src_lang_value],
            value=src_lang_value,
            label="Input audio is spoken in:"
        )
        tgt_lang = gr.Dropdown(
            choices=["English", tgt_lang_value],
            value=tgt_lang_value,
            label="Transcribe in language:"
        )
    # let pnc be anything if src_lang_value == tgt_lang_value, else fix to True
    if src_lang_value == tgt_lang_value:
        pnc = gr.Checkbox(
            value=pnc_value,
            label="Punctuation & Capitalization in transcript?",
            interactive=True
        )
    else:
        pnc = gr.Checkbox(
            value=True,
            label="Punctuation & Capitalization in transcript?",
            interactive=False
        )
    return src_lang, tgt_lang, pnc



with gr.Blocks() as file:
	gr.HTML("<h1 style='text-align: center'>Diabolocom ASR interface: Transcribe & Translate audio</h1>")
	
	# Add model selection dropdown and load button
	with gr.Row():
		with gr.Column():
			gr.HTML("<p><b>Step 1:</b> Select a model and load it.</p>")
			
			model_dropdown = gr.Dropdown(
				choices=["canary-180m-flash", "canary-1b-flash", "canary-1b"],
				value="canary-1b",
				label="Select model:"
			)
			
			load_model_button = gr.Button(
				value="Load selected model",
				variant="primary",
			)
			
			model_status = gr.Textbox(
				label="Model Status",
				value="No model loaded. Please select and load a model.",
				interactive=False
			)

	with gr.Row():
		with gr.Column():
			gr.HTML(
				"<p><b>Step 2:</b> Upload an audio file or record with your microphone.</p>"
			)

			audio_file = gr.Audio(sources=["microphone", "upload"], type="filepath")

			gr.HTML("<p><b>Step 3:</b> Choose the input and output language.</p>")

			src_lang = gr.Dropdown(
				choices=["English", "Spanish", "French", "German"],
				value="English",
				label="Input audio is spoken in:"
			)

			with gr.Column():
				tgt_lang = gr.Dropdown(
					choices=["English", "Spanish", "French", "German"],
					value="English",
					label="Transcribe in language:"
				)
				pnc = gr.Checkbox(
					value=True,
					label="Punctuation & Capitalization in transcript?",
				)

		with gr.Column():

			gr.HTML("<p><b>Step 4:</b> Run the model.</p>")

			go_button = gr.Button(
				value="Run model",
				variant="primary", # make "primary" so it stands out (default is "secondary")
				interactive=False  # Start with button disabled until model is loaded
			)

			model_output_text_box = gr.Textbox(
				label="Model Output",
				elem_id="model_output_text_box",
			)

	with gr.Row():

		gr.HTML(
			"<p style='text-align: center'>"
				"<a href='https://www.diabolocom.com/research/' target='_blank'>Diabolocom Research</a>"
			"</p>"
		)

	# Load model button functionality
	load_model_button.click(
		fn=lambda: gr.update(interactive=False),
		inputs=None,
		outputs=load_model_button
	).then(
		fn=load_model,
		inputs=[model_dropdown],
		outputs=[model_status]
	).then(
		fn=lambda: gr.update(interactive=True),
		inputs=None,
		outputs=load_model_button
	).then(
		fn=lambda: gr.update(interactive=True),
		inputs=None,
		outputs=go_button
	)
	
	# Model dropdown change disables run button until model is loaded
	model_dropdown.change(
		fn=lambda: [gr.update(interactive=False), "No model loaded. Please load the selected model."],
		inputs=None,
		outputs=[go_button, model_status]
	)

	# Run model button functionality with disable during processing
	go_button.click(
		fn=lambda: gr.update(interactive=False),
		inputs=None,
		outputs=go_button
	).then(
		fn=transcribe, 
		inputs=[audio_file, src_lang, tgt_lang, pnc],
		outputs=[model_output_text_box]
	).then(
		fn=lambda: gr.update(interactive=True),
		inputs=None,
		outputs=go_button
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
    css="""
        textarea { font-size: 18px;}
        #model_output_text_box span {
            font-size: 18px;
            font-weight: bold;
        }
    """,
    theme=gr.themes.Default(text_size=gr.themes.sizes.text_lg) # make text slightly bigger (default is text_md )
) as demo:          

	gr.TabbedInterface([file, microphone], ["Offline ASR", "Real-time ASR"])


demo.queue()
demo.launch(share=True)