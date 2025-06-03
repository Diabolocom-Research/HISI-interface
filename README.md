You would need to install a bunch of things. And currently only supports Mac

python fastrtc_server.py \            
  --model small \
  --lan en \
  --min-chunk-size 1.0 \
  --host 0.0.0.0 \
  --port 9880 \
 --backend mlx-whisper




# Bringing Your Own ASR Engine (CustomASR)

This document outlines the process of integrating a custom Automatic Speech Recognition (ASR) engine into the existing Gradio real-time transcription application. The application is designed around an `ASRBase` interface and an `OnlineASRProcessor` that handles audio buffering and streaming logic. Your custom ASR will need to implement this interface.

This guide assumes your ASR engine is entirely independent of Whisper, whisper_timestamped, faster-whisper, or mlx-whisper.

## Core Concepts - Assuming the engine is really meant for offline ASR

1.  **`ASRBase` (in `whisper_online.py`):** This is an abstract base class that defines the contract for any ASR engine to be used by the system. You will create a new class, let's call it `MyCustomASR`, that inherits from `ASRBase`.
2.  **`OnlineASRProcessor` (in `whisper_online.py`):** This class consumes an `ASRBase` instance. It manages:
    * Buffering incoming audio chunks.
    * Deciding when to call the ASR's `transcribe` method.
    * Using a `HypothesisBuffer` to manage and commit recognized words based on consistency across transcription calls.
    * Trimming the audio buffer based on committed segments or sentences.
3.  **`asr_factory` (in `whisper_online.py`):** This function is responsible for instantiating the correct ASR engine based on command-line arguments or UI choices. You'll need to modify this to include your `MyCustomASR`.
4.  **Audio Format:** The `OnlineASRProcessor` will feed your ASR engine audio chunks as NumPy arrays of `np.float32` at a sampling rate of `16000 Hz` (defined by `SAMPLING_RATE` in `whisper_online.py`). Your ASR should be prepared to handle this.

## Steps for Integration

1.  **Create Your `MyCustomASR` Class:**
    * In a new Python file (e.g., `my_custom_asr.py`) or directly within `whisper_online.py` (though a separate file is cleaner for modularity).
    * Implement the required methods from `ASRBase`.
2.  **Modify `asr_factory`:**
    * Update `whisper_online.py` to recognize and instantiate `MyCustomASR`.
3.  **(Optional) Add Command-Line Arguments:**
    * Modify `add_shared_args` in `whisper_online.py` if your ASR needs specific CLI configuration (e.g., custom model paths, engine-specific settings).
4.  **(Optional) Update Gradio UI:**
    * Modify `app.py` if you want to select your custom ASR and its configurations through the Gradio interface.
5.  **Test Thoroughly.**


For an indepeth example there are few engines based on whisper which are already implemented.

## 1. Implementing `MyCustomASR`

1. Your `MyCustomASR` class must inherit from `ASRBase` and implement its methods.


Important Notes for MyCustomASR:

    Word Timestamps: Accurate word-level start and end timestamps (in seconds, relative to the start of the audio_np_array passed to transcribe) are essential for the OnlineASRProcessor's HypothesisBuffer to work correctly. This buffer commits words that are consistently recognized across overlapping audio chunks.
    Segment End Timestamps: These are used by OnlineASRProcessor if buffer_trimming is set to "segment". It helps in deciding how much of the processed audio buffer can be safely discarded.
    sep attribute: This character is used by OnlineASRProcessor.to_flush to join the committed words.
        If your ts_words returns words like " hello ", " world ", then sep should be "".
        If ts_words returns "hello", "world", then sep should be " ".
    Language (lan): The __init__ method receives a language code. Your load_model and transcribe should use this if your ASR is language-dependent. If it's multilingual and detects language, self.original_language might be None (if lan="auto").
    Error Handling: Implement robust error handling within your methods.
    Logging: Use the logging module instead of print for better diagnostics. Access self.logfile (passed in __init__) if you need to write to the specific log stream used by OnlineASRProcessor.

2. Modifying asr_factory in whisper_online.py

Open whisper_online.py and find the asr_factory function. You need to add a condition for your custom ASR backend.
Python


Also In whisper_online.py, inside add_shared_args function

    parser.add_argument('--backend', type=str, default="whisper_timestamped", # Or your preferred default
                        choices=["faster-whisper", "whisper_timestamped", "mlx-whisper", "openai-api", "my_custom_asr"], # Add your backend
                        help='ASR backend to use.')

3. (Optional) Add Command-Line Arguments

If MyCustomASR requires specific configurations (e.g., a path to a unique model config file, special thresholds), add them to the add_shared_args function in whisper_online.py.
Python

# In whisper_online.py, inside add_shared_args function

    # ... existing arguments ...
    parser.add_argument('--my-custom-asr-config', type=str, default=None,
                        help='Path to a special config file for MyCustomASR.')
    parser.add_argument('--my-custom-asr-threshold', type=float, default=0.5,
                        help='A custom threshold for MyCustomASR.')

Then, in asr_factory, you can access these:
Python

# In asr_factory, when instantiating MyCustomASR:
        asr = MyCustomASR(
            # ... other args ...
            logfile=logfile,
            custom_config_path=args.my_custom_asr_config,
            custom_threshold=args.my_custom_asr_threshold
        )

# And in MyCustomASR.__init__:
class MyCustomASR(ASRBase):
    def __init__(self, ..., custom_config_path=None, custom_threshold=None, **kwargs):
        super().__init__(..., **kwargs) # Make sure to pass through other standard args
        self.custom_config_path = custom_config_path
        self.custom_threshold = custom_threshold
        # ... rest of your init, using these values

Remember to pass these through ASRBase.__init__ using **kwargs if ASRBase doesn't explicitly define them, or add them to ASRBase.__init__ if they are general enough. Usually, passing via **kwargs to MyCustomASR.__init__ is cleaner.
4. (Optional) Update Gradio UI (app.py)

If you want to select "my_custom_asr" or configure its specific parameters from the Gradio UI in app.py:

    Model/Backend Selection:
    The current UI has a model dropdown (model_dropdown) based on Whisper model sizes. If your "model" concept is different, you might need to:
        Add a new dropdown for "Backend Engine" where "my_custom_asr" is an option.
        If "my_custom_asr" is chosen, perhaps show different model selection options relevant to it. This involves modifying the gr.Dropdown for models and the setup_and_start_asr function to pass the correct backend argument.

    The initial_cli_args.backend will be used by asr_factory if not overridden by a UI choice. If you want the UI to control the backend, you'll need to:
        Add a backend dropdown to the Gradio UI.
        Pass its value to setup_and_start_asr.
        In setup_and_start_asr, set current_processing_args.backend based on this UI choice.

    