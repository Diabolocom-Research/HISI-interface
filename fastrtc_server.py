from whisper_online import asr_factory, add_shared_args, set_logging, load_audio_chunk
from fastrtc import Stream, StreamHandler, AdditionalOutputs
import argparse
import logging
import numpy as np
import gradio as gr


# 1. Parse common Whisper args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host/IP for Gradio UI")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio UI")
    parser.add_argument("--warmup-file", type=str, default=None,
                        help="Path to a WAV to warm up Whisper before serving")
    # shared args include --min-chunk-size, --model, --lan, --backend, --buffer_trimming, --log-level
    add_shared_args(parser)
    return parser.parse_args()


class WhisperStreamHandler(StreamHandler):
    def __init__(self, asr, online, min_chunk_sec: float, **kwargs):
        # input_sample_rate in super().__init__ is the rate FastRTC expects from the client.
        # Your online.SAMPLING_RATE is what your ASR process expects.
        # If they differ, resampling would be needed in `receive`.
        # For now, assuming client sends at online.SAMPLING_RATE (typically 16000 for Whisper)
        super().__init__(input_sample_rate=online.SAMPLING_RATE, **kwargs)
        self.asr = asr  # Store asr if needed, though online object is used more
        self.online = online
        self.min_chunk = min_chunk_sec
        self.pending = np.zeros((0,), dtype=np.float32)
        self.is_first = True  # To handle potential initial short audio chunks
        self.accumulated_transcript = ""

    def receive(self, frame: tuple[int, np.ndarray]):  # frame is (sample_rate, audio_numpy_array)
        # The audio_numpy_array from FastRTC client is often int16.
        # Your whisper_online likely expects float32.
        client_sample_rate, audio_int16 = frame

        # Ensure audio is float32 and normalized if int16
        if audio_int16.dtype == np.int16:
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
        elif audio_int16.dtype == np.float32:
            audio_float32 = audio_int16
        else:
            logging.warning(f"Unexpected audio dtype: {audio_int16.dtype}. Trying to convert.")
            audio_float32 = audio_int16.astype(np.float32)
            if np.max(np.abs(audio_float32)) > 1.0:  # Basic check if it might need normalization
                audio_float32 = audio_float32 / np.max(np.abs(audio_float32))

        # Resample if client_sample_rate differs from what OnlineASRProcessor expects
        if client_sample_rate != self.online.SAMPLING_RATE:
            logging.debug(f"Resampling from {client_sample_rate} Hz to {self.online.SAMPLING_RATE} Hz")
            # Ensure librosa is imported if you use it here, or use another resampler
            import librosa
            audio_float32 = librosa.resample(audio_float32, orig_sr=client_sample_rate,
                                             target_sr=self.online.SAMPLING_RATE)

        audio_float32 = np.asarray(audio_float32).flatten()
        # self.pending = np.concatenate([self.pending, audio_float32])
        self.online.insert_audio_chunk(audio_float32)

    def old_emit(self):
        """
        This is not used anymore. I am keeping this just to make sure everything works
        Note that if you want to use this you would also need to uncomment the self.pending line in receive and change whisper_online file too!
        """

        needed_samples = int(self.min_chunk * self.input_sample_rate)  # self.input_sample_rate is from super()

        if len(self.pending) < needed_samples:
            # logging.debug("Emit: Not enough pending audio. Returning (None, None).")
            return None, AdditionalOutputs(self.accumulated_transcript)  # No audio frame to send back, no additional output update

        self.is_first = False

        # Take exactly one chunk for processing, leave the rest in pending
        chunk_to_process = self.pending[:needed_samples]
        self.pending = self.pending[needed_samples:]

        # Feed Whisper
        self.online.insert_audio_chunk(chunk_to_process)
        # process_iter() in your whisper_online returns (beg_timestamp, end_timestamp, "text")
        _beg, _end, text_segment = self.online.process_iter()


        if text_segment:  # text_segment is a string
            # logging.debug(f"Emit: Text segment found: '{text_segment}'. Returning AdditionalOutputs.")
            self.accumulated_transcript += " " + text_segment
            print("****" + self.accumulated_transcript)
            return None, AdditionalOutputs(self.accumulated_transcript)  # Audio frame is None, AdditionalOutputs has the text

        # logging.debug("Emit: No text segment from process_iter. Returning (None, None).")

    def emit(self) -> tuple[None, AdditionalOutputs | None] | None:

        _beg, _end, text_segment = self.online.process_iter()

        if text_segment:  # text_segment is the newly committed delta
            if self.accumulated_transcript:
                self.accumulated_transcript += " " + text_segment
            else:
                self.accumulated_transcript = text_segment

        return (None, AdditionalOutputs(self.accumulated_transcript))  # No audio frame, no additional output update

    def copy(self):

        # Let's assume you modify asr_factory or have a way to get a fresh online processor.
        # For this fix, we focus on emit. The copy method is a separate complex issue for multi-client.
        logging.warning(
            "WhisperStreamHandler.copy() needs a robust way to create a new 'online' processor for true multi-client support.")
        # This will likely share state for `online` which is bad for multiple users.
        return WhisperStreamHandler(self.asr, self.online, self.min_chunk)


def main():
    args = get_args()
    set_logging(args, logging.getLogger("whisper_rtc"), other="")

    # asr_factory returns (ASRInterface, OnlineASRProcessor)
    # The ASRInterface (e.g., FasterWhisperASR) is the core model.
    # The OnlineASRProcessor manages the streaming logic around it.
    asr_model_instance, online_processor_template = asr_factory(args)

    # Warm-up
    if args.warmup_file:
        try:
            buf = load_audio_chunk(args.warmup_file, 0, 1)  # Using 1 sec for warmup
            asr_model_instance.transcribe(buf)  # Warm up the base model
            logging.info("Whisper warm-up completed.")
        except Exception as e:
            logging.warning(f"Warm-up failed: {e}")


    handler = WhisperStreamHandler(asr_model_instance, online_processor_template, args.min_chunk_size)

    def revised_additional_outputs_handler(gr_textbox_component, transcript_string_value):
        # transcript_string_value is the actual string from AdditionalOutputs(text).args[0]
        # It could also be None if emit returned (None, None) and FastRTC passed None along.
        if transcript_string_value is not None:
            # Ensure it's a string before updating the textbox
            if not isinstance(transcript_string_value, str):
                text_to_display = str(transcript_string_value)
            else:
                text_to_display = transcript_string_value
            return text_to_display  # Update the textbox with this string
        return gr.update()  # No change if transcript_string_value was None


    stream = Stream(
        handler=handler,
        modality="audio",
        mode="send-receive",
        additional_outputs=[gr.Textbox(label="Transcript", lines=10)],
        additional_outputs_handler=revised_additional_outputs_handler
    )
    stream.ui.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    # Ensure whisper_online.py is in the Python path or same directory
    main()
