#!/usr/bin/env python3
from whisper_online import asr_factory, add_shared_args, set_logging, load_audio_chunk
from fastrtc import Stream, StreamHandler, AdditionalOutputs, WebRTC
import argparse
import logging
import numpy as np
import gradio as gr
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--warmup-file", type=str, default=None)
    add_shared_args(parser)
    return parser.parse_args()

class WhisperStreamHandler(StreamHandler):
    def __init__(self, asr, online, min_chunk_sec: float, **kwargs):
        super().__init__(input_sample_rate=online.SAMPLING_RATE, **kwargs)
        self.asr = asr
        self.online = online
        self.full_audio = np.zeros((0,), dtype=np.float32)
        self.accumulated_transcript = ""
        self.segments = []  # will hold (start, end, text) tuples

    def receive(self, frame):
        sr, pcm = frame
        if pcm.dtype == np.int16:
            audio = pcm.astype(np.float32) / 32768.0
        else:
            audio = pcm.astype(np.float32)
        if sr != self.online.SAMPLING_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr,
                                     target_sr=self.online.SAMPLING_RATE)
        audio = audio.flatten()
        self.online.insert_audio_chunk(audio)
        self.full_audio = np.concatenate([self.full_audio, audio])

    def emit(self):
        beg, end, delta = self.online.process_iter()
        if delta:
            # append newly committed word-chunk
            self.segments.append({"start": beg, "end": end, "text": delta})
            if self.accumulated_transcript:
                self.accumulated_transcript += " " + delta
            else:
                self.accumulated_transcript = delta

        # return transcript, full audio, and the entire segment list
        return None, AdditionalOutputs(
            self.accumulated_transcript,
            self.full_audio.copy(),
            list(self.segments)  # copy of list of dicts
        )

    def copy(self):
        logging.warning("Shared ASR state; override copy() for multi‐client support.")
        return WhisperStreamHandler(self.asr, self.online, 0.0)

def main():
    args = get_args()
    set_logging(args, logging.getLogger("whisper_rtc"), other="")
    asr_model, online_proc = asr_factory(args)

    if args.warmup_file:
        try:
            w = load_audio_chunk(args.warmup_file, 0, 1)
            asr_model.transcribe(w)
        except:
            pass

    handler = WhisperStreamHandler(asr_model, online_proc, args.min_chunk_size)

    def combined_outputs_handler(
        transcript,    # str
        audio_array,   # np.ndarray
        segments_list  # list of {"start", "end", "text"}
    ):
        # 1) update transcript
        out_txt = transcript or ""

        # 2) default no‐change for plots and audio
        out_spec, out_wave, out_audio = gr.update(), gr.update(), gr.update()
        # 3) default DF to no‐change
        out_df = gr.update()

        if audio_array is not None and len(audio_array) > 0:
            import librosa, librosa.display, matplotlib.pyplot as plt

            # Mel-Spectrogram
            S = librosa.feature.melspectrogram(
                y=audio_array, sr=handler.online.SAMPLING_RATE, n_mels=128, fmax=8000
            )
            S_db = librosa.power_to_db(S, ref=np.max)
            fig1, ax1 = plt.subplots(figsize=(6,2))
            librosa.display.specshow(S_db, x_axis="time", y_axis="mel",
                                     sr=handler.online.SAMPLING_RATE, ax=ax1)
            ax1.set_title("Mel Spectrogram")
            plt.tight_layout()
            out_spec = fig1

            # Waveform
            fig2, ax2 = plt.subplots(figsize=(6,1.5))
            ax2.plot(audio_array, linewidth=0.5)
            ax2.set_title("Waveform")
            ax2.set_xlim(0, len(audio_array))
            ax2.set_xlabel("Sample #")
            ax2.set_ylabel("Amplitude")
            plt.tight_layout()
            out_wave = fig2

            # Audio for playback/seek
            out_audio = (handler.online.SAMPLING_RATE, audio_array)

            # Build DataFrame of segments
            df = pd.DataFrame(segments_list)
            out_df = df

        return out_txt, out_spec, out_wave, out_audio, out_df

    # real_time_asr_stream = Stream(
    #     handler=handler,
    #     modality="audio",
    #     mode="send-receive",
    #     additional_outputs=[
    #         gr.Textbox(label="Transcript", lines=3, interactive=False),
    #         gr.Plot(label="Mel Spectrogram"),
    #         gr.Plot(label="Waveform"),
    #         gr.Audio(label="Playback", interactive=False),
    #         gr.Dataframe(
    #             headers=["start","end","text"],
    #             label="Timestamped Transcript",
    #             interactive=False,
    #             datatype=["numeric","numeric","str"],
    #         ),
    #     ],
    #     additional_outputs_handler=combined_outputs_handler
    # )


    with gr.Blocks() as demo:
        gr.Markdown("## Live Audio → Whisper Transcription + Mel/Waveform 🎤")

        # Row 1: recorder
        with gr.Row():
            webrtc = WebRTC(
                label="Click & Speak",
                mode="send",
                modality="audio",
            )

        # Row 2: transcript + mel
        with gr.Row():
            transcript = gr.Textbox(
                label="Transcript", lines=3, interactive=False
            )
            mel_plot   = gr.Plot(
                label="Mel Spectrogram"
            )

        # Row 3: waveform + playback + table
        with gr.Row():
            waveform = gr.Plot(label="Waveform")
            playback = gr.Audio(label="Playback", interactive=False)
            table    = gr.Dataframe(
                label="Timestamped Transcript",
                headers=["start","end","text"],
                datatype=["number","number","str"],
                interactive=False
            )

        webrtc.stream(
            handler,
            inputs=[webrtc, transcript, mel_plot, waveform, playback, table],
            outputs=[webrtc]
        )

        def on_additional_outputs(
                transcript_str: str,
                audio_array: np.ndarray,
                segments_list: list[dict]
        ):
            # — Text
            txt_out = transcript_str or ""

            # prepare defaults
            spec_out, wave_out, aud_out, df_out = (
                gr.update(), gr.update(), gr.update(), gr.update()
            )

            if audio_array is not None and len(audio_array) > 0:
                # Mel Spectrogram
                S = librosa.feature.melspectrogram(
                    y=audio_array,
                    sr=handler.online.SAMPLING_RATE,
                    n_mels=128, fmax=8000
                )
                S_db = librosa.power_to_db(S, ref=np.max)
                fig1, ax1 = plt.subplots(figsize=(6, 2))
                librosa.display.specshow(
                    S_db, x_axis="time", y_axis="mel",
                    sr=handler.online.SAMPLING_RATE, ax=ax1
                )
                ax1.set_title("Mel Spectrogram")
                plt.tight_layout()
                spec_out = fig1
                plt.close(fig1)

                # Waveform
                fig2, ax2 = plt.subplots(figsize=(6, 1.5))
                ax2.plot(audio_array, linewidth=0.5)
                ax2.set_title("Waveform")
                ax2.set_xlim(0, len(audio_array))
                plt.tight_layout()
                wave_out = fig2
                plt.close(fig2)

                # Playback
                aud_out = (handler.online.SAMPLING_RATE, audio_array)

                # Timestamp table
                df_out = pd.DataFrame(segments_list)

            return txt_out, spec_out, wave_out, aud_out, df_out

        webrtc.on_additional_outputs(on_additional_outputs,
                                     outputs=[transcript, mel_plot, waveform, playback, table])

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
    )

if __name__ == "__main__":
    main()