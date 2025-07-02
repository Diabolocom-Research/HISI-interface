# Realtime Automatic Speech Recognition (ASR)

This section of the readme concerns with the real time ASR and how to bring your own custom engine to the interface.

## Integrating a Custom ASR RealTime Engine

This guide details the process for integrating a custom ASR engine into the real-time transcription server. The application is built with a modular, protocol-based architecture, allowing developers to easily "plug in" different ASR systems while leveraging the existing server infrastructure for WebRTC and client communication.

There are two primary integration paths, depending on your needs and how much of the existing real-time logic you want to use.
Integration Paths
1. Path A: Wrapping a Custom Model for the Existing Real-Time Engine

This is the most common and straightforward approach. It involves using the existing real-time processing logic (OnlineASRProcessor) which handles audio buffering, transcript stabilization, and buffer trimming. You simply need to provide it with your own ASR model.

    Choose this path if: You have a pre-trained ASR model (e.g., from Hugging Face, or your own custom-trained model) and want a quick way to make it work in a real-time stream without building your own streaming logic from scratch.

    Core Task: You will create an adapter class that implements the ASRBase interface, which acts as a bridge between the OnlineASRProcessor and your model's specific API.


For a detailed guide, see: **CustomASRusingOnlineWhisper.md**

2. Path B: Providing a Fully Custom Real-Time Engine

This is the advanced approach for developers who want to replace the entire real-time processing pipeline with their own.

    Choose this path if: You have already built a complete real-time engine that includes custom logic for audio buffering, voice activity detection (VAD), endpointing, and transcript generation.

    Core Task: You will create a class that implements the ASRProcessor interface. This class will completely replace the default OnlineASRProcessor. The server will feed raw audio chunks to your class and expect finalized transcript segments in return.

For a detailed guide, see: **CustomRealtimeASRWithoutOnlineWhisper.md**

Final Step: Registering Your Implementation

Regardless of the path you choose, the final step is to make the server aware of your new component. This is done by writing a small amount of "glue code" to register your implementation in the server's ModelLoader registry. This makes your custom ASR engine selectable via the /load_model API endpoint.