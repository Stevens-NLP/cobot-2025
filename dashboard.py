# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
# import numpy as np


# # Set up the Streamlit app layout
# st.title("Capture Audio from Browser")

# # Start the WebRTC stream with only audio mode
# ctx = webrtc_streamer(key="audio-stream")
# print(ctx)

# st.write("Audio is being captured and printed to the console.")
###################

import torch
import whisper
import numpy as np
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# # Load Faster Whisper model (you can use a smaller model like 'base' or 'small' for faster performance)
@st.cache_resource
def load_whisper_model():
    model = whisper.load_model("turbo")  # Use 'small', 'base', or 'large' based on your needs
    return model

# # Load the model once


# model = load_whisper_model()

# # Title of the app
# st.title("Real-Time Audio to Text with Faster Whisper")

# # Instructions
# st.markdown("""
# - Click **Start** to begin capturing audio.
# - The app will transcribe your speech to text in real-time and display it below.
# - The transcribed text will be saved to a text file.
# """)

# class AudioToTextRecorder(AudioProcessorBase):
#     def __init__(self):
#         self.frames = []
#         self.transcribed_text = ""
#         self.start_time = time.time()

#     def recv(self, frame):
#         # print("Here")
#         # print(type(frame))
#         # print(frame)
#         # Convert audio frame to numpy array (16-bit PCM format)
#         # audio_data = np.frombuffer(frame, dtype=np.int16)
#         audio_data = frame.to_ndarray()
#         # print(audio_data)
#         # Accumulate frames for real-time transcription
#         self.frames.append(audio_data)

#         # Limit the frames to a reasonable length (1-2 seconds)
#         if len(self.frames) > 10:  # Adjust this for buffer length
#             self.frames.pop(0)

#         # Convert audio frames to torch tensor for Whisper model
#         audio_tensor = torch.tensor(np.concatenate(self.frames), dtype=torch.float32)
#         # print(audio_tensor)
#         # Process and transcribe audio after 1 second of speech
#         if time.time() - self.start_time > 1:
#             print("Transcribe start")
#             print(audio_tensor.shape)
#             waveform = audio_tensor.unsqueeze(0)  # Add batch dimension
#             print(waveform.shape)
#             # Transcribe the audio using Whisper model
#             # print(waveform)
#             audio_data = whisper.pad_or_trim(waveform.numpy())  # Prepare audio for whisper
#             print("Audio Data Type:", type(audio_data))
#             print("Audio Data Shape:", audio_data.shape)
#             reshaped_audio = audio_data.squeeze()  # Remove batch dimension
            
#             if reshaped_audio.ndim == 2:  # If multi-channel, use the first channel
#                 reshaped_audio = reshaped_audio[0]
#             print("Reshaped Audio Shape:", reshaped_audio.shape)
#             audio_data = reshaped_audio.astype(np.float32)
#             print(audio_data.shape)
#             try:
#                 mel = whisper.log_mel_spectrogram(audio_data, n_mels=model.dims.n_mels).to(model.device)
#                 # print("Mel Spectrogram:", mel)
#             except Exception as e:
#                 print("Error while generating Mel spectrogram:", e)
#             # print(mel)
#             try:
#                 options = whisper.DecodingOptions(fp16=False)
#                 # print(options)
#             except Exception as e:
#                 print(e)
#             try:
#                 result = whisper.decode(model, mel, options)  # Transcribe the audio
#                 # print("result")
#                 # print(result)
#             except Exception as e:
#                 print(e)
#             self.transcribed_text = result.text  # Get the transcribed text
#             self.start_time = time.time()  # Reset the start time

#             # Save transcribed text to file
#             self.save_transcription_to_file(self.transcribed_text)

#         return frame  # Return the frame for WebRTC

#     def save_transcription_to_file(self, text):
#         # Save the transcription into a text file
#         with open("transcription_output.txt", "a") as file:
#             file.write(text + "\n")  # Append text to the file with a newline

# # WebRTC configuration to capture audio
# webrtc_streamer(
#     key="audio-to-text",
#     mode=WebRtcMode.SENDRECV,
#     audio_frame_callback=AudioToTextRecorder,
#     media_stream_constraints={
#         "video": False,  # Disable video
#         "audio": True,   # Enable audio
#     },
#     async_processing=True
# )

# # Display the transcribed text in real-time
# if hasattr(st.session_state, 'transcribed_text'):
#     st.subheader("Real-Time Transcribed Text:")
#     st.write(st.session_state.transcribed_text)
# else:
#     st.write("Start speaking to see the transcribed text appear here.")

# st.info("The transcribed text is being saved to 'transcription_output.txt'.")

import av
import numpy as np
import pydub
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

gain = st.slider("Gain", -10.0, +20.0, 1.0, 0.05)
frames = []
model = load_whisper_model()
def process_audio(frame: av.AudioFrame) -> av.AudioFrame:
    # raw_samples = frame.to_ndarray()
    print("Here")
    audio_data = frame.to_ndarray()
    print("audio data")
    frames.append(audio_data)
    print("frames")
    print(len(frames))
    if len(frames) > 100:
        # frames.pop(0)
        audio_tensor = torch.tensor(np.concatenate(frames), dtype=torch.float32)
        print("audio_tensor")
        waveform = audio_tensor.unsqueeze(0)
        print("waveform")
        audio_data = whisper.pad_or_trim(waveform.numpy())
        print("audio_data")
        reshaped_audio = audio_data.squeeze()
        reshaped_audio = reshaped_audio[0]
        print("reshaped_audio",reshaped_audio.shape)
        try:
            mel = whisper.log_mel_spectrogram(reshaped_audio, n_mels=128).to(model.device)
            # mel = mel.unsqueeze(0)
            print("Mel Spectrogram Shape:", mel.shape)
        except Exception as e:
            print(e)
        options = whisper.DecodingOptions(fp16=False,temperature=0.8)
        print("options")
        try:
            result = whisper.decode(model, mel, options)
            print(result)
        except Exception as e:
            print(e)
            return False
        return frames
    return frame
    # sound = pydub.AudioSegment(
    #     data=raw_samples.tobytes(),
    #     sample_width=frame.format.bytes,
    #     frame_rate=frame.sample_rate,
    #     channels=len(frame.layout.channels),
    # )

    # sound = sound.apply_gain(gain)

    # # Ref: https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentget_array_of_samples  # noqa
    # channel_sounds = sound.split_to_mono()
    # channel_samples = [s.get_array_of_samples() for s in channel_sounds]
    # new_samples: np.ndarray = np.array(channel_samples).T
    # new_samples = new_samples.reshape(raw_samples.shape)

    # new_frame = av.AudioFrame.from_ndarray(new_samples, layout=frame.layout.name)
    # new_frame.sample_rate = frame.sample_rate
    # return new_frame


webrtc_streamer(
    key="audio-filter",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": False,  # Disable video
        "audio": True,   # Enable audio
    },
    audio_frame_callback=process_audio,
    async_processing=True,
)