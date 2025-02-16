import logging
import logging.handlers
import queue
import time
import os
import base64
from TTS.api import TTS
import groq_v1 as gq
import json
import numpy as np
import wave
import pydub
import streamlit as st
from transformers import pipeline
from streamlit_lottie import st_lottie 

from streamlit_webrtc import WebRtcMode, webrtc_streamer

logger = logging.getLogger(__name__)


@st.cache_data  
def get_model():
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo")
    return transcriber


def main(transcriber,tts):
    app_sst(transcriber,tts)


def is_silent(audio_chunk, threshold=0.04):
    """
    Determine if the audio chunk is silent.
    Args:
        audio_chunk (np.ndarray): The audio data as a NumPy array.
        threshold (float): Energy threshold for silence detection.
    Returns:
        bool: True if the audio is silent, False otherwise.
    """
    energy = np.sqrt(np.mean(audio_chunk ** 2))  # RMS energy
    return energy < threshold


def app_sst(transcriber,tts):
    conv = []
    st.title("ALEXA")

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=4096,
        # rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": False, "audio": True},
    )
    print("Starting the model")
    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return
    # status_indicator.write("Loading...")
    text_output = st.empty()
    response = st.empty()
    audio_buffer = np.array([], dtype=np.float32)
    silent = None
    query_complete = False
    last_time = time.time()
    speaking_duration = 0
    while True:
        if query_complete or (time.time() - last_time) < speaking_duration:
            audio_buffer = np.array([], dtype=np.float32)
            time.sleep(0.3)
            continue
        if webrtc_ctx.audio_receiver:
            sound_chunk = pydub.AudioSegment.empty()
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            # status_indicator.write("Running. Say something!")
            # print(len(audio_frames))
            
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound
            # print(sound_chunk)
            if len(sound_chunk) > 0:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(16000)  # Whisper expects 16kHz mono audio
                new_samples  = np.array(sound_chunk.get_array_of_samples(), dtype=np.float32) / 32768.0  # Normalize to [-1, 1]
                
                if is_silent(new_samples):
                    # print("Silent",time.time() - silent)
                    if silent is not None and (time.time() - silent) > 1:
                        print("Stop recording here")
                        silent = None
                        query_complete = True
                    else:
                        continue
                else:
                    silent = time.time()
                    # print("**Status:** Silent frame detected.")
                audio_buffer = np.concatenate([audio_buffer, new_samples])
                # Transcribe using Whisper
                max_length = 16000 * 30
                if len(audio_buffer) > max_length:
                    audio_buffer = audio_buffer[-max_length:]
                # result = transcriber({"sampling_rate": 16000, "raw": audio_buffer})
                # text = result["text"]
                try:
                    result = transcriber({"sampling_rate": 16000, "raw": audio_buffer})
                    text = result["text"]
                    # print(text)
                    text_output.markdown(f"**Text:** {text}")
                    print(speaking_duration,time.time() - last_time)
                    if query_complete:
                        print("Query Complete")
                        #call the main LLM
                        if len(conv) > 10:
                            conv.pop(0)
                            conv.pop(0)
                        conv.append({"role": "user","content": text})
                        llm_response = gq.generate_content(conv)
                        # print(llm_response)
                        conv.append({"role": "assistant","content": llm_response})
                        response.markdown("Response : {}".format(llm_response))
                        tts.tts_to_file(text=llm_response, file_path="output_audio.wav")
                        query_complete = False
                        audio_buffer = np.array([], dtype=np.float32)
                        try:
                            with wave.open("./output_audio.wav", "rb") as audio_file:
                                n_frames = audio_file.getnframes()
                                frame_rate = audio_file.getframerate()
                                speaking_duration = n_frames / float(frame_rate)
                                speaking_duration = speaking_duration+3
                            with open("./output_audio.wav", "rb") as audio_file:
                                audio_bytes = audio_file.read()
                                base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
                                # Use custom HTML to autoplay the .wav file
                                
                                audio_html = f"""
                                <audio autoplay>
                                    <source src="data:audio/wav;base64,{base64_audio}" type="audio/wav">
                                    Your browser does not support the audio element.
                                </audio>
                                """
                                last_time = time.time()
                            st.markdown(audio_html, unsafe_allow_html=True)
                        except FileNotFoundError:
                            st.error("Audio file not found. Please check the path.")
                except ValueError as e:
                    text_output.markdown(f"**Error during transcription:** {e}")
        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break


if "transcriber" not in st.session_state:
    transcriber = get_model()
    st.session_state.transcriber = transcriber

if 'tts' not in st.session_state:
    # tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=True)
    tts = TTS(model_name="tts_models/en/jenny/jenny", progress_bar=True, gpu=True)
    st.session_state.tts = tts

if __name__ == "__main__":
    import os
    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main(st.session_state.transcriber,st.session_state.tts)