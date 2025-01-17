import logging
import logging.handlers
import queue
import time
import os
import groq_v1 as gq
import json
import numpy as np
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


def main(transcriber):
    app_sst(transcriber)


def is_silent(audio_chunk, threshold=0.01):
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

def app_sst(transcriber):
    st.title("ALEXA")
    # path = "./animations/chatbot_animation_3.json"
    # with open(path,"r") as file: 
    #     url = json.load(file) 
    # st_lottie(url, 
    #     reverse=True, 
    #     height=400, 
    #     width=400, 
    #     speed=1, 
    #     loop=True, 
    #     quality='high', 
    #     key='bot'
    # )
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        # rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": False, "audio": True},
    )
    print("Starting the model")
    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return
    status_indicator.write("Loading...")
    text_output = st.empty()
    response = st.empty()
    audio_buffer = np.array([], dtype=np.float32)
    silent = None
    query_complete = False
    while True:
        if query_complete:
            time.sleep(1)
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
                    if silent is not None and (time.time() - silent) > 2:
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
                    if query_complete:
                        print("Query Complete")
                        #call the main LLM
                        llm_response = gq.generate_content(text)
                        # print(llm_response)
                        response.markdown("Response : {}".format(llm_response))
                        query_complete = False
                        audio_buffer = np.array([], dtype=np.float32)

                except ValueError as e:
                    text_output.markdown(f"**Error during transcription:** {e}")
        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break



transcriber = get_model()

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

    main(transcriber)