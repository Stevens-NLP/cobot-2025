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
import canvas
from streamlit_webrtc import WebRtcMode, webrtc_streamer

logger = logging.getLogger(__name__)

class AudioStateManager:
    def __init__(self):
        self.is_mic_enabled = True  # Keep mic enabled to detect wake word
        self.is_wake_word_detected = False
        self.last_activity_time = None
        self.silence_timeout = 3
        self.waiting_for_response = False
        self.speech_detected = False
        self.noise_threshold = 0.04
        self.speech_threshold = 0.001
        self.min_speech_duration = 0.2
        self.speech_start_time = None
        self.post_response_listening = False
        self.listening_for_wake_word = True  # New flag for wake word state
    
    def audio_frame_callback(self, frame):
        # Always return frame for wake word detection
        return frame

    def check_silence_timeout(self):
        if self.last_activity_time is None:
            return False
        return (time.time() - self.last_activity_time) > self.silence_timeout

    def reset_to_wake_word_state(self):
        self.is_wake_word_detected = False
        self.waiting_for_response = False
        self.speech_detected = False
        self.last_activity_time = None
        self.speech_start_time = None
        self.post_response_listening = False
        self.listening_for_wake_word = True
        # Keep mic enabled for wake word detection

def is_silent(audio_chunk, threshold=0.04):
    energy = np.sqrt(np.mean(audio_chunk ** 2))
    return energy < threshold

def is_speech(audio_chunk, threshold=0.1):
    """
    More stringent check for actual speech vs noise
    Returns: (is_speech, energy_level)
    """
    energy = np.sqrt(np.mean(audio_chunk ** 2))
    # Check if energy is significantly above noise threshold
    return energy > threshold, energy

def check_wake_word(transcriber, audio_buffer):
    try:
        result = transcriber({"sampling_rate": 16000, "raw": audio_buffer})
        text = result["text"].lower().strip()
        return "hey alexa" in text or "hay alexa" in text or "alexa" in text
    except:
        return False



def actions(text):
    if "message" in text or "messages" in text:
        messages, num_messages = canvas.get_messages()
        return str(messages) + "Total Messages : "+str(num_messages)
    else:
        return ""

@st.cache_data  
def get_model():
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo")
    return transcriber


def play_audio_and_wait(audio_path):
    """Play audio and return only after it completes"""
    try:
        with wave.open(audio_path, "rb") as audio_file:
            n_frames = audio_file.getnframes()
            frame_rate = audio_file.getframerate()
            duration = n_frames / float(frame_rate)
            
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
            audio_html = f"""
            <audio autoplay onended="this.parentElement.setAttribute('data-ended', '1')" >
                <source src="data:audio/wav;base64,{base64_audio}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
            
        time.sleep(duration)
        return True
        
    except FileNotFoundError:
        st.error("Audio file not found. Please check the path.")
        return False


def app_sst(transcriber, tts):
    conv = []
    st.title("ALEXA")

    if 'audio_state' not in st.session_state:
        st.session_state.audio_state = AudioStateManager()

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=4096,
        media_stream_constraints={"video": False, "audio": True},
        audio_frame_callback=st.session_state.audio_state.audio_frame_callback
    )

    if not webrtc_ctx.state.playing:
        return

    status_indicator = st.empty()
    wake_word_status = st.empty()
    text_output = st.empty()
    response = st.empty()
    audio_buffer = np.array([], dtype=np.float32)
    wake_word_buffer = np.array([], dtype=np.float32)
    silent = None
    query_complete = False

    while True:
        if webrtc_ctx.audio_receiver:
            sound_chunk = pydub.AudioSegment.empty()
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(16000)
                new_samples = np.array(sound_chunk.get_array_of_samples(), dtype=np.float32) / 32768.0

                # Wake word detection mode
                if st.session_state.audio_state.listening_for_wake_word:
                    wake_word_buffer = np.concatenate([wake_word_buffer, new_samples])
                    max_wake_buffer = 16000 * 3
                    if len(wake_word_buffer) > max_wake_buffer:
                        wake_word_buffer = wake_word_buffer[-max_wake_buffer:]
                    
                    is_speech_detected, energy = is_speech(new_samples, st.session_state.audio_state.speech_threshold)
                    if is_speech_detected:
                        if check_wake_word(transcriber, wake_word_buffer):
                            st.session_state.audio_state.is_wake_word_detected = True
                            st.session_state.audio_state.listening_for_wake_word = False
                            st.session_state.audio_state.waiting_for_response = True
                            st.session_state.audio_state.last_activity_time = time.time()
                            wake_word_status.success("Wake word detected! Listening...")
                            audio_buffer = np.array([], dtype=np.float32)
                            wake_word_buffer = np.array([], dtype=np.float32)
                            continue
                    wake_word_status.info("Waiting for wake word 'Hey Alexa'...")
                    continue

                # Post-response listening mode
                if st.session_state.audio_state.post_response_listening:
                    is_speech_detected, energy = is_speech(new_samples, st.session_state.audio_state.speech_threshold)
                    if is_speech_detected:
                        # If speech detected, reset timeout and switch to normal conversation mode
                        st.session_state.audio_state.post_response_listening = False
                        st.session_state.audio_state.waiting_for_response = True
                        st.session_state.audio_state.last_activity_time = time.time()
                        wake_word_status.success("New speech detected! Listening...")
                        audio_buffer = np.array([], dtype=np.float32)
                        audio_buffer = np.concatenate([audio_buffer, new_samples])
                    elif st.session_state.audio_state.check_silence_timeout():
                        # If timeout reached, reset to wake word state
                        st.session_state.audio_state.reset_to_wake_word_state()
                        wake_word_status.info("Timeout reached. Waiting for wake word 'Hey Alexa'...")
                        wake_word_buffer = np.array([], dtype=np.float32)
                    continue

                # Normal conversation mode
                if st.session_state.audio_state.waiting_for_response:
                    is_speech_detected, energy = is_speech(new_samples, st.session_state.audio_state.speech_threshold)
                    
                    if is_speech_detected:
                        if not st.session_state.audio_state.speech_detected:
                            st.session_state.audio_state.speech_detected = True
                            st.session_state.audio_state.speech_start_time = time.time()
                            wake_word_status.success("Speech detected! Listening...")
                        
                        st.session_state.audio_state.last_activity_time = time.time()
                        silent = None
                        
                        if time.time() - st.session_state.audio_state.speech_start_time > st.session_state.audio_state.min_speech_duration:
                            audio_buffer = np.concatenate([audio_buffer, new_samples])
                    
                    else:
                        if st.session_state.audio_state.speech_detected:
                            if silent is None:
                                silent = time.time()
                            elif (time.time() - silent) > 1:
                                if len(audio_buffer) > 0:
                                    silent = None
                                    query_complete = True
                                    st.session_state.audio_state.waiting_for_response = False
                                    st.session_state.audio_state.speech_detected = False

                # Process accumulated speech
                if query_complete:
                    try:
                        result = transcriber({"sampling_rate": 16000, "raw": audio_buffer})
                        text = result["text"]
                        
                        if text.strip():
                            text_output.markdown(f"**Text:** {text}")
                            
                            action_response = actions(text)
                            if action_response != "":
                                text = text + " " + action_response

                            if len(conv) > 10:
                                conv.pop(0)
                                conv.pop(0)
                            conv.append({"role": "user", "content": text})
                            llm_response = gq.generate_content(conv)
                            conv.append({"role": "assistant", "content": llm_response})
                            response.markdown("Response : {}".format(llm_response))
                            
                            # Generate and play audio
                            tts.tts_to_file(text=llm_response, file_path="output_audio.wav")
                            audio_played = play_audio_and_wait("output_audio.wav")
                            
                            # Enter post-response listening mode
                            st.session_state.audio_state.post_response_listening = True
                            st.session_state.audio_state.last_activity_time = time.time()
                            wake_word_status.success("Listening for follow-up question...")
                            
                            query_complete = False
                            audio_buffer = np.array([], dtype=np.float32)

                    except ValueError as e:
                        text_output.markdown(f"**Error during transcription:** {e}")
        else:
            status_indicator.write("AudioReceiver is not set. Abort.")
            break



def main(transcriber, tts):
    app_sst(transcriber, tts)

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

    if "transcriber" not in st.session_state:
        transcriber = get_model()
        st.session_state.transcriber = transcriber

    if 'tts' not in st.session_state:
        tts = TTS(model_name="tts_models/en/jenny/jenny", progress_bar=True, gpu=True)
        st.session_state.tts = tts

    main(st.session_state.transcriber, st.session_state.tts)