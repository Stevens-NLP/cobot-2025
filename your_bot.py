import streamlit as st
import time
import numpy as np
import groq_v1 as gq
import stablediffusion as sd
import RAG as rag
from PIL import Image
import requests
from io import BytesIO
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import speech_recognition as sr
from RealtimeSTT import AudioToTextRecorder
import TTS
recorder = AudioToTextRecorder(spinner=False, model="large-v2", language="en", ensure_sentence_ends_with_period=True, 
                                   enable_realtime_transcription=True, silero_deactivity_detection = True)


st.set_page_config(layout="wide")
st.title("COBOT")
st.sidebar.title("Chat History")

# st.sidebar.write("This is the right sidebar.")
# st.header("This is the header")
# st.markdown("This is the markdown")
# st.subheader("This is the subheader")
# st.caption("This is the caption")
# st.code("x = 2021")
# st.latex(r''' a+a r^1+a r^2+a r^3 ''')


# st.image("kid.jpg", caption="A kid playing")
# st.audio("audio.mp3")
# st.video("video.mp4")

# st.checkbox('Yes')
# st.radio('Pick your role', ['User', 'System'])
# st.selectbox('Pick a fruit', ['Apple', 'Banana', 'Orange'])
# st.multiselect('Choose a planet', ['Jupiter', 'Mars', 'Neptune'])
# st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
# st.slider('Pick a number', 0, 50)

# st.number_input('Pick a number', 0, 10)
# st.text_input('Email address')
# st.date_input('Traveling date')
# st.time_input('School time')
# st.text_area('Description')
# st.button('Click Me')
# st.file_uploader('Upload a photo')
# st.color_picker('Choose your favorite color')

# st.balloons()  # Celebration balloons
# st.progress(10)  # Progress bar
# with st.spinner('Wait for it...'):
#     time.sleep(10) 
# Initialize session state

if "response" not in st.session_state:
    st.session_state.response = ""
if "image" not in st.session_state:
    st.session_state.image = None
if "images" not in st.session_state:
    st.session_state.images = None
if "nxG" not in st.session_state:
    st.session_state.nxG = None
if "image_index" not in st.session_state:
    st.session_state.image_index = 0
def route():
    with st.spinner("Fetching data from internet...."):
        input_text = st.session_state.input
        st.session_state.image = None
        st.session_state.images = None
        st.session_state.nxG = None
        st.session_state.image_index = 0
        try:
            questionClassification = gq.QuestionClassification(input_text).lower()
            is_image_required = gq.is_image_generation_required(input_text).lower()
            images = []
            if questionClassification == 'self':
                st.session_state.response = gq.generate_content(input_text)
            elif questionClassification == 'factual' or questionClassification == "external":
                with st.spinner("Processing... Please wait."):
                    google_search_query = gq.generate_google_search_query(input_text)
                google_search_query = google_search_query.replace("+"," ")
                print(google_search_query)
                context,images,nxG = rag.get_context(google_search_query,input_text)
                print(st.session_state.images)
                st.session_state.nxG = nxG
                print(len(context.split(" ")))
                st.session_state.response = gq.generate_content_context(context,input_text)
            # st.write(is_image_required)
            # print(is_image_required == "yes")
            if is_image_required == "yes":
                print("Image generation required")
                st.session_state.image = sd.generate_image(st.session_state.response)
            elif len(images)>0:
                print(st.session_state.images)
                st.session_state.images = images
        except Exception as e:
            st.session_state.response = gq.generate_content(input_text)
            # st.session_state.response = "error"
# Function to fetch an image from a URL
def fetch_image(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception:
        return None  # Return None if there's an error
    
col1, col2 = st.columns([2,1])
data2 = np.random.randn(100)
with col1:
    with st.form(key="my_form"):
        recordedText = recorder.text()
        st.text_area('Type anything',key = "input")
        st.form_submit_button("Submit", on_click=route)
with col2:
    fig, ax = plt.subplots()
    if st.session_state.nxG is not None:
        nx.draw(st.session_state.nxG, with_labels=True)
        # ax.hist(data2, bins=20)
        st.pyplot(fig)
    
# Display the response

if st.session_state.response:
    st.markdown("### Response:")
    st.markdown(st.session_state.response)
if st.session_state.image is not None:
    st.image(st.session_state.image, caption="")

print(st.session_state.images)

if st.session_state.images is not None:    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Previous"):
            st.session_state.image_index = (st.session_state.image_index - 1) % len(st.session_state.images)
    with col3:
        if st.button("Next ➡️"):
            st.session_state.image_index = (st.session_state.image_index + 1) % len(st.session_state.images)

    # Fetch and display the current image
    print(st.session_state.images)
    
    current_image = fetch_image(st.session_state.images[st.session_state.image_index])

    if current_image:
        st.image(current_image, caption=f"Image {st.session_state.image_index + 1}", use_container_width =True)
    else:
        st.write("Unable to display the image. Blank shown below.")
        st.empty()  # Show a blank placeholder