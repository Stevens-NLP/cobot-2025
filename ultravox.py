import torch
from TTS.api import TTS

# # tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=True)
tts = TTS(model_name="tts_models/en/jenny/jenny", progress_bar=True, gpu=True)
# tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=True, gpu=True)

print(tts.speakers)
print(tts.languages)
# speakers = ['female-en-5', 'female-en-5', 'female-pt-4', 'male-en-2', 'male-en-2', 'male-pt-3']
# Generate speech

text = '''
India, the seventh-largest country in the world, is a vibrant and diverse nation located in South Asia. Known as the world's largest democracy, it is home to over 1.4 billion people, making it the second-most populous country. India's rich cultural heritage is reflected in its many languages, religions, festivals, and cuisines. From the snow-capped Himalayas in the north to the tropical beaches of the south, India's geography is as varied as its culture. It is a land of ancient history, boasting landmarks such as the Taj Mahal, while also being a hub for modern technological and economic growth.
'''

tts.tts_to_file(
    text=text, 
    file_path="output_tortoise.wav",
    # language="en",
    # speaker=tts.speakers[2],  # Specify speaker index for multi-speaker models
    emotion="happy" # Optional: Set emotion (if supported)
)

# from TTS.utils.manage import ModelManager

# manager = ModelManager()
# models = manager.list_models()
# print("\n".join(models))