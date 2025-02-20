from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
import asyncio
import json
import tempfile
import wave
import numpy as np

app = FastAPI()

# Load Whisper model
model = WhisperModel("large-v2", device="cuda", compute_type="float16")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established.")

    while True:
        try:
            # Receive audio data
            data = await websocket.receive_bytes()

            # Save as a temporary WAV file
            with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp_file:
                tmp_file.write(data)
                tmp_file.flush()  # Ensure data is written before reading

                # Transcribe using Whisper
                segments, _ = model.transcribe(tmp_file.name, language="en")

                for segment in segments:
                    print("Transcribed Text:", segment.text)  # Print to console
                    await websocket.send_text(json.dumps({"text": segment.text}))

        except Exception as e:
            print("Error:", str(e))
            await websocket.send_text(json.dumps({"error": str(e)}))
            break

    await websocket.close()
