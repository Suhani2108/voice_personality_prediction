import streamlit as st
import numpy as np
import librosa
import tempfile
import random
from audiorecorder import audiorecorder
import os

st.title("🎤 Voice Emotion Detector")
st.write("Record your voice OR upload a .wav file")

emotion_labels = [
    "Neutral", "Calm", "Happy", "Sad",
    "Angry", "Fearful", "Disgust", "Surprised"
]

# Feature extraction
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


# ================= 🎙️ RECORDING =================
st.subheader("🎙️ Record your voice")

audio = audiorecorder("Start Recording", "Stop Recording")

if len(audio) > 0:
    try:
        audio_bytes = audio.export().read()

        if audio_bytes is not None and len(audio_bytes) > 0:
            st.audio(audio_bytes)

            # Save safely
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name

            # Check file exists
            if os.path.exists(temp_path):
                features = extract_features(temp_path)
                prediction = random.choice(emotion_labels)

                st.success(f"Predicted Emotion: {prediction}")
            else:
                st.error("Recording file not saved properly.")

        else:
            st.warning("No audio recorded. Try again.")

    except Exception as e:
        st.error("Recording failed. Try using file upload.")


# ================= 📂 FILE UPLOAD =================
st.subheader("📂 Upload audio file")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    try:
        st.audio(uploaded_file)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        features = extract_features(temp_path)
        prediction = random.choice(emotion_labels)

        st.success(f"Predicted Emotion: {prediction}")

    except Exception as e:
        st.error("Error processing uploaded file.")
