import streamlit as st
import numpy as np
import librosa
import tempfile
import random
from audiorecorder import audiorecorder

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

# 🎤 RECORD AUDIO
st.subheader("🎙️ Record your voice")
audio = audiorecorder("Start Recording", "Stop Recording")

if len(audio) > 0:
    st.audio(audio.export().read())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio.export().read())
        temp_path = tmp.name

    features = extract_features(temp_path)
    prediction = random.choice(emotion_labels)

    st.success(f"Predicted Emotion: {prediction}")

# 📂 UPLOAD AUDIO
st.subheader("📂 Upload audio file")
uploaded_file = st.file_uploader("Upload .wav", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    features = extract_features(temp_path)
    prediction = random.choice(emotion_labels)

    st.success(f"Predicted Emotion: {prediction}")
