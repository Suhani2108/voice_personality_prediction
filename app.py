import streamlit as st
import numpy as np
import librosa
import tempfile
import random

# Emotion labels
emotion_labels = [
    "Neutral",
    "Calm",
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgust",
    "Surprised"
]

# Feature extraction (for demo)
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

st.title("🎤 Voice Emotion Detector (Demo)")
st.write("Upload a .wav audio file")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        st.audio(uploaded_file, format="audio/wav")

        # Extract features
        features = extract_features(temp_path)

        # Dummy prediction (random for demo)
        prediction = random.choice(emotion_labels)

        st.success(f"Predicted Emotion: {prediction}")

    except Exception as e:
        st.error(f"Error: {e}")
