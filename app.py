import streamlit as st
import numpy as np
import librosa
import tempfile
import random

# Title
st.title("🎤 Voice Emotion Detector")
st.write("Upload a .wav audio file to predict emotion")

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

# Feature extraction function
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# File uploader
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    try:
        # Play audio
        st.audio(uploaded_file, format="audio/wav")

        # Save temp file safely
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        # Extract features
        features = extract_features(temp_path)

        # Dummy prediction (for demo)
        prediction = random.choice(emotion_labels)

        # Show result
        st.success(f"Predicted Emotion: {prediction}")

    except Exception as e:
        st.error("Error processing audio. Please upload a valid .wav file.")
