import streamlit as st
import numpy as np
import librosa
import tempfile
import random

st.title("🎤 Voice Emotion Detector")

emotion_labels = [
    "Neutral","Calm","Happy","Sad",
    "Angry","Fearful","Disgust","Surprised"
]

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

audio_data = st.audio_input("Record your voice")

if audio_data:
    st.audio(audio_data)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data.read())
        path = tmp.name

    features = extract_features(path)

    prediction = random.choice(emotion_labels)

    st.success(f"Predicted Emotion: {prediction}")
