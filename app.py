import streamlit as st
import numpy as np
import tempfile
import librosa
from tensorflow.keras.models import load_model

# Load model safely
@st.cache_resource
def load_my_model():
    return load_model("emotion_model.h5")

model = load_my_model()

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

# Feature extraction
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# UI
st.title("🎤 Voice Emotion Detector")
st.write("Upload a .wav audio file")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        # Play audio
        st.audio(uploaded_file, format="audio/wav")

        # Extract features
        features = extract_features(temp_path)
        features = np.expand_dims(features, axis=0)

        # Predict
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction)

        # Output
        st.success(f"Predicted Emotion: {emotion_labels[predicted_class]}")

    except Exception as e:
        st.error(f"Error: {e}")
