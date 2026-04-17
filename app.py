import streamlit as st
import numpy as np
import librosa
import tempfile
from tensorflow.keras.models import load_model

st.title("🎤 Voice Emotion Detector")

# Load model
model = load_model("emotion_model.h5")

emotion_labels = [
    "Neutral", "Calm", "Happy", "Sad",
    "Angry", "Fearful", "Disgust", "Surprised"
]

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


audio_data = st.audio_input("Record your voice")

if audio_data is not None:
    st.audio(audio_data)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data.read())
        temp_path = tmp.name

    features = extract_features(temp_path)
    features = np.expand_dims(features, axis=0)

    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)

    st.success(f"Predicted Emotion: {emotion_labels[predicted_class]}")
