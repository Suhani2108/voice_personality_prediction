import streamlit as st
import numpy as np
import librosa
import tempfile
import joblib

# ---------- LOAD MODEL ----------
model = joblib.load("emotion_model.pkl")
import os

if not os.path.exists("emotion_model.pkl"):
    st.error("❌ Model file not found. Upload emotion_model.pkl")
    st.stop()

model = joblib.load("emotion_model.pkl")
# ---------- UI ----------
st.set_page_config(page_title="Voice Personality AI", page_icon="🎤")

st.markdown('<div class="title">🎤 Voice Personality AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speak or upload audio to detect emotion</div>', unsafe_allow_html=True)

# ---------- FEATURE EXTRACTION ----------
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    rms = np.mean(librosa.feature.rms(y=audio))

    return np.hstack([mfcc_mean, zcr, rms]).reshape(1, -1)

# ---------- PREDICT ----------
def predict_emotion(file_path):
    features = extract_features(file_path)
    pred = model.predict(features)[0]
    return f"✨ Predicted Emotion: {pred}"

# ---------- RECORD ----------
audio_data = st.audio_input("🎙️ Record")

if audio_data:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data.read())
        path = tmp.name

    st.success(predict_emotion(path))

# ---------- UPLOAD ----------
uploaded_file = st.file_uploader("Upload WAV", type=["wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    st.success(predict_emotion(path))
