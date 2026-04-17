import streamlit as st
import numpy as np
import librosa
import tempfile
import joblib

# ---------- PAGE ----------
st.set_page_config(page_title="Voice Emotion AI", page_icon="🎤")

# ---------- CSS ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #020617);
    color: white;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #cbd5f5;
    margin-bottom: 20px;
}
.divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #38bdf8, transparent);
    margin: 25px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">🎤 Voice Emotion AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speak or upload audio to detect emotion</div>', unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
model = joblib.load("model.pkl")

# ---------- FEATURE ----------
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=4)
    audio = librosa.util.normalize(audio)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

# ---------- RECORD ----------
audio_data = st.audio_input("🎙️ Record your voice")

if audio_data:
    st.audio(audio_data)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data.read())
        path = tmp.name

    features = extract_features(path)
    prediction = model.predict(features)[0]

    st.success(f"✨ Predicted Emotion: {prediction}")

# ---------- DIVIDER ----------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---------- UPLOAD ----------
file = st.file_uploader("📂 Upload WAV", type=["wav"])

if file:
    st.audio(file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file.read())
        path = tmp.name

    features = extract_features(path)
    prediction = model.predict(features)[0]

    st.success(f"✨ Predicted Emotion: {prediction}")
