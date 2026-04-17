import streamlit as st
import numpy as np
import librosa
import tempfile

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
st.markdown('<div class="title">🎤 Voice Personality AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze emotions from your voice</div>', unsafe_allow_html=True)

# ---------- FEATURE EXTRACTION ----------
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=4)
    energy = np.mean(np.abs(audio))
    pitch = np.mean(librosa.yin(audio, fmin=50, fmax=300))
    return energy, pitch

# ---------- SIMPLE AI LOGIC ----------
def predict_emotion(energy, pitch):
    if energy > 0.1 and pitch > 150:
        return "😄 Happy"
    elif energy < 0.05:
        return "😢 Sad"
    elif pitch > 200:
        return "😲 Surprised"
    elif energy > 0.08:
        return "😡 Angry"
    else:
        return "😐 Neutral"

# ---------- RECORD ----------
audio_data = st.audio_input("🎙️ Record your voice")

if audio_data:
    st.audio(audio_data)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data.read())
        path = tmp.name

    energy, pitch = extract_features(path)
    prediction = predict_emotion(energy, pitch)

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

    energy, pitch = extract_features(path)
    prediction = predict_emotion(energy, pitch)

    st.success(f"✨ Predicted Emotion: {prediction}")
