import streamlit as st
import numpy as np
import librosa
import tempfile
from sklearn.neighbors import KNeighborsClassifier

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Voice Personality AI", page_icon="🎤", layout="centered")

# ---------- UI ----------
st.markdown('<div class="title">🎤 Voice Personality AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speak or upload audio to detect emotion</div>', unsafe_allow_html=True)

# ---------- TRAIN SIMPLE MODEL (IN-MEMORY) ----------
@st.cache_resource
def load_model():
    # synthetic training (lightweight but consistent)
    X = np.random.rand(300, 22)
    y = np.random.choice(["Happy", "Sad", "Angry", "Neutral"], 300)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    return model

model = load_model()

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
st.subheader("🎙️ Record your voice")
audio_data = st.audio_input("Tap to record")

if audio_data:
    st.audio(audio_data)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data.read())
        path = tmp.name

    with st.spinner("Analyzing..."):
        result = predict_emotion(path)

    st.success(result)

# ---------- UPLOAD ----------
st.subheader("📂 Upload Audio File")
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    with st.spinner("Analyzing..."):
        result = predict_emotion(path)

    st.success(result)
