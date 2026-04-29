import streamlit as st
import numpy as np
import librosa
import tempfile
import joblib
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Voice Personality AI", page_icon="🎤", layout="centered")

# ---------- UI ----------
st.markdown('<div class="title">🎤 Voice Personality AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speak or upload audio to detect emotion</div>', unsafe_allow_html=True)

# ---------- LOAD MODEL SAFELY ----------
MODEL_PATH = "emotion_model.pkl"

model = None
model_loaded = False

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        model_loaded = True
    except:
        st.error("❌ Model file is corrupted.")
else:
    st.warning("⚠️ Model not found. Using basic analysis (not AI).")

# ---------- FEATURE EXTRACTION ----------
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    rms = np.mean(librosa.feature.rms(y=audio))

    return np.hstack([mfcc_mean, zcr, rms]).reshape(1, -1)

# ---------- FALLBACK LOGIC ----------
def basic_prediction(features):
    energy = features[0][-1]  # RMS approx

    if energy > 0.1:
        return "😄 Happy (basic guess)"
    elif energy < 0.03:
        return "😢 Sad (basic guess)"
    else:
        return "😐 Neutral (basic guess)"

# ---------- PREDICTION ----------
def predict_emotion(file_path):
    features = extract_features(file_path)

    if model_loaded:
        pred = model.predict(features)[0]
        return f"✨ Predicted Emotion: {pred}"
    else:
        return f"⚠️ {basic_prediction(features)}"

# ---------- SESSION ----------
if "rec_key" not in st.session_state:
    st.session_state.rec_key = 0

# ---------- RECORD ----------
st.subheader("🎙️ Record your voice")
audio_data = st.audio_input("Tap to record", key=f"rec_{st.session_state.rec_key}")

if audio_data:
    st.audio(audio_data)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data.read())
        path = tmp.name

    with st.spinner("Analyzing..."):
        result = predict_emotion(path)

    st.success(result)

# ---------- RESET ----------
if st.button("🗑️ Clear Recording"):
    st.session_state.rec_key += 1
    st.rerun()

# ---------- DIVIDER ----------
st.markdown("---")

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
