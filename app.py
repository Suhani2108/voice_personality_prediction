import streamlit as st
import numpy as np
import librosa
import tempfile
import joblib

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Voice Emotion AI", page_icon="🎤")

# ---------- CSS ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #020617);
    color: white;
}

/* Title */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5f5;
    margin-bottom: 30px;
}

/* Divider */
.divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #38bdf8, transparent);
    margin: 25px 0;
}

/* Button */
.stButton > button {
    border-radius: 10px;
    background: linear-gradient(90deg, #6366f1, #38bdf8);
    color: white;
    font-weight: bold;
    border: none;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">🎤 Voice Emotion AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speak or upload audio to detect emotion</div>', unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
model = joblib.load("model.pkl")

# ---------- FEATURE EXTRACTION ----------
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=4)
    audio = librosa.util.normalize(audio)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc.reshape(1, -1)

# ---------- SESSION STATE ----------
if "rec_key" not in st.session_state:
    st.session_state.rec_key = 0

# ---------- RECORD ----------
st.subheader("🎙️ Record your voice")

audio_data = st.audio_input(
    "Tap to record",
    key=f"rec_{st.session_state.rec_key}"
)

if audio_data is not None:
    st.audio(audio_data)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data.read())
        temp_path = tmp.name

    features = extract_features(temp_path)

    prediction = model.predict(features)[0]

    st.success(f"✨ Predicted Emotion: {prediction}")

# RESET
if st.button("🗑️ Clear Recording"):
    st.session_state.rec_key += 1
    st.rerun()

# ---------- DIVIDER ----------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---------- UPLOAD ----------
st.subheader("📂 Upload Audio File")

uploaded_file = st.file_uploader("Upload WAV", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    features = extract_features(temp_path)

    prediction = model.predict(features)[0]

    st.success(f"✨ Predicted Emotion: {prediction}")
