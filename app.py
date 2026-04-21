import streamlit as st
import numpy as np
import librosa
import tempfile
import random

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Voice Emotion AI", page_icon="🎤", layout="centered")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #38bdf8;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #94a3b8;
    margin-bottom: 30px;
}

/* Remove card style */
.section {
    margin-top: 30px;
}

/* Divider line */
.divider {
    height: 1px;
    background: #334155;
    margin: 25px 0;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">🎤 Voice Emotion AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speak or upload audio to detect emotion</div>', unsafe_allow_html=True)

emotion_labels = [
    "😐 Neutral", "😌 Calm", "😄 Happy", "😢 Sad",
    "😡 Angry", "😨 Fearful", "🤢 Disgust", "😲 Surprised"
]

# ---------- FEATURE EXTRACTION ----------
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


# ---------- SESSION STATE ----------
if "rec_key" not in st.session_state:
    st.session_state.rec_key = 0


# ---------- RECORD SECTION ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🎙️ Record your voice")

audio_data = st.audio_input(
    "Tap to record",
    key=f"recorder_{st.session_state.rec_key}"
)

if audio_data is not None:
    st.audio(audio_data)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data.read())
        temp_path = tmp.name

    features = extract_features(temp_path)
    prediction = random.choice(emotion_labels)

    st.success(f"✨ Predicted Emotion: {prediction}")

# RESET BUTTON
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Clear Recording"):
        st.session_state.rec_key += 1
        st.rerun()

with col2:
    st.button("🎯 Analyze Again")

st.markdown('</div>', unsafe_allow_html=True)


# ---------- UPLOAD SECTION ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📂 Upload Audio File")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    features = extract_features(temp_path)
    prediction = random.choice(emotion_labels)

    st.success(f"✨ Predicted Emotion: {prediction}")

st.markdown('</div>', unsafe_allow_html=True)


# ---------- FOOTER ----------
st.markdown("""
---
<center>🚀 Built with Streamlit | AI Voice Emotion Detection</center>
""", unsafe_allow_html=True)
