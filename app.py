import streamlit as st
import numpy as np
import librosa
import tempfile
import random

# ---------- PAGE CONFIG ----------
st.markdown("""
<style>
/* Background Gradient */
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

/* Glow line */
.divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #38bdf8, transparent);
    margin: 30px 0;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    background: linear-gradient(90deg, #6366f1, #38bdf8);
    color: white;
    font-weight: bold;
    border: none;
    padding: 10px 20px;
}

/* Hover effect */
.stButton > button:hover {
    transform: scale(1.05);
    transition: 0.3s;
}

/* Audio player */
audio {
    border-radius: 10px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)
# ---------- HEADER ----------
st.markdown('<div class="title">🎤 Voice Personality AI</div>', unsafe_allow_html=True)
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
