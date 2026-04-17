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
.stButton > button {
    border-radius: 10px;
    background: linear-gradient(90deg, #6366f1, #38bdf8);
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">🎤 Voice Emotion AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speak or upload audio to detect emotion</div>', unsafe_allow_html=True)

# ---------- FEATURE EXTRACTION ----------
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=4)

    energy = np.mean(np.abs(audio))
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    pitch = np.mean(librosa.yin(audio, fmin=50, fmax=300))

    return energy, zcr, spectral_centroid, pitch

# ---------- IMPROVED PREDICTION ----------
def predict_emotion(energy, zcr, centroid, pitch):

    if energy > 0.15 and pitch > 180:
        return "😄 Happy"

    elif energy > 0.18 and zcr > 0.1:
        return "😡 Angry"

    elif pitch > 220 and centroid > 3000:
        return "😲 Surprised"

    elif energy < 0.04:
        return "😢 Sad"

    elif energy < 0.08 and pitch < 140:
        return "😐 Neutral"

    else:
        return "😌 Calm"

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

    energy, zcr, centroid, pitch = extract_features(path)
    prediction = predict_emotion(energy, zcr, centroid, pitch)

    st.success(f"✨ Predicted Emotion: {prediction}")

# ---------- DELETE ----------
if st.button("🗑️ Delete Recording"):
    st.session_state.rec_key += 1
    st.rerun()

# ---------- DIVIDER ----------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---------- UPLOAD ----------
st.subheader("📂 Upload Audio File")

file = st.file_uploader("Upload WAV", type=["wav"])

if file:
    st.audio(file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file.read())
        path = tmp.name

    energy, zcr, centroid, pitch = extract_features(path)
    prediction = predict_emotion(energy, zcr, centroid, pitch)

    st.success(f"✨ Predicted Emotion: {prediction}")
