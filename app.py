import streamlit as st
import numpy as np
import librosa
import tempfile

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
.divider {
    height: 1px;
    background: #334155;
    margin: 25px 0;
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
    pitch = np.mean(librosa.yin(audio, fmin=50, fmax=300))
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

    return energy, zcr, pitch, centroid


# ---------- BETTER LOGIC ----------
def predict_emotion(energy, zcr, pitch, centroid):

    if energy > 0.15 and pitch > 170:
        return "😄 Happy"

    elif energy > 0.18 and zcr > 0.1:
        return "😡 Angry"

    elif pitch > 220 and centroid > 3000:
        return "😲 Surprised"

    elif energy < 0.04:
        return "😢 Sad"

    elif energy < 0.08:
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

    energy, zcr, pitch, centroid = extract_features(path)
    prediction = predict_emotion(energy, zcr, pitch, centroid)

    st.success(f"✨ Predicted Emotion: {prediction}")


# ---------- RESET ----------
if st.button("🗑️ Clear Recording"):
    st.session_state.rec_key += 1
    st.rerun()


# ---------- DIVIDER ----------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ---------- UPLOAD ----------
st.subheader("📂 Upload Audio File")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    energy, zcr, pitch, centroid = extract_features(path)
    prediction = predict_emotion(energy, zcr, pitch, centroid)

    st.success(f"✨ Predicted Emotion: {prediction}")
