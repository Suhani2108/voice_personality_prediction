import streamlit as st
import numpy as np
import librosa
import tempfile
import joblib

# ---------- LOAD MODEL ----------
# Train separately and save as emotion_model.pkl
try:
    model = joblib.load("emotion_model.pkl")
except:
    model = None

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Voice Personality AI", page_icon="🎤", layout="centered")

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
st.markdown('<div class="title">🎤 Voice Personality AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speak or upload audio to detect emotion</div>', unsafe_allow_html=True)


# ---------- FEATURE EXTRACTION ----------
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=7, sr=22050)
        audio, _ = librosa.effects.trim(audio, top_db=18)

        if len(audio) < sr * 0.5:
            return None

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)

        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        rms = np.mean(librosa.feature.rms(y=audio))
        centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

        features = np.hstack([mfcc_mean, zcr, rms, centroid])

        return features.reshape(1, -1)

    except Exception as ex:
        st.error(f"Feature extraction error: {ex}")
        return None


# ---------- EMOTION PREDICTION ----------
def predict_emotion(features):
    if features is None:
        return "⚠️ Audio too short — please record at least 1 second"

    if model is None:
        return "⚠️ Model not found. Please train and save emotion_model.pkl"

    prediction = model.predict(features)[0]

    return f"✨ Predicted Emotion: {prediction}"


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

    with st.spinner("Analyzing audio..."):
        feats = extract_features(path)
        prediction = predict_emotion(feats)

    st.success(prediction)

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

    with st.spinner("Analyzing audio..."):
        feats = extract_features(path)
        prediction = predict_emotion(feats)

    st.success(prediction)
