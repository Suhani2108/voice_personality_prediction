import streamlit as st
import numpy as np
import librosa
import tempfile
import joblib
from tensorflow.keras.models import load_model  # ✅ FIXED

# ---------- LOAD FILES ----------
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Voice Personality AI", page_icon="🎤", layout="centered")

# ---------- CUSTOM CSS (UNCHANGED) ----------
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


# ---------- FEATURE EXTRACTION (MATCH TRAINING) ----------
def extract_features(file_path):
    data, sample_rate = librosa.load(file_path, duration=2.5, offset=0.6)

    result = np.array([])

    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma
    stft = np.abs(librosa.stft(data))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    # RMS
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # Mel
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result


# ---------- PREDICTION ----------
def predict_emotion(file_path):
    features = extract_features(file_path)

    features = features.reshape(1, -1)

    # Scale (same as training)
    features = scaler.transform(features)

    # CNN shape
    features = np.expand_dims(features, axis=2)

    # Predict
    preds = model.predict(features)[0]

    pred_index = np.argmax(preds)
    confidence = float(np.max(preds) * 100)

    # Decode label
    emotion = encoder.categories_[0][pred_index]

    return f"✨ Predicted Emotion: {emotion} (Confidence: {confidence:.2f}%)"


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
        prediction = predict_emotion(path)

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

    with st.spinner("Analyzing..."):
        prediction = predict_emotion(path)

    st.success(prediction)
