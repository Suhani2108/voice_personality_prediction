import streamlit as st
import numpy as np
import librosa
import tempfile
import joblib

# ---------- LOAD MODEL ----------
model = joblib.load("model.pkl")

emotion_labels = [
    "Neutral", "Calm", "Happy", "Sad",
    "Angry", "Fearful", "Disgust", "Surprised"
]

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


# 🎙️ RECORD AUDIO
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

    # 🔥 REAL PREDICTION
    prediction = model.predict(features)[0]

    st.success(f"✨ Predicted Emotion: {prediction}")


# RESET BUTTON
if st.button("🗑️ Clear Recording"):
    st.session_state.rec_key += 1
    st.rerun()


# 📂 UPLOAD AUDIO
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
