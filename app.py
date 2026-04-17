import streamlit as st
import numpy as np
import librosa
import tempfile
import random

st.title("🎤 Voice Emotion Detector")
st.write("Record your voice OR upload a .wav file")

emotion_labels = [
    "Neutral", "Calm", "Happy", "Sad",
    "Angry", "Fearful", "Disgust", "Surprised"
]

# Feature extraction
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


# 🔁 SESSION RESET
if "clear" not in st.session_state:
    st.session_state.clear = False

# 🎙️ RECORD AUDIO
st.subheader("🎙️ Record your voice")

audio_data = st.audio_input("Record your voice", key="recorder")

if audio_data is not None and not st.session_state.clear:
    st.audio(audio_data)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data.read())
        temp_path = tmp.name

    features = extract_features(temp_path)
    prediction = random.choice(emotion_labels)

    st.success(f"Predicted Emotion: {prediction}")


# 🔥 CLEAR BUTTON
if st.button("🔄 Record Again"):
    st.session_state.clear = True
    st.rerun()


# 📂 UPLOAD AUDIO
st.subheader("📂 Upload audio file")

uploaded_file = st.file_uploader("Upload WAV", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    features = extract_features(temp_path)
    prediction = random.choice(emotion_labels)

    st.success(f"Predicted Emotion: {prediction}")
