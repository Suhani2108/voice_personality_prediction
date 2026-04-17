import streamlit as st
import numpy as np
import librosa
import tempfile
import os
from sklearn.ensemble import RandomForestClassifier

# ---------- PAGE CONFIG ----------
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
    font-size: 48px;
    font-weight: bold;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5f5;
    margin-bottom: 25px;
}
.divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #38bdf8, transparent);
    margin: 25px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">🎤 Voice Emotion AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speak or upload audio to detect emotion</div>', unsafe_allow_html=True)

emotion_map = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}

# ---------- FEATURE EXTRACTION ----------
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# ---------- TRAIN MODEL ----------
@st.cache_resource
def train_model():
    X = []
    y = []

    DATA_PATH = "dataset"   # dataset folder required

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".wav"):
                try:
                    emotion = file.split("-")[2]
                    label = emotion_map.get(emotion)

                    file_path = os.path.join(root, file)
                    features = extract_features(file_path)

                    X.append(features)
                    y.append(label)
                except:
                    pass

    X = np.array(X)
    y = np.array(y)

    model = RandomForestClassifier()
    model.fit(X, y)

    return model

# Load / train model
model = train_model()

# ---------- SESSION ----------
if "rec_key" not in st.session_state:
    st.session_state.rec_key = 0

# ---------- RECORD ----------
st.subheader("🎙️ Record your voice")

audio_data = st.audio_input("Tap to record", key=f"rec_{st.session_state.rec_key}")

if audio_data is not None:
    st.audio(audio_data)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data.read())
        temp_path = tmp.name

    features = extract_features(temp_path).reshape(1, -1)

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

    features = extract_features(temp_path).reshape(1, -1)

    prediction = model.predict(features)[0]

    st.success(f"✨ Predicted Emotion: {prediction}")
