import streamlit as st
import numpy as np
import librosa
import tempfile
from sklearn.neighbors import KNeighborsClassifier

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Voice Personality AI", page_icon="🎤")

st.markdown("## 🎤 Voice Personality AI")
st.markdown("Real-time adaptive emotion detection")

# ---------- MODEL (adaptive memory) ----------
@st.cache_resource
def init_model():
    model = KNeighborsClassifier(n_neighbors=3)

    # initial realistic patterns (not random)
    X = np.array([
        [0.02]*22,   # sad
        [0.15]*22,   # happy
        [0.20]*22,   # angry
        [0.08]*22    # neutral
    ])

    y = np.array(["Sad", "Happy", "Angry", "Neutral"])

    model.fit(X, y)
    return model

model = init_model()

# ---------- FEATURE EXTRACTION ----------
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    rms = np.mean(librosa.feature.rms(y=audio))

    return np.hstack([mfcc_mean, zcr, rms]).reshape(1, -1)

# ---------- PREDICT ----------
def predict_emotion(features):
    pred = model.predict(features)[0]

    # confidence (distance-based)
    dist, _ = model.kneighbors(features)
    confidence = max(40, int(100 - dist[0][0]*100))

    return pred, confidence

# ---------- UPDATE MODEL (learning) ----------
def update_model(features, label):
    global model
    X_new = np.vstack([model._fit_X, features])
    y_new = np.append(model._y, label)

    model.fit(X_new, y_new)

# ---------- RECORD ----------
st.subheader("🎙️ Record your voice")
audio_data = st.audio_input("Tap to record")

if audio_data:
    st.audio(audio_data)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data.read())
        path = tmp.name

    with st.spinner("Analyzing..."):
        feats = extract_features(path)
        pred, conf = predict_emotion(feats)

    st.success(f"✨ Predicted Emotion: {pred} (Confidence: {conf}%)")

    # optional learning
    st.markdown("### Improve accuracy (optional)")
    correct = st.selectbox("Correct emotion?", ["", "Happy", "Sad", "Angry", "Neutral"])

    if st.button("Update Model"):
        if correct:
            update_model(feats, correct)
            st.success("✅ Model improved!")

# ---------- UPLOAD ----------
st.subheader("📂 Upload Audio File")
uploaded_file = st.file_uploader("Upload WAV", type=["wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    feats = extract_features(path)
    pred, conf = predict_emotion(feats)

    st.success(f"✨ Predicted Emotion: {pred} (Confidence: {conf}%)")
