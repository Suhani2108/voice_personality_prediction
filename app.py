import streamlit as st
import numpy as np
import librosa
import tempfile
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Voice Personality AI", page_icon="🎤", layout="centered")

# ---------- UI ----------
st.markdown("## 🎤 Voice Personality AI")
st.markdown("Speak or upload audio to detect emotion")

# ---------- FEATURE EXTRACTION ----------
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=4)
        audio, _ = librosa.effects.trim(audio)

        # Energy
        rms = np.mean(librosa.feature.rms(y=audio))

        # ZCR
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

        # Pitch
        f0, _, _ = librosa.pyin(audio, fmin=75, fmax=400)
        f0 = f0[~np.isnan(f0)]

        pitch_mean = np.mean(f0) if len(f0) > 0 else 180
        pitch_std = np.std(f0) if len(f0) > 0 else 20

        # Spectral
        centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

        return rms, zcr, tempo, pitch_mean, pitch_std, centroid

    except:
        return None

# ---------- EMOTION LOGIC ----------
def predict_emotion(f):
    if f is None:
        return "⚠️ Audio too short"

    e, z, t, p, ps, c = f

    scores = {
        "😄 Happy": 0,
        "😢 Sad": 0,
        "😡 Angry": 0,
        "😌 Calm": 0,
        "😨 Fear": 0,
        "🤢 Disgust": 0
    }

    # Energy
    if e > 0.15:
        scores["😡 Angry"] += 2
        scores["😄 Happy"] += 2
    elif e < 0.05:
        scores["😢 Sad"] += 2
        scores["😌 Calm"] += 2
    else:
        scores["😌 Calm"] += 1

    # Pitch
    if p > 240:
        scores["😨 Fear"] += 3
        scores["😄 Happy"] += 1
    elif p < 150:
        scores["😢 Sad"] += 2

    # Pitch variation
    if ps > 50:
        scores["😨 Fear"] += 2
        scores["😡 Angry"] += 1
    elif ps < 15:
        scores["😌 Calm"] += 2

    # Tempo
    if t > 140:
        scores["😄 Happy"] += 2
    elif t < 90:
        scores["😢 Sad"] += 2

    # ZCR
    if z > 0.12:
        scores["😡 Angry"] += 2

    # Spectral
    if c > 3000:
        scores["😄 Happy"] += 1
    elif c < 1500:
        scores["🤢 Disgust"] += 2

    # Final decision
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = int((scores[best] / (total + 1e-6)) * 100)

    if confidence < 35:
        best = "😐 Neutral"
        confidence = 40

    return f"✨ Predicted Emotion: {best} (Confidence: {confidence}%)"


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

    with st.spinner("Analyzing voice..."):
        feats = extract_features(path)
        result = predict_emotion(feats)

    st.success(result)

    # DELETE OPTION
    if st.button("🗑️ Delete Recording"):
        os.unlink(path)
        st.session_state.rec_key += 1
        st.rerun()

# ---------- DIVIDER ----------
st.markdown("---")

# ---------- UPLOAD ----------
st.subheader("📂 Upload Audio File")
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    with st.spinner("Analyzing voice..."):
        feats = extract_features(path)
        result = predict_emotion(feats)

    st.success(result)
