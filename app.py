import streamlit as st
import numpy as np
import librosa
import tempfile
import soundfile as sf
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Voice Personality AI", page_icon="🎤", layout="centered")

# ---------- UI ----------
st.markdown('<div class="title">🎤 Voice Personality AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speak or upload audio to detect emotion</div>', unsafe_allow_html=True)

# ---------- FEATURE EXTRACTION ----------
def extract_features(file_path):
    try:
        try:
            audio, sr = librosa.load(file_path, sr=22050)
        except:
            audio, sr = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

        audio, _ = librosa.effects.trim(audio, top_db=25)

        if len(audio) < sr * 0.7:
            return None

        rms = np.mean(librosa.feature.rms(y=audio))
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

        # Pitch
        f0, voiced, _ = librosa.pyin(audio, fmin=75, fmax=500, sr=sr)
        f0 = f0[~np.isnan(f0)]

        pitch_mean = np.mean(f0) if len(f0) > 0 else 180
        pitch_std = np.std(f0) if len(f0) > 0 else 20

        return rms, zcr, centroid, tempo, pitch_mean, pitch_std

    except:
        return None


# ---------- IMPROVED EMOTION ----------
def predict_emotion(f):
    if f is None:
        return "⚠️ Audio too short"

    e, z, c, t, p, ps = f

    scores = {
        "😄 Happy": 0,
        "😢 Sad": 0,
        "😡 Angry": 0,
        "😌 Calm": 0,
        "😐 Neutral": 0,
        "😨 Fearful": 0
    }

    # ---------- BALANCED LOGIC ----------

    # Energy
    if e > 0.15:
        scores["😡 Angry"] += 2
        scores["😄 Happy"] += 2
    elif e < 0.05:
        scores["😌 Calm"] += 2
        scores["😢 Sad"] += 1
    else:
        scores["😐 Neutral"] += 2

    # Pitch (MOST IMPORTANT)
    if p > 230:
        scores["😨 Fearful"] += 3
        scores["😄 Happy"] += 2
    elif p < 150:
        scores["😢 Sad"] += 2
    else:
        scores["😐 Neutral"] += 2

    # Pitch variation
    if ps > 45:
        scores["😡 Angry"] += 2
        scores["😨 Fearful"] += 2
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
    if c > 2800:
        scores["😄 Happy"] += 1

    # ---------- DECISION ----------
    best = max(scores, key=scores.get)
    total = sum(scores.values())

    # SAFE CONFIDENCE
    conf = int((scores[best] / (total + 1e-6)) * 100)

    # Neutral fallback (VERY IMPORTANT)
    if conf < 35:
        best = "😐 Neutral"
        conf = 40

    return f"✨ Predicted Emotion: {best} (Confidence: {conf}%)"


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
        feats = extract_features(path)
        prediction = predict_emotion(feats)

    st.success(prediction)
    os.unlink(path)

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
        feats = extract_features(path)
        prediction = predict_emotion(feats)

    st.success(prediction)
    os.unlink(path)
