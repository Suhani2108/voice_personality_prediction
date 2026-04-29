import streamlit as st
import numpy as np
import librosa
import tempfile

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

        # Energy
        rms = np.mean(librosa.feature.rms(y=audio))

        # ZCR
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

        # Pitch (IMPORTANT)
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        f0 = f0[~np.isnan(f0)]

        pitch_mean = np.mean(f0) if len(f0) > 0 else 0
        pitch_std = np.std(f0) if len(f0) > 0 else 0

        # Spectral
        centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))

        return {
            "energy": rms,
            "zcr": zcr,
            "tempo": tempo,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "centroid": centroid,
            "bandwidth": bandwidth
        }

    except Exception as ex:
        st.error(f"Feature extraction error: {ex}")
        return None


# ---------- IMPROVED EMOTION LOGIC ----------
def predict_emotion(f):
    if f is None:
        return "⚠️ Audio too short — please record at least 1 second"

    e = f["energy"]
    z = f["zcr"]
    t = f["tempo"]
    p = f["pitch_mean"]
    ps = f["pitch_std"]
    c = f["centroid"]

    scores = {
        "😄 Happy": 0,
        "😢 Sad": 0,
        "😡 Angry": 0,
        "😌 Calm": 0,
        "😐 Neutral": 0,
        "😨 Fearful": 0
    }

    # ENERGY
    if e > 0.12:
        scores["😡 Angry"] += 2
        scores["😄 Happy"] += 1
    elif e < 0.04:
        scores["😢 Sad"] += 2
        scores["😌 Calm"] += 1
    else:
        scores["😐 Neutral"] += 2

    # TEMPO
    if t > 140:
        scores["😄 Happy"] += 2
        scores["😡 Angry"] += 1
    elif t < 90:
        scores["😢 Sad"] += 2
        scores["😌 Calm"] += 1

    # PITCH
    if p > 220:
        scores["😄 Happy"] += 2
        scores["😨 Fearful"] += 1
    elif p < 150:
        scores["😢 Sad"] += 2

    # PITCH VARIATION
    if ps > 40:
        scores["😨 Fearful"] += 2
        scores["😡 Angry"] += 1
    elif ps < 15:
        scores["😌 Calm"] += 2

    # ZCR
    if z > 0.1:
        scores["😡 Angry"] += 2
    elif z < 0.05:
        scores["😌 Calm"] += 2

    # SPECTRAL
    if c > 2500:
        scores["😄 Happy"] += 1
    else:
        scores["😢 Sad"] += 1

    # FINAL
    best = max(scores, key=scores.get)

    return f"✨ Predicted Emotion: {best}"


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
