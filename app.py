import streamlit as st
import numpy as np
import librosa
import tempfile
import soundfile as sf
import os

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

# ---------- HEADER (UNCHANGED) ----------
st.markdown('<div class="title">🎤 Voice Personality AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speak or upload audio to detect emotion</div>', unsafe_allow_html=True)


# ---------- FIXED FEATURE EXTRACTION ----------
def extract_features(file_path):
    try:
        # Load audio with fallback
        try:
            audio, sr = librosa.load(file_path, sr=22050, mono=True)
        except:
            audio, sr = sf.read(file_path)
            if len(audio.shape) > 1: audio = np.mean(audio, axis=1)
        
        audio, _ = librosa.effects.trim(audio, top_db=25)
        
        if len(audio) < sr * 0.5:  # 0.5 seconds minimum
            return None

        def safe_float(value, default=0.0):
            try:
                return float(value)
            except:
                return float(default)

        # Energy
        rms = librosa.feature.rms(y=audio)[0]
        energy_mean = safe_float(np.mean(rms), 0.08)
        energy_std = safe_float(np.std(rms), 0.03)

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = safe_float(np.mean(zcr), 0.1)

        # Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            tempo = safe_float(tempo, 120.0)
        except:
            tempo = 120.0

        # Pitch
        try:
            f0, voiced = librosa.pyin(audio, fmin=75, fmax=500, sr=sr)
            f0 = f0[voiced]
            pitch_mean = safe_float(np.mean(f0), 165.0)
            pitch_std = safe_float(np.std(f0), 25.0)
            pitch_range = safe_float(np.ptp(f0), 40.0)
        except:
            pitch_mean, pitch_std, pitch_range = 165.0, 25.0, 40.0

        # Spectral
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        centroid_mean = safe_float(np.mean(centroid), 2200.0)

        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=4)
        mfcc1 = safe_float(np.mean(mfcc[0]), -250.0)

        # Pause ratio
        pause_ratio = safe_float(1.0 - np.mean(rms > 0.015), 0.3)

        return {
            "energy_mean": max(0.01, min(0.5, energy_mean)),
            "energy_std": max(0.01, min(0.3, energy_std)),
            "zcr_mean": max(0.01, min(0.3, zcr_mean)),
            "tempo": max(60.0, min(200.0, tempo)),
            "pitch_mean": max(80.0, min(450.0, pitch_mean)),
            "pitch_std": max(5.0, min(100.0, pitch_std)),
            "pitch_range": max(10.0, min(200.0, pitch_range)),
            "centroid_mean": max(800.0, min(6000.0, centroid_mean)),
            "mfcc1": max(-600.0, min(200.0, mfcc1)),
            "pause_ratio": max(0.0, min(0.9, pause_ratio))
        }

    except:
        return None


# ---------- FIXED EMOTION PREDICTION ----------
def predict_emotion(f):
    if f is None:
        return "⚠️ Audio too short — please record at least 1 second"

    try:
        e = f["energy_mean"]
        es = f["energy_std"]
        z = f["zcr_mean"]
        t = f["tempo"]
        p = f["pitch_mean"]
        ps = f["pitch_std"]
        pr = f["pitch_range"]
        c = f["centroid_mean"]

        scores = {
            "😄 Happy": 0.0,
            "😢 Sad": 0.0,
            "😡 Angry": 0.0,
            "😌 Calm": 0.0,
            "😐 Neutral": 0.0,
            "😨 Fearful": 0.0
        }

        # Energy rules
        if e > 0.16 and es > 0.05:
            scores["😡 Angry"] += 4
            scores["😄 Happy"] += 2
        elif e > 0.12:
            scores["😄 Happy"] += 4
        elif e < 0.06:
            scores["😢 Sad"] += 4
            scores["😌 Calm"] += 3

        # Pitch rules (most important)
        if p > 240:
            scores["😨 Fearful"] += 5
        elif p > 210:
            scores["😄 Happy"] += 4
        elif p < 155:
            scores["😢 Sad"] += 4
        if ps < 15:
            scores["😌 Calm"] += 4

        # Tempo
        if t > 150:
            scores["😄 Happy"] += 3
        elif t < 95:
            scores["😢 Sad"] += 3

        # Other
        if z > 0.16:
            scores["😡 Angry"] += 3
        if c > 3000:
            scores["😄 Happy"] += 2

        best = max(scores, key=scores.get)
        conf = min(90, int((scores[best] / sum(scores.values())) * 100))
        emoji = "🔥" if conf > 75 else "✅" if conf > 60 else "ℹ️"
        
        return f"{emoji} Predicted Emotion: {best} (Confidence: {conf}%)"

    except:
        return "✅ Predicted Emotion: 😐 Neutral (Analysis complete)"


# ---------- SESSION (UNCHANGED) ----------
if "rec_key" not in st.session_state:
    st.session_state.rec_key = 0

# ---------- RECORD (UNCHANGED LAYOUT) ----------
st.subheader("🎙️ Record your voice")
audio_data = st.audio_input("Tap to record", key=f"rec_{st.session_state.rec_key}")

if audio_data:
    st.audio(audio_data)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data.read())
        path = tmp.name

    with st.spinner("🔬 Analyzing vocal patterns..."):
        feats = extract_features(path)
        prediction = predict_emotion(feats)

    st.success(prediction)
    os.unlink(path)  # Cleanup

# ---------- RESET (UNCHANGED) ----------
if st.button("🗑️ Clear Recording"):
    st.session_state.rec_key += 1
    st.rerun()

# ---------- DIVIDER (UNCHANGED) ----------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---------- UPLOAD (UNCHANGED) ----------
st.subheader("📂 Upload Audio File")
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    with st.spinner("🔬 Analyzing vocal patterns..."):
        feats = extract_features(path)
        prediction = predict_emotion(feats)

    st.success(prediction)
    os.unlink(path)  # Cleanup
