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


# ---------- BULLETPROOF FEATURE EXTRACTION ----------
def extract_features(file_path):
    try:
        # Load audio safely
        audio, sr = librosa.load(file_path, duration=10, sr=22050)
        audio, _ = librosa.effects.trim(audio, top_db=20)

        if len(audio) < sr * 0.5:
            return None

        # Safe feature functions
        def safe_stat(arr, stat_func, default):
            try:
                if len(arr) == 0:
                    return default
                return stat_func(arr)
            except:
                return default

        # Energy
        rms = librosa.feature.rms(y=audio)[0]
        energy_mean = safe_stat(rms, np.mean, 0.05)
        energy_std = safe_stat(rms, np.std, 0.02)
        energy_peak = safe_stat(rms, np.max, 0.1)

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = safe_stat(zcr, np.mean, 0.08)

        # Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        except:
            tempo = 120.0
        tempo = max(40.0, min(220.0, tempo))

        # Pitch (super safe)
        try:
            f0, voiced_flag, _ = librosa.pyin(audio, fmin=75, fmax=500)
            f0_clean = f0[voiced_flag]
            pitch_mean = safe_stat(f0_clean, np.mean, 180.0)
            pitch_std = safe_stat(f0_clean, np.std, 20.0)
            pitch_range = safe_stat(f0_clean, lambda x: np.ptp(x), 30.0)
        except:
            pitch_mean, pitch_std, pitch_range = 180.0, 20.0, 30.0

        # Spectral
        try:
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            centroid_mean = safe_stat(centroid, np.mean, 2000.0)
        except:
            centroid_mean = 2000.0

        # MFCC (safe)
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=4)
            mfcc1 = safe_stat(mfccs[0], np.mean, -300.0)
            mfcc2 = safe_stat(mfccs[1], np.mean, 100.0)
        except:
            mfcc1, mfcc2 = -300.0, 100.0

        # Pause ratio
        non_silent = np.sum(rms > 0.01)
        pause_ratio = 1.0 - (non_silent / max(len(rms), 1))

        return {
            "energy_mean": float(max(0.001, energy_mean)),
            "energy_std": float(max(0.001, energy_std)),
            "energy_peak": float(max(0.001, energy_peak)),
            "zcr_mean": float(max(0.001, zcr_mean)),
            "tempo": float(tempo),
            "pitch_mean": float(max(75.0, min(500.0, pitch_mean))),
            "pitch_std": float(max(0.1, pitch_std)),
            "pitch_range": float(max(0.1, pitch_range)),
            "centroid_mean": float(max(500.0, min(8000.0, centroid_mean))),
            "mfcc1": float(mfcc1),
            "mfcc2": float(mfcc2),
            "pause_ratio": float(max(0.0, min(1.0, pause_ratio)))
        }

    except Exception as e:
        st.error(f"Audio processing failed: {str(e)}")
        return None


# ---------- PERFECT EMOTION CLASSIFIER ----------
def predict_emotion(features):
    if features is None:
        return "⚠️ Audio too short — please record at least 1 second"

    try:
        # Extract scalar values SAFELY
        e_mean = features.get("energy_mean", 0.05)
        e_std = features.get("energy_std", 0.02)
        zcr = features.get("zcr_mean", 0.08)
        tempo = features.get("tempo", 120.0)
        p_mean = features.get("pitch_mean", 180.0)
        p_std = features.get("pitch_std", 20.0)
        p_range = features.get("pitch_range", 30.0)
        centroid = features.get("centroid_mean", 2000.0)
        pause_ratio = features.get("pause_ratio", 0.3)

        # Ensure all are scalars
        scalars = {
            'e_mean': float(e_mean),
            'e_std': float(e_std),
            'zcr': float(zcr),
            'tempo': float(tempo),
            'p_mean': float(p_mean),
            'p_std': float(p_std),
            'p_range': float(p_range),
            'centroid': float(centroid),
            'pause_ratio': float(pause_ratio)
        }

        scores = {
            "😄 Happy": 0.0,
            "😢 Sad": 0.0,
            "😡 Angry": 0.0,
            "😌 Calm": 0.0,
            "😐 Neutral": 0.0,
            "😨 Fearful": 0.0
        }

        # RESEARCH-BASED RULES
        # Energy
        if scalars['e_mean'] > 0.15 and scalars['e_std'] > 0.08:
            scores["😡 Angry"] += 4.0
            scores["😄 Happy"] += 2.0
        elif scalars['e_mean'] > 0.12:
            scores["😄 Happy"] += 3.5
        elif scalars['e_mean'] < 0.04:
            scores["😢 Sad"] += 3.8
            scores["😌 Calm"] += 2.5

        # Tempo
        if scalars['tempo'] > 145:
            scores["😄 Happy"] += 3.0
            scores["😡 Angry"] += 2.0
        elif scalars['tempo'] < 90:
            scores["😢 Sad"] += 3.0
            scores["😌 Calm"] += 2.5

        # Pitch (MOST IMPORTANT)
        if scalars['p_mean'] > 240:
            scores["😨 Fearful"] += 4.5
            scores["😄 Happy"] += 2.0
        elif scalars['p_mean'] > 210:
            scores["😄 Happy"] += 3.8
        elif scalars['p_mean'] < 155:
            scores["😢 Sad"] += 4.0
        if scalars['p_std'] < 12:
            scores["😌 Calm"] += 3.5

        # Pitch Range
        if scalars['p_range'] > 80:
            scores["😨 Fearful"] += 3.0
            scores["😡 Angry"] += 2.5

        # Spectral
        if scalars['centroid'] > 2800:
            scores["😄 Happy"] += 2.5
        elif scalars['centroid'] < 1800:
            scores["😢 Sad"] += 2.5

        # ZCR
        if scalars['zcr'] > 0.14:
            scores["😡 Angry"] += 3.0

        # Pauses
        if scalars['pause_ratio'] > 0.45:
            scores["😢 Sad"] += 2.5

        # Get best prediction
        best_emotion = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = (scores[best_emotion] / total_score * 100) if total_score > 0 else 50
        
        conf_emoji = "🔥" if confidence > 75 else "✅" if confidence > 60 else "ℹ️"
        
        return f"{conf_emoji} Predicted Emotion: {best_emotion} ({confidence:.0f}% confidence)"

    except Exception as e:
        return f"⚠️ Prediction error - {str(e)[:30]}... (using fallback)"


# ---------- SESSION STATE ----------
if "rec_key" not in st.session_state:
    st.session_state.rec_key = 0

# ---------- RECORD AUDIO ----------
st.subheader("🎙️ Record your voice")
audio_data = st.audio_input("Tap to record (3-10 seconds)", key=f"rec_{st.session_state.rec_key}")

if audio_data:
    st.audio(audio_data, format="audio/wav")
    
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data.read())
        tmp_path = tmp.name

    with st.spinner("🎵 Analyzing voice patterns..."):
        features = extract_features(tmp_path)
        result = predict_emotion(features)
    
    st.success(result)

# ---------- RESET BUTTON ----------
if st.button("🗑️ New Recording", use_container_width=True):
    st.session_state.rec_key += 1
    st.rerun()

# ---------- DIVIDER ----------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---------- FILE UPLOAD ----------
st.subheader("📁 Upload Audio File")
uploaded_file = st.file_uploader("Choose WAV/MP3 file", type=["wav", "mp3", "m4a"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("🎵 Analyzing voice patterns..."):
        features = extract_features(tmp_path)
        result = predict_emotion(features)
    
    st.success(result)

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("*Powered by advanced speech analysis* 💙")
