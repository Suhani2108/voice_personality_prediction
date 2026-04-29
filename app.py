import streamlit as st
import numpy as np
import librosa
import tempfile
from sklearn.preprocessing import MinMaxScaler

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


# ---------- ROBUST FEATURE EXTRACTION ----------
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=10, sr=22050)
        audio, _ = librosa.effects.trim(audio, top_db=20)

        if len(audio) < sr * 0.5:
            return None

        # Safe feature extraction with defaults
        def safe_mean(arr, default=0):
            return np.mean(arr) if len(arr) > 0 else default
        
        def safe_std(arr, default=0):
            return np.std(arr) if len(arr) > 1 else default
        
        def safe_ptp(arr, default=0):
            return np.ptp(arr) if len(arr) > 1 else default

        # Energy features
        rms = librosa.feature.rms(y=audio)[0]
        energy_mean = safe_mean(rms, 0.05)
        energy_std = safe_std(rms, 0.02)
        energy_peak = np.max(rms) if len(rms) > 0 else 0.1

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = safe_mean(zcr, 0.08)

        # Tempo (safe fallback)
        try:
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        except:
            tempo, beats = 120.0, np.array([])
        tempo = tempo if tempo > 0 else 120.0

        # Pitch analysis (robust)
        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                frame_length=2048
            )
            f0 = f0[voiced_flag]
        except:
            f0 = np.array([])
        
        pitch_mean = safe_mean(f0, 180.0)
        pitch_std = safe_std(f0, 20.0)
        pitch_range = safe_ptp(f0, 30.0)

        # Spectral features
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        centroid_mean = safe_mean(centroid, 2000.0)
        
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        bandwidth_mean = safe_mean(bandwidth, 1500.0)
        
        # MFCCs (safe)
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            mfcc1 = mfcc_means[0] if len(mfcc_means) > 0 else -300.0
            mfcc2 = mfcc_means[1] if len(mfcc_means) > 1 else 100.0
        except:
            mfcc1, mfcc2 = -300.0, 100.0

        # Formants (safe fallback)
        try:
            formants = librosa.lpc(audio, order=12)
            formant_freqs = np.abs(np.roots(formants)[-6:]).real[:3]
            formant_freqs = formant_freqs[formant_freqs < 5000]  # Filter invalid
            formant_mean = safe_mean(formant_freqs, 1500.0)
        except:
            formant_mean = 1500.0

        # Speaking rate & pauses
        speaking_rate = len(beats) / max(len(audio) / sr, 1) * 60 if len(beats) > 0 else 120.0
        pause_ratio = 1 - (np.sum(rms > 0.01) / max(len(rms), 1))

        return {
            "energy_mean": max(0.001, energy_mean),
            "energy_std": max(0.001, energy_std),
            "energy_peak": max(0.001, energy_peak),
            "zcr_mean": max(0.001, zcr_mean),
            "tempo": max(40.0, min(220.0, tempo)),
            "pitch_mean": max(75.0, min(500.0, pitch_mean)),
            "pitch_std": max(0.1, pitch_std),
            "pitch_range": max(0.1, pitch_range),
            "centroid_mean": max(500.0, min(8000.0, centroid_mean)),
            "bandwidth_mean": max(200.0, bandwidth_mean),
            "mfcc1": mfcc1,
            "mfcc2": mfcc2,
            "formant_mean": max(500.0, min(4000.0, formant_mean)),
            "speaking_rate": max(60.0, min(300.0, speaking_rate)),
            "pause_ratio": max(0.0, min(1.0, pause_ratio))
        }

    except Exception as ex:
        st.error(f"Feature extraction error: {str(ex)}")
        return None


# ---------- BULLETPROOF EMOTION CLASSIFIER ----------
def predict_emotion(f):
    if f is None:
        return "⚠️ Audio too short — please record at least 1 second"

    try:
        # Safe feature extraction
        features = np.array([
            f.get("energy_mean", 0.05),
            f.get("energy_std", 0.02),
            f.get("energy_peak", 0.1),
            f.get("zcr_mean", 0.08),
            f.get("tempo", 120.0),
            f.get("pitch_mean", 180.0),
            f.get("pitch_std", 20.0),
            f.get("pitch_range", 30.0),
            f.get("centroid_mean", 2000.0),
            f.get("bandwidth_mean", 1500.0),
            f.get("mfcc1", -300.0),
            f.get("mfcc2", 100.0),
            f.get("formant_mean", 1500.0),
            f.get("speaking_rate", 120.0),
            f.get("pause_ratio", 0.3)
        ])

        scores = {
            "😄 Happy": 0.0,
            "😢 Sad": 0.0,
            "😡 Angry": 0.0,
            "😌 Calm": 0.0,
            "😐 Neutral": 0.0,
            "😨 Fearful": 0.0
        }

        # Research-based emotion rules (SAFE)
        e_mean, e_std, e_peak = features[0], features[1], features[2]
        zcr, tempo, p_mean = features[3], features[4], features[5]
        p_std, p_range = features[6], features[7]
        centroid, mfcc1 = features[8], features[10]
        pause_ratio = features[14]

        # 1. ENERGY
        if e_mean > 0.15 and e_std > 0.08:
            scores["😡 Angry"] += 3.5
            scores["😄 Happy"] += 1.8
        elif e_mean > 0.12:
            scores["😄 Happy"] += 3.0
        elif e_mean < 0.035:
            scores["😢 Sad"] += 3.2
            scores["😌 Calm"] += 2.0
        else:
            scores["😐 Neutral"] += 2.5

        # 2. TEMPO
        if tempo > 145:
            scores["😄 Happy"] += 2.8
            scores["😡 Angry"] += 1.5
        elif tempo < 85:
            scores["😢 Sad"] += 2.8
            scores["😌 Calm"] += 2.2

        # 3. PITCH
        if p_mean > 240 and p_std > 45:
            scores["😨 Fearful"] += 3.8
        elif p_mean > 210:
            scores["😄 Happy"] += 3.2
        elif p_mean < 155:
            scores["😢 Sad"] += 3.5
        elif p_std < 12:
            scores["😌 Calm"] += 3.0

        # 4. PITCH RANGE
        if p_range > 80:
            scores["😨 Fearful"] += 2.5
            scores["😡 Angry"] += 2.0
        elif p_range < 25:
            scores["😌 Calm"] += 2.8

        # 5. SPECTRAL
        if centroid > 2800:
            scores["😄 Happy"] += 2.2
        elif centroid < 1800:
            scores["😢 Sad"] += 2.5

        # 6. ZCR
        if zcr > 0.14:
            scores["😡 Angry"] += 2.8
        elif zcr < 0.045:
            scores["😌 Calm"] += 2.2

        # 7. PAUSES
        if pause_ratio > 0.45:
            scores["😢 Sad"] += 2.2

        best_emotion = max(scores, key=scores.get)
        confidence = (max(scores.values()) / sum(scores.values())) * 100
        confidence_emoji = "🔥" if confidence > 75 else "✅" if confidence > 60 else "🤔"

        return f"{confidence_emoji} Predicted Emotion: {best_emotion} (Confidence: {confidence:.0f}%)"

    except Exception as e:
        return f"⚠️ Analysis error: {str(e)[:50]}... Please try again"


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

    with st.spinner("🔬 Analyzing vocal patterns..."):
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

    with st.spinner("🔬 Analyzing vocal patterns..."):
        feats = extract_features(path)
        prediction = predict_emotion(feats)

    st.success(prediction)
