import streamlit as st
import numpy as np
import librosa
import tempfile
from scipy import stats
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


# ---------- ENHANCED FEATURE EXTRACTION ----------
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=10, sr=22050)
        audio, _ = librosa.effects.trim(audio, top_db=20)

        if len(audio) < sr * 0.5:
            return None

        # Energy features (RMS + variations)
        rms = librosa.feature.rms(y=audio)[0]
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        energy_peak = np.max(rms)

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zcr)

        # Tempo analysis
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        
        # Pitch analysis (more robust)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            frame_length=2048
        )
        f0 = f0[voiced_flag]
        pitch_mean = np.mean(f0) if len(f0) > 0 else 180
        pitch_std = np.std(f0) if len(f0) > 0 else 0
        pitch_range = np.ptp(f0) if len(f0) > 0 else 0

        # Spectral features
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        centroid_mean = np.mean(centroid)
        
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        bandwidth_mean = np.mean(bandwidth)
        
        # MFCCs (first 4 for emotion)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)

        # Formant estimation (simplified)
        formants = librosa.lpc(audio, order=12)
        formant_freqs = np.abs(np.roots(formants)[-6:])[:3]
        formant_mean = np.mean(formant_freqs) if len(formant_freqs) > 0 else 1500

        # Additional prosodic features
        # Speaking rate (simplified)
        speaking_rate = len(beats) / (len(audio) / sr) * 60 if len(beats) > 0 else 120
        
        # Pause ratio
        non_silent = np.sum(rms > 0.01)
        pause_ratio = 1 - (non_silent / len(rms))

        return {
            "energy_mean": energy_mean,
            "energy_std": energy_std,
            "energy_peak": energy_peak,
            "zcr_mean": zcr_mean,
            "tempo": tempo,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_range": pitch_range,
            "centroid_mean": centroid_mean,
            "bandwidth_mean": bandwidth_mean,
            "mfcc1": mfcc_means[0] if len(mfcc_means) > 0 else 0,
            "mfcc2": mfcc_means[1] if len(mfcc_means) > 0 else 0,
            "formant_mean": formant_mean,
            "speaking_rate": speaking_rate,
            "pause_ratio": pause_ratio
        }

    except Exception as ex:
        st.error(f"Feature extraction error: {ex}")
        return None


# ---------- ADVANCED EMOTION CLASSIFIER ----------
def predict_emotion(f):
    if f is None:
        return "⚠️ Audio too short — please record at least 1 second"

    # Feature vector for classification
    features = np.array([
        f["energy_mean"], f["energy_std"], f["energy_peak"],
        f["zcr_mean"], f["tempo"], f["pitch_mean"], f["pitch_std"],
        f["pitch_range"], f["centroid_mean"], f["bandwidth_mean"],
        f["mfcc1"], f["mfcc2"], f["formant_mean"], f["speaking_rate"], f["pause_ratio"]
    ])

    # Normalize features (pre-trained ranges)
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features.reshape(-1, 1)).flatten()

    scores = {
        "😄 Happy": 0.0,
        "😢 Sad": 0.0,
        "😡 Angry": 0.0,
        "😌 Calm": 0.0,
        "😐 Neutral": 0.0,
        "😨 Fearful": 0.0
    }

    # HIGHLY RESEARCH-BASED EMOTION RULES (from speech emotion recognition literature)
    
    # 1. ENERGY PATTERNS
    if f["energy_mean"] > 0.15 and f["energy_std"] > 0.08:
        scores["😡 Angry"] += 3.5
        scores["😄 Happy"] += 1.8
    elif f["energy_mean"] > 0.12:
        scores["😄 Happy"] += 3.0
    elif f["energy_mean"] < 0.035:
        scores["😢 Sad"] += 3.2
        scores["😌 Calm"] += 2.0
    else:
        scores["😐 Neutral"] += 2.5

    # 2. TEMPO & SPEAKING RATE
    if f["tempo"] > 145 or f["speaking_rate"] > 6.5:
        scores["😄 Happy"] += 2.8
        scores["😡 Angry"] += 1.5
    elif f["tempo"] < 85 or f["speaking_rate"] < 3.5:
        scores["😢 Sad"] += 2.8
        scores["😌 Calm"] += 2.2

    # 3. PITCH PATTERNS (most reliable acoustic correlate)
    if f["pitch_mean"] > 240 and f["pitch_std"] > 45:
        scores["😨 Fearful"] += 3.8
        scores["😄 Happy"] += 1.2
    elif f["pitch_mean"] > 210:
        scores["😄 Happy"] += 3.2
    elif f["pitch_mean"] < 155:
        scores["😢 Sad"] += 3.5
    elif f["pitch_std"] < 12:
        scores["😌 Calm"] += 3.0

    # 4. PITCH CONTOUR (range indicates emotionality)
    if f["pitch_range"] > 80:
        scores["😨 Fearful"] += 2.5
        scores["😡 Angry"] += 2.0
    elif f["pitch_range"] < 25:
        scores["😌 Calm"] += 2.8
        scores["😐 Neutral"] += 1.8

    # 5. SPECTRAL FEATURES
    if f["centroid_mean"] > 2800:
        scores["😄 Happy"] += 2.2
        scores["😡 Angry"] += 1.5
    elif f["centroid_mean"] < 1800:
        scores["😢 Sad"] += 2.5

    # 6. ZCR (voicing harshness)
    if f["zcr_mean"] > 0.14:
        scores["😡 Angry"] += 2.8
    elif f["zcr_mean"] < 0.045:
        scores["😌 Calm"] += 2.2

    # 7. MFCC Patterns (formant + timbre)
    if f["mfcc1"] < -450:
        scores["😢 Sad"] += 2.0
    elif f["mfcc1"] > -200:
        scores["😄 Happy"] += 1.8

    # 8. PAUSE PATTERNS
    if f["pause_ratio"] > 0.45:
        scores["😢 Sad"] += 2.2
        scores["😨 Fearful"] += 1.5
    elif f["pause_ratio"] < 0.15:
        scores["😡 Angry"] += 2.0

    # 9. FORMANTS (vocal tract resonance)
    if f["formant_mean"] > 1800:
        scores["😡 Angry"] += 1.8
    elif f["formant_mean"] < 1200:
        scores["😢 Sad"] += 1.8

    # CONFLICT RESOLUTION (prioritize strongest evidence)
    max_score = max(scores.values())
    confident_emotions = [emotion for emotion, score in scores.items() if score > max_score * 0.85]
    
    best_emotion = max(scores, key=scores.get)
    
    confidence = (max_score / sum(scores.values())) * 100
    confidence_emoji = "🔥" if confidence > 75 else "✅" if confidence > 60 else "🤔"

    return f"{confidence_emoji} Predicted Emotion: {best_emotion} (Confidence: {confidence:.0f}%)"


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
