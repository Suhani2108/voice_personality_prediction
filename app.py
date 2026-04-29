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


# ---------- RICH FEATURE EXTRACTION ----------
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=6, sr=22050)

    # Trim leading/trailing silence
    audio, _ = librosa.effects.trim(audio, top_db=20)

    if len(audio) < sr * 0.3:
        return None  # Too short to analyze

    # --- Pitch (F0) via PYIN for better accuracy ---
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )
    f0_valid = f0[voiced_flag & ~np.isnan(f0)]
    mean_pitch = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0
    pitch_std = float(np.std(f0_valid)) if len(f0_valid) > 0 else 0.0
    voiced_ratio = float(np.sum(voiced_flag) / max(len(voiced_flag), 1))

    # --- Energy & dynamics ---
    rms = librosa.feature.rms(y=audio)[0]
    mean_energy = float(np.mean(rms))
    energy_std = float(np.std(rms))
    # Dynamic range: ratio of high to low energy frames
    energy_range = float(np.percentile(rms, 90) - np.percentile(rms, 10))

    # --- Tempo / rhythm ---
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo = float(np.squeeze(tempo))

    # --- Zero Crossing Rate ---
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    mean_zcr = float(np.mean(zcr))

    # --- MFCCs (13 coefficients) ---
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)

    # --- Spectral features ---
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)))
    spectral_flux = float(np.mean(np.diff(librosa.feature.melspectrogram(y=audio, sr=sr), axis=1) ** 2))
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    mean_contrast = float(np.mean(spectral_contrast))

    # --- Chroma features ---
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mean_chroma = float(np.mean(chroma))
    chroma_std = float(np.std(chroma))

    # --- Speech rate proxy (syllable-like energy peaks) ---
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    duration_sec = len(audio) / sr
    speech_rate = float(len(onsets) / max(duration_sec, 0.1))

    return {
        "mean_pitch": mean_pitch,
        "pitch_std": pitch_std,
        "voiced_ratio": voiced_ratio,
        "mean_energy": mean_energy,
        "energy_std": energy_std,
        "energy_range": energy_range,
        "tempo": tempo,
        "mean_zcr": mean_zcr,
        "mfcc_means": mfcc_means,
        "mfcc_stds": mfcc_stds,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_rolloff": spectral_rolloff,
        "spectral_flux": spectral_flux,
        "mean_contrast": mean_contrast,
        "mean_chroma": mean_chroma,
        "chroma_std": chroma_std,
        "speech_rate": speech_rate,
        "duration": duration_sec,
    }


# ---------- SCORING-BASED EMOTION PREDICTOR ----------
def predict_emotion(feats):
    if feats is None:
        return "⚠️ Audio too short — please record longer"

    p = feats["mean_pitch"]
    p_std = feats["pitch_std"]
    e = feats["mean_energy"]
    e_std = feats["energy_std"]
    e_range = feats["energy_range"]
    zcr = feats["mean_zcr"]
    tempo = feats["tempo"]
    sc = feats["spectral_centroid"]
    sb = feats["spectral_bandwidth"]
    sr_rate = feats["speech_rate"]
    flux = feats["spectral_flux"]
    voiced = feats["voiced_ratio"]
    mfcc = feats["mfcc_means"]
    contrast = feats["mean_contrast"]

    scores = {
        "😄 Happy": 0.0,
        "😡 Angry": 0.0,
        "😢 Sad": 0.0,
        "😨 Fearful": 0.0,
        "😲 Surprised": 0.0,
        "😐 Neutral": 0.0,
        "😌 Calm": 0.0,
        "🤢 Disgusted": 0.0,
    }

    # ── HAPPY: moderate-high pitch, high energy, high tempo, wide pitch variation
    if 160 < p < 280:
        scores["😄 Happy"] += 2.0
    if e > 0.06:
        scores["😄 Happy"] += 1.5
    if p_std > 30:
        scores["😄 Happy"] += 1.5
    if tempo > 110:
        scores["😄 Happy"] += 1.5
    if sr_rate > 4:
        scores["😄 Happy"] += 1.0
    if sc > 2000:
        scores["😄 Happy"] += 1.0
    if mfcc[1] > 10:
        scores["😄 Happy"] += 1.0

    # ── ANGRY: very high energy, high ZCR, high spectral flux, fast tempo, harsh spectral
    if e > 0.12:
        scores["😡 Angry"] += 3.0
    elif e > 0.08:
        scores["😡 Angry"] += 1.5
    if zcr > 0.08:
        scores["😡 Angry"] += 2.0
    if flux > 0.01:
        scores["😡 Angry"] += 2.0
    if tempo > 130:
        scores["😡 Angry"] += 1.5
    if sc > 3000:
        scores["😡 Angry"] += 1.5
    if p_std > 40 and e > 0.1:
        scores["😡 Angry"] += 1.5
    if e_range > 0.1:
        scores["😡 Angry"] += 1.0
    if sr_rate > 5:
        scores["😡 Angry"] += 1.0

    # ── SAD: low pitch, very low energy, slow tempo, low voiced ratio, narrow pitch range
    if e < 0.04:
        scores["😢 Sad"] += 3.0
    elif e < 0.065:
        scores["😢 Sad"] += 1.5
    if p < 140 and p > 0:
        scores["😢 Sad"] += 2.0
    if tempo < 85:
        scores["😢 Sad"] += 2.0
    if p_std < 20:
        scores["😢 Sad"] += 1.5
    if sr_rate < 3:
        scores["😢 Sad"] += 1.5
    if voiced < 0.4:
        scores["😢 Sad"] += 1.0
    if mfcc[1] < -10:
        scores["😢 Sad"] += 1.0
    if e_range < 0.03:
        scores["😢 Sad"] += 1.0

    # ── FEARFUL: high pitch, moderate energy, high flux/zcr, irregular rhythm
    if p > 220:
        scores["😨 Fearful"] += 2.0
    if 0.04 < e < 0.11:
        scores["😨 Fearful"] += 1.0
    if flux > 0.008:
        scores["😨 Fearful"] += 2.0
    if zcr > 0.07 and e < 0.1:
        scores["😨 Fearful"] += 1.5
    if p_std > 45:
        scores["😨 Fearful"] += 1.5
    if sr_rate > 4.5 and e < 0.1:
        scores["😨 Fearful"] += 1.0
    if sb > 2500:
        scores["😨 Fearful"] += 1.0

    # ── SURPRISED: very high pitch, sudden energy burst, wide spectral range
    if p > 250:
        scores["😲 Surprised"] += 2.5
    if e_std > 0.05:
        scores["😲 Surprised"] += 2.0
    if sc > 3500:
        scores["😲 Surprised"] += 2.0
    if p_std > 55:
        scores["😲 Surprised"] += 2.0
    if tempo > 140 or (sr_rate > 5.5):
        scores["😲 Surprised"] += 1.0
    if sb > 3000:
        scores["😲 Surprised"] += 1.0

    # ── NEUTRAL: moderate everything, stable pitch, average energy
    if 0.04 < e < 0.09:
        scores["😐 Neutral"] += 1.5
    if 120 < p < 200:
        scores["😐 Neutral"] += 1.5
    if p_std < 30:
        scores["😐 Neutral"] += 1.5
    if 80 < tempo < 120:
        scores["😐 Neutral"] += 1.5
    if 0.03 < zcr < 0.07:
        scores["😐 Neutral"] += 1.0
    if 2 < sr_rate < 5:
        scores["😐 Neutral"] += 1.0
    if flux < 0.005:
        scores["😐 Neutral"] += 1.0

    # ── CALM: low energy but voiced, slow tempo, low ZCR, low flux, smooth pitch
    if 0.03 < e < 0.07:
        scores["😌 Calm"] += 2.0
    if tempo < 95:
        scores["😌 Calm"] += 2.0
    if zcr < 0.05:
        scores["😌 Calm"] += 1.5
    if flux < 0.004:
        scores["😌 Calm"] += 1.5
    if p_std < 25 and p > 0:
        scores["😌 Calm"] += 1.5
    if voiced > 0.5:
        scores["😌 Calm"] += 1.0
    if sr_rate < 4:
        scores["😌 Calm"] += 1.0

    # ── DISGUSTED: low-mid pitch, moderate energy, harsh spectral, low tempo
    if 100 < p < 170:
        scores["🤢 Disgusted"] += 1.5
    if 0.05 < e < 0.12:
        scores["🤢 Disgusted"] += 1.5
    if contrast < 15:
        scores["🤢 Disgusted"] += 2.0
    if tempo < 100:
        scores["🤢 Disgusted"] += 1.0
    if zcr > 0.06 and e < 0.1:
        scores["🤢 Disgusted"] += 1.0
    if mfcc[2] < -5:
        scores["🤢 Disgusted"] += 1.0

    # Resolve ties: boost dominant emotion slightly using MFCC signature
    # MFCC[0] energy-related, MFCC[1] brightness, MFCC[2] sharpness
    top_emotion = max(scores, key=scores.get)
    top_score = scores[top_emotion]

    # If no clear winner (all low), fallback to neutral
    if top_score < 2.0:
        return "😐 Neutral"

    return top_emotion


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

    st.success(f"✨ Predicted Emotion: {prediction}")

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

    st.success(f"✨ Predicted Emotion: {prediction}")
