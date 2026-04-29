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
def extract_features(audio_bytes):
    """Write bytes to a fresh temp file, extract features, clean up."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        audio, sr = librosa.load(tmp_path, sr=22050, duration=6)
        audio, _ = librosa.effects.trim(audio, top_db=25)

        if len(audio) < int(sr * 0.5):
            return None

        # MFCC
        mfcc      = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).astype(float)
        mfcc_std  = np.std(mfcc,  axis=1).astype(float)

        # Pitch
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        if voiced_flag is not None and len(f0) > 0:
            f0_clean = f0[voiced_flag.astype(bool)]
            f0_clean = f0_clean[~np.isnan(f0_clean)]
        else:
            f0_clean = np.array([])

        pitch_mean  = float(np.mean(f0_clean))  if len(f0_clean) > 0 else 150.0
        pitch_std   = float(np.std(f0_clean))   if len(f0_clean) > 0 else 10.0
        voiced_frac = float(len(f0_clean)) / float(max(len(f0), 1))

        # Energy
        rms_frames = librosa.feature.rms(y=audio)[0]
        rms_mean   = float(np.mean(rms_frames))
        rms_std    = float(np.std(rms_frames))

        # ZCR
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))

        # Tempo
        tempo_result, _ = librosa.beat.beat_track(y=audio, sr=sr)
        tempo = float(tempo_result) if np.ndim(tempo_result) == 0 else float(np.mean(tempo_result))

        # Spectral
        centroid  = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
        flatness  = float(np.mean(librosa.feature.spectral_flatness(y=audio)))

        # Spectral Contrast
        contrast      = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
        contrast_mean = np.mean(contrast, axis=1).astype(float)

        # Chroma
        chroma     = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_std = float(np.std(chroma))

        return {
            "mfcc_mean":   mfcc_mean,
            "mfcc_std":    mfcc_std,
            "pitch_mean":  pitch_mean,
            "pitch_std":   pitch_std,
            "voiced_frac": voiced_frac,
            "rms_mean":    rms_mean,
            "rms_std":     rms_std,
            "zcr":         zcr,
            "tempo":       tempo,
            "centroid":    centroid,
            "bandwidth":   bandwidth,
            "flatness":    flatness,
            "contrast":    contrast_mean,
            "chroma_std":  chroma_std,
        }

    except Exception as ex:
        st.error(f"Feature extraction failed: {ex}")
        return None

    finally:
        # Always clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ---------- EMOTION LOGIC ----------
def predict_emotion(f):
    if f is None:
        return "⚠️ Audio too short or unreadable", {}

    mfcc_m   = f["mfcc_mean"]
    mfcc_s   = f["mfcc_std"]
    pm       = f["pitch_mean"]
    ps       = f["pitch_std"]
    vf       = f["voiced_frac"]
    rms      = f["rms_mean"]
    rms_std  = f["rms_std"]
    zcr      = f["zcr"]
    tempo    = f["tempo"]
    centroid = f["centroid"]
    bw       = f["bandwidth"]
    flatness = f["flatness"]
    contrast = f["contrast"]
    chroma_s = f["chroma_std"]

    scores = {
        "😡 Angry":   0.0,
        "😢 Sad":     0.0,
        "😄 Happy":   0.0,
        "😌 Calm":    0.0,
        "🤢 Disgust": 0.0,
        "😨 Fearful": 0.0,
    }

    # ANGRY
    if rms > 0.10:                       scores["😡 Angry"] += 3.0
    elif rms > 0.07:                     scores["😡 Angry"] += 1.5
    if zcr > 0.10:                       scores["😡 Angry"] += 2.5
    if bw > 2800:                        scores["😡 Angry"] += 2.0
    elif bw > 2200:                      scores["😡 Angry"] += 1.0
    if tempo > 135:                      scores["😡 Angry"] += 1.5
    if contrast[5] > 22:                 scores["😡 Angry"] += 2.0
    if mfcc_m[1] > 10:                   scores["😡 Angry"] += 1.5
    if rms_std > 0.05:                   scores["😡 Angry"] += 1.0

    # SAD
    if rms < 0.035:                      scores["😢 Sad"] += 3.0
    elif rms < 0.06:                     scores["😢 Sad"] += 1.5
    if 60 < pm < 155:                    scores["😢 Sad"] += 2.5
    if ps < 18:                          scores["😢 Sad"] += 2.0
    if tempo < 80:                       scores["😢 Sad"] += 2.0
    if vf < 0.35:                        scores["😢 Sad"] += 1.5
    if bw < 1800:                        scores["😢 Sad"] += 1.5
    if mfcc_m[0] < -250:                 scores["😢 Sad"] += 1.0

    # HAPPY
    if pm > 210:                         scores["😄 Happy"] += 2.5
    elif pm > 185:                       scores["😄 Happy"] += 1.0
    if tempo > 120:                      scores["😄 Happy"] += 2.0
    if centroid > 2600:                  scores["😄 Happy"] += 2.0
    elif centroid > 2000:                scores["😄 Happy"] += 1.0
    if chroma_s > 0.10:                  scores["😄 Happy"] += 2.0
    if 0.05 < rms < 0.13:               scores["😄 Happy"] += 1.0
    if vf > 0.60:                        scores["😄 Happy"] += 1.0
    if mfcc_m[2] > 8:                    scores["😄 Happy"] += 1.0

    # CALM
    if 0.025 <= rms <= 0.075:            scores["😌 Calm"] += 2.0
    if ps < 22:                          scores["😌 Calm"] += 2.5
    if 70 <= tempo <= 110:               scores["😌 Calm"] += 2.0
    if zcr < 0.055:                      scores["😌 Calm"] += 2.0
    if flatness < 0.008:                 scores["😌 Calm"] += 1.5
    if vf > 0.55:                        scores["😌 Calm"] += 1.0
    if rms_std < 0.03:                   scores["😌 Calm"] += 1.5

    # DISGUST
    if pm < 175 and ps > 28:             scores["🤢 Disgust"] += 2.5
    if contrast[0] < 8:                  scores["🤢 Disgust"] += 2.0
    if centroid < 1900:                  scores["🤢 Disgust"] += 1.5
    if mfcc_m[4] < -8:                   scores["🤢 Disgust"] += 1.5
    if rms_std > 0.04 and rms < 0.09:    scores["🤢 Disgust"] += 1.5
    if zcr < 0.07 and flatness > 0.015:  scores["🤢 Disgust"] += 1.5

    # FEARFUL
    if pm > 225:                         scores["😨 Fearful"] += 2.0
    if ps > 55:                          scores["😨 Fearful"] += 3.0
    if flatness > 0.018:                 scores["😨 Fearful"] += 2.0
    if rms_std > 0.045:                  scores["😨 Fearful"] += 2.0
    if zcr > 0.08 and rms < 0.08:       scores["😨 Fearful"] += 2.0
    if vf < 0.42:                        scores["😨 Fearful"] += 1.5
    if mfcc_s[0] > 85:                   scores["😨 Fearful"] += 1.5

    # Final decision
    total = sum(scores.values()) + 1e-9
    best  = max(scores, key=scores.get)
    conf  = int((scores[best] / total) * 100)
    display_scores = {k: round((v / total) * 100, 1) for k, v in scores.items()}

    # Only fall to Neutral if truly no signal at all
    if total < 1.0:
        best = "😐 Neutral"
        conf = 38

    return f"✨ Predicted Emotion: **{best}**  (Confidence: {conf}%)", display_scores


# ---------- SESSION STATE INIT ----------
if "rec_key" not in st.session_state:
    st.session_state.rec_key = 0
if "rec_result" not in st.session_state:
    st.session_state.rec_result = None
if "rec_scores" not in st.session_state:
    st.session_state.rec_scores = {}
if "last_rec_id" not in st.session_state:
    st.session_state.last_rec_id = None

# ---------- RECORD ----------
st.subheader("🎙️ Record your voice")
audio_data = st.audio_input("Tap to record", key=f"rec_{st.session_state.rec_key}")

if audio_data is not None:
    # Use the object id to detect a genuinely new recording
    current_id = id(audio_data)
    if current_id != st.session_state.last_rec_id:
        st.session_state.last_rec_id = current_id
        audio_bytes = audio_data.read()

        st.audio(audio_bytes)

        with st.spinner("Analyzing voice..."):
            feats = extract_features(audio_bytes)
            result, score_map = predict_emotion(feats)

        # Store in session state so result persists across reruns
        st.session_state.rec_result = result
        st.session_state.rec_scores = score_map

    # Always show last result
    if st.session_state.rec_result:
        st.success(st.session_state.rec_result)
        if st.session_state.rec_scores:
            with st.expander("📊 Emotion Score Breakdown"):
                for emotion, pct in sorted(st.session_state.rec_scores.items(), key=lambda x: -x[1]):
                    st.write(f"{emotion}  —  {pct}%")
                    st.progress(min(int(pct), 100))

    if st.button("🗑️ Delete Recording"):
        st.session_state.rec_key += 1
        st.session_state.rec_result = None
        st.session_state.rec_scores = {}
        st.session_state.last_rec_id = None
        st.rerun()

# ---------- DIVIDER ----------
st.markdown("---")

# ---------- UPLOAD ----------
st.subheader("📂 Upload Audio File")
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes)

    with st.spinner("Analyzing voice..."):
        feats = extract_features(audio_bytes)
        result, score_map = predict_emotion(feats)

    st.success(result)

    if score_map:
        with st.expander("📊 Emotion Score Breakdown"):
            for emotion, pct in sorted(score_map.items(), key=lambda x: -x[1]):
                st.write(f"{emotion}  —  {pct}%")
                st.progress(min(int(pct), 100))
