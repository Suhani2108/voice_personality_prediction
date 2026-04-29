import streamlit as st
import tempfile
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# ---------- LOAD MODEL (AUTO DOWNLOAD) ----------
@st.cache_resource
def load_model():
    return EncoderClassifier.from_hparams(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        savedir="tmp_model"
    )

classifier = load_model()

# ---------- UI ----------
st.set_page_config(page_title="Voice Personality AI", page_icon="🎤", layout="centered")

st.markdown('<div class="title">🎤 Voice Personality AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speak or upload audio to detect emotion</div>', unsafe_allow_html=True)

# ---------- PREDICTION ----------
def predict_emotion(file_path):
    signal, fs = torchaudio.load(file_path)
    out_prob, score, index, text_lab = classifier.classify_batch(signal)

    emotion = text_lab[0]
    confidence = float(score[0]) * 100

    return f"✨ Predicted Emotion: {emotion} (Confidence: {confidence:.2f}%)"

# ---------- RECORD ----------
st.subheader("🎙️ Record your voice")
audio_data = st.audio_input("Tap to record")

if audio_data:
    st.audio(audio_data)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data.read())
        path = tmp.name

    with st.spinner("Analyzing..."):
        result = predict_emotion(path)

    st.success(result)

# ---------- UPLOAD ----------
st.subheader("📂 Upload Audio File")
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    with st.spinner("Analyzing..."):
        result = predict_emotion(path)

    st.success(result)
