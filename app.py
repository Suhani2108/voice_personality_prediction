import streamlit as st
import numpy as np
import tempfile
from tensorflow.keras.models import load_model
from feature_extraction import extract_features

model = load_model("emotion_model.h5")

st.title("🎤 Voice Personality Predictor")

uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])

if uploaded_file is not None:
    # temp file create
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.audio(uploaded_file)  # play audio

    # feature extraction
    features = extract_features(temp_path)
    features = np.expand_dims(features, axis=0)

    prediction = model.predict(features)
    pred = np.argmax(prediction)

    st.success(f"Prediction: {pred}")
