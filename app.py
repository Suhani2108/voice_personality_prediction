# ---------- SESSION STATE ----------
if "rec_key" not in st.session_state:
    st.session_state.rec_key = 0

# ---------- RECORD ----------
st.subheader("🎙️ Record your voice")

audio_data = st.audio_input(
    "Tap to record",
    key=f"rec_{st.session_state.rec_key}"
)

if audio_data:
    st.audio(audio_data)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data.read())
        path = tmp.name

    energy, pitch = extract_features(path)
    prediction = predict_emotion(energy, pitch)

    st.success(f"✨ Predicted Emotion: {prediction}")

# ---------- DELETE BUTTON ----------
if st.button("🗑️ Delete Recording"):
    st.session_state.rec_key += 1   # 🔥 resets recorder
    st.rerun()
