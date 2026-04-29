import pickle
import joblib

# load objects from notebook
with open("temp_objects.pkl", "rb") as f:
    model, scaler, encoder = pickle.load(f)

# save properly
model.save("model.h5")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")

print("✅ Export done")
