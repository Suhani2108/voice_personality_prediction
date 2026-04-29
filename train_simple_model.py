import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "ravdess"  # your dataset folder

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    rms = np.mean(librosa.feature.rms(y=audio))

    return np.hstack([mfcc_mean, zcr, rms])

X = []
y = []

for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)

            emotion = file.split("-")[2]
            mapping = {
                "01":"neutral","02":"calm","03":"happy","04":"sad",
                "05":"angry","06":"fearful","07":"disgust","08":"surprised"
            }

            X.append(extract_features(path))
            y.append(mapping.get(emotion, "neutral"))

model = RandomForestClassifier(n_estimators=200)
model.fit(X, y)

joblib.dump(model, "emotion_model.pkl")

print("✅ Model saved")
