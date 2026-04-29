import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

X = np.random.rand(200, 22)
y = np.random.choice(["happy", "sad", "angry", "neutral"], 200)

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

joblib.dump(model, "emotion_model.pkl")

print("✅ Model created")
