import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# FIXED: correct feature size (same as app)
X = np.random.rand(300, 22)
y = np.random.choice(["happy", "sad", "angry", "neutral"], 300)

model = RandomForestClassifier(n_estimators=200)
model.fit(X, y)

joblib.dump(model, "emotion_model.pkl")

print("✅ Fresh model created successfully")
