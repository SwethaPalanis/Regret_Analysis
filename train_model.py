import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "regret_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "regret_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
df = df.dropna()  # remove missing values

X = df.drop("regret_level", axis=1)
y = df["regret_level"]

# Encode categorical columns
categorical_cols = ["decision_type", "external_pressure", "past_regret_experience",
                    "emotional_state", "decision_urgency"]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model + encoders using joblib
joblib.dump((model, encoders), MODEL_PATH)
print(f"✅ Model + encoders saved at {MODEL_PATH}")
