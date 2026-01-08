import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Paths
BASE_DIR = r"C:\regret_project"
DATA_PATH = os.path.join(BASE_DIR, "data", "regret_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "regret_model.pkl")

# Create model folder
os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# 🔥 Remove rows with missing values
df = df.dropna()

# Split features and target
X = df.drop("regret_level", axis=1)
y = df["regret_level"]

# Encode categorical columns
categorical_cols = X.select_dtypes(include=["object"]).columns

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model + encoders
with open(MODEL_PATH, "wb") as f:
    pickle.dump((model, encoders), f)

print("✅ Model trained and saved successfully")
print("📁 Saved at:", MODEL_PATH)
