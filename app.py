import streamlit as st
import pandas as pd
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "regret_model.pkl")

# Load model + encoders
model, encoders = joblib.load(MODEL_PATH)

# Page Layout
st.set_page_config(page_title="Regret Analysis System", layout="centered")
st.title("😕 Regret Analysis System")
st.markdown(
    "Predict your regret level based on your recent decision.\n\n"
    "Fill in the details below and click **Predict**."
)
st.write("---")

# User Inputs
st.header("Decision Details")
user_input = {}
for col, le in encoders.items():
    user_input[col] = st.selectbox(
        col.replace("_", " ").title(),
        le.classes_
    )

# Predict Button
if st.button("🔮 Predict Regret"):
    # Encode input
    input_encoded = {col: encoders[col].transform([val])[0] for col, val in user_input.items()}
    input_df = pd.DataFrame([input_encoded])
    
    # Predict
    result = model.predict(input_df)[0]
    
    # Color-coded output
    color = {"High": "red", "Medium": "orange", "Low": "green"}.get(result, "black")
    st.markdown(f"### Predicted Regret Level: <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)
    
    # Advice message
    advice = {
        "High": "⚠️ Consider reflecting carefully before making similar decisions in the future.",
        "Medium": "🤔 Some caution is advised.",
        "Low": "✅ You are on track; minimal regret expected."
    }.get(result, "")
    
    st.info(advice)
