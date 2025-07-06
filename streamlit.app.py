import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Olist eCommerce Prediction", page_icon="🧾")

st.title("🧾 Olist eCommerce Prediction App")
st.markdown("Predict Review Quality (Classification) and Freight Value (Regression) using product data.")

# Check if all required .pkl files exist
required_files = ["scaler.pkl", "classification_model.pkl", "regression_model.pkl"]
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error(f"❌ File not found: {', '.join(missing_files)}\n\nPlease ensure all .pkl files are in the same directory as this app.")
    st.stop()

# Load models and scaler
scaler = joblib.load("scaler.pkl")
clf_model = joblib.load("classification_model.pkl")
reg_model = joblib.load("regression_model.pkl")

# Input fields
st.header("📦 Enter Product Details")
price = st.number_input("Price", min_value=0.0, format="%.2f")
weight = st.number_input("Product Weight (g)", min_value=0.0, format="%.2f")
review_score = st.slider("Review Score (1 to 5)", min_value=1, max_value=5, value=3)

# Prepare input data
input_data = pd.DataFrame({
    "price": [price],
    "weight": [weight],
    "review_score": [review_score]
})

# Predict button
if st.button("🚀 Predict"):
    try:
        # Scale input
        scaled_input = scaler.transform(input_data)

        # Predict classification (e.g., High/Low review quality)
