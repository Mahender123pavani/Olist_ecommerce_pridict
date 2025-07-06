import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Page config
st.set_page_config(page_title="Olist eCommerce Prediction", page_icon="🛍")

st.title("🛍 Olist eCommerce Prediction App")
st.markdown("Predict Review Quality (Classification) and Freight Value (Regression) using product data.")

# Required files
required_files = ["scaler.pkl", "classification_model.pkl", "regression_model.pkl"]
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error(f"❌ Missing files: {', '.join(missing_files)}")
    st.stop()

# Load files
scaler = joblib.load("scaler.pkl")
clf_model = joblib.load("classification_model.pkl")
reg_model = joblib.load("regression_model.pkl")

# Input fields
st.header("📦 Enter Product Details")
price = st.number_input("Price (R$)", min_value=0.0, value=100.0)
weight = st.number_input("Product Weight (g)", min_value=0.0, value=500.0)
review_score = st.slider("Review Score (1–5)", min_value=1, max_value=5, value=3)

# Predict button
if st.button("🚀 Predict"):
    try:
        # Input DataFrame with matching feature names
        input_df = pd.DataFrame([[price, weight, review_score]],
                                columns=["price", "weight", "review_score"])

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Predict classification and regression
        review_prediction = clf_model.predict(scaled_input)[0]
        freight_prediction = reg_model.predict(scaled_input)[0]

        # Show results
        st.success(f"⭐ Predicted Review Quality: {'Good' if review_prediction == 1 else 'Bad'}")
        st.success(f"🚚 Predicted Freight Value: R$ {freight_prediction:.2f}")

    except Exception as e:
        st.error(f"❗ Error during prediction: {e}")
