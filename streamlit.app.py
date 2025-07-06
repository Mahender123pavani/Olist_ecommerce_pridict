import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# --- Page setup ---
st.set_page_config(page_title="Olist Predictor", page_icon="🛍")
st.title("🛍 Olist eCommerce Prediction App")
st.markdown("Predict Review Quality and Freight Value using product details.")

# --- Check for all required files ---
required_files = ["scaler.pkl", "classification_model.pkl", "regression_model.pkl"]
missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    st.error(f"❌ Missing files: {', '.join(missing)}")
    st.stop()

# --- Load models ---
scaler = joblib.load("scaler.pkl")
clf = joblib.load("classification_model.pkl")
reg = joblib.load("regression_model.pkl")

# --- Input fields ---
st.header("📦 Enter Product Details")
price = st.number_input("Price (R$)", min_value=0.0, value=100.0)
weight = st.number_input("Product Weight (g)", min_value=0.0, value=500.0)
review_score = st.slider("Review Score (1 to 5)", min_value=1, max_value=5, value=3)

# --- Prediction button ---
if st.button("🚀 Predict"):
    try:
        # Match feature names used during training
        input_df = pd.DataFrame([[price, weight, review_score]],
                                columns=["price", "weight", "review_score"])

        scaled_input = scaler.transform(input_df)

        review_pred = clf.predict(scaled_input)[0]
        freight_pred = reg.predict(scaled_input)[0]

        # --- Display results ---
        st.success(f"⭐ Predicted Review Quality: {'Good' if review_pred == 1 else 'Bad'}")
        st.success(f"🚚 Predicted Freight Value: R$ {freight_pred:.2f}")

    except Exception as e:
        st.error(f"❗ Prediction Error: {e}")
