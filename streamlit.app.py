import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Olist Predictor", page_icon="🛍")

st.title("🛍 Olist eCommerce Prediction App")
st.markdown("Predict Review Quality and Freight Value using price and weight.")

# Check files
required_files = ["scaler.pkl", "classification_model.pkl", "regression_model.pkl"]
missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    st.error(f"❌ Missing files: {', '.join(missing)}")
    st.stop()

# Load files
scaler = joblib.load("scaler.pkl")
clf = joblib.load("classification_model.pkl")
reg = joblib.load("regression_model.pkl")

# Inputs
st.header("📦 Enter Product Details")
price = st.number_input("Price (R$)", min_value=0.0, value=100.0)
weight = st.number_input("Weight (g)", min_value=0.0, value=500.0)

# Predict
if st.button("🚀 Predict"):
    try:
        input_df = pd.DataFrame([[price, weight]], columns=["price", "weight"])
        scaled = scaler.transform(input_df)

        review_pred = clf.predict(scaled)[0]
        freight_pred = reg.predict(scaled)[0]

        st.success(f"⭐ Review Quality: {'Good' if review_pred == 1 else 'Bad'}")
        st.success(f"🚚 Freight Value: R$ {freight_pred:.2f}")

    except Exception as e:
        st.error(f"❗ Prediction Error: {e}")
