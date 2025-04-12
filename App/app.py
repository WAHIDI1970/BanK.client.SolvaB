# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib

# =============================================
# PAGE CONFIGURATION
# =============================================
st.set_page_config(
    page_title="Client Solvency Predictor",
    layout="centered",
    page_icon="🏦"
)

st.title("🏦 Client Solvency Prediction App")
st.markdown("This app predicts whether a client is solvable (0) or non-solvable (1) based on their financial data.")

# =============================================
# MODEL LOADING
# =============================================
@st.cache_resource
def load_models():
    try:
        logistic_model = joblib.load("models/logistic_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return {"logistic": logistic_model, "scaler": scaler}
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

models = load_models()

# =============================================
# INPUT FORM
# =============================================
st.sidebar.header("📋 Client Information")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
marital = st.sidebar.selectbox(
    "Marital Status",
    options=[1, 2, 3],
    format_func=lambda x: {1: "Single", 2: "Married", 3: "Divorced"}[x]
)
expenses = st.sidebar.number_input("Monthly Expenses (€)", min_value=0.0, value=500.0, step=50.0)
income = st.sidebar.number_input("Monthly Income (€)", min_value=0.0, value=1500.0, step=100.0)
amount = st.sidebar.number_input("Loan Amount (€)", min_value=0.0, value=2000.0, step=100.0)
price = st.sidebar.number_input("Purchase Price (€)", min_value=0.0, value=2500.0, step=100.0)

# Prepare input data
client_data = pd.DataFrame({
    "Age": [age],
    "Marital": [marital],
    "Expenses": [expenses],
    "Income": [income],
    "Amount": [amount],
    "Price": [price]
})

# =============================================
# PREDICTION FUNCTION
# =============================================
def predict_solvability(data):
    try:
        # Scale features
        scaled_data = models["scaler"].transform(data)

        # Predict solvability
        prediction = models["logistic"].predict(scaled_data)[0]
        probability = models["logistic"].predict_proba(scaled_data)[0][1]

        return prediction, probability
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

# =============================================
# DISPLAY RESULTS
# =============================================
if st.sidebar.button("🔮 Predict Solvability"):
    st.subheader("Client Data")
    st.table(client_data)

    with st.spinner("Analyzing client data..."):
        prediction, probability = predict_solvability(client_data)

        st.subheader("Prediction Results")
        if prediction == 1:
            st.error(f"🚨 Non-Solvable Client (Confidence: {probability:.1%})")
        else:
            st.success(f"✅ Solvable Client (Confidence: {1-probability:.1%})")
            
