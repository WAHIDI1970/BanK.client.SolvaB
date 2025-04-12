# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import pyreadstat

# App configuration
st.set_page_config(page_title="🏦 Credit Scoring App", layout="wide")
st.title("🏦 Credit Solvency Prediction")
st.markdown("Predict client solvency using machine learning models")

# Load data and models with EXACT file paths
@st.cache_resource
def load_resources():
    try:
        # Load data
        df, meta = pyreadstat.read_sav('Data/scoring.sav')
        
        # Load models with your exact filenames
        models = {
            'knn': joblib.load('KNN (1).pkl'),  # With space and (1)
            'logistic': joblib.load('REGLOG (1).pkl'),  # Your logistic model
            'scaler': joblib.load('scaler.pkl')  # Without (1)
        }
        return df, models
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        st.error("Required files:")
        st.error("- Data/scoring.sav")
        st.error("- models/KNN (1).pkl (with space and parentheses)")
        st.error("- models/REGLOG (1).pkl (with space and parentheses)")
        st.error("- models/scaler.pkl")
        
        # Show what files exist
        st.error("Current directory contents:")
        st.error(f"Data folder: {os.listdir('Data') if os.path.exists('Data') else 'Missing'}")
        st.error(f"Models folder: {os.listdir('models') if os.path.exists('models') else 'Missing'}")
        return None, None

df, models = load_resources()
if df is None or models is None:
    st.stop()

# Input form with your variables
with st.form("client_form"):
    st.header("Client Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        Age = st.number_input("Age", min_value=18, max_value=100, value=35)
        Marital_Status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        Expenses = st.number_input("Monthly Expenses (€)", min_value=0, value=500)
    
    with col2:
        Income = st.number_input("Monthly Income (€)", min_value=0, value=2000)
        Amount = st.number_input("Loan Amount (€)", min_value=0, value=10000)
        Price = st.number_input("Purchase Value (€)", min_value=0, value=12000)
    
    submitted = st.form_submit_button("Predict Solvency")

# Prediction function
def predict_solvency(input_df, model_type='logistic'):
    try:
        scaled_input = models['scaler'].transform(input_df)
        model = models[model_type]
        prediction = model.predict(scaled_input)[0]
        proba = model.predict_proba(scaled_input)[0]
        return prediction, proba
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

if submitted:
    # Prepare input with correct variable names
    input_data = {
        'Age': Age,
        'Marital': 1 if Marital_Status == "Single" else 2 if Marital_Status == "Married" else 3,
        'Expenses': Expenses,
        'Income': Income,
        'Amount': Amount,
        'Price': Price
    }
    input_df = pd.DataFrame([input_data])
    
    # Make predictions
    log_pred, log_proba = predict_solvency(input_df, 'logistic')
    knn_pred, knn_proba = predict_solvency(input_df, 'knn')
    
    # Display results
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Logistic Regression (REGLOG)**")
        if log_pred == 1:
            st.error("Non-Solvent 🚨")
        else:
            st.success("Solvent ✅")
        st.metric("Confidence", f"{max(log_proba)*100:.1f}%")
        st.bar_chart(pd.DataFrame({'Probability': log_proba, 
                                 'Status': ['Solvent', 'Non-Solvent']}).set_index('Status'))
    
    with col2:
        st.markdown("**KNN Model**")
        if knn_pred == 1:
            st.error("Non-Solvent 🚨")
        else:
            st.success("Solvent ✅")
        st.metric("Confidence", f"{max(knn_proba)*100:.1f}%")
        st.bar_chart(pd.DataFrame({'Probability': knn_proba, 
                                 'Status': ['Solvent', 'Non-Solvent']}).set_index('Status'))

# Debug section
with st.expander("Technical Details"):
    st.write("**Loaded Data:**", df.shape)
    st.write("**Loaded Models:**")
    st.write(f"- KNN: {type(models['knn'])} (from KNN (1).pkl)")
    st.write(f"- Logistic: {type(models['logistic'])} (from REGLOG (1).pkl)")
    st.write(f"- Scaler: {type(models['scaler'])}")
    
    st.write("**Current Directory:**", os.listdir('.'))
    st.write("**Data Folder:**", os.listdir('Data') if os.path.exists('Data') else "Missing")
    st.write("**Models Folder:**", os.listdir('models') if os.path.exists('models') else "Missing")
