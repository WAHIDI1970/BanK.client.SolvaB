# app.py
import streamlit as st
import pandas as pd
import joblib
import pyreadstat
import os
from sklearn.preprocessing import StandardScaler

# App configuration
st.set_page_config(
    page_title="🏦 Credit Scoring App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 Credit Solvency Prediction")
st.markdown("Predict client solvency using machine learning models")

# Define paths - EXACTLY matching your repository structure
PATHS = {
    'data': 'App/Data/scoring.sav',
    'knn': 'App/Data/models/KNN (1).pkl',
    'logistic': 'App/Data/models/REGLOG (1).pkl',
    'scaler': 'App/Data/models/scaler.pkl'
}

@st.cache_resource
def load_resources():
    try:
        # Verify all files exist
        missing = []
        for name, path in PATHS.items():
            if not os.path.exists(path):
                missing.append(path)
        
        if missing:
            raise FileNotFoundError(f"Missing files: {', '.join(missing)}")

        # Load resources
        df, _ = pyreadstat.read_sav(PATHS['data'])
        models = {
            'knn': joblib.load(PATHS['knn']),
            'logistic': joblib.load(PATHS['logistic']),
            'scaler': joblib.load(PATHS['scaler'])
        }
        return df, models
    
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        st.error("Please verify:")
        st.error("1. All files exist in these exact locations:")
        for path in PATHS.values():
            st.error(f"   - {path}")
        
        st.error("\n2. Current directory contents:")
        st.code(f"""
        Root: {os.listdir('.')}
        App/: {os.listdir('App') if os.path.exists('App') else 'MISSING'}
        App/Data/: {os.listdir('App/Data') if os.path.exists('App/Data') else 'MISSING'}
        App/Data/models/: {os.listdir('App/Data/models') if os.path.exists('App/Data/models') else 'MISSING'}
        """)
        return None, None

# Load data and models
df, models = load_resources()
if df is None or models is None:
    st.stop()

# Input form
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
def predict(input_df):
    try:
        scaled = models['scaler'].transform(input_df)
        return {
            'logistic': {
                'pred': models['logistic'].predict(scaled)[0],
                'proba': models['logistic'].predict_proba(scaled)[0]
            },
            'knn': {
                'pred': models['knn'].predict(scaled)[0],
                'proba': models['knn'].predict_proba(scaled)[0]
            }
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

if submitted:
    # Prepare input data
    input_data = {
        'Age': Age,
        'Marital': 1 if Marital_Status == "Single" else 2 if Marital_Status == "Married" else 3,
        'Expenses': Expenses,
        'Income': Income,
        'Amount': Amount,
        'Price': Price
    }
    input_df = pd.DataFrame([input_data])
    
    # Get predictions
    results = predict(input_df)
    
    if results:
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Logistic Regression**")
            pred = "Non-Solvent 🚨" if results['logistic']['pred'] == 1 else "Solvent ✅"
            st.metric("Status", pred, f"{max(results['logistic']['proba'])*100:.1f}% confidence")
            st.bar_chart(pd.DataFrame({
                'Probability': results['logistic']['proba'],
                'Status': ['Solvent', 'Non-Solvent']
            }).set_index('Status'))
        
        with col2:
            st.markdown("**KNN Model**")
            pred = "Non-Solvent 🚨" if results['knn']['pred'] == 1 else "Solvent ✅"
            st.metric("Status", pred, f"{max(results['knn']['proba'])*100:.1f}% confidence")
            st.bar_chart(pd.DataFrame({
                'Probability': results['knn']['proba'],
                'Status': ['Solvent', 'Non-Solvent']
            }).set_index('Status'))

# Debug section
with st.expander("System Information"):
    st.write("**Loaded Models:**")
    st.write(f"- KNN: {type(models['knn'])}")
    st.write(f"- Logistic Regression: {type(models['logistic'])}")
    st.write(f"- Scaler: {type(models['scaler'])}")
    
    st.write("\n**Sample Data:**")
    st.write(df.head(1))
