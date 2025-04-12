# app.py
import streamlit as st
import os
import sys

# Check requirements before imports
def check_requirements():
    required = {'pandas', 'joblib', 'pyreadstat', 'scikit-learn'}
    try:
        import pkg_resources
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = required - installed
        if missing:
            st.error(f"Missing packages: {', '.join(missing)}")
            st.info("Please add these to requirements.txt:")
            st.code("\n".join(missing))
            return False
        return True
    except:
        return True  # Proceed if we can't check

if not check_requirements():
    st.stop()

# Now safely import everything
import pandas as pd
import joblib
import pyreadstat
from sklearn.preprocessing import StandardScaler

# App configuration
st.set_page_config(page_title="🏦 Credit Scoring App", layout="wide")
st.title("🏦 Credit Solvency Prediction")

# Define paths - EXACTLY matching your repository
PATHS = {
    'data': 'Data/scoring.sav',
    'knn': 'models/KNN (1).pkl',
    'logistic': 'models/REGLOG (1).pkl',
    'scaler': 'models/scaler.pkl'
}

@st.cache_resource
def load_resources():
    try:
        # Verify all files exist
        missing = [name for name, path in PATHS.items() if not os.path.exists(path)]
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
        st.error(f"Loading error: {str(e)}")
        st.error("Required files:")
        st.error(f"- {PATHS['data']}")
        st.error(f"- {PATHS['knn']} (with space and parentheses)")
        st.error(f"- {PATHS['logistic']} (with space and parentheses)")
        st.error(f"- {PATHS['scaler']}")
        
        # Show directory contents
        st.error("Current directory structure:")
        st.code(f"""
        {os.listdir('.')}
        Data/: {os.listdir('Data') if os.path.exists('Data') else 'MISSING'}
        models/: {os.listdir('models') if os.path.exists('models') else 'MISSING'}
        """)
        return None, None

df, models = load_resources()
if df is None or models is None:
    st.stop()

# [Rest of your app code remains the same...]
# Include all the form, prediction, and display code from previous versions
