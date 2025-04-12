# app.py
import streamlit as st
import os

# Check and install required packages
try:
    import pandas as pd
    import joblib
    import pyreadstat
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    st.error(f"Missing required package: {str(e)}")
    st.info("Installing required packages... (this may take a moment)")
    os.system("pip install pandas joblib pyreadstat scikit-learn")
    st.experimental_rerun()

# Now that packages are installed, import them properly
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
        st.error(f"Prediction failed: {str(e)}")
        return None

if submitted:
    input_data = {
        'Age': Age,
        'Marital': 1 if Marital_Status == "Single" else 2 if Marital_Status == "Married" else 3,
        'Expenses': Expenses,
        'Income': Income,
        'Amount': Amount,
        'Price': Price
    }
    input_df = pd.DataFrame([input_data])
    
    results = predict(input_df)
    
    if results:
        st.subheader("Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Logistic Regression**")
            pred = "Non-Solvent 🚨" if results['logistic']['pred'] == 1 else "Solvent ✅"
            st.metric("Status", pred, f"{results['logistic']['proba'][results['logistic']['pred']]*100:.1f}%")
            st.bar_chart(pd.DataFrame({
                'Probability': results['logistic']['proba'],
                'Status': ['Solvent', 'Non-Solvent']
            }).set_index('Status'))
        
        with col2:
            st.markdown("**KNN Model**")
            pred = "Non-Solvent 🚨" if results['knn']['pred'] == 1 else "Solvent ✅"
            st.metric("Status", pred, f"{results['knn']['proba'][results['knn']['pred']]*100:.1f}%")
            st.bar_chart(pd.DataFrame({
                'Probability': results['knn']['proba'],
                'Status': ['Solvent', 'Non-Solvent']
            }).set_index('Status'))

# Debug info
with st.expander("Technical Details"):
    st.write("Loaded models:", list(models.keys()))
    st.write("Sample data:", df.head(1))
