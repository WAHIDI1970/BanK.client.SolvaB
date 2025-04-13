import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Set up the app
st.set_page_config(page_title="Bank Solvency Predictor", layout="centered")
st.title("Bank Client Solvency Prediction")
st.markdown("""
Predict whether a client will be solvent (0) or non-solvent (1) based on financial information.
The model uses Logistic Regression with optimal threshold.
""")

# Load the model and scaler
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load('REGLOG.pkl')
        return model_data['model'], model_data['threshold']
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, optimal_threshold = load_model()

# Input form
with st.form("client_info"):
    st.subheader("Client Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        marital = st.selectbox("Marital Status", 
                             options=[1, 2, 3, 4, 5],
                             format_func=lambda x: ["Single", "Married", "Divorced", "Widowed", "Other"][x-1])
        expenses = st.number_input("Monthly Expenses ($)", min_value=0, value=1000)
    
    with col2:
        income = st.number_input("Monthly Income ($)", min_value=0, value=2000)
        amount = st.number_input("Loan Amount Requested ($)", min_value=0, value=5000)
        price = st.number_input("Item Price ($)", min_value=0, value=6000)
    
    submitted = st.form_submit_button("Predict Solvency")

# Make prediction when form is submitted
if submitted and model:
    # Create input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'Marital': [marital],
        'Expenses': [expenses],
        'Income': [income],
        'Amount': [amount],
        'Price': [price]
    })
    
    # Scale the input data (same as during training)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(input_data)
    
    # Get probabilities
    proba = model.predict_proba(scaled_data)[0]
    prediction = (proba[1] >= optimal_threshold).astype(int)
    
    # Display results
    st.subheader("Prediction Results")
    
    if prediction == 1:
        st.error("Prediction: Non-Solvable (1) - High risk of default")
    else:
        st.success("Prediction: Solvable (0) - Low risk of default")
    
    # Show probabilities
    st.markdown("### Prediction Probabilities")
    proba_df = pd.DataFrame({
        'Status': ['Solvable (0)', 'Non-Solvable (1)'],
        'Probability': [proba[0], proba[1]]
    })
    
    # Display probability bar chart
    st.bar_chart(proba_df.set_index('Status'))
    
    # Show feature importance if available
    try:
        if hasattr(model.named_steps['model'], 'coef_'):
            coefficients = model.named_steps['model'].coef_[0]
            importance_df = pd.DataFrame({
                'Feature': input_data.columns,
                'Importance': np.abs(coefficients)
            }).sort_values('Importance', ascending=False)
            
            st.markdown("### Most Important Features")
            st.dataframe(importance_df.style.format({'Importance': '{:.3f}'}))
    except:
        pass

elif submitted and not model:
    st.warning("Model not loaded properly. Please check the model file.")

# Add some sample data for testing
st.sidebar.markdown("### Sample Data Values")
st.sidebar.write("""
- **Solvable Client**:
  - Age: 30, Married, Expenses: 1000, Income: 3000, Amount: 2000, Price: 2500
- **Non-Solvable Client**:
  - Age: 25, Single, Expenses: 1500, Income: 1500, Amount: 5000, Price: 6000
""")
            
