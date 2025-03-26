import streamlit as st
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin

# --- Page Configuration ---
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Improved Font ---
custom_css = """
<style>
/* Set a modern sans-serif font for the app */
body, .css-18e3th9, .css-1d391kg {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #f0f2f6;
    color: #333333;
}

/* Main content styling */
.reportview-container {
    background-color: #ffffff;
    color: #333333;
}

/* Input widget styling */
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stSelectbox>div>div>input {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Custom Transformer Definition ---
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds engineered features:
    - tenure_group: Bins tenure into groups.
    - AvgMonthlyCharge: TotalCharges divided by tenure (fallback to MonthlyCharges if tenure == 0).
    """
    def __init__(self):
        pass

    def tenure_bin(self, tenure):
        if tenure <= 12:
            return '0-12'
        elif tenure <= 24:
            return '13-24'
        elif tenure <= 48:
            return '25-48'
        elif tenure <= 60:
            return '49-60'
        else:
            return '60+'

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['tenure_group'] = X['tenure'].apply(self.tenure_bin)
        X['AvgMonthlyCharge'] = X.apply(
            lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else row['MonthlyCharges'],
            axis=1
        )
        return X

# --- Model Loading using st.cache_resource ---
@st.cache_resource
def load_model():
    with open("best_churn_model_bayes.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- Sidebar for Instructions ---
st.sidebar.title("Telco Churn App")
st.sidebar.markdown("""
**Overview:**

This application predicts customer churn using a pre-trained machine learning model.

**Instructions:**
- Enter customer details in the main panel.
- Click **Predict Churn**.
- View the prediction result and probability.
""")

# --- Main Application ---
st.title("Telco Customer Churn Prediction")
st.markdown("---")
st.write("Fill out the customer details below:")

# --- Input Widgets ---
col1, col2 = st.columns(2)
with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=840.0)
    Partner = st.selectbox("Partner", options=["Yes", "No"])
    Dependents = st.selectbox("Dependents", options=["Yes", "No"])
with col2:
    MultipleLines = st.selectbox("Multiple Lines", options=["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", options=["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", options=["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", options=["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", options=["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", options=["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", options=["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", options=["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    SeniorCitizen = st.selectbox("Senior Citizen", options=["0", "1"])

# --- Collect Inputs into a DataFrame ---
data = {
    "tenure": [tenure],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges],
    "Partner": [Partner],
    "Dependents": [Dependents],
    "MultipleLines": [MultipleLines],
    "InternetService": [InternetService],
    "OnlineSecurity": [OnlineSecurity],
    "OnlineBackup": [OnlineBackup],
    "DeviceProtection": [DeviceProtection],
    "TechSupport": [TechSupport],
    "StreamingTV": [StreamingTV],
    "StreamingMovies": [StreamingMovies],
    "Contract": [Contract],
    "PaperlessBilling": [PaperlessBilling],
    "PaymentMethod": [PaymentMethod],
    "SeniorCitizen": [SeniorCitizen]
}
input_df = pd.DataFrame(data)

st.markdown("---")
st.subheader("Customer Input Data")
st.dataframe(input_df)

# --- Prediction Section ---
if st.button("Predict Churn"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    if prediction[0] == 1:
        result = "Churn"
        prob = prediction_proba[0][1]
    else:
        result = "No Churn"
        prob = prediction_proba[0][0]
    formatted_prob = f"{prob * 100:.2f}%"
    
    st.markdown("---")
    st.subheader("Prediction Result")
    st.markdown(f"**Prediction:** {result} (**{formatted_prob}**)")
