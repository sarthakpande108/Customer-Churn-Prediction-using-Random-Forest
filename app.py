# app.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open("model/model.pkl", "rb") as file:
    model, scaler = pickle.load(file)

# Define user input functions
def user_input():
    st.sidebar.header("Customer Input Features")
    
    # Collecting user inputs with sidebar interface
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    senior_citizen = st.sidebar.selectbox("Senior Citizen", (0, 1))
    partner = st.sidebar.selectbox("Partner", ("Yes", "No"))
    dependents = st.sidebar.selectbox("Dependents", ("Yes", "No"))
    tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 1)
    phone_service = st.sidebar.selectbox("Phone Service", ("Yes", "No"))
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ("No", "Yes", "No phone service"))
    internet_service = st.sidebar.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
    online_security = st.sidebar.selectbox("Online Security", ("No", "Yes", "No internet service"))
    online_backup = st.sidebar.selectbox("Online Backup", ("No", "Yes", "No internet service"))
    device_protection = st.sidebar.selectbox("Device Protection", ("No", "Yes", "No internet service"))
    tech_support = st.sidebar.selectbox("Tech Support", ("No", "Yes", "No internet service"))
    streaming_tv = st.sidebar.selectbox("Streaming TV", ("No", "Yes", "No internet service"))
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ("No", "Yes", "No internet service"))
    contract = st.sidebar.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ("Yes", "No"))
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        (
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ),
    )
    monthly_charges = st.sidebar.slider("Monthly Charges", 18.0, 120.0, 18.0)
    total_charges = st.sidebar.slider("Total Charges", 18.0, 8700.0, 18.0)

    data = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": senior_citizen,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if phone_service == "Yes" else 0,
        "MultipleLines": ["No", "Yes", "No phone service"].index(multiple_lines),
        "InternetService": ["DSL", "Fiber optic", "No"].index(internet_service),
        "OnlineSecurity": ["No", "Yes", "No internet service"].index(online_security),
        "OnlineBackup": ["No", "Yes", "No internet service"].index(online_backup),
        "DeviceProtection": ["No", "Yes", "No internet service"].index(device_protection),
        "TechSupport": ["No", "Yes", "No internet service"].index(tech_support),
        "StreamingTV": ["No", "Yes", "No internet service"].index(streaming_tv),
        "StreamingMovies": ["No", "Yes", "No internet service"].index(streaming_movies),
        "Contract": ["Month-to-month", "One year", "Two year"].index(contract),
        "PaperlessBilling": 1 if paperless_billing == "Yes" else 0,
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ].index(payment_method),
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Main App Function
def main():
    st.title("Customer Churn Prediction")
    input_df = user_input()

    # Preprocess the input data
    input_features = scaler.transform(input_df)
    
    # Make predictions
    prediction = model.predict(input_features)
    prediction_proba = model.predict_proba(input_features)

    st.subheader("Prediction")
    st.write("Churn" if prediction[0] == 1 else "No Churn")
    st.subheader("Prediction Probability")
    st.write(prediction_proba)

if __name__ == "__main__":
    main()
