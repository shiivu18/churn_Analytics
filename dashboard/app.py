import streamlit as st
import joblib

st.title("Customer Churn Prediction App")

model = joblib.load(
    r"D:\Churn-Analytics\models\churn_model.pkl"
)

st.success("Model Loaded Successfully")

tenure = st.slider(
    "Tenure Months",
    0,
    72,
    12
)

monthly_charge = st.number_input(
    "Monthly Charge",
    min_value=0.0,
    value=50.0
)