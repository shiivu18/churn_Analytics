import streamlit as st
import joblib

st.title("Customer Churn Prediction App")

model = joblib.load(
    r"D:\Churn-Analytics\models\churn_model.pkl"
)

st.success("Model Loaded Successfully")
