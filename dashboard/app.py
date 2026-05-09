import streamlit as st
import joblib
import pandas as pd

model_columns = joblib.load(
    r"D:\Churn-Analytics\models\model_columns.pkl"
)

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


if st.button("Predict"):

    # Create input dataframe
    input_data = pd.DataFrame({
        'Tenure Months': [tenure],
        'Monthly Charge': [monthly_charge]
    })

    # Add missing columns
    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Arrange columns correctly
    input_data = input_data[model_columns]

    # Make prediction
    prediction = model.predict(input_data)

    # Prediction result
    if prediction[0] == 1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is likely to stay")




        