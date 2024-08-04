import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


type_encoder = joblib.load('type_encoder.pkl')
failure_encoder = joblib.load('failure_type.pkl')
scaler = joblib.load('scaler.pkl')
model = joblib.load('rfc_best_model.pkl')

df = pd.read_csv('predictive_maintenance.csv')
st.title("Predictive Maintenance App")

# Function to preprocess user input
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])
    input_df[scaler.feature_names_in_] = scaler.transform(input_df[scaler.feature_names_in_])
    input_df['Type'] = type_encoder.transform(input_df['Type'])
    return input_df

# User input for making predictions
st.write("Make a Prediction")
input_data = {}
for col in model.feature_names_in_:
    if col in scaler.feature_names_in_:
        input_data[col] = st.number_input(f"Enter value for {col}", value=float(df[col].mean()))
    else:
        input_data[col] = st.selectbox(f"Select value for {col}", options=df[col].unique())

# Convert user input to DataFrame and preprocess it
input_df = preprocess_input(input_data)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f"Predicted Failure Type: {failure_encoder.inverse_transform(prediction)[0]}")
