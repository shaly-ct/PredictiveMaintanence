import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

st.title("Predictive Maintenance Prediction Tool")

# Load model
model_path = hf_hub_download(
    repo_id="Shalyn/PredictiveMaintanence-model",
    filename="engine_condition_model_v1.joblib",
    token=os.getenv("HF_TOKEN")
)
model = joblib.load(model_path)

# User input
Engine_RPM = st.number_input("Engine RPM", min_value=0)
Lub_Oil_Pressure = st.number_input("Lub Oil Pressure")
Fuel_Pressure = st.number_input("Fuel Pressure")
Coolant_Pressure = st.number_input("Coolant Pressure")
Lub_Oil_Temperature = st.number_input("Lub Oil Temperature")
Coolant_Temperature = st.number_input("Coolant Temperature")

input_data = pd.DataFrame([{
    'Engine rpm': Engine_RPM,
    'Lub oil pressure': Lub_Oil_Pressure,
    'Fuel pressure': Fuel_Pressure,
    'Coolant pressure': Coolant_Pressure,
    'lub oil temp': Lub_Oil_Temperature,
    'Coolant temp': Coolant_Temperature
}])

# Prediction
classification_threshold = 0.45
if st.button("Predict"):
    prediction_prob = model.predict_proba(input_data)[0,1]
    prediction = int(prediction_prob > classification_threshold)
    result = "Off/False/Active" if prediction == 0 else "On/True/Faulty"
    st.write(f"Vehicle status: **{result}**")
