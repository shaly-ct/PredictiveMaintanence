import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

#Download the model from model hub
model_path = hf_hub_download(repo_id="Shalyn/PredictiveMaintanence-model",filename="engine_condition_model_v1.joblib")

#loading the model
model = joblib.load(model_path)

#Building Streamlit app UI
st.title("Predictive Maintanence Prediction Tool")
st.write("This tool helps to forecast potential failures in vehicles before they occur.")
st.write("Kindly enter the vehicle sensor values to predict if the vehicle needs maintanence.")

#collect user input
Engine_RPM = st.number_input("Engine_RPM(The number of revolutions per minute (RPM) of the engine, indicating engine speed. It is defined in Revolutions per Minute (RPM))")
Lub_Oil_Pressure = st.number_input("Lub_Oil_Pressure (The pressure of the lubricating oil in the engine, essential for reducing friction and wear. It is defined in bar or kilopascals (kPa))")
Fuel_Pressure = st.number_input("Fuel_Pressure (The pressure at which fuel is supplied to the engine, critical for proper combustion. It is defined in bar or kilopascals (kPa))")
Coolant_Pressure = st.number_input("Coolant_Pressure (The pressure of the engine coolant, affecting engine temperature regulation. It is defined in bar or kilopascals (kPa) )")
Lub_Oil_Temperature = st.number_input("Lub_Oil_Temperature (The temperature of the lubricating oil, which impacts viscosity and engine performance. It is defined in degrees Celsius (°C) )")
Coolant_Temperature = st.number_input("Coolant_Temperature (The temperature of the engine coolant, crucial for preventing overheating. It is defined in degrees Celsius (°C))")

#Converting the inputs to match training datarow
input_data = pd.DataFrame([{
    'Engine rpm': Engine_RPM,
    'Lub oil pressure': Lub_Oil_Pressure,
    'Fuel pressure': Fuel_Pressure,
    'Coolant pressure': Coolant_Pressure,
    'lub oil temp' : Lub_Oil_Temperature,
    'Coolant temp': Coolant_Temperature
}])

#setting up classification threshold
classification_threshold = 0.45

#Creating prediction button
if st.button("Predict"):
  prediction_prob = model.predict_proba(input_data)[0,1]
  prediction = (prediction_prob>classification_threshold).astype(int)
  result = "Off/False/Active" if prediction == 0 else "On/True/Faulty"
  st.write(f"Based on the given input the vehicle is {result}")
