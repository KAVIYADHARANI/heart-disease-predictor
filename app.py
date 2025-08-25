import streamlit as st
import numpy as np
import joblib

# Load scaler and model
scaler = joblib.load("artifacts/scaler.joblib")
model = joblib.load("artifacts/stacked_model.joblib")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to check risk of heart disease.")

# Input fields (based on heart.csv columns except 'target')
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (1=Male, 0=Female)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect)", [1, 2, 3])

# Collect features
features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

# Scale
features_scaled = scaler.transform(features)

# Predict
if st.button("Predict"):
    prediction = model.predict(features_scaled)[0]
    if prediction == 1:
        st.error("⚠️ High risk of Heart Disease")
    else:
        st.success("✅ Low risk of Heart Disease")
