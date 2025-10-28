import streamlit as st
import pandas as pd
import joblib
import os
# ============================
# Load the real model
# ============================
interface_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(interface_dir)
lifestyle_nexus_dir = os.path.dirname(project_dir)
model_path = os.path.join(lifestyle_nexus_dir, 'models', 'lightgbm_model.pkl')

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at {model_path}! Ensure it exists.")
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Failed to load model due to: {e}")
    st.stop()
    # ============================
# Streamlit App Interface
# ============================
st.set_page_config(page_title="Health Lifestyle Predictor", page_icon="🏃", layout="centered")

st.title("🏃 Health Lifestyle Predictor")
st.markdown("### Enter your information to find out if your *lifestyle is healthy or not* 💪")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=10, max_value=100, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, step=0.1)
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, step=100)
    sleep_hours = st.number_input("Sleep Hours per Day", min_value=0.0, max_value=24.0, step=0.5)
    water_intake = st.number_input("Liters of Water per Day", min_value=0.0, max_value=10.0, step=0.1)
    calories = st.number_input("Daily Calorie Intake", min_value=1000, max_value=5000, step=50)
with col2:

    smoker = st.selectbox("Do you smoke?", ["No", "Yes"])
    alcohol = st.selectbox("Do you drink alcohol?", ["No", "Yes"])
    resting_hr = st.number_input("Resting Heart Rate (BPM)", min_value=40, max_value=150, step=1)
    systolic_bp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, step=1)
    diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=150, step=1)
    cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, step=1)
    family_history = st.selectbox("Family History of Disease?", ["No", "Yes"])

# Predict Button
if st.button("🔍 Predict Now"):
    gender_val = 0 if gender == "Male" else 1
    smoker_val = 1 if smoker == "Yes" else 0
    alcohol_val = 1 if alcohol == "Yes" else 0
    family_val = 1 if family_history == "Yes" else 0

    # Prepare user input in the same order as the model training
    user_data = pd.DataFrame([[
        age, gender_val, bmi, daily_steps, sleep_hours, water_intake,
        calories, smoker_val, alcohol_val, resting_hr,
        systolic_bp, diastolic_bp, cholesterol, family_val
    ]], columns=[
        'age', 'gender', 'bmi', 'daily_steps', 'sleep_hours',
        'water_intake_l', 'calories_consumed', 'smoker', 'alcohol',
        'resting_hr', 'systolic_bp', 'diastolic_bp', 'cholesterol', 'family_history'
    ])

    # Make prediction and calculate probability
    prediction = model.predict(user_data)[0]
    prob = model.predict_proba(user_data)[0][1] * 100

    st.write("---")
    if prediction == 1:
        st.success(f"✅ Your lifestyle is healthy with a probability of {prob:.1f}%")
    else:
        st.error(f"⚠️ Your lifestyle is unhealthy (Health probability {prob:.1f}%)")

    st.caption("📊 Prediction made using a LightGBM model trained on balanced health data (SMOTE) with 81% accuracy.")
# ============================
# Application interface
# ============================
# st.title("🏃 Lifestyle Nexus")
# st.write("Enter your data to find out if your lifestyle is healthy or not 💪")

# # User Inputs
# age = st.number_input("Age", min_value=10, max_value=100, step=1)
# gender = st.selectbox("Gender", options=["Male", "Female"])  # Dropdown for Gender
# bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, step=0.1)
# daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, step=100)
# water_intake = st.number_input("Liters of water you drink per day", min_value=0.0, max_value=10.0, step=0.1)
# sleep_hours = st.number_input("Hours of sleep per day", min_value=0.0, max_value=24.0, step=0.5)
# calories_consumed = st.number_input("Calories Consumed", min_value=1000, max_value=5000, step=100)  # New input for calories
# smoker = st.selectbox("Do you smoke?", options=["No", "Yes"])  # Dropdown for smoking status
# alcohol = st.selectbox("Do you drink alcohol?", options=["No", "Yes"])  # Dropdown for alcohol consumption
# resting_hr = st.number_input("Resting Heart Rate (BPM)", min_value=40, max_value=120, step=1)  # New input for HR
# systolic_bp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, step=1)  # New input for BP
# diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=150, step=1)  # New input for BP
# cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=300, step=1)  # New input for cholesterol
# family_history = st.selectbox("Family History of Illness?", options=["No", "Yes"])  # Dropdown for family history
# if st.button("Predict"):
#     # Prepare the user's data in the same order as training
#     user_data = pd.DataFrame({
#         'age': [age],
#         'gender': [0],  # Temporarily (Male=0, Female=1) if no selection in the interface
#         'bmi': [bmi],
#         'daily_steps': [daily_steps],
#         'sleep_hours': [sleep_hours],
#         'water_intake_l': [water_intake],
#         'calories_consumed': [2000],  # Placeholder value
#         'smoker': [0],  # 0 = non-smoker
#         'alcohol': [0],  # 0 = does not drink
#         'resting_hr': [70],
#         'systolic_bp': [120],
#         'diastolic_bp': [80],
#         'cholesterol': [180],
#         'family_history': [0],
#     })

#     # Make a prediction
#     prediction = model.predict(user_data)[0]

#     # Display the result
#     if prediction == 1:
#         st.success("✅ Your lifestyle is healthy!")
#     else:
#         st.error("⚠️ Your lifestyle is unhealthy, try to improve it 🙂")
 