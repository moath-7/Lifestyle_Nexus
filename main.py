#from project.app.model import load_model
#def main():
#stearmlit user input
#preprocess user input
 #  load_model()
#give to model to predict
#result
#show result on UI



#if __name__ == "__main__":
 #   main()
#STREAMLIT CODE
## TODO
# Streamlit Code - user input
# This will be prediction
# Give prediction to the model
# Model.predict (this is the output)
# Show result on the UI
#

import streamlit as st
from project.app.model import load_model
import pandas as pd

def preprocess(user_input):
    return pd.DataFrame([user_input])

def main():
    model = load_model()
    st.title("Health Prediction App")
    # Input fields for each feature
    age = st.number_input("Age", min_value=0, value=25)
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    bmi = st.number_input("BMI", min_value=0.0, value=20.5)
    daily_steps = st.number_input("Daily Steps", min_value=0, value=12000)
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, value=8.5)
    calories_consumed = st.number_input("Calories Consumed", min_value=0, value=2000)
    family_history = st.selectbox("Family History (0: No, 1: Yes)", options=[0, 1])
    smoker = st.selectbox("Smoker (0: No, 1: Yes)", options=[0, 1])
    resting_hr = st.number_input("Resting Heart Rate", min_value=0, value=60)
    water_intake_l = st.number_input("Water Intake (Liters)", min_value=0.0, value=3.5)
    systolic_bp = st.number_input("Systolic Blood Pressure", min_value=0, value=110)
    diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=0, value=70)
    cholesterol = st.number_input("Cholesterol", min_value=0, value=155)
    alcohol = st.number_input("Alcohol Consumption (0: No, 1: Yes)", options=[0, 1])

    if st.button("Predict"):
        user_input = {
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "daily_steps": daily_steps,
            "sleep_hours": sleep_hours,
            "calories_consumed": calories_consumed,
            "family_history": family_history,
            "smoker": smoker,
            "resting_hr": resting_hr,
            "water_intake_l": water_intake_l,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "cholesterol": cholesterol,
            "alcohol": alcohol
        }

        user_input_processed = preprocess(user_input)
        result = model.predict(user_input_processed)
        st.success(f"The predicted health risk is:: {result}")

if __name__ == "__main__":
    main()
