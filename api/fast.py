import joblib
import pandas as pd
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)
# ============================
model_path = 'models/lightgbm_model.pkl'

if not os.path.exists(model_path):
    print(f" Model file not found at {model_path}! Ensure it exists.")

try:
    app.state.model = joblib.load(model_path)
except Exception as e:
    print(f" Failed to load model due to: {e}")
# ============================
@app.get("/predict")
def predict(age, gender, bmi, daily_steps, sleep_hours,
                water_intake_l, calories_consumed, smoker, alcohol,
                resting_hr, systolic_bp, diastolic_bp, cholesterol, family_history):
    X_pred = pd.DataFrame(locals(), index=[0])

    float_cols = ['bmi', 'sleep_hours', 'water_intake_l']
    int_cols = ['age', 'gender', 'daily_steps', 'calories_consumed', 'smoker',
                'alcohol', 'resting_hr', 'systolic_bp', 'diastolic_bp',
                'cholesterol', 'family_history']

    for c in float_cols:
        if c in X_pred.columns:
            X_pred[c] = X_pred[c].astype(float)
    for c in int_cols:
        if c in X_pred.columns:
            X_pred[c] = X_pred[c].astype(int)

    model = app.state.model

    prediction = model.predict(X_pred)[0]
    prob = model.predict_proba(X_pred)[0][1] * 100

    return dict(prediction=float(prediction) ,prob=float(prob))
