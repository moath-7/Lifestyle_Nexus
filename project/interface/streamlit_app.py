import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image

# ============================
# تحميل الموديل الحقيقي
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
# إعداد الصفحة العامة
# ============================
st.set_page_config(page_title="Lifestyle Nexus", page_icon="💚", layout="centered")

# ============================
# CSS لتخصيص الألوان والتصميم
# ============================


st.markdown("""
<style>
body, .main, .block-container, .stApp {
    background-color: white !important;
}
.block-container {
    text-align: center;
}
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    padding: 30px 20px;
    border-right: 2px solid #eee;
}
div[role="radiogroup"] label {
    font-size: 16px;
    padding: 8px 0;
    color: #1e3932 !important;
}
div.stButton > button {
    width: auto;
    padding: 12px 28px;
    font-size: 18px;
    background-color: #37D8B8 !important;
    color: white !important;
    font-weight: 600;
    border-radius: 12px;
    border: none;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #0000 !important;
    transform: scale(1.03);
}
h1, h2, h3 {
    color: #0d3b2e !important;
    font-weight: 800;
}
p {
    color: #1e3932 !important;
    font-size: 17px;
    opacity: .85;
}
input, select, textarea {
    border-radius: 10px !important;
    border: 1px solid #bbb !important;
    padding: 10px !important;
}
footer, #MainMenu, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================
# الشعار
# ============================
logo = Image.open("/home/weam1/code/moath-7/Lifestyle_Nexus/project/interface/Black Beige Minimalist  Photography Portfolio Cover Page.png")
st.image(logo, width=200)

# ============================
# Sidebar للتنقل
# ============================
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio("Go to", ["Home", "Prediction", "Results"])
st.session_state.page = page_selection.lower()

# ============================
# الصفحة الرئيسية
# ============================
def home_page():
    st.markdown("<h1>Discover Your Health Risk With AI</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 40px; background-color: #f9f9f9; border-radius: 15px;'>
        <h1 style='color: #2E8B57; font-size: 42px; font-weight: bold;'> Welcome to Lifestyle Nexus</h1>
        <p style='font-size: 20px; color: #555;'>Your journey to a healthier life starts here</p>
        <hr style='width: 60%; margin: 25px auto; border: 1px solid #ddd;'/>
        <p style='font-size: 18px; color: #444;'>
            Utilize our advanced <b>AI</b> to gain insights into your health 
            and take proactive steps toward a better lifestyle.
        </p>
        <p style='font-size: 18px; color: #444;'>
            Small changes can make a big difference — let's build a better, healthier you together.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2,3,2])
    with col2:
        if st.button("Start Your Health Prediction", key="start_button_home"):
            st.session_state.page = "Prediction"
            st.rerun()

# ============================
# صفحة الإدخال والتنبؤ
# ============================
def form_page():
    st.markdown("<h2>Health Data Input Form</h2>", unsafe_allow_html=True)
    st.markdown("<p>Enter your health details below to predict your lifestyle risk.</p>", unsafe_allow_html=True)

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

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Home"):
            st.session_state.page = "home"
            st.experimental_rerun()
    with col2:
        if st.button("Predict Now"):
            gender_val = 0 if gender == "Male" else 1
            smoker_val = 1 if smoker == "Yes" else 0
            alcohol_val = 1 if alcohol == "Yes" else 0
            family_val = 1 if family_history == "Yes" else 0

            user_data = pd.DataFrame([[
                age, gender_val, bmi, daily_steps, sleep_hours, water_intake,
                calories, smoker_val, alcohol_val, resting_hr,
                systolic_bp, diastolic_bp, cholesterol, family_val
            ]], columns=[
                'age', 'gender', 'bmi', 'daily_steps', 'sleep_hours',
                'water_intake_l', 'calories_consumed', 'smoker', 'alcohol',
                'resting_hr', 'systolic_bp', 'diastolic_bp', 'cholesterol', 'family_history'
            ])

            try:
                prediction = model.predict(user_data)[0]
                prob = model.predict_proba(user_data)[0][1] * 100
                st.session_state['prediction'] = (prediction, prob)
                st.session_state.page = "results"
                st.experimental_rerun()
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

# ============================
# صفحة النتائج
# ============================
def results_page():
    st.markdown("<h2>Your Health Prediction Result</h2>", unsafe_allow_html=True)
    if 'prediction' in st.session_state:
        prediction, prob = st.session_state['prediction']
        if prediction == 1:
            st.success(f"✅ Your lifestyle is healthy with a probability of {prob:.1f}%")
        else:
            st.error(f"⚠️ Your lifestyle is unhealthy (Health probability {prob:.1f}%)")
    else:
        st.warning("⚠️ No prediction found. Please complete a prediction first.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Try Again"):
        st.session_state.page = "prediction"
        st.experimental_rerun()
    if st.button("Back to Home"):
        st.session_state.page = "home"
        st.experimental_rerun()

# ============================
# إدارة الصفحات
# ============================
page = st.session_state.page
if page == "home":
    home_page()
elif page == "prediction":
    form_page()
elif page == "results":
    results_page()