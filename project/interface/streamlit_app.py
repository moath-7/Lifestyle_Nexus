import streamlit as st
import pandas as pd
import joblib
import os
import requests
from PIL import Image

# ============================
# تحميل الموديل
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
# إعداد الصفحة
# ============================
st.set_page_config(page_title="Lifestyle Nexus", page_icon="💚", layout="centered")

# ============================
# CSS لتخصيص الألوان والتصميم
# ============================
st.markdown("""
<style>
body, .main, .block-container, .stApp { background-color: white !important; }
.block-container { text-align: center; }
[data-testid="stSidebar"] { background-color: #ffffff !important; padding: 30px 20px; border-right: 2px solid #eee; }
div[role="radiogroup"] label { font-size: 16px; padding: 8px 0; color: #1e3932 !important; }
div.stButton > button { width: auto; padding: 12px 28px; font-size: 18px; background-color: #37D8B8 !important; color: white !important; font-weight: 600; border-radius: 12px; border: none; transition: 0.3s; }
div.stButton > button:hover { background-color: #2FB49C !important; transform: scale(1.03); }
h1, h2, h3 { color: #0d3b2e !important; font-weight: 800; }
p { color: #1e3932 !important; font-size: 17px; opacity: .85; }
input, select, textarea { border-radius: 10px !important; border: 1px solid #bbb !important; padding: 10px !important; }
footer, #MainMenu, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================
# الشعار
# ============================
logo = Image.open("project/interface/Cover.png")
st.image(logo, width=200)

# ============================
# إدارة الصفحة
# ============================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ============================
# Sidebar للتنقل
# ============================
pages = ["home", "prediction", "results"]
page_names = ["Home", "Prediction", "Results"]
current_index = pages.index(st.session_state.page)
page_selection = st.sidebar.radio("Go to", page_names, index=current_index)
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
        <p style='font-size: 18px; color: #444;'>Utilize our advanced <b>AI</b> to gain insights into your health.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2,3,2])
    with col2:
        if st.button("Start Your Health Prediction", key="start_button_home"):
            st.session_state.page = "prediction"

# ============================
# صفحة الإدخال والتنبؤ
# ============================
def form_page():
    st.markdown("<h2>Health Data Input Form</h2>", unsafe_allow_html=True)
    st.markdown("<p>Enter your health details below to predict your lifestyle risk.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # استخدام القيم المخزنة في session_state كقيم افتراضية
    def default(key, val):
        return st.session_state.get(key, val)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, step=1, key="age", value=default("age",25))
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender", index=0 if default("gender","Male")=="Male" else 1)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, step=0.1, key="bmi", value=default("bmi",22.0))
        daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, step=100, key="daily_steps", value=default("daily_steps",5000))
        sleep_hours = st.number_input("Sleep Hours per Day", min_value=0.0, max_value=24.0, step=0.5, key="sleep_hours", value=default("sleep_hours",7.0))
        water_intake = st.number_input("Liters of Water per Day", min_value=0.0, max_value=10.0, step=0.1, key="water_intake", value=default("water_intake",2.0))
        calories = st.number_input("Daily Calorie Intake", min_value=1000, max_value=5000, step=50, key="calories", value=default("calories",2000))

    with col2:
        smoker = st.selectbox("Do you smoke?", ["No", "Yes"], key="smoker", index=0 if default("smoker","No")=="No" else 1)
        alcohol = st.selectbox("Do you drink alcohol?", ["No", "Yes"], key="alcohol", index=0 if default("alcohol","No")=="No" else 1)
        resting_hr = st.number_input("Resting Heart Rate (BPM)", min_value=40, max_value=150, step=1, key="resting_hr", value=default("resting_hr",70))
        systolic_bp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, step=1, key="systolic_bp", value=default("systolic_bp",120))
        diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=150, step=1, key="diastolic_bp", value=default("diastolic_bp",80))
        cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, step=1, key="cholesterol", value=default("cholesterol",180))
        family_history = st.selectbox("Family History of Disease?", ["No", "Yes"], key="family_history", index=0 if default("family_history","No")=="No" else 1)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Home", key="back_home"):
            st.session_state.page = "home"

    with col2:
        if st.button("Predict Now", key="predict_now"):
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
                params = dict(
                user_data
                )
                api_url = 'https://lifestyle-nexus-540389798753.europe-west1.run.app/predict'
                response = requests.get(api_url, params=params)

                prediction = response.json()


                st.session_state['prediction'] = prediction
                st.session_state.page = "results"
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

# ============================
# صفحة النتائج
# ============================
# ============================
# صفحة النتائج
# ============================
def results_page():
    st.markdown("<h2 style='text-align:center;'>Your Health Prediction Result</h2>", unsafe_allow_html=True)

    if 'prediction' in st.session_state:
        prediction, prob = st.session_state['prediction']["prediction"], st.session_state['prediction']["prob"]

        # بطاقة النتيجة
        if prediction == 1:
            st.markdown(f"""
                <div style='background-color:#D4EDDA; padding:25px; border-radius:15px; text-align:center;'>
                    <h3 style='color:#155724;'>✅ Your lifestyle is healthy with a probability of</h3>
                    <p style='font-size:18px; color:#155724;'>Your health Probability: <b>{prob:.1f}%</b> Keep up the good work maintaining a healthy lifestyle.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style='background-color:#F8D7DA; padding:25px; border-radius:15px; text-align:center;'>
                    <h3 style='color:#721C24;'>⚠️ Your lifestyle is unhealthy </h3>
                    <p style <p style='font-size:18px; color:#721C24;'> Your health Probability: <b>{100-prob:.1f}%</b>and you are at risk of lifestyle diseases. Consider making healthier choices.</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # رسم البيانات المدخلة
        st.markdown("<h4>📊 Your Input Data (Visualization):</h4>", unsafe_allow_html=True)
        numeric_cols = ['age','bmi','daily_steps','sleep_hours','water_intake_l','calories_consumed',
                        'resting_hr','systolic_bp','diastolic_bp','cholesterol']
        numeric_data = user_data[numeric_cols].T
        numeric_data.columns = ['Value']
        st.bar_chart(numeric_data)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#2E8B57;'>💡 Tips to Improve Your Health:</h4>", unsafe_allow_html=True)
        tips = [
            "Increase daily physical activity.",
            "Eat a balanced diet with more vegetables.",
            "Ensure 7-8 hours of sleep per day.",
            "Avoid smoking and limit alcohol consumption."
        ]
        if prediction == 1:
            tips = ["Keep up your healthy habits!", "Maintain exercise and balanced diet.", "Monitor sleep and stress levels."]

        for tip in tips:
         st.markdown(f"""
        <p style='color:#0d3b2e; font-size:16px; margin-bottom:4px;'>&#128161; {tip}</p>
    """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ No prediction found. Please complete a prediction first.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔁 Try Again", key="try_again"):
            st.session_state.page = "prediction"
    with col2:
        if st.button("🏠 Back to Home", key="back_home_results"):
            st.session_state.page = "home"

# ============================
# إدارة الصفحات
# ============================
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "prediction":
    form_page()
elif st.session_state.page == "results":
    results_page()
