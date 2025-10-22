import streamlit as st
import pandas as pd
import numpy as np
import joblib

rf_model = joblib.load("rf_model.joblib")
scaler = joblib.load("scaler.joblib")

feature_cols = [
    'sleep_hours', 'phone_usage_hours', 'caffeine_intake',
    'tasks_planned', 'tasks_completed', 'mood_level',
    'social_media_time', 'productivity_ratio', 'sleep_deficit', 'stress_index'
]


st.set_page_config(page_title="Procrastination Predictor", page_icon="🧠", layout="centered")

st.title("🧠 Procrastination Predictor")
st.markdown("""
Predict the likelihood of procrastination based on your daily habits.

Provide your daily data below and get instant feedback on your productivity level.
""")


st.header("📋 Enter Your Daily Stats")

sleep_hours = st.slider("Sleep Hours", 3.0, 12.0, 7.0, step=0.5)
phone_usage_hours = st.slider("Phone Usage (hours)", 0.0, 10.0, 3.0, step=0.5)
caffeine_intake = st.slider("Caffeine Intake (cups)", 0, 10, 2)
tasks_planned = st.slider("Tasks Planned", 1, 15, 5)
tasks_completed = st.slider("Tasks Completed", 0, 15, 3)
mood_level = st.slider("Mood Level (1 = Bad, 10 = Excellent)", 1, 10, 6)
social_media_time = st.slider("Social Media Time (minutes)", 0, 300, 60, step=5)


productivity_ratio = tasks_completed / (tasks_planned + 1e-6)
sleep_deficit = abs(sleep_hours - 8.0)
stress_index = (phone_usage_hours + social_media_time / 60.0) / (mood_level + 1e-6)

input_data = pd.DataFrame([{
    'sleep_hours': sleep_hours,
    'phone_usage_hours': phone_usage_hours,
    'caffeine_intake': caffeine_intake,
    'tasks_planned': tasks_planned,
    'tasks_completed': tasks_completed,
    'mood_level': mood_level,
    'social_media_time': social_media_time,
    'productivity_ratio': productivity_ratio,
    'sleep_deficit': sleep_deficit,
    'stress_index': stress_index
}])


if st.button("🔮 Predict"):
    prediction = rf_model.predict(input_data)[0]
    probability = rf_model.predict_proba(input_data)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error(f"😴 You are likely **Procrastinating!** (Probability: {probability:.2f})")
        st.markdown("""
        ### 💡 Tips to Improve:
        - Get at least **7–8 hours** of quality sleep.
        - Reduce phone and social media usage during work hours.
        - Plan **fewer but achievable** tasks.
        - Maintain a positive mood with breaks and hydration.
        """)
    else:
        st.success(f"💪 You seem **Productive!** (Probability: {probability:.2f})")
        st.markdown("""
        ### 🌟 Keep It Up:
        - Maintain consistent sleep schedule.
        - Continue completing planned tasks.
        - Avoid distractions when possible.
        - Track your habits daily to stay productive!
        """)



