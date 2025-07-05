import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("models/heart_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Title
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("💓 Heart Disease Risk Prediction")
st.markdown("Enter your health information to predict your risk of heart disease.")

# Sidebar for input
st.sidebar.header("📝 Input Health Information")

# Input fields
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
physical_health = st.sidebar.slider("Physical Health (days of poor health in past 30 days)", 0, 30, 5)
mental_health = st.sidebar.slider("Mental Health (days of poor health in past 30 days)", 0, 30, 5)
sleep_time = st.sidebar.slider("Average Sleep Time (hours/day)", 0, 24, 7)

smoking = st.sidebar.selectbox("Do you smoke?", ["No", "Yes"])
alcohol = st.sidebar.selectbox("Do you drink alcohol?", ["No", "Yes"])
stroke = st.sidebar.selectbox("Have you ever had a stroke?", ["No", "Yes"])
diff_walking = st.sidebar.selectbox("Difficulty walking or climbing stairs?", ["No", "Yes"])

diabetic = st.sidebar.selectbox("Are you diabetic?", [
    "No", "Yes", "No, Borderline Diabetes", "Yes (During Pregnancy)"
])

physical_activity = st.sidebar.selectbox("Do you do physical activity?", ["No", "Yes"])
asthma = st.sidebar.selectbox("Do you have asthma?", ["No", "Yes"])
kidney_disease = st.sidebar.selectbox("Do you have kidney disease?", ["No", "Yes"])
skin_cancer = st.sidebar.selectbox("Do you have skin cancer?", ["No", "Yes"])

# Convert categorical inputs
def convert_input(val):
    return 1 if val == "Yes" else 0

diabetic_map = {
    "No": 0,
    "Yes": 1,
    "No, Borderline Diabetes": 0.5,
    "Yes (During Pregnancy)": 0.5
}

input_data = np.array([[
    bmi,
    physical_health,
    mental_health,
    sleep_time,
    convert_input(smoking),
    convert_input(alcohol),
    convert_input(stroke),
    convert_input(diff_walking),
    diabetic_map[diabetic],
    convert_input(physical_activity),
    convert_input(asthma),
    convert_input(kidney_disease),
    convert_input(skin_cancer)
]])

# Scale input
scaled_input = scaler.transform(input_data)

# Predict
if st.sidebar.button("Predict"):
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    st.subheader("📊 Prediction Result")
    if prediction[0] == 1:
        st.error("⚠️ You are **likely at risk** of heart disease.")
    else:
        st.success("✅ You are **not likely at risk** of heart disease.")

    st.markdown(f"**Confidence:** {np.max(prediction_proba) * 100:.2f}%")

# Footer
st.markdown("---")
st.markdown("📌 This tool is for informational purposes only and does not replace professional medical advice.")
