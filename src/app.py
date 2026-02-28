import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Heart Disease Predictor", page_icon="🫀", layout="centered"
)

st.title("🫀 Heart Disease Predictor")
st.write("Fill in the patient details below and click **Predict**.")

# Load models
model = joblib.load("models/best_heart_disease_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")

st.divider()

# ── Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 90, 54)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
    resting_bp = st.number_input("Resting Blood Pressure (mmHg)", 60, 220, 130)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 0, 600, 200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
    exercise_ang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("Oldpeak", -3.0, 7.0, 1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Down", "Flat", "Up"])

st.divider()

# ── Predict
if st.button("Predict", width="stretch", type="primary"):
    raw_num = pd.DataFrame(
        [[resting_bp, cholesterol, max_hr, oldpeak]],
        columns=["RestingBP", "Cholesterol", "MaxHR", "Oldpeak"],
    )
    scaled = scaler.transform(raw_num)[0]

    row = {f: 0 for f in features}
    row["Age"] = age
    row["Sex"] = 1 if sex == "Male" else 0
    row["FastingBS"] = 1 if fasting_bs == "Yes" else 0
    row["ExerciseAngina"] = 1 if exercise_ang == "Yes" else 0
    row["RestingBP"] = scaled[0]
    row["Cholesterol"] = scaled[1]
    row["MaxHR"] = scaled[2]
    row["Oldpeak"] = scaled[3]
    row["ST_Slope"] = {"Down": 0, "Flat": 1, "Up": 2}[st_slope]

    for cp in ["ATA", "NAP", "TA"]:
        row[f"ChestPainType_{cp}"] = 1 if chest_pain == cp else 0

    for ecg in ["Normal", "ST"]:
        row[f"RestingECG_{ecg}"] = 1 if resting_ecg == ecg else 0

    X = pd.DataFrame([row])[features]
    prob = model.predict_proba(X)[0][1]

    # ── Result
    if prob >= 0.5:
        st.error(
            f"### ⚠️ High Risk of Heart Disease ({prob * 100:.1f}%)\nPlease consult a cardiologist."
        )
    else:
        st.success(
            f"### ✅ Low Risk of Heart Disease ({prob * 100:.1f}%)\nNo strong indicators detected."
        )

    st.progress(prob, text=f"Risk probability: {prob * 100:.1f}%")

    # ── SHAP Explanation
    st.divider()
    st.subheader("🔍 Why this prediction?")
    st.caption(
        "The chart below shows which features pushed the risk up (red) or down (blue)."
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # expected_value can be an array for GradientBoosting — extract scalar
    base_val = explainer.expected_value
    if hasattr(base_val, "__len__"):
        base_val = float(base_val[-1])
    else:
        base_val = float(base_val)

    # shap_values can be a list — take positive-class values
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    sv = np.array(sv[0], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=sv,
            base_values=base_val,
            data=X.iloc[0].values,
            feature_names=features,
        ),
        max_display=10,
        show=False,
    )
    st.pyplot(fig, width="stretch")
# streamlit run src/app.py
