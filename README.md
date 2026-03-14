## 🫀 Heart Disease Prediction System

An end-to-end Machine Learning + Explainable AI project that predicts the likelihood of heart disease using clinical patient data.

**This project demonstrates the complete ML lifecycle:**

✅ Data preprocessing & feature engineering

✅ Exploratory Data Analysis (EDA)

✅ Model comparison & evaluation

✅ Hyperparameter tuning

✅ Explainable AI using SHAP

✅ Deployment with Streamlit

---


## 📊 Dataset

The dataset used for this project is publicly available on Kaggle:

👉 https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

Dataset Author: fedesoriano (Kaggle)

**Features Included**

✅ Age

✅ Sex

✅ Chest Pain Type

✅ Resting Blood Pressure

✅ Cholesterol

✅ Fasting Blood Sugar

✅ Resting ECG

✅ Maximum Heart Rate

✅ Exercise-Induced Angina

✅ Oldpeak

✅ ST Slope

✅ Heart Disease (Target)


---

## 🧠 Model Performance

Final tuned Gradient Boosting Classifier results:

| Metric    | Score     |
| --------- | --------- |
| Accuracy  | **0.875** |
| Precision | **0.889** |
| Recall    | **0.897** |
| F1 Score  | **0.893** |
| ROC-AUC   | **0.931** |

**Why This Matters**

✅ High Recall → detects most heart disease cases

✅ Strong ROC-AUC → excellent class separation

✅ Balanced metrics → clinically reliable predictions

**Why Recall Was Optimized**

In medical prediction systems:

➡️ Missing a true heart disease case is dangerous.

Therefore, hyperparameter tuning optimized Recall to minimize false negatives.

---

## 🔍 Explainable AI (SHAP)

The project integrates SHAP **(SHapley Additive Explanations)** to make predictions interpretable.

Users can see:

✅ Features increasing risk 🔴

✅ Features decreasing risk 🔵

This makes the model suitable for healthcare decision support.

---

## 🌐 Streamlit Application (app.py)

The Streamlit interface enables:

✅ Interactive patient input

✅ Real-time prediction

✅ Risk probability visualization

✅ SHAP explanation dashboard

**Example Output**

⚠️ High Risk of Heart Disease

✅ Low Risk of Heart Disease

Feature contribution explanation

---

## 🚀 Live Application

The app is deployed and accessible online — no installation required.

👉 **[Try it live here](https://heart-disease-prediction-gry5jjqbrx9qfq6cwqybsv.streamlit.app/)**

Simply input patient health information and instantly receive:

✅ Risk prediction

✅ Probability score

✅ Explainable AI visualization showing contributing factors

---

## 🛠 Local Installation

Prefer to run it locally? Follow these steps:

1. Clone Repository
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Create a virtual environment and activate it
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

4. Train the model
```bash
python src/train.py
```

5. Run the application
```bash
streamlit run src/app.py
```

---

## ⚠️ Disclaimer

This project is for educational and research purposes only.

It is not a medical diagnostic tool and should not replace professional medical advice.

---

## ⭐ Support

If you found this project useful:

⭐ Star the repository
🍴 Fork the project
🤝 Connect for collaborations

---
