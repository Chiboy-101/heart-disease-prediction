# Import necessary libraries and modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Advanced Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Load the dataset
df = pd.read_csv("heart.csv")
df.head()

# Check if the dataset is evenly distributed
df.info()
df.size
df["HeartDisease"].value_counts()  # Check the distribution of the target variable

# Data Cleaning and Preprocessing
df.isnull().sum()  # Check for missing values
df.drop_duplicates(inplace=True)  # Remove duplicate rows if any

# --- Since there are no missing values, we can proceed with encoding categorical variables

## Encode binary categorical variables
df["Sex"] = df["Sex"].str.strip()
df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
df["ExerciseAngina"] = df["ExerciseAngina"].str.strip()
df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y": 1, "N": 0})

## Encode Nominal Categorical Variables using One-Hot Encoding
df = pd.get_dummies(df, columns=["ChestPainType", "RestingECG"], drop_first=True)

## Encode Ordinal Categorical Variables using Label Encoding
le = LabelEncoder()
df["ST_Slope"] = le.fit_transform(df["ST_Slope"])

# --- Now perform EDA to understand the relationships between features and the target variable

## Correlation Heatmap to show correlations between features and the target variable
corr_matrix = df.corr()
target_corr = corr_matrix["HeartDisease"].sort_values(ascending=False)
print(target_corr)

plt.figure(figsize=(12, 8))
sns.heatmap(target_corr.to_frame(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix - Heart Disease Dataset")
plt.show()

## Bar plot to show the correlation of each feature with the target variable
target_corr.drop("HeartDisease").plot(kind="bar", figsize=(10, 6))

plt.title("Feature Correlation With Heart Disease")
plt.ylabel("Correlation")
plt.show()

## Plot Age distribution for patients with and without heart disease
sns.histplot(data=df, x="Age", hue="HeartDisease", kde=True)

plt.title("Age Distribution by Heart Disease")
plt.show()

## Plot Chest Pain Type distribution for patients with and without heart disease

# correlation with target
corr = df.corr()["HeartDisease"].sort_values(ascending=False)

# show only chest pain columns
corr.filter(like="ChestPainType")

cp_corr = corr.filter(like="ChestPainType")

cp_corr.plot(kind="bar", figsize=(8, 5))

plt.title("Correlation of Chest Pain Types With Heart Disease")
plt.ylabel("Correlation")
plt.show()

# Plot outliers in the dataset using boxplots for numerical features
numerical_features = ["RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[feature])
    plt.title(f"Boxplot of {feature}")
    plt.show()

# Scale numerical features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# df.head()
# df.info()

# Save the preprocessed dataset to a new CSV file
df.to_csv("df_new.csv", index=False)

# Load the preprocessed dataset
df_new = pd.read_csv("df_new.csv")

# Select features and target variable for modeling
X = df_new.drop("HeartDisease", axis=1)
y = df_new["HeartDisease"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a dictionary to store the models and their names
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=28),
    "Random Forest": RandomForestClassifier(random_state=28),
    "Gradient Boosting": GradientBoostingClassifier(random_state=28),
    "Support Vector Machine": SVC(random_state=28),
}

# Train each model and evaluate its performance
results = []

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # If the model supports probability scores, get them for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:  # e.g., SVM without probability=True
        y_prob = model.decision_function(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    # Append results
    results.append(
        {
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": roc,
        }
    )

# Create a DataFrame to display results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="ROC-AUC", ascending=False)
print(results_df)

# Plot the performance of each model based on ROC-AUC
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="ROC-AUC", data=results_df)
plt.title("Model Performance Comparison (ROC-AUC)")
plt.ylabel("ROC-AUC Score")
plt.ylim(0.5, 1)
plt.xticks(rotation=45)
plt.show()

# Hyperparameter Tuning for the best performing model
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 4, 5],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

gb_model = GradientBoostingClassifier(
    random_state=28
)  # Best performing model based on ROC-AUC

# Use GridSearchCV to find the best hyperparameters for the Gradient Boosting model
grid_search = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid,
    cv=5,
    scoring="recall",  # Optimize for recall to minimize false negatives
    n_jobs=-1,
    verbose=2,
)

grid_search.fit(X_train, y_train)  # Fit the model with the training data

print("Best Parameters:", grid_search.best_params_)  # Best hyperparameters
print("Best Recall:", grid_search.best_score_)  # Best recall score

best_model = (
    grid_search.best_estimator_
)  # Get the best model with the optimal hyperparameters

# Make predictions with the best model
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Calculate metrics for the best model
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc}")
print(f"Precision: {prec}")
print(f"Recall: {rec}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC: {roc}")

# Get feature importance
importance = best_model.feature_importances_

feature_importance = pd.Series(importance, index=X_train.columns).sort_values(
    ascending=False
)

# Plot
plt.figure(figsize=(10, 6))
feature_importance.head(15).plot(kind="bar")

plt.title("Top Feature Importance - Heart Disease Prediction")
plt.ylabel("Importance Score")
plt.show()

from sklearn.metrics import roc_curve, auc

# Probability predictions
y_prob = best_model.predict_proba(X_test)[:, 1]

# ROC values
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Heart Disease Model")
plt.legend()
plt.show()

# Model Explanation using SHAP
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# Save the best model, scaler, features and label encoder
import joblib

joblib.dump(best_model, "best_heart_disease_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(X_train.columns.tolist(), "features.pkl")
