"""
Project: Breast Cancer Classification - Model Comparison Dashboard
Machine Learning - (S2-25_AIMLCZG565)  - Assignment 2
Author: Debashis Kundu (2025AC05781)

Dataset: Breast Cancer Wisconsin (Diagnostic) - sklearn built-in
569 instances, 30 features, binary classification (malignant/benign)

Trains 5 classifiers, computes evaluation metrics, saves models + test data
for use by the Streamlit app.
"""
import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------
# Step 1: Load dataset
# ---------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign

print(f"Dataset shape: {X.shape}, classes: {y.unique()}")

# ---------------------------------------------------------------
# Step 2: Train/test split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (helps Logistic Regression, kNN especially)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))

# Save the test data (features + true label) - this is what gets
# uploaded to the Streamlit app and what goes in the GitHub repo
test_data = X_test.copy()
test_data["target"] = y_test.values
test_data.to_csv(os.path.join(OUT_DIR, "..", "test_data.csv"), index=False)
print("Saved test_data.csv")

# ---------------------------------------------------------------
# Step 3: Define models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "kNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=200, random_state=42
    ),
}

results = {}

for name, model in models.items():
    # Logistic Regression and kNN benefit from scaled data
    if name in ["Logistic Regression", "kNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results[name] = metrics

    # save each model
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, os.path.join(OUT_DIR, f"{fname}.pkl"))
    print(f"{name}: {metrics}")

# ---------------------------------------------------------------
# Step 4: Save metrics table (used by README + Streamlit app)
# ---------------------------------------------------------------
results_df = pd.DataFrame(results).T
results_df.index.name = "ML Model Name"
results_df.to_csv(os.path.join(OUT_DIR, "metrics.csv"))

with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n=== Final Comparison Table ===")
print(results_df)
