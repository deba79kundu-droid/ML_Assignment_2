"""
=======================================================================
Assignment 2 - Streamlit App
Breast Cancer Classification - Model Comparison Dashboard

Machine Learning - (S2-25_AIMLCZG565)  - Assignment 2
Author: Debashis Kundu (2025AC05781)
=======================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
)

st.set_page_config(page_title="Breast Cancer Classifier Dashboard", layout="wide")

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}

SCALED_MODELS = {"Logistic Regression", "kNN"}

st.title("🩺 Breast Cancer Classification — Model Comparison App")
st.markdown(
    "This app demonstrates **5 classification models** trained on the "
    "Breast Cancer Wisconsin (Diagnostic) dataset (30 features, 569 instances, "
    "binary classification: malignant vs benign)."
)

# ---------------------------------------------------------------
# Sidebar: dataset upload + model selection
# ---------------------------------------------------------------
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload test data (CSV)", type=["csv"],
    help="Upload the provided test_data.csv, or your own CSV with the same 30 feature columns + 'target' column."
)

model_choice = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
else:
    st.sidebar.info("No file uploaded — using bundled sample test_data.csv")
    df = pd.read_csv("test_data.csv")

if "target" not in df.columns:
    st.error("Uploaded CSV must contain a 'target' column (0 = malignant, 1 = benign).")
    st.stop()

X = df.drop(columns=["target"])
y_true = df["target"]

st.subheader("📄 Preview of Data")
st.dataframe(df.head(10))

# ---------------------------------------------------------------
# Load model + (optionally) scaler
# ---------------------------------------------------------------
model = joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[model_choice]))

if model_choice in SCALED_MODELS:
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    X_input = scaler.transform(X)
else:
    X_input = X

y_pred = model.predict(X_input)
y_proba = model.predict_proba(X_input)[:, 1]

# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------
st.subheader(f"📊 Evaluation Metrics — {model_choice}")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
col2.metric("AUC", f"{roc_auc_score(y_true, y_proba):.4f}")
col3.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
col4.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
col5.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")
col6.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

# ---------------------------------------------------------------
# Confusion matrix + classification report
# ---------------------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Malignant", "Benign"],
                yticklabels=["Malignant", "Benign"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with c2:
    st.markdown("**Classification Report**")
    report = classification_report(y_true, y_pred, target_names=["Malignant", "Benign"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3))

# ---------------------------------------------------------------
# All-model comparison table (precomputed at training time)
# ---------------------------------------------------------------
st.subheader("🏆 All Models — Comparison Table")
if os.path.exists(os.path.join(MODEL_DIR, "metrics.csv")):
    metrics_df = pd.read_csv(os.path.join(MODEL_DIR, "metrics.csv"), index_col=0)
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="lightgreen"))
else:
    st.info("Run model/train_models.py first to generate metrics.csv")

st.caption("Built for BITS WILP M.Tech (AIML/DSE) — Machine Learning(S2-25_AIMLCZG565) Author: Debashis")
