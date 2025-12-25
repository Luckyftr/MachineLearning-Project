import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.sidebar.title("Navigation")
st.sidebar.write("Gunakan Menu di Kiri untuk Berpindah Halaman")

st.title("⚙️ Run Model & Evaluation")

ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = ROOT.parent / "models"

dt = joblib.load(MODELS_PATH / "Decision_Tree.joblib")
xgb = joblib.load(MODELS_PATH / "XGBoost.joblib")
scaler = joblib.load(MODELS_PATH / "scaler.joblib")

uploaded = st.file_uploader("Upload Data Cleaned CSV yang Tersedia untuk Prediksi", type=["csv"])
model_choice = st.selectbox("Pilih Model", ["Decision Tree","XGBoost"])

if uploaded:
    df = pd.read_csv(uploaded)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_scaled = scaler.transform(X)

    model = dt if model_choice == "Decision Tree" else xgb
    y_pred = model.predict(X_scaled)

    st.subheader("Metrics")
    st.write("Accuracy:", accuracy_score(y,y_pred))
    st.write("Precision:", precision_score(y,y_pred))
    st.write("Recall:", recall_score(y,y_pred))
    st.write("F1:", f1_score(y,y_pred))

    cm = confusion_matrix(y,y_pred)
    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Diabetes","Diabetes"],
        yticklabels=["No Diabetes","Diabetes"]
    )
    st.pyplot(fig)
else:
    st.info("Upload data dulu")
