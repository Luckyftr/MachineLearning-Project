import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.sidebar.title("Navigation")
st.sidebar.write("Gunakan Menu di Kiri untuk Berpindah Halaman")


st.title("⬇️ Download Predictions")

ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = ROOT.parent / "models"

dt = joblib.load(MODELS_PATH / "Decision_Tree.joblib")
xgb = joblib.load(MODELS_PATH / "XGBoost.joblib")
scaler = joblib.load(MODELS_PATH / "scaler.joblib")

uploaded = st.file_uploader("Upload File Cleaned Dataset CSV yang Tersedia", type=["csv"])
model_choice = st.selectbox("Pilih Model", ["Decision Tree","XGBoost"])

if uploaded:
    df = pd.read_csv(uploaded)
    X = df.drop("Outcome", axis=1)

    X_scaled = scaler.transform(X)
    model = dt if model_choice=="Decision Tree" else xgb

    df["Prediction"] = model.predict(X_scaled)

    st.write(df.head())

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )
