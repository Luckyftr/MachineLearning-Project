import streamlit as st
import joblib
import shap
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

st.sidebar.title("Navigation")
st.sidebar.write("Gunakan Menu di Kiri untuk Berpindah Halaman")


st.title("🧠 Model Explainability (SHAP)")

# Paths model
ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = ROOT.parent / "models"

# Load model
model = joblib.load(MODELS_PATH / "XGBoost.joblib")

# Upload dataset
uploaded = st.file_uploader("Upload File Cleaned Dataset CSV yang Tersedia untuk SHAP Analysis", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    if "Outcome" in df.columns:
        X = df.drop("Outcome", axis=1)
    else:
        X = df.copy()  # fallback kalau dataset cuma fitur

    st.write("Preview Dataset:")
    st.write(df.head())

    # SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # ======================
    # Summary Plot
    # ======================
    st.subheader("SHAP Summary Plot")
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=10, show=False)
    st.pyplot(plt.gcf())

    # ======================
    # Dependence Plot - Glucose
    # ======================
    st.subheader("Dependence Plot - Glucose")
    shap.dependence_plot("Glucose", shap_values, X, show=False)
    st.pyplot(plt.gcf())

    # ======================
    # Dependence Plot - BMI
    # ======================
    st.subheader("Dependence Plot - BMI")
    shap.dependence_plot("BMI", shap_values, X, show=False)
    st.pyplot(plt.gcf())

    # ======================
    # Prepare SHAP Data for Download
    # ======================
    if isinstance(shap_values, list):
        # jika binary, shap_values[1] biasanya untuk class 1 (positif)
        shap_to_download = pd.DataFrame(shap_values[1], columns=X.columns)
    else:
        shap_to_download = pd.DataFrame(shap_values, columns=X.columns)

    # Ambil kolom Glucose dan BMI untuk dependence plot
    shap_dep = shap_to_download[["Glucose", "BMI"]]
    shap_dep["Glucose_val"] = X["Glucose"].values
    shap_dep["BMI_val"] = X["BMI"].values


else:
    st.info("Silakan upload dataset terlebih dahulu untuk analisis SHAP.")
