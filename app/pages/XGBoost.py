import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
import seaborn as sns
import matplotlib.pyplot as plt
import shap

st.title("Diabetes Prediction + SHAP Explainability (XGBoost)")

# =============================
# Load Model & Scaler
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_PATH = BASE_DIR / "models"

model = joblib.load(MODELS_PATH / "XGBoost.joblib")
scaler = joblib.load(MODELS_PATH / "scaler.joblib")

# =============================
# Input Form Data Pasien
# =============================
st.subheader("Masukkan Data Pasien :")
col1, col2, col3 = st.columns(3)
with col1:
    Pregnancies = st.number_input("Pregnancies", min_value=0, value=2)
    Glucose = st.number_input("Glucose", min_value=0, value=120)
    BloodPressure = st.number_input("Blood Pressure", min_value=0, value=70)
with col2:
    SkinThickness = st.number_input("Skin Thickness", min_value=0, value=20)
    Insulin = st.number_input("Insulin", min_value=0, value=80)
    BMI = st.number_input("BMI", min_value=0.0, value=25.0)
with col3:
    DPF = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
    Age = st.number_input("Age", min_value=1, value=30)

user_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                           Insulin, BMI, DPF, Age]],
                         columns=["Pregnancies","Glucose","BloodPressure","SkinThickness",
                                  "Insulin","BMI","DiabetesPedigreeFunction","Age"])
st.write("### Data Input")
st.write(user_data)

# =============================
# Preprocessing Data Input
# =============================
zero_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
user_data[zero_cols] = user_data[zero_cols].replace(0, np.nan)
user_data = user_data.fillna(user_data.median())
user_scaled = scaler.transform(user_data)

# =============================
# Prediksi Pasien
# =============================
if st.button("Jalankan Prediksi"):
    pred = model.predict(user_scaled)[0]

    st.subheader("📢 Hasil Prediksi Pasien")
    if pred == 1:
        st.error("⚠️ Pasien Diprediksi **DIABETES**")
    else:
        st.success("✅ Pasien Diprediksi **TIDAK DIABETES**")

    # =============================
    # SHAP Explainability - Personal
    # =============================
    st.subheader("SHAP Explainability - Waterfall Plot (Personal)")
    explainer = shap.TreeExplainer(model)
    shap_values_personal = explainer.shap_values(user_scaled)

    if isinstance(shap_values_personal, list):
        shap_val_to_plot = shap_values_personal[1][0]
        base_value = explainer.expected_value[1]
    else:
        shap_val_to_plot = shap_values_personal[0,:] if shap_values_personal.ndim==2 else shap_values_personal[0,:,1]
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value

    explanation = shap.Explanation(
        values=shap_val_to_plot,
        base_values=base_value,
        feature_names=user_data.columns
    )

    shap.plots.waterfall(explanation, max_display=8)
    st.pyplot(plt.gcf())
    plt.clf()

    # =============================
    # Global Evaluation
    # =============================
    st.subheader("Model Evaluation (Global Dataset)")
    test_path = BASE_DIR / "data" / "processed" / "diabetes_clean.csv"
    df = pd.read_csv(test_path)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:,1] if hasattr(model, "predict_proba") else None

    st.write("Accuracy:", accuracy_score(y,y_pred))
    st.write("Precision:", precision_score(y,y_pred))
    st.write("Recall:", recall_score(y,y_pred))
    st.write("F1:", f1_score(y,y_pred))

    # ==== Confusion Matrix Global ====
    st.write("### Confusion Matrix - Global Dataset")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Diabetes","Diabetes"],
                yticklabels=["No Diabetes","Diabetes"])
    st.pyplot(fig)

    # ==== ROC & AUC ====
    if y_prob is not None:
        st.subheader("ROC Curve & AUC")
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)

        fig3, ax3 = plt.subplots()
        ax3.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
        ax3.plot([0,1],[0,1], color='grey', lw=1, linestyle='--')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curve')
        ax3.legend(loc='lower right')
        st.pyplot(fig3)
