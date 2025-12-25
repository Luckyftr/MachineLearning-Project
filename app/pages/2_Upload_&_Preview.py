import streamlit as st
import pandas as pd
import numpy as np

st.sidebar.title("Navigation")
st.sidebar.write("Gunakan Menu di Kiri untuk Berpindah Halaman")

st.title("📂 Upload & Preview Data")

uploaded = st.file_uploader("Upload File Raw Dataset CSV yang Tersedia", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Preview Data (Before Preprocessing)")
    st.write(df.head())
    st.write("Shape:", df.shape)

    # Cleaning: ganti 0 → NaN untuk kolom medis
    zero_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    df[zero_cols] = df[zero_cols].replace(0, np.nan)

    st.subheader("After Preprocessing (Missing handled)")
    st.write(df.head())


else:
    st.info("Silakan upload dataset terlebih dahulu.")
