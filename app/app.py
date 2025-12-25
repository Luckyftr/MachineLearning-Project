import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Diabetes ML App",
    layout="wide"
)

ROOT_DIR = Path(__file__).resolve().parents[1]

st.sidebar.title("Navigation")
st.sidebar.write("Gunakan menu di kiri untuk berpindah halaman")

st.title("Diabetes Prediction System")
st.write("""
Aplikasi ini dibuat sebagai implementasi Machine Learning untuk prediksi diabetes.
Dataset diupload oleh user, preprocessing dilakukan otomatis, dan model ML sudah tersedia.
""")
