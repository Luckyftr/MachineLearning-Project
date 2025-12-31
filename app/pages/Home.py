import streamlit as st

st.sidebar.title("Navigation")
st.sidebar.write("Gunakan Menu di Kiri untuk Berpindah Halaman")


st.title("🏠 Home")
st.write("""
## Diabetes Prediction Project
Project ini memanfaatkan algoritma Machine Learning untuk memprediksi kemungkinan diabetes.

### Dataset
- Sumber: Kaggle (NIDDK)
- Fitur: 8
- Target: Outcome (0 = Tidak Diabetes, 1 = Diabetes)
- Preprocessing otomatis:
  - Nilai 0 dianggap missing
  - Imputasi median
  - Standard Scaler

### Tujuan
- Prediksi probabilitas diabetes
- Interpretasi model dengan SHAP
- Unduh hasil prediksi
""")
