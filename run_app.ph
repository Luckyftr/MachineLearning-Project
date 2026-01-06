#!/bin/bash
# ========================================
# run_app.sh
# Script untuk setup environment & run Streamlit App
# ========================================

echo "=== STEP 0: Buka Anaconda Powershell Prompt ==="
echo "Pastikan Anda sudah membuka Anaconda Powershell Prompt"

echo "=== STEP 1: Buat & aktifkan environment ==="
conda create -n diabetes_app python=3.12 -y
conda activate diabetes_app

echo "=== STEP 2: Install library yang dibutuhkan ==="
pip install streamlit joblib numpy pandas scikit-learn matplotlib -q
pip install ipython -q
pip install xgboost -q
pip install seaborn -q
pip install shap -q



echo "=== STEP 3: Pindah ke folder app & jalankan Streamlit ==="
# Ganti path di bawah sesuai lokasi folder app di komputer/GitHub
cd "C:/laragon/www/MachineLearning-Project/app"
streamlit run app.py
