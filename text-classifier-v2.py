import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Load model and data ---
model_path = "model_xgb.pkl"
csv_path = "X_binary.csv"
txt_folder = "txt_samples"

# Load model and SHAP explainer
model = joblib.load(model_path)
explainer = shap.TreeExplainer(model)

# Config
st.set_page_config(page_title="AI vs Human Classifier", layout="wide")

# Minimize vertical padding
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1E90FF;
    }
    .sub-text {
        font-size: 1.1rem;
        color: #555;
    }
    .essay-box {
        background-color: #f5f7fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        font-family: 'Courier New', monospace;
        font-size: 0.95rem;
        max-height: 250px;
        overflow-y: auto;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-title'>🧠 AI vs Human Essay Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Classifies essays using syntactic complexity features and explains decisions using SHAP.</div>", unsafe_allow_html=True)

# Load TAASSC features
X_full = pd.read_csv(csv_path)
txt_dir = txt_folder

# Sidebar sample selection
sample_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt") and f.split(".")[0].isdigit()]
sample_ids = sorted([int(f.split(".")[0]) for f in sample_files if f.split(".")[0].isd_]()_
