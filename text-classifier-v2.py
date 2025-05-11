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
sample_ids = sorted([int(f.split(".")[0]) for f in sample_files if f.split(".")[0].isdigit()])

if not sample_ids:
    st.error("No valid numeric .txt files found in the folder.")
    st.stop()

sample_id = st.sidebar.selectbox("Select a sample #", sample_ids)

# Load text and features
txt_path = os.path.join(txt_dir, f"{sample_id:03d}.txt")
with open(txt_path, "r", encoding="utf-8") as f:
    text_input = f.read()

features_df = X_full.iloc[[sample_id - 1]]
features = features_df.to_numpy()

# Validate shape
if features.shape[1] != model.n_features_in_:
    st.error(f"Mismatch in feature shape: expected {model.n_features_in_}, got {features.shape[1]}")
    st.stop()

# Predict
pred = model.predict(features)[0]
prob = model.predict_proba(features)[0]
label = "🤖 AI" if pred == 0 else "🧑‍🏫 Human"
confidence = round(np.max(prob) * 100, 2)

# --- Compact dashboard layout ---
col1, col2 = st.columns([3, 2])

# Left: prediction + SHAP
with col1:
    st.markdown(f"### Predicted Label: {label}")
    st.markdown(f"**Confidence:** {confidence:.1f}%")
    st.progress(max(0.0, min(1.0, float(confidence) / 100.0)))
    st.markdown("#### 🔍 SHAP Contribution Plot")
    shap_values = explainer.shap_values(features)
    plt.clf()
    shap.summary_plot(shap_values, pd.DataFrame(features, columns=X_full.columns),
                      plot_type="bar", show=False)
    st.pyplot(plt, clear_figure=True, use_container_width=True)

# Right: essay + feature table
with col2:
    st.markdown("### 📝 Essay Sample")
    st.markdown(f"<div class='essay-box'>{text_input}</div>", unsafe_allow_html=True)
    st.markdown("### 📋 Feature Values")
    st.dataframe(features_df.T.rename(columns={features_df.index[0]: "Value"}), height=300)

