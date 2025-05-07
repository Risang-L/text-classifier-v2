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
model = joblib.load("model_xgb.pkl")
explainer = shap.TreeExplainer(model)

# Config
st.set_page_config(page_title="AI vs Human Classifier", layout="wide")

# Styles
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
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-title'>🧠 AI vs Human Essay Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Classifies essays using syntactic complexity features and explains decisions using SHAP.</div>", unsafe_allow_html=True)

# Load TAASSC features
csv_path = "X_binary.csv"  # You can change this path to match your thesis data
X_full = pd.read_csv(csv_path)
txt_dir = "txt_samples"  # Directory containing .txt files

# Sidebar sample selection
sample_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt") and f.split(".")[0].isdigit()]
sample_ids = []
for f in sample_files:
    try:
        sample_ids.append(int(f.split(".")[0]))
    except ValueError:
        continue
sample_ids = sorted(sample_ids)

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
try:
    if features.shape[1] != model.n_features_in_:
        st.error(f"Mismatch in feature shape: expected {model.n_features_in_}, got {features.shape[1]}")
        st.stop()
except Exception as e:
    st.error(f"Feature shape check failed: {e}")
    st.stop()

# Predict
pred = model.predict(features)[0]
prob = model.predict_proba(features)[0]
label = "🤖 AI" if pred == 0 else "🧑‍🏫 Human"
confidence = round(np.max(prob) * 100, 2)
bar_color = "#1E90FF" if pred == 0 else "#FF4B4B"

# Display results
col1, col2 = st.columns([2, 3])
with col1:
    st.markdown(f"### Predicted Label: {label}")
    st.markdown(f"**Confidence:** {confidence:.1f}%")
    try:
        progress_value = max(0.0, min(1.0, float(confidence) / 100.0))
        st.progress(progress_value)
    except Exception as e:
        st.warning(f"⚠️ Unable to display progress bar: {e}")

with col2:
    st.markdown("#### 🔍 SHAP Contribution Plot")
    shap_values = explainer.shap_values(features)
    shap.summary_plot(shap_values, pd.DataFrame(features, columns=X_full.columns), plot_type="bar", show=False)
    st.pyplot(plt.gcf())

# Essay display
st.markdown("---")
st.subheader("📝 Essay Sample")
st.markdown(f"<div class='essay-box'>{text_input}</div>", unsafe_allow_html=True)

# Feature display
st.markdown("---")
st.subheader("📋 Feature Values")
st.dataframe(features_df.T.rename(columns={features_df.index[0]: "Value"}))

# Footer
st.markdown("---")
st.markdown("<small>Built with ❤️ using Streamlit and SHAP • Thesis project edition</small>", unsafe_allow_html=True)
