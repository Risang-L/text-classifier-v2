import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Load model and data ---
model = joblib.load("model_xgb.pkl")
X_test = pd.read_csv("X_binary.csv")
txt_folder = "txt_samples"

txt_files = [f for f in os.listdir(txt_folder) if f.endswith(".txt")]
sample_numbers = sorted([f.replace(".txt", "") for f in txt_files])

# --- UI Styles ---
st.set_page_config(page_title="AI vs Human Essay Classifier", layout="wide")
st.markdown("""
    <style>
    /* Sticky entire header block including title + input + tabs */
    div[data-testid="stAppViewContainer"] > div:first-child {
        position: sticky;
        top: 0;
        background-color: white;
        z-index: 999;
        border-bottom: 1px solid #e6e6e6;
        padding-bottom: 10px;
    }
    /* Enlarge tab labels */
    button[data-baseweb="tab"] > div {
        font-size: 1.1rem !important;
        padding: 6px 12px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title + Input ---
st.title("AI vs Human Essay Classifier")

selected_input = st.text_input("Sample #:", "1")
if not selected_input.isdigit() or int(selected_input) <= 0:
    st.error("Please enter a valid positive number.")
    st.stop()

selected_sample = selected_input.zfill(3)
if selected_sample not in sample_numbers:
    st.error("Sample not found.")
    st.stop()

# --- Load sample text and features ---
txt_path = os.path.join(txt_folder, f"{selected_sample}.txt")
with open(txt_path, "r", encoding="utf-8") as file:
    sample_text = file.read()

tassc_index = int(selected_sample) - 1
sample = X_test.iloc[[tassc_index]]

# --- Prediction ---
pred = model.predict(sample)[0]
prob = model.predict_proba(sample)[0]
confidence = round(max(prob) * 100, 2)

label = "🧑‍🏫 Human" if pred == 1 else "🤖 AI"
bar_color = "#FF4B4B" if pred == 1 else "#1E90FF"

# --- Tabs ---
tab1, tab2 = st.tabs(["📝 Essay & Features", "🤖 Prediction & Explanation"])

# --- Tab 1 ---
with tab1:
    st.subheader("Essay Sample")
    st.markdown(f"""
    <div style='padding: 12px; background-color: #f9f9f9;
    border: 1px solid #ddd; border-radius: 8px; font-family: monospace; font-size: 0.95rem;'>
    {sample_text}
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Feature Values")
    st.dataframe(sample.T.rename(columns={sample.index[0]: "Value"}))

# --- Tab 2 ---
with tab2:
    st.subheader(f"Predicted Label: {label}")
    st.markdown(f"""
    <div style="position: relative; height: 28px; width: 100%; background-color: #eee;
                border-radius: 20px; margin-top: 8px; box-shadow: inset 0 0 3px rgba(0,0,0,0.1);">
        <div style="height: 100%; width: {confidence}%; background-color: {bar_color};
                    border-radius: 20px; text-align: center; color: white;
                    font-weight: bold; line-height: 28px; font-size: 1.0rem;">
            {confidence:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🔍 SHAP Waterfall Plot")
    explainer = shap.Explainer(model)
    shap_values = explainer(sample)

    single_explanation = shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        data=sample.iloc[0].values,
        feature_names=sample.columns.tolist()
    )

    # Show full waterfall plot
    shap.plots.waterfall(single_explanation, show=False)
    st.pyplot(plt.gcf())

    st.markdown("🔴 = Pushes toward Human (SLW) &nbsp;&nbsp;&nbsp; 🔵 = Pushes toward AI", unsafe_allow_html=True)
