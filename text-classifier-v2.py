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

model = joblib.load(model_path)
X_test = pd.read_csv(csv_path)

txt_files = [f for f in os.listdir(txt_folder) if f.endswith(".txt")]
sample_numbers = sorted([f.replace(".txt", "") for f in txt_files])

# --- Sidebar UI ---
st.sidebar.markdown("## ℹ️ About This App")
st.sidebar.info("This app classifies texts as AI or SLW (Second Language Writers) using syntactic complexity indices and shows SHAP Contribution Plot.")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Choose a sample number (1 to 300)")

selected_input = st.sidebar.text_input(
    label="",
    value="1",
    placeholder="Enter sample number",
    label_visibility="collapsed"
)

# enlarge input box
st.markdown("""
    <style>
    div[data-testid=\"stSidebar\"] input {
        font-size: 1.2rem;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Validate input
try:
    if not selected_input.isdigit() or not (1 <= int(selected_input) <= 300):
        st.sidebar.error("⚠️ Please enter a number between 1 and 300.")
        st.stop()

    selected_sample = selected_input.zfill(3)

except ValueError:
    st.sidebar.error("⚠️ Please enter a valid number.")
    st.stop()

if selected_sample not in sample_numbers:
    st.error("❌ No sample file found for that number.")
    st.stop()

# --- Load sample text and features ---
txt_path = os.path.join(txt_folder, f"{selected_sample}.txt")
with open(txt_path, "r", encoding="utf-8") as file:
    sample_text = file.read()

tassc_index = int(selected_sample) - 1
sample = X_test.iloc[[tassc_index]]

# --- Prediction block ---
pred = model.predict(sample)[0]
prob = model.predict_proba(sample)[0]
confidence = round(max(prob) * 100, 2)

# 🔴 Red for SLW, 🔵 Blue for AI
color = "red" if pred == "SLW" else "dodgerblue"
label = "🧑‍🏫 SLW" if pred == "SLW" else "🤖 AI"

# Dynamic progress bar with embedded text
progress_color = "#1E90FF" if pred == "AI" else "#FF4B4B"  # Blue or Red

st.sidebar.markdown(f"""
<div style='margin-top: 1rem; font-size: 1.1rem;'>
    <b>Classified as:</b><br>
    <span style='font-size: 1.4rem; color:{color}; font-weight:bold;'>{label}</span>
</div>

<div style='margin-top: 1.2rem; font-size: 1.1rem;'>
    <b>Confidence:</b>
    <div style="position: relative; height: 28px; width: 100%; background-color: #ffffff;
                border-radius: 20px; margin-top: 8px; box-shadow: inset 0 0 3px rgba(0,0,0,0.1);">
        <div style="height: 100%; width: {confidence}%; background-color: {progress_color};
                    border-radius: 20px; text-align: center; color: white;
                    font-weight: bold; line-height: 28px; font-size: 1.2rem;">
            {confidence:.1f}%
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- MAIN CONTENT ---
st.title("Streamlit Text Classifier")
st.markdown("""
    <style>
    /* sticky */
    div[data-testid="stTabs"] > div > div:first-child {
        position: sticky;
        top: 3.5rem;
        background-color: white;
        z-index: 999;
        border-bottom: 1px solid #e6e6e6;
        padding-top: 0.5rem;
    }
    
    /* Enlarge */
    button[data-baseweb="tab"] > div {
        font-size: 1.2rem !important;
        padding: 6px 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Tabs for layout
tab1, tab2, tab3 = st.tabs(["📝 Text", "📋 Values", "📉 SHAP Contribution"])

# --- TAB 1: Text Sample ---
with tab1:
    st.subheader(f"📄 Sample #{selected_sample}")
    st.markdown("<div style='margin-top: -10px'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='padding: 12px; background-color: #f9f9f9;
    border: 1px solid #ddd; border-radius: 8px; font-size: 0.95rem;'>
    {sample_text}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 30px; margin-bottom: 10px;'>", unsafe_allow_html=True)

# --- TAB 2: Features ---
with tab2:
    st.subheader("📋 Syntactic Complexity Indices Values")
    st.markdown("<div style='margin-top: -10px'></div>", unsafe_allow_html=True)

    st.dataframe(
        sample.T.rename(columns={sample.index[0]: f"Sample {selected_sample}"}).rename_axis("Index")
    )

    with st.expander("📘 Indices Definitions (Click to Expand)"):
        st.markdown("""
            <div style="padding: 20px; border-radius: 10px;">
                <div style="display: flex; justify-content: space-between; gap: 40px; flex-wrap: wrap;">
                    <ul style="list-style: disc; padding-left: 20px; flex: 1;">
                        <li><b>MLS:</b> <i>Mean Length of Sentence</i></li>
                        <li><b>MLT:</b> <i>Mean Length of T-unit</i></li>
                        <li><b>MLC:</b> <i>Mean Length of Clause</i></li>
                        <li><b>C_S:</b> <i>Clauses per Sentence</i></li>
                        <li><b>VP_T:</b> <i>Verb Phrases per T-unit</i></li>
                        <li><b>CP_T:</b> <i>Coordinate Phrases per T-unit</i></li>
                        <li><b>CP_C:</b> <i>Coordinate Phrases per Clause</i></li>
                    </ul>
                    <ul style="list-style: disc; padding-left: 20px; flex: 1;">
                        <li><b>C_T:</b> <i>Clauses per T-unit</i></li>
                        <li><b>DC_C:</b> <i>Dependent Clauses per Clause</i></li>
                        <li><b>DC_T:</b> <i>Dependent Clauses per T-unit</i></li>
                        <li><b>T_S:</b> <i>T-units per Sentence</i></li>
                        <li><b>CT_T:</b> <i>Complex T-units per T-unit</i></li>
                        <li><b>CN_T:</b> <i>Complex Nominals per T-unit</i></li>
                        <li><b>CN_C:</b> <i>Complex Nominals per Clause</i></li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)


# --- TAB 3: SHAP Contribution ---
with tab3:
    st.subheader("🔍 Top 3 Contributing Indices")
    st.markdown("<div style='margin-top: -10px'></div>", unsafe_allow_html=True)

    explainer = shap.Explainer(model)
    shap_values = explainer(sample)

    single_explanation = shap.Explanation(
        values=shap_values.values[0][:, 1],
        base_values=shap_values.base_values[0][1],
        data=sample.iloc[0].values,
        feature_names=sample.columns.tolist()
    )

    shap_vals = shap_values.values[0][:, 1]
    feature_names = sample.columns.tolist()

    shap_info = []
    for name, val in zip(feature_names, shap_vals):
        direction = "↑ SLW" if val > 0 else "↓ AI"
        shap_info.append((name, val, direction))

    top_features = sorted(shap_info, key=lambda x: abs(x[1]), reverse=True)[:3]

    for i, (name, val, direction) in enumerate(top_features, 1):
        val_display = f"+{val:.3f}" if val > 0 else f"{val:.3f}"
        st.markdown(f"- **{name}** ({val_display}) — *{direction}*")

    st.subheader("📉 SHAP Contribution Plot")
    shap.plots.waterfall(single_explanation, show=False)
    st.pyplot(plt.gcf())

    st.markdown("🔴 = Pushes toward SLW &nbsp;&nbsp;&nbsp; 🔵 = Pushes toward AI", unsafe_allow_html=True)
    st.markdown("<div style='margin-top: 10px'></div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<hr>
<p style='text-align: center; font-size: 0.85rem; color: grey;'>
Built with ❤️ for learning purposes • Powered by Streamlit + SHAP
</p>
""", unsafe_allow_html=True)
