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
explainer = shap.TreeExplainer(model)

st.set_page_config(page_title="AI vs Human Classifier", layout="wide")

st.markdown("""
    <style>
    .essay-box {
        background-color: #f5f7fa;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        font-family: 'Courier New', monospace;
        font-size: 0.92rem;
        max-height: 250px;
        overflow-y: auto;
    }
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    .stDataFrame div {
        font-size: 0.85rem;
    }
    /* 🎯 Completely remove Streamlit footer + hamburger */
    #MainMenu {display: none;}
    footer {display: none;}
    /* 🎯 Make number input full width + remove stepper buttons */
    input[type=number] {
        width: 100% !important;
        -moz-appearance: textfield;
    }
    input[type=number]::-webkit-inner-spin-button,
    input[type=number]::-webkit-outer-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load data ---
X_full = pd.read_csv(csv_path)
txt_dir = txt_folder
sample_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt") and f.split(".")[0].isdigit()]
sample_ids = sorted([int(f.split(".")[0]) for f in sample_files if f.split(".")[0].isdigit()])

if not sample_ids:
    st.error("No valid numeric .txt files found in the folder.")
    st.stop()

# --- Layout ---
col1, col2 = st.columns([2, 3])

with col1:
    sample_id = st.number_input(
        "Enter sample number to display essay:",
        min_value=min(sample_ids),
        max_value=max(sample_ids),
        value=min(sample_ids),
        step=1
    )
    sample_id = int(sample_id)

    txt_path = os.path.join(txt_dir, f"{sample_id:03d}.txt")
    with open(txt_path, "r", encoding="utf-8") as f:
        text_input = f.read()

    features_df = X_full.iloc[[sample_id - 1]]
    features = features_df.to_numpy()

    if features.shape[1] != model.n_features_in_:
        st.error(f"Mismatch in feature shape: expected {model.n_features_in_}, got {features.shape[1]}")
        st.stop()

    st.markdown(f"<div class='essay-box'>{text_input}</div>", unsafe_allow_html=True)
    st.markdown("### 📋 Feature Values")
    st.dataframe(features_df.T.rename(columns={features_df.index[0]: "Value"}), height=300)

with col2:
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    label = "🤖 AI" if pred == 0 else "🧑‍🏫 Human"
    confidence = round(np.max(prob) * 100, 2)

    st.markdown(f"### Predicted Label: {label}")

    # 🎯 Custom rounded confidence bar
    bar_color = "#FF4B4B" if pred == 0 else "#1E90FF"
    st.markdown(f"**Confidence:**")
    st.markdown(f"""
        <div style="background-color: #e0e0e0; border-radius: 25px; height: 25px; width: 100%;">
            <div style="
                background-color: {bar_color};
                width: {confidence}%;
                height: 100%;
                border-radius: 25px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
            ">{confidence:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🔍 SHAP Waterfall Plot")
    shap_values = explainer.shap_values(features)
    plt.clf()
    plt.rcParams.update({'font.size': 8})

    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[0],
        feature_names=X_full.columns
    )

    # 🎯 Manually shrink all SHAP fonts
    ax = plt.gca()
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(6)
    ax.title.set_fontsize(6)
    ax.xaxis.label.set_fontsize(6)
    ax.yaxis.label.set_fontsize(6)

    fig = plt.gcf()
    fig.set_size_inches(4, 3)
    st.pyplot(fig, clear_figure=True, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("<small>Built with ❤️ using Streamlit and SHAP • Thesis project edition</small>", unsafe_allow_html=True)
