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

st.set_page_config(page_title="AI vs Human Essay Classifier")

st.markdown("""
    <style>
    .essay-box {
        background-color: #f5f7fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
        font-family: 'Courier New', monospace;
        font-size: 0.92rem;
        max-height: 300px;
        overflow-y: auto;
    }
    .stDataFrame div {
        font-size: 0.85rem;
    }
    #MainMenu {display: none;}
    footer {display: none;}
    </style>
""", unsafe_allow_html=True)

st.title("AI vs Human Essay Classifier")

# --- Load data ---
X_full = pd.read_csv(csv_path)
txt_dir = txt_folder
sample_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt") and f.split(".")[0].isdigit()]
sample_ids = sorted([int(f.split(".")[0]) for f in sample_files if f.split(".")[0].isdigit()])

if not sample_ids:
    st.error("No valid numeric .txt files found in the folder.")
    st.stop()

sample_input = st.text_input("Sample #:", value=str(min(sample_ids)))
try:
    sample_id = int(sample_input)
    if sample_id not in sample_ids:
        st.error(f"Please enter a valid sample number between {min(sample_ids)} and {max(sample_ids)}")
        st.stop()
except ValueError:
    st.error("Please enter a valid integer sample number.")
    st.stop()

txt_path = os.path.join(txt_dir, f"{sample_id:03d}.txt")
with open(txt_path, "r", encoding="utf-8") as f:
    text_input = f.read()

features_df = X_full.iloc[[sample_id - 1]]
features = features_df.to_numpy()

if features.shape[1] != model.n_features_in_:
    st.error(f"Mismatch in feature shape: expected {model.n_features_in_}, got {features.shape[1]}")
    st.stop()

# --- Tabs layout ---
tab1, tab2 = st.tabs(["📝 Essay & Features", "🤖 Prediction & Explanation"])

with tab1:
    st.subheader("Essay Sample")
    st.markdown(f"<div class='essay-box'>{text_input}</div>", unsafe_allow_html=True)

    st.subheader("Feature Values")
    st.dataframe(features_df.T.rename(columns={features_df.index[0]: "Value"}), height=300)

with tab2:
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    label = "🤖 AI" if pred == 0 else "🧑‍🏫 Human"
    confidence = round(np.max(prob) * 100, 2)

    st.subheader("Prediction Result")
    st.markdown(f"### Predicted Label: {label}")

    # 🎯 Custom confidence bar
    bar_color = "#1E90FF" if pred == 0 else "#FF4B4B" 
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

    st.subheader("SHAP Waterfall Plot")
    shap_values = explainer.shap_values(features)

    # 🎯 Full SHAP waterfall with feature values
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=features[0],
        feature_names=X_full.columns
    ))
    st.pyplot(plt.gcf(), clear_figure=True, use_container_width=True)
