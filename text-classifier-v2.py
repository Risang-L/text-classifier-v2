import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Load model and data ---
model = joblib.load("model_xgb.pkl")
X_full = pd.read_csv("X_binary.csv")
txt_folder = "txt_samples"

# --- Configure page ---
st.set_page_config(page_title="AI vs Human Essay Classifier", layout="centered")

# --- Custom CSS for sticky input + tabs ---
st.markdown("""
    <style>
    /* Sticky input + tabs */
    .sticky-container {
        position: sticky;
        top: 1rem;
        background-color: white;
        z-index: 999;
        padding-bottom: 0.5rem;
    }
    div[data-testid="stTabs"] > div > div:first-child {
        position: sticky;
        top: 5rem;
        background-color: white;
        z-index: 998;
    }
    input, button[data-baseweb="tab"] > div {
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sticky Input ---
with st.container():
    st.markdown('<div class="sticky-container">', unsafe_allow_html=True)
    st.title("AI vs Human Essay Classifier")
    sample_id = st.text_input("Sample #", value="1", max_chars=3)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Validate sample ---
if not sample_id.isdigit() or int(sample_id) < 1:
    st.error("⚠️ Please enter a valid sample number (e.g., 1, 2, 3).")
    st.stop()

sample_num = int(sample_id)
txt_path = os.path.join(txt_folder, f"{sample_num:03d}.txt")

if not os.path.exists(txt_path):
    st.error(f"❌ Sample {sample_num} not found.")
    st.stop()

# --- Load text and features ---
with open(txt_path, "r", encoding="utf-8") as f:
    essay_text = f.read()

features_df = X_full.iloc[[sample_num - 1]]
features = features_df.to_numpy()

# --- Predict ---
pred = model.predict(features)[0]
prob = model.predict_proba(features)[0]
confidence = round(np.max(prob) * 100, 1)

label = "🤖 AI" if pred == 0 else "🧑‍🏫 Human"
color = "#1E90FF" if pred == 0 else "#FF0051"

# --- Tabs ---
tab1, tab2 = st.tabs(["📝 Essay & Features", "🔍 Prediction & Explanation"])

# --- Tab 1 ---
with tab1:
    st.subheader("Essay Sample")
    st.markdown(f"""
    <div style='padding: 1rem; background-color: #f5f5f5; border-radius: 8px;
                 border: 1px solid #ddd; font-family: monospace; font-size: 0.95rem;'>
    {essay_text}
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Feature Values")
    st.dataframe(features_df.T.rename(columns={features_df.index[0]: "Value"}))

# --- Tab 2: Prediction and SHAP ---
with tab2:
    st.subheader(f"Predicted Label: {label}")

    # Custom confidence bar
    st.markdown(f"""
    <div style="height: 30px; background-color: #eee; border-radius: 20px; overflow: hidden; margin-bottom: 1rem;">
        <div style="height: 100%; width: {confidence}%; background-color: {color};
                    text-align: center; line-height: 30px; color: white; font-weight: bold; font-size: 1rem;">
            {confidence:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("SHAP Waterfall Plot")
    
    # SHAP plot fix
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # Handle classifier output shape
    if isinstance(explainer.expected_value, np.ndarray):
        pred_class = int(pred)
        base_value = explainer.expected_value[pred_class]
        values = shap_values[pred_class][0]
    else:
        base_value = explainer.expected_value
        values = shap_values[0]

    shap_expl = shap.Explanation(
        values=values,
        base_values=base_value,
        data=features[0],
        feature_names=X_full.columns.tolist()
    )

    plt.clf()
    plt.close('all')
    shap.plots.waterfall(shap_expl, show=False)
    st.pyplot(plt.gcf())

    # Legend
    st.markdown("""
    <div style="margin-top:1rem; font-size: 0.9rem;">
        <span style="color: crimson;">🔴 Pushes toward Human</span> &nbsp;&nbsp;
        <span style="color: dodgerblue;">🔵 Pushes toward AI</span>
    </div>
    """, unsafe_allow_html=True)

