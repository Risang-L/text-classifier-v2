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
X_data = pd.read_csv(csv_path)
txt_files = [f for f in os.listdir(txt_folder) if f.endswith(".txt")]
sample_numbers = sorted([f.replace(".txt", "") for f in txt_files])

# --- Sidebar input ---
st.sidebar.title("AI vs Human Essay Classifier")
sticky_sample = st.sidebar.text_input("Enter sample #", value="1")

if not sticky_sample.isdigit() or not (1 <= int(sticky_sample) <= len(X_data)):
    st.sidebar.error("Enter a valid number between 1 and {}".format(len(X_data)))
    st.stop()

sample_id = sticky_sample.zfill(3)
if f"{sample_id}.txt" not in txt_files:
    st.error("No text file found for this sample.")
    st.stop()

# --- Load text and features ---
txt_path = os.path.join(txt_folder, f"{sample_id}.txt")
with open(txt_path, "r", encoding="utf-8") as file:
    text_input = file.read()
features_df = X_data.iloc[[int(sample_id)-1]]

# --- Prediction ---
pred = model.predict(features_df)[0]
proba = model.predict_proba(features_df)[0]
confidence = round(np.max(proba) * 100, 2)
label = "🤖 AI" if pred == 0 else "🧑‍🏫 Human"
bar_color = "#1E90FF" if pred == 0 else "#FF4B4B"

# --- Confidence bar ---
st.sidebar.markdown(f"**Predicted Label:** {label}")
st.sidebar.markdown("**Confidence:**")
st.sidebar.markdown(f"""
<div style="position: relative; height: 28px; background-color: #ddd; border-radius: 14px;">
    <div style="background-color: {bar_color}; width: {confidence}%; height: 100%; border-radius: 14px; text-align: center; color: white; font-weight: bold; line-height: 28px;">
        {confidence:.1f}%
    </div>
</div>
""", unsafe_allow_html=True)

# --- Main content ---
st.markdown("<style>div[data-testid=\"stTabs\"] div:first-child {position: sticky; top: 0; background: white; z-index: 99;}</style>", unsafe_allow_html=True)
tabs = st.tabs(["📝 Essay & Features", "🤖 Prediction & Explanation"])

# --- Tab 1: Essay & Features ---
with tabs[0]:
    st.header("Essay Sample")
    st.code(text_input, language="text")

    st.header("Feature Values")
    st.dataframe(features_df.T.rename(columns={features_df.index[0]: "Value"}))

# --- Tab 2: Prediction & SHAP ---
with tabs[1]:
    st.header("Prediction & SHAP Waterfall Plot")
    explainer = shap.Explainer(model)
    shap_values = explainer(features_df)
    single_explanation = shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        data=features_df.iloc[0].values,
        feature_names=features_df.columns.tolist()
    )
    shap.plots.waterfall(single_explanation, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    <div style="text-align: center;">
        🔴 = Pushes toward SLW (Human) &nbsp;&nbsp;&nbsp; 🔵 = Pushes toward AI
    </div>
    """, unsafe_allow_html=True)
