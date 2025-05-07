import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = joblib.load("model_xgb.pkl")
explainer = shap.TreeExplainer(model)

st.set_page_config(page_title="AI vs Human Classifier", layout="wide")

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

st.markdown("<div class='main-title'>🧠 AI vs Human Essay Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Classifies essays using syntactic complexity features and explains decisions using SHAP.</div>", unsafe_allow_html=True)

text_input = st.text_area("✍️ Paste your essay below:", height=300)

st.markdown("---")

def extract_features(text):
    # ⚠️ Placeholder: replace with real feature extraction logic (e.g., from TAASSC)
    return np.zeros(model.n_features_in_)

if st.button("🔍 Classify Text"):
    if not text_input.strip():
        st.warning("Please enter an essay to classify.")
    else:
        features = extract_features(text_input)
        pred = model.predict([features])[0]
        prob = model.predict_proba([features])[0]

        label = "🤖 AI" if pred == 0 else "🧑‍🏫 Human"
        confidence = round(np.max(prob) * 100, 2)
        bar_color = "#1E90FF" if pred == 0 else "#FF4B4B"

        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown(f"### Predicted Label: {label}")
            st.markdown(f"**Confidence:** {confidence:.1f}%")
            st.progress(confidence / 100.0)

        with col2:
            st.markdown("#### 🔍 SHAP Contribution Plot")
            shap_values = explainer.shap_values([features])
            shap.summary_plot(shap_values, pd.DataFrame([features], columns=model.feature_names_in_), plot_type="bar", show=False)
            st.pyplot(plt.gcf())

        st.markdown("---")
        st.subheader("📝 Your Input Essay")
        st.markdown(f"<div class='essay-box'>{text_input}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<small>Built with ❤️ using Streamlit and SHAP • [GitHub](https://github.com)</small>", unsafe_allow_html=True)