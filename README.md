
# Text Classifier v2: AI vs SLW (Second Language Writers)
This project is an extension of [Text Classifier v1](https://github.com/Risang-L/text-classifier), using a stronger **XGBoost model** and larger dataset. It explores linguistic data science and syntactic complexity modeling.

[Live App](https://text-classifier-v2-tglormcyfegrxzctqeiylu.streamlit.app/)

---

## Improvements over v1

- Upgraded from Random Forest to **XGBoost**
- More dataset for training (from 300 → 1000 samples)
- More stable predictions and improved feature interpretability
  
---

## About the syntactic Complexity Indices

The classifier uses L2SCA indices by [TAASSC](https://www.linguisticanalyistools.org/taassc.html).

Predictions are supported by SHAP contribution plots, showing how each feature influences the outcome toward AI or SLW.

---

## Data Overview

The dataset used for model training consists of **1,000 writing samples** (500 human, 500 AI):  

- **Human-written**:  
    500 essays by second language writers (SLW), sourced from [ICNALE](https://language.sakura.ne.jp/icnale/)  
- **AI-generated**:  
    500 essays generated by large language models (LLMs), sourced from [LLM-generated Essay Dataset](https://huggingface.co/datasets/dshihk/llm-generated-essay)  

Data preprocessing by TAASSC.

---

## Data Usage Notice

- The `.txt` files in [`txt_samples/`](./txt_samples) are included **only for demonstration and learning purposes**.  
  They are not licensed for reuse, redistribution, or commercial use.  

- The dataset file `X_binary.csv` is private and is **not licensed** for reuse, redistribution, or modification.  
  It is shared solely for demonstration purposes and should not be used for any other purpose.



