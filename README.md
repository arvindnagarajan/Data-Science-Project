# Data Science Project - Group 7

## Topic: XAI - Switching Behaviour in Health Insurance

### Goal: Explainable AI for Understanding Switching Behaviour in Statutory Health Insurance

## Tasks
- Preprocess and integrate insurer-level data with Customer Monitor survey responses and profile dimensions.
- Perform exploratory analysis to identify dominant switching patterns across demographic and behavioral segments.
- Train interpretable models and black-box models for comparison.
- Apply XAI methods (e.g., SHAP, LIME, counterfactual explanations, etc.) to analyze feature contributions.
- Cluster individuals into personas based on key switching drivers and probabilities.
- Build a dashboard prototype to visualize personas, feature influences, and switching likelihood across segments.



## 🚀 How to Run

To launch the interactive dashboard, simply run the following command in your terminal:

```bash
streamlit run model_and_dashboard.py
```

The model_and_dashboard.py file contains the XGBoost model, SHAP explanations, K means clustering and the Streamlit dashboard.
