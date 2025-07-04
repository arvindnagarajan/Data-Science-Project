# Data Science Project - Group 7

## Topic: XAI - Switching Behaviour in Health Insurance

### Goal: Explainable AI for Understanding Switching Behaviour in Statutory Health Insurance

## Tasks
- Preprocess and integrate insurer-level data with Customer Monitor survey responses and profile dimensions.
- Perform exploratory analysis to identify dominant switching patterns across demographic and behavioral segments.
- Train interpretable models and black-box models for comparison.
- Apply XAI methods (SHAP) to analyze feature contributions.
- Cluster individuals into personas based on key switching drivers and probabilities.
- Build a dashboard prototype to visualize personas, feature influences, and switching likelihood across segments.

## 🛠 Technologies & Tools

- **Python** (Pandas, NumPy, Matplotlib, Seaborn)
- **Machine Learning:** XGBoost, Logistic Regression, Explainable Boosting Machines
- **Explainable AI:** SHAP, LIME, Anchor Explanations, Counterfactuals
- **Dashboard:** Streamlit

## 💡 Key Findings

- Core drivers for switching are price-performance dissatisfaction, poor digital experiences, and perceived unreliability.
- Distinct personas identified:
  - **Active but Dissatisfied Switchers**
  - **Disillusioned Frequent Switchers**
  - **Information Seeking Skeptics**
  - **Skeptical Active Switchers**
- XGBoost + SHAP model achieved an accuracy of **83%**, balancing performance and interpretability.

## 🚀 How to Run?

To launch the interactive dashboard, simply run the following command in your terminal:

```bash
streamlit run model_and_dashboard.py
```

The model_and_dashboard.py file contains the XGBoost model, SHAP explanations, K means clustering and the Streamlit dashboard.


## Python Notebooks

1. Insurer_Level_Dataset_Integration
  For this notebook, please upload the 'Zusatzbeitrag_je Kasse je Quartal' and 'Insurer_Level_Data' files. After the upload, each code block can be run to create the 'Final_Insurer_Level_Dataset' file
2. Insurer_Level_Stats_Visualization
  For this notebook, please upload the 'Final_Insurer_Level_Dataset' file. After the upload, each code block can be run and visualizations will be generated.
3. Kundenmonitor_Dataset_Preprocessing_and_EDA
   For this notebook, please upload the 'Kundenmonitor_Questions_Summary' file. After the upload, each code block can be run and visualizations will be generated.
   
