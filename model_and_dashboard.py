import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import xgboost as xgb
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import shap
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from config import feature_meanings, provider_map, cluster_info
from prep import feature_engineering, data_cleaning, feature_transformation
import streamlit.components.v1 as components

#Importing Dataset
org_survey_data = pd.read_excel("230807_Survey-org.xlsx", sheet_name="Result")
df = org_survey_data
insurer_level_data = pd.read_excel("insurer_details.xlsx") # preprocessed and combined dataset of insurers' details of 2023

# Data Prep
feature_engineering(df)
data_cleaning(df)
feature_transformation(df)

survey_data = pd.read_csv("preprocessed_data.csv") # Using the preprocessed data for clarity, since the original file contains multiple sheets

# Combining the Survey with Insurer level Details Dataset
insurer_level_data['Krankenkasse-with-Q9'] = insurer_level_data['Krankenkasse'].map(provider_map)
insurer_level_data['Krankenkasse-with-Q9'] = insurer_level_data['Krankenkasse-with-Q9'].fillna(0)
survey_data = survey_data.merge(insurer_level_data, how="left", left_on="Q9", right_on="Krankenkasse-with-Q9")
survey_data = survey_data.drop("Krankenkasse", axis=1)
survey_data = survey_data.drop("Jahr", axis=1)
survey_data = survey_data.drop("Regionalität", axis=1)
survey_data = survey_data.drop("Regionale Verteilung", axis=1)

survey_data = survey_data.fillna(0)
df = survey_data
df_final = df
df_renamed = df_final.rename(columns=feature_meanings)

# Logistic Regression Model
x, y = survey_data.drop("Q18", axis=1), survey_data["Q18"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = x_train.copy()
X_test_scaled = x_test.copy()
X_train_scaled[x.columns] = scaler.fit_transform(x_train[x.columns])
X_test_scaled[x.columns] = scaler.transform(x_test[x.columns])
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)
log_pred = logreg.predict_proba(X_test_scaled)[:, 1]
auc_log = roc_auc_score(y_test, log_pred)
print("AUC score:", auc_log)

# SHAP Model for Logistic Regression
x = df_renamed.drop("Switchers l10y", axis=1)
y = df_renamed["Switchers l10y"]
explainer = shap.Explainer(logreg, x)
shap_values = explainer(x)
shap.plots.beeswarm(shap_values, max_display=10)


# XGBoost Model
x, y = df.drop("Q18", axis=1), df["Q18"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)
model = xgb.XGBClassifier(
    n_estimators=350,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42)
model.fit(x_train, y_train)
preds = model.predict(x_test)
accuracy = accuracy_score(y_test, preds)
print(accuracy)
# Finding Switching Probability
switch_probs = model.predict_proba(x)[:, 1]

# SHAP Model for XGBoost
x = df_renamed.drop("Switchers l10y", axis=1)
y = df_renamed["Switchers l10y"]
explainer = shap.Explainer(model, x)
shap_values = explainer(x)
shap_array = shap_values.values
shap.plots.beeswarm(shap_values, max_display=10)


# Clustering Personas
inertias = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(shap_array) 
    inertias.append(kmeans.inertia_)

# Using Elbow Plot to find the number of Clusters
plt.figure(figsize=(8,5))
plt.plot(k_range, inertias, marker='o')
plt.xlabel("Number of clusters k")
plt.ylabel("Inertia")
plt.title("Elbow Plot to choose optimal k")
plt.grid()
plt.show()

# K-means Clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(shap_array) 

x_df = x.copy()
x_df_final = x.copy()
x_shap_input = x.copy()
x_df["cluster"] = cluster_labels
x_df["switching_probability"] = switch_probs
df["cluster"] = cluster_labels # Attaching cluster group to each individual in the survey
df["switching_probability"] = switch_probs # Attaching switching probability to each individual in the survey

# Attaching the participant id back to the list so we now know which participant belongs to which cluster and thier score
x_df_copy = x_df
first_col_name = org_survey_data.columns[0]
first_col = org_survey_data[first_col_name]
x_df_copy.insert(0, first_col_name, first_col)


# Calculating average probability per cluster
switching_prob_by_cluster = x_df.groupby('cluster')['switching_probability'].mean()

# Finding the top features influencing Switching using SHAP Model
shap_df = pd.DataFrame(shap_array, columns=x_test.columns)
shap_df["cluster"] = cluster_labels
cluster_feature_importance = (
    shap_df.groupby("cluster")
    .apply(lambda df: df.iloc[:, :-1].abs().mean()) 
)

N = 8  
for cluster_id in cluster_feature_importance.index:
    top_features = cluster_feature_importance.loc[cluster_id].sort_values(ascending=False).head(N)
    print(f"\nCluster {cluster_id} - Top {N} Features:\n")
    print(top_features)

# Streamlit Layout Begins
st.set_page_config(page_title="Switching Personas Dashboard", layout="wide")
st.title("Switching Personas: Dashboard")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Cluster Personas", "Explainability Outputs", "Policy Insights", "Data-Driven Insights"])
feature_names = x_df.drop(columns=["cluster"]).columns
cluster_top_features = {}
n_clusters = 4

for cluster in range(n_clusters):
    mask = x_df['cluster'] == cluster
    mean_shap = np.abs(shap_array[mask.to_numpy(), :]).mean(axis=0)
    N = 8
    top_idx = mean_shap.argsort()[::-1][:N]
    top_features = [(feature_names[i], mean_shap[i]) for i in top_idx]
    cluster_top_features[cluster] = top_features

feature_names = x_df.drop(columns=["cluster"]).columns

with tab1:
    st.header("Cluster Personas Overview")
        
    # UMAP Projection
    st.subheader('UMAP Projection of SHAP-based Personas')
    st.image('cluster.png', caption='UMAP Projection')

    # Cluster-wise Info
    for cluster in range(n_clusters):
        cluster_data = x_df[x_df["cluster"] == cluster]
        name = cluster_info[cluster]["name"]
        desc = cluster_info[cluster]["description"]

        st.subheader(f"Persona {cluster}: {name}")
        st.write(desc)
        st.write(f"Number of members: {cluster_data.shape[0]}")
        avg_prob = switching_prob_by_cluster.loc[cluster]
        st.write("Average Switching Probability", f"{avg_prob:.1%}")

        # Descriptive stats
        st.write(cluster_data.describe())

        # Pie chart for top contributing features
        st.write("### Top Contributing Features in this Cluster")

        top_feats = cluster_top_features[cluster]
        labels = [f[0] for f in top_feats]
        values = [f[1] for f in top_feats]

        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    st.markdown("---")

with tab2:
    st.header("Global SHAP Summary")
    st.write("This shows the features driving switching behaviour across all clusters.")
    plt.clf()
    plt.figure(figsize=(8, 4)) 
    shap.summary_plot(shap_values, x_shap_input, show=False)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.clf()

    st.header("Cluster-wise SHAP Values")

    for cluster in range(n_clusters):
        st.subheader(f"Cluster {cluster}")
        indices = x_df['cluster'] == cluster
        cluster_shap_values = shap_values[indices.to_numpy()]
        plt.figure(figsize=(8, 4))
        shap.summary_plot(cluster_shap_values, x_shap_input.loc[indices], show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.clf()

    st.subheader("SHAP Heatmap")
    shap_df = pd.DataFrame(shap_values.values, columns=x_df_final.columns)
    heatmap_data = shap_df.sample(100, random_state=42)
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="coolwarm", center=0)
    st.pyplot(plt.gcf())
    plt.clf()
    st.markdown("---")


with tab3:
    st.header("Policy Implications")

    st.subheader("Cluster 0: Active but Dissatisfied Switchers")
    st.markdown("""
    - Introduce loyalty bonuses after 2 years to reduce frequent switching  
    - Invest in faster claims processing to reduce dissatisfaction  
    - Proactively engage them after each contact (e.g., follow-up calls) to address concerns early
    """)

    st.subheader("Cluster 1: Disillusioned Frequent Switchers")
    st.markdown("""
    - Run targeted win-back campaigns with personalized offers  
    - Provide transparency about coverage changes and pricing  
    - Design a claims-resolution guarantee program to rebuild trust
    """)

    st.subheader("Cluster 2: Information-Seeking Skeptics")
    st.markdown("""
    - Offer proactive education about benefits and coverages  
    - Create regular webinars or Q&A sessions with customer advisors  
    - Improve staff friendliness through training programs
    """)

    st.subheader("Cluster 3: Skeptical Active Switchers")
    st.markdown("""
    - Strengthen messaging on guarantees and factual evidence (vs. pure marketing)  
    - Publish service statistics to build trust  
    - Assign dedicated account managers to high-risk members to rebuild loyalty
    """)
    st.markdown("---")

with tab4:
    st.subheader('Insurance Members Distribution across Providers')
    st.image('eda-1.jpg')
    st.subheader('Insurance Membership by HH Income')
    st.image('eda-2.jpg')
    st.subheader('Insured Members vs Co-Insured Members by Age')
    st.image('eda-3.jpg')
    st.subheader('Insured Members vs Co-Insured Members by Intent to continue with Provider')
    st.image('eda-4.jpg')
    st.markdown("---")