import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from config import feature_meanings, provider_map


survey_data = pd.read_csv("preprocessed_data.csv")
insurer_level_data = pd.read_excel("insurer_details.xlsx")


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

# Logistic Regression
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