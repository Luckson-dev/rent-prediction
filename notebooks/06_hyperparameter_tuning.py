#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from  sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


# In[5]:


train_data = pd.read_csv("../data/processed_dataset.csv")
test_data = pd.read_parquet("../data/test_data.parquet")

X_train = train_data.drop(
    ["LoyerMensuel_Log1", "LoyerMensuel_BIF", "IdentifiantMaison"],
    axis=1).select_dtypes(include=["number"])
y_train = train_data["LoyerMensuel_Log1"]

print(X_train)

# correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
# print(correlations.head(10))


# ### Mise en niveau pour les données de test

# In[6]:


# Les variables numériques
feature_cols = ["AgeMaison", "Quartier_Target", "Indicateur_Confort", "Chambres_par_Superficie", "LoyerMensuel_Log1"]

for col in ['Salon', 'SalleDeBainInterieure', 'Parking', 'Meuble', 'Jardin']:
    test_data[col + "_Bin"] = test_data[col].map({"Oui": 1, "Non": 0}).fillna(0).astype(int)

neighbourhood_encoder = joblib.load("neighbourhood_encoder.joblib") 
test_data["Quartier_Target"] = neighbourhood_encoder.transform(test_data[["Quartier"]])[:, 0]

test_data["LoyerMensuel_Log1"] = np.log1p(test_data["LoyerMensuel_BIF"])

# Séparation de X_test et y_test
X_test = test_data.drop(
    ["LoyerMensuel_Log1", "LoyerMensuel_BIF", "IdentifiantMaison"],
    axis=1).select_dtypes(include=["number"])
y_test = test_data["LoyerMensuel_Log1"]

X_test


# ### Entrainement du modèle

# In[7]:


boosting_model = GradientBoostingRegressor(
    learning_rate=0.2,
    n_estimators=150,
    max_depth=2,
    random_state=42
)

boosting_model.fit(X_train, y_train)

y_predict = boosting_model.predict(X_test)

y_predict_series = pd.Series(y_predict, index=y_test.index)

compareson = pd.DataFrame({
    "Valeurs réelles": np.expm1(y_test),
    "Valeurs prédites": np.expm1(y_predict_series)
})

np.expm1(y_predict[:5])

compareson.head(10)


# ### Mésure de performances avec cross_validate

# In[8]:


scoring = ["neg_mean_squared_error", "neg_median_absolute_error", "neg_root_mean_squared_error", "r2"]

boosting_scores = cross_validate(boosting_model, X_train, y_train, scoring=scoring)

print(f"Fit Time : {boosting_scores["fit_time"]}")
print()
print(f"Score Time : {boosting_scores["score_time"]}")
print()
print(f"Test Neg Median Abosulte Error : {boosting_scores["test_neg_median_absolute_error"]}")
print()
print(f"Test Neg Mean Squared Error : {boosting_scores["test_neg_mean_squared_error"]}")
print()
print(f"Test R2 : {boosting_scores["test_r2"]}")

print(r2_score(y_test, y_predict))


# ### Sauvegarde du modèle

# In[9]:


path = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), "data", "best_model_tuned.pkl")
joblib.dump(boosting_model, path)


# In[10]:


importance = boosting_model.feature_importances_

print(X_train.columns)
print(importance)

classement = pd.DataFrame({
    "Variables": X_train.columns,
    "Feature_importance": importance
}).sort_values(ascending=False, by="Feature_importance")

classement

