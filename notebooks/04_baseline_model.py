#!/usr/bin/env python
# coding: utf-8

# ## Baseline Model

# In[22]:


import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score


# ### Sélection de variables

# In[23]:


train_data = pd.read_csv("../data/feature_selection_dataset.csv")
test_data = pd.read_parquet("../data/test_data.parquet")


# In[24]:


X_train = train_data.drop("LoyerMensuel_Log1", axis=1)
y_train = train_data["LoyerMensuel_Log1"]   


# ### Transformation de données de test

# In[25]:


# Les variables numériques
feature_cols = ["AgeMaison", "Quartier_Target", "Indicateur_Confort", "Chambres_par_Superficie", "LoyerMensuel_Log1"]

for col in ['Salon', 'SalleDeBainInterieure', 'Parking', 'Meuble', 'Jardin']:
    test_data[col + "_Bin"] = test_data[col].map({"Oui": 1, "Non": 0}).fillna(0).astype(int)

neighbourhood_encoder = joblib.load("neighbourhood_encoder.joblib") 
test_data["Quartier_Target"] = neighbourhood_encoder.transform(test_data[["Quartier"]])[:, 0]

test_data["LoyerMensuel_Log1"] = np.log1p(test_data["LoyerMensuel_BIF"])

# Création de variables utiles pour le test
cols_confort = ['Salon_Bin', 'SalleDeBainInterieure_Bin', 'Parking_Bin', 'Meuble_Bin', 'Jardin_Bin']
test_data['Indicateur_Confort'] = test_data[cols_confort].sum(axis=1)

test_data['Chambres_par_Superficie'] = test_data['Chambres'] / (test_data['Superficie_m2'] + 0.1)

# Séparation de X_test et y_test
X_test = test_data[feature_cols].drop("LoyerMensuel_Log1", axis=1)
y_test = test_data[feature_cols]["LoyerMensuel_Log1"]

X_test


# ### Dummy Régression

# In[31]:


model_dummy_mean = DummyRegressor(strategy="mean").fit(X_train, y_train)
model_dummy_median = DummyRegressor(strategy="median").fit(X_train, y_train)

y_predict_dummy_mean = model_dummy_mean.predict(y_test)
y_predict_dummy_median = model_dummy_median.predict(y_test)

print(f"y_predict_dummy_mean : {y_predict_dummy_mean}")
print()
print(f"y_predict_dummy_median : {y_predict_dummy_median}")


# ### Régression Linéaire simple

# In[32]:


model = LinearRegression().fit(X_train, y_train)
y_predict = model.predict(X_test)

y_predict_series = pd.Series(y_predict, index=y_test.index)

compareson = pd.DataFrame({
    "Réel": y_test,
    "Prédict": y_predict_series
})

print(compareson.head(3))


# ### Analyse des erreurs

# In[34]:


print(f"LinearModel : {round(mean_squared_error(y_test, y_predict), 2)}")
print(f"dummy median : {round(median_absolute_error(y_test, y_predict_dummy_median), 2)}")
print(f"dummy mean : {round(mean_squared_error(y_test, y_predict_dummy_mean), 2)}")

print()

print(f"r2_Score dummpy mean : {r2_score(y_test, y_predict_dummy_mean)}")
print(f"r2_Score dummpy median : {r2_score(y_test, y_predict_dummy_median)}")
print(f"r2_Score LinearModel : {r2_score(y_test, y_predict)}")


# In[ ]:




