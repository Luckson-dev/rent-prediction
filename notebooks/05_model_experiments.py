#!/usr/bin/env python
# coding: utf-8

# ## Expérimentation de models

# In[10]:


import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score, get_scorer_names


# In[18]:


train_data = pd.read_csv("../data/processed_dataset.csv")
test_data = pd.read_parquet("../data/test_data.parquet")

X_train = train_data.drop(
    ["LoyerMensuel_Log1", "IdentifiantMaison"],
    axis=1).select_dtypes(include=["number"])
y_train = train_data["LoyerMensuel_Log1"]

print(X_train)


# ### Transformation de données de test

# In[21]:


# Les variables numériques
feature_cols = ["AgeMaison", "Quartier_Target", "Indicateur_Confort", "Chambres_par_Superficie", "LoyerMensuel_Log1"]

for col in ['Salon', 'SalleDeBainInterieure', 'Parking', 'Meuble', 'Jardin']:
    test_data[col + "_Bin"] = test_data[col].map({"Oui": 1, "Non": 0}).fillna(0).astype(int)

neighbourhood_encoder = joblib.load("neighbourhood_encoder.joblib") 
test_data["Quartier_Target"] = neighbourhood_encoder.transform(test_data[["Quartier"]])[:, 0]

test_data["LoyerMensuel_Log1"] = np.log1p(test_data["LoyerMensuel_BIF"])

# Création de variables utiles pour le test
# cols_confort = ['Salon_Bin', 'SalleDeBainInterieure_Bin', 'Parking_Bin', 'Meuble_Bin', 'Jardin_Bin']
# test_data['Indicateur_Confort'] = test_data[cols_confort].sum(axis=1)

# test_data['Chambres_par_Superficie'] = test_data['Chambres'] / (test_data['Superficie_m2'] + 0.1)

# Séparation de X_test et y_test
X_test = test_data.drop(
    ["LoyerMensuel_Log1", "IdentifiantMaison"],
    axis=1).select_dtypes(include=["number"])
y_test = test_data["LoyerMensuel_Log1"]

X_test


# In[22]:


def formated_time(second):
    m, s = divmod(second, 60)

    if m > 0:
         f"{int(m)} min {s:.4f} secondes"
    else:
        return f"{s:.4f} secondes"

scoring = ["neg_mean_squared_error", "neg_median_absolute_error", "neg_root_mean_squared_error", "r2"]


# ### Modèle de Régression Linéaire

# In[23]:


linear_model = LinearRegression()

linear_scores = cross_validate(linear_model, X_train, y_train, scoring=scoring)

print(f"Fit Time : {linear_scores["fit_time"]}")
print()
print(f"Score Time : {linear_scores["score_time"]}")
print()
print(f"Test Neg Median Abosulte Error : {linear_scores["test_neg_median_absolute_error"]}")
print()
print(f"Test Neg Mean Squared Error : {linear_scores["test_neg_mean_squared_error"]}")
print()
print(f"Test R2 : {linear_scores["test_r2"]}")


# ### Modèle Lasso

# In[24]:


lasso_model = Lasso(alpha=0.2)

lasso_scores = cross_validate(lasso_model, X_train, y_train, scoring=scoring)

print(f"Fit Time : {lasso_scores["fit_time"]}")
print()
print(f"Score Time : {lasso_scores["score_time"]}")
print()
print(f"Test Neg MSE : {lasso_scores["test_neg_mean_squared_error"]}")
print()
print(f"Test Neg  MAE : {lasso_scores["test_neg_median_absolute_error"]}")
print()
print(f"Test Neg SMSE : {lasso_scores["test_neg_root_mean_squared_error"]}")
print()
print(f"Test R2 : {lasso_scores["test_r2"]}")


# ### Ridget Model

# In[25]:


ridge_model = Ridge(alpha=0.1)
ridge_scores = cross_validate(ridge_model, X_train, y_train, scoring=scoring)

print(f"Fit Time : {ridge_scores["fit_time"]}")
print()
print(f"Score Time : {ridge_scores["score_time"]}")
print()
print(f"Test Neg MSE : {ridge_scores["test_neg_mean_squared_error"]}")
print()
print(f"Test Neg  MAE : {ridge_scores["test_neg_median_absolute_error"]}")
print()
print(f"Test Neg SMSE : {ridge_scores["test_neg_root_mean_squared_error"]}")
print()
print(f"Test R2 : {ridge_scores["test_r2"]}")


# ### Arbres de décision

# In[26]:


tree_model = DecisionTreeRegressor()
tree_scores = cross_validate(tree_model, X_train, y_train, scoring=scoring)

print(f"Fit Time : {tree_scores["fit_time"]}")
print()
print(f"Score Time : {tree_scores["score_time"]}")
print()
print(f"Test Neg MSE : {tree_scores["test_neg_mean_squared_error"]}")
print()
print(f"Test Neg  MAE : {tree_scores["test_neg_median_absolute_error"]}")
print()
print(f"Test Neg SMSE : {tree_scores["test_neg_root_mean_squared_error"]}")
print()
print(f"Test R2 : {tree_scores["test_r2"]}")


# In[27]:


random_model = RandomForestRegressor()

random_scores = cross_validate(random_model, X_train, y_train, scoring=scoring)

print(f"Fit Time : {random_scores["fit_time"]}")
print()
print(f"Score Time : {random_scores["score_time"]}")
print()
print(f"Test Neg MSE : {random_scores["test_neg_mean_squared_error"]}")
print()
print(f"Test Neg  MAE : {random_scores["test_neg_median_absolute_error"]}")
print()
print(f"Test Neg SMSE : {random_scores["test_neg_root_mean_squared_error"]}")
print()
print(f"Test R2 : {random_scores["test_r2"]}")


# In[ ]:




