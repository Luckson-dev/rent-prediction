#!/usr/bin/env python
# coding: utf-8

# ## Sélection de variables plus importantes

# In[36]:


import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from  sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


# ### Création de variables très importantes

# In[37]:


path = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), "data", "best_model_tuned.pkl")

boosting_model = joblib.load(path)

importance = boosting_model.feature_importances_

columns = ['Chambres', 'Superficie_m2', 'DistanceRoute_m', 'AgeMaison',
       'Salon_Bin', 'SalleDeBainInterieure_Bin', 'Parking_Bin', 'Meuble_Bin',
       'Jardin_Bin', 'Quartier_Target']

classement = pd.DataFrame({
    "Variables": columns,
    "Feature_Importance": importance
}).sort_values(ascending=False, by="Feature_Importance")

feature_importance_cols = classement.nlargest(5, "Feature_Importance")["Variables"]


# ### Chargement du modèle d'entrainement

# In[38]:


train_data = pd.read_csv("../data/processed_dataset.csv")
test_data = pd.read_parquet("../data/test_data.parquet")

X_train_all_features = train_data[columns]
X_train_importance = train_data[feature_importance_cols]
y_train = train_data["LoyerMensuel_Log1"]


# ### Mise en niveau de données de test

# In[39]:


# Les variables numériques
feature_cols = ["AgeMaison", "Quartier_Target", "Indicateur_Confort", "Chambres_par_Superficie", "LoyerMensuel_Log1"]

for col in ['Salon', 'SalleDeBainInterieure', 'Parking', 'Meuble', 'Jardin']:
    test_data[col + "_Bin"] = test_data[col].map({"Oui": 1, "Non": 0}).fillna(0).astype(int)

neighbourhood_encoder = joblib.load("neighbourhood_encoder.joblib") 
test_data["Quartier_Target"] = neighbourhood_encoder.transform(test_data[["Quartier"]])[:, 0]

test_data["LoyerMensuel_Log1"] = np.log1p(test_data["LoyerMensuel_BIF"])

# Séparation de X_test et y_test
X_test_all_features = test_data[columns]

X_test_importance = test_data[feature_importance_cols]
y_test = test_data["LoyerMensuel_Log1"]

X_test_importance


# ### Comparaison de modèles via la sélection de variables très important et toutes les variables

# In[45]:


# Entrainement sur toutes les variables
all_features_model = GradientBoostingRegressor(
    learning_rate=0.2,
    n_estimators=150,
    max_depth=2,
    random_state=42
)

all_features_model.fit(X_train, y_train)
y_predict_all_features = all_features_model.predict(X_test)


# In[47]:


# Entrainement les variables plus importantes
importance_features_model = GradientBoostingRegressor(
    learning_rate=0.2,
    n_estimators=150,
    max_depth=2,
    random_state=42
)

importance_features_model.fit(X_train_importance, y_train)
y_predict_importance_features = importance_features_model.predict(X_test_importance)


# In[48]:


# Score de deux modèles
print(f"Score pour all_features_model : {r2_score(y_test, y_predict_all_features)}")
print(f"Score pour importance_features_model : {r2_score(y_test, y_predict_importance_features)}")


# In[ ]: