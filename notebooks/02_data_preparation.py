#!/usr/bin/env python
# coding: utf-8

# In[103]:


import os
import joblib
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, TargetEncoder,
    StandardScaler, RobustScaler,
    FunctionTransformer
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

file_path = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), "data", "rent_prediction.csv")

df = pd.read_csv(file_path)


# In[104]:


numeric_cols = list(df.drop("IdentifiantMaison", axis=1).select_dtypes(include=["number"]).columns)
category_cols = list(df.drop("IdentifiantMaison", axis=1).select_dtypes(include=["object", "str"]).columns)


# ### Traitement de valeurs maquantes

# In[105]:


empty_values = df.isnull().sum().sort_values(ascending=False)
cols_with_empty_values = empty_values[empty_values > 0]

plt.figure(figsize=(10, 6))

if len(cols_with_empty_values) > 0:
    for i, _ in enumerate(cols_with_empty_values):
        col_name = cols_with_empty_values.index[i]
        col_value = cols_with_empty_values.values[i]

        print(col_name, col_value)

        label = f"Colonne avec les valeurs null : {col_name} ({col_value})"
        plt.bar(cols_with_empty_values.index, cols_with_empty_values.values, color="r", label=label)
else:
    label = "Pas de colonnes avec de valeurs nulles"
    plt.bar(cols_with_empty_values.index, cols_with_empty_values.values, color="r", label=label)

plt.legend()
plt.show()


# In[106]:


for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in ['Salon', 'SalleDeBainInterieure', 'Parking', 'Meuble', 'Jardin', 'Quartier']:
    df[col] = df[col].fillna(df[col].mode()[0])

df.isnull().sum()


# ### Lignes en double et Catégories incohérentes

# In[107]:


# df.duplicated()
df.duplicated().sum()


# In[108]:


for col in category_cols:
    print(f"{col} --> {df[col].dropna().unique()}")


# ### Les valeurs aberrantes

# In[109]:


def outlier_values(series, limit_lower_percent=0.25, limit_large_percent=0.75):
    quantile1, quantile3 = series.quantile(limit_lower_percent), series.quantile(limit_large_percent)
    interquantile_range = quantile3 - quantile1

    return quantile1 - (interquantile_range * 1.5), quantile3 + (interquantile_range * 1.5)

for col in numeric_cols:
    lower_limit, large_limit = outlier_values(df[col].dropna())

    outliers = df[(df[col] < lower_limit) | (df[col] > large_limit)]
    print()
    print(f"{col}: {len(outliers)} valeurs abberantes potentielles (bornes : {lower_limit:.1f} à {large_limit:.1f}) sur {len(df[col].dropna())}")


# ### Séparation de données d'entrainement et de test

# In[110]:


# X = df.drop("LoyerMensuel_BIF", axis=1)
# y = df["LoyerMensuel_BIF"]
train_data, test_data = train_test_split(df, test_size=0.25, random_state=45)

path = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), "data", "test_data.parquet")

train_data.to_parquet(path, index=False)
print("Données sauvegardées avec succèss")

train_data.head()


# ### Encodage de variables catégorielles

# In[114]:


for col in ['Salon', 'SalleDeBainInterieure', 'Parking', 'Meuble', 'Jardin']:
    train_data[col + "_Bin"] = train_data[col].map({"Oui": 1, "Non": 0}).fillna(0).astype(int)

encoder = TargetEncoder(smooth=100.0)
encoder.fit(train_data[["Quartier"]], train_data["LoyerMensuel_BIF"])

joblib.dump(encoder, "neighbourhood_encoder.joblib")

encoded_matrix = encoder.transform(train_data[["Quartier"]])
train_data["Quartier_Target"] = encoded_matrix[:, 0]

moyennes_par_quartier = train_data.groupby('Quartier')['LoyerMensuel_BIF'].mean()
print(moyennes_par_quartier.head(10))

# train_data["Quartier_Target"].sort_values(ascending=False)
print(train_data[["Quartier", "Quartier_Target"]].drop_duplicates().sort_values("Quartier_Target"))


# ### Transformation de variables numériques

# In[112]:


train_data["LoyerMensuel_Log1"] = np.log1p(train_data["LoyerMensuel_BIF"])

train_data[["Quartier_Target", "LoyerMensuel_Log1"]].head()


# ### Sauvegarde du dataset (processed_dataset.csv)

# In[113]:


path = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), "data", "processed_dataset.csv")
train_data.to_csv(path, index=False)


# In[ ]:




