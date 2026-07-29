#!/usr/bin/env python
# coding: utf-8

# In[23]:


import os
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt


# In[24]:


path = os.path.abspath(os.getcwd())
file_path = os.path.join(os.path.dirname(path), "data", "rent_prediction.csv")

df = pd.read_csv(file_path)


# ### Dimensions, colonnes et types

# In[19]:


numeric_cols = list(df.drop("IdentifiantMaison", axis=1).select_dtypes(include=["number"]).columns)
category_cols = list(df.drop("IdentifiantMaison", axis=1).select_dtypes(include=["object", "str"]).columns)

print(f"Colonnes dans les données : {list(df.columns)}")
print()
print(f"La forme de données du DataSet : {df.shape}")
print()
print(f"Type de données du DataSet : {df.info()}")


# ### Premier coup d'oeil aux données

# In[8]:


df.head()
df.tail()
df.sample(10)


# ### Statistiques descriptives

# In[9]:


df.describe()
# df.describe(include=["object"])


# ### Identification de valeurs maquantes

# In[15]:


df.isnull()
df.isnull().sum().sort_values(ascending=False)


# ### Lignes en double

# In[17]:


df.duplicated()
df.duplicated().sum()


# ### Analyse de la cible

# In[58]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df["LoyerMensuel_BIF"], kde=True, ax=axes[0], color="skyblue")
axes[0].set_title("Distribution brute de LoyerMensuel_BIF")

# Distribution avec Transformation Log
sns.histplot(np.log1p(df["LoyerMensuel_BIF"]), kde=True, ax=axes[1], color="orange")
axes[1].set_title("Distribution de log1p(LoyerMensuel_BIF)")

plt.tight_layout()
plt.show()


# ### Analyses de features numériques

# In[29]:


fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(14, 4 * len(numeric_cols)))

for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], kde=True, ax=axes[i, 0], color="teal")
    axes[i, 0].set_title(f"Distribution de {col}")

    sns.boxplot(x=df[col], ax=axes[i, 1], color="coral")
    axes[i, 1].set_title(f"Boxplot de {col}")

plt.tight_layout()
plt.show()


# ### Analyses de variables catégorielles

# In[11]:


fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

for i, col in enumerate(category_cols):
    sns.countplot(
        data=df,
        x=col,
        hue=col,
        legend=False,
        ax=axes[i],
        palette="Set2",
        order=df[col].value_counts().index,
    )
    axes[i].set_title(f"Répartition de {col} (Cardinalité: {df[col].nunique()})")
    axes[i].tick_params(axis="x", rotation=45 if df[col].nunique() > 5 else 0)

plt.tight_layout()
plt.show()


# In[22]:


"""
  Calcule le V de Cramer entre deux variables catégorielles x et y

"""

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)

    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape

    # La formule de v de crammer
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)

    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

for col in category_cols:
    corr = cramers_v(df[col], df["LoyerMensuel_BIF"])
    print(col, "-->", corr)


# ### Pattern des valeurs maquantes

# In[56]:


# Visualiser si les manques sont corrélés entre eux (MCAR vs MNAR)
plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Carte thermique des valeurs manquantes (Missing Pattern)")
plt.show()


# ### Corrélation avec la variable cible (LoyerMensuel_BIF)

# In[35]:


plt.figure(figsize=(1, 6))

corr_matrix = df[numeric_cols + ["LoyerMensuel_BIF"]].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Heatmap de corrélation (Variables Numériques)")
plt.show()


# ### Scatter plots / Boxplots croisant chaque feature avec LoyerMensuel_BIF.

# In[ ]:


# Scatter plots pour les variables numériques
for col in numeric_cols:
    plt.figure(figsize=(7, 4))
    sns.scatterplot(
        data=df, x=col, y="LoyerMensuel_BIF", alpha=0.6, color="indigo"
    )
    plt.title(f"{col} vs LoyerMensuel_BIF")
    plt.show()

# Boxplots pour les variables catégorielles
for col in category_cols:
    plt.figure(figsize=(8, 4))

    sns.boxplot(
        data=df,
        x=col,
        hue=col,
        legend=False,
        y="LoyerMensuel_BIF",
        palette="Pastel1",
        order=df.groupby(col)["LoyerMensuel_BIF"]
        .median()
        .sort_values(ascending=False)
        .index,
    )

    plt.title(f"LoyerMensuel_BIF selon {col}")
    plt.xticks(rotation=45 if df[col].nunique() > 5 else 0)
    plt.show()


# In[2]:


# import pandas as pd

# eda_decisions_data = {
#     "Variable / Aspect": [
#         "Lignes Manquantes (Global)",
#         "AgeMaison",
#         "LoyerMensuel_BIF (Variable Cible)",
#         "Chambres & Superficie_m2",
#         "Quartier",
#         "Équipements (Meuble, Jardin, Salon, SdB)",
#         "DistanceRoute_m",
#         "Parking",
#     ],
#     "Données Observées (EDA)": [
#         "25 lignes totalement vides sur l'ensemble des features (485 non-nulls sur 510)[cite: 1].",
#         "Valeur maximale aberrante à 2 600 000 ans (fuite possible de la valeur du loyer)[cite: 1].",
#         "Distribution asymétrique à droite + effet de plafonnement/seuil marqué à 2 600 000 BIF[cite: 1].",
#         "Corrélation quasi-parfaite de 0.98 entre les deux variables + corrélation de 0.62 avec le loyer[cite: 1].",
#         "Forte hiérarchie des prix : Rohero/Kiriri très élevés vs Buyenzi/Bwiza très bas[cite: 1].",
#         "Impact positif net sur le loyer médian (notamment le caractère Meublé)[cite: 1].",
#         "Très faible corrélation linéaire avec le loyer (~0.02)[cite: 1].",
#         "Faible différenciation sur la distribution des loyers entre Oui et Non[cite: 1].",
#     ],
#     "Action / Décision Technique": [
#         "Suppression directe des 25 lignes concernées.",
#         "Remplacement des valeurs > 100 par NaN puis imputation par la médiane selon le Quartier.",
#         "Passage de la cible au log (log1p) pour lisser l'asymétrie + test d'un modèle adapté aux données censurées.",
#         "Conservation de Superficie_m2 pour modèles linéaires (ou création du ratio Superficie/Chambre) ; conservation des deux pour arbres/XGBoost.",
#         "Application d'un Target Encoding (ou Ordinal Encoding basé sur le loyer médian).",
#         "Création d'une feature composite 'Standing_Score' (somme des équipements présents).",
#         "Conservation temporaire ; test d'une binarisation (ex: Proche < 200m vs Éloigné) ou sélection par Ridge/Lasso.",
#         "Simplification ou faible pondération lors du Feature Engineering.",
#     ],
#     "Justification Technique": [
#         "Les données manquent simultanément sur tout le profil (MCAR/MAR), la suppression évite d'injecter du bruit sans perte d'information utile.",
#         "Corrige une erreur manifeste de saisie tout en préservant la cohérence géographique globale.",
#         "Rapproche la cible d'une distribution normale, ce qui stabilise la variance et améliore les performances des régresseurs.",
#         "Évite le problème de multicolinéarité sévère qui déstabilise les coefficients des modèles linéaires.",
#         "Conserve l'information spatiale et la valeur foncière sans exploser le nombre de dimensions (contrairement au One-Hot Encoding).",
#         "Synthétise le niveau de confort général du bien en une seule variable explicative puissante.",
#         "Transforme une variable continue peu explicative en un seuil de proximité plus pertinent.",
#         "Évite de surcharger les modèles avec des variables peu discriminantes.",
#     ],
# }

# # Création du DataFrame
# eda_decisions = pd.DataFrame(eda_decisions_data)

# pd.set_option("display.max_colwidth", None)
# eda_decisions


# In[ ]:




