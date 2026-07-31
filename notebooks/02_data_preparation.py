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
from rent_prediction.utils.get_path import get_filename


class DATAPreparation:

    def __init__(self, df):
        self.df = df
        self.numeric_columns = list(self.df.drop("IdentifiantMaison", axis=1).select_dtypes(include=["number"]).columns)
        self.category_columns = list(self.df.drop("IdentifiantMaison", axis=1).select_dtypes(include=["object", "str"]).columns)

    def get_data(self):
        file_name = get_filename(file_name="rent_prediction.csv")

        df = pd.read_csv(file_name)

        return df

    # Traitement de valeurs maquantes
    def remove_empty_values(self):
        for col in self.numeric_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())

        for col in ['Salon', 'SalleDeBainInterieure', 'Parking', 'Meuble', 'Jardin', 'Quartier']:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        return self.df

    # Lignes en double et Catégories incohérentes
    def check_duplicated_values(self):
        print(f"Show duplicated values : {self.df.duplicated()}")
        # self.df.duplicated().sum()

        for col in self.category_cols:
            print(f"{col} --> {self.df[col].dropna().unique()}")

    def identify_outlier_values(self):
        for col in self.numeric_cols:
            lower_limit, large_limit = self.outlier_values(self.df[col].dropna())
    
            outliers = self.df[(self.df[col] < lower_limit) | (self.df[col] > large_limit)]
            print()
            print(f"{col}: {len(outliers)} valeurs abberantes potentielles (bornes : {lower_limit:.1f} à {large_limit:.1f}) sur {len(self.df[col].dropna())}")

    def train_test_data_split(self):
        train_data, test_data = train_test_split(
            self.df,
            test_size=0.25,
            random_state=45
        )

        return (train_data, test_data)

    def get_feature_encoding(self):

        train_data, _ = self.train_test_data_split()

        for col in ['Salon', 'SalleDeBainInterieure', 'Parking', 'Meuble', 'Jardin']:
            train_data[col + "_Bin"] = train_data[col].map({"Oui": 1, "Non": 0}).fillna(0).astype(int)

        encoder = TargetEncoder(smooth=100.0)
        encoder.fit(train_data[["Quartier"]], train_data["LoyerMensuel_BIF"])

        encoded_matrix = encoder.transform(train_data[["Quartier"]])
        train_data["Quartier_Target"] = encoded_matrix[:, 0]

        moyennes_par_quartier = train_data.groupby('Quartier')['LoyerMensuel_BIF'].mean()
        print(moyennes_par_quartier.head(10))

        # train_data["Quartier_Target"].sort_values(ascending=False)
        print(train_data[["Quartier", "Quartier_Target"]].drop_duplicates().sort_values("Quartier_Target"))

        return (encoder, train_data)

    def numeric_transform(self):
        train_data, _ = self.train_test_data_split()

        train_data["LoyerMensuel_Log1"] = np.log1p(train_data["LoyerMensuel_BIF"])
        train_data[["Quartier_Target", "LoyerMensuel_Log1"]].head()

        return train_data
    

    # Les valeurs aberrantes
    def outlier_values(series, limit_lower_percent=0.25, limit_large_percent=0.75):
        quantile1, quantile3 = series.quantile(limit_lower_percent), series.quantile(limit_large_percent)
        interquantile_range = quantile3 - quantile1

        return quantile1 - (interquantile_range * 1.5), quantile3 + (interquantile_range * 1.5)

# ### Encodage de variables catégorielles



# ### Transformation de variables numériques


# ### Sauvegarde du dataset (processed_dataset.csv)

# In[113]:


path = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), "data", "processed_dataset.csv")
train_data.to_csv(path, index=False)


# In[ ]:




