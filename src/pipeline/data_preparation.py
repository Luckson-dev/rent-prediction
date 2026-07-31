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
from rent_prediction.utils.get_path import get_filename


class DATAPreparation:

    def __init__(self, df):
        self.df = df
        self.train_data = None
        self.test_data = None
        self.encoder = None
        self.numeric_cols = list(self.df.drop("IdentifiantMaison", axis=1).select_dtypes(include=["number"]).columns)
        self.category_cols = list(self.df.drop("IdentifiantMaison", axis=1).select_dtypes(include=["object", "str"]).columns)

    # Traitement de valeurs maquantes
    def clean_data(self):

        self.df = self.df.drop_duplicates(inplace=True)


        for col in self.numeric_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())

        category_cols_to_fill = [
            'Salon',
            'SalleDeBainInterieure',
            'Parking', 'Meuble',
            'Jardin', 'Quartier'
        ]

        for col in category_cols_to_fill:

            if self.df[col].dropna().empty:
                self.df[col] = self.df[col].fillna("Inconnu")
            else:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        return self.df

    @property
    def identify_outlier_values(self):
        for col in self.numeric_cols:
            lower_limit, large_limit = self.outlier_values(self.df[col].dropna())
    
            outliers = self.df[(self.df[col] < lower_limit) | (self.df[col] > large_limit)]
            print()
            print(f"{col}: {len(outliers)} valeurs abberantes potentielles (bornes : {lower_limit:.1f} à {large_limit:.1f}) sur {len(self.df[col].dropna())}")

    def split_data(self):
        self.train_data, self.test_data = train_test_split(
            self.df,
            test_size=0.25,
            random_state=45
        )

        return (self.train_data, self.test_data)

    def encode_features(self):
        if self.train_data is None or self.test_data is None:
            raise ValueError("train_data and test_data can not be null")

        for col in ['Salon', 'SalleDeBainInterieure', 'Parking', 'Meuble', 'Jardin']:

            self.train_data[col + "_Bin"] = self.train_data[col].map({"Oui": 1, "Non": 0}).fillna(0).astype(int)
            self.test_data[col + "_Bin"] = self.test_data[col].map({"Oui": 1, "Non": 0}).fillna(0).astype(int)

        self.encoder = TargetEncoder(smooth=100.0)
        self.train_data["Quartier_Target"] = self.encoder.fit_transform(
            self.train_data[["Quartier"]],
            self.train_data["LoyerMensuel_BIF"]
        )[:, 0]

        self.test_data["Quartier_Target"] = self.encoder.transform(
            self.test_data["Quartier"]
        )[:, 0]

        return (self.encoder, self.train_data, self.test_data)

    def numeric_transform(self):
        train_data, _ = self.train_test_data_split()

        train_data["LoyerMensuel_Log1"] = np.log1p(train_data["LoyerMensuel_BIF"])
        train_data[["Quartier_Target", "LoyerMensuel_Log1"]].head()

        return train_data

    def save_processed_data(self):

        file_name = get_filename("processed_dataset.csv")
        train_data, _ = self.train_test_data_split()

        train_data.to_csv(file_name, index=False)
    

    # Les valeurs aberrantes
    @staticmethod
    def outlier_values(series, limit_lower_percent=0.25, limit_large_percent=0.75):
        quantile1, quantile3 = series.quantile(limit_lower_percent), series.quantile(limit_large_percent)
        interquantile_range = quantile3 - quantile1

        return quantile1 - (interquantile_range * 1.5), quantile3 + (interquantile_range * 1.5)
