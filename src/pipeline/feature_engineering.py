import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from .data_preparation import DATAPreparation


class FTEngineering:

    def __init__(self, df):
        self.df = df
        self.feature_cols = ["AgeMaison", "Quartier_Target", "Indicateur_Confort", "Chambres_par_Superficie", "LoyerMensuel_Log1"]
        self.cols_confort = ['Salon_Bin', 'SalleDeBainInterieure_Bin', 'Parking_Bin', 'Meuble_Bin', 'Jardin_Bin']

    def create_importante_features(self):
        self.df['Indicateur_Confort'] = self.df[self.cols_confort].sum(axis=1)

        self.df['Chambres_par_Superficie'] = self.df['Chambres'] / (self.df['Superficie_m2'] + 0.1)

        return self.df

    # def save_feauture_data(self):

    #     feature_cols = ["AgeMaison", "Quartier_Target", "Indicateur_Confort", "Chambres_par_Superficie", "LoyerMensuel_Log1"]

    #     self.df[feature_cols].to_csv("../data/feature_selection_dataset.csv", index=False)

