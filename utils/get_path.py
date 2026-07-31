import os


def get_filename(file_name) -> str:
    
    path = os.path.abspath(os.getcwd())
    file_path = os.path.join(
        os.path.dirname(path), "data", file_name)

    return file_path

    








# import os
# import pandas as pd
# from sklearn.preprocessing import TargetEncoder
# from sklearn.model_selection import train_test_split


# class DataPreparation:

#     def __init__(self, df):
#         self.df = df

#     # def load_data(self):
#     #     file_path = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), "data", "rent_prediction.csv")

#     #     df = pd.read_csv(file_path)

#     #     return df

#     def get_columns(self):

#         numeric_cols = list(
#             self.df.drop("IdentifiantMaison", axis=1).select_dtypes(include=["number"]).columns)
#         category_cols = list(
#             self.df.drop("IdentifiantMaison", axis=1).select_dtypes(include=["object", "str"]).columns)

#         return numeric_cols, category_cols

#     def removeNanValues(self):
#         df = self.load_data()
#         numeric_cols, _ = self.get_columns()

#         for col in numeric_cols:
#             df[col] = df[col].fillna(df[col].median())

#         for col in ['Salon', 'SalleDeBainInterieure', 'Parking', 'Meuble', 'Jardin']:
#             df[col] = df[col].fillna(df[col].mode()[0])

#     def train_test_data(self):
        
#         train_data, test_data = train_test_split(
#             self.df,
#             test_size=0.25,
#             random_state=45
#         )

#         return train_data, test_data

#     def encode(self):
#         train_data, _ = self.train_test_data()

#         for col in ['Salon', 'SalleDeBainInterieure', 'Parking', 'Meuble', 'Jardin']:
#             train_data[col + "_Bin"] = train_data[col].map({"Oui": 1, "Non": 0}).fillna(0).astype(int)

#         encoder = TargetEncoder(smooth=0)
#         encoded_matrix = encoder.fit_transform(
#             train_data[["Quartier"]],
#             train_data["LoyerMensuel_BIF"]
#         )

#         train_data["Quartier_Bin"] = encoded_matrix[:, 0]

#         return train_data

#     def save_data(self, file_name="processed_dataset.csv"):
#         path = os.path.join(
#             os.path.dirname(os.path.abspath(os.getcwd())),
#             "data", file_name
#         )

#         self.df.to_csv(path, index=False)

# # Sélection de future variables
# class FeatureEngeneering(DataPreparation):

#     def __init__(self, df):
#         self.df = df

#     def createUsefulVariable(self):
#         cols_confort = [
#             'Salon_Bin',
#             'SalleDeBainInterieure_Bin',
#             'Parking_Bin',
#             'Meuble_Bin', 
#             'Jardin_Bin'
#         ]

#         self.df['Indicateur_Confort'] = self.df[cols_confort].sum(axis=1)
#         self.df['Chambres_par_Superficie'] = self.df['Chambres'] / (self.df['Superficie_m2'] + 0.1)

#         return self.df