import joblib
import pandas as pd
import numpy as np

class PredicteurLoyer:
    def __init__(self, chemin_fichier):
        données = joblib.load(chemin_fichier)
        self.modele = données['modele']
        self.encoder = données['encoder']
        self.medianes = données['mean']
        self.modes = données['modes']
        self.colonnes_attendues = données['colonnes_attendues']

    def pretraiter(self, df_brut):
        df = df_brut.copy()

        for col, val in self.mean.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        
        for col, val in self.modes.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        cols_bin = ['Salon', 'SalleDeBainInterieure', 'Parking', 'Meuble', 'Jardin']
        for col in cols_bin:
            if col in df.columns:
                df[col + "_Bin"] = df[col].map({"Oui": 1, "Non": 0}).fillna(0).astype(int)

        if "Quartier" in df.columns:
            enc_vals = self.encoder.transform(df[["Quartier"]])
            df["Quartier_Target"] = enc_vals[:, 0]

        try:
            X_final = df[self.colonnes_attendues]
        except KeyError as e:
            raise ValueError(f"Colonne manquante dans les données d'entrée : {e}")

        return X_final

    def predire(self, df_brut):
        X = self.pretraiter(df_brut)
        prediction_log = self.modele.predict(X)
        
        return np.expm1(prediction_log)

# --- Utilisation ---
# predicteur = PredicteurLoyer("data/modele_production_complet.pkl")
# nouvelles_donnees = pd.DataFrame([...])
# prix = predicteur.predire(nouvelles_donnees)   