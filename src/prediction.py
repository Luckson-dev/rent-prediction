import joblib
import pandas as pd
import numpy as np

class RENTPrediction:
    def __init__(self, file_path):
        data = joblib.load(file_path)
        self.boosting_model = data['boosting_model']
        self.encoder = data['encoder']
        self.mean = data['mean']
        self.modes = data['modes']
        self.columns = data['columns']

    def pipeline(self, in_comming_data):
        df = in_comming_data.copy()

        for col, val in self.mean.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        
        for col, val in self.modes.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        cols_bin = ['Salon', 'SalleDeBainInterieure', 'Parking', 'Meuble', 'Jardin']

        for col in ["Meuble", "Jardin"]:
            if col in df.columns:
                df[col + "_Bin"] = df[col].map({"Oui": 1, "Non": 0}).fillna(0).astype(int)

        if "Quartier" in df.columns:
            enc_vals = self.encoder.transform(df[["Quartier"]])
            df["Quartier_Target"] = enc_vals[:, 0]

        try:
            X_final = df[self.columns]
        except KeyError as e:
            raise ValueError(f"Columns doesn't match : {e}")

        return X_final

    def predict(self, in_comming_data):
        X = self.pipeline(in_comming_data)

        print(f"Processed Data: {X}")
        print(f"Boosting Model: {self.boosting_model}")

        prediction_log = self.boosting_model.predict(X)

        print(f"Prediction (log scale): {prediction_log}")
        
        return np.expm1(prediction_log)  