import pandas as pd
from rent_prediction.utils.get_path import get_filename
from rent_prediction.data.data_preparation import DATAPreparation
from rent_prediction.data.feature_engineering import FTEngineering

df_raw = pd.read_csv(get_filename("rent_prediction.csv"))

# Nettoyage et Préparation
prep = DATAPreparation(df_raw)
prep.clean_data()
train_data, test_data = prep.split_data()
encoder, train, test = prep.encode_features()

# Feature
# fe_train = FTEngineering(train)
# fe_test = FTEngineering(test)

# train_final = fe_train.create_importante_features()
# test_final = fe_test.create_importante_features()

# Sauvegarde ou Entraînement
train_final.to_csv(get_filename("final_train.csv"), index=False)
test_final.to_csv(get_filename("final_test.csv"), index=False)

# OU passer directement au modèle
# model.fit(train_final[features], train_final['target'])