import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import max_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# Zadatak 4.5.2 Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoriˇcku
# varijable „Fuel Type“ kao ulaznu veliˇcinu. Pri tome koristite 1-od-K kodiranje kategoriˇckih
# veliˇcina. Radi jednostavnosti nemojte skalirati ulazne veliˇcine. Komentirajte dobivene rezultate.
# Kolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? O kojem se modelu
# vozila radi?

data = pd.read_csv("lv4/data_C02_emission.csv")

ohe = OneHotEncoder()
def encode_columns(dataframe, columns_to_encode):
    encoded_dfs = []
    for column in columns_to_encode:
        #fit_transformu moramo predati [[]] (DataFrame) a ne [] (Series)
        encoded_array = ohe.fit_transform(dataframe[[column]]).toarray() # veaća numpy arr
        encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out([column]))
        encoded_dfs.append(encoded_df)
    return pd.concat([dataframe.drop(columns_to_encode, axis=1)] + encoded_dfs, axis=1)

def evaluate_regresion_model(original_data, y_real, y_prediction):
    max = max_error(y_real, y_prediction)
    max_error_index = np.argmax(np.abs(y_real - y_prediction))
    print(f"Max Error: {max:.5f}")
    print("Element with biggest error is...", original_data.values[max_error_index, :])


cols_to_encode = ["Fuel Type", "Make", "Model", "Vehicle Class", "Transmission"]

# Spajanje kodiranih stupaca s izvornim podacima (osim kodiranih stupaca)
data_encoded = encode_columns(data, cols_to_encode)
X_cols = data_encoded.drop(["CO2 Emissions (g/km)"], axis=1) # ovo je DataFrame
y_cols = data_encoded["CO2 Emissions (g/km)"] # ovo je Series ( jer nije [[]] nego [] )
X_train, X_test, y_train, y_test = train_test_split(X_cols, y_cols, test_size=0.2, random_state=1)

# izrada modela...
regr = lm.LinearRegression()
regr.fit(X_train, y_train)

# izrada predikcije...
y_test_pred = regr.predict(X_test) 

evaluate_regresion_model(data, y_test, y_test_pred)
# GREŠKA U IZRAČUNU 100% JER SU ZABILJEŽENE POGREŠKE ABNORMALNE