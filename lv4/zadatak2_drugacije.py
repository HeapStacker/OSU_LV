import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn.metrics import max_error

data = pd.read_csv("lv4/data_C02_emission.csv")

def evaluate_regresion_model(original_data, y_real, y_prediction):
    max = max_error(y_real, y_prediction)
    max_error_index = np.argmax(np.abs(y_real - y_prediction))
    print(f"Max Error: {max:.5f}")
    print("Element with biggest error is...", original_data.to_numpy()[max_error_index, :])

# One Hot Encoding
# prefix i prefix_sep su = "" tako da nemam ništa "dodatno" u nazivu stupca
# drop_first - ak hoćeš maknut prvi stupac (te vrijednosti prepoznamo tako da su svi ostali stupci 0)
cols_to_encode = ["Fuel Type"]
num_features = ["Fuel Consumption City (L/100km)","Fuel Consumption Hwy (L/100km)","Fuel Consumption Comb (L/100km)","Fuel Consumption Comb (mpg)", "Engine Size (L)", "Cylinders"]
ohe_cols = pd.get_dummies(data[cols_to_encode], prefix="", prefix_sep="", drop_first=False)

X = pd.concat([data[num_features], ohe_cols], axis=1) # predajemo listu DataFrame-a
y = data["CO2 Emissions (g/km)"]

from sklearn.model_selection import train_test_split
X_tr, X_tst, y_tr, y_tst = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.2, random_state=1)

regr = lm.LinearRegression()
regr.fit(X_tr,y_tr)

evaluate_regresion_model(data, y_tst, regr.predict(X_tst))

