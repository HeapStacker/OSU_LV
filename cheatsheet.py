import numpy as np
import pandas as pd
import matplotlib.colors
from matplotlib import cm
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, max_error


# NUMPY CHEEATSHEET
#**********************************************************************

a = np.array([0, 1, 2])

# INFO
print(type(a))
print(a.shape)

# INICIJALIZACIJA ARRAY-A
a = np.array([6, 2, 9])
x = np.linspace(0, 10, 20)
y = np.sin(x) * np.cos(x)
zeros = np.zeros((10, 10)).astype(np.bool_)
zeros = np.zeros((10, 10)).astype(np.uint16)
zeros = np.zeros((10, 10)).astype(np.int16)
ones = np.array([[1] * 10] * 10)
eye_mat = np.eye(10)

# RANDOM
np.random.seed(100)
x = np.random.uniform(1.0, 5.0, 100)                  # vrijednosti u intervalu [1.0, 5>
x = np.random.uniform(1.0, 5.0, 100).astype(np.uint8) # vrijednosti u skupu {1, 2, 3, 4}
x = np.random.random(10)                              # 10 brojeva u intervalu [0.0, 1.0>
x = np.random.normal(5, 1, 100)                       # 5 je mean, 1 je standardna devijacija
x = np.random.randn(10)                               # 0 je mean, 1 je varijanca (generira 10 brojeva)

# GLAVNE FUNKCIJE
x = np.max(x)
x = np.min(x)
x = np.argmax(x)
x = np.argmin(x)
x = np.mean(x)
x = np.median(x)
x = np.unique(x) 

matrica = np.array(
    [
        [9, 1, 2, 5],
        [3, 7, 2, 4],
        [0, 6, 8, 1],
        [1, 0, 2, 0],
        [4, 3, 1, 4],
        [4, 3, 1, 4],
    ]
)

# SORTIRANJE
prema_gore = np.sort([1, 2, 6, 3, 0, 11, 8, 9])
prema_dolje = np.sort([1, 2, 6, 3, 0, 11, 8, 9])[::-1]
sorted_rows = np.sort(matrica, axis=1)
sorted_columns = np.sort(matrica, axis=0)
transponiranje = matrica.transpose()

# PRIDRUŽIVANJE
b = np.concatenate((a, x))
together_row = np.concatenate((zeros, ones), axis=1)
together_column = np.concatenate((zeros, ones), axis=0) # default

# RESHAPE
matr = matrica.reshape((6, -1)) # isto ko i reshape((6,4))
array = together_column.ravel() # vrati matricu u vektor

# SELEKTIRANJE
a = np.random.uniform(0, 10, 100).astype(np.uint8)
b = np.array(list(range(0, 10)) * 10)
c = a[a == b]  # a == b daje masku (True/False)
d = a[(a == b) * 0 | (a != b) * (len(a) - 1)]
e = a[[0, 1, 2, 1, 1, 2]]


# PANDAS CHEEATSHEET
#**********************************************************************

# SERIES ----------------------------------------------------------------------------------
s1 = pd.Series(["Crvenkapica", "baka", "majka", "lovac", "vuk"])
s2 = pd.Series(5., index=["a", "b", "c", "d", "e"], name="ime objekta")
s3 = pd.Series(np.random.randn(5))

# PRINTANJE
print(s2.max())
print(s2.min())
print(s2.argmax())
print(s2.argmin())
print(s2.mean())
print(s2.median())
print(s2.to_numpy())
print(s2.to_list())
print(s2.unique())

# PRIDRUŽIVANJE
s4 = pd.concat([s1, s2]) # [] možeš zamijeniti s ()

# DATAFRAME -------------------------------------------------------------------------------
data = {"country": ["Italy", "Spain", "Greece", "France", "Portugal"],   "population": [59, 47, 11, 68, 10], "code": [39, 34, 30, 33, 351]}
countries = pd.DataFrame(data, columns=["country", "population", "code"])
car_emmisions = pd.read_csv("lv3/data_C02_emission.csv")

# INFO
print(countries.info())              # za provjeru null vrijednosti i tipova podataka
print(countries.describe())          # za numeričke vrijednosti (count, mean, std, min, ...)
print(countries.isnull())            # vraća masku (True/False) za sve vrijednosti
print(countries.isnull().sum())      # sumacija null vrijednosti za svaki stupac
print(countries.columns.to_numpy())  # vraća nazive stupaca

# BRISANJE
car_emmisions = car_emmisions.dropna(axis=0) # briše redove s null vrijednostima (AXIS SUPROTNO OD SVEGA DRUGOG)
car_emmisions = car_emmisions.dropna(axis=1) # briše stupce s null vrijednostima (AXIS SUPROTNO OD SVEGA DRUGOG)
car_emmisions = car_emmisions.drop_duplicates()
car_emmisions = car_emmisions.reset_index(drop=True)

# SORTIRANJE
sortirano_po_populaciji = countries.sort_values(by=["population"], ascending=False)
sortirano_po_indexu = countries.sort_index(ascending=False)

# SELEKTIRANJE
drzava_s_najvecom_populacijom = sortirano_po_populaciji.to_numpy()[0, :]
ista_stvar_samo_DataFrame = sortirano_po_populaciji.head(1)
svaka_treca = countries[["country", "code"]][::3]
countries_head = countries.head(10)
countries_tail = countries.tail(10)

# GRUPIRANJE 
najvece_emisije_po_tipu_goriva = car_emmisions.groupby("Fuel Type")[["CO2 Emissions (g/km)"]].agg(["max", "min"])
countries[["population", "code"]].agg(["max", "min", "mean", "argmax"])
max_za_populaciju_i_code = countries[["population", "code"]].max()
min_za_populaciju = countries[["population"]].min()

# PRIDRUŽIVANJE
concated_df = pd.concat([countries, car_emmisions], axis=1) # možemo veći dodat manjem (al će bit puno NaN vrijednosti)
concated_df = pd.concat([countries, countries], axis=0) # dodaje se ispod drugog (ima smisla samo za jednake skupove pod.)


# PLOTTING
#**********************************************************************
x = np.linspace(1, 10, 20)
y = np.sin(x)

# lista markera
# . , o v ^ < > 1 2 3 4 8 s p * h H + x D d | _ P X
plt.plot(x, y, "b", linewidth = 2, marker="d", markersize=10, c="gray")
plt.plot(x, y + 10, "b", linewidth = 2, marker="d", markersize=10, c="red")
plt.xlabel("x")
plt.ylabel("vrijednost funkcije")
plt.title("Funkcija")
plt.show()

# različite boje u scatter-u
# plt.scatter(x, y, c=cm.get_cmap("magma")(0.3), label="prvi")
# plt.scatter(x, y, c=cm.get_cmap("magma")(0.6), label="drugi") 
plt.scatter(x, y, c="red", label="prvi")
plt.scatter(x, y + 10, c="black", label="drugi") 
# plt.scatter(x, y, c=(0.2, 1, 0.6), label="prvi")
# plt.scatter(x, y, c=(1, 0.3, 0.9), label="drugi") 
plt.xlabel("CO2 emisija (g/km)")
plt.ylabel("veličina motora (L)")
plt.legend()
plt.show()

plt.hist(y, bins=20, color="blue")
plt.title("Histogram prije skaliranja")
plt.xlabel("Vrijednost")
plt.ylabel("Broj uzoraka")
plt.show()

# bar i boxplot je najlakše napraviti s DataFrame-om...
grouped_by_fuel_types = car_emmisions.groupby("Fuel Type")["Model"].count()
grouped_by_fuel_types.plot.bar()
plt.show()

car_emmisions.boxplot(column="Fuel Consumption Hwy (L/100km)", by="Fuel Type")
plt.show()

car_emmisions["Fuel Consumption Hwy (L/100km)"].plot(kind="box")
plt.show()

# preko plt-a...
grouped_by_fuel_types = car_emmisions.groupby("Fuel Type")["Model"].count()
plt.bar(grouped_by_fuel_types.index, grouped_by_fuel_types.values)
plt.title("Bar plt")
plt.xlabel("Vrijednost")
plt.ylabel("Broj uzoraka")
plt.show()


plt.boxplot(x=car_emmisions["Fuel Consumption Hwy (L/100km)"], vert=False)
plt.title("Boxplt")
plt.xlabel("Vrijednosti")
plt.ylabel("")
plt.show()


# EVALUACIJE
#**********************************************************************
y_real = 2
y_prediction = 2

# GREŠKE
mse = mean_squared_error(y_real, y_prediction)
print(f"MSE - Mean squared error: {mse:.5f}")
print(f"RMSE - Root Mean squared error: {np.sqrt(mse):.5f}")
print(f"MAE - Mean absolute error: {mean_absolute_error(y_real, y_prediction):.5f}")
print(f"MAPE - Mean absolute percentage error: {mean_absolute_percentage_error(y_real, y_prediction) * 100:.5f}%")
print(f"MAXERR - Max error: {max_error(y_real, y_prediction)}")
print(f"R^2 - Coefficient of determination: {r2_score(y_real, y_prediction):.5f}") 

# točnost, preciznost, ...
print("accuracy:    ", accuracy_score(y_real, y_prediction))                        #(True Positive + True Negative) / Total Predictions
print("precision:   ", precision_score(y_real, y_prediction))                       #True Positive / (True Positive + False Positive)
print("sensitivity: ", recall_score(y_real, y_prediction))                          #True Positive / (True Positive + False Negative)
print("specificity: ", recall_score(y_real, y_prediction, pos_label=0))             #True Negative / (True Negative + False Positive)
print("F-score:     ", f1_score(y_real, y_prediction))                              #2 * ((Precision * Sensitivity) / (Precision + Sensitivity))


# LINEARNA REGRESIJA
#**********************************************************************

# Primjer sa skaliranjem ----------------------------------------------------------------------------------
data = pd.read_csv("lv4/data_C02_emission.csv")
num_features = ["Fuel Consumption City (L/100km)","Fuel Consumption Hwy (L/100km)","Fuel Consumption Comb (L/100km)","Fuel Consumption Comb (mpg)", "Engine Size (L)", "Cylinders"]

X = data[num_features]
y = data["CO2 Emissions (g/km)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

regr = lm.LinearRegression()
regr.fit(X_train_scaled, y_train)
print("Koeficijenti mog linearnog reg. modela =", regr.coef_)
y_test_pred = regr.predict(scaler.transform(X_test)) 


# Enkodiranje podataka (Hot Encoding) ---------------------------------------------------------------------
data = pd.read_csv("lv4/data_C02_emission.csv")
cols_to_encode = ["Fuel Type"]

ohe = OneHotEncoder()
def encode_columns(dataframe, columns_to_encode):
    encoded_dfs = []
    for column in columns_to_encode:
        encoded_array = ohe.fit_transform(dataframe[[column]]).toarray()
        encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out([column]))
        encoded_dfs.append(encoded_df)
    return pd.concat(encoded_dfs, axis=1)

data_encoded = encode_columns(data, cols_to_encode)

X_cols = pd.concat([data[num_features], data_encoded], axis=1)
y_cols = data["CO2 Emissions (g/km)"]

X_train, X_test, y_train, y_test = train_test_split(X_cols.to_numpy(), y_cols.to_numpy(), test_size=0.2, random_state=1)

# izrada modela...
regr = lm.LinearRegression()
regr.fit(X_train, y_train)

# izrada predikcije...
y_test_pred = regr.predict(X_test) 

# na kraju samo treba evaluirati model
def evaluate_regresion_model(original_data, y_real, y_prediction):
    max = max_error(y_real, y_prediction)
    max_error_index = np.argmax(np.abs(y_real - y_prediction))
    print(f"Max Error: {max:.5f}")
    print("Element with biggest error is...", original_data.to_numpy()[max_error_index, :])


# LOGISTIČKA REGRESIJA
#**********************************************************************