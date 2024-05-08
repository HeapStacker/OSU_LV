import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn . model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, max_error
from sklearn.preprocessing import OneHotEncoder

def my_print(string):
    print("***********************")
    print(string, "\n")

#a dio zadatka...
data = pd.read_csv("lv4/data_C02_emission.csv")
num_features = ["Fuel Consumption City (L/100km)","Fuel Consumption Hwy (L/100km)","Fuel Consumption Comb (L/100km)","Fuel Consumption Comb (mpg)", "Engine Size (L)", "Cylinders"]
X = data[num_features].to_numpy()
y = data["CO2 Emissions (g/km)"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


#b - matplotlib i dijagram raspršenja - prikažite ovisnost emisije C02 plinova o jednoj numerickoj velicini (podaci ucenja plavom, a podatke testiranja crvenom bojom)
plt.figure(figsize=(6,4))
plt.scatter(X_train[:,0], y_train, color="blue", label="training")
plt.scatter(X_test[:,0], y_test, color="red", label="test")
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.legend()
# plt.show()


#c - standardiziraj ulazne podatke. Prikaz histograma jedne ulazne velicine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja transformirajte ulazne velicine skupa podataka za testiranje.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(X_train[:,0], bins=20, color="blue")
plt.title("Histogram prije skaliranja")
plt.xlabel("Vrijednost")
plt.ylabel("Broj uzoraka")

plt.subplot(1, 2, 2)
plt.hist(X_train_scaled[:,0], bins=20, color="red")
plt.title("Histogram nakon skaliranja")
plt.xlabel("Vrijednost")
plt.ylabel("Broj uzoraka")

plt.tight_layout() #Adjust the padding between and around subplots (možda je dobro za subplotove?)
# plt.show()
X_test_scaled = scaler.transform(X_test)


#d - Izgradnja linearnog regresijskog modela - Ispišite u terminalu parametara modela (povezivanje s izrazom 4.6) 
linear_regression = lm.LinearRegression()
linear_regression.fit(X_train_scaled, y_train)
my_print(linear_regression.coef_) 

#koeficijenti theta su [ -7.83529938  -0.48265607  45.20221679 -14.42360669  1.1945077  10.4901448 ]


#e - Procjena izlazne velicine na temelju ulaznih velicina skupa za testiranje. (Prikazano dijagramom raspršenja odnosa izmedu stvarnih vrijednosti izlazne velicine i procjenjenih)
y_prediction = linear_regression.predict(X_test_scaled)
fuel_consumption = X_test_scaled[:, 0] #0 jer uzimamo prvi stupac (fuel cons...)
plt.figure()
plt.scatter(fuel_consumption, y_test, c="b", label="Real Values", s=5, alpha=0.5)
plt.scatter(fuel_consumption, y_prediction, c="r", label="Prediction", s=5, alpha=0.5)
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.legend()
# plt.show()

#f - Izvršite vrednovanje modela na nacin da izracunate vrijednosti regresijskih metrika na skupu podataka za testiranje.
print("MSE - Mean squared error: {:.5f}".format(mean_squared_error(y_test, y_prediction)))
print("MSE - Mean squared error: {:.5f}".format(np.sqrt(mean_squared_error(y_test, y_prediction))))
print("MAE - Mean absolute error: {:.5f}".format(mean_absolute_error(y_test, y_prediction)))
print("MAPE - Mean absolute percentage error: {:.5f}".format(mean_absolute_percentage_error(y_test, y_prediction))+"%")
print("R^2: {:.5f}".format(r2_score(y_test, y_prediction)))



#Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoricku varijable „Fuel Type“ kao ulaznu velicinu. Pri tome koristite 1-od-K kodiranje kategorickih velicina. Radi jednostavnosti nemojte skalirati ulazne velicine. Komentirajte dobivene rezultate. Kolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? O kojem se modelu vozila radi?

data = pd.read_csv("lv4/data_C02_emission.csv")

encoder = OneHotEncoder()
X_encoded = pd.DataFrame(encoder.fit_transform(data[["Fuel Type"]]).toarray())
data = data.join(X_encoded)

data.columns = ["Make","Model","Vehicle Class","Engine Size (L)","Cylinders","Transmission","Fuel Type","Fuel Consumption City (L/100km)","Fuel Consumption Hwy (L/100km)","Fuel Consumption Comb (L/100km)","Fuel Consumption Comb (mpg)","CO2 Emissions (g/km)","Fuel0", "Fuel1", "Fuel2", "Fuel3"]
y = data["CO2 Emissions (g/km)"].copy()
X = data.drop("CO2 Emissions (g/km)", axis=1)
X_train_all , X_test_all , y_train , y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

X_train = X_train_all[["Engine Size (L)","Cylinders","Fuel Consumption City (L/100km)","Fuel Consumption Hwy (L/100km)","Fuel Consumption Comb (L/100km)","Fuel Consumption Comb (mpg)","Fuel0", "Fuel1", "Fuel2", "Fuel3"]]
X_test = X_test_all[["Engine Size (L)","Cylinders","Fuel Consumption City (L/100km)","Fuel Consumption Hwy (L/100km)","Fuel Consumption Comb (L/100km)","Fuel Consumption Comb (mpg)","Fuel0", "Fuel1", "Fuel2", "Fuel3"]]

linear_regression = lm.LinearRegression()
linear_regression.fit(X_train,y_train)
y_prediction = linear_regression.predict(X_test)

plt.figure()
plt.scatter(X_test["Fuel Consumption City (L/100km)"],y_test, c="b",label="Real values", s=5, alpha=0.5)
plt.scatter(X_test["Fuel Consumption City (L/100km)"],y_prediction, c="r",label="Prediction", s=5, alpha=0.5)
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.legend()
# plt.show()

maxerror = max_error(y_test, y_prediction)
print("Max pogreška u procjeni: {:.3f}".format(maxerror))
print(f"Model vozila s max pogreškom u procjeni: {X_test_all[abs(y_test-y_prediction) == maxerror]["Model"].iloc[0]}")



plt.show()