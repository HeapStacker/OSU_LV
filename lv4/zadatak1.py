import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, max_error

# Skripta zadatak_1.py uˇcitava podatkovni skup iz data_C02_emission.csv.
# Potrebno je izgraditi i vrednovati model koji procjenjuje emisiju C02 plinova na temelju os-
# talih numeriˇckih ulaznih veliˇcina. Detalje oko ovog podatkovnog skupa mogu se prona ́ci u 3.
# laboratorijskoj vježbi.



# a) učitavanje numeričkih veličina i podjela na train/test u omjeru 80%/20%
data = pd.read_csv("lv4/data_C02_emission.csv")
num_features = ["Fuel Consumption City (L/100km)","Fuel Consumption Hwy (L/100km)","Fuel Consumption Comb (L/100km)","Fuel Consumption Comb (mpg)", "Engine Size (L)", "Cylinders"]
X = data[num_features]
y = data["CO2 Emissions (g/km)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# ne mora stavit random test al onda će uvijek podjela bit drugačija



# b) Pomo ́cu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova
# o jednoj numeriˇckoj veliˇcini. Pri tome podatke koji pripadaju skupu za uˇcenje oznaˇcite
# plavom bojom, a podatke koji pripadaju skupu za testiranje oznaˇcite crvenom bojom.
from matplotlib import cm
plt.scatter(X_train["Engine Size (L)"], y_train, c=cm.get_cmap("magma")(0.3), label="train")
plt.scatter(X_test["Engine Size (L)"], y_test, c=cm.get_cmap("magma")(0.6), label="test") 
# c=cm.get_cmap("magma")(0.6) - daje nam boju iz cmape "magma" definiramo broj 0.5 (može biti broj od 0 do 1)
#boja u pythonu može biti općenito def. kao broj od 0 do 1 (npr. c = (0.2, 0.5, 0.1))
plt.xlabel("CO2 emisija (g/km)")
plt.ylabel("veličina motora (L)")
plt.legend()
plt.show()



# c) Izvršite standardizaciju ulaznih veliˇcina skupa za uˇcenje. Prikažite histogram vrijednosti
# jedne ulazne veliˇcine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja
# transformirajte ulazne veliˇcine skupa podataka za testiranje.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(X_train["Engine Size (L)"], bins=20, color="blue")
plt.title("Histogram prije skaliranja")
plt.xlabel("Vrijednost")
plt.ylabel("Broj uzoraka")


print(type(X_train_scaled))
print(X_train_scaled.shape)
#fit_transform ne vraća DataFrame nego numpy array

plt.subplot(1, 2, 2)
plt.hist(X_train_scaled[:, 4], bins=20, color="red")
plt.title("Histogram nakon skaliranja")
plt.xlabel("Vrijednost")
plt.ylabel("Broj uzoraka")

plt.tight_layout()
plt.show()



# d) Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i
# povežite ih s izrazom 4.6.
regr = lm.LinearRegression()
regr.fit(X_train_scaled, y_train)
print("Koeficijenti mog linearnog reg. modela =", regr.coef_)



# e) Izvršite procjenu izlazne veliˇcine na temelju ulaznih veliˇcina skupa za testiranje. Prikažite
# pomo ́cu dijagrama raspršenja odnos izmedu stvarnih vrijednosti izlazne veliˇcine i procjene
# dobivene modelom.
y_test_pred = regr.predict(scaler.transform(X_test)) 
# predict isto vraća numpy array iako smo mu predali dataframe

plt.scatter(X_test.values[:, 0], y_test) # X_test je dataframe al .values daje numpy arr, y_test je dataframe (al ima samo 1 stupac)
plt.scatter(X_test.values[:, 0], y_test_pred) # y_test_pred je numpy arr
plt.legend(["pravi ishod", "predviđeni ishod"]) # mogli smo i label stavit na svaki .plt pa onda prazan legend (plt.legend())
plt.xlabel("proizvoljno odabrani stupac ulaznih (testnih) podataka")
plt.ylabel("izlazni (testni) podaci")
plt.show()



# f) Izvršite vrednovanje modela na naˇcin da izraˇcunate vrijednosti regresijskih metrika na
# skupu podataka za testiranje.

def evaluate_regresion_model(y_real, y_prediction):
    mse = mean_squared_error(y_real, y_prediction)
    print(f"MSE - Mean squared error: {mse:.5f}")
    print(f"RMSE - Root Mean squared error: {np.sqrt(mse):.5f}")
    print(f"MAE - Mean absolute error: {mean_absolute_error(y_real, y_prediction):.5f}")
    print(f"MAPE - Mean absolute percentage error: {mean_absolute_percentage_error(y_real, y_prediction) * 100:.5f}%")
    print(f"MAXERR - Max error: {max_error(y_real, y_prediction)}")

    #izbacuje vrijednost od 0 do 1 (1 znaći da tvoj model odlično funkcionira, 0 znači da je loš)
    print(f"R^2 - Coefficient of determination: {r2_score(y_real, y_prediction):.5f}") 

print("\n\nOvo su neke metrike mojeg linearnog regresijskog modela...")
evaluate_regresion_model(y_test, y_test_pred)



# g) Što se događa s vrijednostima evaluacijskih metrika na testnom skupu kada mijenjate broj
# ulaznih veliˇcina?

# tip podatka .columns je Index, ak staviš .tolist() dobiš listu (naziva stupaca)
# print(X_test.columns.tolist()) #ili .to_numpy()
# ZAPAMTI - .columns za imena stupaca, .values za numpy arr samih vrijednosti

arr = []
print()
for i in range(len(X_train.columns.tolist())):
    arr.append(i)
    print("Evaluacija za različit broj ulaznih stupaca n. n =", i + 1)

    # prvo postavimo pravilan skalar
    X_train_scaled = scaler.fit_transform(X_train.values[:, arr])

    # zatim napravimo model
    regr.fit(X_train_scaled[:, arr], y_train)

    #na kraju napravio predikciju i evaluaciju
    y_test_pred = regr.predict(scaler.transform(X_test.values[:, arr]))
    evaluate_regresion_model(y_test, y_test_pred)
    print()

# na kraju čemo uočiti da ako povećavamo broj stupaca, krajnja predikcija bit će bolja
# vrlo jednostavno za zaključiti