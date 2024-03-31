import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn . preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Učitavanje podatkovnog skupa
data = pd.read_csv('lv4/data_C02_emission.csv')

# Odabir željenih numeričkih veličina (stupaca)
numeric_features = ["Engine Size (L)", "Cylinders", "Fuel Consumption City (L/100km)", "Fuel Consumption Hwy (L/100km)", "Fuel Consumption Comb (L/100km)", "Fuel Consumption Comb (mpg)"]  # Zamijenite s stvarnim imenima stupaca

output_data = ["CO2 Emissions (g/km)"]

# Podijela podataka na skup za učenje i skup za testiranje (80%-20%)
X_train , X_test , y_train , y_test  = train_test_split(data[numeric_features], data[output_data], test_size=0.2, random_state=42)


# Pomocu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova ´
# o jednoj numerickoj veli ˇ cini. Pri tome podatke koji pripadaju skupu za u ˇ cenje ozna ˇ cite ˇ
# plavom bojom, a podatke koji pripadaju skupu za testiranje oznacite crvenom bojom.
plt.figure()
plt.scatter(X_train["Engine Size (L)"], y_train, color='blue')
plt.scatter(X_test["Engine Size (L)"], y_test, color='red')
plt.legend(["Trening", "Testiranje"])
plt.xlabel('Engine Size (L)')
plt.ylabel('emisije C02')


# Izvršite standardizaciju ulaznih velicina skupa za u ˇ cenje. Prikažite histogram vrijednosti ˇ
# jedne ulazne velicine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja ˇ
# transformirajte ulazne velicine skupa podataka za testiranje
sc = MinMaxScaler()
train_data_scaled = sc.fit_transform(X_train[["Engine Size (L)"]])

# Prikaz histograma vrijednosti jedne ulazne veličine prije i nakon skaliranja
feature_index = 0  # Odaberite indeks željene ulazne veličine
feature_name = numeric_features[feature_index]
plt.figure(figsize=(12, 5))
oznake = range(0, 20)
plt.subplot(1, 2, 1)
plt.hist(oznake, X_train[["Engine Size (L)"]], bins=20, color='blue', alpha=0.7)
plt.title('Histogram of ' + feature_name + ' before scaling')
plt.xlabel(feature_name)
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(oznake, train_data_scaled, feature_index], bins=20, color='red', alpha=0.7)
plt.title('Histogram of ' + feature_name + ' after scaling')
plt.xlabel(feature_name)
plt.ylabel('Frequency')

plt.tight_layout()

plt.show()