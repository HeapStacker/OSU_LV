import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def my_print(string):
    print("***********************")
    print(string, "\n")

s1 = pd.Series(["Crvenkapica", "baka", "majka", "lovac", "vuk"])
s2 = pd.Series(5., index=["a", "b", "c", "d", "e"], name="ime objekta")
s3 = pd.Series(np.random.randn(5))
my_print(s1)
my_print(s2)
my_print(s2["b"])
my_print(s3)
my_print(s3[3])

#manualno zadani podaci
data = {"country": ["Italy", "Spain", "Greece", "France", "Portugal"],   "population": [59, 47, 11, 68, 10], "code": [39, 34, 30, 33, 351]}

countries = pd.DataFrame(data, columns=["country", "population", "code"])
my_print(countries)
my_print(countries.info())
my_print(countries.head(2))
my_print(countries.tail(2))
my_print(countries.describe()) #vraca statistiku za svaku veličinu u Data 

#za .mean() i .median moramo izdvojiti pravilan stupac DataFrame-a
my_print(countries["population"].mean())
my_print(countries["population"].median())
my_print(countries.max())
my_print(countries.min())
my_print(countries.sort_values(by=["population"]))

#izdvajanje više od jednog stupca ide ovako...
my_print(countries[["population", "code"]])
#za izdvajanje jednog stupca možeš ovako...
my_print(countries[["code"]])
#ili ovako (al ovako osim Name: code, dobimo i dtype)...
my_print(countries["code"])
#ili isto ko i ovo iznad (pošto nema razmaka u "code")...
my_print(countries.code)

#iloc izdvaja određene redove i stupaca [redovi, stupci] Dataframea
my_print(countries.iloc[2:, :])
my_print(countries.iloc[1:3, [0, 1]])


#učitavanje podataka iz datoteke
data = pd.read_csv("lv3/data_C02_emission.csv")

#logičke provjere
#izbacuje samo true/false za redove u kojima su Cylinders veći od 4
my_print(data["Cylinders"] > 4)

#Izbacuje samu tablicu s navedenim stupcima "Model", "Vehicle Class", "Cylinders" gdje su Cylinders veći od 4
my_print(data[["Model", "Vehicle Class", "Cylinders"]][data["Cylinders"] > 4])

#izbaci sve retke gdje su cilindri > 4 i gdje je motor najveći (bitno je da logiku odvojiš s & i zatvoriš svaku zagradama)
my_print(data[(data.Cylinders > 4) & (data["Engine Size (L)"] == data["Engine Size (L)"].max())])

#dodavanje novih stupaca...
data["jedinice"] = np.ones(len(data))
data["large"] = (data.Cylinders > 10)

#ispišemo nove stupce npr.
my_print(data[["Model", "Engine Size (L)", "jedinice", "large"]][data.large == True])

#grupiranje DataFrame-a...
new_data = data.groupby("Cylinders")
print("count")
my_print(new_data.count())
print("size") #isto ko i .count() samo ispisuje sve stupce 
my_print(new_data.size())
print("sum") #riječi se sumiraju tako da ih povezujemo
my_print(new_data.sum())
print("mean") #srednje vrijednosti veličine motora za svaki broj cilindra
my_print(new_data["Engine Size (L)"].mean())
#count, sum i mean funkcioniraju i na ne grupiranom DataFrame-u


#null vrijednosti i brisanje tih nepotrebnih podataka...
#koliko ima null podataka po svakom stupcu
my_print(data.isnull().sum())
#PROBAJ OBRISATI NEKE STAVKE NA KRAJU CSV-a
#brisanje redova u kojima barem 1 vrijednost nedostaje
my_print(data.dropna(axis=0).tail(5))
#axis=1 briše stupac u kojem fali vrijedost
my_print(data.dropna(axis=1).tail(5))
#brisanje duplikatnih redova
my_print(data.drop_duplicates())
data = data.dropna(axis=0).tail(5)
#bez reset_index(drop=True) se indexi ne budu izmjenili na novu pravilnu vrijednost
data = data.reset_index(drop=True)
my_print(data)


#grafičko prikazivanje podataka Data-frame-a
fig = plt.figure("Data frame graphs", figsize=(10, 2))
data_frame = 1

#postavi row_size i column_size u onoliko grafova koliko misliš postaviti
row_size = 1
column_size = 2

#bins je kolko stupaca postavljamo u histogramu
def add_histogram(data, bins=20):
    global data_frame #moraš ovdje zadati da je global jer inaće python neće otkriti ovu varijablu u lokalnom scope-u
    fig.add_subplot(row_size, column_size, data_frame)
    data.plot(kind="hist", bins = bins, color="red")
    data_frame += 1

def add_box_plot(data):
    global data_frame #moraš ovdje zadati da je global jer inaće python neće otkriti ovu varijablu u lokalnom scope-u
    fig.add_subplot(row_size, column_size, data_frame)
    data.plot(kind="box", color="blue")
    data_frame += 1

add_histogram(data["Fuel Consumption Comb (L/100km)"])
add_box_plot(data["Fuel Consumption Comb (L/100km)"])


#još možemo grupirati podatke i pokazati ih preko boxplota
#radi mini grafove s C02 emisijama za svaki broj cilindara!
data = pd.read_csv("lv3/data_C02_emission.csv")
grouped = data.groupby("Cylinders")
# grouped.boxplot(column=["CO2 Emissions (g/km)"])
#isto možemo i ovako (Bolje ovako)
data.boxplot(column=["CO2 Emissions (g/km)"], by="Cylinders")


#prikaz podataka scatter grafom...
data.plot.scatter(x="Fuel Consumption City (L/100km)", y="Fuel Consumption Hwy (L/100km)", c="Engine Size (L)", cmap="hot", s=50)

print("KORELACIJA")
#Računa linearni odnos odnosno korelaciju između numeričkih veličina dataframea
my_print(data.corr(numeric_only=True))

plt.show()