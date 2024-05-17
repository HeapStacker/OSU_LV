#imports
import pandas as pd
import numpy as np


# Skripta zadatak_1.py učitava podatkovni skup iz data_C02_emission.csv.
# Dodajte programski kod u skriptu pomoću kojeg možete odgovoriti na sljedeća pitanja

print("**************************************************************************************")
# a) Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka veličina? Postoje li izostale ili
# duplicirane vrijednosti? Obrišite ih ako postoje. Kategoričke veličine konvertirajte u tip
# category.


dataFrame = pd.read_csv("lv3/data_C02_emission.csv")
values = []
for line in open("lv3/data_C02_emission.csv"):
    for value in line.strip().split(","):
        print(f"{value} is type {dataFrame[value].dtype}")
    break
print(f"Data frame sadrži {len(dataFrame)} mjerenja.")

#brisanje redova s praznim elementima
if dataFrame.isnull().sum().max() == 0:
    print("Looks like values aren't missing")
else:
    print("Some values are missing, we will delete those rows...")
    dataFrame = dataFrame.dropna(axis=0)
    dataFrame = dataFrame.reset_index(drop=True)

#brisanje dupliciranih vrijednosti
if len(dataFrame) - len(dataFrame.drop_duplicates()) == 0:
    print("Ne postoje duplicirne vrijednosti")
else:
    print("Postoje duplicirane vrijednosti")
    dataFrame = dataFrame.drop_duplicates()
    dataFrame = dataFrame.reset_index(drop=True)


#za kraj konvertiranje kategoričkih veličina u tip kategory
dataFrame.Make = dataFrame.Make.astype("category")
dataFrame.Model = dataFrame.Model.astype("category")
dataFrame["Vehicle Class"] = dataFrame["Vehicle Class"].astype("category")
dataFrame.Transmission = dataFrame.Transmission.astype("category")
dataFrame["Fuel Type"] = dataFrame["Fuel Type"].astype("category")
print(dataFrame.dtypes)


print("**************************************************************************************")
# b) Koja tri automobila ima najve ́cu odnosno najmanju gradsku potrošnju? Ispišite u terminal:
# ime proizvođaća, model vozila i kolika je gradska potrošnja.

dataFrame = pd.read_csv("lv3/data_C02_emission.csv")
vals = ["Make", "Model", "Fuel Consumption City (L/100km)"]

sorted_data = dataFrame.sort_values(by=vals[2], ascending=True)[vals]
#ima još i sort_index()

print(sorted_data.head(3))


print("**************************************************************************************")
# c) Koliko vozila ima veličinu motora između 2.5 i 3.5 L? Kolika je prosjeˇcna C02 emisija
# plinova za ova vozila?
vals += ["Engine Size (L)", "CO2 Emissions (g/km)"]
selected = dataFrame[vals][(dataFrame["Engine Size (L)"] >= 2.5) & (dataFrame["Engine Size (L)"] <= 3.5)].sort_values("Engine Size (L)")

print(f"Ima {len(selected)} vozila s veličinom motora između 2.5 i 3.5 L")
print(f"Prosječna potrošnja C02 emisija za određena vozila je {selected["CO2 Emissions (g/km)"].mean()} g/km")



print("**************************************************************************************")
#d) Koliko mjerenja se odnosi na vozila proizvo  ̄daˇca Audi? Kolika je prosjeˇcna emisija C02
#plinova automobila proizvo  ̄daˇca Audi koji imaju 4 cilindara?

print(f"Izvedeno je {len(dataFrame[dataFrame.Make == "Audi"])} mjerenja na vozilu proizvođača Audi")
audi_4_cil = dataFrame[(dataFrame.Make == "Audi") & (dataFrame.Cylinders == 4)]
print(f"Prosječna emisija C02 plinova na autu proizvođača Audija s 4 cilindra {audi_4_cil["CO2 Emissions (g/km)"].mean()}")



print("**************************************************************************************")
# e) Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjeˇcna emisija C02 plinova s obzirom na
# broj cilindara?

parni_cil = []
for i in range(4, dataFrame.Cylinders.max() + 1, 2): #stavio sam + 1 da se i zadnja stavka postavi u range
    parni_cil.append(i)

vozila_s_parnim_br_cil = dataFrame[dataFrame.Cylinders.isin(parni_cil)]
print(f"Vozila s parnim br. cilindra ima {vozila_s_parnim_br_cil["Cylinders"]}")

prosijek_co2_po_cil = dataFrame.groupby("Cylinders")["CO2 Emissions (g/km)"].mean()
print(prosijek_co2_po_cil)
print("Izgleda da prosijek raste s brojem cilindara :)")

print("**************************************************************************************")
# f) Kolika je prosjeˇcna gradska potrošnja u sluˇcaju vozila koja koriste dizel, a kolika za vozila
# koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?

print(f"Prosjećna gradska potrošnja co2 za dizelaše = {dataFrame[dataFrame["Fuel Type"] == "D"]["CO2 Emissions (g/km)"].mean()}")
print(f"Prosjećna gradska potrošnja co2 za vozila koja troše benzin = {dataFrame[dataFrame["Fuel Type"] == "X"]["CO2 Emissions (g/km)"].mean()}")


print("**************************************************************************************")
#g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najve ́cu gradsku potrošnju goriva

print(dataFrame[(dataFrame["Fuel Type"] == "D") & (dataFrame.Cylinders == 4)].sort_values(by="Fuel Consumption City (L/100km)", ascending=False)[vals + ["Fuel Consumption City (L/100km)"]].head(1))


print("**************************************************************************************")
#h) Koliko ima vozila ima ruˇcni tip mjenjaˇca (bez obzira na broj brzina)?

#isto se riješi kao i ostali primjeri gore :/
print()


print("**************************************************************************************")
#i) Izraˇcunajte korelaciju izme  ̄du numeriˇckih veliˇcina. Komentirajte dobiveni rezultat.

print(f"Korelacija numeričkih veličina iznosi\n {dataFrame.corr(numeric_only=True)}")
