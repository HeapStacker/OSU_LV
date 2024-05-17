import matplotlib.colors
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# MOŽEMO SA SIGURNOŠĆU REĆI DA - ako radimpo data.plot.scatter (na dataframe-u), dobit ćemo
# više informativnih popuna na plotu nego da radimo plt.scater (ovdje ćemo morati sami pisati)
# xlabel, ylabel, title... (ne vrijedi samo za scater, vrijedi i za ostale tipove grafa)

#a) Pomo ́cu histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz
data = pd.read_csv("lv3/data_C02_emission.csv")
data["CO2 Emissions (g/km)"].plot(kind="hist", bins=20)
plt.show()



# b) Pomo ́cu dijagrama raspršenja prikažite odnos izmedu gradske potrošnje goriva i emisije
# C02 plinova. Komentirajte dobiveni prikaz. Kako biste bolje razumjeli odnose izmedu
# veličina, obojite točkice na dijagramu raspršenja s obzirom na tip goriva

# X Regular gasoline
# Z Premium gasoline
# D Diesel
# E Ethanol (E85)
# N Natural gas

#možeš i sam napraviti cmap...
custom_cmap = matplotlib.colors.ListedColormap(["Black", "Sienna", "Red", "Orange", "Yellow"])
linear_custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("cmap_linear", ["Black", "Sienna", "Red", "Orange", "Yellow"])

#++++++++++++++++++++++++++++++++MY_FUNCTION+++++++++++++++++++++++++++++++++++++++++++++++++++
def fill_mapping(unique_list):
   mapping = {}
   for i in range(len(unique_list)):
      mapping[unique_list[i]] = i
   return mapping

def create_mapping(data_frame, col_name):
   unique_list = data_frame[col_name].sort_values().unique().tolist()
   data_frame[col_name] = data_frame[col_name].map(fill_mapping(unique_list))
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

mapped_data = data.copy()
# fuel_map = {"X": 0, "Z": 1, "D": 2, "E": 3, "N": 4} 
# mapped_data["Fuel Type"] = mapped_data["Fuel Type"].map(fuel_map)
create_mapping(mapped_data, "Fuel Type")
mapped_data.plot.scatter(x="Fuel Consumption City (L/100km)", y="CO2 Emissions (g/km)", c="Fuel Type",
 cmap=linear_custom_cmap, s=5) # c je kategorija po kojoj radimo pregled po boji, s je scale točkice
plt.show()
# možeš i ovako pirkazati plot samo se po defaultu neće ispisati xlabel, ylabel, naslov...
# plt.scatter(mapped_data["Fuel Consumption City (L/100km)"], mapped_data["CO2 Emissions (g/km)"], c=mapped_data["Fuel Type"])
# plt.show()

def get_by_fuel_type(type):
    return (data[data["Fuel Type"] == type]["Fuel Consumption City (L/100km)"], data[data["Fuel Type"] == type]["CO2 Emissions (g/km)"])

# ovo su dostupne colormape u pythonu
cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
            'gist_ncar'])]


#basic colors
# 'b'- blue.
# 'c' - cyan.
# 'g' - green.
# 'k' - black.
# 'm' - magenta.
# 'r' - red.
# 'w' - white.
# 'y' - yellow.


#OVAKO SE RADI BOXPLOT (PRIMJERI)
# data["Fuel Consumption Hwy (L/100km)"].plot(kind="box")
# plt.boxplot(x=data["Fuel Consumption Hwy (L/100km)"], vert=False)


#c) Pomoću kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip
# goriva. Primje ́cujete li grubu mjernu pogrešku u podacima?

data.boxplot(column="Fuel Consumption Hwy (L/100km)", by="Fuel Type")
plt.show()

# AGREGACIJSKE FUNKCIJE NA groupby()...
# mean(), count(), max(), min(), agg() - može kombinirati više agg funkcija (npr. .agg(["count", "min", "max"]))
# sum() - suma za svaku grupu, std(), var(), describe(), first(), last(), nth()



#d) Pomo ́cu stupˇcastog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu groupby

# možemo na 2 načina isto :)
grouped_by_fuel_types = data.groupby("Fuel Type")["Model"].count()
grouped_by_fuel_types.plot.bar()
plt.show()

# ili ovako...
# ovo je savršen primjer gdje možemo upotrijebiti .index & .values
# plt.bar(grouped_by_fuel_types.index, grouped_by_fuel_types.values)
# plt.show()



# e) Pomo ́cu stupˇcastog grafa prikažite na istoj slici prosjeˇcnu C02 emisiju vozila s obzirom na broj cilindara.
co2_emission_by_cylinder_count = data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean()

figure = plt.figure("statistics", figsize=(10, 4))
figure.add_subplot(1, 2, 1)
plt.bar(grouped_by_fuel_types.index, grouped_by_fuel_types.values)
plt.xlabel("fuel types")
plt.ylabel("number of cars")
figure.add_subplot(1, 2, 2)
plt.bar(co2_emission_by_cylinder_count.index, co2_emission_by_cylinder_count.values)
plt.xlabel("cylinder count")
plt.ylabel("emissions by cylinder count")
plt.show()