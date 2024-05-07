import numpy as np
import matplotlib.pyplot as plt
import statistics

#2. zadatak

# data = np.genfromtxt('data.csv', delimiter=',')

# # Podijeli podatke na spol, visinu i masu
# spol = data[1:, 0]
# visina = data[1:, 1]
# masa = data[1:, 2]

def load_data(file_path, rows_to_skip = 0, last_row = 0, data_type = float):
    # data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    if last_row:
        data = np.loadtxt(file_path, delimiter=",", dtype=data_type, skiprows=rows_to_skip, max_rows=last_row)
    else:
        data = np.loadtxt(file_path, delimiter=",", dtype=data_type, skiprows=rows_to_skip)
    return data

data = load_data("lv2/data.csv", 1)
print(data.ndim)
spol = data[:, 0].astype(int)
visina = data[:, 1]
masa = data[:, 2]

def plot_scatter(x, y, app_title, title, xlabel, ylabel, color="red"):
    plt.figure(app_title)
    plt.scatter(x, y, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

print(f"Izmjereno je mjerenje na {len(spol)} osoba.")
plot_scatter(visina, masa, "WH_1", "Odnos visine i mase", "visina (cm)", "Masa (kg)", "blue")

#isto ko i u prethodnom ali za svaku 50 osobu
visina_50_ = []
for i in range(0, len(visina)):
    if i % 50 == 0:
        visina_50_.append(visina[i])
masa_50_ = []
for i in range(0, len(masa)):
    if i % 50 == 0:
        masa_50_.append(masa[i])

plot_scatter(visina_50_, masa_50_, "WH_2", "Odnos visine i mase svake 50 osobe", "visina (cm)", "Masa (kg)")
print(f"min visina: {min(visina)}")
print(f"max visina: {max(visina)}")
print(f"srednja visina: {statistics.mean(visina)}")
male_height_ = []
female_height_ = []
for i in range(0, len(spol)):
    if spol[i] == 1:
        male_height_.append(visina[i])
    else:
        female_height_.append(visina[i])

def height_stats(title, heights):
    print(title)
    print(f"min visina: {min(heights)}")
    print(f"max visina: {max(heights)}")
    print(f"srednja visina: {statistics.mean(heights)}\n")    

height_stats("Za muškarce", male_height_)
height_stats("Za žene", female_height_)

plt.show()

# Izdvajanje muškaraca
# ind_muskarci = (data[:, 0] == 1) #vraca true/false vrijednosti prvog stupca koji su == 1 (array boolova)
# print(ind_muskarci.astype(int))
# visina_muskarci = data[ind_muskarci, 1] # vraca data članove 2. stupca koji su muskarci5
# print(visina_muskarci[0:5])