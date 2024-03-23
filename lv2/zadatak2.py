import numpy as np
import matplotlib.pyplot as plt
import statistics

#2. zadatak

# data = np.genfromtxt('data.csv', delimiter=',')

# # Podijeli podatke na spol, visinu i masu
# spol = data[1:, 0]
# visina = data[1:, 1]
# masa = data[1:, 2]

def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    data = np.loadtxt("data.csv", delimiter=",", dtype="str")
    return data

file_path = 'data.csv'
data = load_data(file_path)
spol = data[:, 0].astype(int)
visina = data[:, 1]
masa = data[:, 2]

print(f"Izmjereno je mjerenje na {len(spol)} osoba.")
plt.figure()
plt.scatter(visina, masa, color="blue", label="odnos visine i mase")
plt.title('Odnos visine i mase')
plt.xlabel('Visina (cm)')
plt.ylabel('Masa (kg)')
#isto ko i u prethodnom ali za svaku prethodnu osobu
visina_50_ = []
for i in range(0, len(visina)):
    if i % 50 == 0:
        visina_50_.append(visina[i])
masa_50_ = []
for i in range(0, len(masa)):
    if i % 50 == 0:
        masa_50_.append(masa[i])
plt.figure()
plt.scatter(visina_50_, masa_50_, color="red", label="odnos visine i mase svake 50 osobe")
plt.title('Odnos visine i mase')
plt.xlabel('Visina (cm)')
plt.ylabel('Masa (kg)')
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

print("ZA MUŠKARCE...")
print(f"min visina: {min(male_height_)}")
print(f"max visina: {max(male_height_)}")
print(f"srednja visina: {statistics.mean(male_height_)}")
print("ZA ŽENE...")
print(f"min visina: {min(female_height_)}")
print(f"max visina: {max(female_height_)}")
print(f"srednja visina: {statistics.mean(female_height_)}\n")

# Izdvajanje muškaraca
ind_muskarci = (data[:, 0] == 1) #vraca polje muskaraca
visina_muskarci = data[ind_muskarci, 1] # vraca data članove 2. stupca koji su muskarci