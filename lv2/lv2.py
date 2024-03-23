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
print(f"srednja visina: {statistics.mean(female_height_)}")

print()
# Izdvajanje muškaraca
ind_muskarci = (data[:, 0] == 1) #vraca polje muskaraca
visina_muskarci = data[ind_muskarci, 1] # vraca data članove 2. stupca koji su muskarci

#3. zadatak...
img = plt.imread("road.jpg")

#za ispis matrica
# for i in range(img.ndim):
#     print(f"Matrica {i + 1}:")
#     print(img[:, :, 1])
#     print()

import numpy as np
import matplotlib.pyplot as plt
img = plt.imread ("road.jpg")
img = img[:,:,0].copy()
print(img.shape)
print(img.dtype)
plt.figure()
plt.imshow(img, cmap = "gray")

lighten_factor = 100
lightened_image = np.clip(img.astype(int) + lighten_factor, 0, 255).astype(np.uint8)
plt.figure()
plt.imshow(lightened_image, cmap = "gray")

height, width = img.shape

start_col = width // 4
end_col = width // 2

second_quarter = img[:, start_col:end_col]

plt.figure()
plt.imshow(second_quarter, cmap = "gray")

rotated_image_array = np.rot90(img, k=-1)

plt.figure()
plt.imshow(rotated_image_array, cmap = "gray")

mirrored_image_array = np.fliplr(img)

plt.figure()
plt.imshow(mirrored_image_array, cmap = "gray")


#4. zadatak...

black_square = np.zeros((50, 50))
white_square = np.ones((50, 50)) * 255

top_left = black_square
top_right = white_square
bottom_left = white_square
bottom_right = black_square

top_row = np.hstack((top_left, top_right))
middle_row = np.hstack((bottom_left, bottom_right))
bottom_row = np.hstack((top_left, top_right))
final_image = np.vstack((top_row, middle_row, bottom_row))

plt.figure()
plt.imshow(final_image, cmap = "gray")

plt.show()
    