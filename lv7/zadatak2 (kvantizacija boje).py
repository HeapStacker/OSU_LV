import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.model_selection import GridSearchCV

# 5. Primijenite postupak i na ostale dostupne slike.
# primjenjujem postupak na svim slikama odjednom...
img_paths = [
    "lv7/imgs/test_1.jpg",
    "lv7/imgs/test_2.jpg",
    "lv7/imgs/test_3.jpg",
    "lv7/imgs/test_4.jpg",
    "lv7/imgs/test_5.jpg",
    "lv7/imgs/test_6.jpg"
]

# micanje A kanala na slici tako da bude samo RGB
img_list = []
for path in img_paths:
    img = Image.open(path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img_list.append(np.array(img))

img_scaled_list = []
for img in img_list:
    img_scaled_list.append(img.astype(np.float64) / 255)

img_reshaped_list = []
for img in img_scaled_list:
    w, h, d = img.shape
    img_reshaped_list.append(np.reshape(img, (w * h, d)))



# 1. Otvorite skriptu zadatak_2.py. Ova skripta uˇcitava originalnu RGB sliku test_1.jpg
# te ju transformira u podatkovni skup koji dimenzijama odgovara izrazu (7.2) pri čemu je n
# broj elemenata slike, a m je jednak 3. Koliko je razliˇcitih boja prisutno u ovoj slici?
print(f'Broj boja u originalnoj prvoj slici: {len(np.unique(img_reshaped_list[0], axis=0))}')



def k_means_quantization(reshaped_img, img_true_shape, k_param = 5, create_cluster_images = False):
    km = KMeans(n_clusters=k_param, init='random', n_init=5, random_state=0)
    km.fit(reshaped_img)
    labels = km.predict(reshaped_img)

    # ovaj dio je tu samo ako hoćeš napravit od slike više slika za njezin svaki klaster...
    cluster_imges = []
    if create_cluster_images:
        for i in range(k_param):
            cluster_img_i = np.zeros_like(reshaped_img) # zeros_like prezervira shape od reshaped_img (za razliku od zeros)
            for idx, label in enumerate(labels):
                if label == i:
                    cluster_img_i[idx] = km.cluster_centers_[label]
                else:
                    cluster_img_i[idx] = [1, 1, 1]  # Boja koja nije niti jedan od klaster centara (bijela)

            # moramo reshapeati i cluster slike isto kao i kvantiziranu
            cluster_imges.append(np.reshape(cluster_img_i, img_true_shape))

    quantized = np.reshape(km.cluster_centers_[labels], img_true_shape)
    return quantized, cluster_imges



# 2. Primijenite algoritam K srednjih vrijednosti koji ́ce pronaći grupe u RGB vrijednostima elemenata originalne slike.
# izračunato dolje :)
k_elbow_method = [3, 2, 2, 2, 4, 2]
image_quantized_list = []

for i in range(len(img_reshaped_list)):
   quantized, cluster_imgs = k_means_quantization(img_reshaped_list[i], img_list[i].shape, k_elbow_method[i], i == 0)
   image_quantized_list.append((quantized, cluster_imgs))

imgs_tuple_list = list(zip(img_list, image_quantized_list))



# 4. Usporedite dobivenu sliku s originalnom. Mijenjate broj grupa K. Komentirajte dobivene rezultate.
def plot_images(original, quantized):
    plt.figure("Image quantization", figsize=(11,3))
    plt.subplot(1, 2, 1)
    plt.title("Originalna slika")
    plt.imshow(original)
    plt.subplot(1, 2, 2)
    plt.title("Rezultatna slika nakon kvantizacije")
    plt.imshow(quantized)
    plt.tight_layout()
    plt.show()
# čim je K veći slika će biti bolje kvalitete

def plot_cluster_images(cluster_imges):
    for i in range(len(cluster_imges)):
        plt.figure(f"Cluster image {i + 1}")
        plt.title(f"{i + 1}")
        plt.imshow(cluster_imges[i])
        plt.show()


for img in imgs_tuple_list:
    plot_images(img[0], img[1][0])
    if len(img[1][1]) > 0:
        plot_cluster_images(img[1][1])


# 6. Grafički prikažite ovisnost J o broju grupa K. Koristite atribut inertia objekta klase
# KMeans. Možete li uoˇciti lakat koji upu ́cuje na optimalni broj grupa

# GRID SEARCH NE DAJE DOBRE REZULTATE PA...
# PRIMJER LAKAT METODE, TRAŽENJE NAJBOLJEG K

def plot_elbow_method(reshaped_img, image_id):
    inertia_values = []
    k_values = range(1, 10)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(reshaped_img)
        inertia_values.append(kmeans.inertia_)

    plt.figure()
    plt.plot(k_values, inertia_values, marker='o')
    plt.xlabel('Broj grupa (K)')
    plt.ylabel('J')
    plt.title(f'Ovisnost J o broju grupa K za sliku {image_id + 1}')
    plt.grid(True)
    plt.show()