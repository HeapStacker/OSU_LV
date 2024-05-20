import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.image as Image  

def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)

    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)

    # 2 grupe
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)

    else:
        X = []

    return X  

X = generate_data(500, 1)

plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()


km = KMeans(n_clusters=3, init='random', n_init=5, random_state=0)
km.fit(X)
labels = km.predict(X)  

plt.figure()
plt.scatter(X[:,0],X[:,1], c=labels)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()

img = Image.imread("lv7/imgs/test_1.jpg")  
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

img = img.astype(np.float64) / 255  
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))  
img_array_aprox = img_array.copy()  

print(f'Broj boja u originalnoj slici: {len(np.unique(img_array_aprox, axis=0))}')
km = KMeans(n_clusters=5, init='random', n_init=5, random_state=0)
km.fit(img_array_aprox)
labels = km.predict(img_array_aprox)  
img_array_aprox = km.cluster_centers_[km.labels_]  
img_aprox = np.reshape(img_array_aprox, (w, h, d))

print(f'Broj boja u novoj slici: {len(np.unique(img_array_aprox, axis=0))}')
plt.figure()
plt.title("Rezultatna slika nakon kvantizacije")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

inertia_values = []
k_values = range(1, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(img_array)
    inertia_values.append(kmeans.inertia_)

plt.figure()
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Broj grupa (K)')
plt.ylabel('J')
plt.title('Ovisnost J o broju grupa K')
plt.grid(True)
plt.show()

labels_unique = np.unique(labels)
for i in range(len(labels_unique)):
    binary_image = labels==labels_unique[i]
    binary_image = np.reshape(binary_image, (h,w))
    plt.figure()
    plt.title(f"Binarna slika {i+1}. grupe boja")
    plt.imshow(binary_image)
    plt.tight_layout()
    plt.show()

img2 = Image.imread("lv7/imgs/test_2.jpg")  
plt.figure()
plt.title("Originalna slika")
plt.imshow(img2)
plt.tight_layout()
plt.show()

img2 = img2.astype(np.float64) / 255  
w,h,d = img2.shape
img2_array = np.reshape(img2, (w*h, d))  
img2_array_aprox = img2_array.copy()  
print(f'Broj boja u originalnoj slici: {len(np.unique(img2_array_aprox, axis=0))}') 
km = KMeans(n_clusters=5, init='random', n_init=5, random_state=0)
km.fit(img2_array_aprox)
labels = km.predict(img2_array_aprox)  
img2_array_aprox = km.cluster_centers_[km.labels_]  
img2_aprox = np.reshape(img2_array_aprox, (w, h, d))
print(f'Broj boja u novoj slici: {len(np.unique(img2_array_aprox, axis=0))}')
plt.figure()
plt.title("Rezultatna slika nakon kvantizacije")
plt.imshow(img2_aprox)
plt.tight_layout()
plt.show()


#ima još slika al na istu sforu sve... 