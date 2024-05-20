import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


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

# generiranje podatkovnih primjera
X = generate_data(500, 1)

# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()

# Zadatak 7.5.1 Skripta zadatak_1.py sadrži funkciju generate_data koja služi za generiranje
# umjetnih podatkovnih primjera kako bi se demonstriralo grupiranje. Funkcija prima cijeli broj
# koji definira željeni broju uzoraka u skupu i cijeli broj od 1 do 5 koji definira na koji naˇcin  ́ce
# se generirati podaci, a vra ́ca generirani skup podataka u obliku numpy polja pri ˇcemu su prvi i
# drugi stupac vrijednosti prve odnosno druge ulazne veliˇcine za svaki podatak. Skripta generira
# 500 podatkovnih primjera i prikazuje ih u obliku dijagrama raspršenja.

# 1. Pokrenite skriptu. Prepoznajete li koliko ima grupa u generiranim podacima? Mijenjajte
# naˇcin generiranja podataka.

# mislim da u prvom generiranju ima 3 grupe
# u drugom isto 3
# u trećem 4
# u četvrtom bi reko 2
# u petom 2

# 2. Primijenite metodu K srednjih vrijednosti te ponovo prikažite primjere, ali svaki primjer
# obojite ovisno o njegovoj pripadnosti pojedinoj grupi. Nekoliko puta pokrenite programski
# kod. Mijenjate broj K. Što primje ́cujete?

# • n_clusters – broj grupa (cjelobrojna vrijednost, default=8)
# • init – naˇcini inicijalizacije centara (’k-means++’, ’random’, default=’k-means++’)
# • n_init – koliko puta će se algoritam izvršiti s različitim početnim centrima, a konačan rezultat su centri koji su postigli najmanju vrijednost kriterijske funkcije auto’ ili cjelobrojna vrijednost, default=10
# • max_iter – maksimalni broj iteracija algoritma (cjelobrojna vrijednost, default=300)

km = KMeans(n_clusters=3, init='random', n_init=5, random_state=0)
km.fit(X)
labels = km.predict(X)  

plt.figure()
plt.scatter(X[:,0], X[:,1], c=labels)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('K-means grouping prediction')
plt.show()


# 3. Mijenjajte način definiranja umjetnih primjera te promatrajte rezultate grupiranja podataka
# (koristite optimalni broj grupa). Kako komentirate dobivene rezultate?

# prvi, drugi i treći su relativno dobro predviđeni 
# dok u 4. i 5. dolazi do problema (izgleda da tu se podaci dijele negdje u sredini)
# samo se nadam da sam dobro podijelio grupe, da je u 4. i 5. primjeru točno dodjeljeno 2 grupe