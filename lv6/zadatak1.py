# Zadatak 6.5.1 Skripta zadatak_1.py uˇcitava Social_Network_Ads.csv skup podataka [2].
# Ovaj skup sadrži podatke o korisnicima koji jesu ili nisu napravili kupovinu za prikazani oglas.
# Podaci o korisnicima su spol, dob i procijenjena pla ́ca. Razmatra se binarni klasifikacijski
# problem gdje su dob i procijenjena pla ́ca ulazne veliˇcine, dok je kupovina (0 ili 1) izlazna
# veliˇcina. Za vizualizaciju podatkovnih primjera i granice odluke u skripti je dostupna funkcija
# plot_decision_region [1]. Podaci su podijeljeni na skup za uˇcenje i skup za testiranje modela
# u omjeru 80%-20% te su standardizirani. Izgra  ̄den je model logistiˇcke regresije te je izraˇcunata
# njegova toˇcnost na skupu podataka za uˇcenje i skupu podataka za testiranje. Potrebno je:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("lv6/Social_Network_Ads.csv")
print(data.info())
data.hist() # različito od data.plot.hist() di sve ide na 1 graf
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_scaled, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_scaled)
y_test_p = LogReg_model.predict(X_test_scaled)

print("Logisticka regresija: ")
print(f"Tocnost train: {accuracy_score(y_train, y_train_p):0.3f}")
print(f"Tocnost test: {accuracy_score(y_test, y_test_p):0.3f}")

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_scaled, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()


# 1. Izradite algoritam KNN na skupu podataka za uˇcenje (uz K=5). Izraˇcunajte toˇcnost
# klasifikacije na skupu podataka za uˇcenje i skupu podataka za testiranje. Usporedite
# dobivene rezultate s rezultatima logistiˇcke regresije. Što primje ́cujete vezano uz dobivenu
# granicu odluke KNN modela?
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(X_train_scaled, y_train)
knn_train_prediction = knn.predict(X_train_scaled)
knn_test_prediction = knn.predict(X_test_scaled)

print(f"KNN Točnost (train): {accuracy_score(y_train, knn_train_prediction):0.3f}")
print(f"KNN Točnost (test): {accuracy_score(y_test, knn_test_prediction):0.3f}")

# ako stavimo skalirane podatke u knn rezultat predikcije će također biti bolji kao i kod regresije
# također knn se ponaša bolje od regresijskog modela

plot_decision_regions(X_train_scaled, y_train, knn, 0.08)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
# za dobivenu granicu primjećujem da čim je manje susjeda, to je granica uzburkanija

# Koristimo X_train_scaled i y_train u funkciji plot_decision_regions jer želimo prikazati granicu odluke modela na skupu podataka koji je korišten za treniranje modela. To omogućava vizualizaciju kako model dijeli podatke i kako se ponaša s podacima na kojima je treniran. Međutim, također možemo koristiti testne podatke (X_test_scaled i y_test) za prikazivanje granice odluke na skupu podataka za testiranje, kako bismo vidjeli kako se model ponaša s neviđenim podacima.



# 2. Kako izgleda granica odluke kada je K =1 i kada je K = 100?

# ODGOVOR
# K = 100 je underfitting a K = 1 je overfitting