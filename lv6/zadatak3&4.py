import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

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

data = pd.read_csv("lv6/Social_Network_Ads.csv")

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform((X_test))

# Zadatak 6.5.3 Na podatke iz Zadatka 1 primijenite SVM model koji koristi RBF kernel funkciju
# te prikažite dobivenu granicu odluke. Mijenjajte vrijednost hiperparametra C i γ. Kako promjena
# ovih hiperparametara utjeˇce na granicu odluke te pogrešku na skupu podataka za testiranje?
# Mijenjajte tip kernela koji se koristi. Što primjećujete

# SUPORT VECTOR MACHINE zadatak

c = 13 # najbolje po grid search-u
gamma = 1
kern = "rbf"

def testing_svm_mode(kernel):
    svm_model= svm.SVC(C=c, gamma=gamma, kernel=kernel)
    svm_model.fit(X_train_scaled, y_train)

    y_train_svm = svm_model.predict(X_train_scaled)
    y_test_svm = svm_model.predict(X_test_scaled)

    print("SVM: ")
    print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_svm))))
    print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_svm))))

    # provjera modela na skupu podataka za testiranje (na neviđenim podacima (u zadatku traženo))
    plot_decision_regions(X_test_scaled, y_test, classifier=svm_model)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend(loc='upper left')
    plt.title(f"Tocnost: {accuracy_score(y_train, y_train_svm):0.3f}, C={c}, gamma={gamma}, kernel={kernel}")
    plt.tight_layout()
    plt.show()

testing_svm_mode("linear") # linear je lošiji od rbf-a u našem slučaju
testing_svm_mode("rbf")

# 4. zadatak najbolji parametri pomoću unakrsne validacije
svm_model= svm.SVC(C=c, gamma=gamma, kernel="rbf")
svm_model.fit(X_train_scaled, y_train)

# unakrsna validacija...
param_grid = {"C": np.arange(0, 15), "gamma": [0.01, 0.1, 1]}
knn_grid_search = GridSearchCV(svm_model, param_grid, cv=5)
knn_grid_search.fit(X_train_scaled, y_train)
print("Best params are =>", knn_grid_search.best_params_)
print("SVM (rbf kernel) best score =", knn_grid_search.best_score_)