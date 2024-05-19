import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

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

# Zadatak 6.5.2 Pomo ́cu unakrsne validacije odredite optimalnu vrijednost hiperparametra K
# algoritma KNN za podatke iz Zadatka 1.

knn = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 100) }
knn_grid_search = GridSearchCV(knn, param_grid, cv=5)
knn_grid_search.fit(X_train_scaled, y_train)
print("Best param K num =", knn_grid_search.best_params_)
print("KNN best score =", knn_grid_search.best_score_)