import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# a) Prikažite podatke za učenje i testiranje u x1 - x2 ravnini koristeći biblioteku matplotlib:

# Generiranje umjetnih podataka
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2, random_state=213, n_clusters_per_class=1, class_sep=1)
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
# Priprema za prikazivanje podataka
plt.figure("Podaci za učenje/testiranje u x1 - x2 ravnini", figsize=(10, 5))
# Prikaz podataka za učenje
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', cmap='coolwarm', label='Train Data')
# Prikaz podataka za testiranje
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', cmap='Spectral', label='Test Data')
# Dodavanje legende i oznaka osi
plt.legend()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Binary Classification Problem')
plt.show()


#b) Izgradite model logisticke regresije pomo ˇ cu scikit-learn biblioteke na temelju skupa poda- ´taka za ucenje.

# Inicijalizacija i treniranje modela
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


#c) Pronađite parametre modela:

# Koeficijenti i interceptr modela
theta0 = log_reg.intercept_
theta1, theta2 = log_reg.coef_[0]
# Prikaz granice odluke
x1_min, x1_max = X_train[:, 0].min(), X_train[:, 0].max()
x2_min, x2_max = X_train[:, 1].min(), X_train[:, 1].max()
x1_range = np.linspace(x1_min, x1_max, 100)
x2_boundary = -(theta0 + theta1*x1_range) / theta2
# Prikaži podatke za učenje
plt.figure("Parametri modela")
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', cmap='coolwarm', label='Train Data')
# Prikaz granice odluke
plt.plot(x1_range, x2_boundary, color='black', label='Decision Boundary')
# Dodavanje legendi i oznaka osi
plt.legend()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Decision Boundary')
plt.show()


#d) Klasificirajte skup podataka za testiranje i izračunajte evaluacijske mjere:

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
# Predikcija na testnom skupu
y_pred = log_reg.predict(X_test)
# Izračun evaluacijskih mjera
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
# Ispis evaluacijskih mjera
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall) #odziv


# e) Prikažite skup za testiranje u ravnini x1 - x2 s označenim točno i netočno klasificiranim primjerima:

# Plot correctly classified points (green)
plt.scatter(X_test[y_test == y_pred, 0], X_test[y_test == y_pred, 1], c='green', marker='o', label='Correctly Classified')
# Plot incorrectly classified points (black)
plt.scatter(X_test[y_test != y_pred, 0], X_test[y_test != y_pred, 1], c='black', marker='x', label='Incorrectly Classified')
# Add labels and legend
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Test Data with Correctly/Incorrectly Classified Examples')
plt.legend()
plt.show()