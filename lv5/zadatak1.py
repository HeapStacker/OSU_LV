import numpy as np
import matplotlib
from sklearn import metrics
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ARGUMENTI DOLJE NAVEDENE FUNKCIJE
# n_samples=200: Stvara 200 točkica.
# n_features=2: Svaka točkica ima dvije osobine.
# n_redundant=0: Nijedna osobina nije višak.
# n_informative=2: Obje osobine su važne za razlikovanje grupa.
# random_state=213: Osigurava da su točkice svaki put iste.
# n_clusters_per_class=1: Svaka grupa ima jedan skup točkica.
# class_sep=1: Udaljenost između točkica različitih grupa je 1.
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2, random_state=213, n_clusters_per_class=1, class_sep=1) 
# izbacuje van numpy arr

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5) 
# ulaz je numpy array pa je i izlaz

# Zadatak 5.5.1 Skripta zadatak_1.py generira umjetni binarni klasifikacijski problem s dvije
# ulazne veliˇcine. Podaci su podijeljeni na skup za uˇcenje i skup za testiranje modela.

# a) Prikažite podatke za uˇcenje u x1 −x2 ravnini matplotlib biblioteke pri ˇcemu podatke obojite
# s obzirom na klasu. Prikažite i podatke iz skupa za testiranje, ali za njih koristite drugi
# marker (npr. ’x’). Koristite funkciju scatter koja osim podataka prima i parametre c i
# cmap kojima je mogu ́ce definirati boju svake klase.
plt.scatter(X_train[:,0], X_train[:,1], cmap='coolwarm', c= y_train, marker="o")
plt.scatter(X_test[:,0], X_test[:,1], cmap='coolwarm', c=y_test, marker="x")
plt.xlabel('x1')
plt.ylabel('x2')
plt.title("o - trenirani podaci, x - testni podaci,\nplavo - prva grupa, crveno - druga grupa")
plt.show()



# b) Izgradite model logistiˇcke regresije pomo ́cu scikit-learn biblioteke na temelju skupa poda-
# taka za uˇcenje.

regr = lm.LogisticRegression()
regr.fit(X_train, y_train)



# c) Prona  ̄dite u atributima izgra  ̄denog modela parametre modela. Prikažite granicu odluke
# nauˇcenog modela u ravnini x1 −x2 zajedno s podacima za uˇcenje. Napomena: granica
# odluke u ravnini x1 −x2 definirana je kao krivulja: θ0 +θ1x1 +θ2x2 =0.

coef = regr.coef_[0]
intercept = regr.intercept_
print(coef) # just for clarity
print(intercept) # just for clarity

def decision_boundary(regresion, x1):
    coef = regresion.coef_[0]
    intercept = regresion.intercept_
    return (-coef[0]*x1 - intercept) / coef[1]

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm')
plt.plot(X_train[:, 0], decision_boundary(regr, X_train[:, 0]))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Logistic Regression Decision Boundary')
plt.show()


# d) Provedite klasifikaciju skupa podataka za testiranje pomo ́cu izgra  ̄denog modela logistiˇcke
# regresije. Izraˇcunajte i prikažite matricu zabune na testnim podacima. Izraˇcunate toˇcnost,
# preciznost i odziv na skupu podataka za testiranje.

def calculate_acciracy_and_precision(actual, predicted):
    print("accuracy:    ", metrics.accuracy_score(actual, predicted))
    print("precision:   ", metrics.precision_score(actual, predicted))

y_test_pred = regr.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix: ", cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

calculate_acciracy_and_precision(y_test, y_test_pred)

# e) Prikažite skup za testiranje u ravnini x1 −x2. Zelenom bojom oznaˇcite dobro klasificirane
# primjere dok pogrešno klasificirane primjere oznaˇcite crnom bojom.
correct = y_test == y_test_pred
incorrect = y_test != y_test_pred # ili  p.logical_not(correct) ili ~corrent

# gledaj gornju liniju koda ovako
a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
b = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 0])
# a == b je array [1, 0, 1, 1, 0, 1, 1, 1, 0, 0]
c = np.array([5, 6, 7, 8, 1, 2, 3, 4, 9, 0])
d = c[np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 0]).astype(bool)] 
# da ne stavimo .astype(bool) vraćali bi indexe arraya c
# d je isto ko da napišeš c[a == b] jer a == b vraća bool array

# Prikaz ispravno klasificiranih uzoraka zelenom bojom
plt.scatter(X_test[correct][:, 0], X_test[correct][:, 1], color='green')
plt.scatter(X_test[incorrect][:, 0], X_test[incorrect][:, 1], color='black')
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend(['Correctly classified', 'Incorrectly classified'])
plt.show()