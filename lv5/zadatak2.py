import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


labels = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # ravel stvara iz numpy višedimenzionalnog arraya jednodimenzionalan
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # enumerate izbacuje van indexe i vrijednosti...
    for idx, cl in enumerate(np.unique(y)): 
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=labels[cl])

# Load data
df = pd.read_csv("lv5/penguins.csv")
print(df.info())
# Check for missing values
print(df.isnull().sum())
# Drop 'sex' column due to missing values
df = df.drop(columns=['sex'])
# Drop rows with missing values
df.dropna(axis=0, inplace=True)
# Reset indexes
df = df.reset_index(drop=True)
# Encoding categorical variable 'species'
species_map = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
# df['species'].replace(species_map, inplace=True) #ova komanda baca warning pa koristimo ovu dolje
df["species"] = df["species"].map(species_map)
# Input variables: bill length, flipper length
input_variables = ['bill_length_mm', 'flipper_length_mm']
# Output variables: species
output_variables = ['species']
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df[input_variables].to_numpy(), df[output_variables].to_numpy(), test_size=0.2, random_state=123)

if len(y_test.shape) > 1:
    y_test = y_test.ravel()
    y_train = y_train.ravel()

# Zadatak 5.5.2 Skripta zadatak_2.py uˇcitava podatkovni skup Palmer Penguins [1]. Ovaj
# podatkovni skup sadrži mjerenja provedena na tri razliˇcite vrste pingvina (’Adelie’, ’Chins-
# trap’, ’Gentoo’) na tri razliˇcita otoka u podruˇcju Palmer Station, Antarktika. Vrsta pingvina
# odabrana je kao izlazna veliˇcina i pri tome su klase oznaˇcene s cjelobrojnim vrijednostima
# 0, 1 i 2. Ulazne veliˇcine su duljina kljuna (’bill_length_mm’) i duljina peraje u mm (’flip-
# per_length_mm’). Za vizualizaciju podatkovnih primjera i granice odluke u skripti je dostupna
# funkcija plot_decision_region.

# a) Pomo ́cu stupˇcastog dijagrama prikažite koliko primjera postoji za svaku klasu (vrstu
# pingvina) u skupu podataka za uˇcenje i skupu podataka za testiranje. Koristite numpy
# funkciju unique.
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)
plt.bar(unique_train, counts_train, color="blue")
plt.bar(unique_test, counts_test, color="red")
plt.legend(["train subset", "test subset"])
plt.show()


# b) Izgradite model logistiˇcke regresije pomo ́cu scikit-learn biblioteke na temelju skupa poda-
# taka za uˇcenje.
log_reg = LogisticRegression(max_iter=10000) # max iter moramo postaviti jer inaće izbacuje warning (stavimo na veliki broj)
log_reg.fit(X_train, y_train)
# zapamti na .fit moraš staviti vrijednost čiji je shape (n, ) ne (n, 1) ili pogotovo (n, m) ako imaš takve numpy arr stavi na njih .ravel()


# c) Prona  ̄dite u atributima izgra  ̄denog modela parametre modela. Koja je razlika u odnosu na
# binarni klasifikacijski problem iz prvog zadatka?
print("Coefficients:", log_reg.coef_)
print("Intercept:", log_reg.intercept_)
#razlika je da skup s koeficijentima sad sadrži više elemenata


# d) Pozovite funkciju plot_decision_region pri ˇcemu joj predajte podatke za uˇcenje i
# izgra  ̄deni model logistiˇcke regresije. Kako komentirate dobivene rezultate?
plot_decision_regions(X_train, y_train, classifier=log_reg)
plt.xlabel('Bill Length (mm)')
plt.ylabel('Flipper Length (mm)')
plt.title('Decision Regions for Training Data')
plt.legend(loc='upper left')
plt.show()


# e) Provedite klasifikaciju skupa podataka za testiranje pomo ́cu izgra  ̄denog modela logistiˇcke
# regresije. Izraˇcunajte i prikažite matricu zabune na testnim podacima. Izraˇcunajte toˇcnost.
# Pomo ́cu classification_report funkcije izraˇcunajte vrijednost ˇcetiri glavne metrikea skupu podataka za testiranje.
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)


# f) Dodajte u model još ulaznih veliˇcina. Što se doga  ̄da s rezultatima klasifikacije na skupu
# podataka za testiranje?

input_variables = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
# Output variables: species
output_variables = ['species']
# Train/test split
new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(df[input_variables].to_numpy(), df[output_variables].to_numpy(), test_size=0.2, random_state=123)

if len(y_test.shape) > 1:
    new_y_test = new_y_test.ravel()
    new_y_train = new_y_train.ravel()


log_reg = LogisticRegression(max_iter=10000) # max iter moramo postaviti jer inaće izbacuje warning (stavimo na veliki broj)
log_reg.fit(new_X_train, new_y_train)

new_y_pred = log_reg.predict(new_X_test)
new_acciracy = accuracy_score(new_y_test, new_y_pred)
new_conf_matrix = confusion_matrix(new_y_test, new_y_pred)
new_classification_rep = classification_report(new_y_test, new_y_pred)

print("*******************************************************************")
print("Old acciracy:", accuracy, "new accuracy", new_acciracy)
print("Old confusion matrix:\n", conf_matrix, "\nNew confusion matrix:\n", new_conf_matrix)
print("Old classification report:\n", classification_rep, "\nNew classification report\n", new_classification_rep)
print("*******************************************************************")
# izgledad da se dodavanjem novih numeričkih ulaznih vrijednosti metrički rezultat popravio