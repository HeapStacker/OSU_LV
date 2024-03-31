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
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor='w',
                    label=labels[cl])

# Load data
df = pd.read_csv("lv5/penguins.csv")

# Check for missing values
print(df.isnull().sum())

# Drop 'sex' column due to missing values
df = df.drop(columns=['sex'])

# Drop rows with missing values
df.dropna(axis=0, inplace=True)

# Encoding categorical variable 'species'
df['species'].replace({'Adelie': 0,
                       'Chinstrap': 1,
                       'Gentoo': 2}, inplace=True)

print(df.info())

# Output variable: species
output_variable = ['species']

# Input variables: bill length, flipper length
input_variables = ['bill_length_mm', 'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# a) Count samples for each class in training and test sets
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)
print("Train Data Class Counts:", dict(zip(labels.values(), counts_train)))
print("Test Data Class Counts:", dict(zip(labels.values(), counts_test)))

# b) Build Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train.ravel())

# c) Find model parameters
coefficients = log_reg.coef_
intercept = log_reg.intercept_
print("Coefficients:", coefficients)
print("Intercept:", intercept)

# d) Plot decision regions
plot_decision_regions(X_train, y_train.ravel(), classifier=log_reg)
plt.xlabel('Bill Length (mm)')
plt.ylabel('Flipper Length (mm)')
plt.title('Decision Regions for Training Data')
plt.legend(loc='upper left')
plt.show()

# e) Classification on test data
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)

# f) Add more input variables to the model and observe the changes in classification results
# Not implemented here, but you can add additional input variables to X_train and X_test before fitting the model.