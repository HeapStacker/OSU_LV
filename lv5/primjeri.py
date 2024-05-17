# Model logisticke regresije je jedan od osnovnih algoritama za klasi ˇ fikaciju. U slucaju binarne klasifikacije (kada je y(i) ∈ {0,1})

from sklearn.linear_model import LogisticRegression

# Instantiate LogisticRegression with the specified parameters
log_reg = LogisticRegression(penalty='l2',
                             dual=False,
                             tol=0.0001,
                             C=1.0,
                             fit_intercept=True,
                             intercept_scaling=1,
                             class_weight=None,
                             random_state=None,
                             solver='lbfgs',
                             max_iter=100,
                             multi_class='auto',
                             verbose=0,
                             warm_start=False,
                             n_jobs=None,
                             l1_ratio=None)


# Primjer 5.1 prikazuje isjecak koda koji inicijalizira model logisti ˇ cke regresije te se zatim procje- ˇ
# njuju parametri modela na temelju podataka za ucenje. Izgra ˇ deni model se onda koristi za predikciju ¯
# izlazne velicine na skupu podataka za testiranje.

# # inicijalizacija i ucenje modela logisticke regresije
# LogRegression_model = LogisticRegression()
# LogRegression_model.fit(X_train , y_train)
# # predikcija na skupu podataka za testiranje
# y_test_p = LogRegression_model.predict(X_test)


# U scikit-learn biblioteci dostupne su funkcije za evaluaciju klasifikacijskih modela. U primjeru
# 5.2 koristi se tocnost klasi ˇ fikacije i matrica zabune za evaluaciju izgradenog modela. Izra ¯ cunata ˇ
# matrica zabune prikazuje se u obliku slike. Pomocu funkcije ´ classification_report moguce je ´
# izracunati ˇ cetiri glavne metrike (to ˇ cnost, preciznost, odziv i F1 mjeru).
import numpy as np
import matplotlib . pyplot as plt
from sklearn . metrics import accuracy_score
from sklearn . metrics import confusion_matrix , ConfusionMatrixDisplay
# stvarna vrijednost izlazne velicine i predikcija
y_true = np . array ([1 , 1 , 1 , 0 , 1 , 0 , 1 , 0 , 1])
y_pred = np . array ([0 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 0])
# tocnost
print (" Tocnost : " , accuracy_score ( y_true , y_pred ) )
# matrica zabune
cm = confusion_matrix ( y_true , y_pred )
print (" Matrica zabune : " , cm )
disp = ConfusionMatrixDisplay ( confusion_matrix ( y_true , y_pred ) )
disp . plot ()
plt . show ()
# report
# print ( classification_report ( y_true , y_pred ) )
