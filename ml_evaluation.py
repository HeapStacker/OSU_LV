import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def print_model_scores(actual, predicted, selector):
    # točnost
    if 0 in selector: 
        print("Accuracy:    ", metrics.accuracy_score(actual, predicted))   
    # preciznost
    if 1 in selector:
        print("Precision:   ", metrics.precision_score(actual, predicted))          
    if 2 in selector:
        print("Sensitivity: ", metrics.recall_score(actual, predicted))                  
    if 3 in selector:
        print("Specificity: ", metrics.recall_score(actual, predicted, pos_label=0))     
    if 4 in selector:
        print("F-score:     ", metrics.f1_score(actual, predicted))        
    if 5 in selector:
        print("Confusion Matrix: ", metrics.confusion_matrix(actual, predicted))
    if 6 in selector:
        display = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(actual, predicted))
        display.plot()
        plt.show()

def print_model_errors(y_real, y_prediction, selector):
    mse = metrics.mean_squared_error(y_real, y_prediction)
    if 0 in selector:
        print(f"MSE - Mean squared error: {mse:.5f}")
    if 1 in selector:
        print(f"RMSE - Root Mean squared error: {np.sqrt(mse):.5f}")
    if 2 in selector:
        print(f"MAE - Mean absolute error: {metrics.mean_absolute_error(y_real, y_prediction):.5f}")
    if 3 in selector:
        print(f"MAPE - Mean absolute percentage error: {metrics.mean_absolute_percentage_error(y_real, y_prediction) * 100:.5f}%")
    if 4 in selector:
        print(f"MAXERR - Max error: {metrics.max_error(y_real, y_prediction)}")
    if 5 in selector:
        print(f"R^2 - Coefficient of determination: {metrics.r2_score(y_real, y_prediction):.5f}") 
        # vrijednost između 0 i 1 (1 znači da je dobro)          