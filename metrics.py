import pandas as pd
from math import sqrt
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef, jaccard_score, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, recall_score
from scipy.stats import pointbiserialr

def confusion_matrix_dataframe(y, y_predicted, columns, index):
    cm = confusion_matrix(y, y_predicted)
    matrix = pd.DataFrame(cm, columns = columns, index = index)
    return matrix

def specificity(y, y_predicted):
    tn, fp, tp, fn = confusion_matrix(y, y_predicted).ravel()
    return tn / (tn + fp)

def binary_classification_summary(y, y_predicted):
    acc = accuracy_score(y, y_predicted)
    sen = recall_score(y, y_predicted)
    spe = specificity(y, y_predicted)
    mcc = matthews_corrcoef(y, y_predicted)
    auc = roc_auc_score(y, y_predicted)
    binary_classification_summary = pd.DataFrame([acc, sen, spe, mcc, auc], 
                                                 index = ["Accuracy", "Sensitivity", "Specificity", "Matthews Corr. Coef.", "AUROC"], 
                                                 columns = ["Scores"])
    return binary_classification_summary