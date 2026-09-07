import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def regression_metrics(y_true,y_pred):
    return {"MAE":mean_absolute_error(y_true,y_pred),"RMSE":np.sqrt(mean_squared_error(y_true,y_pred)),"R2":r2_score(y_true,y_pred)}

def classification_metrics(y_true,y_pred):
    return {"Accuracy":accuracy_score(y_true,y_pred),"Precision":precision_score(y_true,y_pred,zero_division=0),"Recall":recall_score(y_true,y_pred,zero_division=0),"F1":f1_score(y_true,y_pred,zero_division=0),"Confusion matrix":confusion_matrix(y_true,y_pred)}
