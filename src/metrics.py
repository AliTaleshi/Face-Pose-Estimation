import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

def calculate_regression_metrics(y_true, y_pred):
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MedAPE': np.median(np.abs((y_true - y_pred) / y_true)) * 100,
        'MedSPE': np.median(((y_true - y_pred) ** 2 / y_true) * 100),
        'EVS': explained_variance_score(y_true, y_pred)
    }
