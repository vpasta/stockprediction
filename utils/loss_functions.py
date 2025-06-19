import numpy as np

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def mse_loss_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / len(y_true)