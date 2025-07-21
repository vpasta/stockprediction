import numpy as np

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def mse_loss_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / len(y_true)

def mae_loss(y_pred, y_true):
    """
    Menghitung Mean Absolute Error (MAE).
    MAE mengukur rata-rata besarnya kesalahan dalam satu set prediksi,
    tanpa mempertimbangkan arahnya.
    """
    return np.mean(np.abs(y_pred - y_true))

def mape_loss(y_pred, y_true):
    """
    Menghitung Mean Absolute Percentage Error (MAPE).
    MAPE mengukur akurasi sebagai persentase dari kesalahan.
    Penting: Hindari pembagian nol jika y_true memiliki nilai 0.
    """
    non_zero_true = y_true[y_true != 0]
    pred_for_non_zero = y_pred[y_true != 0]

    if len(non_zero_true) == 0:
        return np.nan 
    
    return np.mean(np.abs((non_zero_true - pred_for_non_zero) / non_zero_true)) * 100

