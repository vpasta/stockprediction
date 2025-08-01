import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df, lookback_window):
    """
    Mempersiapkan data untuk model GRU menggunakan harga 'Adj Close'.
    """
    # 1. Pilih hanya kolom 'Adj Close' dan pastikan itu adalah DataFrame
    # Menggunakan [['Adj Close']] memastikan kita mendapatkan DataFrame, bukan Series
    close_prices = df[['Adj Close']].copy()

    # Menghapus baris yang mungkin memiliki nilai NaN untuk kebersihan data
    close_prices.dropna(inplace=True)

    # Mengubah ke tipe data float untuk konsistensi
    close_prices = close_prices.astype('float32')

    # 2. Buat dan terapkan scaler pada data 'Adj Close'
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)

    # 3. Buat sequences (X) dan target (y)
    X, y = [], []
    for i in range(len(scaled_prices) - lookback_window):
        # X adalah urutan data dari i sampai i + lookback_window
        X.append(scaled_prices[i:(i + lookback_window), 0])
        # y adalah harga pada hari berikutnya setelah urutan X
        y.append(scaled_prices[i + lookback_window, 0])

    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y = np.array(y)

    return X, y, scaler


