import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def calculate_technical_indicators(df):
    """
    Menghitung berbagai indikator teknikal dan menambahkannya ke DataFrame.
    DataFrame harus memiliki kolom 'Adj Close', 'High', 'Low', 'Close', 'Volume'.
    """
    df_copy = df.copy()

    # SMA (Simple Moving Average)
    df_copy['SMA_10'] = df_copy['Adj Close'].rolling(window=10).mean()
    df_copy['SMA_20'] = df_copy['Adj Close'].rolling(window=20).mean()

    # EMA (Exponential Moving Average)
    df_copy['EMA_10'] = df_copy['Adj Close'].ewm(span=10, adjust=False).mean()
    df_copy['EMA_20'] = df_copy['Adj Close'].ewm(span=20, adjust=False).mean()

    # RSI (Relative Strength Index)
    delta = df_copy['Adj Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0) 

    # Average gain dan average loss menggunakan Exponential Moving Average
    # com = span - 1 untuk mencocokkan definisi EMA di TA libraries
    avg_gain = gain.ewm(com=14 - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=14 - 1, adjust=False).mean()

    # Hitung Relative Strength (RS)
    # Tangani kasus ZeroDivisionError jika avg_loss adalah 0
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss) 
    df_copy['RSI'] = 100 - (100 / (1 + rs))

    # --- Opsional: Tambahkan indikator lain seperti MACD atau ATR jika diinginkan ---
    # MACD
    # exp1 = df_copy['Adj Close'].ewm(span=12, adjust=False).mean()
    # exp2 = df_copy['Adj Close'].ewm(span=26, adjust=False).mean()
    # df_copy['MACD'] = exp1 - exp2
    # df_copy['Signal_Line'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
    
    # ATR (Average True Range)
    # high_low = df_copy['High'] - df_copy['Low']
    # high_close = np.abs(df_copy['High'] - df_copy['Close'].shift())
    # low_close = np.abs(df_copy['Low'] - df_copy['Close'].shift())
    # tr = np.maximum(high_low, np.maximum(high_close, low_close))
    # df_copy['ATR'] = tr.rolling(window=14).mean() # Default 14 period

    # Tangani NaN yang muncul dari perhitungan indikator (misalnya, periode awal untuk rolling/ewm)
    df_copy.ffill(inplace=True)
    df_copy.bfill(inplace=True) # Isi NaN yang mungkin tersisa di awal (jika ada)

    return df_copy


def create_sequences(data_scaled, lookback_window, target_feature_index):
    """
    Mengubah data deret waktu yang diskalakan menjadi pasangan input (X) dan output (y) sequences.

    Args:
        data_scaled (np.ndarray): Data deret waktu yang sudah dinormalisasi (2D array, samples x features).
        lookback_window (int): Jumlah time steps yang akan digunakan sebagai input untuk prediksi.
        target_feature_index (int): Indeks kolom target yang akan diprediksi (misal Adj Close).

    Returns:
        tuple: (X, y) di mana X adalah input sequences dan y adalah target values.
    """
    X, y = [], []
    
    for i in range(len(data_scaled) - lookback_window):
        X.append(data_scaled[i:(i + lookback_window), :]) 
        y.append(data_scaled[i + lookback_window, target_feature_index]) 

    return np.array(X), np.array(y)