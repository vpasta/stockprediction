from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import secrets
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import traceback

app = Flask(__name__)

app.secret_key = secrets.token_hex(16)

# --- Konfigurasi Database ---
DB_NAME = 'prediksi_tsla_db'
DB_USER = 'root'
DB_PASSWORD = ''
DB_HOST = 'localhost'
DB_PORT = '3306'

app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- Model Database untuk Data Saham ---
class StockData(db.Model):
    __tablename__ = 'stock_data'
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), nullable=False)
    date = db.Column(db.Date, nullable=False, index=True)
    open_price = db.Column(db.Float)
    high_price = db.Column(db.Float)
    low_price = db.Column(db.Float)
    close_price = db.Column(db.Float)
    adj_close_price = db.Column(db.Float)
    volume = db.Column(db.BigInteger)

    __table_args__ = (db.UniqueConstraint('ticker', 'date', name='_ticker_date_uc'),)

    def __repr__(self):
        return f'<StockData {self.ticker} - {self.date}>'

DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# --- Fungsi Aktivasi dan Turunannya ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# --- Fungsi Loss ---
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def mse_loss_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / len(y_true)

# --- Kelas GRU dari Nol ---
class GRU:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.Wz = np.random.randn(input_dim, hidden_dim) * 0.01
        self.Uz = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.bz = np.zeros((1, hidden_dim))

        self.Wr = np.random.randn(input_dim, hidden_dim) * 0.01
        self.Ur = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.br = np.zeros((1, hidden_dim))

        self.Wh = np.random.randn(input_dim, hidden_dim) * 0.01
        self.Uh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.bh = np.zeros((1, hidden_dim))

        self.Wo = np.random.randn(hidden_dim, output_dim) * 0.01
        self.bo = np.zeros((1, output_dim))

        self.cache = {}

    def forward(self, X):
        sequence_length = X.shape[0]
        
        h_prev = np.zeros((1, self.hidden_dim))

        hidden_states = []
        sequence_cache = []

        for t in range(sequence_length):
            x_t = X[t].reshape(1, self.input_dim)

            z_pre = np.dot(x_t, self.Wz) + np.dot(h_prev, self.Uz) + self.bz
            r_pre = np.dot(x_t, self.Wr) + np.dot(h_prev, self.Ur) + self.br
            
            z_t = sigmoid(z_pre)
            r_t = sigmoid(r_pre)
            
            h_tilde_pre = np.dot(x_t, self.Wh) + np.dot(r_t * h_prev, self.Uh) + self.bh
            h_tilde_t = tanh(h_tilde_pre)

            h_t = (1 - z_t) * h_prev + z_t * h_tilde_t

            sequence_cache.append({
                'x_t': x_t,
                'h_prev': h_prev.copy(), 
                'z_pre': z_pre, 'r_pre': r_pre, 'h_tilde_pre': h_tilde_pre,
                'z_t': z_t,
                'r_t': r_t,
                'h_tilde_t': h_tilde_t,
                'h_t': h_t
            })

            h_prev = h_t
            hidden_states.append(h_t)

        final_h_t = hidden_states[-1]
        output = np.dot(final_h_t, self.Wo) + self.bo

        self.cache['sequence_cache'] = sequence_cache
        self.cache['final_h_t'] = final_h_t
        self.cache['output'] = output
        return output

    def backward(self, d_output, learning_rate):
        sequence_cache = self.cache['sequence_cache']
        final_h_t = self.cache['final_h_t']

        dWz, dUz, dbz = np.zeros_like(self.Wz), np.zeros_like(self.Uz), np.zeros_like(self.bz)
        dWr, dUr, dbr = np.zeros_like(self.Wr), np.zeros_like(self.Ur), np.zeros_like(self.br)
        dWh, dUh, dbh = np.zeros_like(self.Wh), np.zeros_like(self.Uh), np.zeros_like(self.bh)
        dWo, dbo = np.zeros_like(self.Wo), np.zeros_like(self.bo)

        dWo += np.dot(final_h_t.T, d_output)
        dbo += d_output

        dh_next = np.dot(d_output, self.Wo.T)

        for t in reversed(range(len(sequence_cache))):
            cache_t = sequence_cache[t]
            x_t, h_prev, z_pre, r_pre, h_tilde_pre, z_t, r_t, h_tilde_t, h_t = \
                cache_t['x_t'], cache_t['h_prev'], cache_t['z_pre'], cache_t['r_pre'], cache_t['h_tilde_pre'], \
                cache_t['z_t'], cache_t['r_t'], cache_t['h_tilde_t'], cache_t['h_t']

            dh_t = dh_next

            dh_prev_term1 = dh_t * (1 - z_t)

            dz_t = dh_t * (h_tilde_t - h_prev)
            dz_pre = dz_t * sigmoid_derivative(z_pre)

            dh_tilde_t = dh_t * z_t
            dh_tilde_pre = dh_tilde_t * tanh_derivative(h_tilde_pre)

            dWh += np.dot(x_t.T, dh_tilde_pre)
            dbh += dh_tilde_pre
            
            dUh += np.dot((r_t * h_prev).T, dh_tilde_pre)
            
            dr_t_from_h_tilde = np.dot(dh_tilde_pre, self.Uh.T) * h_prev

            dr_t = dr_t_from_h_tilde
            dr_pre = dr_t * sigmoid_derivative(r_pre)

            dWr += np.dot(x_t.T, dr_pre)
            dbr += dr_pre
            dUr += np.dot(h_prev.T, dr_pre)

            dWz += np.dot(x_t.T, dz_pre)
            dbz += dz_pre
            dUz += np.dot(h_prev.T, dz_pre)

            dh_next = dh_prev_term1 + \
                      np.dot(dz_pre, self.Uz.T) + \
                      np.dot(dr_pre, self.Ur.T) + \
                      (np.dot(dh_tilde_pre, self.Uh.T) * r_t)


        self.grads = {
            'dWz': dWz, 'dUz': dUz, 'dbz': dbz,
            'dWr': dWr, 'dUr': dUr, 'dbr': dbr,
            'dWh': dWh, 'dUh': dUh, 'dbh': dbh,
            'dWo': dWo, 'dbo': dbo
        }

    def update_weights(self, learning_rate):
        self.Wz -= learning_rate * self.grads['dWz']
        self.Uz -= learning_rate * self.grads['dUz']
        self.bz -= learning_rate * self.grads['dbz']

        self.Wr -= learning_rate * self.grads['dWr']
        self.Ur -= learning_rate * self.grads['dUr']
        self.br -= learning_rate * self.grads['dbr'] 

        self.Wh -= learning_rate * self.grads['dWh']
        self.Uh -= learning_rate * self.grads['dUh']
        self.bh -= learning_rate * self.grads['dbh']

        self.Wo -= learning_rate * self.grads['dWo']
        self.bo -= learning_rate * self.grads['dbo']

    def get_weights(self):
        return {
            'Wz': self.Wz, 'Uz': self.Uz, 'bz': self.bz,
            'Wr': self.Wr, 'Ur': self.Ur, 'br': self.br,
            'Wh': self.Wh, 'Uh': self.Uh, 'bh': self.bh,
            'Wo': self.Wo, 'bo': self.bo
        }

    def set_weights(self, weights_dict):
        self.Wz = weights_dict['Wz']
        self.Uz = weights_dict['Uz']
        self.bz = weights_dict['bz']
        self.Wr = weights_dict['Wr']
        self.Ur = weights_dict['Ur']
        self.br = weights_dict['br']
        self.Wh = weights_dict['Wh']
        self.Uh = weights_dict['Uh']
        self.bh = weights_dict['bh']
        self.Wo = weights_dict['Wo']
        self.bo = weights_dict['bo']

# --- Fungsi untuk membuat sequence data ---
# data_scaled: sekarang akan menjadi array 2D dengan banyak fitur
def create_sequences(data_scaled, lookback_window, target_feature_index): # NEW: target_feature_index
    X, y = [], []
    
    for i in range(len(data_scaled) - lookback_window):
        X.append(data_scaled[i:(i + lookback_window), :]) # Mengambil SEMUA kolom fitur untuk input
        y.append(data_scaled[i + lookback_window, target_feature_index]) # Memprediksi target feature (misal Adj Close)

    return np.array(X), np.array(y)

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    default_ticker = 'TSLA'
    default_end_date = datetime.now().strftime('%Y-%m-%d')
    default_start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')

    if request.method == 'POST':
        ticker = request.form.get('ticker', default_ticker).upper()
        start_date_str = request.form.get('start_date')
        end_date_str = request.form.get('end_date')

        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            if start_date >= end_date:
                flash("Tanggal mulai harus sebelum tanggal selesai.", 'error')
                return render_template('index.html', default_ticker=ticker, default_start_date=start_date_str, default_end_date=end_date_str)
            if start_date > datetime.now().date() or end_date > datetime.now().date() + timedelta(days=1):
                flash("Tanggal tidak boleh di masa depan.", 'error')
                return render_template('index.html', default_ticker=ticker, default_start_date=start_date_str, default_end_date=end_date_str)
        except ValueError:
            flash("Format tanggal tidak valid. GunakanYYYY-MM-DD.", 'error')
            return render_template('index.html', default_ticker=ticker, default_start_date=start_date_str, default_end_date=end_date_str)

        try:
            data_yf = yf.download(ticker, start=start_date, end=end_date)

            if data_yf.empty:
                flash(f"Tidak ada data ditemukan untuk {ticker} pada periode tersebut.", 'error')
                return render_template('index.html', default_ticker=ticker, default_start_date=start_date_str, default_end_date=end_date_str)

            new_records_count = 0
            for index, row in data_yf.iterrows():
                record_date = index.date()

                existing_record = StockData.query.filter_by(ticker=ticker, date=record_date).first()

                if not existing_record:
                    open_val = float(row['Open'])
                    high_val = float(row['High'])
                    low_val = float(row['Low'])
                    close_val = float(row['Close'])
                    
                    adj_close_val = float(row['Adj Close']) if 'Adj Close' in row else float(row['Close'])
                    if 'Adj Close' not in row:
                        flash(f"Peringatan: Kolom 'Adj Close' tidak ditemukan untuk {ticker} pada {record_date}. Menggunakan 'Close' sebagai gantinya.", 'warning')

                    volume_val = int(row['Volume'])

                    stock_data = StockData(
                        ticker=ticker,
                        date=record_date,
                        open_price=open_val,
                        high_price=high_val,
                        low_price=low_val,
                        close_price=close_val,
                        adj_close_price=adj_close_val,
                        volume=volume_val
                    )
                    db.session.add(stock_data)
                    new_records_count += 1
            db.session.commit()

            file_name = f"{ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            file_path = os.path.join(DATA_DIR, file_name)
            data_yf.to_csv(file_path)

            flash(f"Data {ticker} berhasil diunduh dan {new_records_count} record baru disimpan ke database. File CSV juga disimpan sebagai {file_name}", 'success')
            return render_template('index.html', downloaded_file=file_name, default_ticker=ticker, default_start_date=start_date_str, default_end_date=end_date_str)
        except Exception as e:
            db.session.rollback()
            flash(f"Terjadi kesalahan saat mengambil atau menyimpan data: {e}. Coba lagi.", 'error')
            return render_template('index.html', default_ticker=ticker, default_start_date=start_date_str, default_end_date=end_date_str)

    return render_template('index.html', default_ticker=default_ticker, default_start_date=default_start_date, default_end_date=default_end_date)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(DATA_DIR, filename), as_attachment=True)

@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    ticker = request.args.get('ticker', 'TSLA').upper()
    
    stock_records = StockData.query.filter_by(ticker=ticker).order_by(StockData.date.asc()).all()

    if not stock_records:
        flash(f"Tidak ada data ditemukan di database untuk ticker {ticker}. Silakan unduh terlebih dahulu.", 'error')
        return render_template('preprocess.html', current_ticker=ticker)

    data_list = []
    for record in stock_records:
        data_list.append({
            'Date': record.date,
            'Open': record.open_price,
            'High': record.high_price,
            'Low': record.low_price,
            'Close': record.close_price,
            'Adj Close': record.adj_close_price,
            'Volume': record.volume
        })
    df = pd.DataFrame(data_list)
    df.set_index('Date', inplace=True)

    # --- NEW: Hitung Indikator Teknikal ---
    # Pastikan semua perhitungan dilakukan sebelum ffill/bfill terakhir untuk NaN
    df['SMA_10'] = df['Adj Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Adj Close'].rolling(window=20).mean()

    df['EMA_10'] = df['Adj Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Adj Close'].ewm(span=20, adjust=False).mean()

    # RSI Calculation (Period = 14)
    delta = df['Adj Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(com=14 - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=14 - 1, adjust=False).mean()

    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss) 
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- FIXED: Menggunakan ffill() dan bfill() tanpa 'method' argumen ---
    # Lakukan ffill/bfill setelah semua fitur (termasuk indikator) dihitung
    df.ffill(inplace=True)
    df.bfill(inplace=True) # Isi NaN yang mungkin tersisa di awal (jika ada)

    # --- START OF PRINTING ---
    print("\n--- DataFrame Head setelah Feature Engineering ---")
    print(df.head().to_string()) # to_string() untuk memastikan semua kolom tercetak
    print("\n--- DataFrame Tail setelah Feature Engineering ---")
    print(df.tail().to_string())
    print("\n--- Info DataFrame ---")
    df.info()
    print("\n--- Cek NaN setelah Ffill/Bfill ---")
    print(df.isnull().sum().to_string())
    print("-------------------------------------------------")
    # --- END OF PRINTING ---

    # --- NEW: Tentukan fitur-fitur yang akan diskalakan dan digunakan sebagai input ---
    features_to_scale = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
                         'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20', 'RSI'] 
    
    # Pastikan semua fitur ini ada di DataFrame
    for feature in features_to_scale:
        if feature not in df.columns:
            flash(f"Peringatan: Fitur '{feature}' tidak ditemukan di DataFrame. Mungkin ada kesalahan perhitungan indikator.", 'warning')

    data_for_scaling = df[features_to_scale].values # Ambil semua fitur yang dipilih

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_for_scaling)
    
    app.config['features_to_scale'] = features_to_scale
    app.config['scaler_all_features'] = scaler
    app.config['df_full_preprocessed'] = df.copy() # Simpan DataFrame lengkap setelah preprocessing, untuk date indexing

    LOOKBACK_WINDOW = 90
    
    # --- NEW: Dapatkan indeks Adj Close untuk fungsi create_sequences ---
    adj_close_index_for_target = features_to_scale.index('Adj Close')
    X, y = create_sequences(scaled_data, LOOKBACK_WINDOW, adj_close_index_for_target) # Pass index target

    TRAIN_SPLIT_RATIO = 0.8
    total_sequence_samples = X.shape[0]
    train_size_sequences = int(total_sequence_samples * TRAIN_SPLIT_RATIO)

    X_train_data = X[:train_size_sequences]
    y_train_data = y[:train_size_sequences]

    X_val_data = X[train_size_sequences:]
    y_val_data = y[train_size_sequences:]

    INPUT_DIM = len(features_to_scale)
    HIDDEN_DIM = 128
    OUTPUT_DIM = 1

    gru_model = None
    model_filename = os.path.join(MODEL_DIR, f'GRU_weights_{ticker}_H{HIDDEN_DIM}_L{LOOKBACK_WINDOW}_F{INPUT_DIM}.npz')

    if os.path.exists(model_filename):
        try:
            loaded_weights_data = np.load(model_filename)
            loaded_weights = {key: loaded_weights_data[key] for key in loaded_weights_data}

            temp_gru = GRU(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
            temp_gru.set_weights(loaded_weights) 
            gru_model = temp_gru
            flash(f"Model GRU untuk {ticker} dimuat dari cache: {os.path.basename(model_filename)}", 'info')
            print(f"Menggunakan model GRU yang dimuat dari {model_filename}.")

        except Exception as e:
            flash(f"Gagal memuat model dari cache ({os.path.basename(model_filename)}): {e}. Menginisialisasi model baru.", 'warning')
            print(f"Gagal memuat model dari cache ({model_filename}): {e}. Menginisialisasi model baru.")
            gru_model = GRU(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
            flash("Menginisialisasi model GRU baru.", 'info')
    else:
        print(f"File model tidak ditemukan ({model_filename}). Menginisialisasi model GRU baru.")
        gru_model = GRU(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
        flash("Menginisialisasi model GRU baru (tidak ada di cache).", 'info')

    app.config['gru_model'] = gru_model
    app.config[f'X_train_{ticker}'] = X_train_data
    app.config[f'y_train_{ticker}'] = y_train_data
    app.config[f'X_val_{ticker}'] = X_val_data
    app.config[f'y_val_{ticker}'] = y_val_data

    app.config['lookback_window'] = LOOKBACK_WINDOW
    app.config['input_dim'] = INPUT_DIM
    app.config['hidden_dim'] = HIDDEN_DIM
    app.config['output_dim'] = OUTPUT_DIM
    app.config['train_size_sequences'] = train_size_sequences

    flash(f"Data {ticker} telah dimuat dari database dan diproses awal. Total {len(df)} record. "
          f"Sequence data (Train X: {X_train_data.shape}, Train y: {y_train_data.shape}, "
          f"Val X: {X_val_data.shape}, Val y: {y_val_data.shape}) "
          f"dibuat dengan lookback window {LOOKBACK_WINDOW}. "
          f"Model GRU diinisialisasi dengan input_dim {INPUT_DIM} dan hidden_dim {HIDDEN_DIM}.", 'info')

    display_df = df.head(10).to_html(classes='data-table') + "<br>" + df.tail(10).to_html(classes='data-table')
    
    return render_template('preprocess.html', 
                           data_html=display_df, 
                           current_ticker=ticker,
                           total_records=len(df),
                           show_preprocess_results=True,
                           lookback_window=LOOKBACK_WINDOW,
                           X_shape_train=X_train_data.shape,
                           y_shape_train=y_train_data.shape,
                           X_shape_val=X_val_data.shape,
                           y_shape_val=y_val_data.shape,
                           hidden_dim=HIDDEN_DIM,
                           input_dim=INPUT_DIM)

@app.route('/train', methods=['GET'])
def train_model():
    ticker = 'TSLA'
    X_train = app.config.get(f'X_train_{ticker}')
    y_train = app.config.get(f'y_train_{ticker}')
    X_val = app.config.get(f'X_val_{ticker}')
    y_val = app.config.get(f'y_val_{ticker}')

    gru_model = app.config.get('gru_model')
    scaler = app.config.get('scaler_all_features')
    lookback_window = app.config.get('lookback_window')
    
    INPUT_DIM = app.config.get('input_dim')
    HIDDEN_DIM = app.config.get('hidden_dim')
    OUTPUT_DIM = app.config.get('output_dim')

    if X_train is None or y_train is None or X_val is None or y_val is None or gru_model is None or scaler is None:
        flash("Data atau model belum disiapkan. Silakan kunjungi halaman preprocess terlebih dahulu.", 'error')
        return redirect(url_for('preprocess', ticker=ticker))

    if gru_model.input_dim != INPUT_DIM or gru_model.hidden_dim != HIDDEN_DIM or gru_model.output_dim != OUTPUT_DIM:
        flash("Dimensi model yang dimuat tidak cocok dengan konfigurasi saat ini. Silakan inisialisasi ulang model di halaman Preprocessing.", 'error')
        return redirect(url_for('preprocess', ticker=ticker))

    EPOCHS = 200
    LEARNING_RATE = 0.005
    
    losses = []
    val_losses = []
    start_time = time.time()

    flash(f"Memulai pelatihan model GRU untuk {ticker}...", 'info')
    print(f"--- Memulai pelatihan model GRU untuk {ticker} (Epochs: {EPOCHS}, LR: {LEARNING_RATE}) ---")

    try:
        for epoch in range(EPOCHS):
            total_loss = 0
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            for i in range(X_train_shuffled.shape[0]):
                x_sample = X_train_shuffled[i]
                y_sample = y_train_shuffled[i].reshape(1, 1)

                y_pred_scaled = gru_model.forward(x_sample)

                loss = mse_loss(y_pred_scaled, y_sample)
                total_loss += loss

                d_output = mse_loss_derivative(y_pred_scaled, y_sample)
                gru_model.backward(d_output, LEARNING_RATE)

                gru_model.update_weights(LEARNING_RATE)
            
            avg_loss = total_loss / X_train_shuffled.shape[0]
            losses.append(avg_loss)

            total_val_loss = 0
            for i in range(X_val.shape[0]):
                val_x_sample = X_val[i]
                val_y_sample = y_val[i].reshape(1, 1)
                val_y_pred_scaled = gru_model.forward(val_x_sample)
                total_val_loss += mse_loss(val_y_pred_scaled, val_y_sample)
            avg_val_loss = total_val_loss / X_val.shape[0]
            val_losses.append(avg_val_loss)

            flash(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}", 'info')
            print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        print("--- Pelatihan selesai dengan sukses ---")

    except Exception as e:
        flash(f"Terjadi error fatal saat pelatihan: {e}", 'error')
        print(f"--- ERROR FATAL SAAT PELATIHAN ---")
        traceback.print_exc()
        print(f"--- ERROR FATAL SAAT PELATIHAN ---")
        return render_template('train.html', 
                               ticker=ticker,
                               epochs=EPOCHS,
                               learning_rate=LEARNING_RATE,
                               final_loss=losses[-1] if losses else 0,
                               final_val_loss=val_losses[-1] if val_losses else 0,
                               training_duration=time.time() - start_time,
                               losses=losses,
                               val_losses=val_losses,
                               error_occurred=True)


    end_time = time.time()
    training_duration = end_time - start_time

    model_filename = os.path.join(MODEL_DIR, f'GRU_weights_{ticker}_H{HIDDEN_DIM}_L{lookback_window}_F{INPUT_DIM}.npz')
    
    try:
        np.savez(model_filename, **gru_model.get_weights())
        flash(f"Model weights berhasil disimpan ke: {os.path.basename(model_filename)}", 'success')
    except Exception as e:
        flash(f"Gagal menyimpan model weights: {e}", 'error')
        print(f"ERROR: Gagal menyimpan model weights: {e}")

    flash(f"Pelatihan selesai dalam {training_duration:.2f} detik. Final Train Loss: {losses[-1]:.6f}, Final Val Loss: {val_losses[-1]:.6f}", 'success')

    return render_template('train.html', 
                           ticker=ticker,
                           epochs=EPOCHS,
                           learning_rate=LEARNING_RATE,
                           final_loss=losses[-1],
                           final_val_loss=val_losses[-1],
                           training_duration=training_duration,
                           losses=losses,
                           val_losses=val_losses)

@app.route('/predict', methods=['GET'])
def predict_price():
    ticker = 'TSLA'
    X_test_samples = app.config.get(f'X_val_{ticker}')
    y_true_scaled_test = app.config.get(f'y_val_{ticker}')

    gru_model = app.config.get('gru_model')
    scaler = app.config.get('scaler_all_features')
    lookback_window = app.config.get('lookback_window')
    features_to_scale = app.config.get('features_to_scale')
    df_full_preprocessed = app.config.get('df_full_preprocessed')

    if X_test_samples is None or y_true_scaled_test is None or gru_model is None or scaler is None or features_to_scale is None or df_full_preprocessed is None:
        flash("Data, model, atau scaler belum disiapkan atau dilatih. Silakan kunjungi halaman preprocess dan latih model terlebih dahulu.", 'error')
        return redirect(url_for('train_model'))

    test_size = X_test_samples.shape[0]
    
    predictions_scaled = []
    
    for i in range(X_test_samples.shape[0]):
        x_sample_test = X_test_samples[i]
        y_pred_scaled = gru_model.forward(x_sample_test)
        predictions_scaled.append(y_pred_scaled[0, 0])
        
    adj_close_index = features_to_scale.index('Adj Close')

    predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)

    num_features = len(features_to_scale)
    predictions_scaled_padded = np.zeros((predictions_scaled.shape[0], num_features))
    predictions_scaled_padded[:, adj_close_index] = predictions_scaled.flatten()

    predictions_original_full = scaler.inverse_transform(predictions_scaled_padded)
    predictions_original = predictions_original_full[:, adj_close_index].reshape(-1, 1)
    
    y_true_scaled_padded = np.zeros((y_true_scaled_test.shape[0], num_features))
    y_true_scaled_padded[:, adj_close_index] = y_true_scaled_test.flatten()

    y_true_original_full = scaler.inverse_transform(y_true_scaled_padded)
    y_true_original = y_true_original_full[:, adj_close_index].reshape(-1, 1)

    rmse = np.sqrt(np.mean((predictions_original - y_true_original)**2))

    train_size_sequences = app.config.get('train_size_sequences')
    
    # Calculate the start index for validation dates in the original full dataframe
    # total_samples_in_X_before_val = train_size_sequences
    # The first 'y' in the sequence is at index `lookback_window` of the original scaled_data.
    # So, the date for X_val[0] is at index `train_size_sequences + lookback_window` in the original df_full_preprocessed
    start_index_for_val_dates_in_df = train_size_sequences + lookback_window 
    
    predicted_dates = df_full_preprocessed.index[start_index_for_val_dates_in_df : start_index_for_val_dates_in_df + test_size]


    prediction_results = []
    for i in range(test_size):
        prediction_results.append({
            'Date': predicted_dates[i].strftime('%Y-%m-%d'),
            'True Price': f"{y_true_original[i, 0]:.2f}",
            'Predicted Price': f"{predictions_original[i, 0]:.2f}"
        })
    
    display_results = prediction_results[-20:]

    flash(f"Prediksi berhasil dilakukan untuk {test_size} hari terakhir. RMSE: {rmse:.4f}", 'success')

    return render_template('predict.html', 
                           ticker=ticker,
                           rmse=rmse,
                           prediction_results=display_results,
                           total_predictions=test_size)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)