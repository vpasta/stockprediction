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
    return 2 * (y_pred - y_true) / len(y_true) # Turunan MSE terhadap y_pred

# --- Kelas GRU dari Nol ---
class GRU:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Inisialisasi Bobot dan Bias
        # Bobot dan Bias untuk Update Gate (z)
        self.Wz = np.random.randn(input_dim, hidden_dim) * 0.01
        self.Uz = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.bz = np.zeros((1, hidden_dim))

        # Bobot dan Bias untuk Reset Gate (r)
        self.Wr = np.random.randn(input_dim, hidden_dim) * 0.01
        self.Ur = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.br = np.zeros((1, hidden_dim))

        # Bobot dan Bias untuk Candidate Hidden State (h_tilde)
        self.Wh = np.random.randn(input_dim, hidden_dim) * 0.01
        self.Uh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.bh = np.zeros((1, hidden_dim))

        # Bobot dan Bias untuk Output Layer
        self.Wo = np.random.randn(hidden_dim, output_dim) * 0.01
        self.bo = np.zeros((1, output_dim))

        # Variabel untuk menyimpan nilai-nilai perantara (untuk backward pass nanti)
        self.cache = {}

    def forward(self, X):
        """
        Melakukan forward pass pada GRU untuk satu sequence.
        X (np.ndarray): Input sequence, shape (sequence_length, input_dim)
        """
        sequence_length = X.shape[0]
        
        h_prev = np.zeros((1, self.hidden_dim))

        hidden_states = []
        
        # Cache untuk setiap time step dalam sequence (penting untuk BPTT)
        sequence_cache = [] # List of dicts, each dict contains cache for one time step

        # Iterate over each time step in the input sequence
        for t in range(sequence_length):
            x_t = X[t].reshape(1, self.input_dim) # Ambil input pada time step t, reshape ke (1, input_dim)

            # Pre-activations (simpan untuk turunan nanti)
            z_pre = np.dot(x_t, self.Wz) + np.dot(h_prev, self.Uz) + self.bz
            r_pre = np.dot(x_t, self.Wr) + np.dot(h_prev, self.Ur) + self.br
            h_tilde_pre = np.dot(x_t, self.Wh) + np.dot(r_t * h_prev, self.Uh) + self.bh # Note: r_t here is from previous step

            # Hitung Update Gate (z_t)
            z_t = sigmoid(z_pre)

            # Hitung Reset Gate (r_t)
            r_t = sigmoid(r_pre)

            # Hitung Candidate Hidden State (h_tilde_t)
            # Penting: gunakan r_t dari perhitungan saat ini
            h_tilde_t = tanh(np.dot(x_t, self.Wh) + np.dot(r_t * h_prev, self.Uh) + self.bh)

            # Hitung Hidden State (h_t)
            h_t = (1 - z_t) * h_prev + z_t * h_tilde_t

            # Simpan nilai-nilai yang dibutuhkan untuk backward pass
            sequence_cache.append({
                'x_t': x_t,
                'h_prev': h_prev.copy(), 
                'z_pre': z_pre, 'r_pre': r_pre, 'h_tilde_pre': h_tilde_pre, # pre-activations
                'z_t': z_t,
                'r_t': r_t,
                'h_tilde_t': h_tilde_t,
                'h_t': h_t
            })

            h_prev = h_t # Update hidden state untuk time step berikutnya
            hidden_states.append(h_t)

        final_h_t = hidden_states[-1]
        output = np.dot(final_h_t, self.Wo) + self.bo

        self.cache['sequence_cache'] = sequence_cache
        self.cache['final_h_t'] = final_h_t # Output dari GRU sebelum layer akhir
        self.cache['output'] = output # Output prediksi akhir
        return output

    def backward(self, d_output, learning_rate):
        """
        Melakukan backward pass (BPTT) pada GRU untuk satu sequence.
        Menghitung gradien dan memperbarui bobot.
        
        Args:
            d_output (np.ndarray): Gradien dari loss terhadap output akhir model, shape (1, output_dim)
            learning_rate (float): Tingkat pembelajaran
        """
        sequence_cache = self.cache['sequence_cache']
        final_h_t = self.cache['final_h_t']

        # Inisialisasi gradien untuk bobot dan bias
        dWz, dUz, dbz = np.zeros_like(self.Wz), np.zeros_like(self.Uz), np.zeros_like(self.bz)
        dWr, dUr, dbr = np.zeros_like(self.Wr), np.zeros_like(self.Ur), np.zeros_like(self.br)
        dWh, dUh, dbh = np.zeros_like(self.Wh), np.zeros_like(self.Uh), np.zeros_like(self.bh)
        dWo, dbo = np.zeros_like(self.Wo), np.zeros_like(self.bo)

        # Gradien untuk layer output (sebelumnya: self.Wo, self.bo)
        # d_output_layer = d_output (gradien dari loss terhadap output)
        dWo += np.dot(final_h_t.T, d_output)
        dbo += d_output

        # Inisialisasi gradien hidden state dari time step berikutnya
        # Ini penting karena gradien dari h_t disebarkan mundur melalui waktu
        dh_next = np.dot(d_output, self.Wo.T) # Gradien dari output layer kembali ke final_h_t

        # Iterate backward through the sequence
        for t in reversed(range(len(sequence_cache))):
            cache_t = sequence_cache[t]
            x_t, h_prev, z_pre, r_pre, h_tilde_pre, z_t, r_t, h_tilde_t, h_t = \
                cache_t['x_t'], cache_t['h_prev'], cache_t['z_pre'], cache_t['r_pre'], cache_t['h_tilde_pre'], \
                cache_t['z_t'], cache_t['r_t'], cache_t['h_tilde_t'], cache_t['h_t']

            # Total gradien untuk h_t (dari current time step + dari future time step)
            dh_t = dh_next # Gradien dari h_t pada time step selanjutnya (dh_next akan di update)

            # Gradien untuk h_prev (dari 1 - z_t) * h_prev
            dh_prev_term1 = dh_t * (1 - z_t)

            # Gradien terhadap z_t
            dz_t = dh_t * (h_tilde_t - h_prev)
            dz_pre = dz_t * sigmoid_derivative(z_pre) # Gradien melalui sigmoid

            # Gradien terhadap h_tilde_t
            dh_tilde_t = dh_t * z_t
            dh_tilde_pre = dh_tilde_t * tanh_derivative(h_tilde_pre) # Gradien melalui tanh

            # Gradien terhadap Wh, Uh, bh dari h_tilde_t
            dWh += np.dot(x_t.T, dh_tilde_pre)
            db_h += dh_tilde_pre
            # Ini adalah bagian tricky untuk Uh dan h_prev
            dUh_h_prev_product = dh_tilde_pre # Gradien yang masuk ke (r_t * h_prev) * Uh
            dUh += np.dot((r_t * h_prev).T, dUh_h_prev_product)
            
            # Gradien untuk r_t (dari r_t * h_prev * Uh)
            dr_t_from_h_tilde = dUh_h_prev_product * self.Uh.T * h_prev # Ini salah, harus dot product, bukan element-wise
            dr_t_from_h_tilde = np.dot(dUh_h_prev_product, self.Uh.T) * h_prev # Perbaikan ini

            # Gradien terhadap r_t
            dr_t = dr_t_from_h_tilde # Ini adalah gradien r_t yang berasal dari h_tilde_t
            dr_pre = dr_t * sigmoid_derivative(r_pre) # Gradien melalui sigmoid

            # Gradien terhadap Wr, Ur, br dari r_t
            dWr += np.dot(x_t.T, dr_pre)
            dbr += dr_pre
            dUr += np.dot(h_prev.T, dr_pre)

            # Gradien terhadap z_t
            dWz += np.dot(x_t.T, dz_pre)
            dbz += dz_pre
            dUz += np.dot(h_prev.T, dz_pre)

            # Gradien untuk h_prev yang berasal dari r_t * h_prev di h_tilde
            dh_prev_term2 = np.dot(dr_t, self.Ur.T) + np.dot(dh_tilde_pre * r_t, self.Uh.T) # Perbaikan ini
            
            # Total gradien untuk h_prev pada time step ini, yang akan diteruskan ke time step sebelumnya
            dh_next = dh_prev_term1 + dh_prev_term2


        # Akumulasi gradien yang telah dihitung ke dalam self.grads
        self.grads = {
            'dWz': dWz, 'dUz': dUz, 'dbz': dbz,
            'dWr': dWr, 'dUr': dUr, 'dbr': dbr,
            'dWh': dWh, 'dUh': dUh, 'dbh': dbh,
            'dWo': dWo, 'dbo': dbo
        }
        # return self.grads # Kita tidak perlu mengembalikan grads secara eksplisit karena langsung di update_weights

    def update_weights(self, learning_rate):
        """
        Memperbarui bobot model menggunakan gradien yang telah dihitung.
        """
        # Perbarui bobot dan bias menggunakan gradien
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

# --- Fungsi untuk membuat sequence data (tetap sama) ---
def create_sequences(data_scaled, lookback_window):
    X, y = [], []
    for i in range(len(data_scaled) - lookback_window):
        X.append(data_scaled[i:(i + lookback_window), 0])
        y.append(data_scaled[i + lookback_window, 0])
    return np.array(X), np.array(y)

# --- NEW ROUTE: Pelatihan Model ---
@app.route('/train', methods=['GET'])
def train_model():
    # Pastikan data dan model sudah disiapkan
    ticker = 'TSLA' # Saat ini kita hanya fokus pada TSLA
    X_train = app.config.get(f'X_train_{ticker}')
    y_train = app.config.get(f'y_train_{ticker}')
    gru_model = app.config.get('gru_model')
    scaler = app.config.get('scaler_adj_close')
    lookback_window = app.config.get('lookback_window')

    if X_train is None or y_train is None or gru_model is None or scaler is None:
        flash("Data atau model belum disiapkan. Silakan kunjungi halaman preprocess terlebih dahulu.", 'error')
        return redirect(url_for('preprocess', ticker=ticker))

    # --- Hyperparameters Pelatihan ---
    EPOCHS = 20 # Jumlah epoch pelatihan
    LEARNING_RATE = 0.01 # Tingkat pembelajaran
    
    losses = []
    start_time = time.time()

    flash(f"Memulai pelatihan model GRU untuk {ticker}...", 'info')

    # Loop pelatihan
    for epoch in range(EPOCHS):
        total_loss = 0
        for i in range(X_train.shape[0]):
            x_sample = X_train[i] # Satu sequence input
            y_sample = y_train[i].reshape(1, 1) # Satu target output, reshape ke (1,1)

            # Forward pass
            y_pred_scaled = gru_model.forward(x_sample)

            # Hitung Loss
            loss = mse_loss(y_pred_scaled, y_sample)
            total_loss += loss

            # Backward pass
            d_output = mse_loss_derivative(y_pred_scaled, y_sample)
            gru_model.backward(d_output, LEARNING_RATE) # Ini akan mengisi self.grads di model

            # Update bobot
            gru_model.update_weights(LEARNING_RATE)
        
        avg_loss = total_loss / X_train.shape[0]
        losses.append(avg_loss)
        flash(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}", 'info')
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}") # Tetap print ke konsol untuk melihat progres real-time

    end_time = time.time()
    training_duration = end_time - start_time

    flash(f"Pelatihan selesai dalam {training_duration:.2f} detik. Final Loss: {losses[-1]:.6f}", 'success')

    # Setelah pelatihan, Anda bisa menambahkan logika untuk menyimpan model (bobotnya)
    # Misalnya, simpan bobot ke file .npy atau database agar bisa dimuat kembali tanpa retraining
    # Contoh: np.save('gru_weights.npy', gru_model.get_weights_as_dict())
    # Untuk project ini, kita simpan di app.config saja untuk sementara.

    return render_template('train.html', 
                           ticker=ticker,
                           epochs=EPOCHS,
                           learning_rate=LEARNING_RATE,
                           final_loss=losses[-1],
                           training_duration=training_duration,
                           losses=losses) # Kirim list losses untuk potensi visualisasi

# ... (kode Flask routes lainnya seperti index dan download_file tetap sama) ...
@app.route('/', methods=['GET', 'POST'])
def index():
    # ... (kode di sini tetap sama seperti sebelumnya) ...
    default_ticker = 'TSLA'
    default_end_date = datetime.now().strftime('%Y-%m-%d')
    default_start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')

    return render_template('index.html', default_ticker=default_ticker, default_start_date=default_start_date, default_end_date=default_end_date)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(DATA_DIR, filename), as_attachment=True)

@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    # ... (kode di sini tetap sama seperti sebelumnya) ...
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

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    data_for_scaling = df[['Adj Close']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_for_scaling)

    LOOKBACK_WINDOW = 60
    X, y = create_sequences(scaled_data, LOOKBACK_WINDOW)

    INPUT_DIM = 1
    HIDDEN_DIM = 50
    OUTPUT_DIM = 1

    if 'gru_model' not in app.config or app.config['gru_model'].input_dim != INPUT_DIM or \
       app.config['gru_model'].hidden_dim != HIDDEN_DIM or app.config['gru_model'].output_dim != OUTPUT_DIM:
        print(f"Inisialisasi model GRU dengan input_dim={INPUT_DIM}, hidden_dim={HIDDEN_DIM}, output_dim={OUTPUT_DIM}")
        gru_model = GRU(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
        app.config['gru_model'] = gru_model
    else:
        gru_model = app.config['gru_model']
        print("Menggunakan model GRU yang sudah ada.")

    app.config[f'X_train_{ticker}'] = X
    app.config[f'y_train_{ticker}'] = y
    app.config['scaler_adj_close'] = scaler
    app.config['lookback_window'] = LOOKBACK_WINDOW
    app.config['input_dim'] = INPUT_DIM
    app.config['hidden_dim'] = HIDDEN_DIM
    app.config['output_dim'] = OUTPUT_DIM

    flash(f"Data {ticker} telah dimuat dari database dan diproses awal. Total {len(df)} record. "
          f"Sequence data (X: {X.shape}, y: {y.shape}) dibuat dengan lookback window {LOOKBACK_WINDOW}. "
          f"Model GRU diinisialisasi dengan hidden_dim {HIDDEN_DIM}.", 'info')

    display_df = df.head(10).to_html(classes='data-table') + "<br>" + df.tail(10).to_html(classes='data-table')
    
    return render_template('preprocess.html', 
                           data_html=display_df, 
                           current_ticker=ticker,
                           total_records=len(df),
                           show_preprocess_results=True,
                           lookback_window=LOOKBACK_WINDOW,
                           X_shape=X.shape,
                           y_shape=y.shape,
                           hidden_dim=HIDDEN_DIM)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)