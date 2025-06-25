from flask import Flask, render_template, request, send_file, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from functools import wraps
import time
import traceback
import sys
import config
from utils.db_models import db, StockData, User, SavedModel, PredictionDetail
from utils.gru_model import GRU
from utils.data_preprocessing import calculate_technical_indicators, create_sequences
from utils.loss_functions import mse_loss, mse_loss_derivative, mae_loss, mape_loss
from sklearn.preprocessing import MinMaxScaler 

app = Flask(__name__)

app.secret_key = config.SECRET_KEY 
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+mysqlconnector://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.permanent_session_lifetime = timedelta(days=7)

db.init_app(app) 

DATA_DIR = config.DATA_DIR
MODEL_DIR = config.MODEL_DIR
LOG_DIR = config.LOG_DIR

def log_to_both(message):
    print(message)

def get_last_n_days_data(df, n_days):
    """
    Mengambil n_days terakhir dari DataFrame, pastikan data lengkap.
    """
    if len(df) < n_days:
        return None
    return df.iloc[-n_days:]

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            flash('Anda Harus Login Terlebih Dahulu', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            flash('Anda harus Login terlebih dahulu.', 'error')
            return redirect(url_for('login'))
        if session.get('user_role') != 'admin':
            flash('Anda tidak memiliki izin untuk mengakses halaman ini.', 'error')
            return redirect(url_for('dashboard_user'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form.get('role', 'free_user') 

        if not username or not password:
            flash('Username dan password tidak boleh kosong.', 'error')
            return render_template('register.html')

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username sudah ada. Silakan pilih username lain.', 'error')
            return render_template('register.html')

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_password, role=role)
        db.session.add(new_user)
        try:
            db.session.commit()
            flash(f'Pengguna {username} dengan peran {role} berhasil didaftarkan. Silakan login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f'Terjadi kesalahan saat pendaftaran: {e}', 'error')
            return render_template('register.html')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password_hash, password):
            session['logged_in'] = True
            session['username'] = user.username
            session['user_role'] = user.role
            flash(f'Selamat datang, {user.username}!', 'success')

            if user.role == 'admin':
                return redirect(url_for('admin_dashboard')) 
            else:
                return redirect(url_for('dashboard_user')) 
        else:
            flash('Username atau password salah.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    session.pop('user_role', None)
    session.clear()
    flash('Anda telah berhasil logout.', 'info')
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@admin_required
def index():
    default_ticker = config.DEFAULT_TICKER
    default_end_date = datetime.now().strftime('%Y-%m-%d')
    default_start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')

    default_epochs = config.DEFAULT_EPOCHS
    default_learning_rate = config.DEFAULT_LEARNING_RATE
    default_dropout_rate = config.DEFAULT_DROPOUT_RATE
    default_patience = config.EARLY_STOPPING_PATIENCE

    if request.method == 'POST':
        ticker = request.form.get('ticker', default_ticker).upper()
        start_date_str = request.form.get('start_date')
        end_date_str = request.form.get('end_date')

        try:
            epochs = int(request.form.get('epochs', default_epochs))
            learning_rate = float(request.form.get('learning_rate', default_learning_rate))
            dropout_rate = float(request.form.get('dropout_rate', default_dropout_rate))
            patience = int(request.form.get('patience', default_patience))

            app.config['current_epochs'] = epochs
            app.config['current_learning_rate'] = learning_rate
            app.config['current_dropout_rate'] = dropout_rate
            app.config['current_patience'] = patience

            flash("Parameter pelatihan model telah diatur.", 'info')

        except ValueError:
            flash("Input parameter pelatihan tidak valid. Harap masukkan angka yang benar.", 'error')
            return render_template('index.html', 
                                   default_ticker=ticker, 
                                   default_start_date=start_date_str, 
                                   default_end_date=end_date_str,
                                   default_epochs=default_epochs, 
                                   default_learning_rate=default_learning_rate,
                                   default_dropout_rate=default_dropout_rate,
                                   default_patience=default_patience)


        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            if start_date >= end_date:
                flash("Tanggal mulai harus sebelum tanggal selesai.", 'error')
                return render_template('index.html', default_ticker=ticker, default_start_date=start_date_str, default_end_date=end_date_str,
                                       default_epochs=epochs,
                                       default_learning_rate=learning_rate,
                                       default_dropout_rate=dropout_rate,
                                       default_patience=patience)
            if start_date > datetime.now().date() or end_date > datetime.now().date() + timedelta(days=1):
                flash("Tanggal tidak boleh di masa depan.", 'error')
                return render_template('index.html', default_ticker=ticker, default_start_date=start_date_str, default_end_date=end_date_str,
                                       default_epochs=epochs, 
                                       default_learning_rate=learning_rate,
                                       default_dropout_rate=dropout_rate,
                                       default_patience=patience)
        except ValueError:
            flash("Format tanggal tidak valid. GunakanYYYY-MM-DD.", 'error')
            return render_template('index.html', default_ticker=ticker, default_start_date=start_date_str, default_end_date=end_date_str,
                                   default_epochs=epochs, 
                                   default_learning_rate=learning_rate,
                                   default_dropout_rate=dropout_rate,
                                   default_patience=patience)

        try:
            data_yf = yf.download(ticker, start=start_date, end=end_date)

            if data_yf.empty:
                flash(f"Tidak ada data ditemukan untuk {ticker} pada periode tersebut.", 'error')
                return render_template('index.html', default_ticker=ticker, default_start_date=start_date_str, default_end_date=end_date_str,
                                       default_epochs=epochs, default_learning_rate=learning_rate, default_dropout_rate=dropout_rate, default_patience=patience)

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
            return render_template('index.html', downloaded_file=file_name, default_ticker=ticker, default_start_date=start_date_str, default_end_date=end_date_str,
                                   default_epochs=epochs, default_learning_rate=learning_rate, default_dropout_rate=dropout_rate, default_patience=patience)
        except Exception as e:
            db.session.rollback()
            flash(f"Terjadi kesalahan saat mengambil atau menyimpan data: {e}. Coba lagi.", 'error')
            return render_template('index.html', default_ticker=ticker, default_start_date=start_date_str, default_end_date=end_date_str,
                                   default_epochs=epochs, default_learning_rate=learning_rate, default_dropout_rate=dropout_rate, default_patience=patience)

    return render_template('index.html', 
                           default_ticker=default_ticker, 
                           default_start_date=default_start_date, 
                           default_end_date=default_end_date,
                           default_epochs=default_epochs,
                           default_learning_rate=default_learning_rate,
                           default_dropout_rate=default_dropout_rate,
                           default_patience=default_patience)

@app.route('/download/<filename>')
@admin_required
def download_file(filename):
    return send_file(os.path.join(DATA_DIR, filename), as_attachment=True)

@app.route('/preprocess', methods=['GET', 'POST'])
@admin_required
def preprocess():
    ticker = request.args.get('ticker', config.DEFAULT_TICKER).upper()

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

    df = calculate_technical_indicators(df)

    print("\n--- DataFrame Head setelah Feature Engineering ---")
    print(df.head().to_string())
    print("\n--- DataFrame Tail setelah Feature Engineering ---")
    print(df.tail().to_string())
    print("\n--- Info DataFrame ---")
    df.info()
    print("\n--- Cek NaN setelah Ffill/Bfill ---")
    print(df.isnull().sum().to_string())
    print("-------------------------------------------------")

    features_to_scale = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                         'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20', 'RSI',
                         'MACD', 'Signal_Line', 'MACD_Hist', 'ATR']

    for feature in features_to_scale:
        if feature not in df.columns:
            flash(f"Peringatan: Fitur '{feature}' tidak ditemukan di DataFrame. Mungkin ada kesalahan perhitungan indikator.", 'warning')

    data_for_scaling = df[features_to_scale].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_for_scaling)

    app.config['features_to_scale'] = features_to_scale
    app.config['scaler_all_features'] = scaler
    app.config['df_full_preprocessed'] = df.copy()

    LOOKBACK_WINDOW = config.DEFAULT_LOOKBACK_WINDOW 

    adj_close_index_for_target = features_to_scale.index('Adj Close')
    X, y = create_sequences(scaled_data, LOOKBACK_WINDOW, adj_close_index_for_target)

    TRAIN_SPLIT_RATIO = config.DEFAULT_TRAIN_SPLIT_RATIO 
    total_sequence_samples = X.shape[0]
    train_size_sequences = int(total_sequence_samples * TRAIN_SPLIT_RATIO)

    X_train_data = X[:train_size_sequences]
    y_train_data = y[:train_size_sequences]

    X_val_data = X[train_size_sequences:]
    y_val_data = y[train_size_sequences:]

    INPUT_DIM = len(features_to_scale)
    HIDDEN_DIM = config.DEFAULT_HIDDEN_DIM 
    OUTPUT_DIM = 1
    EPOCHS = app.config.get('current_epochs', config.DEFAULT_EPOCHS)
    LEARNING_RATE = app.config.get('current_learning_rate', config.DEFAULT_LEARNING_RATE)
    DROPOUT_RATE = app.config.get('current_dropout_rate', config.DEFAULT_DROPOUT_RATE)
    EARLY_STOPPING_PATIENCE = app.config.get('current_patience', config.EARLY_STOPPING_PATIENCE)


    lr_str = str(LEARNING_RATE).replace('.', '')
    do_str = str(DROPOUT_RATE).replace('.', '')


    gru_model = None
    model_filename = os.path.join(MODEL_DIR, f'GRU_weights_{ticker}_H{HIDDEN_DIM}_L{LOOKBACK_WINDOW}_F{INPUT_DIM}_LR{lr_str}_DO{do_str}.npz')


    model_loaded_from_cache = False 

    if os.path.exists(model_filename):
        try:
            loaded_weights_data = np.load(model_filename)
            loaded_weights = {key: loaded_weights_data[key] for key in loaded_weights_data}

            temp_gru = GRU(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout_rate=DROPOUT_RATE)
            temp_gru.set_weights(loaded_weights)
            gru_model = temp_gru
            flash(f"Model GRU untuk {ticker} dimuat dari cache: {os.path.basename(model_filename)}", 'info')
            print(f"Menggunakan model GRU yang dimuat dari {model_filename}.")
            model_loaded_from_cache = True 

        except Exception as e:
            flash(f"Gagal memuat model dari cache ({os.path.basename(model_filename)}): {e}. Menginisialisasi model baru.", 'warning')
            print(f"Gagal memuat model dari cache ({model_filename}): {e}. Menginisialisasi model baru.")
            gru_model = GRU(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout_rate=DROPOUT_RATE)
            flash("Menginisialisasi model GRU baru.", 'info')
    else:
        print(f"File model tidak ditemukan ({model_filename}). Menginisialisasi model GRU baru.")
        gru_model = GRU(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout_rate=DROPOUT_RATE)
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
    app.config['dropout_rate'] = DROPOUT_RATE
    app.config['learning_rate'] = LEARNING_RATE
    app.config['epochs'] = EPOCHS
    app.config['early_stopping_patience'] = EARLY_STOPPING_PATIENCE

    flash(f"Data {ticker} telah dimuat dari database dan diproses awal. Total {len(df)} record. "
          f"Sequence data (Train X: {X_train_data.shape}, Train y: {y_train_data.shape}, "
          f"Val X: {X_val_data.shape}, Val y: {y_val_data.shape}) "
          f"dibuat dengan lookback window {LOOKBACK_WINDOW}. "
          f"Model GRU diinisialisasi dengan input_dim {INPUT_DIM}, hidden_dim {HIDDEN_DIM}, dropout_rate {DROPOUT_RATE}.", 'info')

    pd.set_option('display.max_rows', None) 
    head_html = df.head(10).to_html(classes='table table-bordered table-striped table-sm', index=True, border=0)
    tail_html = df.tail(10).to_html(classes='table table-bordered table-striped table-sm', index=True, border=0)
    display_df = f"{head_html}<br><br>{tail_html}"

    return render_template('preprocess.html',
                           data_html=display_df,
                           current_ticker=ticker,
                           total_records=len(df),
                           show_preprocess_results=True,
                           lookback_window=LOOKBACK_WINDOW,
                           learning_rate=LEARNING_RATE,
                           epochs=EPOCHS,
                           early_stopping_patience=EARLY_STOPPING_PATIENCE,
                           X_shape_train=X_train_data.shape,
                           y_shape_train=y_train_data.shape,
                           X_shape_val=X_val_data.shape,
                           y_shape_val=y_val_data.shape,
                           hidden_dim=HIDDEN_DIM,
                           input_dim=INPUT_DIM,
                           dropout_rate=DROPOUT_RATE,
                           model_loaded_from_cache=model_loaded_from_cache)

@app.route('/train', methods=['GET'])
@admin_required
def train_model():
    ticker = config.DEFAULT_TICKER 
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
    EPOCHS = app.config.get('current_epochs', config.DEFAULT_EPOCHS)
    LEARNING_RATE = app.config.get('current_learning_rate', config.DEFAULT_LEARNING_RATE)
    DROPOUT_RATE = app.config.get('current_dropout_rate', config.DEFAULT_DROPOUT_RATE)
    EARLY_STOPPING_PATIENCE = app.config.get('current_patience', config.EARLY_STOPPING_PATIENCE)


    if X_train is None or y_train is None or X_val is None or y_val is None or gru_model is None or scaler is None:
        flash("Data atau model belum disiapkan. Silakan kunjungi halaman preprocess terlebih dahulu.", 'error')
        return redirect(url_for('preprocess', ticker=ticker))

    if gru_model.hidden_dim != HIDDEN_DIM or \
       gru_model.output_dim != OUTPUT_DIM or \
       gru_model.dropout_rate != DROPOUT_RATE or \
       gru_model.input_dim != INPUT_DIM: 
        flash("Dimensi atau dropout rate model yang dimuat tidak cocok dengan konfigurasi saat ini. Silakan inisialisasi ulang model di halaman Preprocessing.", 'error')
        return redirect(url_for('preprocess', ticker=ticker))

    BATCH_SIZE = config.DEFAULT_BATCH_SIZE 
    
    lr_str = str(LEARNING_RATE).replace('.', '')
    do_str = str(DROPOUT_RATE).replace('.', '')

    losses = []
    val_losses = []
    start_time = time.time()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'train_log_{ticker}_HD{HIDDEN_DIM}_LR{lr_str}_DO{do_str}_BS{BATCH_SIZE}_W{lookback_window}_{timestamp}.txt')

    current_training_logs = []
    def log_to_both(message):
        print(message) 
        current_training_logs.append(message) 

    log_to_both(f"--- Memulai pelatihan model GRU untuk {ticker} ---")
    log_to_both(f"Tanggal dan Waktu Pelatihan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_to_both("==================================================")
    log_to_both("Parameter Konfigurasi Model:")
    log_to_both(f"- Ticker Saham: {ticker}")
    log_to_both(f"- Lookback Window: {lookback_window} hari")
    log_to_both(f"- Dimensi Tersembunyi (Hidden Dim): {HIDDEN_DIM}")
    log_to_both(f"- Jumlah Fitur Input: {INPUT_DIM}") 
    log_to_both(f"- Epochs Maksimal: {EPOCHS}")
    log_to_both(f"- Learning Rate: {LEARNING_RATE}")
    log_to_both(f"- Dropout Rate: {DROPOUT_RATE}")
    log_to_both(f"- Batch Size: {BATCH_SIZE}")
    log_to_both(f"- Rasio Pembagian Data (Train/Val): {config.DEFAULT_TRAIN_SPLIT_RATIO*100:.0f}% / {(1 - config.DEFAULT_TRAIN_SPLIT_RATIO)*100:.0f}%")
    log_to_both(f"- Early Stopping Patience: {EARLY_STOPPING_PATIENCE} Epoch")
    log_to_both("==================================================\n")

    try:
        for epoch in range(EPOCHS):
            total_loss = 0
            gru_model.set_training_mode(True) 

            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            num_batches = int(np.ceil(X_train_shuffled.shape[0] / BATCH_SIZE)) 
            for i in range(num_batches):
                start_idx = i * BATCH_SIZE
                end_idx = min((i + 1) * BATCH_SIZE, X_train_shuffled.shape[0])

                x_batch = X_train_shuffled[start_idx:end_idx] 
                y_batch = y_train_shuffled[start_idx:end_idx].reshape(-1, 1) 

                y_pred_scaled = gru_model.forward(x_batch) 

                loss = mse_loss(y_pred_scaled, y_batch)
                total_loss += loss

                d_output = mse_loss_derivative(y_pred_scaled, y_batch)
                gru_model.backward(d_output) 

                gru_model.update_weights(LEARNING_RATE)

            avg_loss = total_loss / num_batches
            losses.append(avg_loss)

            gru_model.set_training_mode(False) 
            total_val_loss = 0
            num_val_batches = int(np.ceil(X_val.shape[0] / BATCH_SIZE))
            for i in range(num_val_batches):
                start_idx_val = i * BATCH_SIZE
                end_idx_val = min((i + 1) * BATCH_SIZE, X_val.shape[0])

                val_x_batch = X_val[start_idx_val:end_idx_val]
                val_y_batch = y_val[start_idx_val:end_idx_val].reshape(-1, 1)
                val_y_pred_scaled = gru_model.forward(val_x_batch)
                total_val_loss += mse_loss(val_y_pred_scaled, val_y_batch)
            avg_val_loss = total_val_loss / num_val_batches
            val_losses.append(avg_val_loss)

            log_message = f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            log_to_both(log_message)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_weights = gru_model.get_weights()
                log_to_both("Validation loss meningkat. Menyimpan bobot model terbaik.")
            else:
                patience_counter += 1
                log_to_both(f"Validation loss tidak meningkat. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                log_to_both(f"Early stopping dipicu setelah {epoch + 1} epoch tanpa peningkatan validasi loss.")
                log_to_both("--- Early stopping dipicu ---")
                break

        log_to_both("--- Pelatihan selesai dengan sukses ---")

    except Exception as e:
        flash(f"Terjadi error fatal saat pelatihan: {e}", 'error')
        log_to_both(f"--- ERROR FATAL SAAT PELATIHAN: {e} ---")
        traceback.print_exc() 
        log_to_both(traceback.format_exc()) 
        log_to_both(f"--- ERROR FATAL SAAT PELATIHAN ---")
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

    model_filename = os.path.join(MODEL_DIR, f'GRU_weights_{ticker}_H{HIDDEN_DIM}_L{lookback_window}_F{INPUT_DIM}_LR{lr_str}_DO{do_str}.npz')

    try:
        if best_model_weights is not None:
            np.savez(model_filename, **best_model_weights)
            log_to_both(f"Bobot model TERBAIK berhasil disimpan ke: {os.path.basename(model_filename)}")
            gru_model.set_weights(best_model_weights) 
        else: 
            np.savez(model_filename, **gru_model.get_weights())
            log_to_both(f"Bobot model terakhir berhasil disimpan ke: {os.path.basename(model_filename)}")
    except Exception as e:
        log_to_both(f"ERROR: Gagal menyimpan bobot model: {e}")
        print(f"ERROR: Gagal menyimpan bobot model: {e}")

    final_loss = losses[-1] if losses else 0
    final_val_loss = val_losses[-1] if val_losses else 0
    if best_model_weights is not None and best_val_loss != float('inf'):
        final_val_loss = best_val_loss 

    log_to_both(f"\n--- Ringkasan Pelatihan Akhir ---")
    log_to_both(f"Durasi Pelatihan Total: {training_duration:.2f} detik")
    log_to_both(f"Final Train Loss (Epoch terakhir): {final_loss:.6f}")
    log_to_both(f"Final Val Loss (Best): {final_val_loss:.6f}")
    log_to_both(f"Model Terbaik Disimpan: {os.path.basename(model_filename)}")
    log_to_both("--------------------------------------------------")

    with open(log_filename, 'w') as f:
        for log_line in current_training_logs:
            f.write(log_line + '\n')
    flash(f"Log pelatihan disimpan ke: {os.path.basename(log_filename)}", 'info')
    
    return render_template('train.html',
                           ticker=ticker,
                           epochs=EPOCHS,
                           learning_rate=LEARNING_RATE,
                           final_loss=final_loss,
                           final_val_loss=final_val_loss,
                           training_duration=training_duration,
                           losses=losses,
                           val_losses=val_losses,
                           dropout_rate=DROPOUT_RATE)


@app.route('/predict', methods=['GET'])
@admin_required
def predict_price():
    ticker = config.DEFAULT_TICKER 
    X_test_samples = app.config.get(f'X_val_{ticker}')
    y_true_scaled_test = app.config.get(f'y_val_{ticker}')

    gru_model = app.config.get('gru_model')
    scaler = app.config.get('scaler_all_features')
    lookback_window = app.config.get('lookback_window')
    features_to_scale = app.config.get('features_to_scale')
    df_full_preprocessed = app.config.get('df_full_preprocessed')
    
    HIDDEN_DIM = app.config.get('hidden_dim')
    INPUT_DIM = app.config.get('input_dim')
    LEARNING_RATE = app.config.get('learning_rate')
    DROPOUT_RATE = app.config.get('dropout_rate') 

    if X_test_samples is None or y_true_scaled_test is None or gru_model is None or scaler is None or features_to_scale is None or df_full_preprocessed is None:
        flash("Data, model, atau scaler belum disiapkan atau dilatih. Silakan kunjungan halaman preprocess dan latih model terlebih dahulu.", 'error')
        return redirect(url_for('train_model'))

    test_size = X_test_samples.shape[0]

    predictions_scaled = []

    gru_model.set_training_mode(False) 
    BATCH_SIZE = config.DEFAULT_BATCH_SIZE 
    num_test_batches = int(np.ceil(X_test_samples.shape[0] / BATCH_SIZE))

    for i in range(num_test_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, X_test_samples.shape[0])

        x_batch_test = X_test_samples[start_idx:end_idx]
        y_pred_batch_scaled = gru_model.forward(x_batch_test)
        predictions_scaled.extend(y_pred_batch_scaled.flatten()) 

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
    mae = mae_loss(predictions_original, y_true_original) 
    mape = mape_loss(predictions_original, y_true_original) 

    train_size_sequences = app.config.get('train_size_sequences')
    start_index_for_val_dates_in_df = train_size_sequences + lookback_window
    predicted_dates = df_full_preprocessed.index[start_index_for_val_dates_in_df : start_index_for_val_dates_in_df + test_size]


    prediction_results = []
    for i in range(test_size):
        prediction_results.append({
            'Date': predicted_dates[i].strftime('%Y-%m-%d'),
            'True Price': f"{y_true_original[i, 0]:.2f}",
            'Predicted Price': f"{predictions_original[i, 0]:.2f}"
        })

    full_prediction_data = [\
        {'Date': predicted_dates[i].strftime('%Y-%m-%d'),\
         'True Price': y_true_original[i, 0], \
         'Predicted Price': predictions_original[i, 0]} \
        for i in range(test_size)\
    ]
    
    full_prediction_data_for_db = [
        {'date': predicted_dates[i], 
         'true_price': float(y_true_original[i, 0]),
         'predicted_price': float(predictions_original[i, 0])}
        for i in range(test_size)
    ]

    display_results = prediction_results[-20:]

    flash(f"Prediksi berhasil dilakukan untuk {test_size} hari terakhir. RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%", 'success')

    try:
        lr_str = str(LEARNING_RATE).replace('.', '')
        do_str = str(DROPOUT_RATE).replace('.', '')
        model_filepath = os.path.join(MODEL_DIR, f'GRU_weights_{ticker}_H{HIDDEN_DIM}_L{lookback_window}_F{INPUT_DIM}_LR{lr_str}_DO{do_str}.npz')

        saved_model_entry = SavedModel.query.filter_by(model_filepath=model_filepath).first()

        if saved_model_entry:
            saved_model_entry.rmse = rmse
            saved_model_entry.mae = mae
            saved_model_entry.mape = mape
            saved_model_entry.training_timestamp = datetime.now() 
            db.session.add(saved_model_entry)
            flash(f"Metrik model {os.path.basename(model_filepath)} berhasil diperbarui di database.", 'info')

            PredictionDetail.query.filter_by(model_id=saved_model_entry.id).delete()
            flash("Prediksi lama untuk model ini telah dihapus.", 'info')

        else:
            saved_model_entry = SavedModel(
                ticker=ticker,
                lookback_window=lookback_window,
                hidden_dim=HIDDEN_DIM,
                input_dim=INPUT_DIM,
                learning_rate=LEARNING_RATE,
                dropout_rate=DROPOUT_RATE,
                rmse=rmse,
                mae=mae,
                mape=mape,
                model_filepath=model_filepath,
                training_timestamp=datetime.now()
            )
            db.session.add(saved_model_entry)
            db.session.flush() 
            flash(f"Model {os.path.basename(model_filepath)} berhasil disimpan ke database.", 'success')
        
        for pred_data in full_prediction_data_for_db:
            prediction_detail = PredictionDetail(
                model_id=saved_model_entry.id,
                prediction_date=pred_data['date'],
                true_price=pred_data['true_price'],
                predicted_price=pred_data['predicted_price']
            )
            db.session.add(prediction_detail)
        
        db.session.commit()
        flash("Detail prediksi berhasil disimpan ke database.", 'success')

    except Exception as e:
        db.session.rollback()
        flash(f"Gagal menyimpan model atau prediksi ke database: {e}", 'error')
        print(f"ERROR: Gagal menyimpan model atau prediksi ke database: {e}")
        traceback.print_exc()
    
    return render_template('predict.html',
                           ticker=ticker,
                           rmse=rmse,
                           mae=mae, 
                           mape=mape, 
                           prediction_results=display_results,
                           total_predictions=test_size,
                           full_prediction_data=full_prediction_data)

@app.route('/predict_future', methods=['GET'])
@admin_required 
def predict_future():
    ticker = config.DEFAULT_TICKER
    lookback_window = app.config.get('lookback_window', config.DEFAULT_LOOKBACK_WINDOW)
    gru_model = app.config.get('gru_model')
    scaler = app.config.get('scaler_all_features')
    features_to_scale = app.config.get('features_to_scale') 
    df_full_preprocessed = app.config.get('df_full_preprocessed') 

    if gru_model is None or scaler is None or df_full_preprocessed is None or features_to_scale is None:
        flash("Model atau data belum disiapkan. Silakan proses data dan latih model terlebih dahulu.", 'error')
        return redirect(url_for('preprocess', ticker=ticker))

    if len(df_full_preprocessed) < lookback_window:
        flash(f"Tidak cukup data historis ({len(df_full_preprocessed)} hari). Diperlukan setidaknya {lookback_window} hari untuk lookback window.", 'error')
        return redirect(url_for('index', ticker=ticker))

    FUTURE_PREDICTION_DAYS = config.DEFAULT_FUTURE_PREDICTION_DAYS 

    # 1. Ambil lookback_window data TERAKHIR dari DataFrame yang sudah diproses
    #    Ini akan menjadi input awal untuk prediksi masa depan.
    last_historical_data_for_sequence_df = df_full_preprocessed.iloc[-lookback_window:]
    
    # 2. Transformasi lookback_window data TERAKHIR ke skala (ini akan menjadi current_sequence awal)
    current_sequence_scaled = scaler.transform(last_historical_data_for_sequence_df[features_to_scale].values)
    
    # 3. Ambil baris data AKTUAL TERAKHIR dari DataFrame asli (original scale)
    last_actual_row_original = df_full_preprocessed.iloc[-1][features_to_scale].values.copy()
    
    adj_close_index = features_to_scale.index('Adj Close')

    future_predictions_original = [] 
    future_dates = [] 
    
    current_sequence = current_sequence_scaled.copy() 

    gru_model.set_training_mode(False) 

    last_historical_date_obj = df_full_preprocessed.index[-1] 

    for i in range(FUTURE_PREDICTION_DAYS):
        next_prediction_date = last_historical_date_obj + timedelta(days=i + 1)
        future_dates.append(next_prediction_date.strftime('%Y-%m-%d'))

        input_for_prediction = current_sequence.reshape(1, lookback_window, len(features_to_scale))

        predicted_scaled_adj_close = gru_model.forward(input_for_prediction)[0, 0]

        dummy_row_for_inverse = np.zeros((1, len(features_to_scale)))
        dummy_row_for_inverse[0, adj_close_index] = predicted_scaled_adj_close
        predicted_original_adj_close = scaler.inverse_transform(dummy_row_for_inverse)[0, adj_close_index]
        future_predictions_original.append(predicted_original_adj_close)

        next_original_row_features = last_actual_row_original.copy() 
        
        next_original_row_features[adj_close_index] = predicted_original_adj_close
        
        open_idx = features_to_scale.index('Open')
        high_idx = features_to_scale.index('High')
        low_idx = features_to_scale.index('Low')
        close_idx = features_to_scale.index('Close')

        next_original_row_features[open_idx] = predicted_original_adj_close * (1 + np.random.uniform(-0.005, 0.005)) 
        next_original_row_features[high_idx] = max(predicted_original_adj_close, predicted_original_adj_close * (1 + np.random.uniform(0.001, 0.01))) 
        next_original_row_features[low_idx] = min(predicted_original_adj_close, predicted_original_adj_close * (1 - np.random.uniform(0.001, 0.01))) 
        next_original_row_features[close_idx] = predicted_original_adj_close 
        
        volume_idx = features_to_scale.index('Volume')
        next_original_row_features[volume_idx] = df_full_preprocessed.iloc[-1]['Volume'] * (1 + np.random.uniform(-0.1, 0.1))
        next_original_row_features[volume_idx] = max(0, next_original_row_features[volume_idx])
            
        next_input_row_scaled = scaler.transform(next_original_row_features.reshape(1, -1))[0]
        
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = next_input_row_scaled

    future_predictions_formatted = []
    for i, pred_price in enumerate(future_predictions_original):
        future_predictions_formatted.append({
            'Date': future_dates[i],
            'Predicted Price': f"{pred_price:.2f}"
        })

    historical_for_chart_df = df_full_preprocessed.iloc[-lookback_window:] 
    
    historical_for_chart = []
    for date, row in historical_for_chart_df.iterrows():
        historical_for_chart.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Price': row['Adj Close'] 
        })
    
    future_for_chart = []
    for i, pred_price_orig in enumerate(future_predictions_original):
        future_for_chart.append({
            'Date': future_dates[i],
            'Price': pred_price_orig 
        })

    flash(f"Prediksi harga saham {ticker} untuk {config.DEFAULT_FUTURE_PREDICTION_DAYS} hari ke depan berhasil dilakukan.", 'success')

    return render_template('predict_future.html',
                           ticker=ticker,
                           future_predictions=future_predictions_formatted, 
                           future_prediction_days=config.DEFAULT_FUTURE_PREDICTION_DAYS,
                           historical_for_chart=historical_for_chart, 
                           future_for_chart=future_for_chart)
    
@app.route('/admin/models')
@admin_required
def admin_models_manage():
    all_saved_models = SavedModel.query.order_by(SavedModel.training_timestamp.desc()).all()
    return render_template('admin_models.html', models=all_saved_models)

@app.route('/admin/dashboard') 
@admin_required
def admin_dashboard():
    total_stock_records = StockData.query.count()

    unique_tickers = db.session.query(StockData.ticker).distinct().count()

    total_saved_models = SavedModel.query.count()

    best_model_by_rmse = SavedModel.query.order_by(SavedModel.rmse.asc()).first()
    
    latest_models = SavedModel.query.order_by(SavedModel.training_timestamp.desc()).limit(5).all()


    return render_template('admin_dashboard.html',
                           total_stock_records=total_stock_records,
                           unique_tickers=unique_tickers,
                           total_saved_models=total_saved_models,
                           best_model_by_rmse=best_model_by_rmse,
                           latest_models=latest_models)
    
@app.route('/dashboard_user')
@login_required 
def dashboard_user():
    available_models = SavedModel.query.filter_by(ticker=config.DEFAULT_TICKER).order_by(SavedModel.training_timestamp.desc()).all()
    
    model_choices = []
    if available_models:
        for model in available_models:
            choice_text = (
                f"TSLA - L{model.lookback_window} HD{model.hidden_dim} "
                f"LR{model.learning_rate} DO{model.dropout_rate} "
                f"(RMSE: {model.rmse:.2f} MAPE: {model.mape:.2f}%) - {model.training_timestamp.strftime('%Y-%m-%d %H:%M')}"
            )
            model_choices.append({'id': model.id, 'text': choice_text})
    else:
        flash("Belum ada model prediksi TSLA yang tersedia di database.", 'info')

    flash("Selamat datang di Dashboard Pengguna! Pilih model prediksi yang ingin Anda lihat.", 'info')
    return render_template('user_dashboard.html', model_choices=model_choices)

@app.route('/free_user_predict_view', methods=['GET'])
@login_required
def free_user_predict_view():
    model_id = request.args.get('model_id', type=int) 

    if not model_id:
        flash("Pilih model prediksi dari daftar di bawah.", 'info')
        return redirect(url_for('dashboard_user')) 

    saved_model = SavedModel.query.get(model_id)

    if not saved_model:
        flash("Model prediksi yang dipilih tidak ditemukan.", 'error')
        return redirect(url_for('dashboard_user'))

    prediction_details = PredictionDetail.query.filter_by(model_id=saved_model.id).order_by(PredictionDetail.prediction_date.asc()).all()

    if not prediction_details:
        flash(f"Tidak ada detail prediksi untuk model ini ({saved_model.id}).", 'warning')
        return redirect(url_for('dashboard_user'))

    prediction_results = [] 
    full_prediction_data = [] 

    for detail in prediction_details:
        prediction_results.append({
            'Date': detail.prediction_date.strftime('%Y-%m-%d'),
            'True Price': f"{detail.true_price:.2f}",
            'Predicted Price': f"{detail.predicted_price:.2f}"
        })
        full_prediction_data.append({
            'Date': detail.prediction_date.strftime('%Y-%m-%d'),
            'True Price': detail.true_price,
            'Predicted Price': detail.predicted_price
        })

    rmse = saved_model.rmse
    mae = saved_model.mae
    mape = saved_model.mape

    display_results = prediction_results[-20:]
    total_predictions = len(prediction_details)

    flash(f"Hasil prediksi untuk model {saved_model.ticker} (LR: {saved_model.learning_rate}, DO: {saved_model.dropout_rate}) dimuat.", 'success')

    return render_template('free_user_predict.html',
                           ticker=saved_model.ticker,
                           rmse=rmse,
                           mae=mae,
                           mape=mape,
                           prediction_results=display_results,
                           full_prediction_data=full_prediction_data,
                           total_predictions=total_predictions,
                           selected_model_text=saved_model.training_timestamp.strftime('%Y-%m-%d %H:%M'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            hashed_password = generate_password_hash('adminpass')
            admin_user = User(username='admin', password_hash=hashed_password, role='admin')
            db.session.add(admin_user)
            db.session.commit()
            print("Pengguna admin default 'admin' dengan password 'adminpass' telah dibuat.")
    app.run(debug=True)