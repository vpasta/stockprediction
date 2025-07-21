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
from utils.db_models import db, StockData, User, SavedModel, PredictionDetail, FuturePrediction
from utils.gru_model import GRU
from utils.data_preprocessing import preprocess_data
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

# --- FUNGSI LOGIN DAN OTORISASI (Tidak Berubah) ---
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
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('dashboard_user'))
        else:
            flash('Username atau password salah.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Anda telah berhasil logout.', 'info')
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@admin_required
def index():
    default_ticker = session.get('current_ticker', config.DEFAULT_TICKER)
    try:
        distinct_tickers_query = db.session.query(StockData.ticker).distinct().all()
        available_tickers = [item[0] for item in distinct_tickers_query]
    except Exception as e:
        flash("Gagal mengambil daftar ticker dari database.", "error")
        available_tickers = []
    default_end_date = datetime.now().strftime('%Y-%m-%d')
    default_start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    default_epochs = config.DEFAULT_EPOCHS
    default_learning_rate = config.DEFAULT_LEARNING_RATE
    default_dropout_rate = config.DEFAULT_DROPOUT_RATE
    default_patience = config.EARLY_STOPPING_PATIENCE

    if request.method == 'POST':
        ticker = request.form.get('ticker', default_ticker).upper()
        session['current_ticker'] = ticker
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
            flash("Input parameter pelatihan tidak valid.", 'error')
            return redirect(url_for('index'))
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            if start_date >= end_date:
                flash("Tanggal mulai harus sebelum tanggal selesai.", 'error')
                return redirect(url_for('index'))
        except ValueError:
            flash("Format tanggal tidak valid. Gunakan YYYY-MM-DD.", 'error')
            return redirect(url_for('index'))
        try:
            data_yf = yf.download(ticker, start=start_date, end=end_date)
            if data_yf.empty:
                flash(f"Tidak ada data ditemukan untuk {ticker}.", 'error')
                return redirect(url_for('index'))
            new_records_count = 0
            for index, row in data_yf.iterrows():
                record_date = index.date()
                if not StockData.query.filter_by(ticker=ticker, date=record_date).first():
                    stock_data = StockData(
                        ticker=ticker, date=record_date,
                        open_price=float(row['Open']), high_price=float(row['High']),
                        low_price=float(row['Low']), close_price=float(row['Close']),
                        adj_close_price=float(row['Adj Close']), volume=int(row['Volume'])
                    )
                    db.session.add(stock_data)
                    new_records_count += 1
            db.session.commit()
            flash(f"Data {ticker} berhasil diunduh, {new_records_count} record baru disimpan.", 'success')
            return redirect(url_for('index'))
        except Exception as e:
            db.session.rollback()
            flash(f"Terjadi kesalahan saat mengambil atau menyimpan data: {e}", 'error')
            return redirect(url_for('index'))
            
    return render_template('index.html',
                           available_tickers=available_tickers, default_ticker=default_ticker,
                           default_start_date=default_start_date, default_end_date=default_end_date,
                           default_epochs=default_epochs, default_learning_rate=default_learning_rate,
                           default_dropout_rate=default_dropout_rate, default_patience=default_patience)

@app.route('/process_existing', methods=['POST'])
@admin_required
def process_existing():
    ticker_to_process = request.form.get('ticker_to_process')
    if ticker_to_process:
        session['current_ticker'] = ticker_to_process
        flash(f"Ticker aktif sekarang adalah {ticker_to_process}. Anda bisa memulai preprocessing.", 'info')
        return redirect(url_for('preprocess'))
    else:
        flash("Anda harus memilih ticker untuk diproses.", 'error')
        return redirect(url_for('index'))

@app.route('/delete_model/<int:model_id>', methods=['POST'])
@admin_required
def delete_model(model_id):
    model_to_delete = SavedModel.query.get_or_404(model_id)
    try:
        if os.path.exists(model_to_delete.model_filepath):
            os.remove(model_to_delete.model_filepath)
        db.session.delete(model_to_delete)
        db.session.commit()
        flash(f'Model ID {model_id} ({model_to_delete.ticker}) berhasil dihapus.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Gagal menghapus model: {e}', 'error')
    return redirect(url_for('admin_models_manage'))

@app.route('/delete_stock_data/<string:ticker>', methods=['POST'])
@admin_required
def delete_stock_data(ticker):
    try:
        models_to_delete = SavedModel.query.filter_by(ticker=ticker).all()
        for model in models_to_delete:
            if os.path.exists(model.model_filepath):
                os.remove(model.model_filepath)
            db.session.delete(model)
        StockData.query.filter_by(ticker=ticker).delete()
        db.session.commit()
        flash(f'Semua data dan model untuk ticker {ticker} telah dihapus.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Gagal menghapus data untuk {ticker}: {e}', 'error')
    return redirect(url_for('index'))

@app.route('/preprocess', methods=['GET', 'POST'])
@admin_required
def preprocess():
    ticker = session.get('current_ticker', config.DEFAULT_TICKER)
    stock_records = StockData.query.filter_by(ticker=ticker).order_by(StockData.date.asc()).all()
    if not stock_records:
        flash(f"Tidak ada data untuk {ticker}. Silakan unduh dulu.", 'error')
        return render_template('preprocess.html', current_ticker=ticker)
    df = pd.DataFrame(
        [{'Date': r.date, 'Adj Close': r.adj_close_price} for r in stock_records]
    ).set_index('Date')
    LOOKBACK_WINDOW = config.DEFAULT_LOOKBACK_WINDOW
    X, y, scaler = preprocess_data(df, LOOKBACK_WINDOW)
    TRAIN_SPLIT_RATIO = config.DEFAULT_TRAIN_SPLIT_RATIO
    train_size = int(len(X) * TRAIN_SPLIT_RATIO)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    INPUT_DIM = 1
    HIDDEN_DIM = config.DEFAULT_HIDDEN_DIM
    OUTPUT_DIM = 1
    EPOCHS = app.config.get('current_epochs', config.DEFAULT_EPOCHS)
    LEARNING_RATE = app.config.get('current_learning_rate', config.DEFAULT_LEARNING_RATE)
    DROPOUT_RATE = app.config.get('current_dropout_rate', config.DEFAULT_DROPOUT_RATE)
    EARLY_STOPPING_PATIENCE = app.config.get('current_patience', config.EARLY_STOPPING_PATIENCE)
    gru_model = GRU(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout_rate=DROPOUT_RATE)
    flash("Menginisialisasi model GRU baru dengan 1 fitur input ('Adj Close').", 'info')
    app.config['gru_model'] = gru_model
    app.config['scaler'] = scaler
    app.config['df_original'] = df
    app.config['X_train'] = X_train
    app.config['y_train'] = y_train
    app.config['X_val'] = X_val
    app.config['y_val'] = y_val
    app.config['train_size'] = train_size
    app.config['lookback_window'] = LOOKBACK_WINDOW
    app.config['input_dim'] = INPUT_DIM
    app.config['hidden_dim'] = HIDDEN_DIM
    app.config['learning_rate'] = LEARNING_RATE
    app.config['dropout_rate'] = DROPOUT_RATE
    app.config['epochs'] = EPOCHS
    app.config['early_stopping_patience'] = EARLY_STOPPING_PATIENCE
    flash(f"Data {ticker} berhasil diproses. Total {len(df)} record. Train: {X_train.shape}, Val: {X_val.shape}. Model GRU diinisialisasi dengan input_dim={INPUT_DIM}.", 'success')
    head_html = df[['Adj Close']].head(10).to_html(classes='table table-bordered table-striped table-sm')
    tail_html = df[['Adj Close']].tail(10).to_html(classes='table table-bordered table-striped table-sm')
    display_df = f"{head_html}<br><br>{tail_html}"
    return render_template('preprocess.html',
                           data_html=display_df, current_ticker=ticker, total_records=len(df),
                           show_preprocess_results=True, lookback_window=LOOKBACK_WINDOW,
                           learning_rate=LEARNING_RATE, epochs=EPOCHS,
                           early_stopping_patience=EARLY_STOPPING_PATIENCE,
                           X_shape_train=X_train.shape, y_shape_train=y_train.shape,
                           X_shape_val=X_val.shape, y_shape_val=y_val.shape,
                           hidden_dim=HIDDEN_DIM, input_dim=INPUT_DIM, dropout_rate=DROPOUT_RATE)

@app.route('/train', methods=['GET'])
@admin_required
def train_model():
    ticker = session.get('current_ticker')
    X_train, y_train = app.config.get('X_train'), app.config.get('y_train')
    X_val, y_val = app.config.get('X_val'), app.config.get('y_val')
    gru_model = app.config.get('gru_model')
    lookback_window = app.config.get('lookback_window')
    INPUT_DIM = app.config.get('input_dim')
    HIDDEN_DIM = app.config.get('hidden_dim')
    LEARNING_RATE = app.config.get('learning_rate')
    DROPOUT_RATE = app.config.get('dropout_rate')
    EPOCHS = app.config.get('epochs')
    PATIENCE = app.config.get('early_stopping_patience')
    if X_train is None or gru_model is None:
        flash("Data atau model belum disiapkan. Lakukan preprocessing dulu.", 'error')
        return redirect(url_for('preprocess'))
    BATCH_SIZE = config.DEFAULT_BATCH_SIZE
    losses, val_losses = [], []
    start_time = time.time()
    best_val_loss, patience_counter, best_model_weights = float('inf'), 0, None
    lr_str = str(LEARNING_RATE).replace('.', '')
    do_str = str(DROPOUT_RATE).replace('.', '')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'train_log_{ticker}_F{INPUT_DIM}_H{HIDDEN_DIM}_LR{lr_str}_DO{do_str}_{timestamp}.txt')
    current_training_logs = []
    def log_and_store(message):
        print(message)
        current_training_logs.append(message)
    log_and_store(f"--- Memulai pelatihan untuk {ticker} ---")
    log_and_store(f"Parameter: Lookback={lookback_window}, Hidden={HIDDEN_DIM}, LR={LEARNING_RATE}, Dropout={DROPOUT_RATE}, Epochs={EPOCHS}")
    log_and_store(f"Tanggal dan Waktu Pelatihan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_and_store("==================================================")
    log_and_store("Parameter Konfigurasi Model:")
    log_and_store(f"- Ticker Saham: {ticker}")
    log_and_store(f"- Lookback Window: {lookback_window} hari")
    log_and_store(f"- Dimensi Tersembunyi (Hidden Dim): {HIDDEN_DIM}")
    log_and_store(f"- Jumlah Fitur Input: {INPUT_DIM}") 
    log_and_store(f"- Epochs Maksimal: {EPOCHS}")
    log_and_store(f"- Learning Rate: {LEARNING_RATE}")
    log_and_store(f"- Dropout Rate: {DROPOUT_RATE}")
    log_and_store(f"- Batch Size: {BATCH_SIZE}")
    log_and_store(f"- Rasio Pembagian Data (Train/Val): {config.DEFAULT_TRAIN_SPLIT_RATIO*100:.0f}% / {(1 - config.DEFAULT_TRAIN_SPLIT_RATIO)*100:.0f}%")
    log_and_store(f"- Early Stopping Patience: {PATIENCE} Epoch")
    log_and_store("==================================================\n")
    try:
        for epoch in range(EPOCHS):
            total_loss = 0
            gru_model.set_training_mode(True)
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled, y_train_shuffled = X_train[permutation], y_train[permutation]
            for i in range(0, X_train.shape[0], BATCH_SIZE):
                x_batch = X_train_shuffled[i:i+BATCH_SIZE]
                y_batch = y_train_shuffled[i:i+BATCH_SIZE].reshape(-1, 1)
                y_pred = gru_model.forward(x_batch)
                loss = mse_loss(y_pred, y_batch)
                total_loss += loss
                d_output = mse_loss_derivative(y_pred, y_batch)
                gru_model.backward(d_output)
                gru_model.update_weights(LEARNING_RATE)
            avg_loss = total_loss / (X_train.shape[0] / BATCH_SIZE)
            losses.append(avg_loss)
            gru_model.set_training_mode(False)
            y_val_pred = gru_model.forward(X_val)
            avg_val_loss = mse_loss(y_val_pred, y_val.reshape(-1, 1))
            val_losses.append(avg_val_loss)
            log_and_store(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            if avg_val_loss < best_val_loss:
                best_val_loss, patience_counter = avg_val_loss, 0
                best_model_weights = gru_model.get_weights()
                log_and_store("Validation loss meningkat. Menyimpan bobot model terbaik.")
            else:
                patience_counter += 1
                log_and_store(f"Validation loss tidak meningkat. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                log_and_store(f"Early stopping dipicu pada epoch {epoch + 1}.")
                break
        log_and_store("--- Pelatihan selesai ---")
    except Exception as e:
        flash(f"Error saat pelatihan: {e}", 'error')
        log_and_store(f"--- ERROR: {e} ---")
        traceback.print_exc(file=sys.stdout)
        return redirect(url_for('preprocess'))
    training_duration = time.time() - start_time
    if best_model_weights:
        gru_model.set_weights(best_model_weights)
    model_filename = os.path.join(MODEL_DIR, f'GRU_weights_{ticker}_F{INPUT_DIM}_H{HIDDEN_DIM}_L{lookback_window}_LR{lr_str}_DO{do_str}.npz')
    np.savez(model_filename, **gru_model.get_weights())
    
    final_loss = losses[-1] if losses else 0
    final_val_loss = val_losses[-1] if val_losses else 0
    if best_model_weights is not None and best_val_loss != float('inf'):
        final_val_loss = best_val_loss 
    
    log_and_store(f"\n--- Ringkasan Pelatihan Akhir ---")
    log_and_store(f"Durasi Pelatihan Total: {training_duration:.2f} detik")
    log_and_store(f"Final Train Loss (Epoch terakhir): {final_loss:.6f}")
    log_and_store(f"Final Val Loss (Best): {final_val_loss:.6f}")
    log_and_store(f"Model Terbaik Disimpan: {os.path.basename(model_filename)}")
    log_and_store("--------------------------------------------------")
    
    with open(log_filename, 'w') as f:
        f.write('\n'.join(current_training_logs))
    flash(f"Log pelatihan disimpan ke: {os.path.basename(log_filename)}", 'info')
    return render_template('train.html',
                           ticker=ticker, epochs=EPOCHS, learning_rate=LEARNING_RATE,
                           final_loss=final_loss,
                           final_val_loss=final_val_loss,
                           training_duration=training_duration, losses=losses, val_losses=val_losses,
                           dropout_rate=DROPOUT_RATE)

@app.route('/predict', methods=['GET'])
@admin_required
def predict_price():
    ticker = session.get('current_ticker')
    X_val, y_val = app.config.get('X_val'), app.config.get('y_val')
    gru_model = app.config.get('gru_model')
    scaler = app.config.get('scaler')
    df_original = app.config.get('df_original')
    train_size = app.config.get('train_size')
    lookback_window = app.config.get('lookback_window')
    if X_val is None or gru_model is None or scaler is None:
        flash("Data, model, atau scaler belum disiapkan. Lakukan preprocessing dulu.", 'error')
        return redirect(url_for('preprocess'))
    gru_model.set_training_mode(False)
    predictions_scaled = gru_model.forward(X_val)
    predictions_original = scaler.inverse_transform(predictions_scaled)
    y_true_original = scaler.inverse_transform(y_val.reshape(-1, 1))
    rmse = np.sqrt(np.mean((predictions_original - y_true_original)**2))
    mae = mae_loss(predictions_original, y_true_original)
    mape = mape_loss(predictions_original, y_true_original)
    val_dates = df_original.index[train_size + lookback_window:]
    prediction_results = [{'Date': val_dates[i].strftime('%Y-%m-%d'), 'True Price': f"{y_true_original[i, 0]:.2f}", 'Predicted Price': f"{predictions_original[i, 0]:.2f}"} for i in range(len(predictions_original))]
    full_prediction_data = [{'Date': val_dates[i].strftime('%Y-%m-%d'), 'True Price': float(y_true_original[i, 0]), 'Predicted Price': float(predictions_original[i, 0])} for i in range(len(predictions_original))]
    flash(f"Prediksi pada data validasi selesai. RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%", 'success')
    try:
        INPUT_DIM = app.config.get('input_dim')
        HIDDEN_DIM = app.config.get('hidden_dim')
        LEARNING_RATE = app.config.get('learning_rate')
        DROPOUT_RATE = app.config.get('dropout_rate')
        lr_str = str(LEARNING_RATE).replace('.', '')
        do_str = str(DROPOUT_RATE).replace('.', '')
        model_filepath = os.path.join(MODEL_DIR, f'GRU_weights_{ticker}_F{INPUT_DIM}_H{HIDDEN_DIM}_L{lookback_window}_LR{lr_str}_DO{do_str}.npz')
        saved_model = SavedModel.query.filter_by(model_filepath=model_filepath).first()
        if not saved_model:
            saved_model = SavedModel(
                ticker=ticker, lookback_window=lookback_window,
                hidden_dim=HIDDEN_DIM, input_dim=INPUT_DIM,
                learning_rate=LEARNING_RATE, dropout_rate=DROPOUT_RATE,
                model_filepath=model_filepath
            )
            db.session.add(saved_model)
        saved_model.rmse, saved_model.mae, saved_model.mape = float(rmse), float(mae), float(mape)
        saved_model.training_timestamp = datetime.now()
        PredictionDetail.query.filter_by(model_id=saved_model.id).delete()
        db.session.flush()
        for i in range(len(full_prediction_data)):
            pred_detail = PredictionDetail(
                model_id=saved_model.id,
                prediction_date=val_dates[i],
                true_price=float(y_true_original[i, 0]),
                predicted_price=float(predictions_original[i, 0])
            )
            db.session.add(pred_detail)
        db.session.commit()
        flash("Hasil prediksi dan metrik model berhasil disimpan ke database.", 'success')
    except Exception as e:
        db.session.rollback()
        flash(f"Gagal menyimpan ke database: {e}", 'error')
        traceback.print_exc()
    return render_template('predict.html',
                           ticker=ticker, rmse=rmse, mae=mae, mape=mape,
                           prediction_results=prediction_results[-20:],
                           total_predictions=len(prediction_results),
                           full_prediction_data=full_prediction_data)

@app.route('/predict_future', methods=['GET'])
@admin_required
def predict_future():
    ticker = session.get('current_ticker')
    gru_model = app.config.get('gru_model')
    scaler = app.config.get('scaler')
    df_original = app.config.get('df_original')
    lookback_window = app.config.get('lookback_window')
    if gru_model is None or scaler is None or df_original is None:
        flash("State tidak lengkap. Lakukan preprocessing dulu.", 'error')
        return redirect(url_for('preprocess'))
    FUTURE_PREDICTION_DAYS = config.DEFAULT_FUTURE_PREDICTION_DAYS
    gru_model.set_training_mode(False)
    last_sequence_original = df_original[['Adj Close']].values[-lookback_window:]
    current_sequence_scaled = scaler.transform(last_sequence_original)
    future_predictions_np = []
    for _ in range(FUTURE_PREDICTION_DAYS):
        input_for_prediction = current_sequence_scaled.reshape(1, lookback_window, 1)
        predicted_scaled = gru_model.forward(input_for_prediction)
        future_predictions_np.append(scaler.inverse_transform(predicted_scaled)[0, 0])
        current_sequence_scaled = np.append(current_sequence_scaled[1:], predicted_scaled, axis=0)
    future_predictions = [float(p) for p in future_predictions_np]
    last_historical_date = df_original.index[-1]
    future_dates = [last_historical_date + timedelta(days=i+1) for i in range(FUTURE_PREDICTION_DAYS)]
    future_predictions_formatted = [{'Date': date.strftime('%Y-%m-%d'), 'Predicted Price': f"{price:.2f}"} for date, price in zip(future_dates, future_predictions)]
    try:
        INPUT_DIM = app.config.get('input_dim')
        HIDDEN_DIM = app.config.get('hidden_dim')
        LEARNING_RATE = app.config.get('learning_rate')
        DROPOUT_RATE = app.config.get('dropout_rate')
        lr_str = str(LEARNING_RATE).replace('.', '')
        do_str = str(DROPOUT_RATE).replace('.', '')
        model_filepath = os.path.join(MODEL_DIR, f'GRU_weights_{ticker}_F{INPUT_DIM}_H{HIDDEN_DIM}_L{lookback_window}_LR{lr_str}_DO{do_str}.npz')
        saved_model = SavedModel.query.filter_by(model_filepath=model_filepath).first()
        if saved_model:
            FuturePrediction.query.filter_by(model_id=saved_model.id).delete()
            for i in range(len(future_predictions)):
                future_pred_entry = FuturePrediction(
                    model_id=saved_model.id,
                    forecast_date=future_dates[i],
                    predicted_price=float(future_predictions[i])
                )
                db.session.add(future_pred_entry)
            db.session.commit()
            flash(f"Prediksi masa depan untuk model ID {saved_model.id} berhasil disimpan ke database.", 'success')
        else:
            flash("Model yang sesuai tidak ditemukan di database. Prediksi masa depan tidak disimpan.", 'warning')
    except Exception as e:
        db.session.rollback()
        flash(f"Gagal menyimpan prediksi masa depan ke database: {e}", "error")
        traceback.print_exc()
    historical_for_chart_df = df_original.iloc[-lookback_window:]
    historical_for_chart = [{'Date': date.strftime('%Y-%m-%d'), 'Price': float(row['Adj Close'])} for date, row in historical_for_chart_df.iterrows()]
    return render_template('predict_future.html',
                           ticker=ticker,
                           future_predictions=future_predictions_formatted,
                           future_prediction_days=FUTURE_PREDICTION_DAYS,
                           historical_for_chart=historical_for_chart,
                           avg_predictions=future_predictions)

@app.route('/load_model/<int:model_id>')
@admin_required
def load_model(model_id):
    saved_model = SavedModel.query.get_or_404(model_id)

    if saved_model.input_dim != 1:
        flash(f"Model ID {model_id} tidak kompatibel. Model ini dilatih dengan {saved_model.input_dim} fitur, sedangkan sistem saat ini hanya mendukung 1 fitur.", 'error')
        return redirect(url_for('admin_models_manage'))

    try:
        gru_model = GRU(input_dim=saved_model.input_dim, hidden_dim=saved_model.hidden_dim, output_dim=1, dropout_rate=saved_model.dropout_rate)
        
        weights_npz = np.load(saved_model.model_filepath, allow_pickle=True)
        weights_dict = {key: weights_npz[key] for key in weights_npz.files}
        gru_model.set_weights(weights_dict)

        stock_records = StockData.query.filter_by(ticker=saved_model.ticker).order_by(StockData.date.asc()).all()
        df = pd.DataFrame([{'Date': r.date, 'Adj Close': r.adj_close_price} for r in stock_records]).set_index('Date')
        
        X, y, scaler = preprocess_data(df, saved_model.lookback_window)
        
        train_size = int(len(X) * config.DEFAULT_TRAIN_SPLIT_RATIO)
        X_val, y_val = X[train_size:], y[train_size:]

        session['current_ticker'] = saved_model.ticker
        app.config['gru_model'] = gru_model
        app.config['scaler'] = scaler
        app.config['df_original'] = df
        app.config['X_val'] = X_val
        app.config['y_val'] = y_val
        app.config['train_size'] = train_size
        app.config['lookback_window'] = saved_model.lookback_window
        app.config['input_dim'] = saved_model.input_dim
        app.config['hidden_dim'] = saved_model.hidden_dim
        app.config['learning_rate'] = saved_model.learning_rate
        app.config['dropout_rate'] = saved_model.dropout_rate
        
        flash(f"Model ID {model_id} ({saved_model.ticker}) berhasil dimuat dan siap untuk prediksi.", 'success')
        return redirect(url_for('predict_price'))

    except Exception as e:
        flash(f"Gagal memuat model: {e}", 'error')
        traceback.print_exc()
        return redirect(url_for('admin_models_manage'))

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
    all_saved_models = SavedModel.query.filter_by(input_dim=1).order_by(SavedModel.training_timestamp.desc()).all()
    models_by_ticker = {}
    for model in all_saved_models:
        if model.ticker not in models_by_ticker:
            models_by_ticker[model.ticker] = []
        models_by_ticker[model.ticker].append({
            "id": model.id, "lookback_window": model.lookback_window,
            "hidden_dim": model.hidden_dim, "learning_rate": model.learning_rate,
            "dropout_rate": model.dropout_rate, "rmse": model.rmse,
            "training_timestamp": model.training_timestamp.isoformat()
        })
    logo_map = config.TICKER_LOGO_MAP
    default_logo_url = 'https://via.placeholder.com/60/1e2a3b/FFFFFF?text=?'
    ticker_logos = {ticker: logo_map.get(ticker, default_logo_url) for ticker in models_by_ticker.keys()}
    return render_template('user_dashboard.html',
                           models_by_ticker=models_by_ticker,
                           ticker_logos=ticker_logos)

@app.route('/free_user_predict_view', methods=['GET'])
@login_required
def free_user_predict_view():
    model_id = request.args.get('model_id', type=int)
    if not model_id:
        flash("Model tidak dipilih.", 'error')
        return redirect(url_for('dashboard_user'))
    saved_model = SavedModel.query.get_or_404(model_id)
    if saved_model.input_dim != 1:
        flash("Model ini tidak dapat ditampilkan untuk pengguna gratis.", 'error')
        return redirect(url_for('dashboard_user'))
    prediction_details = PredictionDetail.query.filter_by(model_id=saved_model.id).order_by(PredictionDetail.prediction_date.asc()).all()
    future_prediction_details = FuturePrediction.query.filter_by(model_id=saved_model.id).order_by(FuturePrediction.forecast_date.asc()).all()
    if not prediction_details:
        flash("Tidak ada detail prediksi historis untuk model ini.", 'warning')
        return redirect(url_for('dashboard_user'))
    full_prediction_data_historical_chart = [{'Date': d.prediction_date.strftime('%Y-%m-%d'), 'True Price': float(d.true_price), 'Predicted Price': float(d.predicted_price)} for d in prediction_details]
    prediction_results_historical_table = [{'Date': d['Date'], 'True Price': f"{d['True Price']:.2f}", 'Predicted Price': f"{d['Predicted Price']:.2f}"} for d in full_prediction_data_historical_chart]
    if future_prediction_details:
        future_predictions = [float(fp.predicted_price) for fp in future_prediction_details]
        future_dates_obj = [fp.forecast_date for fp in future_prediction_details]
        future_dates = [d.strftime('%Y-%m-%d') for d in future_dates_obj]
    else:
        flash("Membuat prediksi masa depan baru dan menyimpannya ke database...", "info")
        try:
            gru_model = GRU(input_dim=1, hidden_dim=saved_model.hidden_dim, output_dim=1, dropout_rate=saved_model.dropout_rate)
            weights_npz = np.load(saved_model.model_filepath, allow_pickle=True)
            weights_dict = {key: weights_npz[key] for key in weights_npz.files}
            gru_model.set_weights(weights_dict)
            stock_records = StockData.query.filter_by(ticker=saved_model.ticker).order_by(StockData.date.asc()).all()
            df = pd.DataFrame([{'Date': r.date, 'Adj Close': r.adj_close_price} for r in stock_records]).set_index('Date')
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(df[['Adj Close']].values)
            FUTURE_PREDICTION_DAYS = config.DEFAULT_FUTURE_PREDICTION_DAYS
            last_sequence_original = df[['Adj Close']].values[-saved_model.lookback_window:]
            current_sequence_scaled = scaler.transform(last_sequence_original)
            future_predictions_np = []
            for _ in range(FUTURE_PREDICTION_DAYS):
                input_for_prediction = current_sequence_scaled.reshape(1, saved_model.lookback_window, 1)
                predicted_scaled = gru_model.forward(input_for_prediction)
                future_predictions_np.append(scaler.inverse_transform(predicted_scaled)[0, 0])
                current_sequence_scaled = np.append(current_sequence_scaled[1:], predicted_scaled, axis=0)
            future_predictions = [float(p) for p in future_predictions_np]
            last_historical_date = df.index[-1]
            future_dates_obj = [last_historical_date + timedelta(days=i+1) for i in range(FUTURE_PREDICTION_DAYS)]
            future_dates = [d.strftime('%Y-%m-%d') for d in future_dates_obj]
            FuturePrediction.query.filter_by(model_id=saved_model.id).delete()
            for i in range(len(future_predictions)):
                db.session.add(FuturePrediction(model_id=saved_model.id, forecast_date=future_dates_obj[i], predicted_price=float(future_predictions[i])))
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            flash(f"Gagal membuat atau menyimpan prediksi masa depan: {e}", 'error')
            traceback.print_exc()
            return redirect(url_for('dashboard_user'))
    future_predictions_formatted = [{'Date': future_dates[i], 'Predicted Price': f"{price:.2f}"} for i, price in enumerate(future_predictions)]
    last_historical_data = PredictionDetail.query.filter_by(model_id=saved_model.id).order_by(PredictionDetail.prediction_date.desc()).limit(saved_model.lookback_window).all()
    historical_for_future_chart = [{'Date': d.prediction_date.strftime('%Y-%m-%d'), 'Price': float(d.true_price)} for d in reversed(last_historical_data)]
    return render_template('free_user_predict.html',
                           ticker=saved_model.ticker, rmse=saved_model.rmse, mae=saved_model.mae, mape=saved_model.mape,
                           prediction_results=prediction_results_historical_table[-20:],
                           total_predictions=len(prediction_details),
                           full_prediction_data=full_prediction_data_historical_chart,
                           selected_model_text=(f"{saved_model.ticker} - F{saved_model.input_dim} L{saved_model.lookback_window} H{saved_model.hidden_dim}"),
                           historical_for_future_chart=historical_for_future_chart,
                           avg_predictions=future_predictions,
                           future_predictions=future_predictions_formatted,
                           future_prediction_days=len(future_predictions))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            hashed_password = generate_password_hash('adminpass')
            admin_user = User(username='admin', password_hash=hashed_password, role='admin')
            db.session.add(admin_user)
            db.session.commit()
            print("Pengguna admin default 'admin' dengan password 'adminpass' telah dibuat.")
    app.run(debug=True, port=5001)
