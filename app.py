from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import time
import traceback
import config
from utils.db_models import db, StockData
from utils.gru_model import GRU
from utils.data_preprocessing import calculate_technical_indicators, create_sequences
from utils.loss_functions import mse_loss, mse_loss_derivative, mae_loss, mape_loss
from sklearn.preprocessing import MinMaxScaler 

app = Flask(__name__)

app.secret_key = config.SECRET_KEY # Gunakan dari config
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+mysqlconnector://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app) 

DATA_DIR = config.DATA_DIR
MODEL_DIR = config.MODEL_DIR

@app.route('/', methods=['GET', 'POST'])
def index():
    default_ticker = config.DEFAULT_TICKER
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
                         'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20', 'RSI']

    for feature in features_to_scale:
        if feature not in df.columns:
            flash(f"Peringatan: Fitur '{feature}' tidak ditemukan di DataFrame. Mungkin ada kesalahan perhitungan indikator.", 'warning')

    data_for_scaling = df[features_to_scale].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_for_scaling)

    app.config['features_to_scale'] = features_to_scale
    app.config['scaler_all_features'] = scaler
    app.config['df_full_preprocessed'] = df.copy()

    LOOKBACK_WINDOW = config.DEFAULT_LOOKBACK_WINDOW # Gunakan dari config

    adj_close_index_for_target = features_to_scale.index('Adj Close')
    X, y = create_sequences(scaled_data, LOOKBACK_WINDOW, adj_close_index_for_target)

    TRAIN_SPLIT_RATIO = config.DEFAULT_TRAIN_SPLIT_RATIO # Gunakan dari config
    total_sequence_samples = X.shape[0]
    train_size_sequences = int(total_sequence_samples * TRAIN_SPLIT_RATIO)

    X_train_data = X[:train_size_sequences]
    y_train_data = y[:train_size_sequences]

    X_val_data = X[train_size_sequences:]
    y_val_data = y[train_size_sequences:]

    INPUT_DIM = len(features_to_scale)
    HIDDEN_DIM = config.DEFAULT_HIDDEN_DIM 
    OUTPUT_DIM = 1
    DROPOUT_RATE = config.DEFAULT_DROPOUT_RATE 

    gru_model = None
    model_filename = os.path.join(MODEL_DIR, f'GRU_weights_{ticker}_H{HIDDEN_DIM}_L{LOOKBACK_WINDOW}_F{INPUT_DIM}_D{int(DROPOUT_RATE*100)}.npz')

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

    flash(f"Data {ticker} telah dimuat dari database dan diproses awal. Total {len(df)} record. "
          f"Sequence data (Train X: {X_train_data.shape}, Train y: {y_train_data.shape}, "
          f"Val X: {X_val_data.shape}, Val y: {y_val_data.shape}) "
          f"dibuat dengan lookback window {LOOKBACK_WINDOW}. "
          f"Model GRU diinisialisasi dengan input_dim {INPUT_DIM}, hidden_dim {HIDDEN_DIM}, dropout_rate {DROPOUT_RATE}.", 'info')

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
                           input_dim=INPUT_DIM,
                           dropout_rate=DROPOUT_RATE,
                           model_loaded_from_cache=model_loaded_from_cache)

@app.route('/train', methods=['GET'])
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
    DROPOUT_RATE = app.config.get('dropout_rate')

    if X_train is None or y_train is None or X_val is None or y_val is None or gru_model is None or scaler is None:
        flash("Data atau model belum disiapkan. Silakan kunjungi halaman preprocess terlebih dahulu.", 'error')
        return redirect(url_for('preprocess', ticker=ticker))

    if gru_model.input_dim != INPUT_DIM or gru_model.hidden_dim != HIDDEN_DIM or gru_model.output_dim != OUTPUT_DIM or gru_model.dropout_rate != DROPOUT_RATE:
        flash("Dimensi atau dropout rate model yang dimuat tidak cocok dengan konfigurasi saat ini. Silakan inisialisasi ulang model di halaman Preprocessing.", 'error')
        return redirect(url_for('preprocess', ticker=ticker))

    EPOCHS = config.DEFAULT_EPOCHS
    LEARNING_RATE = config.DEFAULT_LEARNING_RATE
    EARLY_STOPPING_PATIENCE = config.EARLY_STOPPING_PATIENCE
    BATCH_SIZE = config.DEFAULT_BATCH_SIZE 

    losses = []
    val_losses = []
    start_time = time.time()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    flash(f"Memulai pelatihan model GRU untuk {ticker}...", 'info')
    print(f"--- Memulai pelatihan model GRU untuk {ticker} (Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Dropout: {DROPOUT_RATE}, Patience: {EARLY_STOPPING_PATIENCE}, Batch Size: {BATCH_SIZE}) ---")

    try:
        for epoch in range(EPOCHS):
            total_loss = 0
            gru_model.set_training_mode(True) # Aktifkan dropout untuk pelatihan

            # Acak indeks data pelatihan
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            # Loop melalui mini-batch
            num_batches = int(np.ceil(X_train_shuffled.shape[0] / BATCH_SIZE)) # Hitung jumlah batch
            for i in range(num_batches):
                start_idx = i * BATCH_SIZE
                end_idx = min((i + 1) * BATCH_SIZE, X_train_shuffled.shape[0])

                x_batch = X_train_shuffled[start_idx:end_idx] # Ambil batch X
                y_batch = y_train_shuffled[start_idx:end_idx].reshape(-1, 1) # Ambil batch y dan reshape

                y_pred_scaled = gru_model.forward(x_batch) # Kirim batch ke forward pass

                loss = mse_loss(y_pred_scaled, y_batch)
                total_loss += loss

                d_output = mse_loss_derivative(y_pred_scaled, y_batch)
                gru_model.backward(d_output) # Kirim d_output ke backward pass

                gru_model.update_weights(LEARNING_RATE)

            avg_loss = total_loss / num_batches
            losses.append(avg_loss)

            gru_model.set_training_mode(False) # Nonaktifkan dropout untuk evaluasi validasi
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

            flash(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}", 'info')
            print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            # Logika Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_weights = gru_model.get_weights()
                flash("Validation loss meningkat. Menyimpan bobot model terbaik.", 'info')
            else:
                patience_counter += 1
                flash(f"Validation loss tidak meningkat. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}", 'warning')

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                flash(f"Early stopping dipicu setelah {epoch + 1} epoch tanpa peningkatan validasi loss.", 'warning')
                print(f"--- Early stopping dipicu setelah {epoch + 1} epoch ---")
                break

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

    model_filename = os.path.join(MODEL_DIR, f'GRU_weights_{ticker}_H{HIDDEN_DIM}_L{lookback_window}_F{INPUT_DIM}_D{int(DROPOUT_RATE*100)}.npz')

    try:
        if best_model_weights is not None:
            np.savez(model_filename, **best_model_weights)
            flash(f"Bobot model TERBAIK berhasil disimpan ke: {os.path.basename(model_filename)}", 'success')
            gru_model.set_weights(best_model_weights) 
        else: 
            np.savez(model_filename, **gru_model.get_weights())
            flash(f"Bobot model terakhir berhasil disimpan ke: {os.path.basename(model_filename)}", 'success')
    except Exception as e:
        flash(f"Gagal menyimpan bobot model: {e}", 'error')
        print(f"ERROR: Gagal menyimpan bobot model: {e}")

    final_loss = losses[-1] if losses else 0
    final_val_loss = val_losses[-1] if val_losses else 0
    if best_model_weights is not None and best_val_loss != float('inf'):
        final_val_loss = best_val_loss 

    flash(f"Pelatihan selesai dalam {training_duration:.2f} detik. Final Train Loss: {final_loss:.6f}, Final Val Loss (best): {final_val_loss:.6f}", 'success')

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
def predict_price():
    ticker = config.DEFAULT_TICKER 
    X_test_samples = app.config.get(f'X_val_{ticker}')
    y_true_scaled_test = app.config.get(f'y_val_{ticker}')

    gru_model = app.config.get('gru_model')
    scaler = app.config.get('scaler_all_features')
    lookback_window = app.config.get('lookback_window')
    features_to_scale = app.config.get('features_to_scale')
    df_full_preprocessed = app.config.get('df_full_preprocessed')
    DROPOUT_RATE = app.config.get('dropout_rate') 

    if X_test_samples is None or y_true_scaled_test is None or gru_model is None or scaler is None or features_to_scale is None or df_full_preprocessed is None:
        flash("Data, model, atau scaler belum disiapkan atau dilatih. Silakan kunjungan halaman preprocess dan latih model terlebih dahulu.", 'error')
        return redirect(url_for('train_model'))

    test_size = X_test_samples.shape[0]

    predictions_scaled = []

    gru_model.set_training_mode(False) # Set model ke mode inferensi (dropout nonaktif)
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

    # Hitung metrik evaluasi
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

    display_results = prediction_results[-20:]

    flash(f"Prediksi berhasil dilakukan untuk {test_size} hari terakhir. RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%", 'success')

    return render_template('predict.html',
                           ticker=ticker,
                           rmse=rmse,
                           mae=mae, 
                           mape=mape, 
                           prediction_results=display_results,
                           total_predictions=test_size)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)