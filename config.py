import os
import secrets

SECRET_KEY = secrets.token_hex(16)

DB_NAME = 'prediksi_tsla_db'
DB_USER = 'root'
DB_PASSWORD = '' 
DB_HOST = 'localhost'
DB_PORT = '3306'

DATA_DIR = 'data'
MODEL_DIR = 'models'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

DEFAULT_TICKER = 'TSLA'
DEFAULT_LOOKBACK_WINDOW = 90
DEFAULT_HIDDEN_DIM = 128
DEFAULT_EPOCHS = 200
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_TRAIN_SPLIT_RATIO = 0.8