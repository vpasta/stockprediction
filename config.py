import os
import secrets

SECRET_KEY = '2bc740f863cc778aea16789d2ac2fbaefa6105e5eb379148013e72af8e3161d1'

DB_NAME = 'prediksi_tsla_db'
DB_USER = 'root'
DB_PASSWORD = '' 
DB_HOST = 'localhost'
DB_PORT = '3306'
DATA_DIR = 'data'
MODEL_DIR = 'models'
LOG_DIR = 'logs'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

DEFAULT_TICKER = 'TSLA'
TICKER_LOGO_MAP = {
    'TSLA': 'https://logo.clearbit.com/tesla.com',
    'AAPL': 'https://logo.clearbit.com/apple.com',
    'AMZN': 'https://logo.clearbit.com/amazon.com',
    'GOOGL': 'https://logo.clearbit.com/google.com',
    'MSFT': 'https://logo.clearbit.com/microsoft.com'
}
DEFAULT_LOOKBACK_WINDOW = 90
DEFAULT_HIDDEN_DIM = 256
DEFAULT_EPOCHS = 200
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_DROPOUT_RATE = 0.5
DEFAULT_TRAIN_SPLIT_RATIO = 0.8
EARLY_STOPPING_PATIENCE = 30
DEFAULT_BATCH_SIZE = 16
DEFAULT_FUTURE_PREDICTION_DAYS = 7