from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import UniqueConstraint, ForeignKey
from sqlalchemy.orm import relationship

db = SQLAlchemy()

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

    __table_args__ = (UniqueConstraint('ticker', 'date', name='_ticker_date_uc'),)

    def __repr__(self):
        return f'<StockData {self.ticker} - {self.date}>'
    
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False) 
    role = db.Column(db.String(20), nullable=False, default='free_user') 

    def __repr__(self):
        return f'<User {self.username} ({self.role})>'
    
class SavedModel(db.Model):
    __tablename__ = 'saved_models'
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), nullable=False)
    lookback_window = db.Column(db.Integer, nullable=False)
    hidden_dim = db.Column(db.Integer, nullable=False)
    input_dim = db.Column(db.Integer, nullable=False)
    learning_rate = db.Column(db.Float, nullable=False)
    dropout_rate = db.Column(db.Float, nullable=False)
    rmse = db.Column(db.Float)
    mae = db.Column(db.Float)
    mape = db.Column(db.Float)
    model_filepath = db.Column(db.String(255), nullable=False, unique=True) 
    training_timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    
    predictions = relationship('PredictionDetail', backref='saved_model', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f'<SavedModel {self.ticker} L{self.lookback_window} LR{self.learning_rate} DO{self.dropout_rate}>'

class PredictionDetail(db.Model):
    __tablename__ = 'prediction_details'
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, ForeignKey('saved_models.id'), nullable=False)
    prediction_date = db.Column(db.Date, nullable=False) 
    true_price = db.Column(db.Float)
    predicted_price = db.Column(db.Float)

    __table_args__ = (UniqueConstraint('model_id', 'prediction_date', name='_model_date_uc'),)

    def __repr__(self):
        return f'<PredictionDetail ModelID:{self.model_id} Date:{self.prediction_date}>'