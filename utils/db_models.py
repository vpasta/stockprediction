from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import UniqueConstraint # Import ini juga

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