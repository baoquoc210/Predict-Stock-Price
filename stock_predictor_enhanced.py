import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import sqlite3
import logging
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
from datetime import datetime

# Configure logging
logging.basicConfig(filename='stock_predictor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Database Functions
def initialize_database(db_name='stock_data_static.db'):
    """Create SQLite database and table for stock data."""
    try:
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS prices
                     (date TEXT, symbol TEXT, close REAL, volume INTEGER, PRIMARY KEY (date, symbol))''')
        conn.commit()
        logging.info("Database initialized.")
        return conn
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        raise

def load_historical_data(conn, symbol='TSLA', start_date='2020-01-01', end_date='2025-03-31'):
    """Load static historical data into the database if not already present."""
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM prices WHERE symbol = ?", (symbol,))
    if c.fetchone()[0] == 0:  # Only load if table is empty for this symbol
        data = yf.download(symbol, start=start_date, end=end_date)
        for index, row in data.iterrows():
            date = index.strftime('%Y-%m-%d')
            close_price = float(row['Close'])
            volume = int(row['Volume'])
            c.execute("INSERT OR IGNORE INTO prices VALUES (?, ?, ?, ?)",
                      (date, symbol, close_price, volume))
        conn.commit()
        logging.info(f"Loaded historical data for {symbol} from {start_date} to {end_date}.")
    else:
        logging.info(f"Data for {symbol} already exists in database.")

# 2. Data Processing Functions
def fetch_data_from_db(conn, symbol='TSLA'):
    """Retrieve data from SQLite database."""
    query = f"SELECT date, close, volume FROM prices WHERE symbol = '{symbol}' ORDER BY date"
    df = pd.read_sql_query(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def add_features(df):
    """Add technical indicators and sentiment to the dataset."""
    df['SMA_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
    df['EMA_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
    df['RSI'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(df['close'])
    df['MACD'] = macd.macd()
    bb = BollingerBands(df['close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    # Placeholder for X sentiment (replace with real analysis)
    df['Sentiment'] = np.random.uniform(-1, 1, len(df))  # Random for now
    return df.dropna()

def prepare_sequences(df, sequence_length=60):
    """Create sequences for LSTM input."""
    scaler = MinMaxScaler()
    features = ['close', 'volume', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'Sentiment']
    scaled_data = scaler.fit_transform(df[features])
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Predict 'close'
    
    X, y = np.array(X), np.array(y)
    logging.info(f"Prepared {len(X)} sequences with {sequence_length} timesteps and {len(features)} features.")
    return X, y, scaler

# 3. Model Functions
def build_lstm_model(sequence_length, n_features):
    """Construct and compile a deeper LSTM model."""
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(sequence_length, n_features)))
    model.add(Dropout(0.3))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_model(model, X, y, scaler, epochs=50, batch_size=128):
    """Train the LSTM model with time series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_test, y_test), verbose=1)
        predictions_scaled = model.predict(X_test)
        dummy = np.zeros((len(predictions_scaled), X_test.shape[2]-1))
        predictions = scaler.inverse_transform(np.concatenate([predictions_scaled, dummy], axis=1))[:, 0]
        y_test_full = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), dummy], axis=1))[:, 0]
        mae = np.mean(np.abs(predictions - y_test_full))
        mae_scores.append(mae)
    print(f"Cross-validated MAE: {np.mean(mae_scores):.2f} (+/- {np.std(mae_scores):.2f})")
    # Final train/test split for plotting
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test), verbose=1)
    return model, X_test, y_test, history

# 4. Evaluation and Prediction
def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model performance and return predictions."""
    predictions_scaled = model.predict(X_test)
    dummy = np.zeros((len(predictions_scaled), X_test.shape[2]-1))
    predictions = scaler.inverse_transform(np.concatenate([predictions_scaled, dummy], axis=1))[:, 0]
    y_test_full = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), dummy], axis=1))[:, 0]
    
    mae = np.mean(np.abs(predictions - y_test_full))
    logging.info(f"Test set MAE: {mae:.2f}")
    return predictions, y_test_full, mae

def plot_results(actual, predicted, dates, symbol='TSLA'):
    """Plot actual vs predicted prices with specific dates on x-axis."""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual Prices', color='blue')
    plt.plot(dates, predicted, label='Predicted Prices', color='red')
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    
    # Format x-axis with specific dates
    ax = plt.gca()
    ax.xaxis.set_major_locator(MonthLocator())  # Show ticks every month
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))  # Format as YYYY-MM-DD
    plt.xticks(rotation=45)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# 5. Main Execution
def main():
    symbol = 'TSLA'
    sequence_length = 60
    
    # Setup database and load data
    conn = initialize_database()
    load_historical_data(conn, symbol=symbol)
    
    # Process data
    df = fetch_data_from_db(conn, symbol)
    df = add_features(df)
    X, y, scaler = prepare_sequences(df, sequence_length)
    
    # Build and train model
    model = build_lstm_model(sequence_length, X.shape[2])
    model, X_test, y_test, history = train_model(model, X, y, scaler)
    
    # Evaluate and visualize
    predictions, actual, mae = evaluate_model(model, X_test, y_test, scaler)
    print(f"Test set MAE: {mae:.2f}")
    
    # Get dates for the test set
    test_dates = df.index[-len(actual):]
    plot_results(actual, predictions, test_dates, symbol)
    
    # Predict next day
    last_sequence = X[-1].reshape(1, sequence_length, X.shape[2])
    next_day_scaled = model.predict(last_sequence)
    next_day_price = scaler.inverse_transform(np.concatenate([next_day_scaled, np.zeros((1, X.shape[2]-1))], axis=1))[0, 0]
    print(f"Predicted price for next day: ${next_day_price:.2f}")
    
    conn.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Program failed: {e}")
        print(f"An error occurred: {e}")