import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import sqlite3
import logging
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
from datetime import datetime
import keras_tuner as kt

# Configure logging for professional tracking
logging.basicConfig(filename='stock_predictor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Database and Data Management
def initialize_database(db_name='stock_data_static.db'):
    """Initialize SQLite database with stock price table."""
    try:
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS prices
                     (date TEXT, symbol TEXT, close REAL, volume INTEGER, PRIMARY KEY (date, symbol))''')
        conn.commit()
        logging.info("Database initialized for stock data storage.")
        return conn
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        raise

def load_historical_data(conn, symbol='TSLA', start_date='2020-01-01', end_date='2025-03-31'):
    """Load and store historical stock data into SQLite."""
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM prices WHERE symbol = ?", (symbol,))
    if c.fetchone()[0] == 0:
        data = yf.download(symbol, start=start_date, end=end_date)
        for index, row in data.iterrows():
            date = index.strftime('%Y-%m-%d')
            close_price = float(row['Close'])
            volume = int(row['Volume'])
            c.execute("INSERT OR IGNORE INTO prices VALUES (?, ?, ?, ?)",
                      (date, symbol, close_price, volume))
        conn.commit()
        logging.info(f"Loaded {symbol} data from {start_date} to {end_date}.")
    else:
        logging.info(f"Data for {symbol} already exists.")

# 2. Data Preprocessing and Feature Engineering
def fetch_and_preprocess_data(conn, symbol='TSLA'):
    """Fetch data from database and preprocess for analysis."""
    df = pd.read_sql_query(f"SELECT date, close, volume FROM prices WHERE symbol = '{symbol}' ORDER BY date", conn)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Feature engineering for AI model
    df['SMA_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
    df['EMA_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
    df['RSI'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(df['close'])
    df['MACD'] = macd.macd()
    bb = BollingerBands(df['close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['Sentiment'] = np.random.uniform(-1, 1, len(df))  # Placeholder for real sentiment
    return df.dropna()

def prepare_sequences(df, sequence_length=60):
    """Prepare time series data with engineered features for LSTM."""
    scaler = MinMaxScaler()
    features = ['close', 'volume', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'Sentiment']
    scaled_data = scaler.fit_transform(df[features])
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Predict 'close'
    X, y = np.array(X), np.array(y)
    logging.info(f"Prepared {len(X)} sequences with {sequence_length} timesteps.")
    return X, y, scaler, df.index[sequence_length:]

# 3. Model Development and Training with Hyperparameter Optimization
def build_tunable_model(hp):
    """Define a tunable LSTM model for hyperparameter optimization."""
    model = Sequential()
    model.add(LSTM(units=hp.Int('units_1', min_value=32, max_value=128, step=32),
                   return_sequences=True, input_shape=(hp.Int('sequence_length', 60, 60, step=1), 
                                                     hp.Int('n_features', 9, 9, step=1))))
    model.add(Dropout(hp.Float('dropout_1', 0.1, 0.3, step=0.1)))
    model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=64, step=32),
                   return_sequences=False))
    model.add(Dropout(hp.Float('dropout_2', 0.1, 0.3, step=0.1)))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1))
    
    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                  loss='mean_squared_error')
    return model

def train_optimized_model(X, y, scaler, max_trials=10):
    """Train model with hyperparameter tuning using keras-tuner."""
    tuner = kt.RandomSearch(
        build_tunable_model,
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='stock_prediction'
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    tuner.search(X_train, y_train, epochs=30, batch_size=64,
                 validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)
    
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model, X_test, y_test

# 4. Evaluation and Visualization
def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model accuracy and generate predictions."""
    predictions_scaled = model.predict(X_test)
    dummy = np.zeros((len(predictions_scaled), X_test.shape[2]-1))
    predictions = scaler.inverse_transform(np.concatenate([predictions_scaled, dummy], axis=1))[:, 0]
    y_test_full = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), dummy], axis=1))[:, 0]
    mae = np.mean(np.abs(predictions - y_test_full))
    logging.info(f"Test MAE: {mae:.2f}")
    return predictions, y_test_full, mae

def plot_results(actual, predicted, dates, symbol='TSLA'):
    """Visualize predictions with detailed date labels."""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual Prices', color='blue')
    plt.plot(dates, predicted, label='Predicted Prices', color='red')
    plt.title(f'{symbol} Stock Price Prediction for AI-Driven Insights')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def summarize_results(mae, next_day_price):
    """Summarize model performance for professional reporting."""
    print(f"\n=== Prediction Summary ===")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Predicted Price for Next Day: ${next_day_price:.2f}")
    print("This model leverages advanced AI techniques and hyperparameter optimization for financial forecasting, suitable for AI-driven investment strategies in Australia.")

# 5. Main Execution
def main():
    symbol = 'TSLA'
    sequence_length = 60
    
    # Database and data setup
    conn = initialize_database()
    load_historical_data(conn, symbol=symbol)
    
    # Data preprocessing and feature engineering
    df = fetch_and_preprocess_data(conn)
    X, y, scaler, dates = prepare_sequences(df, sequence_length)
    
    # Model training with optimization
    model, X_test, y_test = train_optimized_model(X, y, scaler)
    predictions, actual, mae = evaluate_model(model, X_test, y_test, scaler)
    
    # Visualization and reporting
    test_dates = dates[-len(actual):]
    plot_results(actual, predictions, test_dates, symbol)
    
    # Next-day prediction
    last_sequence = X[-1].reshape(1, sequence_length, X.shape[2])
    next_day_scaled = model.predict(last_sequence)
    next_day_price = scaler.inverse_transform(np.concatenate([next_day_scaled, np.zeros((1, X.shape[2]-1))], axis=1))[0, 0]
    summarize_results(mae, next_day_price)
    
    conn.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error in execution: {e}")
        print(f"An error occurred: {e}")
