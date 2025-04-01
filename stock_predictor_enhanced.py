import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN for consistent numerical results

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
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
    """Create SQLite database and table for stock data, or migrate existing table."""
    try:
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        
        # Create the table if it doesn't exist
        c.execute('''CREATE TABLE IF NOT EXISTS prices
                     (date TEXT, symbol TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER, 
                     PRIMARY KEY (date, symbol))''')
        
        # Check if the table has the new columns (open, high, low)
        c.execute("PRAGMA table_info(prices)")
        columns = [info[1] for info in c.fetchall()]
        
        # Add missing columns if they don't exist
        if 'open' not in columns:
            c.execute("ALTER TABLE prices ADD COLUMN open REAL")
            logging.info("Added 'open' column to prices table.")
        if 'high' not in columns:
            c.execute("ALTER TABLE prices ADD COLUMN high REAL")
            logging.info("Added 'high' column to prices table.")
        if 'low' not in columns:
            c.execute("ALTER TABLE prices ADD COLUMN low REAL")
            logging.info("Added 'low' column to prices table.")
        
        conn.commit()
        logging.info("Database initialized or migrated successfully.")
        return conn
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        raise

def load_historical_data(conn, symbol='TSLA', start_date='2020-01-01', end_date='2025-03-31'):
    """Load static historical data into the database if not already present."""
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM prices WHERE symbol = ? AND open IS NOT NULL AND high IS NOT NULL AND low IS NOT NULL", (symbol,))
    if c.fetchone()[0] == 0:  # Only load if table is empty or missing new columns for this symbol
        # Clear existing data for this symbol to ensure consistency
        c.execute("DELETE FROM prices WHERE symbol = ?", (symbol,))
        data = yf.download(symbol, start=start_date, end=end_date)
        for index, row in data.iterrows():
            date = index.strftime('%Y-%m-%d')
            open_price = float(row['Open'])
            high_price = float(row['High'])
            low_price = float(row['Low'])
            close_price = float(row['Close'])
            volume = int(row['Volume'])
            c.execute("INSERT OR IGNORE INTO prices VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (date, symbol, open_price, high_price, low_price, close_price, volume))
        conn.commit()
        logging.info(f"Loaded historical data for {symbol} from {start_date} to {end_date}.")
    else:
        logging.info(f"Data for {symbol} already exists in database with required columns.")

# 2. Data Processing Functions
def fetch_data_from_db(conn, symbol='TSLA'):
    """Retrieve data from SQLite database."""
    query = f"SELECT date, open, high, low, close, volume FROM prices WHERE symbol = '{symbol}' ORDER BY date"
    df = pd.read_sql_query(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    # If open, high, low are missing, fill with close as a fallback
    for col in ['open', 'high', 'low']:
        if df[col].isnull().any():
            logging.warning(f"Missing {col} values for {symbol}; filling with close price.")
            df[col].fillna(df['close'], inplace=True)
    return df

def add_features(df):
    """Add technical indicators, sentiment, and price change to the dataset."""
    df['SMA_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
    df['EMA_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
    df['RSI'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(df['close'])
    df['MACD'] = macd.macd()
    bb = BollingerBands(df['close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    df['OBV'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    # Simple rule-based sentiment based on price change (replace with real sentiment data)
    df['Sentiment'] = df['close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    # Add daily price change
    df['Price_Change'] = df['close'].diff()
    # Add lagged features
    df['Lag_1'] = df['close'].shift(1)
    df['Lag_2'] = df['close'].shift(2)
    df['Lag_3'] = df['close'].shift(3)
    return df.dropna()

def prepare_sequences(df, sequence_length=15):
    """Create sequences for LSTM input."""
    scaler = RobustScaler()  # Switch to RobustScaler
    features = ['close', 'volume', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'ATR', 'OBV', 'Sentiment', 'Price_Change', 'Lag_1', 'Lag_2', 'Lag_3']
    scaled_data = scaler.fit_transform(df[features])
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Predict 'close'
    
    X, y = np.array(X), np.array(y)
    logging.info(f"Prepared {len(X)} sequences with {sequence_length} timesteps and {len(features)} features.")
    return X, y, scaler, len(features)  # Return number of features for validation

# 3. Model Functions
def build_lstm_model(sequence_length, n_features):
    """Construct and compile a simpler LSTM model."""
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(sequence_length, n_features)))
    model.add(Dropout(0.2))  # Reduced dropout
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1))
    
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')  # Lower learning rate
    return model

def train_model(model, X, y, scaler, epochs=50, batch_size=64):
    """Train the LSTM model with time series cross-validation, early stopping, and learning rate scheduling."""
    tscv = TimeSeriesSplit(n_splits=3)
    mae_scores = []
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_test, y_test), verbose=1, 
                            callbacks=[early_stopping, reduce_lr])
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
                        validation_data=(X_test, y_test), verbose=1, 
                        callbacks=[early_stopping, reduce_lr])
    
    # Save the trained model in native Keras format
    model.save('stock_predictor_model.keras')
    logging.info("Model saved to stock_predictor_model.keras")
    return model, X_test, y_test, history

# 4. Evaluation and Prediction
def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model performance with multiple metrics and return predictions."""
    predictions_scaled = model.predict(X_test)
    dummy = np.zeros((len(predictions_scaled), X_test.shape[2]-1))
    predictions = scaler.inverse_transform(np.concatenate([predictions_scaled, dummy], axis=1))[:, 0]
    y_test_full = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), dummy], axis=1))[:, 0]
    
    mae = np.mean(np.abs(predictions - y_test_full))
    rmse = np.sqrt(np.mean((predictions - y_test_full) ** 2))
    mape = np.mean(np.abs((predictions - y_test_full) / y_test_full)) * 100
    logging.info(f"Test set MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    print(f"Test set MAE: {mae:.2f}")
    print(f"Test set RMSE: {rmse:.2f}")
    print(f"Test set MAPE: {mape:.2f}%")
    return predictions, y_test_full, mae

def predict_multiple_days(model, last_sequence, scaler, days=5):
    """Predict stock prices for multiple days ahead."""
    predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(days):
        next_day_scaled = model.predict(current_sequence)
        next_day_price = scaler.inverse_transform(np.concatenate([next_day_scaled, np.zeros((1, current_sequence.shape[2]-1))], axis=1))[0, 0]
        predictions.append(next_day_price)
        # Update the sequence with the new prediction
        next_day_full = np.zeros((1, current_sequence.shape[2]))
        next_day_full[0, 0] = next_day_scaled[0, 0]
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, :] = next_day_full
    return predictions

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
    # Save the plot as an image
    plt.savefig('tsla_prediction_plot.png', bbox_inches='tight')
    plt.show(block=False)  # Non-blocking plot for Jupyter Notebook

# 5. Main Execution
def main():
    symbol = 'TSLA'
    sequence_length = 15  # Reduced sequence length
    
    # Setup database and load data
    conn = initialize_database()
    load_historical_data(conn, symbol=symbol)
    
    # Process data
    df = fetch_data_from_db(conn, symbol)
    df = add_features(df)
    X, y, scaler, n_features = prepare_sequences(df, sequence_length)
    
    # Check if a trained model exists
    model_path = 'stock_predictor_model.keras'
    if os.path.exists(model_path):
        logging.info("Loading existing model from stock_predictor_model.keras")
        try:
            model = load_model(model_path)
            # Check if the model's input shape matches the current number of features
            expected_n_features = model.input_shape[-1]
            if expected_n_features != n_features:
                logging.warning(f"Feature mismatch: Model expects {expected_n_features} features, but data has {n_features} features. Retraining model.")
                model = build_lstm_model(sequence_length, n_features)
                model, X_test, y_test, history = train_model(model, X, y, scaler)
            else:
                # Recompile the model to build the metrics
                model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
                # Final train/test split for evaluation
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                history = None  # No training history since we're loading the model
        except Exception as e:
            logging.warning(f"Failed to load model: {e}. Retraining model.")
            model = build_lstm_model(sequence_length, n_features)
            model, X_test, y_test, history = train_model(model, X, y, scaler)
    else:
        # Build and train model
        model = build_lstm_model(sequence_length, n_features)
        model, X_test, y_test, history = train_model(model, X, y, scaler)
    
    # Evaluate and visualize
    predictions, actual, mae = evaluate_model(model, X_test, y_test, scaler)
    
    # Get dates for the test set
    test_dates = df.index[-len(actual):]
    plot_results(actual, predictions, test_dates, symbol)
    
    # Predict next day
    logging.info("Starting next-day prediction...")
    last_sequence = X[-1].reshape(1, sequence_length, X.shape[2])
    next_day_scaled = model.predict(last_sequence)
    next_day_price = scaler.inverse_transform(np.concatenate([next_day_scaled, np.zeros((1, X.shape[2]-1))], axis=1))[0, 0]
    print(f"Predicted price for next day: ${next_day_price:.2f}")
    logging.info(f"Next-day prediction completed: ${next_day_price:.2f}")
    
    # Predict multiple days ahead
    future_days = predict_multiple_days(model, last_sequence, scaler, days=5)
    print("Predicted prices for the next 5 days:", [f"${price:.2f}" for price in future_days])
    
    conn.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Program failed: {e}")
        print(f"An error occurred: {e}")