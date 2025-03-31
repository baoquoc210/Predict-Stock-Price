# Stock Price Prediction with LSTM

This project predicts stock prices using a Long Short-Term Memory (LSTM) neural network. It uses historical data for Tesla (TSLA) from 2020 to March 31, 2025, stored in a static SQLite database, and incorporates technical indicators to improve predictions. The model is optimized for faster training while maintaining accuracy.

## Features
- **Data Storage**: Historical stock data is stored in a SQLite database (`stock_data_static.db`) for efficient reuse.
- **Technical Indicators**: Includes Simple Moving Average (SMA), Exponential Moving Average (EMA), Relative Strength Index (RSI), MACD, and Bollinger Bands.
- **Sentiment Placeholder**: A placeholder for sentiment analysis (e.g., from X posts) is included.
- **Model**: A two-layer LSTM model with dropout for regularization.
- **Visualization**: Plots actual vs. predicted prices with specific dates on the x-axis.
- **Optimized Training**: Uses early stopping and a batch size of 128 to reduce training time.

## Prerequisites
- **Python**: Version 3.7 to 3.11 (TensorFlow compatibility).
- **Dependencies**: Install the required libraries using:
  ```bash
  pip install pandas yfinance numpy tensorflow scikit-learn matplotlib ta
