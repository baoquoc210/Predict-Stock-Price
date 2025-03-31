# Stock Price Prediction with LSTM

This project uses a Long Short-Term Memory (LSTM) neural network to predict stock prices for Tesla (TSLA) based on historical data from January 2020 to March 31, 2025. The data is stored in a static SQLite database, and the model incorporates various technical indicators to enhance prediction accuracy. The project is optimized for both performance and precision, with a focus on reducing lag in predictions.

## Features
- **Data Storage**: Historical stock data is stored in a SQLite database (`stock_data_static.db`) for efficient access.
- **Technical Indicators**: Includes Simple Moving Average (SMA), Exponential Moving Average (EMA), Relative Strength Index (RSI), MACD, Bollinger Bands, daily price change, and a sentiment placeholder.
- **Enhanced Model**: A two-layer LSTM model with 128 and 64 units, dropout for regularization, a smaller learning rate (0.0005), and a shorter sequence length (30 days) for better precision.
- **Training**: Uses 50 epochs with early stopping, a batch size of 128, and a train/test split to balance accuracy and training time.
- **Visualization**: Generates a plot comparing actual vs. predicted prices, with specific dates on the x-axis, saved as `tsla_prediction_plot.png`.
- **Next-Day Prediction**: Outputs a predicted price for the next trading day.

## Prerequisites
- **Python**: Version 3.7 to 3.11 (for TensorFlow compatibility).
- **Dependencies**: Install the required libraries using the following command:
  ```bash
  pip install pandas yfinance numpy tensorflow scikit-learn matplotlib ta
