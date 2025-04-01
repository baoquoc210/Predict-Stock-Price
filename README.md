# Stock Price Prediction with LSTM

This project uses a Long Short-Term Memory (LSTM) neural network to predict stock prices for Tesla (TSLA) from January 2020 to March 31, 2025. The model leverages historical data stored in a SQLite database, enhanced technical indicators, and a simplified LSTM architecture for improved accuracy. The trained model is saved for reuse, and predictions include both next-day and multi-day forecasts.

## Features
- **Data Storage**: Historical data in a SQLite database (`stock_data_static.db`).
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, daily price change, and a rule-based sentiment (based on price change).
- **New Features**: Added lagged prices (`Lag_1`, `Lag_2`, `Lag_3`) for short-term dependencies.
- **Model**: Two-layer LSTM (64, 32 units) with dropout (0.2) and a sequence length of 15.
- **Training**: 50 epochs, batch size of 64, early stopping, learning rate scheduling, and 3-fold time series cross-validation.
- **Model Persistence**: Saves the model as `stock_predictor_model.keras` for reuse.
- **Predictions**: Next-day and 5-day ahead forecasts.
- **Evaluation**: MAE, RMSE, and MAPE metrics.
- **Visualization**: Plot of actual vs. predicted prices, saved as `tsla_prediction_plot.png`.

## Prerequisites
- **Python**: 3.7 to 3.11 (TensorFlow compatibility).
- **Dependencies**:
  ```bash
  pip install pandas yfinance numpy tensorflow scikit-learn matplotlib ta

## Full Report
For a detailed explanation of the methodology, model architecture, evaluation metrics, results, and discussion, please refer to the full report
[Stock Price Prediction Report: Tesla (TSLA) Using LSTM](https://docs.google.com/document/d/17UIqmWPD65O6nokmnc4Mly8vfG2Ph3pphW3uM_IPq-s/edit?usp=sharing)

## License
MIT License. See [LICENSE](LICENSE).

**Copyright Notice**: The methodology, code, and findings in this project are the intellectual property of Quoc Bao Huynh. Unauthorized use or reproduction is prohibited.
