import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# -----------------------------
# SETTINGS
# -----------------------------
# List of tickers to analyze. Expand this list as desired.
tickers = ['AAPL', 'MSFT', 'GOOG','MA','V','NVDA', 'AMD', 'TSLA','AXP','KO','CRM','PLTR','SHOP','COST']  # Example tickers ['AAPL', 'MSFT', 'GOOG']
start_date = '2010-01-01'
end_date   = '2020-12-31'
sequence_length = 60     # Number of prior days used as input
forecast_horizon = 10     # For multi-step forecast: predict 5 days ahead

# Root folder where forecasts and charts will be saved.
output_root = 'RNN forecasts'
os.makedirs(output_root, exist_ok=True)

# Dictionary for summary metrics
summary_stats = {}

# -----------------------------
# PROCESS EACH TICKER
# -----------------------------
for ticker in tickers:
    print("\n" + "=" * 60)
    print(f"Processing data for ticker: {ticker}")
    
    # Create a folder for the current ticker.
    ticker_folder = os.path.join(output_root, ticker)
    os.makedirs(ticker_folder, exist_ok=True)
    
    # -----------------------------
    # 1. Download Data and Prepare Features (Close & Volume)
    # -----------------------------
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    print("Data downloaded. DataFrame columns:")
    print(data.columns)
    
    # Handle potential MultiIndex columns (yfinance sometimes returns a MultiIndex):
    if isinstance(data.columns, pd.MultiIndex):
        print("Detected MultiIndex columns. Extracting 'Close' and 'Volume'...")
        if 'Close' in data.columns.levels[0] and 'Volume' in data.columns.levels[0]:
            df_close = data['Close']
            df_volume = data['Volume']
            if ticker in df_close.columns and ticker in df_volume.columns:
                print(f"Selecting '{ticker}' columns for Close and Volume.")
                df = pd.DataFrame({
                    'Close': df_close[ticker],
                    'Volume': df_volume[ticker]
                })
            else:
                print("Ticker not found in sub-columns; using first available columns.")
                df = pd.DataFrame({
                    'Close': df_close.iloc[:, 0],
                    'Volume': df_volume.iloc[:, 0]
                })
        else:
            raise ValueError("MultiIndex detected but required columns 'Close' and/or 'Volume' not found.")
    else:
        if 'Close' in data.columns and 'Volume' in data.columns:
            df = data[['Close', 'Volume']].copy()
        elif 'Adj Close' in data.columns and 'Volume' in data.columns:
            df = data[['Adj Close', 'Volume']].copy()
            df.rename(columns={'Adj Close': 'Close'}, inplace=True)
        else:
            raise ValueError("Required columns ('Close' and 'Volume') not found in data.")
    
    print("DataFrame head:")
    print(df.head())
    print("DataFrame shape:", df.shape)
    
    # -----------------------------
    # 2. Scaling
    # -----------------------------
    # Features: both Close and Volume; Target: Close only.
    features = df[['Close', 'Volume']].values
    target   = df[['Close']].values
    
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target   = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = scaler_features.fit_transform(features)
    scaled_target   = scaler_target.fit_transform(target)
    
    print("Scaled features shape:", scaled_features.shape)
    print("Scaled target shape:", scaled_target.shape)
    
    # -----------------------------
    # 3. Split Data into Training and Test Sets
    # -----------------------------
    training_data_len = int(np.ceil(len(df) * 0.8))
    print("Training data length (rows):", training_data_len)
    
    # -------------------------------------------------------------------
    # SECTION A: ONE-STEP FORECAST (Predict next day Close)
    # -------------------------------------------------------------------
    train_features = scaled_features[:training_data_len, :]   # shape: (train_rows, 2)
    train_target   = scaled_target[:training_data_len, :]       # shape: (train_rows, 1)
    
    x_train = []
    y_train = []
    for i in range(sequence_length, len(train_features)):
        x_train.append(train_features[i-sequence_length:i, :])
        y_train.append(train_target[i, 0])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print("x_train (one-step) shape:", x_train.shape)
    print("y_train (one-step) shape:", y_train.shape)
    
    # Build the one-step forecasting model.
    model_one_step = Sequential()
    model_one_step.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model_one_step.add(Dropout(0.2))
    model_one_step.add(LSTM(units=50, return_sequences=False))
    model_one_step.add(Dropout(0.2))
    model_one_step.add(Dense(units=25))
    model_one_step.add(Dense(units=1))  # Output: next day's close
    
    model_one_step.compile(optimizer='adam', loss='mean_squared_error')
    print(f"\nOne-Step Model summary for {ticker}:")
    model_one_step.summary()
    
    print(f"Training one-step forecast model for {ticker}...")
    model_one_step.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)
    
    # Prepare test data for one-step forecast.
    test_features = scaled_features[training_data_len - sequence_length:, :]
    x_test = []
    for i in range(sequence_length, len(test_features)):
        x_test.append(test_features[i-sequence_length:i, :])
    x_test = np.array(x_test)
    print("x_test (one-step) shape:", x_test.shape)
    
    y_test = target[training_data_len:, :]  # Actual unscaled close prices
    
    # Predict and inverse transform predictions.
    predictions_one_step = model_one_step.predict(x_test)
    predictions_one_step_inv = scaler_target.inverse_transform(predictions_one_step)
    print("Predictions (one-step) shape:", predictions_one_step_inv.shape)
    
    # Accuracy metrics for one-step forecasts.
    mae_one = mean_absolute_error(y_test, predictions_one_step_inv)
    mse_one = mean_squared_error(y_test, predictions_one_step_inv)
    rmse_one = np.sqrt(mse_one)
    mape_one = np.mean(np.abs((y_test - predictions_one_step_inv) / y_test)) * 100
    
    print(f"\nOne-Step Forecast Accuracy Metrics for {ticker}:")
    print(f"MAE:  {mae_one:.4f}")
    print(f"RMSE: {rmse_one:.4f}")
    print(f"MAPE: {mape_one:.2f}%")
    
    # Save One-Step Forecast Chart as PDF.
    plt.figure(figsize=(14, 7))
    plt.title('Stock Price Prediction (Test Data) - One Step Forecast')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(df.index[training_data_len:], y_test, label='Actual Price')
    plt.plot(df.index[training_data_len:], predictions_one_step_inv, label='Predicted Price')
    plt.legend()
    one_step_pdf = os.path.join(ticker_folder, f"Test_Data_Prediction_{ticker}.pdf")
    plt.savefig(one_step_pdf)
    plt.close()
    print(f"One-Step Forecast chart saved to {one_step_pdf}")
    
    # -------------------------------------------------------------------
    # SECTION B: MULTI-STEP FORECAST (Predict next 5 days Close)
    # -------------------------------------------------------------------
    x_train_multi = []
    y_train_multi = []
    for i in range(sequence_length, len(train_features) - forecast_horizon + 1):
        x_train_multi.append(train_features[i-sequence_length:i, :])
        y_train_multi.append(train_target[i:i+forecast_horizon, 0])
    x_train_multi = np.array(x_train_multi)
    y_train_multi = np.array(y_train_multi)
    print("\nMulti-output training set shapes for", ticker)
    print("x_train_multi shape:", x_train_multi.shape)    # Expected: (samples, 60, 2)
    print("y_train_multi shape:", y_train_multi.shape)      # Expected: (samples, forecast_horizon)
    
    # Build the multi-step forecasting model.
    model_multi_step = Sequential()
    model_multi_step.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_multi.shape[1], x_train_multi.shape[2])))
    model_multi_step.add(Dropout(0.2))
    model_multi_step.add(LSTM(units=50, return_sequences=False))
    model_multi_step.add(Dropout(0.2))
    model_multi_step.add(Dense(units=25))
    model_multi_step.add(Dense(units=forecast_horizon))  # Output: vector for next 5 days
    
    model_multi_step.compile(optimizer='adam', loss='mean_squared_error')
    print(f"\nMulti-Step Model summary for {ticker}:")
    model_multi_step.summary()
    
    print(f"Training multi-step forecast model for {ticker}...")
    model_multi_step.fit(x_train_multi, y_train_multi, batch_size=32, epochs=10, verbose=1)
    
    # Prepare test data for multi-step forecast.
    test_features_multi = scaled_features[training_data_len - sequence_length:, :]
    x_test_multi = []
    y_test_multi = []
    for i in range(sequence_length, len(test_features_multi) - forecast_horizon + 1):
        x_test_multi.append(test_features_multi[i-sequence_length:i, :])
        y_test_multi.append(scaled_target[training_data_len + i - sequence_length : training_data_len + i - sequence_length + forecast_horizon, 0])
    x_test_multi = np.array(x_test_multi)
    y_test_multi = np.array(y_test_multi)
    print("x_test_multi shape:", x_test_multi.shape)
    print("y_test_multi shape:", y_test_multi.shape)
    
    # Predict multi-step outputs.
    predictions_multi = model_multi_step.predict(x_test_multi)
    # Inverse transform: scaler_target expects 2D arrays.
    predictions_multi_inv = scaler_target.inverse_transform(predictions_multi)
    y_test_multi_inv = scaler_target.inverse_transform(y_test_multi)
    print("Multi-step predictions shape:", predictions_multi_inv.shape)
    
    # Accuracy metrics for multi-step forecasts (aggregated over all forecast horizons).
    mae_multi = mean_absolute_error(y_test_multi_inv.flatten(), predictions_multi_inv.flatten())
    mse_multi = mean_squared_error(y_test_multi_inv.flatten(), predictions_multi_inv.flatten())
    rmse_multi = np.sqrt(mse_multi)
    mape_multi = np.mean(np.abs((y_test_multi_inv.flatten() - predictions_multi_inv.flatten()) / y_test_multi_inv.flatten())) * 100
    
    print(f"\nMulti-Step Forecast Accuracy Metrics for {ticker}:")
    print(f"MAE:  {mae_multi:.4f}")
    print(f"RMSE: {rmse_multi:.4f}")
    print(f"MAPE: {mape_multi:.2f}%")
    
    # Save Multi-Step Forecast Chart (for one sample) as PDF.
    sample_index = 0
    plt.figure(figsize=(10, 5))
    plt.title('Multi-Step Forecast for One Test Sample')
    plt.xlabel('Forecast Horizon (Days)')
    plt.ylabel('Close Price USD ($)')
    forecast_days = np.arange(1, forecast_horizon + 1)
    plt.plot(forecast_days, y_test_multi_inv[sample_index], 'o-', label='Actual Future Prices')
    plt.plot(forecast_days, predictions_multi_inv[sample_index], 'o-', label='Predicted Future Prices')
    plt.legend()
    multi_step_pdf = os.path.join(ticker_folder, f"Multi_Step_Forecast_Sample_{ticker}.pdf")
    plt.savefig(multi_step_pdf)
    plt.close()
    print(f"Multi-Step Forecast chart saved to {multi_step_pdf}")
    
    # Collect summary statistics for this ticker.
    summary_stats[ticker] = {
        "Single Step": {"MAE": mae_one, "RMSE": rmse_one, "MAPE": mape_one},
        "Multi-Step":  {"MAE": mae_multi, "RMSE": rmse_multi, "MAPE": mape_multi}
    }

# -----------------------------
# FINAL: Build and Print Summary Table
# -----------------------------
rows = []
for ticker, stats in summary_stats.items():
    row = {("Single Step", "MAE"): stats["Single Step"]["MAE"],
           ("Single Step", "RMSE"): stats["Single Step"]["RMSE"],
           ("Single Step", "MAPE"): stats["Single Step"]["MAPE"],
           ("Multi-Step", "MAE"): stats["Multi-Step"]["MAE"],
           ("Multi-Step", "RMSE"): stats["Multi-Step"]["RMSE"],
           ("Multi-Step", "MAPE"): stats["Multi-Step"]["MAPE"],
           "Ticker": ticker}
    rows.append(row)

df_summary = pd.DataFrame(rows)
df_summary.set_index("Ticker", inplace=True)
df_summary.columns = pd.MultiIndex.from_tuples(df_summary.columns)

# Append an "Average" row for MAPE values.
avg_single_mape = df_summary[("Single Step", "MAPE")].mean()
avg_multi_mape  = df_summary[("Multi-Step", "MAPE")].mean()

avg_row = {
    ("Single Step", "MAE"): np.nan,
    ("Single Step", "RMSE"): np.nan,
    ("Single Step", "MAPE"): avg_single_mape,
    ("Multi-Step", "MAE"): np.nan,
    ("Multi-Step", "RMSE"): np.nan,
    ("Multi-Step", "MAPE"): avg_multi_mape
}
df_summary.loc["Average"] = avg_row

# Reindex to move "Average" row to the bottom.
new_index = list(df_summary.index[df_summary.index != "Average"]) + ["Average"]
df_summary = df_summary.reindex(new_index)

print("\nSummary Table of Accuracy Metrics for All Tickers:")
print(df_summary.to_string(float_format="%.4f"))

# Optionally, save the summary table to CSV.
summary_csv = os.path.join(output_root, "Summary_Accuracy_Metrics.csv")
df_summary.to_csv(summary_csv)
print(f"\nSummary table saved to {summary_csv}")