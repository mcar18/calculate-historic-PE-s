import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# -----------------------------
# SETTINGS
# -----------------------------
tickers = ['SPY', 'AAPL', 'MSFT']  # Expand this list as needed
start_date = '2010-01-01'
end_date   = '2020-12-31'
sequence_length = 60      # Number of prior days used as input
forecast_horizon = 10     # Predict next 10 days (for multi-step)
output_root = 'RNN forecasts'
os.makedirs(output_root, exist_ok=True)

# Choose scaling method: "robust" (global) or "rolling" (rolling window normalization)
scaling_method = "rolling"  # Change to "robust" for global scaling
rolling_window_size = 20    # Window size for rolling normalization

# Feature and target column names:
feature_cols = ['Close', 'Volume', 'Volume_MA20', 'Volume_Ratio',
                'MA20', 'MA50', 'MA200', 'Pct_Move',
                'SPY_Close', 'SPY_MA20', 'SPY_MA50', 'SPY_MA200']
target_col = ['Close']

# -----------------------------
# Function: Rolling Window Normalization
# -----------------------------
def rolling_normalize_columns(df, cols, window):
    """
    For each column in cols, computes the rolling median and IQR over the specified window,
    and returns a normalized DataFrame along with DataFrames of the rolling medians and IQRs.
    """
    df_norm = df.copy()
    median_dict = {}
    iqr_dict = {}
    for col in cols:
        median = df[col].rolling(window=window, min_periods=window).median()
        q75 = df[col].rolling(window=window, min_periods=window).quantile(0.75)
        q25 = df[col].rolling(window=window, min_periods=window).quantile(0.25)
        iqr = q75 - q25
        df_norm[col] = (df[col] - median) / iqr
        median_dict[col] = median
        iqr_dict[col] = iqr
    df_norm = df_norm.dropna()
    median_df = pd.DataFrame({col: median_dict[col] for col in cols})
    iqr_df = pd.DataFrame({col: iqr_dict[col] for col in cols})
    median_df = median_df.loc[df_norm.index]
    iqr_df = iqr_df.loc[df_norm.index]
    return df_norm, median_df, iqr_df

# -----------------------------
# Download SPY Data and Compute Its Moving Averages
# -----------------------------
print("Downloading SPY data...")
spy_data = yf.download("SPY", start=start_date, end=end_date)
if isinstance(spy_data.columns, pd.MultiIndex):
    spy_close = spy_data['Close']
    if isinstance(spy_close, pd.DataFrame):
        spy_close.columns = ['SPY_Close']
        spy_data = spy_close
    else:
        spy_data = spy_close.to_frame(name='SPY_Close')
else:
    spy_data = spy_data[['Close']].copy()
    spy_data.rename(columns={'Close': 'SPY_Close'}, inplace=True)
# Compute SPY moving averages
spy_data['SPY_MA20'] = spy_data['SPY_Close'].rolling(window=20).mean()
spy_data['SPY_MA50'] = spy_data['SPY_Close'].rolling(window=50).mean()
spy_data['SPY_MA200'] = spy_data['SPY_Close'].rolling(window=200).mean()
spy_data.dropna(inplace=True)
print("SPY data downloaded and processed with moving averages.")

# Dictionary to collect summary metrics
summary_stats = {}

# -----------------------------
# PROCESS EACH TICKER
# -----------------------------
for ticker in tickers:
    print("\n" + "="*60)
    print(f"Processing data for ticker: {ticker}")
    
    # Create ticker folder
    ticker_folder = os.path.join(output_root, ticker)
    os.makedirs(ticker_folder, exist_ok=True)
    
    # -----------------------------
    # 1. Download Ticker Data and Basic Features
    # -----------------------------
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    print("Data downloaded. Columns:")
    print(data.columns)
    
    # Handle MultiIndex if needed
    if isinstance(data.columns, pd.MultiIndex):
        print("Detected MultiIndex columns. Extracting 'Close' and 'Volume'...")
        if 'Close' in data.columns.levels[0] and 'Volume' in data.columns.levels[0]:
            df_close = data['Close']
            df_volume = data['Volume']
            if ticker in df_close.columns and ticker in df_volume.columns:
                print(f"Selecting '{ticker}' columns for Close and Volume.")
                df = pd.DataFrame({'Close': df_close[ticker],
                                   'Volume': df_volume[ticker]})
            else:
                print("Ticker not found in sub-columns; using first available columns.")
                df = pd.DataFrame({'Close': df_close.iloc[:,0],
                                   'Volume': df_volume.iloc[:,0]})
        else:
            raise ValueError("MultiIndex detected but required columns not found.")
    else:
        if 'Close' in data.columns and 'Volume' in data.columns:
            df = data[['Close', 'Volume']].copy()
        elif 'Adj Close' in data.columns and 'Volume' in data.columns:
            df = data[['Adj Close', 'Volume']].copy()
            df.rename(columns={'Adj Close': 'Close'}, inplace=True)
        else:
            raise ValueError("Required columns not found in data.")
    
    print("DataFrame head:")
    print(df.head())
    print("DataFrame shape:", df.shape)
    
    # -----------------------------
    # 2. Additional Features
    # -----------------------------
    # Ticker’s moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    # Daily percentage change (Pct_Move)
    df['Pct_Move'] = df['Close'].pct_change() * 100
    # Volume features: 20-day moving average and ratio
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
    
    # Merge SPY data into ticker DataFrame by date
    df = df.merge(spy_data, left_index=True, right_index=True, how='left')
    df.dropna(inplace=True)
    
    # -----------------------------
    # 3. Scaling / Normalization
    # -----------------------------
    # Our feature set and target:
    # Features: Ticker’s Close, Volume, Volume_MA20, Volume_Ratio, MA20, MA50, MA200, Pct_Move,
    #           SPY_Close, SPY_MA20, SPY_MA50, SPY_MA200
    # Target: Ticker’s Close
    # We'll use our chosen scaling method.
    if scaling_method == "robust":
        print("Using global RobustScaler...")
        features = df[feature_cols].values
        target = df[target_col].values
        
        scaler_features = RobustScaler()
        scaler_target = RobustScaler()
        scaled_features = scaler_features.fit_transform(features)
        scaled_target = scaler_target.fit_transform(target)
    elif scaling_method == "rolling":
        print("Using rolling window normalization...")
        all_cols = feature_cols + target_col
        df_norm, roll_medians, roll_iqrs = rolling_normalize_columns(df, all_cols, rolling_window_size)
        df = df_norm
        features = df[feature_cols].values
        target = df[target_col].values
        scaled_features = features  # Data is already normalized.
        scaled_target = target
        # Store rolling medians and IQRs for the target column ("Close")
        rolling_target_median = roll_medians[target_col[0]].values
        rolling_target_iqr = roll_iqrs[target_col[0]].values
    else:
        raise ValueError("Invalid scaling_method. Use 'robust' or 'rolling'.")
    
    print("Scaled features shape:", scaled_features.shape)
    print("Scaled target shape:", scaled_target.shape)
    
    # -----------------------------
    # 4. Train/Test Split
    # -----------------------------
    training_data_len = int(np.ceil(len(df) * 0.8))
    print("Training data length (rows):", training_data_len)
    
    # -------------------------------------------------------------------
    # SECTION A: ONE-STEP FORECAST
    # -------------------------------------------------------------------
    train_features = scaled_features[:training_data_len, :]
    train_target = scaled_target[:training_data_len, :]
    
    x_train = []
    y_train = []
    for i in range(sequence_length, len(train_features)):
        x_train.append(train_features[i-sequence_length:i, :])
        y_train.append(train_target[i, 0])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print("x_train (one-step) shape:", x_train.shape)
    print("y_train (one-step) shape:", y_train.shape)
    
    model_one_step = Sequential()
    model_one_step.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model_one_step.add(Dropout(0.2))
    model_one_step.add(LSTM(units=50, return_sequences=False))
    model_one_step.add(Dropout(0.2))
    model_one_step.add(Dense(units=25))
    model_one_step.add(Dense(units=1))
    model_one_step.compile(optimizer='adam', loss='mean_squared_error')
    print(f"\nOne-Step Model summary for {ticker}:")
    model_one_step.summary()
    
    print(f"Training one-step forecast model for {ticker}...")
    model_one_step.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)
    
    test_features = scaled_features[training_data_len - sequence_length:, :]
    x_test = []
    for i in range(sequence_length, len(test_features)):
        x_test.append(test_features[i-sequence_length:i, :])
    x_test = np.array(x_test)
    print("x_test (one-step) shape:", x_test.shape)
    
    y_test = target[training_data_len:, :]
    predictions_one_step = model_one_step.predict(x_test)
    if scaling_method == "robust":
        predictions_one_step_inv = scaler_target.inverse_transform(predictions_one_step)
        y_test_inv = scaler_target.inverse_transform(y_test)
    elif scaling_method == "rolling":
        # Inverse transform using the stored rolling stats; note that the first prediction corresponds to df.index[training_data_len]
        r_median_test = rolling_target_median[training_data_len:]
        r_iqr_test = rolling_target_iqr[training_data_len:]
        predictions_one_step_inv = predictions_one_step.flatten() * r_iqr_test + r_median_test
        y_test_inv = y_test.flatten() * r_iqr_test + r_median_test
    print("Predictions (one-step) shape:", predictions_one_step_inv.shape)
    
    mae_one = mean_absolute_error(y_test_inv, predictions_one_step_inv)
    mse_one = mean_squared_error(y_test_inv, predictions_one_step_inv)
    rmse_one = np.sqrt(mse_one)
    mape_one = np.mean(np.abs((y_test_inv - predictions_one_step_inv) / y_test_inv)) * 100
    
    print(f"\nOne-Step Forecast Accuracy Metrics for {ticker}:")
    print(f"MAE:  {mae_one:.4f}")
    print(f"RMSE: {rmse_one:.4f}")
    print(f"MAPE: {mape_one:.2f}%")
    
    plt.figure(figsize=(14,7))
    plt.title('Stock Price Prediction (Test Data) - One Step Forecast')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(df.index[training_data_len:], y_test_inv, label='Actual Price')
    plt.plot(df.index[training_data_len:], predictions_one_step_inv, label='Predicted Price')
    plt.legend()
    one_step_pdf = os.path.join(ticker_folder, f"Test_Data_Prediction_{ticker}.pdf")
    plt.savefig(one_step_pdf)
    plt.close()
    print(f"One-Step Forecast chart saved to {one_step_pdf}")
    
    # -------------------------------------------------------------------
    # SECTION B: MULTI-STEP FORECAST
    # -------------------------------------------------------------------
    x_train_multi = []
    y_train_multi = []
    for i in range(sequence_length, len(train_features) - forecast_horizon + 1):
        x_train_multi.append(train_features[i-sequence_length:i, :])
        y_train_multi.append(train_target[i:i+forecast_horizon, 0])
    x_train_multi = np.array(x_train_multi)
    y_train_multi = np.array(y_train_multi)
    print("\nMulti-output training set shapes for", ticker)
    print("x_train_multi shape:", x_train_multi.shape)
    print("y_train_multi shape:", y_train_multi.shape)
    
    model_multi_step = Sequential()
    model_multi_step.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_multi.shape[1], x_train_multi.shape[2])))
    model_multi_step.add(Dropout(0.2))
    model_multi_step.add(LSTM(units=50, return_sequences=False))
    model_multi_step.add(Dropout(0.2))
    model_multi_step.add(Dense(units=25))
    model_multi_step.add(Dense(units=forecast_horizon))
    model_multi_step.compile(optimizer='adam', loss='mean_squared_error')
    print(f"\nMulti-Step Model summary for {ticker}:")
    model_multi_step.summary()
    
    print(f"Training multi-step forecast model for {ticker}...")
    model_multi_step.fit(x_train_multi, y_train_multi, batch_size=32, epochs=10, verbose=1)
    
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
    
    predictions_multi = model_multi_step.predict(x_test_multi)
    if scaling_method == "robust":
        predictions_multi_inv = scaler_target.inverse_transform(predictions_multi)
        y_test_multi_inv = scaler_target.inverse_transform(y_test_multi)
    elif scaling_method == "rolling":
        raw_preds_multi = []
        raw_y_test_multi = []
        # For multi-step, the first sample corresponds to index training_data_len in df.
        for i in range(y_test_multi.shape[0]):
            j = training_data_len + i
            # Get rolling stats for the forecast horizon
            median_slice = rolling_target_median[j:j+forecast_horizon]
            iqr_slice = rolling_target_iqr[j:j+forecast_horizon]
            raw_pred = predictions_multi[i] * iqr_slice + median_slice
            raw_y = y_test_multi[i] * iqr_slice + median_slice
            raw_preds_multi.append(raw_pred)
            raw_y_test_multi.append(raw_y)
        raw_preds_multi = np.array(raw_preds_multi)
        raw_y_test_multi = np.array(raw_y_test_multi)
        predictions_multi_inv = raw_preds_multi
        y_test_multi_inv = raw_y_test_multi
    print("Multi-step predictions shape:", predictions_multi_inv.shape)
    
    mae_multi = mean_absolute_error(y_test_multi_inv.flatten(), predictions_multi_inv.flatten())
    mse_multi = mean_squared_error(y_test_multi_inv.flatten(), predictions_multi_inv.flatten())
    rmse_multi = np.sqrt(mse_multi)
    mape_multi = np.mean(np.abs((y_test_multi_inv.flatten() - predictions_multi_inv.flatten()) / y_test_multi_inv.flatten())) * 100
    
    print(f"\nMulti-Step Forecast Accuracy Metrics for {ticker}:")
    print(f"MAE:  {mae_multi:.4f}")
    print(f"RMSE: {rmse_multi:.4f}")
    print(f"MAPE: {mape_multi:.2f}%")
    
    sample_index = 0
    plt.figure(figsize=(10,5))
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
    
    summary_stats[ticker] = {
        "Single Step": {"MAE": mae_one, "RMSE": rmse_one, "MAPE": mape_one},
        "Multi-Step": {"MAE": mae_multi, "RMSE": rmse_multi, "MAPE": mape_multi}
    }

# -----------------------------
# Build and Print Summary Table
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
avg_single_mape = df_summary[("Single Step", "MAPE")].mean()
avg_multi_mape = df_summary[("Multi-Step", "MAPE")].mean()
avg_row = {("Single Step", "MAE"): np.nan,
           ("Single Step", "RMSE"): np.nan,
           ("Single Step", "MAPE"): avg_single_mape,
           ("Multi-Step", "MAE"): np.nan,
           ("Multi-Step", "RMSE"): np.nan,
           ("Multi-Step", "MAPE"): avg_multi_mape}
df_summary.loc["Average"] = avg_row
new_index = list(df_summary.index[df_summary.index != "Average"]) + ["Average"]
df_summary = df_summary.reindex(new_index)
print("\nSummary Table of Accuracy Metrics for All Tickers:")
print(df_summary.to_string(float_format="%.4f"))
summary_csv = os.path.join(output_root, "Summary_Accuracy_Metrics.csv")
df_summary.to_csv(summary_csv)
print(f"\nSummary table saved to {summary_csv}")