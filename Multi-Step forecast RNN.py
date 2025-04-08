import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# -----------------------------
# Download data and prepare the "Close" prices
# -----------------------------
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2020-12-31'

print("Downloading data for ticker:", ticker)
data = yf.download(ticker, start=start_date, end=end_date)
print("\nData downloaded. Columns:")
print(data.columns)

if isinstance(data.columns, pd.MultiIndex):
    print("\nDetected MultiIndex columns. Extracting 'Close' prices...")
    if 'Close' in data.columns.levels[0]:
        df = data['Close']
        if ticker in df.columns:
            print(f"Selecting the '{ticker}' column from the 'Close' data.")
            df = df[[ticker]]
            df.columns = ['Close']
        else:
            print("Ticker not found in sub-columns, using first available column.")
            df = df.iloc[:, 0].to_frame(name='Close')
    else:
        raise ValueError("MultiIndex detected but 'Close' is not in the primary level.")
else:
    if 'Close' in data.columns:
        df = data[['Close']].copy()
    elif 'Adj Close' in data.columns:
        df = data[['Adj Close']].copy()
        df.rename(columns={'Adj Close': 'Close'}, inplace=True)
    else:
        raise ValueError("No 'Close' or 'Adj Close' found.")

print("\nDataFrame head (Close prices):")
print(df.head())
print("\nDataFrame tail:")
print(df.tail())
print("DataFrame shape:", df.shape)

dataset = df.values

# -----------------------------
# Scale the data and split into training and test sets
# -----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
print("\nScaled data shape:", scaled_data.shape)

training_data_len = int(np.ceil(len(dataset) * 0.8))
print("Training data length (rows):", training_data_len)

train_data = scaled_data[0:training_data_len, :]
sequence_length = 60  # using the prior 60 days to predict the next day
x_train, y_train = [], []
for i in range(sequence_length, len(train_data)):
    x_train.append(train_data[i-sequence_length:i, 0])
    y_train.append(train_data[i, 0])
x_train = np.array(x_train)
y_train = np.array(y_train)
print("x_train shape (before reshape):", x_train.shape)
print("y_train shape:", y_train.shape)

# Reshape for LSTM [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print("x_train shape after reshape:", x_train.shape)

# -----------------------------
# Build and train the one-step forecasting LSTM model
# -----------------------------
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))  # one output: next-day prediction

model.compile(optimizer='adam', loss='mean_squared_error')
print("\nModel summary:")
model.summary()

print("\nTraining one-step forecast model...")
model.fit(x_train, y_train, batch_size=32, epochs=10)

# -----------------------------
# Prepare test data
# -----------------------------
test_data = scaled_data[training_data_len - sequence_length:, :]
x_test = []
for i in range(sequence_length, len(test_data)):
    x_test.append(test_data[i-sequence_length:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print("x_test shape:", x_test.shape)

y_test = dataset[training_data_len:, :]  # actual prices for test period

# -----------------------------
# Predict and measure accuracy
# -----------------------------
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
print("Predictions shape:", predictions.shape)

# Calculate error metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

print("\nOne-Step Forecast Accuracy Metrics:")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

# -----------------------------
# (Optional) Walk-Forward Validation Concept:
# A walk-forward validation would iterate over each time step in the test set, updating the model input as new actual data becomes available.
# For example:
#
# errors = []
# history_scaled = list(train_data)
# for t in range(len(y_test)):
#     x_input = np.array(history_scaled[-sequence_length:]).reshape(1, sequence_length, 1)
#     yhat = model.predict(x_input)
#     # In a rolling forecast, you could append the actual observed test value here.
#     history_scaled.append(test_data[t + sequence_length])  
#     errors.append(np.abs(y_test[t] - scaler.inverse_transform(yhat)))
# Aggregate errors could then be computed over the test set.
#
# -----------------------------
# Plot test data: Actual vs. Predicted Prices
plt.figure(figsize=(14, 7))
plt.title('Stock Price Prediction (Test Data) - One Step Forecast')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(df.index[training_data_len:], y_test, label='Actual Price')
plt.plot(df.index[training_data_len:], predictions, label='Predicted Price')
plt.legend()
plt.show()


# -----------------------------
# Multi-Output Forecasting Setup
# -----------------------------
forecast_horizon = 5  # For example, predict the next 5 days

# To build multi-output data, adjust the training samples creation.
x_train_multi, y_train_multi = [], []
# Note: Use len(train_data) - forecast_horizon to avoid index overflow
for i in range(sequence_length, len(train_data) - forecast_horizon + 1):
    x_train_multi.append(train_data[i-sequence_length:i, 0])
    # Create a vector of the next 'forecast_horizon' days
    y_train_multi.append(train_data[i:i+forecast_horizon, 0])
x_train_multi = np.array(x_train_multi)
y_train_multi = np.array(y_train_multi)
print("\nMulti-output training set shapes:")
print("x_train_multi shape (before reshape):", x_train_multi.shape)  # (samples, 60)
print("y_train_multi shape:", y_train_multi.shape)                # (samples, forecast_horizon)

# Reshape x_train_multi for LSTM input
x_train_multi = np.reshape(x_train_multi, (x_train_multi.shape[0], x_train_multi.shape[1], 1))
print("x_train_multi shape after reshape:", x_train_multi.shape)

# Build the multi-output LSTM model
model_multi = Sequential()
model_multi.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_multi.shape[1], 1)))
model_multi.add(Dropout(0.2))
model_multi.add(LSTM(units=50, return_sequences=False))
model_multi.add(Dropout(0.2))
model_multi.add(Dense(units=25))
model_multi.add(Dense(units=forecast_horizon))  # Now output is a vector of length 'forecast_horizon'

model_multi.compile(optimizer='adam', loss='mean_squared_error')
print("\nMulti-output Model summary:")
model_multi.summary()

print("\nTraining multi-output forecast model...")
model_multi.fit(x_train_multi, y_train_multi, batch_size=32, epochs=10)

# -----------------------------
# Prepare multi-output test data
# -----------------------------
# For the test data, we must similarly create multi-step target vectors.
test_data_multi = scaled_data[training_data_len - sequence_length:, :]
x_test_multi, y_test_multi = [], []
for i in range(sequence_length, len(test_data_multi) - forecast_horizon + 1):
    x_test_multi.append(test_data_multi[i-sequence_length:i, 0])
    y_test_multi.append(test_data_multi[i:i+forecast_horizon, 0])
x_test_multi = np.array(x_test_multi)
y_test_multi = np.array(y_test_multi)
x_test_multi = np.reshape(x_test_multi, (x_test_multi.shape[0], x_test_multi.shape[1], 1))
print("x_test_multi shape:", x_test_multi.shape)
print("y_test_multi shape:", y_test_multi.shape)

# Generate multi-step predictions
multi_predictions = model_multi.predict(x_test_multi)
multi_predictions = scaler.inverse_transform(multi_predictions)
# To compare, convert y_test_multi back to original scale.
y_test_multi_original = scaler.inverse_transform(y_test_multi)

print("Multi-output predictions shape:", multi_predictions.shape)

# Calculate error metrics for the multi-output forecast.
# Here, we simply calculate metrics across all forecast horizons.
mae_multi = mean_absolute_error(y_test_multi_original.flatten(), multi_predictions.flatten())
mse_multi = mean_squared_error(y_test_multi_original.flatten(), multi_predictions.flatten())
rmse_multi = np.sqrt(mse_multi)
mape_multi = np.mean(np.abs((y_test_multi_original.flatten() - multi_predictions.flatten()) / y_test_multi_original.flatten())) * 100

print("\nMulti-Step Forecast Accuracy Metrics (aggregated over forecast horizon):")
print(f"MAE:  {mae_multi:.4f}")
print(f"RMSE: {rmse_multi:.4f}")
print(f"MAPE: {mape_multi:.2f}%")

# -----------------------------
# Plot multi-step predictions for one example from the test set
# -----------------------------
# For illustration, plot the predicted vs. actual future values for the first multi-output sample.
sample_index = 0
plt.figure(figsize=(10, 5))
plt.title('Multi-Step Forecast for One Test Sample')
plt.xlabel('Forecast Horizon (Days)')
plt.ylabel('Close Price USD ($)')
forecast_days = np.arange(1, forecast_horizon + 1)
plt.plot(forecast_days, y_test_multi_original[sample_index], 'o-', label='Actual Future Prices')
plt.plot(forecast_days, multi_predictions[sample_index], 'o-', label='Predicted Future Prices')
plt.legend()
plt.show()