import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# ------------------------------------------
# 1. Downloading Data and Debugging Output
# ------------------------------------------
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2020-12-31'

print("Downloading data for ticker:", ticker)
data = yf.download(ticker, start=start_date, end=end_date)
print("\nData downloaded. Columns:")
print(data.columns)

# If data has a MultiIndex for columns (as is now common with yfinance),
# extract the "Close" price.
if isinstance(data.columns, pd.MultiIndex):
    print("\nDetected MultiIndex columns. Extracting 'Close' prices...")
    if 'Close' in data.columns.levels[0]:
        df = data['Close']
        # Check if the desired ticker exists in the sub-columns
        if ticker in df.columns:
            print(f"Selecting the '{ticker}' column from the 'Close' data.")
            df = df[[ticker]]
            df.columns = ['Close']  # Rename for later consistency
        else:
            print("Ticker not found in sub-columns, using first available column.")
            df = df.iloc[:, 0].to_frame(name='Close')
    else:
        raise ValueError("MultiIndex detected but 'Close' is not one of the primary levels.")
else:
    # Handle non-MultiIndex DataFrames by checking for 'Close' or 'Adj Close'
    if 'Close' in data.columns:
        df = data[['Close']].copy()
    elif 'Adj Close' in data.columns:
        df = data[['Adj Close']].copy()
        df.rename(columns={'Adj Close': 'Close'}, inplace=True)
    else:
        raise ValueError("No 'Close' or 'Adj Close' column found in data.")

print("\nDataFrame head after selecting 'Close':")
print(df.head())
print("\nDataFrame tail:")
print(df.tail())
print("DataFrame shape:", df.shape)

# Convert the DataFrame into a NumPy array
dataset = df.values

# ------------------------------------------
# 2. Scaling and Preparing the Data
# ------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
print("\nScaled data shape:", scaled_data.shape)

# Define training data as 80% of the overall data
training_data_len = int(np.ceil(len(dataset) * 0.8))
print("Training data length (number of rows):", training_data_len)

train_data = scaled_data[0:training_data_len, :]

# Create sequences of 60 time-steps for training: each sample uses 60 previous days to predict the next day
sequence_length = 60  # using the prior 60 days of trading
x_train, y_train = [], []
for i in range(sequence_length, len(train_data)):
    x_train.append(train_data[i-sequence_length:i, 0])
    y_train.append(train_data[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)
print("x_train shape (before reshape):", x_train.shape)
print("y_train shape:", y_train.shape)

# Reshape x_train to be [samples, time steps, features] for the LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print("x_train shape after reshape:", x_train.shape)

# ------------------------------------------
# 3. Building and Training the LSTM Model
# ------------------------------------------
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
print("\nModel summary:")
model.summary()

print("\nTraining model...")
history = model.fit(x_train, y_train, batch_size=32, epochs=10)

# ------------------------------------------
# 4. Visualization: Historic Training Data Chart
# ------------------------------------------
# Plot the historical stock price (training period only)
plt.figure(figsize=(14, 7))
plt.title('Historic Stock Price (Training Data)')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')

# Plot only the training period. We start the plot at index equal to the sequence length,
# since the first 'sequence_length' rows were used to form the first training sample.
plt.plot(df.index[sequence_length:training_data_len], dataset[sequence_length:training_data_len], label='Training Actual Price')
plt.legend()
plt.show()

# ------------------------------------------
# 5. Prepare Test Data and Visualization: Predictions vs. Actual
# ------------------------------------------
# Prepare the test data. Notice we include the last 'sequence_length' points from training
# to form the first test sample.
test_data = scaled_data[training_data_len - sequence_length:, :]
x_test = []
for i in range(sequence_length, len(test_data)):
    x_test.append(test_data[i-sequence_length:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print("x_test shape:", x_test.shape)

# Get the actual prices for the test period
y_test = dataset[training_data_len:, :]

# Generate predictions using the trained model
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
print("Predictions shape:", predictions.shape)

# Plot the actual vs. predicted prices for the test data.
plt.figure(figsize=(14, 7))
plt.title('Stock Price Prediction (Test Data)')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(df.index[training_data_len:], y_test, label='Actual Price')
plt.plot(df.index[training_data_len:], predictions, label='Predicted Price')
plt.legend()
plt.show()