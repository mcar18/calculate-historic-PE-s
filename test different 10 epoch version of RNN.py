import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -----------------------------
# 1. Download and Debug the Data
# -----------------------------
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2020-12-31'

print("Downloading data for ticker:", ticker)
data = yf.download(ticker, start=start_date, end=end_date)
print("\nData downloaded. The columns are:\n", data.columns)

# -----------------------------
# 2. Extract and Prepare 'Close' Price Data with Debugging
# -----------------------------
# Handle MultiIndex columns if present
if isinstance(data.columns, pd.MultiIndex):
    print("\nDetected MultiIndex columns. Attempting to select the 'Close' prices...")
    if 'Close' in data.columns.levels[0]:
        # Extract the 'Close' column. This returns a DataFrame with ticker symbols as sub-columns.
        df = data['Close']
        if ticker in df.columns:
            print(f"Selecting '{ticker}' column from the 'Close' data.")
            df = df[[ticker]]
            df.columns = ['Close']  # Rename column for consistency in later code.
        else:
            # If the ticker is not found in the sub-columns, take the first available column.
            print("Specified ticker not found in the sub-columns. Using the first available column.")
            df = df.iloc[:, 0].to_frame(name='Close')
    else:
        raise ValueError("MultiIndex columns detected, but 'Close' is not present in the first level.")
else:
    # For DataFrames without MultiIndex, try 'Close' or 'Adj Close'
    if 'Close' in data.columns:
        df = data[['Close']].copy()
    elif 'Adj Close' in data.columns:
        df = data[['Adj Close']].copy()
        df.rename(columns={'Adj Close': 'Close'}, inplace=True)
    else:
        raise ValueError("Neither 'Close' nor 'Adj Close' columns were found in the data.")

print("\nDataFrame head after selecting the 'Close' column:")
print(df.head())
print("Shape of the DataFrame:", df.shape)

# Convert DataFrame values into a NumPy array
dataset = df.values

# -----------------------------
# 3. Scaling and Creating Training Data with Debugging
# -----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
print("\nScaled data shape:", scaled_data.shape)

# Use 80% of the data for training
training_data_len = int(np.ceil(len(dataset) * 0.8))
print("Training data length:", training_data_len)

train_data = scaled_data[0:training_data_len, :]

x_train, y_train = [], []
sequence_length = 60  # Use 60 past time-steps (days) for each prediction

for i in range(sequence_length, len(train_data)):
    x_train.append(train_data[i-sequence_length:i, 0])
    y_train.append(train_data[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)
print("x_train shape (before reshape):", x_train.shape)
print("y_train shape:", y_train.shape)

# Reshape x_train for the LSTM: [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print("x_train shape after reshape:", x_train.shape)

# -----------------------------
# 4. Building and Training the LSTM Model with Debugging
# -----------------------------
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

# -----------------------------
# 5. Testing the Model and Generating Predictions with Debugging
# -----------------------------
print("\nPreparing test data...")
test_data = scaled_data[training_data_len - sequence_length:, :]
x_test = []
y_test = dataset[training_data_len:, :]  # Actual close prices for test period

for i in range(sequence_length, len(test_data)):
    x_test.append(test_data[i-sequence_length:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print("x_test shape:", x_test.shape)

print("Generating predictions...")
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
print("Predictions shape:", predictions.shape)

# -----------------------------
# 6. Visualizing the Results
# -----------------------------
plt.figure(figsize=(14, 7))
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')

# Plot actual close prices and predictions.
plt.plot(df.index[training_data_len:], y_test, label='Actual Price')
plt.plot(df.index[training_data_len:], predictions, label='Predicted Price')
plt.legend()
plt.show()