# Import required libraries
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -----------------------------
# 1. Download the Data
# -----------------------------
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2020-12-31'

# Download the historical data with default auto_adjust=True
data = yf.download(ticker, start=start_date, end=end_date)
print("Downloaded Data Columns:", data.columns)

# -----------------------------
# 2. Extract the 'Close' Column with Debugging Checks
# -----------------------------
# If columns are MultiIndex, extract 'Close' values accordingly.
if isinstance(data.columns, pd.MultiIndex):
    # Extract the 'Close' data; this will drop the ticker level
    df = data['Close']
    print("Extracted 'Close' data from MultiIndex, columns now:", df.columns)
else:
    # For non-MultiIndex, check for 'Close' or 'Adj Close'
    if 'Close' in data.columns:
        df = data.filter(['Close'])
    elif 'Adj Close' in data.columns:
        df = data.filter(['Adj Close']).rename(columns={'Adj Close': 'Close'})
    else:
        raise ValueError("Neither 'Close' nor 'Adj Close' columns were found in the data.")
        
# Ensure we have a DataFrame with one column. If it's a Series, convert it.
if isinstance(df, pd.Series):
    df = df.to_frame(name='Close')

# Debugging: print first few rows and shape of dataframe
print("First 5 rows of df:\n", df.head())
print("Shape of df:", df.shape)

# Convert to numpy array and ensure it has shape (n, 1)
dataset = df.values.reshape(-1, 1)
print("Shape of dataset after reshaping:", dataset.shape)

# -----------------------------
# 3. Data Preparation
# -----------------------------
training_data_len = int(np.ceil(len(dataset) * 0.8))
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Debug: verify scaling worked
print("Scaled data shape:", scaled_data.shape)

train_data = scaled_data[0:training_data_len, :]
x_train, y_train = [], []
sequence_length = 60  # using the previous 60 time-steps

for i in range(sequence_length, len(train_data)):
    x_train.append(train_data[i - sequence_length:i, 0])
    y_train.append(train_data[i, 0])

# Convert lists to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

# Reshape input to be [samples, time_steps, features] for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

# -----------------------------
# 4. Build and Train the LSTM Model
# -----------------------------
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=32, epochs=10)

# -----------------------------
# 5. Testing and Prediction
# -----------------------------
test_data = scaled_data[training_data_len - sequence_length:, :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(sequence_length, len(test_data)):
    x_test.append(test_data[i - sequence_length:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# Get model predictions and invert the scaling
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# -----------------------------
# 6. Visualization
# -----------------------------
plt.figure(figsize=(14, 7))
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')

# Since the df index carries date information, slice it for the test portion
plt.plot(df.index[training_data_len:], y_test, label='Actual Price')
plt.plot(df.index[training_data_len:], predictions, label='Predicted Price')
plt.legend()
plt.show()