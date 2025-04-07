# Import required libraries
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -----------------------------
# 1. Downloading the Data
# -----------------------------
# Set the ticker symbol and the period for the historical data
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2020-12-31'

# Download the historical data from yfinance
data = yf.download(ticker, start=start_date, end=end_date)

# Focus on the 'Close' prices for prediction
df = data.filter(['Close'])
dataset = df.values

# -----------------------------
# 2. Data Preparation
# -----------------------------
# Define training data length (e.g., 80% of dataset for training)
training_data_len = int(np.ceil(len(dataset) * 0.8))

# Scale the data to be in the range (0, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create the training dataset and generate sequences of 60 time-steps
train_data = scaled_data[0:training_data_len, :]
x_train, y_train = [], []
sequence_length = 60  # number of previous time-steps to use for prediction

for i in range(sequence_length, len(train_data)):
    x_train.append(train_data[i-sequence_length:i, 0])
    y_train.append(train_data[i, 0])

# Convert the training data to numpy arrays and reshape for LSTM input
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# -----------------------------
# 3. Building the LSTM Model
# -----------------------------
model = Sequential()

# First LSTM layer with Dropout regularization
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

# Second LSTM layer with Dropout regularization
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Dense layers: one hidden dense layer and one output layer
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model using 'adam' optimizer and mean squared error loss function
model.compile(optimizer='adam', loss='mean_squared_error')

# -----------------------------
# 4. Training the Model
# -----------------------------
# You can experiment with the number of epochs and batch size
model.fit(x_train, y_train, batch_size=32, epochs=10)

# -----------------------------
# 5. Testing and Predicting
# -----------------------------
# Create the testing dataset by appending the last 60 days of training data
test_data = scaled_data[training_data_len - sequence_length:, :]
x_test, y_test = [], dataset[training_data_len:, :]

for i in range(sequence_length, len(test_data)):
    x_test.append(test_data[i-sequence_length:i, 0])

# Convert test data to numpy array and reshape it
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the model's predicted price values and revert scaling
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# -----------------------------
# 6. Visualization of Results
# -----------------------------
plt.figure(figsize=(14, 7))
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')

# Plot the actual closing prices
plt.plot(df.index[training_data_len:], y_test, label='Actual Price')

# Plot the predicted closing prices
plt.plot(df.index[training_data_len:], predictions, label='Predicted Price')
plt.legend()
plt.show()