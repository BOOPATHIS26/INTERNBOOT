# ================================================
# Stock Market Price Prediction using LSTM
# ================================================

# Step 1: Install required libraries
# pip install yfinance pandas numpy matplotlib scikit-learn tensorflow

# Step 2: Import Libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 3: Load Historical Stock Data
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2025-01-01', auto_adjust=False)

# Keep only standard OHLCV columns
data = data[['Open','High','Low','Close','Volume']]

print("Historical Stock Data:")
print(data.head())

# Step 4: Exploratory Data Analysis (EDA)
plt.figure(figsize=(12,6))
plt.plot(data['Close'], color='blue')
plt.title(f'{ticker} Stock Closing Price')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.show()

# Optional: Moving averages
data['MA50'] = data['Close'].rolling(50).mean()
data['MA200'] = data['Close'].rolling(200).mean()

plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['MA50'], label='50-day MA')
plt.plot(data['MA200'], label='200-day MA')
plt.legend()
plt.show()

# Step 5: Preprocessing for LSTM
close_prices = data['Close'].values.reshape(-1,1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_prices)

# Create sequences for LSTM
sequence_length = 60
x_train = []
y_train = []

for i in range(sequence_length, len(scaled_data)):
    x_train.append(scaled_data[i-sequence_length:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Step 6: Build LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Step 7: Train the Model
model.fit(x_train, y_train, batch_size=32, epochs=50)

# Step 8: Prepare Test Data and Make Predictions
test_data = yf.download(ticker, start='2025-01-01', end='2025-10-01', auto_adjust=False)
test_data = test_data[['Open','High','Low','Close','Volume']]
actual_prices = test_data['Close'].values

# Combine training and test data for scaling
total_data = pd.concat((data['Close'], test_data['Close']), axis=0)
inputs = total_data[len(total_data) - len(test_data) - sequence_length:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

x_test = []
for i in range(sequence_length, len(inputs)):
    x_test.append(inputs[i-sequence_length:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Step 9: Visualize Predictions
plt.figure(figsize=(14,5))
plt.plot(actual_prices[sequence_length:], color='blue', label='Actual Stock Price')
plt.plot(predicted_prices, color='red', label='Predicted Stock Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()

# Step 10: Print Comparison of Actual vs Predicted
# Trim actual prices to match predictions
actual_prices_trimmed = actual_prices[sequence_length:]

# Ensure both arrays are 1D and have the same length
actual_prices_trimmed = np.array(actual_prices_trimmed).flatten()
predicted_prices = np.array(predicted_prices).flatten()

# Make sure lengths match
min_len = min(len(actual_prices_trimmed), len(predicted_prices))
actual_prices_trimmed = actual_prices_trimmed[:min_len]
predicted_prices = predicted_prices[:min_len]

# Create comparison DataFrame
comparison = pd.DataFrame({
    'Actual': actual_prices_trimmed,
    'Predicted': predicted_prices
})

print("Comparison of Actual vs Predicted Prices:")
print(comparison.head(10))
