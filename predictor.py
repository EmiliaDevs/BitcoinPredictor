import pandas as pd
import requests
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime
import pyfiglet

# Fetch historical data from CoinGecko API
def fetch_historical_data():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': '365'
    }
    response = requests.get(url, params=params)
    data = response.json()
    prices = data['prices']
    timestamps = [timestamp for timestamp, _ in prices]
    prices = [price for _, price in prices]
    return timestamps, prices

# Create features and labels for training
def create_features_and_labels(df, window_size):
    X = []
    y = []
    for i in range(len(df) - window_size):
        X.append(df[i:i+window_size])
        y.append(df.iloc[i+window_size])
    X = np.array(X)
    y = np.array(y)
    return X, y

# Fetch historical data
timestamps, prices = fetch_historical_data()

# Create a DataFrame from the fetched data
data = {'Timestamp': timestamps, 'Price': prices}
df = pd.DataFrame(data)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
df.set_index('Timestamp', inplace=True)

# Define the number of previous days to consider for prediction
window_size = 7

# Prepare the data for training
X, y = create_features_and_labels(df, window_size)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)

# Ask for user input for the number of next days
next_days = int(input("Enter the number of days for price prediction: "))

# Make predictions for the next n days
last_data = df.iloc[-window_size:].values.reshape(1, -1)
predictions = model.predict(last_data)

# Print the predicted prices for the next n days
print('\nPredicted Prices for the Next', next_days, 'Days:')
current_date = df.index[-1]
figlet_text = pyfiglet.figlet_format("Drickworld")
print(figlet_text)
for i in range(next_days):
    current_date += datetime.timedelta(days=1)
    print(current_date.date(), ':', round(predictions[0], 2))
    predictions = model.predict(np.append(last_data[:, 1:], predictions).reshape(1, -1))

