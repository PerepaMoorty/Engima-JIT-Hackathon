import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Loading the dataset and checking for Date and CLose
data = pd.read_csv('AAPL_history_processed.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace = True)

# Moving averages
data['SMA_50'] = data['Close'].rolling(window = 50).mean()
data['SMA_200'] = data['Close'].rolling(window = 200).mean()

# Normalization for LSTM
scalar = MinMaxScaler(feature_range = (0, 1))
data['Normalized_Close'] = scalar.fit_transform(data['Close'].values.reshape(-1, 1))

# Preparing dataset for LSTM
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:1 + time_steps])
        y.append(data[i - time_steps])
    return np.array(X), np.array(y)

time_steps = 60
X, y = create_sequences(data['Normalized_Close'].values, time_steps)

# Spliting into training and testing
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshaping
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Building LSTM
