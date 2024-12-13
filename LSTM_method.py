import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class StockPredictor:
    def __init__(self, filepath, feature_column='Close', prediction_days=60):
        self.filepath = filepath
        self.feature_column = feature_column
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def load_data(self):
        """Load stock data from a CSV file."""
        self.data = pd.read_csv(self.filepath)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.sort_values('Date', inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        print(f"Data loaded successfully with {len(self.data)} records.")

    def preprocess_data(self):
        """Preprocess the data for LSTM model."""
        dataset = self.data[[self.feature_column]].values
        dataset = self.scaler.fit_transform(dataset)
        
        self.train_size = int(len(dataset) * 0.8)
        self.train_data = dataset[:self.train_size]
        self.test_data = dataset[self.train_size - self.prediction_days:]
        
        self.X_train, self.y_train = self.create_sequences(self.train_data)
        self.X_test, self.y_test = self.create_sequences(self.test_data)
        
        # Reshape input to be [samples, time steps, features]
        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
        print("Data preprocessing completed.")

    def create_sequences(self, data):
        """Create sequences of data for LSTM."""
        X, y = [], []
        for i in range(self.prediction_days, len(data)):
            X.append(data[i - self.prediction_days:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def build_model(self):
        """Build the LSTM model using PyTorch."""
        self.model = LSTMModel(input_size=1, hidden_size=256, num_layers=2, output_size=1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.NAdam(self.model.parameters(), lr=0.001)
        print("PyTorch LSTM model built and compiled.")

    def train(self, epochs=100, batch_size=32):
        """Train the LSTM model using PyTorch."""
        dataset = TensorDataset(torch.from_numpy(self.X_train).float(), torch.from_numpy(self.y_train).float())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict(self):
        """Make predictions and visualize the results using PyTorch."""
        self.model.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(self.X_test).float()
            predictions = self.model(inputs).numpy()
            predictions = self.scaler.inverse_transform(predictions)
            actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        
        # Plot the data
        plt.figure(figsize=(14,5))
        plt.plot(self.data['Date'][self.train_size:], actual, color='blue', label='Actual Stock Price')
        plt.plot(self.data['Date'][self.train_size:], predictions, color='red', label='Predicted Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

    def evaluate(self):
        """Evaluate the model performance."""
        from sklearn.metrics import mean_squared_error, r2_score
        predictions = self.model.predict(self.X_test)
        predictions = self.scaler.inverse_transform(predictions)
        actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        mse = mean_squared_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        print(f"Mean Squared Error: {mse}")
        print(f"R² Score: {r2}")

if __name__ == "__main__":
    # Example usage
    predictor = StockPredictor(filepath='data/AAPL_history_processed.csv', feature_column='Close', prediction_days=60)
    predictor.load_data()
    predictor.preprocess_data()
    predictor.build_model()
    predictor.train(epochs=150, batch_size=64)
    predictor.predict()
    predictor.evaluate()
