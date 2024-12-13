import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class StockPredictor:
    def __init__(self, filepath, feature_column="Close", prediction_days=60):
        self.filepath = filepath
        self.feature_column = feature_column
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def load_data(self):
        """Load stock data from a CSV file."""
        self.data = pd.read_csv(self.filepath)
        self.data["Date"] = pd.to_datetime(self.data["Date"], utc=True)
        self.data.sort_values("Date", inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        print(f"Data loaded successfully with {len(self.data)} records.")

    def preprocess_data(self):
        """Preprocess the data for LSTM model."""
        dataset = self.data[[self.feature_column]].values
        self.train_size = int(len(dataset) * 0.8)
        self.val_size = int(len(dataset) * 0.1)
        self.test_size = len(dataset) - self.train_size - self.val_size

        self.train_data = dataset[: self.train_size]
        self.val_data = dataset[self.train_size : self.train_size + self.val_size]
        self.test_data = dataset[
            self.train_size + self.val_size - self.prediction_days :
        ]

        self.scaler.fit(self.train_data)  # Fit only on training data
        dataset = self.scaler.transform(dataset)

        self.X_train, self.y_train = self.create_sequences(self.train_data)
        self.X_val, self.y_val = self.create_sequences(self.val_data)
        self.X_test, self.y_test = self.create_sequences(self.test_data)

        # Reshape input to be [samples, time steps, features]
        self.X_train = self.X_train.reshape(
            (self.X_train.shape[0], self.X_train.shape[1], 1)
        )
        self.X_val = self.X_val.reshape((self.X_val.shape[0], self.X_val.shape[1], 1))
        self.X_test = self.X_test.reshape(
            (self.X_test.shape[0], self.X_test.shape[1], 1)
        )
        print("Data preprocessing completed.")

    def create_sequences(self, data):
        """Create sequences of data for LSTM."""
        X, y = [], []
        for i in range(self.prediction_days, len(data)):
            X.append(data[i - self.prediction_days : i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def build_model(self):
        """Build the LSTM model using PyTorch."""
        self.model = LSTMModel(
            input_size=1, hidden_size=256, num_layers=2, output_size=1
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.NAdam(self.model.parameters(), lr=0.0008)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Learning rate scheduler
        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        print(f"PyTorch LSTM model built and compiled on {self.device}.")

    def train(self, epochs=300, batch_size=64):
        """Train the LSTM model using PyTorch."""
        train_dataset = TensorDataset(
            torch.from_numpy(self.X_train).float(),
            torch.from_numpy(self.y_train).float(),
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        val_dataset = TensorDataset(
            torch.from_numpy(self.X_val).float(), torch.from_numpy(self.y_val).float()
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.model.train()
        best_loss = float("inf")
        patience = 64
        trigger_times = 0
        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                y_batch = y_batch.view(-1, 1)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )  # Gradient clipping
                self.optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_dataloader)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_val_batch, y_val_batch in val_dataloader:
                    X_val_batch, y_val_batch = X_val_batch.to(
                        self.device
                    ), y_val_batch.to(self.device)
                    val_outputs = self.model(X_val_batch)
                    y_val_batch = y_val_batch.view(-1, 1)
                    val_loss += self.criterion(val_outputs, y_val_batch).item()
            avg_val_loss = val_loss / len(val_dataloader)
            self.model.train()

            print(
                f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
            )

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                trigger_times = 0
                # Save the best model
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("Early stopping!")
                    break

            # Scheduler step
            self.scheduler.step()

    def predict(self):
        """Make predictions and visualize the results using PyTorch."""
        self.model.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(self.X_test).float().to(self.device)
            predictions = self.model(inputs).cpu().numpy()
            predictions = self.scaler.inverse_transform(predictions)
            actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

        # Adjust date indexing
        prediction_range = self.data["Date"][self.train_size + self.val_size :]

        # Plot the data
        plt.figure(figsize=(14, 5))
        plt.plot(prediction_range, actual, color="blue", label="Actual Stock Price")
        plt.plot(
            prediction_range, predictions, color="red", label="Predicted Stock Price"
        )
        plt.title("Stock Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.show()

    def evaluate(self):
        """Evaluate the model performance."""
        # Load the best model
        self.model.load_state_dict(torch.load("best_model.pth"))
        self.model.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(self.X_test).float().to(self.device)
            predictions = self.model(inputs).cpu().numpy()

        # Inverse transform the scaled data back to original scale
        predictions = self.scaler.inverse_transform(predictions)
        actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

        mse = mean_squared_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        print(f"Mean Squared Error: {mse}")
        print(f"R² Score: {r2}")


if __name__ == "__main__":
    # Example usage
    predictor = StockPredictor(
        filepath="AAPL_history_processed.csv",
        feature_column="Close",
        prediction_days=60,
    )
    predictor.load_data()
    predictor.preprocess_data()
    predictor.build_model()
    predictor.train(epochs=300, batch_size=32)
    predictor.predict()
    predictor.evaluate()
