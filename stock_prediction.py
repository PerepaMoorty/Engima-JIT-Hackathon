import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
from datetime import datetime
import time

def fetch_stock_data(ticker: str, period: str = "10y") -> pd.DataFrame:
    """
    Fetches historical stock data for the given ticker and period.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    hist.to_csv(f'data/{ticker}_history.csv')
    print(f"Fetched data for {ticker} and saved to data/{ticker}_history.csv")
    return hist

def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Processes the stock data by handling missing values and scaling features.
    """
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    
    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    
    # Feature engineering: Moving averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Drop rows with NaN values after rolling
    df.dropna(inplace=True)
    
    # Scaling
    scaler = StandardScaler()
    df[['Close', 'MA50', 'MA200']] = scaler.fit_transform(df[['Close', 'MA50', 'MA200']])
    
    processed_file = file_path.replace('.csv', '_processed.csv')
    df.to_csv(processed_file)
    print(f"Processed data saved to {processed_file}")
    return df

def train_model(file_path: str, model_path: str = 'models/linear_regression_model.pkl') -> LinearRegression:
    """
    Trains a Linear Regression model on the processed stock data.
    """
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    
    # Features and target
    X = df[['MA50', 'MA200']]
    y = df['Close']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")
    
    # Evaluate
    score = model.score(X_test, y_test)
    print(f"Model R^2 Score: {score}")
    
    return model

def get_real_time_data(ticker: str) -> pd.DataFrame:
    """
    Fetches the latest real-time data for the given ticker.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d", interval="1m")
    return hist

def make_prediction(model_path: str, recent_features: pd.DataFrame) -> float:
    """
    Makes a prediction based on the most recent features.
    """
    model = joblib.load(model_path)
    prediction = model.predict(recent_features)[0]
    return prediction

def decide_action(current_price: float, predicted_price: float) -> str:
    """
    Decides whether to sell or hold based on the prediction.
    """
    if predicted_price < current_price:
        return "Sell"
    else:
        return "Hold"

def update_overlay(ticker: str, model_path: str):
    """
    Updates the overlay with the prediction.
    """
    real_time_data = get_real_time_data(ticker)
    
    # Load the latest processed historical data
    historical = pd.read_csv(f'data/{ticker}_history_processed.csv', parse_dates=['Date'], index_col='Date')
    latest_features = historical[['MA50', 'MA200']].tail(1)
    
    predicted_price = make_prediction(model_path, latest_features)
    current_price = real_time_data['Close'].iloc[-1]
    
    action = decide_action(current_price, predicted_price)
    
    print(f"Timestamp: {datetime.now()}")
    print(f"Current Price: {current_price}")
    print(f"Predicted Price: {predicted_price}")
    print(f"Action: {action}\n")

def main():
    ticker = "AAPL"
    period = "10y"
    data_file = f'data/{ticker}_history.csv'
    processed_file = f'data/{ticker}_history_processed.csv'
    model_path = 'models/linear_regression_model.pkl'
    
    # Ensure directories exist
    import os
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Fetch Data
    if not os.path.exists(data_file):
        fetch_stock_data(ticker, period)
    else:
        print(f"Data file {data_file} already exists.")
    
    # Step 2: Process Data
    if not os.path.exists(processed_file):
        preprocess_data(data_file)
    else:
        print(f"Processed file {processed_file} already exists.")
    
    # Step 3: Train Model
    if not os.path.exists(model_path):
        train_model(processed_file, model_path)
    else:
        print(f"Model file {model_path} already exists.")
    
    # Step 4: Real-Time Prediction Loop
    print("Starting real-time prediction. Press Ctrl+C to stop.")
    try:
        while True:
            update_overlay(ticker, model_path)
            time.sleep(20)  # Wait for 1 minute before next update
    except KeyboardInterrupt:
        print("Stopping real-time prediction.")

if __name__ == "__main__":
    main() 