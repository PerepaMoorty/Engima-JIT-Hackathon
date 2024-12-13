from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd

def train_model(data):
    # Define features, including sentiment_score
    features = ["RSI", "MACD", "Signal_Line", "BB_High", "BB_Low", "sentiment_score"]

    # Drop rows with missing values
    data = data.dropna()

    # Feature matrix and target variable
    X = data[features]
    y = data["close"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize RandomForestRegressor
    model = RandomForestRegressor(random_state=42)

    # Define hyperparameter grid for GridSearchCV
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring="neg_mean_squared_error"
    )

    # Fit GridSearchCV
    grid_search.fit(X_scaled, y)

    # Best estimator from GridSearch
    best_model = grid_search.best_estimator_

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Fit the best model on training data
    best_model.fit(X_train, y_train)

    # Calculate accuracy (R-squared score)
    accuracy = best_model.score(X_test, y_test)

    # Make predictions for the entire dataset
    data["prediction"] = best_model.predict(X_scaled)

    # Predict for the next 30 days
    future_predictions = []
    future_dates = pd.date_range(data["date"].iloc[-1] + pd.Timedelta(days=1), periods=30, freq="D")

    # Simulate future data to predict the next 30 days
    last_features = data[features].iloc[-1].values.reshape(1, -1)  # Last row features
    for _ in range(30):  # Predict for 30 days
        last_scaled = scaler.transform(last_features)
        next_prediction = best_model.predict(last_scaled)[0]
        future_predictions.append(next_prediction)

        # Update features for the next prediction based on the new prediction
        # For simplicity, let's just add the predicted value to 'BB_High' and 'BB_Low' as an example
        last_features[0, features.index("BB_High")] = next_prediction  # Update feature (you can modify other features as well)
        last_features[0, features.index("BB_Low")] = next_prediction  # Similarly update other relevant features

    return best_model, accuracy, data, future_dates, future_predictions
