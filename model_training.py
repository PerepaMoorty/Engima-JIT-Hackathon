from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def train_model(data):
    # Define target variable
    data["target"] = (data["close"].shift(-1) > data["close"]).astype(int)

    # Define features, including sentiment_score
    features = ["RSI", "MACD", "Signal_Line", "BB_High", "BB_Low", "sentiment_score"]

    # Drop rows with missing values
    data = data.dropna()

    # Feature matrix and target vector
    X = data[features]
    y = data["target"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize RandomForestClassifier
    model = RandomForestClassifier(random_state=42)

    # Define hyperparameter grid for GridSearchCV
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring="accuracy"
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

    # Calculate accuracy on testing data
    accuracy = best_model.score(X_test, y_test)

    # Predict on the entire dataset
    data["prediction"] = best_model.predict(X_scaled)

    return best_model, accuracy, data
