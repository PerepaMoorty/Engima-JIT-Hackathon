from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_model(data):
    data["target"] = (data["close"].shift(-1) > data["close"]).astype(int)
    features = ["RSI", "MACD", "Signal_Line", "BB_High", "BB_Low"]
    data = data.dropna()
    X = data[features]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    data["prediction"] = model.predict(X[features])
    return model, accuracy, data
