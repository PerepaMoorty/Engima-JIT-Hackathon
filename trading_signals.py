import pandas as pd


def detect_trading_signals(data, window=5):
    """
    Detects buy and sell signals based on price dips and peaks.

    Args:
        data (pd.DataFrame): DataFrame containing stock data with 'close' prices.
        window (int): Number of periods to consider for detecting peaks and dips.

    Returns:
        pd.DataFrame: DataFrame with additional 'buy_signal' and 'sell_signal' columns.
    """
    data = data.copy()
    data["buy_signal"] = False
    data["sell_signal"] = False

    for i in range(window, len(data) - window):
        window_slice = data["close"][i - window : i + window + 1]
        current_price = data["close"].iloc[i]

        # Detect local minimum for buy signal
        if current_price == window_slice.min():
            data.at[data.index[i], "buy_signal"] = True

        # Detect local maximum for sell signal
        elif current_price == window_slice.max():
            data.at[data.index[i], "sell_signal"] = True

    return data
