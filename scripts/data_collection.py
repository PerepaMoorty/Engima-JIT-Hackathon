import yfinance as yf
import pandas as pd


def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetches historical stock data using the yfinance library.

    Args:
        symbol (str): Stock symbol (e.g., 'RELIANCE.NS' for NSE or 'TCS.BO' for BSE).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame containing the stock data with columns ['date', 'open', 'high', 'low', 'close', 'volume'].
    """
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        df.reset_index(inplace=True)
        df = df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "open", "high", "low", "close", "volume"]]
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
