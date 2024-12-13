from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def calculate_indicators(df):
    df['RSI'] = RSIIndicator(df['close'].astype(float), window=14).rsi()
    macd = MACD(df['close'].astype(float))
    df['MACD'] = macd.macd()
    df['Signal_Line'] = macd.macd_signal()
    bb = BollingerBands(df['close'].astype(float))
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    return df
