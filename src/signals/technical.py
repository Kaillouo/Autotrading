import pandas as pd
import pandas_ta as ta


def compute_indicators(df):
    """Compute RSI(14), MACD(12,26,9), Bollinger Bands(20,2) on an OHLCV DataFrame."""
    df = df.copy()

    # RSI
    df["rsi"] = ta.rsi(df["close"], length=14)

    # MACD — returns None if not enough data (need 26+ rows)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd"] = macd.iloc[:, 0]        # MACD line
        df["macd_signal"] = macd.iloc[:, 1]  # Signal line
        df["macd_hist"] = macd.iloc[:, 2]    # Histogram
    else:
        df["macd"] = None
        df["macd_signal"] = None
        df["macd_hist"] = None

    # Bollinger Bands — returns None if not enough data (need 20+ rows)
    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None:
        df["bb_lower"] = bb.iloc[:, 0]
        df["bb_mid"] = bb.iloc[:, 1]
        df["bb_upper"] = bb.iloc[:, 2]
    else:
        df["bb_lower"] = None
        df["bb_mid"] = None
        df["bb_upper"] = None

    # ATR — needed for stop-loss sizing (risk rules use ATR multiples)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # EMA cross — core signal input
    df["ema_fast"] = ta.ema(df["close"], length=20)
    df["ema_slow"] = ta.ema(df["close"], length=50)
    # 1=bullish (fast > slow), 0=bearish, NaN when insufficient data
    ema_fast = pd.to_numeric(df["ema_fast"], errors="coerce")
    ema_slow = pd.to_numeric(df["ema_slow"], errors="coerce")
    df["ema_cross"] = (ema_fast > ema_slow).where(ema_slow.notna()).astype("Int64")

    return df
