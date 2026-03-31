import ccxt
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()


def get_exchange():
    """Create a ccxt Bybit exchange instance in testnet/sandbox mode."""
    exchange = ccxt.bybit({
        "apiKey": os.getenv("BYBIT_API_KEY"),
        "secret": os.getenv("BYBIT_API_SECRET"),
        "options": {"defaultType": "spot"},
    })
    exchange.set_sandbox_mode(True)
    # OHLCV is a public endpoint — disable the authenticated fetchCurrencies
    # call that ccxt triggers during load_markets (causes auth errors on testnet
    # if keys are not yet configured).
    exchange.has["fetchCurrencies"] = False
    return exchange


def fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=200):
    """Fetch OHLCV candles from Bybit testnet and return as DataFrame."""
    exchange = get_exchange()
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["symbol"] = symbol
    return df
