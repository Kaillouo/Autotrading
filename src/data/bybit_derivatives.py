"""
Bybit derivatives data: funding rates, open interest, liquidations.
Uses the ccxt client from bybit_client — no duplicate connection setup.
"""

import logging
import numpy as np
import pandas as pd

from src.data.bybit_client import get_exchange

logger = logging.getLogger(__name__)

# Bybit linear perpetual symbol format
BTC_PERP = "BTC/USDT:USDT"


def get_derivatives_exchange():
    """Return the shared ccxt Bybit exchange instance reconfigured for linear perpetuals."""
    exchange = get_exchange()
    exchange.options["defaultType"] = "linear"
    return exchange


def fetch_funding_rate_history(symbol: str = BTC_PERP, limit: int = 200) -> pd.DataFrame:
    """
    Fetch funding rate history for a perpetual contract.

    Returns DataFrame indexed by UTC timestamp with columns:
        funding_rate, funding_rate_zscore (rolling 48-period z-score)

    Returns empty DataFrame on failure.
    """
    exchange = get_derivatives_exchange()
    try:
        raw = exchange.fetch_funding_rate_history(symbol, limit=limit)
    except Exception as e:
        logger.warning(f"Failed to fetch funding rate history for {symbol}: {e}")
        return pd.DataFrame(columns=["timestamp", "funding_rate", "funding_rate_zscore"])

    if not raw:
        logger.warning(f"No funding rate data returned for {symbol}")
        return pd.DataFrame(columns=["timestamp", "funding_rate", "funding_rate_zscore"])

    df = pd.DataFrame(raw)
    df = df[["timestamp", "fundingRate"]].rename(columns={"fundingRate": "funding_rate"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Rolling 48-period z-score — NaN for first ~48 rows (warmup)
    roll_mean = df["funding_rate"].rolling(48, min_periods=2).mean()
    roll_std = df["funding_rate"].rolling(48, min_periods=2).std()
    # Avoid division by zero: if std is 0, z-score is 0
    df["funding_rate_zscore"] = (df["funding_rate"] - roll_mean) / roll_std.replace(0.0, np.nan)

    return df


def fetch_open_interest_history(
    symbol: str = BTC_PERP, timeframe: str = "1h", limit: int = 200
) -> pd.DataFrame:
    """
    Fetch open interest history for a perpetual contract.

    Returns DataFrame with columns: timestamp, open_interest
    Returns empty DataFrame on failure.
    """
    exchange = get_derivatives_exchange()
    try:
        raw = exchange.fetch_open_interest_history(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        logger.warning(f"Failed to fetch open interest history for {symbol}: {e}")
        return pd.DataFrame(columns=["timestamp", "open_interest"])

    if not raw:
        logger.warning(f"No open interest data returned for {symbol}")
        return pd.DataFrame(columns=["timestamp", "open_interest"])

    df = pd.DataFrame(raw)

    # ccxt normalises the field to openInterestAmount or openInterest depending on exchange
    oi_col = next(
        (c for c in ("openInterestAmount", "openInterest") if c in df.columns),
        None,
    )
    if oi_col is None:
        logger.warning(f"Unexpected OI response columns: {list(df.columns)}")
        return pd.DataFrame(columns=["timestamp", "open_interest"])

    df = df[["timestamp", oi_col]].rename(columns={oi_col: "open_interest"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_liquidations(symbol: str = BTC_PERP, limit: int = 200) -> pd.DataFrame:
    """
    Fetch recent liquidations.

    Graceful fallback: if the testnet does not support liquidations, logs a warning
    and returns an empty DataFrame — never raises.

    Returns DataFrame with columns: timestamp, side, amount, price
    """
    exchange = get_derivatives_exchange()
    try:
        raw = exchange.fetch_liquidations(symbol, limit=limit)
    except Exception as e:
        logger.warning(f"Liquidations unavailable (likely testnet limitation): {e}")
        return pd.DataFrame(columns=["timestamp", "side", "amount", "price"])

    if not raw:
        return pd.DataFrame(columns=["timestamp", "side", "amount", "price"])

    records = []
    for liq in raw:
        records.append(
            {
                "timestamp": pd.to_datetime(liq.get("timestamp", 0), unit="ms", utc=True),
                "side": liq.get("side", ""),
                "amount": float(liq.get("amount") or 0.0),
                "price": float(liq.get("price") or 0.0),
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df
