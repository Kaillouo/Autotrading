"""
Bybit derivatives data: funding rates, open interest, liquidations.
Uses the ccxt client from bybit_client — no duplicate connection setup.

Usage:
    python src/data/bybit_derivatives.py               # testnet, 6 months
    python src/data/bybit_derivatives.py --months 12 --live  # production, 12 months
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.data.bybit_client import get_exchange
from src.db.database import init_db, insert_derivatives_data, get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

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


# ── Historical backfill ───────────────────────────────────────────────────────

FUNDING_INTERVAL_MS = 8 * 3_600_000   # funding rate every 8 hours
OI_INTERVAL_MS      = 3_600_000        # OI every 1 hour
REQUEST_DELAY       = 0.20             # seconds between API calls


def backfill_derivatives(
    symbol: str = BTC_PERP,
    months: int = 6,
    live: bool = False,
) -> int:
    """
    Backfill funding rate + open interest history for `months` back.

    When live=False, uses testnet. When live=True, uses production Bybit public API.
    Returns the number of new derivative rows inserted into the DB.
    """
    if live:
        exchange = ccxt.bybit({"options": {"defaultType": "linear"}})
        exchange.has["fetchCurrencies"] = False
        print(f"[INFO] Derivatives backfill: production Bybit, {months} months")
    else:
        exchange = get_derivatives_exchange()
        print(f"[INFO] Derivatives backfill: testnet Bybit, {months} months")

    init_db()

    now = datetime.now(timezone.utc)
    since_target = now - timedelta(days=months * 30)
    since_ms = int(since_target.timestamp() * 1000)
    now_ms = int(now.timestamp() * 1000)

    conn = get_connection()
    count_before = conn.execute(
        "SELECT COUNT(*) FROM derivatives_snapshots WHERE symbol = ?", (symbol,)
    ).fetchone()[0]
    conn.close()

    print(f"  Range: {since_target.strftime('%Y-%m-%d')} -> {now.strftime('%Y-%m-%d')}")
    print(f"  Existing derivative rows: {count_before}")
    print()

    # ── Funding rate pagination ───────────────────────────────────────────────
    print("Fetching funding rate history...")
    funding_rows = []
    current_since = since_ms

    while current_since < now_ms:
        try:
            raw = exchange.fetch_funding_rate_history(symbol, since=current_since, limit=200)
        except Exception as e:
            print(f"  [WARN] Funding rate fetch failed: {e} — stopping early")
            time.sleep(2)
            break

        if not raw:
            break

        for r in raw:
            funding_rows.append({
                "timestamp": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
                "funding_rate": float(r.get("fundingRate") or r.get("fundingRate", 0)),
            })

        last_ts = raw[-1]["timestamp"]
        print(f"  Funding: fetched up to {pd.to_datetime(last_ts, unit='ms')}  ({len(funding_rows)} rows)")
        next_since = last_ts + FUNDING_INTERVAL_MS
        if next_since <= current_since:
            break
        current_since = next_since
        time.sleep(REQUEST_DELAY)

    if funding_rows:
        funding_df = pd.DataFrame(funding_rows).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        # Rolling 48-period z-score
        roll_mean = funding_df["funding_rate"].rolling(48, min_periods=2).mean()
        roll_std  = funding_df["funding_rate"].rolling(48, min_periods=2).std()
        funding_df["funding_rate_zscore"] = (funding_df["funding_rate"] - roll_mean) / roll_std.replace(0.0, float("nan"))
    else:
        funding_df = pd.DataFrame(columns=["timestamp", "funding_rate", "funding_rate_zscore"])

    # ── OI pagination ─────────────────────────────────────────────────────────
    print("Fetching open interest history...")
    oi_rows = []
    current_since = since_ms

    while current_since < now_ms:
        try:
            raw = exchange.fetch_open_interest_history(symbol, timeframe="1h", since=current_since, limit=200)
        except Exception as e:
            print(f"  [WARN] OI fetch failed: {e} — stopping early")
            time.sleep(2)
            break

        if not raw:
            break

        for r in raw:
            oi_col = r.get("openInterestAmount") or r.get("openInterest") or 0
            oi_rows.append({
                "timestamp": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
                "open_interest": float(oi_col),
            })

        last_ts = raw[-1]["timestamp"]
        print(f"  OI: fetched up to {pd.to_datetime(last_ts, unit='ms')}  ({len(oi_rows)} rows)")
        next_since = last_ts + OI_INTERVAL_MS
        if next_since <= current_since:
            break
        current_since = next_since
        time.sleep(REQUEST_DELAY)

    if oi_rows:
        oi_df = pd.DataFrame(oi_rows).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    else:
        oi_df = pd.DataFrame(columns=["timestamp", "open_interest"])

    # ── Insert into DB ────────────────────────────────────────────────────────
    print("Inserting into DB...")
    insert_derivatives_data(funding_df, oi_df, symbol=symbol)

    conn = get_connection()
    count_after = conn.execute(
        "SELECT COUNT(*) FROM derivatives_snapshots WHERE symbol = ?", (symbol,)
    ).fetchone()[0]
    conn.close()

    new_rows = count_after - count_before
    print()
    print(f"Derivatives backfill complete.")
    print(f"  Funding rows fetched : {len(funding_rows)}")
    print(f"  OI rows fetched      : {len(oi_rows)}")
    print(f"  New rows inserted    : {new_rows}")
    print(f"  Total in DB now      : {count_after}")
    return new_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill derivatives (funding rate + OI)")
    parser.add_argument("--months", type=int, default=6, help="Months of history to fetch (default: 6)")
    parser.add_argument("--live", action="store_true", help="Use production Bybit (public data, no auth)")
    args = parser.parse_args()
    backfill_derivatives(months=args.months, live=args.live)
