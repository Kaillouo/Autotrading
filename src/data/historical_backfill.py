"""
Historical OHLCV backfill — fetches OHLCV candles from Bybit.

Safe to run multiple times: INSERT OR IGNORE skips rows that already exist.
Progress is printed every batch so you can see it working.

Usage:
    python src/data/historical_backfill.py               # testnet, 6 months
    python src/data/historical_backfill.py --months 12 --live  # production, 12 months
"""

import argparse
import os
import sys
import time

# Make sure the project root is importable when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import ccxt
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.data.bybit_client import get_exchange
from src.signals.technical import compute_indicators
from src.db.database import init_db, insert_candles, get_connection

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
CANDLE_MS = 3_600_000  # 1 hour in milliseconds
BATCH_SIZE = 200
MONTHS = 6
REQUEST_DELAY = 0.15  # seconds between API calls — stay well under rate limits


def backfill_historical(
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME,
    months: int = MONTHS,
    live: bool = False,
) -> int:
    """
    Fetch `months` of OHLCV history from Bybit for `symbol`.

    When live=False (default), uses testnet/sandbox mode.
    When live=True, uses production Bybit public API (no auth needed for OHLCV).

    Paginates forward in BATCH_SIZE=200 candle chunks starting from
    `months` ago and stops when it reaches the present.

    Returns the number of NEW candles inserted (duplicates are skipped).
    """
    if live:
        exchange = ccxt.bybit({"options": {"defaultType": "spot"}})
        exchange.has["fetchCurrencies"] = False
        print(f"[INFO] Using production Bybit (public OHLCV, no auth)")
    else:
        exchange = get_exchange()
        print(f"[INFO] Using testnet/sandbox Bybit")

    init_db()

    now = datetime.now(timezone.utc)
    since_target = now - timedelta(days=months * 30)
    since_ms = int(since_target.timestamp() * 1000)
    now_ms = int(now.timestamp() * 1000)

    total_estimate = (now_ms - since_ms) // CANDLE_MS

    # Count existing rows so we can report new insertions
    conn = get_connection()
    count_before = conn.execute(
        "SELECT COUNT(*) FROM candles WHERE symbol = ?", (symbol,)
    ).fetchone()[0]
    conn.close()

    print("=" * 60)
    print(f"HISTORICAL BACKFILL - {symbol} {timeframe}")
    print("=" * 60)
    print(f"Range : {since_target.strftime('%Y-%m-%d')} -> {now.strftime('%Y-%m-%d')}")
    print(f"Target: ~{total_estimate} candles  (batch size: {BATCH_SIZE})")
    print(f"Existing candles in DB: {count_before}")
    print()

    total_fetched = 0
    current_since = since_ms

    while current_since < now_ms:
        try:
            raw = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=current_since,
                limit=BATCH_SIZE,
            )
        except Exception as e:
            print(f"[ERROR] Fetch failed: {e}  — waiting 2s and stopping early")
            time.sleep(2)
            break

        if not raw:
            break

        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["symbol"] = symbol
        df = compute_indicators(df)
        insert_candles(df)

        batch_start = df["timestamp"].iloc[0]
        batch_end = df["timestamp"].iloc[-1]
        total_fetched += len(df)

        print(
            f"Fetched {batch_start.strftime('%Y-%m-%d')} -> {batch_end.strftime('%Y-%m-%d')}"
            f"  ({total_fetched}/{total_estimate} candles)"
        )

        # Advance since to one period after the last returned candle
        last_ts_ms = int(df["timestamp"].iloc[-1].timestamp() * 1000)
        next_since = last_ts_ms + CANDLE_MS

        if next_since <= current_since:
            # Safety: no progress — shouldn't happen but prevents infinite loop
            break

        current_since = next_since
        time.sleep(REQUEST_DELAY)

    # Count new insertions
    conn = get_connection()
    count_after = conn.execute(
        "SELECT COUNT(*) FROM candles WHERE symbol = ?", (symbol,)
    ).fetchone()[0]
    conn.close()
    new_candles = count_after - count_before

    print()
    print(f"Backfill complete.")
    print(f"  API candles fetched : {total_fetched}")
    print(f"  New rows inserted   : {new_candles}  (duplicates skipped: {total_fetched - new_candles})")
    print(f"  Total in DB now     : {count_after}")
    return new_candles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical OHLCV candles")
    parser.add_argument("--months", type=int, default=6, help="Months of history to fetch (default: 6)")
    parser.add_argument("--live", action="store_true", help="Use production Bybit (public data, no auth)")
    args = parser.parse_args()
    backfill_historical(months=args.months, live=args.live)
