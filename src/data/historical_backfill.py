"""
Historical OHLCV backfill — fetches 6 months of BTC/USDT 1h candles from Bybit testnet.

Safe to run multiple times: INSERT OR IGNORE skips rows that already exist.
Progress is printed every batch so you can see it working.

Usage:
    python -m src.data.historical_backfill
    # or directly:
    python src/data/historical_backfill.py
"""

import os
import sys
import time

# Make sure the project root is importable when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from datetime import datetime, timezone, timedelta

from src.data.bybit_client import get_exchange
from src.signals.technical import compute_indicators
from src.db.database import init_db, insert_candles

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
) -> int:
    """
    Fetch `months` of OHLCV history from Bybit testnet for `symbol`.

    Paginates forward in BATCH_SIZE=200 candle chunks starting from
    `months` ago and stops when it reaches the present.

    Returns the total number of candles fetched (including any skipped
    duplicates — idempotency is handled at the DB layer).
    """
    exchange = get_exchange()
    init_db()

    now = datetime.now(timezone.utc)
    since_target = now - timedelta(days=months * 30)
    since_ms = int(since_target.timestamp() * 1000)
    now_ms = int(now.timestamp() * 1000)

    total_estimate = (now_ms - since_ms) // CANDLE_MS

    print("=" * 60)
    print(f"HISTORICAL BACKFILL - {symbol} {timeframe}")
    print("=" * 60)
    print(f"Range : {since_target.strftime('%Y-%m-%d')} -> {now.strftime('%Y-%m-%d')}")
    print(f"Target: ~{total_estimate} candles  (batch size: {BATCH_SIZE})")
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

    print()
    print(f"Backfill complete. Total candles fetched: {total_fetched}")
    return total_fetched


if __name__ == "__main__":
    backfill_historical()
