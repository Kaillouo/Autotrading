"""Main pipeline: fetch OHLCV -> compute indicators -> store to DB -> print summary."""

import sys
import io

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.data.bybit_client import fetch_ohlcv
from src.signals.technical import compute_indicators
from src.db.database import init_db, insert_candles, query_recent


def run():
    print("=" * 60)
    print("TRADING DATA PIPELINE - Session 1")
    print("=" * 60)

    # 1. Fetch OHLCV
    print("\n[1/3] Fetching BTC/USDT 1h candles from Bybit testnet...")
    df = fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=200)
    print(f"      Got {len(df)} candles | {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # 2. Compute indicators
    print("[2/3] Computing technical indicators (RSI, MACD, BB)...")
    df = compute_indicators(df)
    valid = df.dropna(subset=["rsi", "macd", "bb_mid"])
    print(f"      {len(valid)} rows with complete indicators (after warmup)")

    # 3. Store to DB
    print("[3/3] Storing to SQLite...")
    init_db()
    insert_candles(df)

    # 4. Print summary
    recent = query_recent(limit=5)
    print("\n" + "=" * 60)
    print("LATEST 5 CANDLES WITH INDICATORS")
    print("=" * 60)
    display_cols = ["timestamp", "close", "volume", "rsi", "macd", "bb_lower", "bb_mid", "bb_upper"]
    available = [c for c in display_cols if c in recent.columns]
    print(recent[available].to_string(index=False))
    print("\nPipeline complete.")


if __name__ == "__main__":
    run()
