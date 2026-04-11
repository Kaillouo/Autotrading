"""Main pipeline: fetch OHLCV -> compute indicators -> store to DB -> fetch derivatives -> print summary."""

import argparse
import sys
import io

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.utils.logger import get_logger
from src.data.bybit_client import fetch_ohlcv
from src.data.bybit_derivatives import fetch_funding_rate_history, fetch_open_interest_history
from src.signals.technical import compute_indicators
from src.db.database import init_db, insert_candles, insert_derivatives_data, query_recent

logger = get_logger("pipeline")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()

    SYMBOL_PERP = args.symbol + ":USDT"

    logger.info("=" * 60)
    logger.info("TRADING DATA PIPELINE")
    logger.info("=" * 60)

    # 1. Fetch OHLCV
    logger.info(f"[1/4] Fetching {args.symbol} {args.timeframe} candles from Bybit testnet...")
    df = fetch_ohlcv(symbol=args.symbol, timeframe=args.timeframe, limit=200)
    logger.info(f"      Got {len(df)} candles | {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # 2. Compute indicators
    logger.info("[2/4] Computing technical indicators (RSI, MACD, BB)...")
    df = compute_indicators(df)
    valid = df.dropna(subset=["rsi", "macd", "bb_mid"])
    logger.info(f"      {len(valid)} rows with complete indicators (after warmup)")

    # 3. Store candles to DB
    logger.info("[3/4] Storing candles to SQLite...")
    init_db()
    insert_candles(df)

    # 4. Fetch and store derivatives
    logger.info(f"[4/4] Fetching derivatives for {SYMBOL_PERP}...")
    funding_df = fetch_funding_rate_history(symbol=SYMBOL_PERP)
    oi_df = fetch_open_interest_history(symbol=SYMBOL_PERP, timeframe=args.timeframe)
    insert_derivatives_data(funding_df, oi_df, symbol=SYMBOL_PERP)
    logger.info(f"      funding rows={len(funding_df)}  OI rows={len(oi_df)}")

    # Summary
    recent = query_recent(limit=5)
    logger.info("\n" + "=" * 60)
    logger.info("LATEST 5 CANDLES WITH INDICATORS")
    logger.info("=" * 60)
    display_cols = ["timestamp", "close", "volume", "rsi", "macd", "bb_lower", "bb_mid", "bb_upper"]
    available = [c for c in display_cols if c in recent.columns]
    logger.info("\n" + recent[available].to_string(index=False))

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    run()
