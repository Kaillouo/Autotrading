"""Main pipeline: fetch OHLCV -> compute indicators -> store to DB -> fetch derivatives
-> Fast Brain signal -> Rules Engine + execute."""

import argparse
import json
import os
import sys
import io
from datetime import datetime, timezone

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.utils.logger import get_logger
from src.data.bybit_client import fetch_ohlcv
from src.data.bybit_derivatives import fetch_funding_rate_history, fetch_open_interest_history
from src.signals.technical import compute_indicators
from src.db.database import init_db, insert_candles, insert_derivatives_data, query_recent

logger = get_logger("pipeline")

_POSITIONS_PATH = os.path.join(os.path.dirname(__file__), "data", "positions.json")


# ── Position state helpers ─────────────────────────────────────────────────────

def _read_positions_file() -> dict:
    if not os.path.exists(_POSITIONS_PATH):
        return {"open_positions": [], "equity": 10000.0, "peak_equity": 10000.0}
    with open(_POSITIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_positions_file(data: dict) -> None:
    os.makedirs(os.path.dirname(_POSITIONS_PATH), exist_ok=True)
    with open(_POSITIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_positions() -> list:
    return _read_positions_file().get("open_positions", [])


def load_equity() -> float:
    return float(_read_positions_file().get("equity", 10000.0))


def load_peak_equity() -> float:
    return float(_read_positions_file().get("peak_equity", 10000.0))


def update_positions(action: str, sizing: dict, execution: dict) -> None:
    """
    Update positions.json after a successful order:
    - open_long: append new position entry
    - close_long: remove all open positions, update equity with realized PnL
    """
    data = _read_positions_file()
    filled_price = execution.get("filled_price", 0.0)
    now = datetime.now(timezone.utc).isoformat()

    if action == "open_long":
        position = {
            "entry_price": filled_price or sizing.get("stop_price", 0.0),
            "quantity": sizing["quantity"],
            "stop": sizing["stop_price"],
            "tp": sizing["tp_price"],
            "opened_at": now,
        }
        data["open_positions"].append(position)
        logger.info(f"Position opened: {position}")

    elif action == "close_long":
        commission_pct = 0.001
        total_pnl = 0.0
        for pos in data["open_positions"]:
            qty = float(pos.get("quantity", 0.0))
            entry = float(pos.get("entry_price", 0.0))
            if entry > 0 and filled_price > 0:
                gross_pnl = (filled_price - entry) * qty
                commission = (entry + filled_price) * qty * commission_pct
                total_pnl += gross_pnl - commission
        data["equity"] = round(data["equity"] + total_pnl, 4)
        data["peak_equity"] = round(max(data["equity"], data["peak_equity"]), 4)
        data["open_positions"] = []
        logger.info(f"Position(s) closed. PnL={total_pnl:.4f} | New equity={data['equity']:.2f}")

    _write_positions_file(data)


# ── Main pipeline ──────────────────────────────────────────────────────────────

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
    logger.info(f"[1/6] Fetching {args.symbol} {args.timeframe} candles from Bybit testnet...")
    df = fetch_ohlcv(symbol=args.symbol, timeframe=args.timeframe, limit=200)
    logger.info(f"      Got {len(df)} candles | {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # 2. Compute indicators
    logger.info("[2/6] Computing technical indicators (RSI, MACD, BB)...")
    df = compute_indicators(df)
    valid = df.dropna(subset=["rsi", "macd", "bb_mid"])
    logger.info(f"      {len(valid)} rows with complete indicators (after warmup)")

    # 3. Store candles to DB
    logger.info("[3/6] Storing candles to SQLite...")
    init_db()
    insert_candles(df)

    # 4. Fetch and store derivatives
    logger.info(f"[4/6] Fetching derivatives for {SYMBOL_PERP}...")
    funding_df = fetch_funding_rate_history(symbol=SYMBOL_PERP)
    oi_df = fetch_open_interest_history(symbol=SYMBOL_PERP, timeframe=args.timeframe)
    insert_derivatives_data(funding_df, oi_df, symbol=SYMBOL_PERP)
    logger.info(f"      funding rows={len(funding_df)}  OI rows={len(oi_df)}")

    # 5. Fast Brain
    logger.info("[5/6] Running Fast Brain (LLM signal generation)...")
    from src.ai.fast_brain import run_fast_brain
    signal = run_fast_brain(symbol=args.symbol)
    logger.info(
        f"      Fast Brain -> direction={signal['direction']} "
        f"confidence={signal['confidence']:.2f} regime={signal['regime']}"
    )

    # 6. Rules Engine + Execute
    logger.info("[6/6] Rules Engine evaluation...")
    from src.risk.rules_engine import evaluate_signal
    from src.risk.position_sizer import calculate_position_size
    from src.execution.order_executor import execute_signal

    open_positions = load_positions()
    current_equity = load_equity()
    peak_equity = load_peak_equity()

    result = evaluate_signal(signal, open_positions, current_equity, peak_equity=peak_equity)
    logger.info(
        f"      Rules Engine -> approved={result['approved']} "
        f"action={result['action']} reason={result['reason']}"
    )

    if result["approved"] and result["action"] != "hold":
        signal_price = float(signal.get("current_price", 0.0))
        sizing = calculate_position_size(signal, current_equity, current_price=signal_price)
        execution = execute_signal(result, sizing, symbol=args.symbol)
        if execution["success"]:
            logger.info(f"      Order placed: {execution}")
            update_positions(result["action"], sizing, execution)
        else:
            logger.error(f"      Order failed: {execution['error']}")

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
