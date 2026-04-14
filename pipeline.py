"""Main pipeline: fetch OHLCV -> compute indicators -> store to DB -> fetch derivatives
-> sync positions -> Fast Brain signal -> Rules Engine + execute."""

import argparse
import json
import os
import sys
import io
import time
from datetime import datetime, timezone

# ── Absolute base dir — safe when run from any CWD (e.g. Task Scheduler) ──────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Force UTF-8 output on Windows (safe-guarded for no-console environments like Task Scheduler)
try:
    if sys.stdout and hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

from src.utils.logger import get_logger
from src.data.bybit_client import fetch_ohlcv
from src.data.bybit_derivatives import fetch_funding_rate_history, fetch_open_interest_history
from src.signals.technical import compute_indicators
from src.db.database import init_db, insert_candles, insert_derivatives_data, query_recent, insert_trade

logger = get_logger("pipeline")

_POSITIONS_PATH = os.path.join(BASE_DIR, "data", "positions.json")
_LOCK_FILE = os.path.join(BASE_DIR, "data", "pipeline.lock")
_LOCK_MAX_AGE_SECONDS = 600  # 10 minutes


# ── Run lock (prevents overlapping Task Scheduler runs) ───────────────────────

def acquire_lock() -> None:
    if os.path.exists(_LOCK_FILE):
        age = time.time() - os.path.getmtime(_LOCK_FILE)
        if age < _LOCK_MAX_AGE_SECONDS:
            logger.warning(
                f"Pipeline already running — lock file is {age:.0f}s old (< {_LOCK_MAX_AGE_SECONDS}s). Exiting."
            )
            sys.exit(0)
        else:
            logger.warning(f"Stale lock file found ({age:.0f}s old). Removing and continuing.")
    os.makedirs(os.path.dirname(_LOCK_FILE), exist_ok=True)
    with open(_LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))


def release_lock() -> None:
    if os.path.exists(_LOCK_FILE):
        os.remove(_LOCK_FILE)


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


def update_positions(action: str, sizing: dict, execution: dict, signal: dict | None = None) -> None:
    """
    Update positions.json after a successful order.
    - open_long: append new position with entry + stop/TP order IDs + signal context
    - close_long: write completed trades to DB, remove positions, update equity
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
            "entry_order_id": execution.get("entry_order_id", ""),
            "stop_order_id": execution.get("stop_order_id", ""),
            "tp_order_id": execution.get("tp_order_id", ""),
        }
        if signal:
            position["signal_scores"] = signal.get("signals", {})
            position["regime"] = signal.get("regime", "unknown")
            position["entry_confidence"] = signal.get("confidence", 0.0)
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

        for pos in data["open_positions"]:
            scores = pos.get("signal_scores", {})
            entry = float(pos.get("entry_price", 0.0))
            insert_trade({
                "timestamp": now,
                "asset": signal.get("asset") if signal else None,
                "direction": "buy",
                "entry_price": entry,
                "exit_price": filled_price,
                "quantity": pos.get("quantity"),
                "regime": pos.get("regime"),
                "signal_source": "fast_brain",
                "confidence": pos.get("entry_confidence"),
                "pnl_pct": (filled_price - entry) / entry if entry > 0 and filled_price > 0 else None,
                "exit_reason": "signal",
                "technical_score": scores.get("technical_score"),
                "funding_rate_score": scores.get("funding_rate_score"),
                "oi_delta_score": scores.get("oi_delta_score"),
                "ema_cross_score": scores.get("ema_cross_score"),
                "composite_score": scores.get("composite_score"),
            })

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

    acquire_lock()
    try:
        _run_pipeline(args, SYMBOL_PERP)
    finally:
        release_lock()


def _run_pipeline(args, SYMBOL_PERP: str) -> None:
    logger.info("=" * 60)
    logger.info("TRADING DATA PIPELINE")
    logger.info("=" * 60)

    # 0. Sync positions against exchange (detect externally-closed positions)
    logger.info("[0/6] Syncing open positions against exchange...")
    from src.execution.position_monitor import sync_positions
    try:
        sync_positions(symbol=args.symbol)
    except Exception as e:
        logger.warning(f"      Position sync failed (non-fatal): {e}")

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

    if not result["approved"] and "drawdown" in result["reason"].lower():
        from src.notifications.telegram import send_drawdown_warning
        dd_pct = (peak_equity - current_equity) / peak_equity * 100 if peak_equity > 0 else 0.0
        send_drawdown_warning(current_equity, peak_equity, dd_pct)

    if result["approved"] and result["action"] != "hold":
        signal_price = float(signal.get("current_price", 0.0))
        sizing = calculate_position_size(signal, current_equity, current_price=signal_price)
        execution = execute_signal(result, sizing, symbol=args.symbol)
        if execution["success"]:
            logger.info(f"      Order placed: {execution}")

            # Compute close PnL before update_positions clears positions
            close_pnl = None
            if result["action"] == "close_long":
                commission_pct = 0.001
                fp = execution["filled_price"]
                close_pnl = sum(
                    (fp - float(p["entry_price"])) * float(p["quantity"])
                    - (float(p["entry_price"]) + fp) * float(p["quantity"]) * commission_pct
                    for p in open_positions
                    if float(p.get("entry_price", 0)) > 0 and fp > 0
                )

            update_positions(result["action"], sizing, execution, signal=signal)

            from src.notifications.telegram import send_trade_alert
            new_equity = load_equity()
            if result["action"] == "open_long":
                send_trade_alert(
                    "open", args.symbol, execution["filled_price"],
                    signal["regime"], signal["confidence"],
                    stop_price=sizing["stop_price"], tp_price=sizing["tp_price"],
                    equity=new_equity,
                )
            elif result["action"] == "close_long":
                send_trade_alert(
                    "close", args.symbol, execution["filled_price"],
                    signal["regime"], signal["confidence"],
                    pnl=close_pnl, exit_reason="signal", equity=new_equity,
                )
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
