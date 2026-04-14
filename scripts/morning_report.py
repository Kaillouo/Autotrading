"""
Morning report script — runs daily at 08:00 via Task Scheduler.

Reads: positions.json, trading.db, trading.log
Sends: morning summary via Telegram
"""

import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is on path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.utils.logger import get_logger
from src.notifications.telegram import send_morning_report

logger = get_logger("morning_report")

_POSITIONS_PATH = BASE_DIR / "data" / "positions.json"
_DB_PATH = BASE_DIR / "data" / "trading.db"
_LOG_PATH = BASE_DIR / "logs" / "trading.log"


def _read_positions() -> dict:
    if not _POSITIONS_PATH.exists():
        return {"open_positions": [], "equity": 10000.0, "peak_equity": 10000.0}
    with open(_POSITIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _query_trade_stats() -> dict:
    """Query last-24h and 7d trade stats from DB."""
    if not _DB_PATH.exists():
        return {"trades_today": 0, "pnl_today": 0.0, "wins_7d": 0, "total_7d": 0}

    conn = sqlite3.connect(_DB_PATH)
    try:
        now_utc = datetime.now(timezone.utc)
        cutoff_24h = (now_utc - timedelta(hours=24)).isoformat()
        cutoff_7d = (now_utc - timedelta(days=7)).isoformat()

        # Last 24h: count + approximate USD PnL (pnl_pct * entry_price * quantity)
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS cnt,
                COALESCE(SUM(
                    CASE WHEN pnl_pct IS NOT NULL AND entry_price IS NOT NULL AND quantity IS NOT NULL
                         THEN pnl_pct * entry_price * quantity
                         ELSE 0 END
                ), 0) AS pnl_usd
            FROM trades
            WHERE timestamp >= ?
            """,
            (cutoff_24h,),
        ).fetchone()
        trades_today = row[0] if row else 0
        pnl_today = float(row[1]) if row else 0.0

        # 7d win rate
        row7 = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) AS wins
            FROM trades
            WHERE timestamp >= ?
            """,
            (cutoff_7d,),
        ).fetchone()
        total_7d = row7[0] if row7 else 0
        wins_7d = int(row7[1] or 0) if row7 else 0
    finally:
        conn.close()

    return {
        "trades_today": trades_today,
        "pnl_today": pnl_today,
        "wins_7d": wins_7d,
        "total_7d": total_7d,
    }


def _get_current_regime() -> str:
    """Get regime from latest candle in DB via regime_detector."""
    try:
        from src.db.database import query_recent
        from src.signals.regime_detector import classify_regime, get_regime_confidence
        df = query_recent(symbol="BTC/USDT", limit=100)
        if df.empty:
            return "unknown"
        df = df.sort_values("timestamp").reset_index(drop=True)
        idx = len(df) - 1
        regime = classify_regime(df, idx)
        confidence = get_regime_confidence(df, idx)
        return f"{regime} (confidence: {confidence:.2f})"
    except Exception as e:
        logger.warning(f"Could not determine regime: {e}")
        return "unknown"


def _get_last_run() -> tuple[str, bool]:
    """
    Parse trading.log for the most recent pipeline run timestamp.
    Returns (human-readable time string, bot_running bool).
    bot_running = True if last run was within 30 minutes.
    """
    try:
        if not _LOG_PATH.exists():
            return "unknown", False

        # Read last 2KB — enough to find the last timestamp
        with open(_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            f.seek(max(0, os.path.getsize(_LOG_PATH) - 2048))
            tail = f.read()

        # Log format: "2026-04-13 07:45:00,123 INFO ..."
        timestamps = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", tail)
        if not timestamps:
            return "unknown", False

        last_ts_str = timestamps[-1]
        last_ts = datetime.strptime(last_ts_str, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
        age_min = (datetime.now(timezone.utc) - last_ts).total_seconds() / 60
        bot_running = age_min <= 30

        # Format as HH:MM
        run_label = last_ts.strftime("%H:%M")
        return run_label, bot_running
    except Exception as e:
        logger.warning(f"Could not parse last run from log: {e}")
        return "unknown", True


def _compute_max_drawdown(equity: float, peak_equity: float) -> float:
    if peak_equity <= 0:
        return 0.0
    return (peak_equity - equity) / peak_equity * 100


def main() -> None:
    logger.info("Generating morning report...")

    pos_data = _read_positions()
    equity = float(pos_data.get("equity", 10000.0))
    peak_equity = float(pos_data.get("peak_equity", 10000.0))
    open_positions = pos_data.get("open_positions", [])

    stats = _query_trade_stats()
    regime = _get_current_regime()
    last_run, bot_running = _get_last_run()
    max_drawdown_pct = _compute_max_drawdown(equity, peak_equity)

    win_rate_7d = (
        stats["wins_7d"] / stats["total_7d"] if stats["total_7d"] > 0 else 0.0
    )

    send_morning_report(
        equity=equity,
        peak_equity=peak_equity,
        open_positions=open_positions,
        regime=regime,
        trades_today=stats["trades_today"],
        pnl_today=stats["pnl_today"],
        win_rate_7d=win_rate_7d,
        wins_7d=stats["wins_7d"],
        total_7d=stats["total_7d"],
        max_drawdown_pct=max_drawdown_pct,
        last_run=last_run,
        bot_running=bot_running,
    )
    logger.info("Morning report sent.")


if __name__ == "__main__":
    main()
