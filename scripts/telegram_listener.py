"""
Telegram command listener — polls for inbound messages and handles bot commands.
Runs every 1 minute via Task Scheduler.

Supported commands:
  /history [N]  — last N trades (default 5, max 20)
  /status       — current equity, regime, open positions, last run
"""

import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.utils.logger import get_logger
from src.notifications.telegram import send_message, _BOT_TOKEN, _CHAT_ID

logger = get_logger("telegram_listener")

_OFFSET_PATH = BASE_DIR / "data" / "telegram_offset.json"
_POSITIONS_PATH = BASE_DIR / "data" / "positions.json"
_DB_PATH = BASE_DIR / "data" / "trading.db"
_LOG_PATH = BASE_DIR / "logs" / "trading.log"


# ── Offset persistence ─────────────────────────────────────────────────────────

def _load_offset() -> int:
    try:
        if _OFFSET_PATH.exists():
            with open(_OFFSET_PATH, "r") as f:
                return int(json.load(f).get("offset", 0))
    except Exception:
        pass
    return 0


def _save_offset(offset: int) -> None:
    try:
        os.makedirs(_OFFSET_PATH.parent, exist_ok=True)
        with open(_OFFSET_PATH, "w") as f:
            json.dump({"offset": offset}, f)
    except Exception as e:
        logger.warning(f"Could not save offset: {e}")


# ── Telegram polling ───────────────────────────────────────────────────────────

def _get_updates(offset: int) -> list:
    if not _BOT_TOKEN:
        return []
    try:
        url = f"https://api.telegram.org/bot{_BOT_TOKEN}/getUpdates"
        resp = requests.get(url, params={"offset": offset, "timeout": 5}, timeout=10)
        if resp.ok:
            return resp.json().get("result", [])
    except Exception as e:
        logger.warning(f"getUpdates error: {e}")
    return []


# ── Command handlers ───────────────────────────────────────────────────────────

def _read_positions() -> dict:
    if not _POSITIONS_PATH.exists():
        return {"open_positions": [], "equity": 10000.0, "peak_equity": 10000.0}
    with open(_POSITIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_current_regime() -> str:
    try:
        from src.db.database import query_recent
        from src.signals.regime_detector import classify_regime
        df = query_recent(symbol="BTC/USDT", limit=100)
        if df.empty:
            return "unknown"
        df = df.sort_values("timestamp").reset_index(drop=True)
        return classify_regime(df, len(df) - 1)
    except Exception:
        return "unknown"


def _get_last_run_label() -> str:
    try:
        if not _LOG_PATH.exists():
            return "unknown"
        with open(_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            f.seek(max(0, os.path.getsize(_LOG_PATH) - 2048))
            tail = f.read()
        timestamps = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", tail)
        if not timestamps:
            return "unknown"
        last_ts = datetime.strptime(timestamps[-1], "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
        age_min = (datetime.now(timezone.utc) - last_ts).total_seconds() / 60
        status = "✅" if age_min <= 30 else "⚠️"
        return f"{status} {last_ts.strftime('%H:%M')} ({age_min:.0f}m ago)"
    except Exception:
        return "unknown"


def _handle_history(n: int = 5) -> None:
    n = max(1, min(n, 20))
    if not _DB_PATH.exists():
        send_message("No trade history yet.")
        return

    conn = sqlite3.connect(_DB_PATH)
    try:
        rows = conn.execute(
            """
            SELECT timestamp, exit_reason, entry_price, exit_price, quantity, pnl_pct
            FROM trades
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        send_message("No trades recorded yet.")
        return

    pos_data = _read_positions()
    equity = float(pos_data.get("equity", 10000.0))

    # 7d win rate
    try:
        conn = sqlite3.connect(_DB_PATH)
        cutoff_7d = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        r = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) FROM trades WHERE timestamp >= ?",
            (cutoff_7d,),
        ).fetchone()
        conn.close()
        total_7d, wins_7d = (r[0] or 0), int(r[1] or 0)
        win_rate_str = f"{wins_7d/total_7d:.0%} ({wins_7d}/{total_7d})" if total_7d else "N/A"
    except Exception:
        win_rate_str = "N/A"

    lines = [f"📋 <b>Last {len(rows)} Trades</b>", ""]
    for i, (ts, exit_reason, entry_price, exit_price, qty, pnl_pct) in enumerate(rows, 1):
        reason_map = {"tp": "TP", "stop": "STOP", "signal": "SIGNAL"}
        reason_label = reason_map.get(exit_reason or "", exit_reason or "?")

        if pnl_pct is not None and entry_price is not None and qty is not None:
            pnl_usd = pnl_pct * float(entry_price) * float(qty)
            icon = "✅" if pnl_usd >= 0 else "❌"
            sign = "+" if pnl_usd >= 0 else ""
            pnl_str = f"{sign}${pnl_usd:,.0f}"
        else:
            icon = "❓"
            pnl_str = "N/A"

        # Format timestamp
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            ts_label = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            ts_label = ts[:16]

        entry_str = f"${float(entry_price):,.0f}" if entry_price else "?"
        exit_str = f"${float(exit_price):,.0f}" if exit_price else "?"

        lines.append(f"{i}. {icon} LONG → {reason_label} | {pnl_str} | {ts_label}")
        lines.append(f"   Entry {entry_str} → Exit {exit_str}")
        lines.append("")

    lines.append(f"Equity: ${equity:,.2f} | 7d win rate: {win_rate_str}")

    send_message("\n".join(lines))


def _handle_status() -> None:
    pos_data = _read_positions()
    equity = float(pos_data.get("equity", 10000.0))
    peak_equity = float(pos_data.get("peak_equity", 10000.0))
    open_positions = pos_data.get("open_positions", [])

    drawdown_pct = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0.0
    regime = _get_current_regime()
    last_run = _get_last_run_label()

    lines = [
        "📈 <b>Bot Status</b>",
        f"Equity: <b>${equity:,.2f}</b> (peak: ${peak_equity:,.2f})",
        f"Drawdown: {drawdown_pct:.1f}%",
        f"Open positions: {len(open_positions)}",
        f"Regime: {regime}",
        f"Last run: {last_run}",
    ]
    send_message("\n".join(lines))


# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    if not _BOT_TOKEN or not _CHAT_ID:
        logger.warning("Telegram config missing — listener exiting.")
        return

    offset = _load_offset()
    updates = _get_updates(offset)

    for update in updates:
        update_id = update.get("update_id", 0)
        offset = update_id + 1  # advance past this update

        message = update.get("message", {})
        chat_id = str(message.get("chat", {}).get("id", ""))
        text = (message.get("text") or "").strip()

        # Only respond to messages from the configured chat
        if chat_id != _CHAT_ID:
            continue

        if not text.startswith("/"):
            continue

        parts = text.split()
        cmd = parts[0].lower().split("@")[0]  # strip @botname suffix

        logger.info(f"Command received: {text!r}")

        if cmd == "/history":
            n = 5
            if len(parts) > 1:
                try:
                    n = int(parts[1])
                except ValueError:
                    pass
            _handle_history(n)

        elif cmd == "/status":
            _handle_status()

        else:
            send_message(
                f"Unknown command: <code>{cmd}</code>\n\n"
                "Available commands:\n"
                "/history [N] — last N trades (default 5)\n"
                "/status — current equity and bot state"
            )

    _save_offset(offset)


if __name__ == "__main__":
    main()
