"""
Telegram notification module — outbound only.

All functions are wrapped in try/except so a network failure or missing config
never crashes the trading pipeline.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger("telegram")

_BASE_DIR = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _BASE_DIR / "config" / "telegram.json"

# Loaded once at import; None means notifications are silently disabled.
_BOT_TOKEN: str | None = None
_CHAT_ID: str | None = None


def _load_config() -> None:
    global _BOT_TOKEN, _CHAT_ID
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        _BOT_TOKEN = cfg.get("bot_token")
        _CHAT_ID = str(cfg.get("chat_id", ""))
        if not _BOT_TOKEN or not _CHAT_ID:
            logger.warning("telegram.json missing bot_token or chat_id — notifications disabled")
            _BOT_TOKEN = None
            _CHAT_ID = None
    except FileNotFoundError:
        logger.warning("config/telegram.json not found — notifications disabled")
    except Exception as e:
        logger.warning(f"Failed to load telegram config: {e}")


_load_config()


def send_message(text: str) -> bool:
    """
    Send a plain-text (HTML-formatted) message to the configured chat.
    Returns True on success, False on any failure. Never raises.
    """
    if not _BOT_TOKEN or not _CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": _CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if not resp.ok:
            logger.warning(f"Telegram sendMessage failed: {resp.status_code} {resp.text[:200]}")
            return False
        return True
    except Exception as e:
        logger.warning(f"Telegram sendMessage error: {e}")
        return False


def send_trade_alert(
    action: str,
    symbol: str,
    price: float,
    regime: str,
    confidence: float,
    pnl: float | None = None,
    exit_reason: str | None = None,
    stop_price: float | None = None,
    tp_price: float | None = None,
    equity: float | None = None,
) -> None:
    """
    Send a trade open or close alert.

    action: "open" or "close"
    exit_reason: "signal" | "tp" | "stop"  (for closes)
    stop_price/tp_price: for open alerts only
    """
    try:
        if action == "open":
            lines = [
                "🟢 <b>LONG OPENED</b>",
                f"{symbol} @ <b>${price:,.0f}</b>",
                f"Regime: {regime} | Confidence: {confidence:.2f}",
            ]
            if stop_price is not None and tp_price is not None:
                lines.append(f"Stop: ${stop_price:,.0f} | TP: ${tp_price:,.0f}")
            if equity is not None:
                lines.append(f"Equity: ${equity:,.2f}")

        elif action == "close":
            reason_label = {
                "tp": "TP HIT",
                "stop": "STOP HIT",
                "signal": "SIGNAL",
            }.get(exit_reason or "", "CLOSED")
            lines = [
                f"🔴 <b>LONG CLOSED — {reason_label}</b>",
                f"{symbol} @ <b>${price:,.0f}</b>",
                f"Regime: {regime} | Confidence: {confidence:.2f}",
            ]
            if pnl is not None:
                sign = "+" if pnl >= 0 else "-"
                lines.append(f"PnL: <b>{sign}${abs(pnl):,.2f}</b>")
            if equity is not None:
                lines.append(f"Equity: ${equity:,.2f}")

        else:
            logger.warning(f"send_trade_alert: unknown action {action!r}")
            return

        send_message("\n".join(lines))
    except Exception as e:
        logger.warning(f"send_trade_alert error: {e}")


def send_morning_report(
    equity: float,
    peak_equity: float,
    open_positions: list,
    regime: str,
    trades_today: int,
    pnl_today: float,
    win_rate_7d: float = 0.0,
    wins_7d: int = 0,
    total_7d: int = 0,
    max_drawdown_pct: float = 0.0,
    last_run: str = "",
    bot_running: bool = True,
) -> None:
    """Send daily morning summary."""
    try:
        start_equity = 10_000.0  # baseline
        equity_change_pct = (equity - start_equity) / start_equity * 100

        status_icon = "✅" if bot_running else "⚠️"
        status_label = "Running" if bot_running else "Check needed"

        pnl_sign = "+" if pnl_today >= 0 else ""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        win_rate_str = (
            f"{win_rate_7d:.0%} ({wins_7d}/{total_7d})"
            if total_7d > 0
            else "N/A (no trades)"
        )

        lines = [
            "📊 <b>Trading Bot — Morning Report</b>",
            f"Date: {date_str}",
            "",
            f"Equity: <b>${equity:,.2f}</b> ({equity_change_pct:+.2f}% from start)",
            f"Open positions: {len(open_positions)}",
            f"Regime: {regime}",
            "",
            f"Last 24h: {trades_today} trades | {pnl_sign}${pnl_today:,.2f}",
            f"Win rate (7d): {win_rate_str}",
            f"Max drawdown: {max_drawdown_pct:.1f}%",
            "",
            f"Bot status: {status_icon} {status_label}",
        ]
        if last_run:
            lines.append(f"Last run: {last_run}")

        send_message("\n".join(lines))
    except Exception as e:
        logger.warning(f"send_morning_report error: {e}")


def send_drawdown_warning(
    current_equity: float,
    peak_equity: float,
    drawdown_pct: float,
) -> None:
    """
    Send drawdown alert.
    >= 20%: halt alert (🚨)
    >= 10%: warning (⚠️)
    < 10%: no-op
    """
    try:
        if drawdown_pct >= 20.0:
            lines = [
                "🚨 <b>BOT HALTED — MAX DRAWDOWN</b>",
                f"Drawdown: {drawdown_pct:.1f}% — exceeds 20% limit",
                "All activity paused. Manual review required.",
                f"Equity: ${current_equity:,.2f} (peak: ${peak_equity:,.2f})",
            ]
            send_message("\n".join(lines))
        elif drawdown_pct >= 10.0:
            lines = [
                "⚠️ <b>DRAWDOWN WARNING</b>",
                f"Current drawdown: {drawdown_pct:.1f}% from peak",
                f"Equity: ${current_equity:,.2f} (peak: ${peak_equity:,.2f})",
                "Bot is still running. Review advised.",
            ]
            send_message("\n".join(lines))
        # else: below threshold, silent
    except Exception as e:
        logger.warning(f"send_drawdown_warning error: {e}")
