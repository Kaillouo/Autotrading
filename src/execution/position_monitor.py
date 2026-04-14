"""
Position Monitor — syncs positions.json against actual exchange state.

Detects positions that were closed externally (stop or TP hit while PC was off)
and updates positions.json accordingly.
"""

import json
import os
from datetime import datetime, timezone

from src.data.bybit_client import get_exchange
from src.db.database import insert_trade
from src.utils.logger import get_logger

logger = get_logger("position_monitor")

_POSITIONS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "positions.json"
)


def _read_positions() -> dict:
    if not os.path.exists(_POSITIONS_PATH):
        return {"open_positions": [], "equity": 10000.0, "peak_equity": 10000.0}
    with open(_POSITIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_positions(data: dict) -> None:
    os.makedirs(os.path.dirname(_POSITIONS_PATH), exist_ok=True)
    with open(_POSITIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def sync_positions(symbol: str = "BTC/USDT") -> None:
    """
    Compare positions.json against actual open orders on Bybit.

    If a position's stop or TP order is no longer open (filled or cancelled),
    mark the position as closed and log the outcome.
    """
    data = _read_positions()
    open_positions = data.get("open_positions", [])

    if not open_positions:
        logger.debug("No open positions to sync.")
        return

    exchange = get_exchange()
    commission_pct = 0.001
    positions_to_keep = []
    total_pnl = 0.0
    pending_alerts = []

    for pos in open_positions:
        stop_order_id = pos.get("stop_order_id", "")
        tp_order_id = pos.get("tp_order_id", "")

        # If no order IDs recorded, position predates this feature — skip sync
        if not stop_order_id and not tp_order_id:
            logger.debug("Position has no order IDs — skipping exchange sync for this entry.")
            positions_to_keep.append(pos)
            continue

        stop_status = None
        tp_status = None
        filled_price = 0.0
        trigger_label = None

        if stop_order_id:
            try:
                stop_order = exchange.fetch_order(stop_order_id, symbol=symbol)
                stop_status = stop_order.get("status", "unknown")
                logger.debug(f"Stop order {stop_order_id} status: {stop_status}")

                if stop_status in ("closed", "filled"):
                    filled_price = float(
                        stop_order.get("average") or stop_order.get("price") or 0.0
                    )
                    trigger_label = "stop-loss"
            except Exception as e:
                logger.warning(f"Could not fetch stop order {stop_order_id}: {e}")

        if tp_order_id and trigger_label is None:
            try:
                tp_order = exchange.fetch_order(tp_order_id, symbol=symbol)
                tp_status = tp_order.get("status", "unknown")
                logger.debug(f"TP order {tp_order_id} status: {tp_status}")

                if tp_status in ("closed", "filled"):
                    filled_price = float(
                        tp_order.get("average") or tp_order.get("price") or 0.0
                    )
                    trigger_label = "take-profit"
            except Exception as e:
                logger.warning(f"Could not fetch TP order {tp_order_id}: {e}")

        if trigger_label is not None:
            # Position was closed externally by stop or TP
            qty = float(pos.get("quantity", 0.0))
            entry = float(pos.get("entry_price", 0.0))
            if entry > 0 and filled_price > 0:
                gross_pnl = (filled_price - entry) * qty
                commission = (entry + filled_price) * qty * commission_pct
                pnl = gross_pnl - commission
            else:
                pnl = 0.0
            total_pnl += pnl

            scores = pos.get("signal_scores", {})
            insert_trade({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "asset": symbol,
                "direction": "buy",
                "entry_price": entry,
                "exit_price": filled_price,
                "quantity": qty,
                "regime": pos.get("regime"),
                "signal_source": "fast_brain",
                "confidence": pos.get("entry_confidence"),
                "pnl_pct": (filled_price - entry) / entry if entry > 0 and filled_price > 0 else None,
                "exit_reason": "stop" if trigger_label == "stop-loss" else "tp",
                "technical_score": scores.get("technical_score"),
                "funding_rate_score": scores.get("funding_rate_score"),
                "oi_delta_score": scores.get("oi_delta_score"),
                "ema_cross_score": scores.get("ema_cross_score"),
                "composite_score": scores.get("composite_score"),
            })

            pending_alerts.append({
                "exit_price": filled_price,
                "pnl": pnl,
                "regime": pos.get("regime", "unknown"),
                "confidence": pos.get("entry_confidence", 0.0),
                "exit_reason": "stop" if trigger_label == "stop-loss" else "tp",
            })

            logger.info(
                f"Position externally closed via {trigger_label}: "
                f"entry={entry:.2f} exit={filled_price:.2f} qty={qty:.6f} pnl={pnl:.4f}"
            )
        else:
            # Both orders still open — position live
            positions_to_keep.append(pos)

    closed_count = len(open_positions) - len(positions_to_keep)
    if closed_count > 0:
        data["open_positions"] = positions_to_keep
        data["equity"] = round(data.get("equity", 10000.0) + total_pnl, 4)
        data["peak_equity"] = round(
            max(data["equity"], data.get("peak_equity", data["equity"])), 4
        )
        _write_positions(data)
        logger.info(
            f"Sync complete: {closed_count} position(s) closed externally. "
            f"Total PnL={total_pnl:.4f} | New equity={data['equity']:.2f}"
        )

        from src.notifications.telegram import send_trade_alert
        for alert in pending_alerts:
            send_trade_alert(
                "close", symbol, alert["exit_price"],
                alert["regime"], alert["confidence"],
                pnl=alert["pnl"], exit_reason=alert["exit_reason"],
                equity=data["equity"],
            )
    else:
        logger.info("Sync complete: all positions still open on exchange.")
