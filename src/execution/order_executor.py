"""
Order Executor — the ONLY file that touches the exchange.

The LLM never calls this. Only pipeline.py calls execute_signal after the
Rules Engine has approved the signal.
"""

import json
import os

from src.data.bybit_client import get_exchange
from src.utils.logger import get_logger

logger = get_logger("order_executor")

_POSITIONS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "positions.json"
)


def _read_positions() -> dict:
    if not os.path.exists(_POSITIONS_PATH):
        return {"open_positions": [], "equity": 10000.0, "peak_equity": 10000.0}
    with open(_POSITIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def execute_signal(
    approved: dict,
    sizing: dict,
    symbol: str = "BTC/USDT",
) -> dict:
    """
    Place a market order on Bybit testnet.

    Args:
        approved: Rules Engine result dict with "action" key.
        sizing: Position sizer result with "quantity".
        symbol: Trading pair, e.g. "BTC/USDT".

    Returns:
        {"success": bool, "order_id": str, "filled_price": float, "error": str | None}
    """
    action = approved.get("action", "hold")

    if action == "hold":
        return {"success": False, "order_id": "", "filled_price": 0.0, "error": "Action is hold — nothing to execute"}

    try:
        exchange = get_exchange()

        if action == "open_long":
            qty = sizing["quantity"]
            if qty <= 0:
                return {"success": False, "order_id": "", "filled_price": 0.0, "error": "Quantity is zero or negative"}

            logger.info(f"Placing market BUY order: {qty:.6f} {symbol}")
            order = exchange.create_market_buy_order(symbol, qty)

        elif action == "close_long":
            pos_data = _read_positions()
            open_positions = pos_data.get("open_positions", [])
            total_qty = sum(float(p.get("quantity", 0.0)) for p in open_positions)

            if total_qty <= 0:
                return {
                    "success": False,
                    "order_id": "",
                    "filled_price": 0.0,
                    "error": "No open quantity to sell",
                }

            logger.info(f"Placing market SELL order: {total_qty:.6f} {symbol} (closing all)")
            order = exchange.create_market_sell_order(symbol, total_qty)

        else:
            return {
                "success": False,
                "order_id": "",
                "filled_price": 0.0,
                "error": f"Unknown action: {action!r}",
            }

        filled_price = float(order.get("average") or order.get("price") or 0.0)
        order_id = str(order.get("id", ""))

        logger.info(
            f"Order filled: id={order_id} side={order.get('side')} "
            f"qty={order.get('amount')} price={filled_price} timestamp={order.get('datetime')}"
        )

        return {
            "success": True,
            "order_id": order_id,
            "filled_price": filled_price,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Order placement failed ({action}): {e}")
        return {"success": False, "order_id": "", "filled_price": 0.0, "error": str(e)}
