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
    Place orders on Bybit testnet.

    For open_long: places market buy + stop-market sell + limit TP sell.
    For close_long: cancels open stop/TP orders first, then places market sell.

    Returns:
        {
            "success": bool,
            "entry_order_id": str,
            "stop_order_id": str,
            "tp_order_id": str,
            "filled_price": float,
            "error": str | None,
        }
    """
    action = approved.get("action", "hold")

    if action == "hold":
        return {
            "success": False,
            "entry_order_id": "",
            "stop_order_id": "",
            "tp_order_id": "",
            "filled_price": 0.0,
            "error": "Action is hold — nothing to execute",
        }

    try:
        exchange = get_exchange()

        if action == "open_long":
            qty = sizing["quantity"]
            if qty <= 0:
                return {
                    "success": False,
                    "entry_order_id": "",
                    "stop_order_id": "",
                    "tp_order_id": "",
                    "filled_price": 0.0,
                    "error": "Quantity is zero or negative",
                }

            # 1. Market buy
            logger.info(f"Placing market BUY order: {qty:.6f} {symbol}")
            entry_order = exchange.create_market_buy_order(symbol, qty)
            entry_order_id = str(entry_order.get("id", ""))
            filled_price = float(entry_order.get("average") or entry_order.get("price") or 0.0)
            logger.info(
                f"Entry filled: id={entry_order_id} qty={entry_order.get('amount')} "
                f"price={filled_price} timestamp={entry_order.get('datetime')}"
            )

            stop_order_id = ""
            tp_order_id = ""

            # 2. Stop-loss order
            try:
                logger.info(
                    f"Placing stop-market SELL order: {qty:.6f} {symbol} @ stop={sizing['stop_price']}"
                )
                stop_order = exchange.create_order(
                    symbol=symbol,
                    type="stop_market",
                    side="sell",
                    amount=qty,
                    params={"stopPrice": sizing["stop_price"], "reduceOnly": True},
                )
                stop_order_id = str(stop_order.get("id", ""))
                logger.info(f"Stop order placed: id={stop_order_id}")
            except Exception as e:
                logger.critical(
                    f"Stop-loss placement FAILED after entry fill: {e}. "
                    "Attempting emergency market sell to avoid unprotected position."
                )
                try:
                    exchange.create_market_sell_order(symbol, qty)
                    logger.info("Emergency sell executed. Position closed.")
                except Exception as sell_err:
                    logger.critical(f"Emergency sell also failed: {sell_err}")
                return {
                    "success": False,
                    "entry_order_id": entry_order_id,
                    "stop_order_id": "",
                    "tp_order_id": "",
                    "filled_price": filled_price,
                    "error": f"Stop placement failed: {e}",
                }

            # 3. Take-profit order
            try:
                logger.info(
                    f"Placing limit TP SELL order: {qty:.6f} {symbol} @ tp={sizing['tp_price']}"
                )
                tp_order = exchange.create_order(
                    symbol=symbol,
                    type="limit",
                    side="sell",
                    amount=qty,
                    price=sizing["tp_price"],
                    params={"reduceOnly": True},
                )
                tp_order_id = str(tp_order.get("id", ""))
                logger.info(f"TP order placed: id={tp_order_id}")
            except Exception as e:
                logger.critical(
                    f"TP placement FAILED after entry fill: {e}. "
                    "Cancelling stop order and attempting emergency sell."
                )
                if stop_order_id:
                    try:
                        exchange.cancel_order(stop_order_id, symbol=symbol)
                    except Exception as cancel_err:
                        logger.warning(f"Could not cancel stop order {stop_order_id}: {cancel_err}")
                try:
                    exchange.create_market_sell_order(symbol, qty)
                    logger.info("Emergency sell executed. Position closed.")
                except Exception as sell_err:
                    logger.critical(f"Emergency sell also failed: {sell_err}")
                return {
                    "success": False,
                    "entry_order_id": entry_order_id,
                    "stop_order_id": stop_order_id,
                    "tp_order_id": "",
                    "filled_price": filled_price,
                    "error": f"TP placement failed: {e}",
                }

            return {
                "success": True,
                "entry_order_id": entry_order_id,
                "stop_order_id": stop_order_id,
                "tp_order_id": tp_order_id,
                "filled_price": filled_price,
                "error": None,
            }

        elif action == "close_long":
            pos_data = _read_positions()
            open_positions = pos_data.get("open_positions", [])
            total_qty = sum(float(p.get("quantity", 0.0)) for p in open_positions)

            if total_qty <= 0:
                return {
                    "success": False,
                    "entry_order_id": "",
                    "stop_order_id": "",
                    "tp_order_id": "",
                    "filled_price": 0.0,
                    "error": "No open quantity to sell",
                }

            # Cancel outstanding stop/TP orders before closing
            for pos in open_positions:
                for order_id in [pos.get("stop_order_id"), pos.get("tp_order_id")]:
                    if order_id:
                        try:
                            exchange.cancel_order(order_id, symbol=symbol)
                            logger.info(f"Cancelled order {order_id} before closing position")
                        except Exception as e:
                            logger.warning(f"Could not cancel order {order_id}: {e}")

            logger.info(f"Placing market SELL order: {total_qty:.6f} {symbol} (closing all)")
            order = exchange.create_market_sell_order(symbol, total_qty)

            filled_price = float(order.get("average") or order.get("price") or 0.0)
            order_id = str(order.get("id", ""))
            logger.info(
                f"Close filled: id={order_id} qty={order.get('amount')} "
                f"price={filled_price} timestamp={order.get('datetime')}"
            )

            return {
                "success": True,
                "entry_order_id": order_id,
                "stop_order_id": "",
                "tp_order_id": "",
                "filled_price": filled_price,
                "error": None,
            }

        else:
            return {
                "success": False,
                "entry_order_id": "",
                "stop_order_id": "",
                "tp_order_id": "",
                "filled_price": 0.0,
                "error": f"Unknown action: {action!r}",
            }

    except Exception as e:
        logger.error(f"Order placement failed ({action}): {e}")
        return {
            "success": False,
            "entry_order_id": "",
            "stop_order_id": "",
            "tp_order_id": "",
            "filled_price": 0.0,
            "error": str(e),
        }
