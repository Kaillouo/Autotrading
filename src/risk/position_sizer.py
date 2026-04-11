"""
Kelly-lite position sizer — ATR-adjusted.

Takes a validated signal + account state and returns concrete sizing:
quantity, stop price, TP price, and USD size.
"""

import json
import os

from src.utils.logger import get_logger

logger = get_logger("position_sizer")

_RISK_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "risk.json"
)
_MIN_ORDER_USD = 10.0  # Bybit testnet minimum


def _load_risk_config() -> dict:
    with open(_RISK_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_position_size(
    signal: dict,
    current_equity: float,
    current_price: float,
) -> dict:
    """
    Compute position size using Kelly-lite with ATR-adjusted stops.

    Logic:
    - Base size: min(signal.max_position_pct, risk.max_position_pct) * equity
    - Kelly adjustment: base_size * min(confidence, 1.0) * kelly_fraction
    - Minimum size: $10 (Bybit testnet minimum)
    - stop_price: current_price * (1 - suggested_stop_pct)
    - tp_price: current_price * (1 + suggested_tp_pct)

    Returns:
        {"size_usd": float, "quantity": float, "stop_price": float, "tp_price": float}
    """
    risk = _load_risk_config()

    # Guard against degenerate price
    if current_price <= 0:
        current_price = 1.0

    signal_max_pct = float(signal.get("max_position_pct", risk["max_position_pct"]))
    risk_max_pct = risk["max_position_pct"]
    base_pct = min(signal_max_pct, risk_max_pct)

    base_size_usd = base_pct * current_equity

    confidence = min(float(signal.get("confidence", 1.0)), 1.0)
    kelly_fraction = risk["kelly_fraction"]
    size_usd = base_size_usd * confidence * kelly_fraction

    # Enforce minimum order size
    size_usd = max(size_usd, _MIN_ORDER_USD)

    quantity = size_usd / current_price

    stop_pct = float(signal.get("suggested_stop_pct", 0.02))
    tp_pct = float(signal.get("suggested_tp_pct", 0.04))
    stop_price = current_price * (1.0 - stop_pct)
    tp_price = current_price * (1.0 + tp_pct)

    result = {
        "size_usd": round(size_usd, 4),
        "quantity": round(quantity, 6),
        "stop_price": round(stop_price, 2),
        "tp_price": round(tp_price, 2),
    }

    logger.info(
        f"Position sized: ${size_usd:.2f} | qty={quantity:.6f} BTC | "
        f"stop={stop_price:.2f} | tp={tp_price:.2f}"
    )
    return result
