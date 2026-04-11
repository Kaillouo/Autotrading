"""
Hard Rules Engine — pure Python, no LLM.

Reads an approved signal dict and validates it against all risk constraints.
Returns an approval dict with action and reason. The LLM never calls this
directly — only pipeline.py orchestrates the flow.
"""

import json
import os

from config.signal_contract import is_signal_expired
from src.utils.logger import get_logger

logger = get_logger("rules_engine")

_RISK_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "risk.json"
)


def _load_risk_config() -> dict:
    with open(_RISK_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_signal(
    signal: dict,
    open_positions: list,
    current_equity: float,
    peak_equity: float,
) -> dict:
    """
    Validate a signal against all risk constraints.

    Args:
        signal: Full signal dict (from signal.json / fast_brain output).
        open_positions: List of open position dicts (from positions.json).
        current_equity: Total account equity right now.
        peak_equity: Highest equity ever recorded (for drawdown check).

    Returns:
        {
            "approved": bool,
            "reason": str,
            "action": "open_long" | "close_long" | "hold",
        }

    Check order (first failure rejects):
        1. Signal expired (TTL)
        2. direction == "hold"
        3. confidence < min_signal_confidence
        4. direction == "buy":
            a. Already at max_open_positions
            b. Drawdown from peak_equity > max_drawdown_halt (HALT)
            c. Approved → action = "open_long"
        5. direction == "sell":
            a. No open positions
            b. Approved → action = "close_long"
    """
    risk = _load_risk_config()

    # 1. TTL check
    if is_signal_expired(signal):
        reason = "Signal expired (TTL exceeded)"
        logger.warning(f"Rules Engine REJECTED: {reason}")
        return {"approved": False, "reason": reason, "action": "hold"}

    direction = signal.get("direction", "hold")
    confidence = float(signal.get("confidence", 0.0))
    min_confidence = risk["min_signal_confidence"]

    # 2. Hold direction — nothing to do, not a rejection
    if direction == "hold":
        reason = "Signal direction is hold"
        logger.info(f"Rules Engine HOLD: {reason}")
        return {"approved": True, "reason": reason, "action": "hold"}

    # 3. Confidence gate
    if confidence < min_confidence:
        reason = f"Confidence {confidence:.3f} below minimum {min_confidence}"
        logger.warning(f"Rules Engine REJECTED: {reason}")
        return {"approved": False, "reason": reason, "action": "hold"}

    # 4. Buy signal checks
    if direction == "buy":
        # 4a. Max open positions
        if len(open_positions) >= risk["max_open_positions"]:
            reason = f"Max open positions reached ({len(open_positions)}/{risk['max_open_positions']})"
            logger.warning(f"Rules Engine REJECTED: {reason}")
            return {"approved": False, "reason": reason, "action": "hold"}

        # 4b. Drawdown halt
        if peak_equity > 0:
            drawdown = (peak_equity - current_equity) / peak_equity
            if drawdown > risk["max_drawdown_halt"]:
                reason = (
                    f"HALT: drawdown {drawdown:.1%} exceeds max "
                    f"{risk['max_drawdown_halt']:.1%} — trading suspended"
                )
                logger.critical(f"Rules Engine HALT: {reason}")
                return {"approved": False, "reason": reason, "action": "hold"}

        reason = f"Buy signal approved (confidence={confidence:.3f}, open_positions={len(open_positions)})"
        logger.info(f"Rules Engine APPROVED open_long: {reason}")
        return {"approved": True, "reason": reason, "action": "open_long"}

    # 5. Sell signal checks
    if direction == "sell":
        if not open_positions:
            reason = "Sell signal but no open positions to close"
            logger.warning(f"Rules Engine REJECTED: {reason}")
            return {"approved": False, "reason": reason, "action": "hold"}

        reason = f"Sell signal approved — closing {len(open_positions)} position(s)"
        logger.info(f"Rules Engine APPROVED close_long: {reason}")
        return {"approved": True, "reason": reason, "action": "close_long"}

    # Unknown direction
    reason = f"Unknown direction: {direction!r}"
    logger.error(f"Rules Engine REJECTED: {reason}")
    return {"approved": False, "reason": reason, "action": "hold"}
