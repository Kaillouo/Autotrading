"""
signal.json contract — the ONLY interface between the LLM (Fast Brain) and the Rules Engine.

The LLM writes this file. The Rules Engine reads it. No direct LLM → exchange calls.

Pure Python, stdlib only. No LLM calls, no external dependencies.
"""

import json
import os
from datetime import datetime, timezone
from typing import TypedDict

# Canonical location of the signal file, relative to project root
_SIGNAL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "signal.json")

VALID_DIRECTIONS = frozenset({"buy", "sell", "hold"})

REQUIRED_FIELDS = {
    "timestamp",
    "asset",
    "direction",
    "confidence",
    "regime",
    "signals",
    "reasoning",
    "max_position_pct",
    "suggested_stop_pct",
    "suggested_tp_pct",
    "ttl_minutes",
}


class SignalDict(TypedDict):
    timestamp: str          # ISO-8601 UTC  e.g. "2026-03-31T14:23:00Z"
    asset: str              # e.g. "BTC/USDT"
    direction: str          # "buy" | "sell" | "hold"
    confidence: float       # 0.0 – 1.0
    regime: str             # e.g. "trending_up" | "ranging" | "high_vol" | "low_vol"
    signals: dict           # sub-scores, e.g. {"technical_score": 0.65, "funding_rate_score": 0.8}
    reasoning: str          # human-readable explanation from the LLM
    max_position_pct: float # fraction of capital, e.g. 0.08
    suggested_stop_pct: float
    suggested_tp_pct: float
    ttl_minutes: int        # signal expires after this many minutes from timestamp


class SignalValidationError(ValueError):
    """Raised when a signal dict fails validation."""


def _validate(signal: dict) -> None:
    """
    Validate signal dict against contract rules.
    Raises SignalValidationError with a descriptive message on failure.
    """
    missing = REQUIRED_FIELDS - set(signal.keys())
    if missing:
        raise SignalValidationError(f"Signal missing required fields: {sorted(missing)}")

    direction = signal["direction"]
    if direction not in VALID_DIRECTIONS:
        raise SignalValidationError(
            f"Invalid direction '{direction}'. Must be one of {sorted(VALID_DIRECTIONS)}"
        )

    confidence = signal["confidence"]
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        raise SignalValidationError(f"confidence must be a float, got {type(confidence).__name__}")
    if not (0.0 <= confidence <= 1.0):
        raise SignalValidationError(
            f"confidence must be between 0.0 and 1.0, got {confidence}"
        )

    if not isinstance(signal["signals"], dict):
        raise SignalValidationError("signals must be a dict of sub-scores")

    for pct_field in ("max_position_pct", "suggested_stop_pct", "suggested_tp_pct"):
        try:
            float(signal[pct_field])
        except (TypeError, ValueError):
            raise SignalValidationError(f"{pct_field} must be a number")

    try:
        int(signal["ttl_minutes"])
    except (TypeError, ValueError):
        raise SignalValidationError("ttl_minutes must be an integer")


def write_signal(signal: dict, path: str | None = None) -> None:
    """
    Validate and write a signal dict to data/signal.json (or a custom path).

    Raises SignalValidationError if the signal is invalid.
    """
    _validate(signal)
    target = path or _SIGNAL_PATH
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(signal, f, indent=2)


def read_signal(path: str | None = None) -> dict | None:
    """
    Read signal.json and return the dict, or None if the file does not exist.

    Does NOT validate on read — the Rules Engine should call is_signal_expired()
    to decide whether to act on the signal.
    """
    target = path or _SIGNAL_PATH
    if not os.path.exists(target):
        return None
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def is_signal_expired(signal: dict) -> bool:
    """
    Return True if the signal has passed its TTL.

    Compares signal["timestamp"] + signal["ttl_minutes"] against now (UTC).
    Returns True (treat as expired) on any parse error.
    """
    try:
        ts_raw = signal["timestamp"]
        # Accept both "Z" suffix and explicit "+00:00"
        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        ttl_minutes = int(signal["ttl_minutes"])
        expiry = ts.timestamp() + ttl_minutes * 60
        return datetime.now(timezone.utc).timestamp() > expiry
    except Exception:
        return True  # treat malformed signal as expired
