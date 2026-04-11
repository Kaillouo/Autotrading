"""
Fast Brain — LLM decision layer.

Reads the last 50 candles + recent derivatives from DB, classifies the market
regime, computes sub-signal scores, then calls Haiku to produce a directional
signal. Writes data/signal.json via the signal contract.
"""

import json
import math
import os
from datetime import datetime, timezone

import pandas as pd

from src.db.database import get_connection
from src.signals.regime_detector import classify_regime, get_regime_confidence
from src.signals.technical import compute_indicators
from src.utils.logger import get_logger
from config.signal_contract import write_signal, SignalValidationError
from src.ai.utils.ai_client import call_haiku

logger = get_logger("fast_brain")

_REGIME_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "regime_weights.json"
)
_RISK_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "risk.json"
)
_STRATEGY_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "strategy.json"
)

_HAIKU_SYSTEM = """You are a crypto trading signal generator for a BTC/USDT spot trading bot.
You will receive market context as JSON and must output a trading signal as JSON.
Rules:
- Only generate "buy" signals. "sell" means close an existing long. "hold" means do nothing.
- Only output "buy" if composite_score > 0.62 AND regime_confidence > 0.5
- Only output "sell" if composite_score < 0.35
- Otherwise output "hold"
- confidence must reflect how strong the signal is (0.0-1.0)
- reasoning must be 1-2 sentences explaining the key factor driving the decision
- suggested_stop_pct = (atr * 2.0) / current_price
- suggested_tp_pct = (atr * 4.0) / current_price

Respond with ONLY this JSON, no markdown, no explanation:
{
  "direction": "buy" | "sell" | "hold",
  "confidence": <float>,
  "reasoning": "<string>",
  "suggested_stop_pct": <float>,
  "suggested_tp_pct": <float>
}"""


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fetch_candles(symbol: str = "BTC/USDT", limit: int = 50) -> pd.DataFrame:
    """Return the last `limit` candles in ascending order with indicators computed."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM candles WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
        conn,
        params=(symbol, limit),
    )
    conn.close()
    if df.empty:
        return df
    # Reverse to ascending order (oldest first)
    df = df.iloc[::-1].reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _fetch_derivatives(symbol: str = "BTC/USDT:USDT", limit: int = 10) -> pd.DataFrame:
    """Return the most recent `limit` derivatives rows in ascending order."""
    conn = get_connection()
    df = pd.read_sql_query(
        """SELECT timestamp, funding_rate, funding_rate_zscore, open_interest
           FROM derivatives_snapshots
           WHERE symbol = ?
           ORDER BY timestamp DESC LIMIT ?""",
        conn,
        params=(symbol, limit),
    )
    conn.close()
    if not df.empty:
        df = df.iloc[::-1].reset_index(drop=True)
    return df


# ── Signal score helpers ───────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _clip(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _compute_technical_score(row: pd.Series) -> float:
    rsi = _safe_float(row.get("rsi"), None)
    macd_hist = _safe_float(row.get("macd_hist"), None)
    bb_lower = _safe_float(row.get("bb_lower"), None)
    bb_upper = _safe_float(row.get("bb_upper"), None)
    close = _safe_float(row.get("close"), 0.0)
    atr = _safe_float(row.get("atr"), 0.0)

    scores = []

    if rsi is not None and math.isfinite(rsi):
        # 0 at RSI=30, 1 at RSI=70
        rsi_score = _clip((rsi - 30.0) / 40.0, 0.0, 1.0)
        scores.append(rsi_score)

    if macd_hist is not None and math.isfinite(macd_hist):
        if atr > 0:
            macd_score = _clip(0.5 + macd_hist / (atr * 2.0), 0.0, 1.0)
        else:
            macd_score = 0.5 if macd_hist == 0 else (1.0 if macd_hist > 0 else 0.0)
        scores.append(macd_score)

    if (
        bb_lower is not None and bb_upper is not None
        and math.isfinite(bb_lower) and math.isfinite(bb_upper)
        and bb_upper > bb_lower
    ):
        bb_score = _clip((close - bb_lower) / (bb_upper - bb_lower), 0.0, 1.0)
        scores.append(bb_score)

    return sum(scores) / len(scores) if scores else 0.5


def _compute_funding_rate_score(zscore: float | None) -> float:
    """Low/negative funding → bullish (score > 0.5). High funding → bearish (score < 0.5)."""
    if zscore is None or not math.isfinite(zscore):
        return 0.5
    clamped = _clip(zscore, -10.0, 10.0)
    # sigmoid(-zscore): negative zscore → high score, positive zscore → low score
    return 1.0 / (1.0 + math.exp(clamped))


def _compute_oi_delta_score(oi_series: pd.Series) -> float:
    """Rising OI → bullish. Score = 0.5 + change_pct * 2, clipped to [0, 1]."""
    oi = oi_series.dropna()
    if len(oi) < 2:
        return 0.5
    first = _safe_float(oi.iloc[0], 0.0)
    last = _safe_float(oi.iloc[-1], 0.0)
    if first <= 0:
        return 0.5
    change_pct = (last - first) / first
    return _clip(0.5 + change_pct * 2.0, 0.0, 1.0)


def _compute_ema_cross_score(row: pd.Series) -> float:
    val = row.get("ema_cross")
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.5
    return 1.0 if int(val) == 1 else 0.0


def _compute_composite_score(scores: dict, regime: str, regime_weights: dict) -> float:
    weights = regime_weights.get(regime, regime_weights.get("ranging", {}))
    total_weight = 0.0
    weighted_sum = 0.0
    for key, score in scores.items():
        w = weights.get(key, 0.0)
        weighted_sum += w * score
        total_weight += w
    if total_weight <= 0:
        return 0.5
    return _clip(weighted_sum / total_weight, 0.0, 1.0)


# ── Main entry point ───────────────────────────────────────────────────────────

def run_fast_brain(symbol: str = "BTC/USDT") -> dict:
    """
    Run the Fast Brain pipeline:
    1. Fetch candles + derivatives from DB
    2. Classify regime + confidence
    3. Compute signal scores
    4. Call Haiku for the directional decision
    5. Write signal.json
    6. Return the full signal dict
    """
    symbol_perp = symbol + ":USDT"
    risk_config = _load_json(_RISK_CONFIG_PATH)
    strategy_config = _load_json(_STRATEGY_CONFIG_PATH)
    regime_weights = _load_json(_REGIME_WEIGHTS_PATH)

    # ── Fetch data ─────────────────────────────────────────────────────────────
    candles = _fetch_candles(symbol=symbol, limit=50)
    derivatives = _fetch_derivatives(symbol=symbol_perp, limit=10)

    # Fallback hold signal builder for error cases
    def _hold_signal(reason: str) -> dict:
        ts = datetime.now(timezone.utc).isoformat()
        signal = {
            "timestamp": ts,
            "asset": symbol,
            "direction": "hold",
            "confidence": 0.0,
            "regime": "ranging",
            "signals": {
                "technical_score": 0.5,
                "funding_rate_score": 0.5,
                "oi_delta_score": 0.5,
                "ema_cross_score": 0.5,
                "composite_score": 0.5,
            },
            "reasoning": reason,
            "max_position_pct": risk_config["max_position_pct"],
            "suggested_stop_pct": 0.02,
            "suggested_tp_pct": 0.04,
            "ttl_minutes": strategy_config["signal_ttl_minutes"],
            "current_price": 0.0,
        }
        try:
            write_signal(signal)
        except Exception:
            pass
        return signal

    if candles.empty or len(candles) < 10:
        logger.warning("Not enough candles in DB — writing hold signal")
        return _hold_signal("Insufficient candle data in DB")

    # ── Compute regime ─────────────────────────────────────────────────────────
    # Regime detector expects a full indicator DataFrame
    # candles from DB already have indicators — just rename if needed
    df = candles.copy()
    last_idx = len(df) - 1
    regime = classify_regime(df, last_idx)
    regime_confidence = get_regime_confidence(df, last_idx)

    last = df.iloc[last_idx]
    current_price = _safe_float(last.get("close"), 0.0)
    atr = _safe_float(last.get("atr"), 0.0)

    # Price changes
    price_change_1h = 0.0
    price_change_24h = 0.0
    if len(df) >= 2:
        prev_close = _safe_float(df.iloc[-2].get("close"), current_price)
        price_change_1h = (current_price - prev_close) / prev_close if prev_close > 0 else 0.0
    if len(df) >= 25:
        prev_24h = _safe_float(df.iloc[-25].get("close"), current_price)
        price_change_24h = (current_price - prev_24h) / prev_24h if prev_24h > 0 else 0.0

    # ── Compute sub-scores ─────────────────────────────────────────────────────
    technical_score = _compute_technical_score(last)
    ema_cross_score = _compute_ema_cross_score(last)

    # Funding rate (latest available)
    funding_rate = None
    funding_rate_zscore = None
    if not derivatives.empty:
        latest_deriv = derivatives.iloc[-1]
        funding_rate = _safe_float(latest_deriv.get("funding_rate"), None)
        funding_rate_zscore_raw = latest_deriv.get("funding_rate_zscore")
        if funding_rate_zscore_raw is not None:
            funding_rate_zscore = _safe_float(funding_rate_zscore_raw, None)

    funding_rate_score = _compute_funding_rate_score(funding_rate_zscore)

    # OI delta over last 5 derivatives rows
    oi_series = derivatives["open_interest"] if not derivatives.empty else pd.Series(dtype=float)
    oi_delta_score = _compute_oi_delta_score(oi_series.tail(6))

    # OI % change over last 5 candles (for context dict)
    oi_change_pct = 0.0
    if len(oi_series.dropna()) >= 2:
        oi_vals = oi_series.dropna().tail(6)
        first_oi = _safe_float(oi_vals.iloc[0], 0.0)
        last_oi = _safe_float(oi_vals.iloc[-1], 0.0)
        if first_oi > 0:
            oi_change_pct = (last_oi - first_oi) / first_oi

    scores = {
        "technical_score": technical_score,
        "funding_rate_score": funding_rate_score,
        "oi_delta_score": oi_delta_score,
        "ema_cross_score": ema_cross_score,
    }
    composite_score = _compute_composite_score(scores, regime, regime_weights)

    # BB position for context
    bb_lower = _safe_float(last.get("bb_lower"), 0.0)
    bb_upper = _safe_float(last.get("bb_upper"), 0.0)
    bb_position = 0.5
    if bb_upper > bb_lower > 0:
        bb_position = _clip((current_price - bb_lower) / (bb_upper - bb_lower), 0.0, 1.0)

    # ── Build Haiku context ────────────────────────────────────────────────────
    context = {
        "asset": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
        "regime_confidence": round(regime_confidence, 4),
        "current_price": round(current_price, 2),
        "price_change_1h": round(price_change_1h, 6),
        "price_change_24h": round(price_change_24h, 6),
        "atr": round(atr, 2),
        "signals": {
            "technical_score": round(technical_score, 4),
            "funding_rate_score": round(funding_rate_score, 4),
            "oi_delta_score": round(oi_delta_score, 4),
            "ema_cross_score": round(ema_cross_score, 4),
            "composite_score": round(composite_score, 4),
        },
        "indicators": {
            "rsi": round(_safe_float(last.get("rsi"), 50.0), 2),
            "macd_hist": round(_safe_float(last.get("macd_hist"), 0.0), 4),
            "bb_position": round(bb_position, 4),
            "ema_cross": int(_safe_float(last.get("ema_cross"), 0.0)),
            "funding_rate": funding_rate,
            "funding_rate_zscore": funding_rate_zscore,
            "open_interest_change_pct": round(oi_change_pct, 6),
        },
    }

    prompt = f"Market context:\n{json.dumps(context, indent=2)}\n\nGenerate a trading signal:"

    # ── Call Haiku ─────────────────────────────────────────────────────────────
    try:
        raw_response = call_haiku(prompt=prompt, system=_HAIKU_SYSTEM, max_tokens=256)
        logger.info(f"Haiku raw response: {raw_response}")
        # Strip markdown code fences if Haiku added them despite instructions
        clean = raw_response.strip()
        if clean.startswith("```"):
            # Drop the opening fence line (e.g. "```json\n" or "```\n")
            first_newline = clean.find("\n")
            if first_newline != -1:
                clean = clean[first_newline + 1:]
            else:
                clean = clean[3:]
            # Drop the closing fence
            if clean.rstrip().endswith("```"):
                clean = clean.rstrip()[:-3].rstrip()
            clean = clean.strip()
        haiku_output = json.loads(clean)
    except json.JSONDecodeError as e:
        logger.error(f"Haiku returned invalid JSON: {e} | raw: {raw_response!r}")
        return _hold_signal("LLM returned invalid JSON — defaulting to hold")
    except Exception as e:
        logger.error(f"Haiku call failed: {e}")
        return _hold_signal(f"LLM call failed: {e}")

    # ── Build and write full signal dict ───────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signal = {
        "timestamp": ts,
        "asset": symbol,
        "direction": haiku_output.get("direction", "hold"),
        "confidence": float(haiku_output.get("confidence", 0.0)),
        "regime": regime,
        "signals": {
            "technical_score": round(technical_score, 4),
            "funding_rate_score": round(funding_rate_score, 4),
            "oi_delta_score": round(oi_delta_score, 4),
            "ema_cross_score": round(ema_cross_score, 4),
            "composite_score": round(composite_score, 4),
        },
        "reasoning": haiku_output.get("reasoning", ""),
        "max_position_pct": risk_config["max_position_pct"],
        "suggested_stop_pct": float(haiku_output.get("suggested_stop_pct", atr * 2.0 / max(current_price, 1))),
        "suggested_tp_pct": float(haiku_output.get("suggested_tp_pct", atr * 4.0 / max(current_price, 1))),
        "ttl_minutes": strategy_config["signal_ttl_minutes"],
        "current_price": current_price,
    }

    try:
        write_signal(signal)
        logger.info(
            f"Signal written: direction={signal['direction']} confidence={signal['confidence']:.2f} "
            f"regime={regime} composite={composite_score:.3f}"
        )
    except SignalValidationError as e:
        logger.error(f"Signal validation failed: {e} — writing hold fallback")
        return _hold_signal(f"Validation error: {e}")

    return signal
