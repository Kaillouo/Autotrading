"""
Regime Detector — classifies market state for the Fast Brain.

Regimes (priority order):
  1. high_vol  — ATR spike (takes precedence over trend labels)
  2. low_vol   — ATR compression
  3. trending_up   — confirmed uptrend (EMA cross + slope + persistence)
  4. trending_down — confirmed downtrend (mirror of trending_up)
  5. ranging   — default / inconclusive
"""

import numpy as np
import pandas as pd

# ── Thresholds ────────────────────────────────────────────────────────────────

ATR_LOOKBACK = 30          # candles for ATR mean baseline
HIGH_VOL_MULT = 1.5        # ATR > mean * this → high_vol
LOW_VOL_MULT = 0.6         # ATR < mean * this → low_vol
EMA_SLOPE_WINDOW = 5       # candles to measure EMA fast slope direction
CROSS_PERSISTENCE = 3      # minimum candles cross must hold before trend is confirmed


def classify_regime(df: pd.DataFrame, idx: int) -> str:
    """
    Classify the market regime at candle index `idx`.

    Returns one of: 'trending_up', 'trending_down', 'ranging', 'high_vol', 'low_vol'

    Priority: volatility → trend → ranging (default)
    Requires at least 50 warmup candles to produce non-default labels.
    """
    min_required = max(ATR_LOOKBACK, EMA_SLOPE_WINDOW + 1, CROSS_PERSISTENCE + 1, 50)
    if idx < min_required:
        return "ranging"

    row = df.iloc[idx]
    atr = row.get("atr")
    if not atr or pd.isna(atr) or atr <= 0:
        return "ranging"

    # ── 1. Volatility regime (takes precedence) ───────────────────────────────
    atr_window = df["atr"].iloc[idx - ATR_LOOKBACK : idx]
    atr_mean = atr_window.mean()
    if pd.isna(atr_mean) or atr_mean <= 0:
        return "ranging"

    if atr > HIGH_VOL_MULT * atr_mean:
        return "high_vol"
    if atr < LOW_VOL_MULT * atr_mean:
        return "low_vol"

    # ── 2. Trend regime ───────────────────────────────────────────────────────
    ema_cross = row.get("ema_cross")
    ema_slow = row.get("ema_slow")
    ema_fast = row.get("ema_fast")
    close = row.get("close")

    if pd.isna(ema_cross) or pd.isna(ema_slow) or pd.isna(ema_fast) or pd.isna(close):
        return "ranging"

    # EMA fast slope: compare current value to value 5 candles ago
    ema_fast_series = pd.to_numeric(df["ema_fast"].iloc[idx - EMA_SLOPE_WINDOW : idx + 1], errors="coerce")
    if ema_fast_series.isna().any() or len(ema_fast_series) < EMA_SLOPE_WINDOW + 1:
        return "ranging"
    slope_positive = float(ema_fast_series.iloc[-1]) > float(ema_fast_series.iloc[0])
    slope_negative = float(ema_fast_series.iloc[-1]) < float(ema_fast_series.iloc[0])

    # Cross persistence: last CROSS_PERSISTENCE candles must all share the same cross direction
    recent_cross = pd.to_numeric(
        df["ema_cross"].iloc[idx - CROSS_PERSISTENCE + 1 : idx + 1], errors="coerce"
    )
    if recent_cross.isna().any() or len(recent_cross) < CROSS_PERSISTENCE:
        return "ranging"

    cross_up_persistent = bool((recent_cross == 1).all())
    cross_down_persistent = bool((recent_cross == 0).all())

    # Trending up: EMA cross bullish + price above slow EMA + slope rising + persistent
    if (
        int(ema_cross) == 1
        and float(close) > float(ema_slow)
        and slope_positive
        and cross_up_persistent
    ):
        return "trending_up"

    # Trending down: EMA cross bearish + price below slow EMA + slope falling + persistent
    if (
        int(ema_cross) == 0
        and float(close) < float(ema_slow)
        and slope_negative
        and cross_down_persistent
    ):
        return "trending_down"

    return "ranging"


def get_regime_confidence(df: pd.DataFrame, idx: int) -> float:
    """
    Confidence score [0.0, 1.0] for the regime at candle `idx`.

    high_vol  — scales with how far ATR is above the 1.5× threshold (max at 3×)
    low_vol   — scales with how far ATR is below the 0.6× threshold (max at 0.3×)
    trending  — based on slope magnitude + how long the cross has held
    ranging   — fixed 0.5
    """
    regime = classify_regime(df, idx)
    min_required = max(ATR_LOOKBACK, EMA_SLOPE_WINDOW + 1, CROSS_PERSISTENCE + 1, 50)
    if idx < min_required:
        return 0.5

    row = df.iloc[idx]
    atr = row.get("atr")
    if not atr or pd.isna(atr) or atr <= 0:
        return 0.5

    atr_window = df["atr"].iloc[idx - ATR_LOOKBACK : idx]
    atr_mean = atr_window.mean()
    if pd.isna(atr_mean) or atr_mean <= 0:
        return 0.5

    if regime == "high_vol":
        # 1.5× = 0.5 confidence, 3.0× = 1.0 confidence
        ratio = atr / atr_mean
        return float(min(1.0, max(0.5, (ratio - HIGH_VOL_MULT) / (3.0 - HIGH_VOL_MULT) * 0.5 + 0.5)))

    if regime == "low_vol":
        # 0.6× = 0.5 confidence, 0.3× = 1.0 confidence
        ratio = atr / atr_mean
        return float(min(1.0, max(0.5, (LOW_VOL_MULT - ratio) / (LOW_VOL_MULT - 0.3) * 0.5 + 0.5)))

    if regime in ("trending_up", "trending_down"):
        # Component 1: slope magnitude (ema_fast change over window as % of price)
        ema_fast_series = pd.to_numeric(
            df["ema_fast"].iloc[idx - EMA_SLOPE_WINDOW : idx + 1], errors="coerce"
        )
        close = float(row.get("close") or 1)
        slope_magnitude = abs(
            float(ema_fast_series.iloc[-1]) - float(ema_fast_series.iloc[0])
        ) / close
        slope_score = min(0.5, slope_magnitude * 100)  # 0.5% price move in EMA = 0.5

        # Component 2: how many consecutive candles the cross has held (max 20)
        cross_val = 1 if regime == "trending_up" else 0
        streak = 0
        for back in range(idx, max(idx - 20, -1), -1):
            v = pd.to_numeric(df["ema_cross"].iloc[back], errors="coerce")
            if pd.isna(v) or int(v) != cross_val:
                break
            streak += 1
        streak_score = min(0.5, streak / 20 * 0.5)

        return float(min(1.0, max(0.3, 0.5 + slope_score + streak_score)))

    # ranging — moderate baseline confidence
    return 0.5


def get_regime_summary(df: pd.DataFrame) -> dict:
    """
    Classify every candle in `df` and return summary statistics.

    Returns:
        {
          "labels": List[str],               # one per row
          "counts": {regime: int},           # total candle counts
          "transitions": List[dict],         # where regime changed
          "dominant": str,                   # most common overall
        }
    """
    labels = []
    for i in range(len(df)):
        labels.append(classify_regime(df, i))

    counts: dict = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1

    transitions = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            ts = df.iloc[i].get("timestamp", i)
            transitions.append({"idx": i, "timestamp": str(ts), "from": labels[i - 1], "to": labels[i]})

    dominant = max(counts, key=lambda k: counts[k]) if counts else "ranging"

    return {
        "labels": labels,
        "counts": counts,
        "transitions": transitions,
        "dominant": dominant,
    }
