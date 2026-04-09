"""
Tests for src/signals/regime_detector.py

Tests use synthetic DataFrames — no DB required.
Run with: pytest tests/test_regime.py -v
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.signals.regime_detector import (
    classify_regime,
    get_regime_confidence,
    get_regime_summary,
    ATR_LOOKBACK,
    CROSS_PERSISTENCE,
    EMA_SLOPE_WINDOW,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_df(n: int = 100, **overrides) -> pd.DataFrame:
    """
    Build a minimal candle DataFrame with flat indicators.
    All ATR values default to 100.0, EMA cross=1, close > ema_slow.
    Override any column by passing keyword args (scalar or array).
    """
    base = {
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "open": 50_000.0,
        "high": 50_500.0,
        "low": 49_500.0,
        "close": 50_000.0,
        "volume": 100.0,
        "atr": 100.0,
        "ema_fast": 50_100.0,
        "ema_slow": 49_900.0,
        "ema_cross": 1,
        "rsi": 50.0,
        "macd_hist": 0.0,
        "bb_lower": 49_000.0,
        "bb_upper": 51_000.0,
        "funding_rate_zscore": 0.0,
        "open_interest": 1_000.0,
    }
    for k, v in overrides.items():
        base[k] = v
    return pd.DataFrame(base)


def set_cross_persistent(df: pd.DataFrame, idx: int, value: int, lookback: int = CROSS_PERSISTENCE) -> pd.DataFrame:
    """Set ema_cross to `value` for the last `lookback` candles up to idx."""
    df = df.copy()
    start = max(0, idx - lookback + 1)
    df.loc[start:idx, "ema_cross"] = value
    return df


# ── Warmup guard ──────────────────────────────────────────────────────────────


def test_warmup_returns_ranging():
    df = make_df(100)
    # idx < 50 (min_required) should always be ranging
    for i in range(50):
        assert classify_regime(df, i) == "ranging", f"Expected ranging at idx={i}"


# ── Volatility regimes ────────────────────────────────────────────────────────


def test_high_vol_detection():
    """ATR spike above 1.5× 30-period mean → high_vol."""
    n = 100
    atr_vals = [100.0] * n
    idx = 80
    atr_vals[idx] = 200.0  # 2× mean → high_vol
    df = make_df(n, atr=atr_vals)
    assert classify_regime(df, idx) == "high_vol"


def test_low_vol_detection():
    """ATR compression below 0.6× 30-period mean → low_vol."""
    n = 100
    atr_vals = [100.0] * n
    idx = 80
    atr_vals[idx] = 50.0  # 0.5× mean → low_vol
    df = make_df(n, atr=atr_vals)
    assert classify_regime(df, idx) == "low_vol"


def test_vol_takes_priority_over_trend():
    """high_vol should win even when EMA cross signals trending_up."""
    n = 100
    atr_vals = [100.0] * n
    idx = 80
    atr_vals[idx] = 200.0  # high_vol spike
    df = make_df(n, atr=atr_vals)
    # All other trend conditions are met (ema_cross=1, close > ema_slow, slope positive)
    # high_vol must still win
    assert classify_regime(df, idx) == "high_vol"


# ── Trending regimes ──────────────────────────────────────────────────────────


def test_trending_up_all_conditions_met():
    """All conditions for trending_up → must return trending_up."""
    n = 100
    idx = 80
    # Rising ema_fast over last 5 candles: 50100, 50110, 50120, 50130, 50140, 50150
    ema_fast_vals = [50_100.0] * n
    for j in range(5):
        ema_fast_vals[idx - EMA_SLOPE_WINDOW + j] = 50_100.0 + j * 10
    ema_fast_vals[idx] = 50_150.0

    df = make_df(n, ema_fast=ema_fast_vals)
    df = set_cross_persistent(df, idx, value=1, lookback=CROSS_PERSISTENCE)
    assert classify_regime(df, idx) == "trending_up"


def test_trending_up_fails_without_slope():
    """Flat ema_fast (no slope) → should NOT return trending_up."""
    n = 100
    idx = 80
    df = make_df(n, ema_fast=50_100.0)  # flat — slope == 0, not positive
    df = set_cross_persistent(df, idx, value=1, lookback=CROSS_PERSISTENCE)
    result = classify_regime(df, idx)
    assert result != "trending_up"


def test_trending_up_fails_without_cross_persistence():
    """Cross that flipped only 1 candle ago (not persistent) → not trending_up."""
    n = 100
    idx = 80
    # Rising ema_fast
    ema_fast_vals = list(range(49_900, 49_900 + n))
    df = make_df(n, ema_fast=ema_fast_vals)
    # Only current candle has cross=1; earlier candles have cross=0
    df["ema_cross"] = 0
    df.loc[idx, "ema_cross"] = 1
    result = classify_regime(df, idx)
    assert result != "trending_up"


def test_trending_down_all_conditions_met():
    """Mirror of trending_up — all down conditions met → trending_down."""
    n = 100
    idx = 80
    # Falling ema_fast over last 5 candles
    ema_fast_vals = [49_900.0] * n
    for j in range(EMA_SLOPE_WINDOW + 1):
        ema_fast_vals[idx - EMA_SLOPE_WINDOW + j] = 49_900.0 - j * 10
    ema_fast_vals[idx] = 49_900.0 - EMA_SLOPE_WINDOW * 10

    df = make_df(
        n,
        ema_fast=ema_fast_vals,
        ema_slow=50_100.0,
        close=49_800.0,   # below ema_slow
    )
    df = set_cross_persistent(df, idx, value=0, lookback=CROSS_PERSISTENCE)
    assert classify_regime(df, idx) == "trending_down"


def test_ranging_is_default():
    """Inconclusive signals → ranging."""
    n = 100
    idx = 80
    # Mix: cross=1 half the time, flat slope, close near ema_slow
    df = make_df(n, ema_fast=50_000.0, ema_slow=50_000.0, close=50_000.0)
    df["ema_cross"] = [i % 2 for i in range(n)]  # alternating
    assert classify_regime(df, idx) == "ranging"


# ── Confidence scores ─────────────────────────────────────────────────────────


def test_confidence_high_vol_scales():
    """Confidence for high_vol should be >= 0.5 and increase with ATR ratio."""
    n = 100
    atr_vals_1 = [100.0] * n
    atr_vals_2 = [100.0] * n
    idx = 80
    atr_vals_1[idx] = 160.0   # 1.6× mean — just above threshold
    atr_vals_2[idx] = 250.0   # 2.5× mean — well above threshold

    df1 = make_df(n, atr=atr_vals_1)
    df2 = make_df(n, atr=atr_vals_2)

    c1 = get_regime_confidence(df1, idx)
    c2 = get_regime_confidence(df2, idx)

    assert c1 >= 0.5
    assert c2 >= 0.5
    assert c2 > c1


def test_confidence_bounded():
    """Confidence must always be in [0.0, 1.0]."""
    n = 100
    atr_vals = [100.0] * n
    atr_vals[80] = 10_000.0  # extreme spike
    df = make_df(n, atr=atr_vals)
    c = get_regime_confidence(df, 80)
    assert 0.0 <= c <= 1.0


# ── get_regime_summary ────────────────────────────────────────────────────────


def test_summary_structure():
    """get_regime_summary must return all required keys with correct types."""
    df = make_df(100)
    summary = get_regime_summary(df)

    assert "labels" in summary
    assert "counts" in summary
    assert "transitions" in summary
    assert "dominant" in summary
    assert len(summary["labels"]) == 100
    assert isinstance(summary["counts"], dict)
    assert isinstance(summary["transitions"], list)
    assert isinstance(summary["dominant"], str)


def test_summary_counts_match_labels():
    """counts dict must sum to total number of candles."""
    df = make_df(100)
    summary = get_regime_summary(df)
    assert sum(summary["counts"].values()) == 100


def test_summary_dominant_is_most_common():
    """dominant must match the regime with the highest count."""
    df = make_df(100)
    summary = get_regime_summary(df)
    expected = max(summary["counts"], key=lambda k: summary["counts"][k])
    assert summary["dominant"] == expected
