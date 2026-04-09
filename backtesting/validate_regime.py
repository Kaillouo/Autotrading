"""
Regime Validation — last 30 days of BTC/USDT candles.

Prints:
  - Day-by-day: dominant regime, BTC price range, ATR
  - Regime transition log (when label changed)
  - Overall distribution

Run from repo root:
    python backtesting/validate_regime.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db.database import get_connection
from src.signals.technical import compute_indicators
from src.signals.regime_detector import classify_regime, get_regime_summary


def load_last_30_days() -> pd.DataFrame:
    conn = get_connection()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    df = pd.read_sql_query(
        "SELECT * FROM candles WHERE symbol='BTC/USDT' AND timestamp >= ? ORDER BY timestamp ASC",
        conn,
        params=(cutoff,),
    )
    conn.close()
    return df


def main():
    print("Loading last 30 days of BTC/USDT candles...")
    df = load_last_30_days()

    if df.empty:
        print("ERROR: No data found. Run historical_backfill.py first.")
        sys.exit(1)

    print(f"  {len(df)} candles loaded")

    df = compute_indicators(df)
    for col in ("atr", "ema_fast", "ema_slow", "ema_cross", "rsi"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    summary = get_regime_summary(df)
    labels = summary["labels"]

    # ── Day-by-day breakdown ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'DATE':<12} {'DOMINANT':<16} {'PRICE LOW':>11} {'PRICE HIGH':>11} {'ATR':>9} {'CANDLES':>8}")
    print("-" * 72)

    df["date"] = df["timestamp"].dt.date
    df["_regime"] = labels

    for day, group in df.groupby("date"):
        dom_regime = group["_regime"].value_counts().idxmax()
        price_low = group["low"].min()
        price_high = group["high"].max()
        atr_mean = group["atr"].mean()
        n = len(group)
        print(
            f"{str(day):<12} {dom_regime:<16} "
            f"${price_low:>10,.0f} ${price_high:>10,.0f} "
            f"{atr_mean:>9.1f} {n:>8d}"
        )

    # ── Transition log ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("REGIME TRANSITIONS")
    print("-" * 72)
    transitions = summary["transitions"]
    if not transitions:
        print("  No transitions (all candles same regime)")
    else:
        print(f"  Total transitions: {len(transitions)}")
        for t in transitions:
            print(f"  {t['timestamp'][:19]}  {t['from']:16s} -> {t['to']}")

    # ── Overall distribution ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("REGIME DISTRIBUTION (candles)")
    print("-" * 72)
    total = sum(summary["counts"].values())
    for regime, count in sorted(summary["counts"].items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total else 0
        bar = "#" * int(pct / 2)
        print(f"  {regime:<16} {count:5d}  ({pct:5.1f}%)  {bar}")

    print(f"\n  Dominant overall: {summary['dominant']}")
    print(f"  Date range: {df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()}")


if __name__ == "__main__":
    main()
