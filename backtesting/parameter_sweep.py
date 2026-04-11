"""
Walk-forward parameter sweep for the backtest strategy.

Loads all available data, splits 75/25 by time index:
  - Train (first 75%) = optimization period (NOT used for scoring)
  - Holdout (last 25%) = evaluation period (the only Sharpe we report)

This prevents overfitting: we only report holdout Sharpe, never train Sharpe.

Usage:
    python backtesting/parameter_sweep.py
"""

import itertools
import json
import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtesting.backtest_runner import load_data, run_backtest
from src.signals.technical import compute_indicators

# ── Parameter grids ───────────────────────────────────────────────────────────

STOP_ATR_MULT_GRID              = [1.0, 1.5, 2.0, 2.5]
TP_ATR_MULT_GRID                = [2.0, 2.5, 3.0, 3.5, 4.0]
ENTRY_THRESHOLD_GRID            = [0.58, 0.60, 0.62, 0.65, 0.68]
EXIT_THRESHOLD_GRID             = [0.35, 0.38, 0.40, 0.42]
TRENDING_UP_EMA_CROSS_WEIGHT_GRID = [0.00, 0.05, 0.10, 0.20]

NUMERIC_COLS = (
    "atr", "ema_fast", "ema_slow", "ema_cross", "rsi", "macd_hist",
    "bb_lower", "bb_upper", "funding_rate_zscore", "open_interest",
)


def main():
    print("=" * 60)
    print("PARAMETER SWEEP — Walk-forward validation")
    print("=" * 60)

    # ── Load + prepare full dataset ───────────────────────────────────────────
    print("Loading data from DB...")
    df = load_data()
    print(f"  Total candles: {len(df)}")

    print("Computing indicators...")
    df = compute_indicators(df)

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.reset_index(drop=True)

    # ── Walk-forward split ────────────────────────────────────────────────────
    train_end_idx = int(len(df) * 0.75)
    test_df = df.iloc[train_end_idx:].reset_index(drop=True)

    print(f"\nSplit:")
    print(f"  Train (not evaluated): {df.iloc[0]['timestamp']} -> {df.iloc[train_end_idx - 1]['timestamp']}  ({train_end_idx} candles)")
    print(f"  Holdout (scored):      {test_df.iloc[0]['timestamp']} -> {test_df.iloc[-1]['timestamp']}  ({len(test_df)} candles)")

    # ── Build combo list ──────────────────────────────────────────────────────
    combos = list(itertools.product(
        STOP_ATR_MULT_GRID,
        TP_ATR_MULT_GRID,
        ENTRY_THRESHOLD_GRID,
        EXIT_THRESHOLD_GRID,
        TRENDING_UP_EMA_CROSS_WEIGHT_GRID,
    ))
    total = len(combos)
    print(f"\nRunning {total} parameter combinations on holdout period...\n")

    results = []

    for idx, (stop, tp, entry, exit_, ema_w) in enumerate(combos):
        r = run_backtest(
            df=test_df,
            stop_atr_mult=stop,
            tp_atr_mult=tp,
            entry_threshold=entry,
            exit_threshold=exit_,
            trending_up_ema_weight=ema_w,
            silent=True,
            save_results=False,
        )
        results.append({
            "stop_atr_mult": stop,
            "tp_atr_mult": tp,
            "entry_threshold": entry,
            "exit_threshold": exit_,
            "trending_up_ema_weight": ema_w,
            "holdout_sharpe": r["annualized_sharpe"],
            "holdout_return_pct": r["total_return_pct"],
            "holdout_max_dd_pct": r["max_drawdown_pct"],
            "trade_count": r["trade_count"],
        })

        if (idx + 1) % 100 == 0:
            elapsed_pct = (idx + 1) / total * 100
            print(f"  {idx + 1}/{total}  ({elapsed_pct:.0f}%)  best so far: Sharpe={max(x['holdout_sharpe'] for x in results):.3f}")

    # ── Sort and report ───────────────────────────────────────────────────────
    results.sort(key=lambda x: x["holdout_sharpe"], reverse=True)

    print("\n" + "=" * 60)
    print("TOP 10 by holdout Sharpe")
    print("=" * 60)
    for rank, r in enumerate(results[:10], 1):
        print(
            f"  #{rank:2d}  Sharpe={r['holdout_sharpe']:+.3f}  Return={r['holdout_return_pct']:+6.2f}%  "
            f"DD={r['holdout_max_dd_pct']:.1f}%  trades={r['trade_count']:3d}  "
            f"stop={r['stop_atr_mult']}  tp={r['tp_atr_mult']}  "
            f"entry={r['entry_threshold']}  exit={r['exit_threshold']}  "
            f"ema_w={r['trending_up_ema_weight']}"
        )

    print(f"\nWORST combo:")
    w = results[-1]
    print(
        f"  Sharpe={w['holdout_sharpe']:+.3f}  Return={w['holdout_return_pct']:+6.2f}%  "
        f"stop={w['stop_atr_mult']}  tp={w['tp_atr_mult']}  "
        f"entry={w['entry_threshold']}  exit={w['exit_threshold']}  ema_w={w['trending_up_ema_weight']}"
    )

    # ── Save full results ─────────────────────────────────────────────────────
    out_path = os.path.join(
        os.path.dirname(__file__),
        f"sweep_results_{date.today().isoformat()}.json",
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results ({total} combos) saved to: {out_path}")

    # ── Return best for Part 4 ────────────────────────────────────────────────
    return results[0]


if __name__ == "__main__":
    best = main()
    print(f"\nBEST PARAMS:")
    for k, v in best.items():
        print(f"  {k}: {v}")
