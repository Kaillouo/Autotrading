"""
Session 3 Backtester — Pure Python signal scoring on historical DB data.
No LLM calls, no live exchange. Reads from trading.db only.
"""

import json
import math
import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db.database import get_connection
from src.signals.technical import compute_indicators
from src.signals.regime_detector import classify_regime

# ── Load regime weights ──────────────────────────────────────────────────────

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "regime_weights.json")
with open(WEIGHTS_PATH) as f:
    REGIME_WEIGHTS = json.load(f)

# ── Load risk config ─────────────────────────────────────────────────────────

RISK_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "risk.json")
with open(RISK_PATH) as f:
    RISK_CONFIG = json.load(f)

STOP_ATR_MULT = RISK_CONFIG["stop_loss_atr_multiple"]
TP_ATR_MULT = RISK_CONFIG["take_profit_atr_multiple"]
COMMISSION_PCT = RISK_CONFIG["commission_pct"]

# ── Load strategy config ──────────────────────────────────────────────────────

STRATEGY_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "strategy.json")
with open(STRATEGY_PATH) as f:
    STRATEGY_CONFIG = json.load(f)

_BT = STRATEGY_CONFIG["backtest"]
ENTRY_THRESHOLD = _BT["entry_threshold"]
EXIT_THRESHOLD  = _BT["exit_threshold"]
WARMUP          = _BT["warmup_candles"]
POSITION_SIZE   = _BT["position_size_pct"]
INITIAL_CAPITAL = _BT["initial_capital"]

# ── Signal Scoring Functions ─────────────────────────────────────────────────


def score_technical(row) -> float:
    """RSI + MACD + BB composite -> 0.0 to 1.0"""
    rsi = row.get("rsi", 50)
    rsi_score = max(0, min(1, (70 - rsi) / 40)) if rsi else 0.5

    macd_score = 0.7 if (row.get("macd_hist") or 0) > 0 else 0.3

    close = row.get("close", 0)
    bb_lower = row.get("bb_lower")
    bb_upper = row.get("bb_upper")
    if bb_lower and bb_upper and bb_upper != bb_lower:
        bb_score = max(0, min(1, 1 - (close - bb_lower) / (bb_upper - bb_lower)))
    else:
        bb_score = 0.5

    return 0.4 * rsi_score + 0.3 * macd_score + 0.3 * bb_score


def score_funding_rate(row) -> float:
    """Funding rate z-score -> 0.0 to 1.0 (negative z = bullish)"""
    z = row.get("funding_rate_zscore", 0) or 0
    return max(0.05, min(0.95, 0.5 - z / 3.0))


def score_oi_delta(df, idx) -> float:
    """OI change direction vs price change -> 0.0 to 1.0"""
    if idx < 5:
        return 0.5
    oi_now = df.iloc[idx].get("open_interest")
    oi_prev = df.iloc[idx - 5].get("open_interest")
    price_now = df.iloc[idx]["close"]
    price_prev = df.iloc[idx - 5]["close"]
    if not oi_now or not oi_prev:
        return 0.5
    oi_up = oi_now > oi_prev
    price_up = price_now > price_prev
    if oi_up and price_up:
        return 0.7
    elif oi_up and not price_up:
        return 0.3
    elif not oi_up and price_up:
        return 0.4
    else:
        return 0.5


def score_ema_cross(row) -> float:
    """EMA cross: 1 (fast > slow) -> bullish"""
    return 0.7 if row.get("ema_cross", 0) == 1 else 0.3


# ── Main Backtest ────────────────────────────────────────────────────────────


def load_data() -> pd.DataFrame:
    """Load candles + derivatives from DB, merge on timestamp."""
    conn = get_connection()

    candles = pd.read_sql_query(
        "SELECT * FROM candles WHERE symbol = 'BTC/USDT' ORDER BY timestamp ASC",
        conn,
    )
    derivatives = pd.read_sql_query(
        "SELECT timestamp, funding_rate, funding_rate_zscore, open_interest "
        "FROM derivatives_snapshots WHERE symbol LIKE 'BTC%' ORDER BY timestamp ASC",
        conn,
    )
    conn.close()

    if candles.empty:
        print("ERROR: No candle data in DB. Run: python src/data/historical_backfill.py")
        sys.exit(1)

    # Merge derivatives onto candles by nearest timestamp
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    if not derivatives.empty:
        derivatives["timestamp"] = pd.to_datetime(derivatives["timestamp"], utc=True)
        candles = pd.merge_asof(
            candles.sort_values("timestamp"),
            derivatives.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("4h"),
        )

    return candles


def run_backtest():
    print("Loading data from DB...")
    df = load_data()
    print(f"  Loaded {len(df)} candles")

    # Recompute indicators to ensure ATR/EMA are populated
    print("Computing indicators...")
    df = compute_indicators(df)

    # Convert to numeric where needed
    for col in ("atr", "ema_fast", "ema_slow", "ema_cross", "rsi", "macd_hist",
                "bb_lower", "bb_upper", "funding_rate_zscore", "open_interest"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    capital = INITIAL_CAPITAL
    position = None  # dict with entry_price, entry_idx, stop, tp, size_usd
    trades = []
    equity_curve = []
    regime_counts = {}

    print(f"Running backtest on {len(df) - WARMUP} candles (warmup={WARMUP})...")

    for i in range(WARMUP, len(df)):
        row = df.iloc[i]
        close = row["close"]

        # Track equity
        unrealized = 0
        if position:
            unrealized = (close - position["entry_price"]) / position["entry_price"] * position["size_usd"]
        equity_curve.append(capital + unrealized)

        # Check exit conditions if in position
        if position:
            low = row["low"]
            high = row["high"]

            # Stop loss hit
            if low <= position["stop"]:
                exit_price = position["stop"]
                pnl = (exit_price - position["entry_price"]) / position["entry_price"] * position["size_usd"]
                commission = position["size_usd"] * COMMISSION_PCT * 2
                pnl -= commission
                capital += pnl
                trades.append({
                    "entry_idx": position["entry_idx"],
                    "exit_idx": i,
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_pct": (exit_price - position["entry_price"]) / position["entry_price"],
                    "exit_reason": "stop",
                    "regime": position["regime"],
                    "commission": commission,
                })
                position = None
                continue

            # Take profit hit
            if high >= position["tp"]:
                exit_price = position["tp"]
                pnl = (exit_price - position["entry_price"]) / position["entry_price"] * position["size_usd"]
                commission = position["size_usd"] * COMMISSION_PCT * 2
                pnl -= commission
                capital += pnl
                trades.append({
                    "entry_idx": position["entry_idx"],
                    "exit_idx": i,
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_pct": (exit_price - position["entry_price"]) / position["entry_price"],
                    "exit_reason": "tp",
                    "regime": position["regime"],
                    "commission": commission,
                })
                position = None
                continue

        # Classify regime
        regime = classify_regime(df, i)
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # Score signals
        weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["ranging"])
        tech = score_technical(row)
        funding = score_funding_rate(row)
        oi = score_oi_delta(df, i)
        ema = score_ema_cross(row)

        composite = (
            weights["technical_score"] * tech
            + weights["funding_rate_score"] * funding
            + weights["oi_delta_score"] * oi
            + weights["ema_cross_score"] * ema
        )

        # Sell signal exits position
        if position and composite < EXIT_THRESHOLD:
            exit_price = close
            pnl = (exit_price - position["entry_price"]) / position["entry_price"] * position["size_usd"]
            commission = position["size_usd"] * COMMISSION_PCT * 2
            pnl -= commission
            capital += pnl
            trades.append({
                "entry_idx": position["entry_idx"],
                "exit_idx": i,
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": (exit_price - position["entry_price"]) / position["entry_price"],
                "exit_reason": "sell_signal",
                "regime": position["regime"],
                "commission": commission,
            })
            position = None

        # Buy signal opens position
        if not position and composite > ENTRY_THRESHOLD and i + 1 < len(df):
            entry_price = df.iloc[i + 1]["open"]  # enter at next candle's open
            atr = row.get("atr")
            if not atr or pd.isna(atr) or atr <= 0:
                continue  # can't set stops without ATR

            stop = entry_price - STOP_ATR_MULT * atr
            tp = entry_price + TP_ATR_MULT * atr
            size_usd = capital * POSITION_SIZE

            position = {
                "entry_price": entry_price,
                "entry_idx": i + 1,
                "stop": stop,
                "tp": tp,
                "size_usd": size_usd,
                "regime": regime,
            }

    # Close any open position at last close
    if position:
        exit_price = df.iloc[-1]["close"]
        pnl = (exit_price - position["entry_price"]) / position["entry_price"] * position["size_usd"]
        commission = position["size_usd"] * COMMISSION_PCT * 2
        pnl -= commission
        capital += pnl
        trades.append({
            "entry_idx": position["entry_idx"],
            "exit_idx": len(df) - 1,
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": (exit_price - position["entry_price"]) / position["entry_price"],
            "exit_reason": "end_of_data",
            "regime": position["regime"],
            "commission": commission,
        })
        position = None

    # ── Compute Stats ────────────────────────────────────────────────────────

    total_return_pct = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    trade_count = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    win_rate = len(wins) / trade_count * 100 if trade_count else 0

    # Max drawdown from equity curve
    max_dd = 0
    peak = equity_curve[0] if equity_curve else INITIAL_CAPITAL
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    # Annualized Sharpe (hourly returns -> annualized)
    if len(equity_curve) > 1:
        eq_series = pd.Series(equity_curve)
        returns = eq_series.pct_change().dropna()
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * math.sqrt(8760)  # ~8760 hours/year
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Regime breakdown
    regime_trades = {}
    for t in trades:
        r = t["regime"]
        if r not in regime_trades:
            regime_trades[r] = {"count": 0, "wins": 0, "total_pnl": 0}
        regime_trades[r]["count"] += 1
        regime_trades[r]["total_pnl"] += t["pnl"]
        if t["pnl"] > 0:
            regime_trades[r]["wins"] += 1

    # ── Print Results ────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Period:          {df.iloc[WARMUP]['timestamp']} -> {df.iloc[-1]['timestamp']}")
    print(f"  Candles:         {len(df)} ({len(df) - WARMUP} after warmup)")
    print(f"  Initial capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final capital:   ${capital:,.2f}")
    print(f"  Total return:    {total_return_pct:+.2f}%")
    print(f"  Annualized Sharpe: {sharpe:.3f}")
    print(f"  Win rate:        {win_rate:.1f}% ({len(wins)}/{trade_count})")
    print(f"  Max drawdown:    {max_dd * 100:.2f}%")
    print(f"  Trade count:     {trade_count}")

    print("\n  Regime distribution (candles):")
    for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        print(f"    {regime:16s} {count:5d}")

    print("\n  Regime trade breakdown:")
    for regime, stats in sorted(regime_trades.items(), key=lambda x: -x[1]["count"]):
        wr = stats["wins"] / stats["count"] * 100 if stats["count"] else 0
        print(f"    {regime:16s}  trades={stats['count']:3d}  wins={stats['wins']:3d}  "
              f"win_rate={wr:.0f}%  pnl=${stats['total_pnl']:+.2f}")

    print("\n  Exit reason breakdown:")
    exit_reasons = {}
    for t in trades:
        r = t["exit_reason"]
        exit_reasons[r] = exit_reasons.get(r, 0) + 1
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:16s} {count:3d}")

    # ── Save Results JSON ────────────────────────────────────────────────────

    results = {
        "date": str(date.today()),
        "period_start": str(df.iloc[WARMUP]["timestamp"]),
        "period_end": str(df.iloc[-1]["timestamp"]),
        "candle_count": len(df),
        "initial_capital": INITIAL_CAPITAL,
        "final_capital": round(capital, 2),
        "total_return_pct": round(total_return_pct, 4),
        "annualized_sharpe": round(sharpe, 4),
        "win_rate_pct": round(win_rate, 2),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "trade_count": trade_count,
        "commission_pct": COMMISSION_PCT,
        "regime_distribution": regime_counts,
        "regime_trade_breakdown": regime_trades,
        "exit_reasons": exit_reasons,
    }

    results_path = os.path.join(
        os.path.dirname(__file__),
        f"results_{date.today().isoformat()}.json",
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")

    # ── Exit Criteria ────────────────────────────────────────────────────────

    if sharpe > 0.8:
        print("\nEXIT CRITERIA MET -- Sharpe > 0.8 -- ready for S4.")
    else:
        print(f"\nSharpe {sharpe:.3f} < 0.8 -- not yet ready for S4.")
        print("  Weakest signals by regime — review regime_weights.json tuning.")

    return results


if __name__ == "__main__":
    run_backtest()
