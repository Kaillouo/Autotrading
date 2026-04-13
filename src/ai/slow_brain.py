"""
Slow Brain — reviews trade history, suggests regime weight adjustments.

Usage:
    python -m src.ai.slow_brain --mode 6h
    python -m src.ai.slow_brain --mode weekly
"""
import argparse
import copy
import json
import logging
import os
from datetime import datetime, timezone, timedelta

import pandas as pd

from src.db.database import get_connection
from src.ai.utils.ai_client import call_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
WEIGHTS_PATH = os.path.join(_BASE_DIR, "config", "regime_weights.json")
WEIGHT_HISTORY_PATH = os.path.join(_BASE_DIR, "data", "weight_history.json")
SYNCED_BRIDGE_PATH = os.path.join(
    r"C:\Users\chunk\OneDrive\Desktop\OBI Second brain",
    "synced-bridge", "context", "trading-weekly.md"
)

SIGNAL_COLS = ["technical_score", "funding_rate_score", "oi_delta_score", "ema_cross_score"]
WEIGHT_FLOOR = 0.05
WEIGHT_CEIL = 0.60


def load_regime_weights() -> dict:
    with open(WEIGHTS_PATH) as f:
        return json.load(f)


def save_regime_weights(weights: dict) -> None:
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(weights, f, indent=2)


def load_recent_trades(hours: int = 6) -> pd.DataFrame:
    """Load completed trades from the last `hours` from the trades table."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM trades WHERE timestamp >= ? AND exit_reason IS NOT NULL",
        conn,
        params=(cutoff,),
    )
    conn.close()
    return df


def has_enough_data(trades_df: pd.DataFrame, min_trades_per_regime: int = 5) -> bool:
    """Return True only if at least one regime has >= min_trades_per_regime completed trades."""
    if trades_df.empty:
        return False
    return (trades_df.groupby("regime").size() >= min_trades_per_regime).any()


def build_analysis_prompt(trades_df: pd.DataFrame, current_weights: dict, hours: int) -> str:
    """Build the LLM prompt with per-regime trade summaries and current weights."""
    summary_lines = []
    for regime, grp in trades_df.groupby("regime"):
        count = len(grp)
        if count < 5:
            continue
        wins = grp[grp["pnl_pct"] > 0]
        losses = grp[grp["pnl_pct"] <= 0]
        win_rate = len(wins) / count * 100
        avg_pnl = grp["pnl_pct"].mean() * 100

        win_scores = {
            col: round(float(wins[col].mean()), 4) if len(wins) > 0 and col in grp.columns and wins[col].notna().any() else None
            for col in SIGNAL_COLS
        }
        loss_scores = {
            col: round(float(losses[col].mean()), 4) if len(losses) > 0 and col in grp.columns and losses[col].notna().any() else None
            for col in SIGNAL_COLS
        }

        summary_lines.append(
            f"Regime: {regime} | trades={count} | win_rate={win_rate:.1f}% | avg_pnl={avg_pnl:+.2f}%\n"
            f"  Winning trade avg scores:  {win_scores}\n"
            f"  Losing trade avg scores:   {loss_scores}"
        )

    per_regime_summary = "\n\n".join(summary_lines) if summary_lines else "No regimes with 5+ trades."

    return (
        f"You are reviewing the performance of a BTC/USDT trading bot.\n"
        f"Below is a summary of recent trades grouped by market regime.\n"
        f"The bot uses 4 signals weighted per regime. Your job is to suggest small weight adjustments.\n\n"
        f"Current regime weights:\n{json.dumps(current_weights, indent=2)}\n\n"
        f"Trade performance summary (last {hours}h):\n{per_regime_summary}\n\n"
        f"For each regime with enough data (5+ trades), suggest weight adjustments as a JSON object.\n"
        f"Adjustments must be small floats (e.g. +0.02, -0.03). Only include regimes you want to change.\n"
        f"Respond ONLY with valid JSON, no explanation:\n"
        f'{{\n  "regime_name": {{"signal_name": adjustment_float, ...}},\n  ...\n}}\n\n'
        f"Rules:\n"
        f"- If a signal scores high on winning trades but low on losing trades, increase its weight\n"
        f"- If a signal shows no difference between wins and losses, decrease its weight slightly\n"
        f"- trending_up regime: ema_cross_score is almost always 0.7 (not discriminating) — consider reducing it\n"
        f"- Don't suggest changes larger than \u00b10.05 per signal\n"
        f"- If data is insufficient or no clear pattern, return {{}}"
    )


def parse_weight_adjustments(llm_response: str) -> dict:
    """Extract JSON weight adjustments from LLM response. Returns empty dict on parse failure."""
    try:
        text = llm_response.strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        logger.warning(f"Failed to parse weight adjustments: {e}\nResponse: {llm_response[:200]}")
        return {}


def apply_weight_adjustments(current_weights: dict, adjustments: dict, max_delta: float) -> dict:
    """
    Apply adjustments with guardrails:
    1. Clamp each change to [-max_delta, +max_delta]
    2. After applying, renormalize each regime's weights to sum to 1.0
    3. Clamp each weight to [WEIGHT_FLOOR, WEIGHT_CEIL]
    4. Only apply adjustments for regimes present in current_weights
    """
    new_weights = copy.deepcopy(current_weights)

    for regime, changes in adjustments.items():
        if regime not in new_weights:
            logger.warning(f"Slow Brain suggested unknown regime '{regime}' — skipping")
            continue
        for signal, delta in changes.items():
            if signal not in new_weights[regime]:
                logger.warning(f"Slow Brain suggested unknown signal '{signal}' in regime '{regime}' — skipping")
                continue
            clamped = max(-max_delta, min(max_delta, float(delta)))
            new_weights[regime][signal] = new_weights[regime][signal] + clamped

        # Clamp each weight to [WEIGHT_FLOOR, WEIGHT_CEIL]
        for signal in new_weights[regime]:
            new_weights[regime][signal] = max(WEIGHT_FLOOR, min(WEIGHT_CEIL, new_weights[regime][signal]))

        # Renormalize to sum to 1.0
        total = sum(new_weights[regime].values())
        if total > 0:
            for signal in new_weights[regime]:
                new_weights[regime][signal] = round(new_weights[regime][signal] / total, 6)

    return new_weights


def log_weight_changes(old_weights: dict, new_weights: dict) -> None:
    """Append a timestamped before/after record to data/weight_history.json."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "before": old_weights,
        "after": new_weights,
    }
    history = []
    if os.path.exists(WEIGHT_HISTORY_PATH):
        with open(WEIGHT_HISTORY_PATH) as f:
            try:
                history = json.load(f)
            except Exception:
                history = []
    history.append(record)
    with open(WEIGHT_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def write_synced_bridge_summary(trades_df: pd.DataFrame, adjustments: dict, new_weights: dict) -> None:
    """Write weekly performance summary to synced-bridge for morning briefing."""
    count = len(trades_df)
    win_rate = float((trades_df["pnl_pct"] > 0).mean() * 100) if count > 0 else 0.0
    total_pnl = float(trades_df["pnl_pct"].sum() * 100) if count > 0 else 0.0

    regime_pnl = trades_df.groupby("regime")["pnl_pct"].mean()
    best = regime_pnl.idxmax() if not regime_pnl.empty else "n/a"
    worst = regime_pnl.idxmin() if not regime_pnl.empty else "n/a"

    regime_stats = []
    for regime, grp in trades_df.groupby("regime"):
        wr = float((grp["pnl_pct"] > 0).mean() * 100)
        avg = float(grp["pnl_pct"].mean() * 100)
        regime_stats.append(f"| {regime} | {len(grp)} | {wr:.1f}% | {avg:+.2f}% |")

    changes_md = []
    for regime, changes in adjustments.items():
        for signal, delta in changes.items():
            changes_md.append(f"- {regime}.{signal}: {float(delta):+.4f}")

    content = (
        f"# Trading Bot — Weekly Summary\n"
        f"**Updated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"**Period:** last 7 days\n\n"
        f"## Performance\n"
        f"- Trades: {count} | Win rate: {win_rate:.1f}% | Total PnL: {total_pnl:+.2f}%\n"
        f"- Best regime: {best} | Worst regime: {worst}\n\n"
        f"## Weight Adjustments Applied\n"
        + ("\n".join(changes_md) if changes_md else "- None")
        + "\n\n"
        f"## Regime Breakdown\n"
        f"| Regime | Trades | Win Rate | Avg PnL |\n"
        f"|---|---|---|---|\n"
        + "\n".join(regime_stats)
        + "\n"
    )

    os.makedirs(os.path.dirname(SYNCED_BRIDGE_PATH), exist_ok=True)
    with open(SYNCED_BRIDGE_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Weekly summary written to synced-bridge/context/trading-weekly.md")


def run_slow_brain(mode: str = "6h") -> None:
    hours = 6 if mode == "6h" else 168
    model = "claude-sonnet-4-6" if mode == "6h" else "claude-opus-4-6"
    max_delta = 0.03 if mode == "6h" else 0.05

    trades = load_recent_trades(hours)
    if not has_enough_data(trades):
        logger.info(
            f"Slow Brain ({mode}): insufficient trade data ({len(trades)} trades). Skipping."
        )
        return

    current_weights = load_regime_weights()
    prompt = build_analysis_prompt(trades, current_weights, hours)
    response = call_model(model, prompt)
    adjustments = parse_weight_adjustments(response)

    if not adjustments:
        logger.info("Slow Brain: no adjustments suggested.")
        return

    new_weights = apply_weight_adjustments(current_weights, adjustments, max_delta)
    log_weight_changes(current_weights, new_weights)
    save_regime_weights(new_weights)
    logger.info("Slow Brain: weights updated. Changes logged to data/weight_history.json")

    if mode == "weekly":
        write_synced_bridge_summary(trades, adjustments, new_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slow Brain — regime weight optimizer")
    parser.add_argument("--mode", choices=["6h", "weekly"], default="6h",
                        help="6h = incremental review (Sonnet, ±0.03); weekly = full review (Opus, ±0.05)")
    args = parser.parse_args()
    run_slow_brain(args.mode)
