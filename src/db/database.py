import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "trading.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()

    # ── Session 1 ─────────────────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            macd_hist REAL,
            bb_lower REAL,
            bb_mid REAL,
            bb_upper REAL,
            UNIQUE(timestamp, symbol)
        )
    """)

    # ── Session 2: new tables ─────────────────────────────────────────────────

    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            asset TEXT,
            direction TEXT,
            entry_price REAL,
            exit_price REAL,
            quantity REAL,
            regime TEXT,
            signal_source TEXT,
            confidence REAL,
            pnl_pct REAL,
            exit_reason TEXT,
            notes TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_accuracy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            signal_type TEXT,
            regime TEXT,
            predicted_direction TEXT,
            actual_direction TEXT,
            confidence REAL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS regime_log (
            timestamp TEXT PRIMARY KEY,
            regime TEXT,
            btc_price REAL,
            funding_rate REAL,
            open_interest REAL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS polymarket_bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            market_id TEXT,
            question TEXT,
            our_prob REAL,
            market_prob REAL,
            edge REAL,
            stake REAL,
            outcome TEXT,
            pnl REAL,
            is_paper INTEGER DEFAULT 1
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created TEXT,
            pattern_type TEXT,
            description TEXT,
            confidence REAL,
            still_valid INTEGER DEFAULT 1
        )
    """)

    # Derivatives snapshot table — one row per funding/OI data point
    conn.execute("""
        CREATE TABLE IF NOT EXISTS derivatives_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            funding_rate REAL,
            funding_rate_zscore REAL,
            open_interest REAL,
            UNIQUE(timestamp, symbol)
        )
    """)

    conn.commit()
    conn.close()


def insert_candles(df):
    """Insert OHLCV + indicator data. Skips duplicates."""
    conn = get_connection()
    cols = [
        "timestamp", "symbol", "open", "high", "low", "close", "volume",
        "rsi", "macd", "macd_signal", "macd_hist", "bb_lower", "bb_mid", "bb_upper",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = None

    df_insert = df[cols].copy()
    df_insert["timestamp"] = df_insert["timestamp"].astype(str)

    df_insert.to_sql("candles", conn, if_exists="append", index=False, method=_insert_or_ignore)
    conn.close()


def _insert_or_ignore(table, conn, keys, data_iter):
    """Custom insert method that ignores duplicate rows."""
    from sqlite3 import OperationalError
    cols = ", ".join(keys)
    placeholders = ", ".join(["?"] * len(keys))
    sql = f"INSERT OR IGNORE INTO {table.name} ({cols}) VALUES ({placeholders})"
    data = list(data_iter)
    conn.execute(f"BEGIN")
    for row in data:
        conn.execute(sql, row)
    conn.execute("COMMIT")


def query_recent(symbol="BTC/USDT", limit=5):
    """Return the most recent candles with indicators."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM candles WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
        conn,
        params=(symbol, limit),
    )
    conn.close()
    return df


# ── Session 2 methods ─────────────────────────────────────────────────────────

def insert_derivatives_data(funding_df: pd.DataFrame, oi_df: pd.DataFrame, symbol: str = "BTC/USDT:USDT") -> None:
    """
    Merge funding rate and open interest DataFrames on timestamp and insert
    into derivatives_snapshots. Skips rows that already exist (idempotent).

    Both DataFrames must have a 'timestamp' column (datetime, UTC).
    """
    if funding_df.empty and oi_df.empty:
        return

    conn = get_connection()

    # Merge on nearest timestamp — funding and OI may have different cadences
    if not funding_df.empty and not oi_df.empty:
        merged = pd.merge_asof(
            funding_df.sort_values("timestamp"),
            oi_df.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("2h"),
        )
    elif not funding_df.empty:
        merged = funding_df.copy()
        merged["open_interest"] = None
    else:
        merged = oi_df.copy()
        merged["funding_rate"] = None
        merged["funding_rate_zscore"] = None

    for col in ("funding_rate", "funding_rate_zscore", "open_interest"):
        if col not in merged.columns:
            merged[col] = None

    merged["symbol"] = symbol
    merged["timestamp"] = merged["timestamp"].astype(str)

    for _, row in merged.iterrows():
        conn.execute(
            """
            INSERT OR IGNORE INTO derivatives_snapshots
                (timestamp, symbol, funding_rate, funding_rate_zscore, open_interest)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                row["timestamp"],
                row["symbol"],
                row.get("funding_rate"),
                row.get("funding_rate_zscore"),
                row.get("open_interest"),
            ),
        )

    conn.commit()
    conn.close()


def insert_regime_snapshot(
    timestamp: str,
    regime: str,
    btc_price: float,
    funding_rate: float | None = None,
    open_interest: float | None = None,
) -> None:
    """
    Upsert a single regime snapshot row.
    timestamp: ISO string (e.g. '2026-03-31T14:00:00+00:00')
    """
    conn = get_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO regime_log
            (timestamp, regime, btc_price, funding_rate, open_interest)
        VALUES (?, ?, ?, ?, ?)
        """,
        (timestamp, regime, btc_price, funding_rate, open_interest),
    )
    conn.commit()
    conn.close()


def insert_polymarket_snapshot(
    market_id: str,
    question: str,
    our_prob: float | None,
    market_prob: float,
    edge: float | None,
    stake: float = 0.0,
    outcome: str = "pending",
    pnl: float = 0.0,
    is_paper: bool = True,
    timestamp: str | None = None,
) -> None:
    """Insert a Polymarket bet/snapshot row. All values parameterized."""
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat()
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO polymarket_bets
            (timestamp, market_id, question, our_prob, market_prob,
             edge, stake, outcome, pnl, is_paper)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (timestamp, market_id, question, our_prob, market_prob,
         edge, stake, outcome, pnl, int(is_paper)),
    )
    conn.commit()
    conn.close()


def get_funding_rate_history(symbol: str = "BTC/USDT:USDT", limit: int = 200) -> pd.DataFrame:
    """Return the most recent `limit` funding rate rows for a symbol, oldest first."""
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT timestamp, funding_rate, funding_rate_zscore
        FROM derivatives_snapshots
        WHERE symbol = ? AND funding_rate IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        conn,
        params=(symbol, limit),
    )
    conn.close()
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df
