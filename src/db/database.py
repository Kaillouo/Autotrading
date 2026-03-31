import sqlite3
import os
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "trading.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
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
