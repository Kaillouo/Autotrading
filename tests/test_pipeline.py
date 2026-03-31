"""Tests for the trading data pipeline."""

import os
import sys
import pytest
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.bybit_client import get_exchange, fetch_ohlcv
from src.signals.technical import compute_indicators
from src.db.database import init_db, insert_candles, query_recent, DB_PATH


class TestBybitConnection:
    def test_exchange_connects_to_testnet(self):
        exchange = get_exchange()
        assert exchange.urls["api"]["public"].find("testnet") != -1 or "test" in str(exchange.urls).lower()
        # Verify we can load markets
        exchange.load_markets()
        assert "BTC/USDT" in exchange.symbols

    def test_fetch_ohlcv_returns_dataframe(self):
        df = fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=50)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert list(df.columns[:6]) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert df["close"].dtype in ["float64", "int64"]


class TestIndicators:
    def test_indicators_computed(self):
        df = fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=50)
        df = compute_indicators(df)
        assert "rsi" in df.columns
        assert "macd" in df.columns
        assert "bb_mid" in df.columns

    def test_no_nan_after_warmup(self):
        df = fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=100)
        df = compute_indicators(df)
        # After 30 rows of warmup, all indicators should be populated
        tail = df.iloc[35:]
        assert tail["rsi"].notna().all(), "RSI has NaN after warmup"
        assert tail["macd"].notna().all(), "MACD has NaN after warmup"
        assert tail["bb_mid"].notna().all(), "BB has NaN after warmup"


class TestDatabase:
    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path, monkeypatch):
        """Use a temp DB for tests."""
        test_db = str(tmp_path / "test.db")
        monkeypatch.setattr("src.db.database.DB_PATH", test_db)
        init_db()

    def test_write_and_read_roundtrip(self):
        df = fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=50)
        df = compute_indicators(df)
        insert_candles(df)

        result = query_recent(symbol="BTC/USDT", limit=5)
        assert len(result) == 5
        assert "close" in result.columns
        assert "rsi" in result.columns

    def test_no_duplicate_inserts(self):
        df = fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=30)
        df = compute_indicators(df)
        insert_candles(df)
        insert_candles(df)  # Insert same data again

        result = query_recent(symbol="BTC/USDT", limit=40)
        assert len(result) == 30  # Should not duplicate
