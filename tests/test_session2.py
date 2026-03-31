"""
Session 2 tests.

Covers:
- Bybit derivatives fetcher (funding rate, OI, z-score)
- Polymarket client (mocked HTTP, empty list on failure)
- Historical backfill idempotency
- signal_contract: valid write/read roundtrip + rejection of bad signals
"""

import json
import os
import sys
import types
import unittest.mock as mock

import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.bybit_derivatives import fetch_funding_rate_history, fetch_open_interest_history
from src.data.polymarket_client import fetch_markets
from src.data.historical_backfill import backfill_historical
from src.db.database import init_db, query_recent, DB_PATH, get_funding_rate_history
from config.signal_contract import (
    write_signal,
    read_signal,
    is_signal_expired,
    SignalValidationError,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _minimal_signal(**overrides) -> dict:
    base = {
        "timestamp": "2026-03-31T14:23:00Z",
        "asset": "BTC/USDT",
        "direction": "buy",
        "confidence": 0.72,
        "regime": "trending_up",
        "signals": {"technical_score": 0.65, "funding_rate_score": 0.80},
        "reasoning": "Test signal",
        "max_position_pct": 0.08,
        "suggested_stop_pct": 0.025,
        "suggested_tp_pct": 0.055,
        "ttl_minutes": 15,
    }
    base.update(overrides)
    return base


# ── Bybit derivatives ─────────────────────────────────────────────────────────

class TestBybitDerivatives:
    def test_funding_rate_returns_dataframe(self):
        df = fetch_funding_rate_history(limit=100)
        assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
        assert "funding_rate" in df.columns, "Missing funding_rate column"
        assert "funding_rate_zscore" in df.columns, "Missing funding_rate_zscore column"
        assert "timestamp" in df.columns, "Missing timestamp column"

    def test_funding_rate_non_empty(self):
        df = fetch_funding_rate_history(limit=100)
        assert len(df) > 0, "Expected non-empty funding rate data"

    def test_funding_rate_zscore_is_float_column(self):
        df = fetch_funding_rate_history(limit=100)
        if df.empty:
            pytest.skip("No funding rate data available from testnet")
        zscore_col = df["funding_rate_zscore"].dropna()
        assert len(zscore_col) > 0, "All z-score values are NaN — warmup period too long?"
        assert zscore_col.dtype in ("float64", "float32"), "z-score must be a float column"

    def test_funding_rate_zscore_not_all_nan(self):
        df = fetch_funding_rate_history(limit=100)
        if df.empty:
            pytest.skip("No funding rate data available from testnet")
        # After warmup (48 rows), there should be valid z-scores
        non_nan_count = df["funding_rate_zscore"].notna().sum()
        assert non_nan_count > 0, (
            f"All {len(df)} z-score rows are NaN. "
            "Need at least 2 rows of funding rate history for rolling z-score."
        )

    def test_open_interest_returns_dataframe(self):
        df = fetch_open_interest_history(limit=100)
        assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
        assert "open_interest" in df.columns, "Missing open_interest column"
        assert "timestamp" in df.columns, "Missing timestamp column"


# ── Polymarket client ─────────────────────────────────────────────────────────

class TestPolymarketClient:
    def test_returns_list_on_success(self):
        """Mock the HTTP call — we can't rely on Polymarket being reachable in tests."""
        sample_response = {
            "data": [
                {
                    "condition_id": "abc123",
                    "question": "Will BTC exceed $90K by April 5?",
                    "end_date_iso": "2026-04-05T23:59:00+00:00",
                    "volume": 50000,
                    "tokens": [
                        {"outcome": "YES", "price": 0.62},
                        {"outcome": "NO",  "price": 0.38},
                    ],
                }
            ],
            "next_cursor": None,
        }

        def _mock_urlopen(req, timeout=10):
            cm = mock.MagicMock()
            cm.__enter__ = lambda s: s
            cm.__exit__ = mock.MagicMock(return_value=False)
            cm.read.return_value = json.dumps(sample_response).encode("utf-8")
            return cm

        with mock.patch("urllib.request.urlopen", side_effect=_mock_urlopen):
            results = fetch_markets()

        assert isinstance(results, list), "Should always return a list"
        assert len(results) == 1
        assert results[0]["market_id"] == "abc123"
        assert results[0]["yes_price"] == pytest.approx(0.62)
        assert results[0]["implied_prob_yes"] == pytest.approx(0.62)

    def test_returns_empty_list_on_api_failure(self):
        """API down → empty list, never raises."""
        import urllib.error

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            results = fetch_markets()

        assert results == [], "Should return empty list on failure"
        assert isinstance(results, list)

    def test_returns_empty_list_on_timeout(self):
        import socket

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=socket.timeout("timed out"),
        ):
            results = fetch_markets()

        assert results == []


# ── Historical backfill idempotency ───────────────────────────────────────────

class TestHistoricalBackfill:
    @pytest.fixture(autouse=True)
    def _use_temp_db(self, tmp_path, monkeypatch):
        test_db = str(tmp_path / "backfill_test.db")
        monkeypatch.setattr("src.db.database.DB_PATH", test_db)
        # Also patch the reference inside historical_backfill (imported separately)
        import src.db.database as db_mod
        monkeypatch.setattr(db_mod, "DB_PATH", test_db)
        init_db()

    def test_backfill_inserts_candles(self):
        # Use 1 month, smaller range to keep test fast
        count = backfill_historical(months=1)
        assert count > 0, "Backfill should have fetched at least some candles"

        recent = query_recent(symbol="BTC/USDT", limit=500)
        assert len(recent) > 0, "Candles should be in DB after backfill"

    def test_backfill_is_idempotent(self):
        backfill_historical(months=1)
        count_after_first = len(query_recent(symbol="BTC/USDT", limit=5000))

        backfill_historical(months=1)
        count_after_second = len(query_recent(symbol="BTC/USDT", limit=5000))

        assert count_after_first == count_after_second, (
            f"Row count changed after second run: {count_after_first} → {count_after_second}. "
            "INSERT OR IGNORE should prevent duplicates."
        )


# ── Signal contract ───────────────────────────────────────────────────────────

class TestSignalContract:
    def test_valid_signal_writes_and_reads_back(self, tmp_path):
        path = str(tmp_path / "signal.json")
        sig = _minimal_signal()
        write_signal(sig, path=path)

        loaded = read_signal(path=path)
        assert loaded is not None
        assert loaded["direction"] == "buy"
        assert loaded["confidence"] == pytest.approx(0.72)
        assert loaded["asset"] == "BTC/USDT"
        assert loaded["ttl_minutes"] == 15

    def test_read_returns_none_when_no_file(self, tmp_path):
        result = read_signal(path=str(tmp_path / "nonexistent.json"))
        assert result is None

    def test_rejects_missing_field(self, tmp_path):
        sig = _minimal_signal()
        del sig["direction"]
        with pytest.raises(SignalValidationError, match="missing required fields"):
            write_signal(sig, path=str(tmp_path / "signal.json"))

    def test_rejects_invalid_direction(self, tmp_path):
        sig = _minimal_signal(direction="long")  # invalid
        with pytest.raises(SignalValidationError, match="direction"):
            write_signal(sig, path=str(tmp_path / "signal.json"))

    def test_rejects_confidence_above_one(self, tmp_path):
        sig = _minimal_signal(confidence=1.5)
        with pytest.raises(SignalValidationError, match="confidence"):
            write_signal(sig, path=str(tmp_path / "signal.json"))

    def test_rejects_confidence_below_zero(self, tmp_path):
        sig = _minimal_signal(confidence=-0.1)
        with pytest.raises(SignalValidationError, match="confidence"):
            write_signal(sig, path=str(tmp_path / "signal.json"))

    def test_all_valid_directions_accepted(self, tmp_path):
        for direction in ("buy", "sell", "hold"):
            sig = _minimal_signal(direction=direction)
            write_signal(sig, path=str(tmp_path / f"signal_{direction}.json"))

    def test_signal_not_expired_when_fresh(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        sig = _minimal_signal(timestamp=now, ttl_minutes=15)
        assert is_signal_expired(sig) is False

    def test_signal_expired_when_old(self):
        sig = _minimal_signal(timestamp="2020-01-01T00:00:00Z", ttl_minutes=15)
        assert is_signal_expired(sig) is True

    def test_signal_expired_on_malformed_timestamp(self):
        sig = _minimal_signal(timestamp="not-a-date")
        assert is_signal_expired(sig) is True
