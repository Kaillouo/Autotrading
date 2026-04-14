"""Tests for src/notifications/telegram.py"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_ok_response():
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.status_code = 200
    return mock_resp


def _make_fail_response():
    mock_resp = MagicMock()
    mock_resp.ok = False
    mock_resp.status_code = 400
    mock_resp.text = "Bad Request"
    return mock_resp


def _patched_config(tmp_path) -> dict:
    """Write a temp telegram.json and return its path."""
    cfg = {"bot_token": "test_token", "chat_id": "123456789"}
    p = tmp_path / "telegram.json"
    p.write_text(json.dumps(cfg))
    return cfg, p


# ── send_message ───────────────────────────────────────────────────────────────

class TestSendMessage:
    def test_success(self, tmp_path, monkeypatch):
        cfg, cfg_path = _patched_config(tmp_path)
        monkeypatch.setattr("src.notifications.telegram._BOT_TOKEN", cfg["bot_token"])
        monkeypatch.setattr("src.notifications.telegram._CHAT_ID", cfg["chat_id"])

        with patch("requests.post", return_value=_make_ok_response()) as mock_post:
            from src.notifications.telegram import send_message
            result = send_message("hello")

        assert result is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "sendMessage" in call_kwargs[0][0]
        payload = call_kwargs[1]["json"]
        assert payload["text"] == "hello"
        assert payload["chat_id"] == cfg["chat_id"]

    def test_network_error_returns_false(self, monkeypatch):
        monkeypatch.setattr("src.notifications.telegram._BOT_TOKEN", "tok")
        monkeypatch.setattr("src.notifications.telegram._CHAT_ID", "123")

        with patch("requests.post", side_effect=ConnectionError("timeout")):
            from src.notifications.telegram import send_message
            result = send_message("hello")

        assert result is False  # must not raise

    def test_api_error_returns_false(self, monkeypatch):
        monkeypatch.setattr("src.notifications.telegram._BOT_TOKEN", "tok")
        monkeypatch.setattr("src.notifications.telegram._CHAT_ID", "123")

        with patch("requests.post", return_value=_make_fail_response()):
            from src.notifications.telegram import send_message
            result = send_message("hello")

        assert result is False

    def test_no_config_returns_false(self, monkeypatch):
        monkeypatch.setattr("src.notifications.telegram._BOT_TOKEN", None)
        monkeypatch.setattr("src.notifications.telegram._CHAT_ID", None)

        with patch("requests.post") as mock_post:
            from src.notifications.telegram import send_message
            result = send_message("hello")

        assert result is False
        mock_post.assert_not_called()


# ── send_trade_alert ───────────────────────────────────────────────────────────

class TestSendTradeAlert:
    def _captured_text(self, monkeypatch, **kwargs) -> str:
        monkeypatch.setattr("src.notifications.telegram._BOT_TOKEN", "tok")
        monkeypatch.setattr("src.notifications.telegram._CHAT_ID", "123")
        captured = {}
        with patch("requests.post") as mock_post:
            mock_post.return_value = _make_ok_response()
            from src.notifications.telegram import send_trade_alert
            send_trade_alert(**kwargs)
            if mock_post.called:
                captured["text"] = mock_post.call_args[1]["json"]["text"]
        return captured.get("text", "")

    def test_open_alert_content(self, monkeypatch):
        text = self._captured_text(
            monkeypatch,
            action="open",
            symbol="BTC/USDT",
            price=84230.0,
            regime="ranging",
            confidence=0.71,
            stop_price=82814.0,
            tp_price=87462.0,
            equity=10340.0,
        )
        assert "LONG OPENED" in text
        assert "84,230" in text
        assert "82,814" in text
        assert "87,462" in text
        assert "ranging" in text
        assert "0.71" in text
        assert "10,340" in text

    def test_close_tp_hit(self, monkeypatch):
        text = self._captured_text(
            monkeypatch,
            action="close",
            symbol="BTC/USDT",
            price=87462.0,
            regime="ranging",
            confidence=0.71,
            pnl=126.40,
            exit_reason="tp",
            equity=10466.0,
        )
        assert "LONG CLOSED" in text
        assert "TP HIT" in text
        assert "126" in text
        assert "10,466" in text

    def test_close_stop_hit(self, monkeypatch):
        text = self._captured_text(
            monkeypatch,
            action="close",
            symbol="BTC/USDT",
            price=82500.0,
            regime="ranging",
            confidence=0.65,
            pnl=-85.20,
            exit_reason="stop",
            equity=9914.80,
        )
        assert "STOP HIT" in text
        assert "-$85.20" in text

    def test_close_signal(self, monkeypatch):
        text = self._captured_text(
            monkeypatch,
            action="close",
            symbol="BTC/USDT",
            price=84000.0,
            regime="trending_up",
            confidence=0.70,
            exit_reason="signal",
        )
        assert "SIGNAL" in text


# ── send_morning_report ────────────────────────────────────────────────────────

class TestSendMorningReport:
    def test_morning_report_content(self, monkeypatch):
        monkeypatch.setattr("src.notifications.telegram._BOT_TOKEN", "tok")
        monkeypatch.setattr("src.notifications.telegram._CHAT_ID", "123")

        with patch("requests.post") as mock_post:
            mock_post.return_value = _make_ok_response()
            from src.notifications.telegram import send_morning_report
            send_morning_report(
                equity=10466.0,
                peak_equity=10500.0,
                open_positions=[],
                regime="ranging",
                trades_today=3,
                pnl_today=94.20,
                win_rate_7d=0.42,
                wins_7d=8,
                total_7d=19,
                max_drawdown_pct=3.1,
                last_run="07:45",
                bot_running=True,
            )

        assert mock_post.called
        text = mock_post.call_args[1]["json"]["text"]
        assert "Morning Report" in text
        assert "10,466" in text
        assert "ranging" in text
        assert "3 trades" in text
        assert "07:45" in text


# ── send_drawdown_warning ──────────────────────────────────────────────────────

class TestSendDrawdownWarning:
    def _captured_text(self, monkeypatch, drawdown_pct: float) -> str:
        monkeypatch.setattr("src.notifications.telegram._BOT_TOKEN", "tok")
        monkeypatch.setattr("src.notifications.telegram._CHAT_ID", "123")
        with patch("requests.post") as mock_post:
            mock_post.return_value = _make_ok_response()
            from src.notifications.telegram import send_drawdown_warning
            send_drawdown_warning(8880.0, 10000.0, drawdown_pct)
            if mock_post.called:
                return mock_post.call_args[1]["json"]["text"]
        return ""

    def test_warning_at_11_percent(self, monkeypatch):
        text = self._captured_text(monkeypatch, 11.2)
        assert "DRAWDOWN WARNING" in text
        assert "11.2" in text
        assert "still running" in text.lower()

    def test_halt_at_21_percent(self, monkeypatch):
        text = self._captured_text(monkeypatch, 21.0)
        assert "HALTED" in text
        assert "21.0" in text
        assert "paused" in text.lower()

    def test_no_alert_below_threshold(self, monkeypatch):
        monkeypatch.setattr("src.notifications.telegram._BOT_TOKEN", "tok")
        monkeypatch.setattr("src.notifications.telegram._CHAT_ID", "123")
        with patch("requests.post") as mock_post:
            from src.notifications.telegram import send_drawdown_warning
            send_drawdown_warning(9600.0, 10000.0, 4.0)
        mock_post.assert_not_called()
