"""Offline integration tests for the market data recorder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from market_data.connection_health import ConnectionHealth, HeartbeatStats
from market_data.manifest import build_manifest, write_manifest
from market_data.recorder_config import RecorderConfig
from market_data.recorder import MarketDataRecorder
from market_data.status import read_status, write_status, format_status_summary


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ws_messages.json"


def _load_fixtures() -> dict:
    with FIXTURES.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class TestRecorderIntegration:
    def test_handle_message_trades(self, tmp_path):
        config = RecorderConfig(
            coins=("BTC",),
            output_dir=tmp_path / "market_stream",
            raw_only=True,
        )
        recorder = MarketDataRecorder(config)
        fixtures = _load_fixtures()
        msg = fixtures["trade"]
        recorder._handle_message("trades", msg, 1718000001000)
        recorder._raw_writer.shutdown()
        assert recorder._raw_writer.row_count == 1
        assert recorder._health.channel_message_counts.get("trades") == 1

    def test_handle_malformed_message(self, tmp_path):
        config = RecorderConfig(
            coins=("BTC",),
            output_dir=tmp_path / "market_stream",
            raw_only=True,
        )
        recorder = MarketDataRecorder(config)
        recorder._handle_message("__malformed__", "bad json", 1000)
        assert recorder._health.malformed_count == 1

    def test_handle_pong_heartbeat(self, tmp_path):
        config = RecorderConfig(
            coins=("BTC",),
            output_dir=tmp_path / "market_stream",
            raw_only=True,
        )
        recorder = MarketDataRecorder(config)
        recorder._health.heartbeat.record_ping(1000)
        recorder._handle_message("pong", None, 1050)
        assert recorder._health.heartbeat.pong_count == 1
        assert recorder._health.heartbeat.last_round_trip_ms == 50

    def test_handle_subscription_response(self, tmp_path):
        config = RecorderConfig(
            coins=("BTC",),
            output_dir=tmp_path / "market_stream",
            raw_only=True,
        )
        recorder = MarketDataRecorder(config)
        fixtures = _load_fixtures()
        recorder._handle_message(
            "subscriptionResponse",
            fixtures["subscription_response"]["data"],
            1000,
        )
        assert recorder._subscription_tracker.active_count >= 1

    def test_pong_not_stored_as_market_event(self, tmp_path):
        config = RecorderConfig(
            coins=("BTC",),
            output_dir=tmp_path / "market_stream",
            raw_only=True,
        )
        recorder = MarketDataRecorder(config)
        recorder._handle_message("pong", None, 1000)
        recorder._raw_writer.shutdown()
        assert recorder._raw_writer.row_count == 0

    def test_normalization_pipeline(self, tmp_path):
        config = RecorderConfig(
            coins=("BTC", "ETH", "SOL"),
            output_dir=tmp_path / "market_stream",
        )
        recorder = MarketDataRecorder(config)
        fixtures = _load_fixtures()
        for key in ("trade", "bbo", "l2_book", "candle", "active_asset_ctx", "all_mids"):
            msg = fixtures[key]
            recorder._handle_message(msg["channel"], msg, 1718000001000)
        recorder._raw_writer.shutdown()
        if recorder._normalized_writer:
            recorder._normalized_writer.shutdown()
        assert recorder._raw_writer.row_count >= 6
        if recorder._normalized_writer:
            assert recorder._normalized_writer.row_count >= 1

    def test_manifest_generation(self, tmp_path):
        manifests_dir = tmp_path / "manifests"
        manifest_data = build_manifest(
            recorder_session_id="test-id",
            config_dict={"network": "mainnet", "coins": ["BTC"]},
            start_time_ms=1000,
            end_time_ms=2000,
            raw_row_count=10,
            normalized_row_count=8,
            duplicate_count=2,
            malformed_count=0,
            reconnect_count=1,
            heartbeat_stats={},
            channel_stats={"trades": 5},
            latency_stats={"count": 5},
            gap_stats={"total_gaps": 0},
            subscription_summary={"by_state": {"ACTIVE": 3}},
            files_written=["raw/events.jsonl"],
            shutdown_reason="duration_limit",
            configuration_hash="abc123",
        )
        path = write_manifest(manifests_dir, "test-id", manifest_data)
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["paper_only"] is True
        assert loaded["execution_enabled"] is False
        assert loaded["recorder_session_id"] == "test-id"

    def test_configuration_hash_stable(self):
        config = RecorderConfig(coins=("BTC", "ETH", "SOL"))
        h1 = config.configuration_hash()
        h2 = config.configuration_hash()
        assert h1 == h2
        assert len(h1) == 64

    def test_status_read_write(self, tmp_path):
        status = {
            "running": True,
            "recorder_session_id": "abc",
            "uptime_seconds": 60.0,
            "connection_state": "CONNECTED",
            "reconnect_count": 0,
            "malformed_count": 0,
            "duplicate_count": 0,
            "gap_count": 0,
            "output_dir": str(tmp_path),
        }
        write_status(tmp_path, status)
        loaded = read_status(tmp_path)
        assert loaded["recorder_session_id"] == "abc"
        summary = format_status_summary(loaded)
        assert "abc" in summary

    def test_graceful_shutdown_writes_manifest(self, tmp_path):
        config = RecorderConfig(
            coins=("BTC",),
            output_dir=tmp_path / "market_stream",
            raw_only=True,
        )
        recorder = MarketDataRecorder(config)
        fixtures = _load_fixtures()
        recorder._handle_message("trades", fixtures["trade"], 1000)
        import asyncio

        async def shutdown():
            await recorder._shutdown()

        asyncio.run(shutdown())
        manifest_path = tmp_path / "market_stream" / "manifests" / f"{recorder.recorder_session_id}.json"
        assert manifest_path.exists()

    def test_exponential_backoff_boundaries(self):
        config = RecorderConfig(
            reconnect_initial_delay_seconds=1.0,
            reconnect_max_delay_seconds=10.0,
        )
        delay = config.reconnect_initial_delay_seconds
        for _ in range(20):
            delay = min(delay * 2, config.reconnect_max_delay_seconds)
        assert delay == 10.0

    def test_heartbeat_stats(self):
        hb = HeartbeatStats()
        hb.record_ping(1000)
        rtt = hb.record_pong(1050)
        assert rtt == 50
        assert hb.ping_count == 1
        assert hb.pong_count == 1

    def test_connection_health_summary(self):
        health = ConnectionHealth()
        health.record_message("trades")
        health.record_message("trades")
        summary = health.summary()
        assert summary["channel_message_counts"]["trades"] == 2

    def test_no_writes_to_live_data_files(self, tmp_path):
        live_paper = Path(__file__).resolve().parent.parent / "data" / "paper_trades.csv"
        before = live_paper.read_bytes() if live_paper.exists() else b""
        config = RecorderConfig(
            coins=("BTC",),
            output_dir=tmp_path / "market_stream",
            raw_only=True,
        )
        recorder = MarketDataRecorder(config)
        recorder._raw_writer.shutdown()
        after = live_paper.read_bytes() if live_paper.exists() else b""
        assert before == after

    def test_paper_bot_behavior_unchanged(self):
        import paper_trader
        from config import PAPER_TRADING

        trader = paper_trader.PaperTrader()
        assert trader.cash_balance == PAPER_TRADING.starting_balance_usdc
