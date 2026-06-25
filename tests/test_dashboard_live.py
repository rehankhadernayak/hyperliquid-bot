"""Tests for live market dashboard loaders and aggregations."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

import dashboard
import dashboard_charts
import dashboard_data
import dashboard_live
import dashboard_market_archive
from config import VERSION
from exit_config import ExitConfig


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "ws_messages.json"
RAW_PREVIEW_SESSION_ID = "rawpreview1234567890abcdef12"


def _raw_record(channel: str, payload: dict, session_id: str = RAW_PREVIEW_SESSION_ID) -> dict:
    return {
        "schema_version": "1.0.0",
        "recorder_session_id": session_id,
        "network": "mainnet",
        "channel": channel,
        "subscription": {"type": channel, "coin": "BTC"} if channel != "allMids" else None,
        "exchange_timestamp_ms": 1718000001000,
        "received_timestamp_ms": 1718000001000,
        "sequence": 1,
        "raw_payload": payload,
        "raw_payload_hash": "abc",
        "reconnect_generation": 0,
        "source": "hyperliquid_websocket",
    }


def _write_raw_preview_session(
    tmp_path: Path,
    *,
    session_id: str = RAW_PREVIEW_SESSION_ID,
    channels: tuple[str, ...] = ("trade", "bbo", "candle", "all_mids"),
) -> Path:
    stream = tmp_path / "market_stream"
    raw_dir = stream / "raw" / "2026-06-22"
    raw_dir.mkdir(parents=True)
    raw_file = raw_dir / "events-10.jsonl"
    with FIXTURES.open(encoding="utf-8") as handle:
        fixtures = json.load(handle)
    with raw_file.open("w", encoding="utf-8") as handle:
        for key in channels:
            msg = fixtures[key]
            record = _raw_record(msg["channel"], msg, session_id)
            handle.write(json.dumps(record) + "\n")
    return stream


class TestRawPreviewLoader:
    def test_parses_trade_price_rows(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path, channels=("trade",))
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        preview = dashboard_live.load_raw_preview_market_data(session_id=RAW_PREVIEW_SESSION_ID)
        trade_rows = [row for row in preview["rows"] if row["source_channel"] == "trades"]
        assert trade_rows
        assert trade_rows[0]["coin"] == "BTC"
        assert trade_rows[0]["price"] == pytest.approx(63747.5)
        assert trade_rows[0]["volume"] == pytest.approx(0.1)

    def test_parses_bbo_midpoint_rows(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path, channels=("bbo",))
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        preview = dashboard_live.load_raw_preview_market_data(session_id=RAW_PREVIEW_SESSION_ID)
        bbo_rows = [row for row in preview["rows"] if row["source_channel"] == "bbo"]
        assert bbo_rows
        assert bbo_rows[0]["price"] == pytest.approx(63747.5)
        assert bbo_rows[0]["best_bid"] == pytest.approx(63747.0)
        assert bbo_rows[0]["best_ask"] == pytest.approx(63748.0)
        assert bbo_rows[0]["spread_bps"] == pytest.approx(0.16, abs=0.01)

    def test_parses_candle_close_rows(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path, channels=("candle",))
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        preview = dashboard_live.load_raw_preview_market_data(session_id=RAW_PREVIEW_SESSION_ID)
        candle_rows = [row for row in preview["rows"] if row["source_channel"] == "candle"]
        assert candle_rows
        assert candle_rows[0]["coin"] == "SOL"
        assert candle_rows[0]["price"] == pytest.approx(150.5)
        assert candle_rows[0]["close"] == pytest.approx(150.5)
        assert candle_rows[0]["volume"] == pytest.approx(1000.0)

    def test_handles_missing_files(self, tmp_path, monkeypatch):
        stream = tmp_path / "market_stream"
        stream.mkdir(parents=True)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        preview = dashboard_live.load_raw_preview_market_data(session_id=RAW_PREVIEW_SESSION_ID)
        assert preview["available"] is False
        assert preview["rows"] == []
        assert preview["files_read"] == []

    def test_does_not_write_files(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path)
        raw_file = next(stream.rglob("*.jsonl"))
        before_mtime = raw_file.stat().st_mtime
        before_size = raw_file.stat().st_size
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        dashboard_live.load_raw_preview_market_data(session_id=RAW_PREVIEW_SESSION_ID)

        assert raw_file.stat().st_mtime == before_mtime
        assert raw_file.stat().st_size == before_size


class TestRawPreviewChartIntegration:
    def test_uses_raw_preview_when_normalized_missing(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)
        monkeypatch.setattr(dashboard_data, "DATA", dashboard_data.DATA.__class__(
            data_dir=data_dir,
            price_log_csv=data_dir / "price_log.csv",
            paper_trades_csv=data_dir / "paper_trades.csv",
            backtest_results_json=data_dir / "backtest_results.json",
            recommendations_json=data_dir / "recommendations.json",
            equity_log_csv=data_dir / "equity_log.csv",
            session_configs_dir=data_dir / "session_configs",
            session_reports_dir=data_dir / "session_reports",
            experiments_dir=data_dir / "experiments",
        ))

        chart_data = dashboard_charts.build_chart_data(
            session_id=RAW_PREVIEW_SESSION_ID,
            historical=False,
            timeline=[],
        )
        assert chart_data["chart_data_source"] == "raw_preview"
        assert chart_data["raw_preview_available"] is True
        assert chart_data["normalized_data_available"] is False
        assert chart_data["coins"]["BTC"]["has_data"] is True

    def test_prefers_normalized_data_when_available(self, tmp_path, monkeypatch):
        if not dashboard_live._PARQUET_AVAILABLE:
            pytest.skip("pyarrow not installed")

        import pyarrow as pa
        import pyarrow.parquet as pq

        stream = _write_raw_preview_session(tmp_path)
        coin_dir = stream / "normalized" / "event_type=trade" / "date=2026-06-20" / "coin=BTC"
        coin_dir.mkdir(parents=True)
        table = pa.table(
            {
                "coin": ["BTC"],
                "price": [999.0],
                "size": [1.0],
                "side": ["B"],
                "received_timestamp_ms": [1_718_000_001_000],
            }
        )
        pq.write_table(table, coin_dir / "part-0.parquet")
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        chart_data = dashboard_charts.build_chart_data(
            session_id=RAW_PREVIEW_SESSION_ID,
            historical=True,
            range_key="All local",
            timeline=[],
        )
        assert chart_data["chart_data_source"] == "normalized"
        assert chart_data["raw_preview_available"] is False
        assert chart_data["coins"]["BTC"]["latest_price"] == pytest.approx(999.0)

    def test_debug_fields_include_raw_preview_status(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)
        chart_data = dashboard_charts.build_chart_data(
            session_id=RAW_PREVIEW_SESSION_ID,
            historical=False,
            timeline=[],
        )
        snapshot = dashboard_live.live_dashboard_debug_snapshot(
            {},
            chart_data=chart_data,
            bundle_loaded=True,
        )
        assert snapshot["raw_preview_available"] is True
        assert snapshot["raw_preview_rows"] > 0
        assert snapshot["raw_preview_session_id"] == RAW_PREVIEW_SESSION_ID
        assert snapshot["chart_data_source"] == "raw_preview"
        assert snapshot["raw_preview_channels_used"]


class TestLiveDashboardContext:
    def test_load_live_dashboard_context_missing_files(self, tmp_path, monkeypatch):
        stream_dir = tmp_path / "market_stream"
        stream_dir.mkdir(parents=True)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream_dir)

        context = dashboard_live.load_live_dashboard_context()
        assert context["data_mode"] == "missing"
        assert context["recorder_session_id"] is None

    def test_load_live_dashboard_context_historical_from_manifest(self, tmp_path, monkeypatch):
        stream_dir = tmp_path / "market_stream"
        manifests_dir = stream_dir / "manifests"
        manifests_dir.mkdir(parents=True)
        session_id = "abc123" * 2
        manifest = {
            "recorder_session_id": session_id,
            "generated_at": "2026-06-22T12:00:00+00:00",
            "paper_only": True,
            "execution_enabled": False,
        }
        (manifests_dir / f"{session_id}.json").write_text(json.dumps(manifest), encoding="utf-8")
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream_dir)

        context = dashboard_live.load_live_dashboard_context()
        assert context["data_mode"] == "historical"
        assert context["recorder_session_id"] == session_id


class TestStaleStatusDisplay:
    def test_websocket_panel_marks_stale(self):
        context = {
            "data_mode": "live",
            "recorder_session_id": "deadbeef",
            "stale": True,
            "status": {
                "connection_state": "CONNECTED",
                "state": "STALE_STATUS",
                "running": False,
            },
        }
        panel = dashboard_live.build_websocket_status_panel(context)
        assert panel["stale"] is True
        assert panel["state"] == "STALE_STATUS"
        assert panel["display_connection_status"] == "STALE"

    def test_websocket_panel_connected(self):
        context = {
            "data_mode": "live",
            "recorder_session_id": "abc",
            "stale": False,
            "status": {
                "connection_state": "CONNECTED",
                "running": True,
                "heartbeat_age_seconds": 2.5,
            },
        }
        panel = dashboard_live.build_websocket_status_panel(context)
        assert panel["display_connection_status"] == "CONNECTED"
        assert panel["heartbeat_age_seconds"] == 2.5


class TestHistoricalMarketLoad:
    def test_historical_mode_skips_time_cutoff(self, tmp_path, monkeypatch):
        if not dashboard_live._PARQUET_AVAILABLE:
            pytest.skip("pyarrow not installed")

        import pyarrow as pa
        import pyarrow.parquet as pq

        stream_dir = tmp_path / "market_stream"
        coin_dir = stream_dir / "normalized" / "event_type=trade" / "date=2026-06-20" / "coin=BTC"
        coin_dir.mkdir(parents=True)
        old_ts = 1_700_000_000_000
        table = pa.table(
            {
                "coin": ["BTC"],
                "price": [100.0],
                "size": [1.0],
                "side": ["B"],
                "received_timestamp_ms": [old_ts],
            }
        )
        pq.write_table(table, coin_dir / "part-0.parquet")
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream_dir)

        live_events = dashboard_live.load_recent_market_events(session_id="sess1", historical=False)
        hist_events = dashboard_live.load_recent_market_events(session_id="sess1", historical=True)
        assert live_events == []
        assert len(hist_events) == 1
        assert hist_events[0]["price"] == 100.0


class TestRiskCooldownStatus:
    def test_risk_cooldown_status_labels(self):
        assert dashboard_live.risk_cooldown_status("OK — all risk checks passing") == (
            "OK — all risk checks passing"
        )
        assert "COOLDOWN" in dashboard_live.risk_cooldown_status("Cooldown active — wait 30s")
        assert "LIMIT" in dashboard_live.risk_cooldown_status("Session trade limit reached")


class TestPerSecondAggregation:
    def test_build_per_second_market_rows_from_events(self):
        now_ms = 1_782_123_600_000
        events = [
            {
                "_event_type": "bbo",
                "coin": "BTC",
                "best_bid_price": 100.0,
                "best_ask_price": 100.2,
                "best_bid_size": 3.0,
                "best_ask_size": 1.0,
                "spread": 0.2,
                "spread_bps": 20.0,
                "received_timestamp_ms": now_ms,
            },
            {
                "_event_type": "trade",
                "coin": "BTC",
                "price": 100.1,
                "size": 0.5,
                "side": "B",
                "received_timestamp_ms": now_ms + 100,
            },
            {
                "_event_type": "trade",
                "coin": "BTC",
                "price": 100.15,
                "size": 0.25,
                "side": "A",
                "received_timestamp_ms": now_ms + 200,
            },
        ]

        rows = dashboard_live.build_per_second_market_rows(events, coins=("BTC",))
        assert len(rows) == 1
        row = rows[0]
        assert row["coin"] == "BTC"
        assert row["latest_price"] == 100.15
        assert row["best_bid"] == 100.0
        assert row["best_ask"] == 100.2
        assert row["spread_bps"] == 20.0
        assert row["bbo_imbalance"] == pytest.approx(0.5)
        assert row["trades_per_second"] >= 0


class TestDecisionTimeline:
    def test_append_and_load_decision_timeline(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        monkeypatch.setattr(dashboard_live, "DATA", dashboard_data.DATA.__class__(
            data_dir=data_dir,
            price_log_csv=data_dir / "price_log.csv",
            paper_trades_csv=data_dir / "paper_trades.csv",
            backtest_results_json=data_dir / "backtest_results.json",
            recommendations_json=data_dir / "recommendations.json",
            equity_log_csv=data_dir / "equity_log.csv",
            session_configs_dir=data_dir / "session_configs",
            session_reports_dir=data_dir / "session_reports",
            experiments_dir=data_dir / "experiments",
        ))

        snapshot = {
            "timestamp": "2026-06-22 12:00:00 UTC",
            "decisions": [
                {
                    "coin": "BTC",
                    "recommendation": "HOLD",
                    "primary_reason": "flat",
                    "current_price": 100.0,
                }
            ],
        }
        dashboard_live.append_decision_timeline_from_snapshot(snapshot)
        rows = dashboard_live.load_decision_timeline(limit=10)
        assert len(rows) == 1
        assert rows[0]["coin"] == "BTC"
        assert rows[0]["decision"] == "HOLD"
        assert rows[0]["action_taken"] == "NONE"

    def test_load_decision_timeline_missing_file(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        monkeypatch.setattr(dashboard_live, "DATA", dashboard_data.DATA.__class__(
            data_dir=data_dir,
            price_log_csv=data_dir / "price_log.csv",
            paper_trades_csv=data_dir / "paper_trades.csv",
            backtest_results_json=data_dir / "backtest_results.json",
            recommendations_json=data_dir / "recommendations.json",
            equity_log_csv=data_dir / "equity_log.csv",
            session_configs_dir=data_dir / "session_configs",
            session_reports_dir=data_dir / "session_reports",
            experiments_dir=data_dir / "experiments",
        ))
        assert dashboard_live.load_decision_timeline() == []


class TestMissingFileHandling:
    def test_load_live_market_dashboard_bundle_no_crash(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        stream_dir = data_dir / "market_stream"
        stream_dir.mkdir(parents=True)
        monkeypatch.setattr(dashboard_data, "DATA", dashboard_data.DATA.__class__(
            data_dir=data_dir,
            price_log_csv=data_dir / "price_log.csv",
            paper_trades_csv=data_dir / "paper_trades.csv",
            backtest_results_json=data_dir / "backtest_results.json",
            recommendations_json=data_dir / "recommendations.json",
            equity_log_csv=data_dir / "equity_log.csv",
            session_configs_dir=data_dir / "session_configs",
            session_reports_dir=data_dir / "session_reports",
            experiments_dir=data_dir / "experiments",
        ))
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream_dir)
        monkeypatch.setattr(dashboard_live, "DATA", dashboard_data.DATA)

        bundle = dashboard_live.load_live_market_dashboard_bundle()
        assert "websocket_status" in bundle
        assert len(bundle["per_second_market"]) == 3
        assert bundle["decision_timeline"] == []


class TestSafetyGuards:
    FORBIDDEN_PATTERNS = (
        "private_key",
        "place_order",
        "execute_order",
        "sign_transaction",
        "wallet",
    )

    def test_dashboard_live_has_no_execution_code(self):
        raw = (REPO_ROOT / "dashboard_live.py").read_text(encoding="utf-8")
        source = raw.split('"""', 2)[-1].lower() if raw.count('"""') >= 2 else raw.lower()
        for pattern in ("private_key", "place_order", "execute_order", "sign_transaction"):
            assert pattern not in source

    def test_dashboard_live_has_no_wallet_signing_imports(self):
        tree = ast.parse((REPO_ROOT / "dashboard_live.py").read_text(encoding="utf-8"))
        forbidden = {"wallet", "signing", "web3", "eth_account"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in forbidden
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in forbidden

    def test_live_bot_decisions_are_paper_only(self):
        recommendations = {
            "strategy_profile": "contrarian_v1_test",
            "paper_trader": {"profile_id": "contrarian_v1_test", "session_trades": 0},
            "decisions": [
                {
                    "coin": "BTC",
                    "recommendation": "PAPER BUY",
                    "primary_reason": "test",
                    "signal_strength": "MODERATE",
                    "risk_status": "OK — all risk checks passing",
                }
            ],
        }
        rows = dashboard_live.build_live_bot_decisions(recommendations, [])
        assert rows[0]["decision"] == dashboard_live.DECISION_PAPER_BUY
        assert rows[0]["execution_enabled"] is False
        assert rows[0]["paper_only"] is True


class TestExistingDashboardSections:
    def test_dashboard_module_exports_existing_sections(self):
        assert hasattr(dashboard, "_render_session_summary")
        assert hasattr(dashboard, "_render_current_recommendations")
        assert hasattr(dashboard, "_render_performance_section")
        assert hasattr(dashboard, "_render_market_data_recorder")
        assert hasattr(dashboard, "_render_price_section")
        assert hasattr(dashboard, "_render_experiment_analysis")
        assert hasattr(dashboard, "_render_backtest_results")

    def test_dashboard_has_live_market_section(self):
        assert hasattr(dashboard, "_render_live_market_dashboard")
        assert hasattr(dashboard, "_render_live_market_charts")


class TestMeetingHeaderCards:
    def test_header_cards_labels(self):
        bundle = {
            "websocket_status": {
                "running": True,
                "connection_state": "CONNECTED",
                "stale": False,
                "display_connection_status": "CONNECTED",
            },
            "paper_portfolio": {"paused": False},
            "recommendations": {"timestamp": "now"},
        }
        cards = dashboard_live.meeting_header_cards(bundle)
        assert cards["websocket"] == "CONNECTED"
        assert cards["paper_bot"] == "RUNNING"
        assert cards["safety"] == "PAPER ONLY"

    def test_header_cards_stale(self):
        bundle = {
            "websocket_status": {
                "running": False,
                "connection_state": "CONNECTED",
                "stale": True,
                "display_connection_status": "STALE",
            },
            "paper_portfolio": {"paused": False},
            "recommendations": None,
        }
        cards = dashboard_live.meeting_header_cards(bundle)
        assert cards["websocket"] == "STALE"


class TestRecommendationMapping:
    def test_map_recommendation_to_decision(self):
        assert dashboard_live.map_recommendation_to_decision("PAPER BUY") == "PAPER_BUY_SIGNAL"
        assert dashboard_live.map_recommendation_to_decision("PAPER SELL") == "PAPER_SELL_SIGNAL"
        assert dashboard_live.map_recommendation_to_decision("BLOCKED") == "BLOCKED"

    def test_strategy_version_uses_repo_version(self):
        recommendations = {
            "paper_trader": {"profile_id": "test_profile"},
            "decisions": [
                {
                    "coin": "ETH",
                    "recommendation": "HOLD",
                    "primary_reason": "flat",
                    "signal_strength": "WEAK",
                    "risk_status": "OK — no trade proposed",
                }
            ],
        }
        rows = dashboard_live.build_live_bot_decisions(recommendations, [])
        assert rows[0]["strategy_version"] == VERSION


class TestLiveMarketCharts:
    def test_build_price_line_from_events(self):
        now_ms = 1_782_123_600_000
        events = [
            {
                "_event_type": "trade",
                "coin": "BTC",
                "price": 100.0,
                "received_timestamp_ms": now_ms,
            },
            {
                "_event_type": "trade",
                "coin": "BTC",
                "price": 101.0,
                "received_timestamp_ms": now_ms + 1000,
            },
            {
                "_event_type": "bbo",
                "coin": "BTC",
                "best_bid_price": 100.5,
                "best_ask_price": 100.7,
                "received_timestamp_ms": now_ms + 2000,
            },
        ]
        reference_time_s = now_ms / 1000.0 + 2
        result = dashboard_charts.build_price_line_from_events(
            events,
            "BTC",
            resolution_seconds=1,
            window_seconds=60,
            reference_time_s=reference_time_s,
        )
        assert result["has_data"] is True
        assert result["latest_price"] == pytest.approx(100.6)
        assert result["best_bid"] == 100.5
        assert result["best_ask"] == 100.7
        assert len(result["series"]) >= 2

    def test_missing_market_data_handling(self):
        result = dashboard_charts.build_price_line_from_events([], "ETH", window_seconds=60)
        assert result["has_data"] is False
        assert result["latest_price"] is None

        chart_data = dashboard_charts.build_chart_data(
            session_id=None,
            historical=True,
            recommendations=None,
            timeline=[],
        )
        assert chart_data["coins"]["SOL"]["has_data"] is False
        assert chart_data["paper_only"] is True

    def test_missing_decision_timeline_handling(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        monkeypatch.setattr(dashboard_live, "DATA", dashboard_data.DATA.__class__(
            data_dir=data_dir,
            price_log_csv=data_dir / "price_log.csv",
            paper_trades_csv=data_dir / "paper_trades.csv",
            backtest_results_json=data_dir / "backtest_results.json",
            recommendations_json=data_dir / "recommendations.json",
            equity_log_csv=data_dir / "equity_log.csv",
            session_configs_dir=data_dir / "session_configs",
            session_reports_dir=data_dir / "session_reports",
            experiments_dir=data_dir / "experiments",
        ))

        chart_data = dashboard_charts.build_chart_data(
            session_id=None,
            historical=True,
            timeline=[],
        )
        assert chart_data["decision_timeline_available"] is False

    def test_format_paper_signal_marker(self):
        entry = {
            "timestamp": "2026-06-22 12:00:00 UTC",
            "coin": "BTC",
            "decision": dashboard_live.DECISION_PAPER_BUY,
            "reason": "contrarian dip",
            "price": 100.0,
            "spread_bps": 12.0,
            "bbo_imbalance": 0.2,
            "action_taken": "SIGNAL_ONLY",
        }
        marker = dashboard_charts.format_paper_signal_marker(
            entry,
            strategy_name="contrarian_v1",
            profile_name="contrarian_v1_medium_tp35_sl25",
            signal_strength="MODERATE",
        )
        assert marker["marker_type"] == dashboard_live.DECISION_PAPER_BUY
        assert marker["symbol"] == "triangle-up"
        assert marker["label"] == "paper signal marker"
        assert "Paper signal marker" in marker["hover_text"]
        assert marker["signal_strength"] == "MODERATE"
        assert marker["paper_only"] is True

    def test_planned_exit_display_with_tp_sl(self):
        position = {
            "coin": "BTC",
            "average_entry": 100.0,
            "current_price": 100.4,
            "unrealized_pnl": 0.4,
        }
        exit_config = ExitConfig(
            exit_mode="cost_aware",
            take_profit_pct=0.35,
            stop_loss_pct=0.25,
        )
        detail = dashboard_charts.build_planned_exit_display(
            position,
            exit_config,
            strategy_name="contrarian_v1",
            profile_name="contrarian_v1_medium_tp35_sl25",
            holding_time_seconds=125.0,
        )
        assert detail["planned_take_profit"] == pytest.approx(100.35)
        assert detail["planned_stop_loss"] == pytest.approx(99.75)
        assert detail["has_fixed_tp_sl"] is True
        assert "Planned paper TP" in detail["exit_summary"]
        assert detail["holding_time_label"] == "2m 5s"

    def test_planned_exit_display_without_tp_sl(self):
        position = {
            "coin": "ETH",
            "average_entry": 50.0,
            "current_price": 50.1,
            "unrealized_pnl": 0.1,
        }
        exit_config = ExitConfig(
            exit_mode="signal_exit_v1",
            take_profit_pct=None,
            stop_loss_pct=None,
            signal_exit_enabled=True,
        )
        detail = dashboard_charts.build_planned_exit_display(
            position,
            exit_config,
            strategy_name="contrarian_v1",
            profile_name="contrarian_v1_medium_signal_exit",
        )
        assert detail["has_fixed_tp_sl"] is False
        assert "Signal-based exit" in detail["exit_summary"]

    def test_planned_exit_display_no_exit_config(self):
        position = {
            "coin": "SOL",
            "average_entry": 10.0,
            "current_price": 10.05,
            "unrealized_pnl": 0.05,
        }
        detail = dashboard_charts.build_planned_exit_display(position, None)
        assert detail["exit_summary"] == dashboard_charts.NO_FIXED_EXIT_MESSAGE

    def test_chart_controls_do_not_trigger_execution(self):
        source = inspect.getsource(dashboard._render_live_market_charts)
        assert dashboard_charts.chart_controls_are_visualization_only(source)
        forbidden = ("place_order", "execute_order", "websocket_recorder", "subprocess")
        for token in forbidden:
            assert token not in source

    def test_dashboard_charts_has_no_execution_code(self):
        tree = ast.parse((REPO_ROOT / "dashboard_charts.py").read_text(encoding="utf-8"))
        forbidden = {"wallet", "signing", "web3", "eth_account"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in forbidden
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in forbidden

        raw = (REPO_ROOT / "dashboard_charts.py").read_text(encoding="utf-8")
        body = raw.split('"""', 2)[-1].lower()
        for pattern in ("place_order(", "execute_order(", "sign_transaction("):
            assert pattern not in body
        assert dashboard_charts.chart_controls_are_visualization_only(
            inspect.getsource(dashboard._render_live_market_charts)
        )

    def test_dashboard_loads_in_historical_mode(self, tmp_path, monkeypatch):
        stream_dir = tmp_path / "market_stream"
        manifests_dir = stream_dir / "manifests"
        manifests_dir.mkdir(parents=True)
        session_id = "abc123" * 2
        manifest = {
            "recorder_session_id": session_id,
            "generated_at": "2026-06-22T12:00:00+00:00",
            "paper_only": True,
        }
        (manifests_dir / f"{session_id}.json").write_text(json.dumps(manifest), encoding="utf-8")
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream_dir)
        monkeypatch.setattr(dashboard_data, "DATA", dashboard_data.DATA.__class__(
            data_dir=data_dir,
            price_log_csv=data_dir / "price_log.csv",
            paper_trades_csv=data_dir / "paper_trades.csv",
            backtest_results_json=data_dir / "backtest_results.json",
            recommendations_json=data_dir / "recommendations.json",
            equity_log_csv=data_dir / "equity_log.csv",
            session_configs_dir=data_dir / "session_configs",
            session_reports_dir=data_dir / "session_reports",
            experiments_dir=data_dir / "experiments",
        ))
        monkeypatch.setattr(dashboard_live, "DATA", dashboard_data.DATA)

        bundle = dashboard_live.load_live_market_dashboard_bundle()
        assert bundle["context"]["data_mode"] == "historical"
        chart_data = dashboard_charts.build_chart_data(
            session_id=bundle["context"]["recorder_session_id"],
            historical=True,
            recommendations=bundle.get("recommendations"),
            timeline=bundle.get("decision_timeline"),
        )
        assert chart_data["data_mode"] == "historical"
        assert chart_data["paper_only"] is True

    def test_build_reference_lines(self):
        detail = {
            "paper_entry_price": 100.0,
            "planned_take_profit": 100.35,
            "planned_stop_loss": 99.75,
        }
        lines = dashboard_charts.build_reference_lines(detail, current_price=100.1)
        labels = {line["label"] for line in lines}
        assert "Paper entry" in labels
        assert "Planned paper TP" in labels
        assert "Planned paper SL" in labels
        assert "Current price" in labels

    def test_build_paper_markers_filters_by_coin(self):
        timeline = [
            {
                "timestamp": "2026-06-22 12:00:10 UTC",
                "coin": "BTC",
                "decision": dashboard_live.DECISION_BLOCKED,
                "reason": "spread",
                "price": 100.0,
                "action_taken": "BLOCKED",
            },
            {
                "timestamp": "2026-06-22 12:00:20 UTC",
                "coin": "ETH",
                "decision": dashboard_live.DECISION_HOLD,
                "reason": "flat",
                "price": 50.0,
                "action_taken": "NONE",
            },
        ]
        markers = dashboard_charts.build_paper_markers(
            timeline,
            None,
            coin="BTC",
            window_start_s=1_782_000_000.0,
            reference_time_s=1_783_000_000.0,
        )
        assert len(markers) == 1
        assert markers[0]["coin"] == "BTC"
        assert markers[0]["symbol"] == "x"


class TestPhase1ChartDashboard:
    def test_charts_render_from_raw_preview_without_paper_trades(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        chart_data = dashboard_charts.build_chart_data(
            session_id=RAW_PREVIEW_SESSION_ID,
            historical=False,
            range_key="Live 5m",
            timeline=[],
            recommendations=None,
        )
        assert chart_data["coins"]["BTC"]["has_data"] is True
        assert chart_data["decision_timeline_available"] is False
        assert chart_data["chart_data_source"] == "raw_preview"

    def test_auto_refresh_default_enabled_when_connected(self):
        ws = {
            "display_connection_status": "CONNECTED",
            "running": True,
            "connection_state": "CONNECTED",
        }
        assert dashboard_live.default_auto_refresh_enabled(ws) is True

        ws_stopped = {
            "display_connection_status": "DISCONNECTED",
            "running": False,
            "connection_state": "DISCONNECTED",
        }
        assert dashboard_live.default_auto_refresh_enabled(ws_stopped) is False

    def test_selected_range_is_applied(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        chart_data = dashboard_charts.build_chart_data(
            session_id=RAW_PREVIEW_SESSION_ID,
            historical=False,
            range_key="Live 1m",
            timeline=[],
        )
        assert chart_data["selected_chart_range"] == "Live 1m"
        assert chart_data["window_seconds"] == 60

    def test_all_local_uses_archive_label(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        import dashboard_market_archive as market_archive

        chart_data = dashboard_charts.build_chart_data(
            session_id=RAW_PREVIEW_SESSION_ID,
            historical=True,
            range_key="All local",
            timeline=[],
        )
        assert chart_data["is_all_local"] is True
        assert market_archive.ALL_LOCAL_LABEL in chart_data["data_mode_label"]

    def test_6m_range_shows_warning_when_only_recent_local_data(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        import dashboard_market_archive as market_archive

        chart_data = dashboard_charts.build_chart_data(
            session_id=RAW_PREVIEW_SESSION_ID,
            historical=True,
            range_key="6m",
            timeline=[],
        )
        assert chart_data["missing_history_warning"] == market_archive.MISSING_HISTORY_WARNING

    def test_missing_data_does_not_crash(self, tmp_path, monkeypatch):
        stream_dir = tmp_path / "market_stream"
        stream_dir.mkdir(parents=True)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream_dir)

        chart_data = dashboard_charts.build_chart_data(
            session_id=None,
            historical=True,
            range_key="30d",
            timeline=[],
        )
        assert chart_data["chart_data_source"] == "none"
        assert chart_data["coins"]["BTC"]["has_data"] is False

    def test_paper_decision_absence_keeps_market_charts(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        chart_data = dashboard_charts.build_chart_data(
            session_id=RAW_PREVIEW_SESSION_ID,
            historical=False,
            range_key="Live 5m",
            timeline=[],
            recommendations=None,
        )
        assert chart_data["decision_timeline_available"] is False
        assert chart_data["coins"]["BTC"]["has_data"] is True
        assert chart_data["open_paper_positions"] == []

    def test_dashboard_does_not_start_recorder(self):
        source = inspect.getsource(dashboard._render_live_market_charts)
        forbidden = ("websocket_recorder", "subprocess", "Popen", "main.py")
        for token in forbidden:
            assert token not in source

    def test_dashboard_does_not_modify_raw_files(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path)
        raw_file = next(stream.rglob("*.jsonl"))
        before_mtime = raw_file.stat().st_mtime
        before_size = raw_file.stat().st_size
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        dashboard_market_archive.build_market_archive_index(stream_dir=stream)
        dashboard_charts.build_chart_data(
            session_id=RAW_PREVIEW_SESSION_ID,
            historical=False,
            range_key="Live 5m",
            timeline=[],
            archive_index=dashboard_market_archive.build_market_archive_index(stream_dir=stream),
        )

        assert raw_file.stat().st_mtime == before_mtime
        assert raw_file.stat().st_size == before_size

    def test_no_wallet_signing_order_execution_in_new_modules(self):
        for module_path in (
            REPO_ROOT / "dashboard_market_archive.py",
            REPO_ROOT / "dashboard_charts.py",
        ):
            raw = module_path.read_text(encoding="utf-8")
            body = raw.split('"""', 2)[-1].lower() if raw.count('"""') >= 2 else raw.lower()
            for pattern in ("place_order(", "execute_order(", "sign_transaction("):
                assert pattern not in body
            if module_path.name == "dashboard_market_archive.py":
                assert "private_key" not in body

    def test_downsample_preserves_endpoints(self):
        series = [{"time_ms": idx * 1000, "price": float(idx)} for idx in range(100)]
        sampled, count = dashboard_charts.downsample_series(series, max_points=10)
        assert count <= 11
        assert sampled[0]["price"] == 0.0
        assert sampled[-1]["price"] == 99.0

    def test_archive_index_scans_raw_preview(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        index = dashboard_market_archive.build_market_archive_index(stream_dir=stream)
        assert "raw_preview" in index["data_sources"]
        assert index["raw_preview"]["available"] is True
        assert RAW_PREVIEW_SESSION_ID in index["sessions"] or index["raw_preview"]["file_count"] >= 1


class TestDashboardSchemaSafety:
    def test_default_debug_snapshot_has_required_keys(self):
        snapshot = dashboard_live.default_live_dashboard_debug_snapshot()
        required = (
            "normalized_data_available",
            "raw_preview_available",
            "raw_preview_rows",
            "raw_preview_session_id",
            "raw_preview_files_read",
            "raw_preview_channels_used",
            "chart_data_source",
            "chart_rows_loaded",
            "chart_rows_displayed",
            "auto_refresh_enabled",
            "refresh_interval_seconds",
            "selected_chart_range",
            "earliest_local_timestamp",
            "latest_local_timestamp",
            "missing_history_warning",
            "bundle_loaded",
            "errors",
            "normalization_queue_count",
            "newest_unprocessed_session_id",
            "normalization_sessions_scanned",
            "normalization_queue_warnings",
        )
        for key in required:
            assert key in snapshot

    def test_missing_normalized_data_available_does_not_crash(self):
        chart_data = {"chart_data_source": "none", "coins": {}}
        snapshot = dashboard_live.live_dashboard_debug_snapshot(
            {}, chart_data=chart_data, bundle_loaded=False
        )
        assert snapshot.get("normalized_data_available") is False
        assert snapshot.get("chart_data_source") == "none"

    def test_debug_snapshot_without_chart_data_uses_defaults(self):
        snapshot = dashboard_live.live_dashboard_debug_snapshot({}, bundle_loaded=False)
        assert snapshot["normalized_data_available"] is False
        assert snapshot["raw_preview_available"] is False
        assert snapshot["raw_preview_rows"] == 0
        assert snapshot["chart_data_source"] == "none"

    def test_handles_missing_raw_preview(self, tmp_path, monkeypatch):
        stream_dir = tmp_path / "market_stream"
        stream_dir.mkdir(parents=True)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream_dir)

        chart_data = dashboard_charts.build_chart_data(
            session_id=None,
            historical=True,
            range_key="30d",
            timeline=[],
        )
        assert chart_data.get("raw_preview_available") is False
        snapshot = dashboard_live.live_dashboard_debug_snapshot(
            {}, chart_data=chart_data, bundle_loaded=True
        )
        assert snapshot.get("raw_preview_available") is False

    def test_handles_missing_normalized_data(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        chart_data = dashboard_charts.build_chart_data(
            session_id=RAW_PREVIEW_SESSION_ID,
            historical=False,
            timeline=[],
        )
        assert chart_data.get("normalized_data_available") is False
        assert chart_data.get("raw_preview_available") is True

    def test_handles_missing_paper_decision_timeline(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        chart_data = dashboard_charts.build_chart_data(
            session_id=RAW_PREVIEW_SESSION_ID,
            historical=False,
            timeline=[],
            recommendations=None,
        )
        assert chart_data.get("decision_timeline_available") is False

    def test_build_research_readiness_labels(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)
        index = dashboard_market_archive.build_market_archive_index(stream_dir=stream)

        readiness = dashboard_live.build_research_readiness(
            index,
            session_id=RAW_PREVIEW_SESSION_ID,
            replay_count=0,
            feature_count=0,
            backtest_count=0,
        )
        assert readiness["labels"]["RAW_READY"] is True
        assert readiness["labels"]["NORMALIZED_READY"] is False
        assert readiness.get("next_recommended_command")

    def test_build_data_coverage_rows_schema_safe(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)
        index = dashboard_market_archive.build_market_archive_index(stream_dir=stream)

        rows = dashboard_live.build_data_coverage_rows(index)
        assert rows
        for row in rows:
            assert "coin" in row
            assert "normalized_available" in row
            assert "raw_available" in row


class TestDashboardStateAndFallback:
    def test_charts_render_from_raw_preview_when_recorder_stopped(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path)
        session_id = RAW_PREVIEW_SESSION_ID
        manifests_dir = stream / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        (manifests_dir / f"{session_id}.json").write_text(
            json.dumps(
                {
                    "recorder_session_id": session_id,
                    "generated_at": "2026-06-22T12:00:00+00:00",
                    "raw_row_count": 4,
                    "manifest_finalized": True,
                    "shutdown_reason": "duration_limit",
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        chart_data = dashboard_charts.build_chart_data(
            session_id=session_id,
            historical=True,
            range_key="Live 5m",
            timeline=[],
            recorder_running=False,
        )
        assert chart_data["chart_data_source"] == "raw_preview"
        assert chart_data["coins"]["BTC"]["has_data"] is True

    def test_chart_data_source_is_never_pending(self, tmp_path, monkeypatch):
        stream_dir = tmp_path / "market_stream"
        stream_dir.mkdir(parents=True)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream_dir)

        chart_data = dashboard_charts.build_chart_data(
            session_id=None,
            historical=True,
            range_key="30d",
            timeline=[],
            recorder_running=False,
        )
        assert chart_data["chart_data_source"] in {
            "normalized",
            "raw_preview",
            "archive",
            "local_summary",
            "none",
            "error",
        }

        header_source = inspect.getsource(dashboard._render_header_status_row)
        assert '"pending"' not in header_source

    def test_empty_selected_range_uses_latest_fallback_when_enabled(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        chart_data = dashboard_charts.build_chart_data(
            session_id=RAW_PREVIEW_SESSION_ID,
            historical=True,
            range_key="6m",
            timeline=[],
            show_latest_fallback=True,
            recorder_running=False,
        )
        assert chart_data["chart_data_source"] == "raw_preview"
        assert chart_data["coins"]["BTC"]["has_data"] is True

    def test_recorder_reliability_reads_manifest_while_stopped(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path, session_id=RAW_PREVIEW_SESSION_ID)
        _write_recorder_manifest(
            stream,
            session_id=RAW_PREVIEW_SESSION_ID,
            raw_row_count=12,
            shutdown_reason="duration_limit",
        )
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        context = dashboard_live.load_live_dashboard_context()
        manifest = dashboard_live.resolve_latest_recorder_manifest(context=context)
        summary = dashboard_live.build_recorder_reliability_summary(
            context.get("status") or {},
            context=context,
            manifest=manifest,
        )
        assert summary["session_id"] == RAW_PREVIEW_SESSION_ID
        assert summary["raw_events"] == 12
        assert summary["shutdown_reason"] == "duration_limit"
        assert summary["manifest_consistency"] == "finalized"

    def test_review_mode_render_function_called(self):
        import dashboard_review

        source = inspect.getsource(dashboard.main)
        assert "dashboard_review.render_review_mode" in source
        review_source = inspect.getsource(dashboard_review.render_review_mode)
        assert "Review Mode loaded." in review_source
        assert "_render_review_mode_content" in review_source

    def test_review_mode_has_fallback_content(self):
        import dashboard_review

        source = inspect.getsource(dashboard_review.render_review_mode)
        assert "Review Mode Placeholders" in source
        assert "st.info" in source


class TestDashboardReviewModeSafety:
    def test_review_mode_does_not_run_pytest_on_import(self):
        import inspect

        import dashboard_review

        source = inspect.getsource(dashboard_review._render_health_and_tests_section)
        assert "run_test_summary" in source
        assert "st.button" in source
        module_source = inspect.getsource(dashboard_review.render_review_mode)
        render_body = module_source.split("def render_review_mode")[1]
        assert "run_test_summary" not in render_body

    def test_debug_expander_uses_safe_get(self):
        source = inspect.getsource(dashboard._render_live_dashboard_debug_expander)
        assert "snapshot.get(" in source
        assert 'snapshot["normalized_data_available"]' not in source

    def test_raw_tables_hidden_in_expanders(self):
        source = inspect.getsource(dashboard._render_paper_portfolio_section)
        assert "st.expander" in source
        assert "Per-second market data" in source
        assert "Decision timeline" in source

    def test_internal_debug_expander_collapsed(self):
        source = inspect.getsource(dashboard._render_live_dashboard_debug_expander)
        assert "expanded=False" in source
        assert "Debug (internal)" in source

    def test_no_exception_for_missing_backtest_blocks(self):
        source = inspect.getsource(dashboard._render_backtest_results)
        assert "st.info" in source
        assert 'results["test_mode"]' not in source


def _write_recorder_manifest(
    stream: Path,
    *,
    session_id: str,
    start_ms: int = 1718000000000,
    end_ms: int = 1718003600000,
    raw_row_count: int = 4,
    raw_only_session: bool = True,
    manifest_finalized: bool = True,
    shutdown_reason: str = "duration_limit",
) -> None:
    manifests_dir = stream / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "recorder_session_id": session_id,
        "schema_version": "1.0.0",
        "generated_at": "2026-06-22T10:00:00+00:00",
        "start_time_ms": start_ms,
        "end_time_ms": end_ms,
        "raw_row_count": raw_row_count,
        "normalized_row_count": 0,
        "raw_only_session": raw_only_session,
        "live_normalization": False,
        "manifest_finalized": manifest_finalized,
        "shutdown_reason": shutdown_reason,
        "paper_only": True,
        "execution_enabled": False,
        "channel_counts": {"trades": 2, "bbo": 2},
    }
    (manifests_dir / f"{session_id}.json").write_text(json.dumps(manifest), encoding="utf-8")


class TestNormalizationQueue:
    UNPROCESSED_SESSION_ID = "unprocessed1234567890abcd01"
    PROCESSED_SESSION_ID = "processed1234567890abcdef01"

    def test_raw_only_usable_session_appears_in_queue(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path, session_id=self.UNPROCESSED_SESSION_ID)
        _write_recorder_manifest(stream, session_id=self.UNPROCESSED_SESSION_ID)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        queue = dashboard_live.build_normalization_queue(show_processed=False, stream_dir=stream)
        session_ids = [row["session_id"] for row in queue["rows"]]
        assert self.UNPROCESSED_SESSION_ID in session_ids
        assert queue["normalization_queue_count"] >= 1

    def test_already_normalized_session_excluded_by_default(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path, session_id=self.PROCESSED_SESSION_ID)
        _write_recorder_manifest(stream, session_id=self.PROCESSED_SESSION_ID)
        reports_dir = stream / "normalization_reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / f"{self.PROCESSED_SESSION_ID}.json").write_text(
            json.dumps({"session_id": self.PROCESSED_SESSION_ID, "completed": True}),
            encoding="utf-8",
        )
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        queue = dashboard_live.build_normalization_queue(show_processed=False, stream_dir=stream)
        session_ids = [row["session_id"] for row in queue["rows"]]
        assert self.PROCESSED_SESSION_ID not in session_ids

        processed_queue = dashboard_live.build_normalization_queue(
            show_processed=True, stream_dir=stream
        )
        processed_ids = [row["session_id"] for row in processed_queue["rows"]]
        assert self.PROCESSED_SESSION_ID in processed_ids

    def test_recommended_command_displayed_not_executed(self):
        command = dashboard_live.build_normalization_recommended_command(
            self.UNPROCESSED_SESSION_ID
        )
        assert "python3 market_data_normalize.py" in command
        assert self.UNPROCESSED_SESSION_ID in command
        assert "--skip-allmids-expansion" in command

        render_source = inspect.getsource(dashboard._render_normalization_queue_section)
        assert "st.dataframe" in render_source
        forbidden = ("subprocess", "Popen", "os.system", "market_data_normalize.main")
        for token in forbidden:
            assert token not in render_source

    def test_missing_quality_report_does_not_crash(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path, session_id=self.UNPROCESSED_SESSION_ID)
        _write_recorder_manifest(stream, session_id=self.UNPROCESSED_SESSION_ID)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        queue = dashboard_live.build_normalization_queue(show_processed=False, stream_dir=stream)
        row = next(row for row in queue["rows"] if row["session_id"] == self.UNPROCESSED_SESSION_ID)
        assert row["quality_label"] == "unknown"

    def test_inconsistent_manifest_is_labeled(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path, session_id=self.UNPROCESSED_SESSION_ID)
        _write_recorder_manifest(
            stream,
            session_id=self.UNPROCESSED_SESSION_ID,
            manifest_finalized=True,
            shutdown_reason="running",
        )
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        queue = dashboard_live.build_normalization_queue(show_processed=False, stream_dir=stream)
        row = next(row for row in queue["rows"] if row["session_id"] == self.UNPROCESSED_SESSION_ID)
        assert row["manifest_consistency"] == "INCONSISTENT_MANIFEST"
        assert self.UNPROCESSED_SESSION_ID in str(queue["normalization_queue_warnings"])

    def test_raw_files_not_modified_by_queue_builder(self, tmp_path, monkeypatch):
        stream = _write_raw_preview_session(tmp_path, session_id=self.UNPROCESSED_SESSION_ID)
        _write_recorder_manifest(stream, session_id=self.UNPROCESSED_SESSION_ID)
        raw_file = next(stream.rglob("*.jsonl"))
        before_mtime = raw_file.stat().st_mtime
        before_size = raw_file.stat().st_size
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)

        dashboard_live.build_normalization_queue(show_processed=False, stream_dir=stream)
        dashboard_live.build_normalization_queue(show_processed=True, stream_dir=stream)

        assert raw_file.stat().st_mtime == before_mtime
        assert raw_file.stat().st_size == before_size

    def test_no_wallet_signing_order_execution_in_queue_code(self):
        for module_path in (REPO_ROOT / "dashboard_live.py", REPO_ROOT / "dashboard.py"):
            raw = module_path.read_text(encoding="utf-8")
            body = raw.split('"""', 2)[-1].lower() if raw.count('"""') >= 2 else raw.lower()
            for pattern in ("place_order(", "execute_order(", "sign_transaction(", "private_key"):
                assert pattern not in body
            assert "subprocess" not in inspect.getsource(dashboard_live.build_normalization_queue)
            assert "subprocess" not in inspect.getsource(dashboard._render_normalization_queue_section)
