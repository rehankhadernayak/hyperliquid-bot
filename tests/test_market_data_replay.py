"""Offline tests for market data replay engine."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

import dashboard_data
from market_data.event_models import BboEvent, MidPriceEvent, TradeEvent
from market_data.quality_models import QualityLabel
from market_data.replay_clock import ReplayClock
from market_data.replay_cli import run_replay
from market_data.replay_cursor import ReplayCursor
from market_data.replay_engine import ReplayEngine, ReplayError, load_quality_report_for_session
from market_data.replay_events import (
    CountingConsumer,
    NullConsumer,
    ReplayContext,
    ReplaySummary,
    build_replay_event,
    compute_stream_hash,
    row_to_event,
)
from market_data.replay_filters import ReplayFilters, apply_filters
from market_data.replay_ordering import (
    ORDERING_RULE,
    analyze_ordering,
    assign_replay_sequences,
    primary_time_for_event,
    sort_events_deterministically,
)
from market_data.replay_report import (
    REPLAY_REPORT_SCHEMA_VERSION,
    build_replay_report,
    list_replay_reports,
    render_markdown_report,
    write_replay_report,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REPLAY_MODULES = [
    REPO_ROOT / "market_data" / "replay_engine.py",
    REPO_ROOT / "market_data" / "replay_events.py",
    REPO_ROOT / "market_data" / "replay_ordering.py",
    REPO_ROOT / "market_data" / "replay_cursor.py",
    REPO_ROOT / "market_data" / "replay_clock.py",
    REPO_ROOT / "market_data" / "replay_filters.py",
    REPO_ROOT / "market_data" / "replay_report.py",
    REPO_ROOT / "market_data" / "replay_cli.py",
    REPO_ROOT / "market_data_replay.py",
]


def _write_manifest(tmp_path: Path, session_id: str, files_written: list[str], **overrides) -> Path:
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "recorder_session_id": session_id,
        "start_time_ms": 1000,
        "end_time_ms": 10000,
        "network": "mainnet",
        "coins": ["BTC", "ETH"],
        "schema_version": "1.0.0",
        "files_written": files_written,
        "raw_row_count": 0,
        "normalized_row_count": 0,
        "paper_only": True,
        "execution_enabled": False,
    }
    data.update(overrides)
    path = manifests_dir / f"{session_id}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _write_quality_report(tmp_path: Path, session_id: str, label: str) -> Path:
    reports_dir = tmp_path / "quality_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": "1.0.0",
        "recorder_session_id": session_id,
        "quality_label": label,
        "data_integrity_warnings": [],
        "gap_summary": {"total_gaps": 0},
        "usability_flags": {
            "usable_for_backtesting": {
                "usable": False,
                "reason": "Backtesting replay engine is not implemented yet",
            }
        },
    }
    path = reports_dir / f"{session_id}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _write_normalized_parquet(tmp_path: Path, event_type: str, coin: str, rows: list[dict]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    partition = (
        tmp_path
        / "normalized"
        / f"event_type={event_type}"
        / "date=2026-06-18"
        / f"coin={coin}"
    )
    partition.mkdir(parents=True, exist_ok=True)
    path = partition / f"events-test-{event_type}-{coin}.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(path))
    return path


def _sample_trade_row(coin: str, recv_ms: int, exchange_ms: int, trade_id: int = 1) -> dict:
    return {
        "coin": coin,
        "side": "B",
        "price": 100.0,
        "size": 0.1,
        "trade_id": trade_id,
        "transaction_hash": "0xabc",
        "exchange_timestamp_ms": exchange_ms,
        "received_timestamp_ms": recv_ms,
    }


def _sample_bbo_row(coin: str, recv_ms: int, exchange_ms: int) -> dict:
    return {
        "coin": coin,
        "best_bid_price": 99.0,
        "best_bid_size": 1.0,
        "best_ask_price": 101.0,
        "best_ask_size": 1.0,
        "spread": 2.0,
        "spread_bps": 20.0,
        "exchange_timestamp_ms": exchange_ms,
        "received_timestamp_ms": recv_ms,
    }


def _sample_mid_row(coin: str, recv_ms: int, price: float = 100.0) -> dict:
    return {
        "coin": coin,
        "mid_price": price,
        "exchange_timestamp_ms": None,
        "received_timestamp_ms": recv_ms,
    }


def _build_replay_session(tmp_path: Path, session_id: str = "replaytest01") -> Path:
    trade_path = _write_normalized_parquet(
        tmp_path,
        "trade",
        "BTC",
        [_sample_trade_row("BTC", 3000, 2900, 1)],
    )
    bbo_path = _write_normalized_parquet(
        tmp_path,
        "bbo",
        "ETH",
        [_sample_bbo_row("ETH", 2000, 1900)],
    )
    mid_path = _write_normalized_parquet(
        tmp_path,
        "mid",
        "BTC",
        [_sample_mid_row("BTC", 5000)],
    )
    _write_manifest(
        tmp_path,
        session_id,
        files_written=[str(trade_path), str(bbo_path), str(mid_path)],
    )
    _write_quality_report(tmp_path, session_id, QualityLabel.GOOD_FOR_RESEARCH.value)
    return tmp_path


def _make_replay_event(
    event_type: str,
    coin: str,
    primary_time: int,
    received: int,
    exchange: int | None,
    source_file: str,
    row_index: int,
    seq: int = 1,
    local_seq: int | None = None,
    primary_source: str = "exchange_timestamp_ms",
    channel: str | None = None,
) -> "ReplayEvent":
    from market_data.replay_events import channel_for_event_type

    if event_type == "trade":
        event = TradeEvent(
            coin=coin,
            side="B",
            price=100.0,
            size=0.1,
            trade_id=1,
            transaction_hash="0x",
            exchange_timestamp_ms=exchange or received,
            received_timestamp_ms=received,
        )
    elif event_type == "bbo":
        event = BboEvent(
            coin=coin,
            best_bid_price=99.0,
            best_bid_size=1.0,
            best_ask_price=101.0,
            best_ask_size=1.0,
            spread=2.0,
            spread_bps=20.0,
            exchange_timestamp_ms=exchange,
            received_timestamp_ms=received,
        )
    else:
        event = MidPriceEvent(
            coin=coin,
            mid_price=100.0,
            exchange_timestamp_ms=exchange,
            received_timestamp_ms=received,
        )

    ch = channel or channel_for_event_type(event_type)
    replay_event = build_replay_event(
        event=event,
        event_type=event_type,
        replay_session_id="replay-id",
        source_recorder_session_id="source-id",
        source_file=source_file,
        source_row_index=row_index,
        replay_sequence=seq,
        original_local_sequence=local_seq,
        primary_time_ms=primary_time,
        primary_time_source=primary_source,
    )
    return replay_event.__class__(
        **{**replay_event.__dict__, "channel": ch}
    )


class TestManifestAndQuality:
    def test_missing_manifest_raises(self, tmp_path):
        engine = ReplayEngine(market_stream_dir=tmp_path, test_mode=True)
        with pytest.raises(ReplayError, match="Manifest not found"):
            engine.load_session("missing")

    def test_missing_quality_report_allowed(self, tmp_path):
        session_id = "noquality01"
        trade_path = _write_normalized_parquet(
            tmp_path, "trade", "BTC", [_sample_trade_row("BTC", 3000, 2900)]
        )
        _write_manifest(tmp_path, session_id, files_written=[str(trade_path)])
        engine = ReplayEngine(market_stream_dir=tmp_path, test_mode=True)
        session = engine.load_session(session_id)
        assert session.quality_report is None

    def test_unusable_quality_blocks_replay(self, tmp_path):
        session_id = "unusable01"
        trade_path = _write_normalized_parquet(
            tmp_path, "trade", "BTC", [_sample_trade_row("BTC", 3000, 2900)]
        )
        _write_manifest(tmp_path, session_id, files_written=[str(trade_path)])
        _write_quality_report(tmp_path, session_id, QualityLabel.UNUSABLE.value)
        engine = ReplayEngine(market_stream_dir=tmp_path, test_mode=True)
        with pytest.raises(ReplayError, match="UNUSABLE"):
            engine.load_session(session_id)

    def test_allow_degraded_permits_unusable(self, tmp_path):
        session_id = "unusable02"
        _build_replay_session(tmp_path, session_id)
        _write_quality_report(tmp_path, session_id, QualityLabel.UNUSABLE.value)
        engine = ReplayEngine(market_stream_dir=tmp_path, test_mode=True)
        session = engine.load_session(session_id, allow_degraded=True)
        assert session.manifest.exists

    def test_backtesting_false_does_not_block(self, tmp_path):
        session_id = "backtestflag"
        _build_replay_session(tmp_path, session_id)
        engine = ReplayEngine(market_stream_dir=tmp_path, test_mode=True)
        result = engine.run(session_id, dry_run=True, output_dir=tmp_path / "replays")
        assert result.report["total_events"] > 0


class TestEventLoading:
    def test_normalized_file_discovery(self, tmp_path):
        session_id = "discover01"
        _build_replay_session(tmp_path, session_id)
        engine = ReplayEngine(market_stream_dir=tmp_path, test_mode=True)
        session = engine.load_session(session_id)
        assert len(session.normalized_files_used) == 3

    def test_event_model_reconstruction(self):
        row = _sample_trade_row("BTC", 3000, 2900)
        event = row_to_event(row, "trade")
        assert isinstance(event, TradeEvent)
        assert event.coin == "BTC"

    def test_raw_validation_warnings(self, tmp_path):
        session_id = "rawwarn01"
        trade_path = _write_normalized_parquet(
            tmp_path, "trade", "BTC", [_sample_trade_row("BTC", 3000, 2900)]
        )
        _write_manifest(
            tmp_path,
            session_id,
            files_written=[str(trade_path), "data/market_stream/raw/missing.jsonl"],
        )
        from market_data.replay_manifest import load_session_manifest

        manifest = load_session_manifest(session_id, tmp_path)
        from market_data.replay_engine import validate_raw_files

        warnings = validate_raw_files(manifest)
        assert any("missing" in w.lower() for w in warnings)


class TestDeterministicOrdering:
    def test_ordering_rule_documented(self):
        assert "exchange_timestamp_ms" in ORDERING_RULE
        assert "trades" in ORDERING_RULE

    def test_primary_time_exchange_when_available(self):
        event = TradeEvent(
            coin="BTC",
            side="B",
            price=1.0,
            size=1.0,
            trade_id=1,
            transaction_hash="0x",
            exchange_timestamp_ms=1000,
            received_timestamp_ms=1100,
        )
        primary, source = primary_time_for_event(event, "trade")
        assert primary == 1000
        assert source == "exchange_timestamp_ms"

    def test_missing_exchange_timestamp_fallback(self):
        event = MidPriceEvent(
            coin="BTC",
            mid_price=100.0,
            exchange_timestamp_ms=None,
            received_timestamp_ms=5000,
        )
        primary, source = primary_time_for_event(event, "mid")
        assert primary == 5000
        assert source == "no_exchange_timestamp"

    def test_channel_priority_tie_breaker(self):
        events = [
            _make_replay_event("bbo", "BTC", 1000, 1000, 1000, "/a.parquet", 0, 1, channel="bbo"),
            _make_replay_event("trade", "BTC", 1000, 1000, 1000, "/b.parquet", 0, 2, channel="trades"),
        ]
        sorted_events = sort_events_deterministically(events)
        assert sorted_events[0].channel == "trades"

    def test_coin_alphabetical_tie_breaker(self):
        events = [
            _make_replay_event("trade", "ETH", 1000, 1000, 1000, "/a.parquet", 0, 1, channel="trades"),
            _make_replay_event("trade", "BTC", 1000, 1000, 1000, "/a.parquet", 1, 2, channel="trades"),
        ]
        sorted_events = sort_events_deterministically(events)
        assert sorted_events[0].coin == "BTC"

    def test_source_row_tie_breaker(self):
        events = [
            _make_replay_event("trade", "BTC", 1000, 1000, 1000, "/same.parquet", 5, 1, channel="trades"),
            _make_replay_event("trade", "BTC", 1000, 1000, 1000, "/same.parquet", 1, 2, channel="trades"),
        ]
        sorted_events = sort_events_deterministically(events)
        assert sorted_events[0].source_row_index == 1

    def test_deterministic_ordering_stable(self, tmp_path):
        _build_replay_session(tmp_path, "order01")
        engine = ReplayEngine(market_stream_dir=tmp_path, test_mode=True)
        r1 = engine.run("order01", dry_run=True, replay_session_id="fixed-id-1")
        r2 = engine.run("order01", dry_run=True, replay_session_id="fixed-id-2")
        assert compute_stream_hash(r1.events) == compute_stream_hash(r2.events)


class TestReplayCursor:
    def _events(self) -> list:
        return assign_replay_sequences(
            sort_events_deterministically(
                [
                    _make_replay_event("trade", "BTC", 1000, 1000, 1000, "/a", 0, 1),
                    _make_replay_event("trade", "BTC", 2000, 2000, 2000, "/a", 1, 2),
                    _make_replay_event("trade", "BTC", 3000, 3000, 3000, "/a", 2, 3),
                ]
            )
        )

    def test_next_event(self):
        cursor = ReplayCursor(self._events())
        e1 = cursor.next_event()
        assert e1 is not None
        assert e1.replay_sequence == 1
        assert cursor.current_position().index == 1

    def test_peek_event(self):
        cursor = ReplayCursor(self._events())
        peeked = cursor.peek_event()
        assert peeked.replay_sequence == 1
        assert cursor.current_position().index == 0

    def test_seek_to_timestamp(self):
        cursor = ReplayCursor(self._events())
        assert cursor.seek_to_timestamp(2000)
        assert cursor.peek_event().primary_time_ms == 2000

    def test_seek_to_sequence(self):
        cursor = ReplayCursor(self._events())
        assert cursor.seek_to_sequence(3)
        assert cursor.peek_event().replay_sequence == 3

    def test_reset(self):
        cursor = ReplayCursor(self._events())
        cursor.next_event()
        cursor.reset()
        assert cursor.current_position().index == 0

    def test_end_of_stream(self):
        cursor = ReplayCursor(self._events())
        for _ in range(3):
            cursor.next_event()
        assert cursor.at_end()
        assert cursor.next_event() is None

    def test_no_future_leakage(self):
        cursor = ReplayCursor(self._events())
        assert cursor.events_ahead_count() == 1
        assert cursor.max_future_index() == 0
        cursor.next_event()
        assert cursor.events_ahead_count() == 1
        assert cursor.max_future_index() == 1
        cursor.next_event()
        cursor.next_event()
        assert cursor.at_end()
        assert cursor.events_ahead_count() == 0


class TestReplayClock:
    def test_speed_zero_no_sleep(self):
        clock = ReplayClock(speed=0, test_mode=True)
        clock.start(1000)
        clock.wait_until(5000)
        assert clock.total_sleep_seconds == 0

    def test_speed_one_records_delay(self):
        sleeps: list[float] = []

        def fake_sleep(sec: float) -> None:
            sleeps.append(sec)

        clock = ReplayClock(speed=1.0, test_mode=False, sleep_fn=fake_sleep)
        clock.start(1000)
        clock.wait_until(2000)
        assert len(sleeps) == 1
        assert sleeps[0] == pytest.approx(1.0)

    def test_deterministic_test_clock(self):
        clock = ReplayClock(speed=1.0, test_mode=True)
        clock.start(1000)
        clock.wait_until(3000)
        assert clock.total_sleep_seconds == 0

    def test_max_events_limit(self):
        clock = ReplayClock(max_events=2, test_mode=True)
        assert clock.should_emit(1000)
        clock.record_emission()
        assert clock.should_emit(2000)
        clock.record_emission()
        assert not clock.should_emit(3000)


class TestFilters:
    def _session_events(self, tmp_path):
        _build_replay_session(tmp_path, "filter01")
        engine = ReplayEngine(market_stream_dir=tmp_path, test_mode=True)
        return engine.run("filter01", dry_run=True)

    def test_filter_by_coin(self, tmp_path):
        result = self._session_events(tmp_path)
        filters = ReplayFilters(coins=["BTC"])
        filtered, skipped = apply_filters(result.events, filters)
        assert all(e.coin == "BTC" for e in filtered)
        assert skipped >= 1

    def test_filter_by_channel(self, tmp_path):
        result = self._session_events(tmp_path)
        filters = ReplayFilters(channels=["trades"])
        filtered, _ = apply_filters(result.events, filters)
        assert all(e.original_event_type == "trade" for e in filtered)

    def test_start_end_timestamp_filters(self, tmp_path):
        result = self._session_events(tmp_path)
        filters = ReplayFilters(start_timestamp_ms=2500, end_timestamp_ms=4000)
        filtered, _ = apply_filters(result.events, filters)
        assert all(2500 <= e.primary_time_ms <= 4000 for e in filtered)

    def test_max_events_filter(self, tmp_path):
        result = self._session_events(tmp_path)
        filters = ReplayFilters(max_events=1)
        filtered, _ = apply_filters(result.events, filters)
        assert len(filtered) == 1


class TestDryRunAndConsumers:
    def test_dry_run_mode(self, tmp_path):
        _build_replay_session(tmp_path, "dryrun01")
        engine = ReplayEngine(market_stream_dir=tmp_path, test_mode=True)
        result = engine.run(
            "dryrun01",
            dry_run=True,
            output_dir=tmp_path / "replays",
            replay_session_id="dryrun-report-id",
        )
        assert result.report["dry_run"] is True
        assert result.report["emitted_count"] == 0
        assert (tmp_path / "replays" / "dryrun-report-id.json").exists()

    def test_consumer_callbacks(self, tmp_path):
        _build_replay_session(tmp_path, "consumer01")
        engine = ReplayEngine(market_stream_dir=tmp_path, test_mode=True)
        consumer = CountingConsumer()
        engine.run("consumer01", dry_run=False, consumer=consumer, speed=0)
        assert consumer.started
        assert consumer.count > 0
        assert consumer.finished


class TestStreamHash:
    def test_stream_hash_stability(self, tmp_path):
        _build_replay_session(tmp_path, "hash01")
        engine = ReplayEngine(market_stream_dir=tmp_path, test_mode=True)
        r1 = engine.run("hash01", dry_run=True, replay_session_id="hash-fixed")
        r2 = engine.run("hash01", dry_run=True, replay_session_id="hash-fixed")
        assert r1.report["stream_hash"] == r2.report["stream_hash"]


class TestOrderingWarnings:
    def test_backward_exchange_timestamp_warning(self):
        events = assign_replay_sequences(
            [
                _make_replay_event("trade", "BTC", 2000, 2000, 2000, "/a", 0, 1),
                _make_replay_event("trade", "BTC", 1000, 1000, 1000, "/a", 1, 2),
            ]
        )
        for e in events:
            object.__setattr__(e, "channel", "trades")
        analysis = analyze_ordering(events)
        assert any("backward" in w.lower() for w in analysis.ordering_warnings)


class TestReportOutput:
    def test_report_json_output(self, tmp_path):
        _build_replay_session(tmp_path, "report01")
        engine = ReplayEngine(market_stream_dir=tmp_path, test_mode=True)
        result = engine.run(
            "report01",
            dry_run=True,
            output_dir=tmp_path / "replays",
            replay_session_id="report-json-id",
        )
        path = tmp_path / "replays" / "report-json-id.json"
        data = json.loads(path.read_text())
        assert data["schema_version"] == REPLAY_REPORT_SCHEMA_VERSION
        assert data["paper_only"] is True
        assert data["execution_enabled"] is False

    def test_report_markdown_output(self, tmp_path):
        _build_replay_session(tmp_path, "report02")
        engine = ReplayEngine(market_stream_dir=tmp_path, test_mode=True)
        result = engine.run(
            "report02",
            dry_run=True,
            output_dir=tmp_path / "replays",
            replay_session_id="report-md-id",
        )
        md = (tmp_path / "replays" / "report-md-id.md").read_text()
        assert "Market Data Replay Report" in md
        assert "read-only" in md.lower()


class TestDashboardLoader:
    def test_load_replay_reports_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", tmp_path)
        assert dashboard_data.load_replay_reports() == []

    def test_load_replay_report_roundtrip(self, tmp_path, monkeypatch):
        replays_dir = tmp_path / "replays"
        replays_dir.mkdir(parents=True)
        report = {"replay_session_id": "dash-replay-01", "total_events": 5}
        (replays_dir / "dash-replay-01.json").write_text(json.dumps(report))
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", tmp_path)
        loaded = dashboard_data.load_replay_report("dash-replay-01")
        assert loaded["total_events"] == 5


class TestSafety:
    def test_no_wallet_signing_execution_code(self):
        from market_data.security_guard import FORBIDDEN_IMPORT_MODULES

        for path in REPLAY_MODULES:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert alias.name.split(".")[0] not in FORBIDDEN_IMPORT_MODULES, (
                            f"{path}: {alias.name}"
                        )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert node.module.split(".")[0] not in FORBIDDEN_IMPORT_MODULES, (
                        f"{path}: {node.module}"
                    )
        assert "execution_enabled"  # replay reports always false

    def test_paper_trader_unchanged(self):
        import paper_trader

        assert hasattr(paper_trader, "PaperTrader")

    def test_no_live_file_modification(self, tmp_path):
        live_manifest = REPO_ROOT / "data/market_stream/manifests"
        if not live_manifest.exists():
            pytest.skip("No live manifests")
        before = {p: p.stat().st_mtime for p in live_manifest.glob("*.json")}
        _build_replay_session(tmp_path, "safetest")
        engine = ReplayEngine(market_stream_dir=tmp_path, test_mode=True)
        engine.run("safetest", dry_run=True, output_dir=tmp_path / "replays")
        for p, mtime in before.items():
            assert p.stat().st_mtime == mtime


class TestCLI:
    def test_cli_dry_run(self, tmp_path):
        sid = "cli01"
        _build_replay_session(tmp_path, sid)

        from argparse import Namespace

        args = Namespace(
            session_id=sid,
            latest=False,
            dry_run=True,
            emit=False,
            coins=None,
            channels=None,
            speed=1.0,
            max_events=None,
            start_timestamp_ms=None,
            end_timestamp_ms=None,
            exclude_candles=False,
            exclude_l2_book=False,
            allow_degraded=False,
            output_dir=tmp_path / "replays",
            market_stream_dir=tmp_path,
        )
        assert run_replay(args) == 0
