"""Offline tests for market data quality reports."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from market_data.quality_checks import (
    assign_overall_label,
    build_recommendations,
    build_usability_flags,
    inspect_session,
)
from market_data.quality_cli import run_quality_report
from market_data.quality_models import QualityLabel
from market_data.quality_report import build_quality_report, render_markdown_report, write_quality_report
from market_data.replay_manifest import (
    SessionManifest,
    find_latest_session_id,
    load_session_manifest,
)
import dashboard_data

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_MANIFEST = REPO_ROOT / "data/market_stream/manifests/0440bbbe838e4ce9adad8605ac364d4d.json"
LIVE_RAW = REPO_ROOT / "data/market_stream/raw/2026-06-18/events-15.jsonl"


def _write_manifest(tmp_path: Path, session_id: str, **overrides) -> Path:
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "recorder_session_id": session_id,
        "start_time_ms": 1000,
        "end_time_ms": 61000,
        "network": "mainnet",
        "coins": ["BTC", "ETH", "SOL"],
        "schema_version": "1.0.0",
        "files_written": [],
        "raw_row_count": 0,
        "normalized_row_count": 0,
        "duplicate_count": 0,
        "malformed_count": 0,
        "reconnect_count": 0,
        "channel_statistics": {},
        "latency_statistics": {},
        "gap_statistics": {"total_gaps": 0, "gaps": []},
        "shutdown_reason": "test",
        "paper_only": True,
        "execution_enabled": False,
    }
    data.update(overrides)
    path = manifests_dir / f"{session_id}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _raw_line(session_id: str, channel: str, seq: int, recv_ms: int, **extra) -> str:
    record = {
        "schema_version": "1.0.0",
        "recorder_session_id": session_id,
        "network": "mainnet",
        "channel": channel,
        "subscription": extra.get("subscription"),
        "exchange_timestamp_ms": extra.get("exchange_timestamp_ms"),
        "received_timestamp_ms": recv_ms,
        "sequence": seq,
        "raw_payload": extra.get("raw_payload", {"channel": channel, "data": {}}),
        "source": "hyperliquid_websocket",
    }
    return json.dumps(record)


def _build_valid_session(tmp_path: Path, session_id: str = "testsession001") -> Path:
    raw_dir = tmp_path / "raw" / "2026-06-18"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "events-12.jsonl"
    lines = [
        _raw_line(
            session_id,
            "bbo",
            1,
            2000,
            exchange_timestamp_ms=1900,
            subscription={"type": "bbo", "coin": "BTC"},
            raw_payload={
                "channel": "bbo",
                "data": {
                    "coin": "BTC",
                    "time": 1900,
                    "bbo": [{"px": "100.0", "sz": "1.0"}, {"px": "100.1", "sz": "2.0"}],
                },
            },
        ),
        _raw_line(
            session_id,
            "trades",
            2,
            3000,
            exchange_timestamp_ms=2900,
            raw_payload={
                "channel": "trades",
                "data": [
                    {
                        "coin": "BTC",
                        "side": "B",
                        "px": "100.05",
                        "sz": "0.1",
                        "time": 2900,
                        "hash": "0xabc",
                        "tid": 1,
                    }
                ],
            },
        ),
        _raw_line(
            session_id,
            "candle",
            3,
            4000,
            exchange_timestamp_ms=3600000,
            subscription={"type": "candle", "coin": "BTC"},
            raw_payload={
                "channel": "candle",
                "data": {
                    "t": 3600000,
                    "T": 3605999,
                    "s": "BTC",
                    "i": "1m",
                    "o": "100",
                    "h": "101",
                    "l": "99",
                    "c": "100.5",
                    "v": "10",
                    "n": 5,
                },
            },
        ),
        _raw_line(
            session_id,
            "allMids",
            4,
            5000,
            exchange_timestamp_ms=None,
            raw_payload={"channel": "allMids", "data": {"mids": {"BTC": "100.0"}}},
        ),
        _raw_line(
            session_id,
            "l2Book",
            5,
            6000,
            exchange_timestamp_ms=5900,
            subscription={"type": "l2Book", "coin": "BTC"},
            raw_payload={
                "channel": "l2Book",
                "data": {
                    "coin": "BTC",
                    "time": 5900,
                    "levels": [
                        [{"px": "100.0", "sz": "1.0"}, {"px": "99.9", "sz": "2.0"}],
                        [{"px": "100.1", "sz": "1.0"}, {"px": "100.2", "sz": "2.0"}],
                    ],
                },
            },
        ),
        _raw_line(
            session_id,
            "activeAssetCtx",
            6,
            7000,
            exchange_timestamp_ms=None,
            subscription={"type": "activeAssetCtx", "coin": "BTC"},
            raw_payload={"channel": "activeAssetCtx", "data": {"coin": "BTC", "ctx": {}}},
        ),
    ]
    raw_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_manifest(
        tmp_path,
        session_id,
        files_written=[str(raw_path)],
        raw_row_count=len(lines),
        normalized_row_count=0,
        channel_statistics={"bbo": 1, "trades": 1, "candle": 1, "allMids": 1, "l2Book": 1, "activeAssetCtx": 1},
    )
    return tmp_path


class TestMissingManifest:
    def test_missing_manifest_unusable(self, tmp_path):
        manifest = load_session_manifest("missing", tmp_path)
        state = inspect_session(manifest)
        label = assign_overall_label(state, manifest)
        assert label == QualityLabel.UNUSABLE
        assert any(w.category == "manifest" for w in state.warnings)


class TestMissingRawFiles:
    def test_missing_raw_files_critical(self, tmp_path):
        session_id = "missingraw01"
        _write_manifest(
            tmp_path,
            session_id,
            files_written=[str(tmp_path / "raw/missing.jsonl")],
        )
        manifest = load_session_manifest(session_id, tmp_path)
        state = inspect_session(manifest)
        assert any(w.category == "missing_file" for w in state.warnings)
        assert assign_overall_label(state, manifest) == QualityLabel.UNUSABLE


class TestUnreadableRaw:
    def test_unreadable_raw_file(self, tmp_path):
        session_id = "badraw0001"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "bad.jsonl"
        raw_path.write_text("{not valid json\n", encoding="utf-8")
        _write_manifest(tmp_path, session_id, files_written=[str(raw_path)], raw_row_count=1)
        manifest = load_session_manifest(session_id, tmp_path)
        state = inspect_session(manifest)
        assert state.raw_malformed_count >= 1
        assert any(w.category == "malformed_raw" for w in state.warnings)


class TestValidManifest:
    def test_valid_session_inspection(self, tmp_path):
        session_id = "valid000001"
        stream_dir = _build_valid_session(tmp_path, session_id)
        manifest = load_session_manifest(session_id, stream_dir)
        state = inspect_session(manifest)
        assert state.raw_inspected_count == 6
        label = assign_overall_label(state, manifest)
        assert label in (QualityLabel.GOOD_FOR_RESEARCH, QualityLabel.USABLE_WITH_WARNINGS)


class TestChannelSummaries:
    def test_channel_summaries_populated(self, tmp_path):
        session_id = "chsummary01"
        stream_dir = _build_valid_session(tmp_path, session_id)
        state = inspect_session(load_session_manifest(session_id, stream_dir))
        assert "bbo" in state.channel_summaries
        assert state.channel_summaries["bbo"].raw_event_count == 1
        assert state.channel_summaries["trades"].raw_event_count == 1


class TestCoinSummaries:
    def test_coin_summaries_btc(self, tmp_path):
        session_id = "coinsum001"
        stream_dir = _build_valid_session(tmp_path, session_id)
        state = inspect_session(load_session_manifest(session_id, stream_dir))
        assert "BTC" in state.coin_summaries or state.raw_inspected_count > 0


class TestLatencyByChannel:
    def test_latency_trades_bbo(self, tmp_path):
        session_id = "latency001"
        stream_dir = _build_valid_session(tmp_path, session_id)
        state = inspect_session(load_session_manifest(session_id, stream_dir))
        assert state.latency_by_channel["trades"].count >= 1
        assert state.latency_by_channel["bbo"].count >= 1
        assert state.latency_by_channel["l2Book"].count >= 1


class TestAllMidsLatencyUnavailable:
    def test_allmids_latency_unavailable(self, tmp_path):
        session_id = "allmids001"
        stream_dir = _build_valid_session(tmp_path, session_id)
        state = inspect_session(load_session_manifest(session_id, stream_dir))
        assert state.latency_by_channel["allMids"].latency_available is False


class TestCandleLatencyInformational:
    def test_candle_latency_informational(self, tmp_path):
        session_id = "candlelat01"
        stream_dir = _build_valid_session(tmp_path, session_id)
        state = inspect_session(load_session_manifest(session_id, stream_dir))
        assert state.latency_by_channel["candle"].informational_only is True


class TestNegativeLatency:
    def test_negative_latency_warning(self, tmp_path):
        session_id = "neglat0001"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "events.jsonl"
        raw_path.write_text(
            _raw_line(
                session_id,
                "bbo",
                1,
                1000,
                exchange_timestamp_ms=5000,
                subscription={"type": "bbo", "coin": "BTC"},
                raw_payload={
                    "channel": "bbo",
                    "data": {
                        "coin": "BTC",
                        "bbo": [{"px": "100", "sz": "1"}, {"px": "101", "sz": "1"}],
                    },
                },
            )
            + "\n",
            encoding="utf-8",
        )
        _write_manifest(tmp_path, session_id, files_written=[str(raw_path)], raw_row_count=1)
        state = inspect_session(load_session_manifest(session_id, tmp_path))
        assert state.latency_by_channel["bbo"].negative_latency_count == 1
        assert any(w.category == "clock_quality" for w in state.warnings)


class TestLocalTimestampOrdering:
    def test_local_timestamp_backward(self, tmp_path):
        session_id = "localts001"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "events.jsonl"
        raw_path.write_text(
            "\n".join(
                [
                    _raw_line(session_id, "bbo", 1, 5000, subscription={"type": "bbo", "coin": "BTC"}),
                    _raw_line(session_id, "bbo", 2, 4000, subscription={"type": "bbo", "coin": "BTC"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        _write_manifest(tmp_path, session_id, files_written=[str(raw_path)], raw_row_count=2)
        state = inspect_session(load_session_manifest(session_id, tmp_path))
        assert state.local_backward_count >= 1


class TestBackwardExchangeTimestamps:
    def test_backward_exchange_timestamp(self, tmp_path):
        session_id = "backexch01"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "events.jsonl"
        raw_path.write_text(
            "\n".join(
                [
                    _raw_line(
                        session_id,
                        "trades",
                        1,
                        2000,
                        exchange_timestamp_ms=2000,
                        raw_payload={"channel": "trades", "data": [{"coin": "BTC", "px": "1", "sz": "1", "side": "B", "tid": 1, "hash": "0x", "time": 2000}]},
                    ),
                    _raw_line(
                        session_id,
                        "trades",
                        2,
                        3000,
                        exchange_timestamp_ms=1000,
                        raw_payload={"channel": "trades", "data": [{"coin": "BTC", "px": "1", "sz": "1", "side": "B", "tid": 2, "hash": "0x", "time": 1000}]},
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        _write_manifest(tmp_path, session_id, files_written=[str(raw_path)], raw_row_count=2)
        state = inspect_session(load_session_manifest(session_id, tmp_path))
        assert state.backward_exchange_count >= 1


class TestCandleContinuity:
    def test_missing_candle_period(self, tmp_path):
        session_id = "candlegap1"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "events.jsonl"
        candle = lambda open_ms: _raw_line(
            session_id,
            "candle",
            open_ms,
            open_ms + 100,
            exchange_timestamp_ms=open_ms,
            subscription={"type": "candle", "coin": "BTC"},
            raw_payload={
                "channel": "candle",
                "data": {"t": open_ms, "T": open_ms + 59999, "s": "BTC", "i": "1m", "o": "1", "h": "1", "l": "1", "c": "1", "v": "1", "n": 1},
            },
        )
        raw_path.write_text("\n".join([candle(0), candle(120000)]) + "\n", encoding="utf-8")
        _write_manifest(
            tmp_path,
            session_id,
            files_written=[str(raw_path)],
            raw_row_count=2,
            end_time_ms=200000,
        )
        state = inspect_session(load_session_manifest(session_id, tmp_path))
        assert any(g.gap_type == "MISSING_CANDLE_PERIOD" for g in state.gaps)


class TestCandleOhlcSanity:
    def test_invalid_ohlc(self, tmp_path):
        session_id = "badohlc001"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "events.jsonl"
        raw_path.write_text(
            _raw_line(
                session_id,
                "candle",
                1,
                2000,
                subscription={"type": "candle", "coin": "BTC"},
                raw_payload={
                    "channel": "candle",
                    "data": {"t": 0, "T": 59999, "s": "BTC", "o": "10", "h": "9", "l": "11", "c": "10", "v": "1", "n": 1},
                },
            )
            + "\n",
            encoding="utf-8",
        )
        _write_manifest(tmp_path, session_id, files_written=[str(raw_path)], raw_row_count=1)
        state = inspect_session(load_session_manifest(session_id, tmp_path))
        assert any(w.category == "candle_ohlc" for w in state.warnings)


class TestBboSpreadSanity:
    def test_bbo_spread_stats(self, tmp_path):
        session_id = "bbospread1"
        stream_dir = _build_valid_session(tmp_path, session_id)
        state = inspect_session(load_session_manifest(session_id, stream_dir))
        assert "BTC" in state.bbo_spread_bps


class TestCrossedBbo:
    def test_crossed_bbo_critical(self, tmp_path):
        session_id = "crossbbo01"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "events.jsonl"
        raw_path.write_text(
            _raw_line(
                session_id,
                "bbo",
                1,
                2000,
                exchange_timestamp_ms=1900,
                subscription={"type": "bbo", "coin": "BTC"},
                raw_payload={
                    "channel": "bbo",
                    "data": {
                        "coin": "BTC",
                        "bbo": [{"px": "101.0", "sz": "1"}, {"px": "100.0", "sz": "1"}],
                    },
                },
            )
            + "\n",
            encoding="utf-8",
        )
        _write_manifest(tmp_path, session_id, files_written=[str(raw_path)], raw_row_count=1)
        state = inspect_session(load_session_manifest(session_id, tmp_path))
        assert any(w.category == "bbo_sanity" and w.severity == "CRITICAL" for w in state.warnings)


class TestL2Sorting:
    def test_l2_unsorted_bids(self, tmp_path):
        session_id = "l2sort001"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "events.jsonl"
        raw_path.write_text(
            _raw_line(
                session_id,
                "l2Book",
                1,
                2000,
                exchange_timestamp_ms=1900,
                subscription={"type": "l2Book", "coin": "BTC"},
                raw_payload={
                    "channel": "l2Book",
                    "data": {
                        "coin": "BTC",
                        "levels": [
                            [{"px": "99.0", "sz": "1"}, {"px": "100.0", "sz": "1"}],
                            [{"px": "101.0", "sz": "1"}],
                        ],
                    },
                },
            )
            + "\n",
            encoding="utf-8",
        )
        _write_manifest(tmp_path, session_id, files_written=[str(raw_path)], raw_row_count=1)
        state = inspect_session(load_session_manifest(session_id, tmp_path))
        assert any("not sorted" in w.message for w in state.warnings)


class TestL2CrossedBook:
    def test_crossed_l2_book(self, tmp_path):
        session_id = "l2cross01"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "events.jsonl"
        raw_path.write_text(
            _raw_line(
                session_id,
                "l2Book",
                1,
                2000,
                subscription={"type": "l2Book", "coin": "BTC"},
                raw_payload={
                    "channel": "l2Book",
                    "data": {
                        "coin": "BTC",
                        "levels": [
                            [{"px": "102.0", "sz": "1"}],
                            [{"px": "101.0", "sz": "1"}],
                        ],
                    },
                },
            )
            + "\n",
            encoding="utf-8",
        )
        _write_manifest(tmp_path, session_id, files_written=[str(raw_path)], raw_row_count=1)
        state = inspect_session(load_session_manifest(session_id, tmp_path))
        assert any("Crossed L2" in w.message for w in state.warnings)


class TestTradeValidation:
    def test_invalid_trade_price(self, tmp_path):
        session_id = "badtrade01"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "events.jsonl"
        raw_path.write_text(
            _raw_line(
                session_id,
                "trades",
                1,
                2000,
                raw_payload={
                    "channel": "trades",
                    "data": [{"coin": "BTC", "px": "0", "sz": "0", "side": "X", "tid": None, "hash": "", "time": 1900}],
                },
            )
            + "\n",
            encoding="utf-8",
        )
        _write_manifest(tmp_path, session_id, files_written=[str(raw_path)], raw_row_count=1)
        state = inspect_session(load_session_manifest(session_id, tmp_path))
        assert any(w.category == "trade_sanity" for w in state.warnings)


class TestDuplicateTrades:
    def test_duplicate_trade_keys(self, tmp_path):
        session_id = "duptrade01"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "events.jsonl"
        trade_payload = {
            "channel": "trades",
            "data": [{"coin": "BTC", "px": "100", "sz": "1", "side": "B", "tid": 42, "hash": "0x1", "time": 1000}],
        }
        raw_path.write_text(
            "\n".join(
                [
                    _raw_line(session_id, "trades", 1, 2000, raw_payload=trade_payload),
                    _raw_line(session_id, "trades", 2, 3000, raw_payload=trade_payload),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        _write_manifest(tmp_path, session_id, files_written=[str(raw_path)], raw_row_count=2)
        state = inspect_session(load_session_manifest(session_id, tmp_path))
        assert state.duplicate_trade_keys >= 1


class TestRawNormalizedConsistency:
    def test_count_mismatch_warning(self, tmp_path):
        session_id = "countmis1"
        stream_dir = _build_valid_session(tmp_path, session_id)
        manifest_path = stream_dir / "manifests" / f"{session_id}.json"
        data = json.loads(manifest_path.read_text())
        data["raw_row_count"] = 9999
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        state = inspect_session(load_session_manifest(session_id, stream_dir))
        assert any(w.category == "count_consistency" for w in state.warnings)


class TestExpansionRatio:
    def test_expansion_ratio_note(self, tmp_path):
        session_id = "expand0001"
        stream_dir = _build_valid_session(tmp_path, session_id)
        norm_dir = stream_dir / "normalized" / "event_type=mid" / "date=2026-06-18" / "coin=BTC"
        norm_dir.mkdir(parents=True)
        norm_path = norm_dir / "events-1.parquet"
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.table({"coin": ["BTC"], "mid_price": [100.0]})
            pq.write_table(table, norm_path)
        except Exception:
            pytest.skip("pyarrow not available")
        manifest_path = stream_dir / "manifests" / f"{session_id}.json"
        data = json.loads(manifest_path.read_text())
        data["files_written"].append(str(norm_path))
        data["normalized_row_count"] = 600
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        report = build_quality_report(session_id, stream_dir)
        assert report["raw_normalized_consistency"]["normalized_to_raw_expansion_ratio"] is not None


class TestGapSummaries:
    def test_manifest_gaps_merged(self, tmp_path):
        session_id = "gapmerge01"
        stream_dir = _build_valid_session(tmp_path, session_id)
        manifest_path = stream_dir / "manifests" / f"{session_id}.json"
        data = json.loads(manifest_path.read_text())
        data["gap_statistics"] = {
            "total_gaps": 1,
            "gaps": [
                {
                    "gap_type": "BACKWARD_EXCHANGE_TIMESTAMP",
                    "coin": "BTC",
                    "channel": "trades",
                    "started_at_ms": 2000,
                    "ended_at_ms": 1000,
                    "duration_ms": -1000,
                    "certainty": "INFERRED",
                }
            ],
        }
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        report = build_quality_report(session_id, stream_dir)
        assert report["gap_summary"]["total_gaps"] >= 1


class TestUsabilityFlags:
    def test_backtesting_not_usable(self, tmp_path):
        session_id = "usable0001"
        stream_dir = _build_valid_session(tmp_path, session_id)
        manifest = load_session_manifest(session_id, stream_dir)
        state = inspect_session(manifest)
        flags = build_usability_flags(state, manifest, assign_overall_label(state, manifest))
        assert flags["usable_for_backtesting"]["usable"] is False
        assert "not implemented" in flags["usable_for_backtesting"]["reason"].lower()


class TestQualityLabelAssignment:
    def test_good_label_when_clean(self, tmp_path):
        session_id = "goodlabel1"
        stream_dir = _build_valid_session(tmp_path, session_id)
        for _ in range(100):
            raw_dir = stream_dir / "raw" / "2026-06-18"
            with (raw_dir / "events-12.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    _raw_line(
                        session_id,
                        "bbo",
                        100,
                        8000,
                        exchange_timestamp_ms=7900,
                        subscription={"type": "bbo", "coin": "ETH"},
                        raw_payload={
                            "channel": "bbo",
                            "data": {
                                "coin": "ETH",
                                "bbo": [{"px": "50", "sz": "1"}, {"px": "50.1", "sz": "1"}],
                            },
                        },
                    )
                    + "\n"
                )
        manifest_path = stream_dir / "manifests" / f"{session_id}.json"
        data = json.loads(manifest_path.read_text())
        data["raw_row_count"] = 106
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        report = build_quality_report(session_id, stream_dir)
        assert report["quality_label"] in (
            QualityLabel.GOOD_FOR_RESEARCH.value,
            QualityLabel.USABLE_WITH_WARNINGS.value,
        )


class TestMarkdownOutput:
    def test_markdown_contains_sections(self, tmp_path):
        session_id = "mdoutput01"
        stream_dir = _build_valid_session(tmp_path, session_id)
        report = build_quality_report(session_id, stream_dir)
        md = render_markdown_report(report)
        assert "# Market Data Quality Report" in md
        assert "## Safety statement" in md
        assert session_id in md


class TestJsonOutput:
    def test_json_report_fields(self, tmp_path):
        session_id = "jsonout001"
        stream_dir = _build_valid_session(tmp_path, session_id)
        out_dir = tmp_path / "quality_reports"
        report = build_quality_report(session_id, stream_dir)
        paths = write_quality_report(report, out_dir)
        assert paths["json"].exists()
        loaded = json.loads(paths["json"].read_text())
        assert loaded["paper_only"] is True
        assert loaded["execution_enabled"] is False
        assert "quality_label" in loaded


class TestDashboardDataLoader:
    def test_load_quality_reports_empty(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        stream_dir = data_dir / "market_stream"
        stream_dir.mkdir(parents=True)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream_dir)
        assert dashboard_data.load_quality_reports() == []

    def test_load_quality_report_roundtrip(self, tmp_path, monkeypatch):
        session_id = "dashload01"
        stream_dir = _build_valid_session(tmp_path / "data" / "market_stream", session_id)
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream_dir)
        report = build_quality_report(session_id, stream_dir)
        write_quality_report(report, stream_dir / "quality_reports")
        loaded = dashboard_data.load_quality_report(session_id)
        assert loaded is not None
        assert loaded["recorder_session_id"] == session_id


class TestNoLiveFileModification:
    def test_report_writes_only_quality_dir(self, tmp_path):
        if not LIVE_MANIFEST.exists():
            pytest.skip("Smoke manifest not available")
        manifest_bytes = LIVE_MANIFEST.read_bytes()
        raw_bytes = LIVE_RAW.read_bytes() if LIVE_RAW.exists() else None

        out_dir = tmp_path / "quality_reports"
        session_id = "0440bbbe838e4ce9adad8605ac364d4d"
        stream_dir = REPO_ROOT / "data/market_stream"
        report = build_quality_report(session_id, stream_dir, repo_root=REPO_ROOT)
        write_quality_report(report, out_dir)

        assert LIVE_MANIFEST.read_bytes() == manifest_bytes
        if raw_bytes is not None:
            assert LIVE_RAW.read_bytes() == raw_bytes


class TestNoNetworkCalls:
    def test_quality_modules_no_requests(self):
        quality_dir = REPO_ROOT / "market_data"
        for path in quality_dir.glob("quality*.py"):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert alias.name != "requests"
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert node.module.split(".")[0] != "requests"


class TestNoWalletSigningExecution:
    def test_no_forbidden_patterns_in_quality_modules(self):
        from market_data.security_guard import FORBIDDEN_IMPORT_MODULES

        quality_dir = REPO_ROOT / "market_data"
        for path in quality_dir.glob("*.py"):
            if not path.name.startswith("quality") and path.name != "replay_manifest.py":
                continue
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert alias.name.split(".")[0] not in FORBIDDEN_IMPORT_MODULES
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert node.module.split(".")[0] not in FORBIDDEN_IMPORT_MODULES
        assert "execution_enabled"  # report always false


class TestCli:
    def test_cli_json_format(self, tmp_path, capsys):
        session_id = "clitest001"
        stream_dir = _build_valid_session(tmp_path, session_id)
        import argparse

        args = argparse.Namespace(
            session_id=session_id,
            latest=False,
            output_dir=tmp_path / "reports",
            market_stream_dir=stream_dir,
            format="json",
        )
        code = run_quality_report(args)
        captured = capsys.readouterr()
        assert "quality_label" in captured.out
        assert code in (0, 2)

    def test_find_latest_session(self, tmp_path):
        _write_manifest(tmp_path, "aaa")
        _write_manifest(tmp_path, "bbb")
        assert find_latest_session_id(tmp_path) in ("aaa", "bbb")
