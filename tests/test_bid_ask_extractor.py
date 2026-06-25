"""Focused offline tests for the bid/ask local data extractor."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from bid_ask_extractor import (
    DEFAULT_OUTPUT_ROOT,
    ExtractionReport,
    discover_local_bid_ask_sources,
    extract_bid_ask_rows_from_normalized,
    extract_bid_ask_rows_from_raw_jsonl,
    extract_coin_bid_ask_1s,
    sample_to_1s,
    save_extraction_report,
    write_bid_ask_csv,
)
from bid_ask_schema import BID_ASK_CSV_COLUMNS, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS

EXTRACTOR_FILE = Path(__file__).resolve().parent.parent / "bid_ask_extractor.py"
BASE_TS = 1_718_000_000_000


def _raw_line(
    session_id: str,
    channel: str,
    seq: int,
    recv_ms: int,
    *,
    raw_payload: dict,
    exchange_timestamp_ms: int | None = None,
) -> str:
    record = {
        "schema_version": "1.0.0",
        "recorder_session_id": session_id,
        "network": "mainnet",
        "channel": channel,
        "subscription": {"type": channel, "coin": "BTC"},
        "exchange_timestamp_ms": exchange_timestamp_ms,
        "received_timestamp_ms": recv_ms,
        "sequence": seq,
        "raw_payload": raw_payload,
        "source": "hyperliquid_websocket",
    }
    return json.dumps(record)


def _write_raw_bbo_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bbo_payload(
    *,
    coin: str = "BTC",
    time_ms: int = BASE_TS,
    bid_px: str = "63747.0",
    ask_px: str = "63748.0",
    bid_sz: str = "1.5",
    ask_sz: str = "0.8",
) -> dict:
    return {
        "channel": "bbo",
        "data": {
            "coin": coin,
            "time": time_ms,
            "bbo": [
                {"px": bid_px, "sz": bid_sz},
                {"px": ask_px, "sz": ask_sz},
            ],
        },
    }


class TestRawJsonlParse:
    def test_extract_bbo_rows_from_raw_jsonl(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw" / "2026-06-18" / "events-12.jsonl"
        lines = [
            _raw_line(
                "session-1",
                "bbo",
                1,
                BASE_TS,
                exchange_timestamp_ms=BASE_TS,
                raw_payload=_bbo_payload(time_ms=BASE_TS),
            ),
            _raw_line(
                "session-1",
                "bbo",
                2,
                BASE_TS + 500,
                exchange_timestamp_ms=BASE_TS + 500,
                raw_payload=_bbo_payload(time_ms=BASE_TS + 500, bid_px="63748.0", ask_px="63749.0"),
            ),
        ]
        _write_raw_bbo_file(raw_path, lines)

        rows, stats = extract_bid_ask_rows_from_raw_jsonl(raw_path, coins=("BTC",))
        assert stats.rows_extracted == 2
        assert stats.invalid_rows_skipped == 0
        assert rows[0]["coin"] == "BTC"
        assert rows[0]["best_bid"] == pytest.approx(63747.0)
        assert rows[0]["best_ask"] == pytest.approx(63748.0)
        assert rows[0]["source_session_id"] == "session-1"
        assert rows[0]["source_channel"] == "bbo"

    def test_invalid_bid_ask_skipped(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw" / "2026-06-18" / "events-12.jsonl"
        lines = [
            _raw_line(
                "session-1",
                "bbo",
                1,
                BASE_TS,
                raw_payload=_bbo_payload(bid_px="63750.0", ask_px="63749.0"),
            ),
            _raw_line(
                "session-1",
                "bbo",
                2,
                BASE_TS + 1000,
                raw_payload=_bbo_payload(bid_px="63747.0", ask_px="63748.0"),
            ),
        ]
        _write_raw_bbo_file(raw_path, lines)

        rows, stats = extract_bid_ask_rows_from_raw_jsonl(raw_path)
        assert len(rows) == 1
        assert stats.invalid_rows_skipped == 1


class TestNormalizedParse:
    def test_extract_bbo_rows_from_normalized_jsonl(self, tmp_path: Path) -> None:
        normalized_path = (
            tmp_path
            / "normalized"
            / "event_type=bbo"
            / "date=2024-06-10"
            / "coin=BTC"
            / "events-1.jsonl"
        )
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "coin": "BTC",
            "best_bid_price": 63747.0,
            "best_bid_size": 1.5,
            "best_ask_price": 63748.0,
            "best_ask_size": 0.8,
            "spread": 1.0,
            "spread_bps": 0.15,
            "exchange_timestamp_ms": BASE_TS,
            "received_timestamp_ms": BASE_TS + 10,
        }
        normalized_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

        rows, stats = extract_bid_ask_rows_from_normalized(
            normalized_path,
            coins=("BTC",),
            source_channel="bbo",
        )
        assert stats.rows_extracted == 1
        assert rows[0]["best_bid"] == pytest.approx(63747.0)
        assert rows[0]["best_ask"] == pytest.approx(63748.0)
        assert rows[0]["source_channel"] == "bbo"


class TestSampleTo1s:
    def test_last_observation_per_second(self) -> None:
        rows = [
            {
                "timestamp_ms": BASE_TS,
                "best_bid": 100.0,
                "best_ask": 101.0,
            },
            {
                "timestamp_ms": BASE_TS + 500,
                "best_bid": 102.0,
                "best_ask": 103.0,
            },
            {
                "timestamp_ms": BASE_TS + 2000,
                "best_bid": 104.0,
                "best_ask": 105.0,
            },
        ]
        result = sample_to_1s(rows)
        assert len(result.rows) == 2
        assert result.rows[0]["best_bid"] == pytest.approx(102.0)
        assert result.rows[1]["best_bid"] == pytest.approx(104.0)

    def test_missing_seconds_count(self) -> None:
        rows = [
            {"timestamp_ms": BASE_TS, "best_bid": 100.0, "best_ask": 101.0},
            {"timestamp_ms": BASE_TS + 3000, "best_bid": 102.0, "best_ask": 103.0},
        ]
        result = sample_to_1s(rows)
        assert result.missing_seconds == 2
        assert result.duplicate_timestamps == 0


class TestWriteCsvAndReport:
    def test_output_csv_schema(self, tmp_path: Path) -> None:
        output_root = tmp_path / "bid_ask_csv"
        output_path = output_root / "1s" / "BTC_1s.csv"
        row = {
            "timestamp_ms": BASE_TS,
            "timestamp_iso": "2024-06-10T10:13:20Z",
            "coin": "BTC",
            "best_bid": 63747.0,
            "best_ask": 63748.0,
            "mid_price": 63747.5,
            "spread": 1.0,
            "spread_bps": 0.15,
            "bid_size": 1.5,
            "ask_size": 0.8,
            "bbo_imbalance": 0.304,
            "source_session_id": "session-1",
            "source_channel": "bbo",
        }
        write_bid_ask_csv([row], output_path, output_root=output_root)

        content = output_path.read_text(encoding="utf-8").strip().splitlines()
        header = content[0].split(",")
        assert header == list(BID_ASK_CSV_COLUMNS)
        assert "BTC" in content[1]

    def test_report_creation(self, tmp_path: Path) -> None:
        output_root = tmp_path / "bid_ask_csv"
        report = ExtractionReport(
            coin="BTC",
            source_files=["/tmp/raw/events.jsonl"],
            output_path=str(output_root / "1s" / "BTC_1s.csv"),
            row_count=3,
            coverage_start_ms=BASE_TS,
            coverage_end_ms=BASE_TS + 2000,
            missing_seconds=1,
            duplicate_timestamps=0,
            invalid_rows_skipped=0,
            locally_collected_data_only=True,
        )
        report_path = save_extraction_report(report, output_root=output_root)
        assert report_path.exists()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["coin"] == "BTC"
        assert payload["locally_collected_data_only"] is True
        assert payload["missing_seconds"] == 1


class TestDiscoveryAndPipeline:
    def test_discover_local_bid_ask_sources(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw" / "2026-06-18" / "events-12.jsonl"
        _write_raw_bbo_file(raw_path, [_raw_line("session-1", "bbo", 1, BASE_TS, raw_payload=_bbo_payload())])
        normalized_path = (
            tmp_path
            / "normalized"
            / "event_type=bbo"
            / "date=2024-06-10"
            / "coin=BTC"
            / "events-1.jsonl"
        )
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_path.write_text("{}\n", encoding="utf-8")

        sources = discover_local_bid_ask_sources(tmp_path, coins=("BTC",))
        kinds = {source.source_kind for source in sources}
        assert "raw_jsonl" in kinds
        assert "normalized_jsonl" in kinds

    def test_end_to_end_extraction(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw" / "2026-06-18" / "events-12.jsonl"
        lines = [
            _raw_line(
                "session-1",
                "bbo",
                1,
                BASE_TS,
                raw_payload=_bbo_payload(time_ms=BASE_TS),
            ),
            _raw_line(
                "session-1",
                "bbo",
                2,
                BASE_TS + 1500,
                raw_payload=_bbo_payload(time_ms=BASE_TS + 1500, bid_px="63748.0", ask_px="63749.0"),
            ),
            _raw_line(
                "session-1",
                "bbo",
                3,
                BASE_TS + 3000,
                raw_payload=_bbo_payload(time_ms=BASE_TS + 3000, bid_px="63749.0", ask_px="63750.0"),
            ),
        ]
        _write_raw_bbo_file(raw_path, lines)

        output_root = tmp_path / "bid_ask_csv"
        report = extract_coin_bid_ask_1s(
            "BTC",
            market_stream_dir=tmp_path,
            output_root=output_root,
        )
        csv_path = output_root / "1s" / "BTC_1s.csv"
        assert csv_path.exists()
        assert report.row_count == 3
        assert report.missing_seconds == 1
        assert len(list((output_root / "reports").glob("*.json"))) == 1


class TestSafety:
    def test_raw_source_files_not_modified(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw" / "2026-06-18" / "events-12.jsonl"
        _write_raw_bbo_file(raw_path, [_raw_line("session-1", "bbo", 1, BASE_TS, raw_payload=_bbo_payload())])
        before = raw_path.read_bytes()
        before_mtime = raw_path.stat().st_mtime

        extract_bid_ask_rows_from_raw_jsonl(raw_path)
        extract_coin_bid_ask_1s(
            "BTC",
            market_stream_dir=tmp_path,
            output_root=tmp_path / "bid_ask_csv",
        )

        assert raw_path.read_bytes() == before
        assert raw_path.stat().st_mtime == before_mtime

    def test_write_bid_ask_csv_rejects_outside_output_root(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Output path must be under"):
            write_bid_ask_csv([], tmp_path / "outside.csv", output_root=DEFAULT_OUTPUT_ROOT)

    def test_no_network_calls(self) -> None:
        source = EXTRACTOR_FILE.read_text(encoding="utf-8").lower()
        forbidden_network_tokens = (
            "requests.",
            "urllib.",
            "http.client",
            "socket.",
            "websocket",
            "aiohttp",
            "httpx",
        )
        for token in forbidden_network_tokens:
            assert token not in source

    def test_no_wallet_signing_order_execution_code(self) -> None:
        source = EXTRACTOR_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        lowered = source.lower()
        for pattern in FORBIDDEN_SOURCE_PATTERNS:
            assert pattern not in lowered
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES

    def test_output_only_under_data_bid_ask_csv(self, tmp_path: Path) -> None:
        output_root = tmp_path / "bid_ask_csv"
        report = ExtractionReport(
            coin="ETH",
            source_files=[],
            output_path=str(output_root / "1s" / "ETH_1s.csv"),
            row_count=0,
            coverage_start_ms=None,
            coverage_end_ms=None,
            missing_seconds=0,
            duplicate_timestamps=0,
            invalid_rows_skipped=0,
        )
        save_extraction_report(report, output_root=output_root)
        assert (output_root / "reports").exists()
        assert not (tmp_path / "1s").exists()
