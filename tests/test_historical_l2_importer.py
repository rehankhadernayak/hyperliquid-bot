"""Focused offline tests for the historical L2 importer skeleton."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from bid_ask_schema import BID_ASK_CSV_COLUMNS, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS
from historical_l2_importer import (
    DEFAULT_OUTPUT_ROOT,
    HistoricalImportReport,
    build_archive_s3_path,
    default_historical_output_path,
    import_decompressed_l2_file,
    l2_snapshot_to_bid_ask_row,
    parse_l2_snapshot_line,
    summarize_historical_coverage,
    write_historical_bid_ask_csv,
    write_historical_import_report,
)

IMPORTER_FILE = Path(__file__).resolve().parent.parent / "historical_l2_importer.py"
RESEARCH_FILE = Path(__file__).resolve().parent.parent / "historical_backfill_research.py"
BASE_TS = 1_718_000_000_000


def _l2_archive_line(
    *,
    coin: str = "SOL",
    time_ms: int = BASE_TS,
    bid_px: str = "150.0",
    ask_px: str = "150.5",
    bid_sz: str = "10.0",
    ask_sz: str = "8.0",
) -> str:
    payload = {
        "channel": "l2Book",
        "data": {
            "coin": coin,
            "time": time_ms,
            "levels": [
                [{"px": bid_px, "sz": bid_sz, "n": 2}],
                [{"px": ask_px, "sz": ask_sz, "n": 3}],
            ],
        },
    }
    return json.dumps(payload)


class TestArchivePath:
    def test_build_archive_s3_path(self) -> None:
        path = build_archive_s3_path("20230916", 9, "SOL")
        assert path == "s3://hyperliquid-archive/market_data/20230916/9/l2Book/SOL.lz4"

    def test_build_archive_s3_path_normalizes_coin(self) -> None:
        path = build_archive_s3_path("20230916", 9, " sol ")
        assert path.endswith("/SOL.lz4")


class TestParseL2Snapshot:
    def test_parse_valid_l2_snapshot(self) -> None:
        snapshot = parse_l2_snapshot_line(_l2_archive_line())
        assert snapshot is not None
        assert snapshot["timestamp_ms"] == BASE_TS
        assert snapshot["coin"] == "SOL"
        assert snapshot["best_bid"] == pytest.approx(150.0)
        assert snapshot["best_ask"] == pytest.approx(150.5)
        assert snapshot["bid_size"] == pytest.approx(10.0)
        assert snapshot["ask_size"] == pytest.approx(8.0)
        assert snapshot["source_channel"] == "historical_l2Book"

    def test_malformed_line_skipped(self) -> None:
        assert parse_l2_snapshot_line("not-json") is None
        assert parse_l2_snapshot_line("") is None
        assert parse_l2_snapshot_line('{"channel":"l2Book"}') is None

    def test_top_bid_ask_extraction(self) -> None:
        line = _l2_archive_line(
            bid_px="149.5",
            ask_px="150.25",
            bid_sz="4.0",
            ask_sz="6.0",
        )
        snapshot = parse_l2_snapshot_line(line)
        assert snapshot is not None
        assert snapshot["best_bid"] == pytest.approx(149.5)
        assert snapshot["best_ask"] == pytest.approx(150.25)
        assert snapshot["bid_size"] == pytest.approx(4.0)
        assert snapshot["ask_size"] == pytest.approx(6.0)


class TestBidAskRowSchema:
    def test_l2_snapshot_to_bid_ask_row(self) -> None:
        snapshot = parse_l2_snapshot_line(_l2_archive_line())
        assert snapshot is not None
        row = l2_snapshot_to_bid_ask_row(snapshot)
        assert row is not None
        assert list(row.keys()) == list(BID_ASK_CSV_COLUMNS)
        assert row["coin"] == "SOL"
        assert row["source_session_id"] == "historical_archive"
        assert row["source_channel"] == "historical_l2Book"
        assert row["mid_price"] == pytest.approx(150.25)
        assert row["spread"] == pytest.approx(0.5)

    def test_invalid_snapshot_returns_none(self) -> None:
        assert l2_snapshot_to_bid_ask_row({"best_bid": 151.0, "best_ask": 150.0}) is None


class TestLocalImport:
    def test_import_decompressed_l2_file_writes_csv(self, tmp_path: Path) -> None:
        input_path = tmp_path / "SOL"
        output_root = tmp_path / "bid_ask_csv"
        output_path = output_root / "historical_1s" / "SOL_1s.csv"
        lines = [
            _l2_archive_line(time_ms=BASE_TS),
            _l2_archive_line(time_ms=BASE_TS + 500, bid_px="150.1", ask_px="150.6"),
            _l2_archive_line(time_ms=BASE_TS + 2000, bid_px="150.2", ask_px="150.7"),
        ]
        input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        csv_path, stats = import_decompressed_l2_file(
            input_path,
            "SOL",
            output_path,
            output_root=output_root,
        )
        assert csv_path == output_path
        assert stats.rows_written == 3
        assert output_path.exists()

        content = output_path.read_text(encoding="utf-8").strip().splitlines()
        assert content[0].split(",") == list(BID_ASK_CSV_COLUMNS)
        assert "historical_archive" in content[1]
        assert "historical_l2Book" in content[1]

    def test_default_historical_output_path(self) -> None:
        path = default_historical_output_path("sol", output_root=DEFAULT_OUTPUT_ROOT)
        assert path == DEFAULT_OUTPUT_ROOT / "historical_1s" / "SOL_1s.csv"


class TestCoverageAndReport:
    def test_summarize_historical_coverage(self, tmp_path: Path) -> None:
        output_root = tmp_path / "bid_ask_csv"
        output_path = output_root / "historical_1s" / "SOL_1s.csv"
        rows = []
        for offset_ms in (0, 500, 3000):
            snapshot = parse_l2_snapshot_line(
                _l2_archive_line(time_ms=BASE_TS + offset_ms)
            )
            assert snapshot is not None
            row = l2_snapshot_to_bid_ask_row(snapshot)
            assert row is not None
            rows.append(row)
        write_historical_bid_ask_csv(rows, output_path, output_root=output_root)

        summary = summarize_historical_coverage(output_path)
        assert summary.row_count == 3
        assert summary.start_timestamp_ms == BASE_TS
        assert summary.end_timestamp_ms == BASE_TS + 3000
        assert summary.missing_seconds == 2
        assert summary.duplicate_timestamps == 0
        assert summary.invalid_bid_ask_rows == 0
        assert summary.source == "historical_archive_l2Book"

    def test_write_historical_import_report(self, tmp_path: Path) -> None:
        output_root = tmp_path / "bid_ask_csv"
        report = HistoricalImportReport(
            coin="SOL",
            input_path="/tmp/SOL",
            output_path=str(output_root / "historical_1s" / "SOL_1s.csv"),
            row_count=3,
            coverage_start_ms=BASE_TS,
            coverage_end_ms=BASE_TS + 3000,
            missing_seconds=2,
            duplicate_timestamps=0,
            invalid_bid_ask_rows=0,
            malformed_lines_skipped=1,
            invalid_rows_skipped=0,
        )
        report_path = write_historical_import_report(report, output_root=output_root)
        assert report_path.exists()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["coin"] == "SOL"
        assert payload["source"] == "historical_archive_l2Book"
        assert payload["download_enabled"] is False


class TestSafety:
    def test_no_network_calls(self) -> None:
        source = IMPORTER_FILE.read_text(encoding="utf-8").lower()
        for token in (
            "requests.",
            "urllib.",
            "http.client",
            "socket.",
            "websocket",
            "aiohttp",
            "httpx",
            "boto3",
            "botocore",
        ):
            assert token not in source

    def test_no_subprocess_download_calls(self) -> None:
        source = IMPORTER_FILE.read_text(encoding="utf-8").lower()
        assert "subprocess" not in source
        assert "aws s3" not in source

    def test_no_wallet_signing_order_execution_code(self) -> None:
        for path in (IMPORTER_FILE, RESEARCH_FILE):
            source = path.read_text(encoding="utf-8")
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

    def test_write_historical_bid_ask_csv_rejects_outside_output_root(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="Output path must be under"):
            write_historical_bid_ask_csv(
                [],
                tmp_path / "outside.csv",
                output_root=DEFAULT_OUTPUT_ROOT,
            )

    def test_paper_only_constants(self) -> None:
        from historical_backfill_research import DOWNLOAD_ENABLED, EXECUTION_ENABLED, PAPER_ONLY

        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False
        assert DOWNLOAD_ENABLED is False
