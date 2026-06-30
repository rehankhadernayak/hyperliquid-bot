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
    main,
    merge_historical_bid_ask_rows,
    parse_l2_snapshot_line,
    read_historical_bid_ask_csv,
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
        assert stats.new_rows_imported == 3
        assert stats.rows_written == 3
        assert stats.rows_after_merge == 3
        assert stats.existing_rows_before == 0
        assert stats.mode == "overwrite"
        assert output_path.exists()

        content = output_path.read_text(encoding="utf-8").strip().splitlines()
        assert content[0].split(",") == list(BID_ASK_CSV_COLUMNS)
        assert "historical_archive" in content[1]
        assert "historical_l2Book" in content[1]

    def test_second_import_appends_to_same_coin_csv(self, tmp_path: Path) -> None:
        output_root = tmp_path / "bid_ask_csv"
        output_path = output_root / "historical_1s" / "SOL_1s.csv"

        hour_one = tmp_path / "hour1"
        hour_one.write_text(
            "\n".join(
                [
                    _l2_archive_line(time_ms=BASE_TS),
                    _l2_archive_line(time_ms=BASE_TS + 1000),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        hour_two = tmp_path / "hour2"
        hour_two.write_text(
            "\n".join(
                [
                    _l2_archive_line(time_ms=BASE_TS + 3600_000, bid_px="151.0", ask_px="151.5"),
                    _l2_archive_line(time_ms=BASE_TS + 3601_000, bid_px="151.1", ask_px="151.6"),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        _, stats_one = import_decompressed_l2_file(
            hour_one, "SOL", output_path, output_root=output_root
        )
        assert stats_one.new_rows_imported == 2
        assert stats_one.rows_after_merge == 2

        _, stats_two = import_decompressed_l2_file(
            hour_two, "SOL", output_path, output_root=output_root
        )
        assert stats_two.mode == "append_merge"
        assert stats_two.existing_rows_before == 2
        assert stats_two.new_rows_imported == 2
        assert stats_two.rows_after_merge == 4
        assert stats_two.duplicate_rows_removed == 0

        summary = summarize_historical_coverage(output_path)
        assert summary.row_count == 4

    def test_duplicate_timestamp_rows_removed_on_merge(self, tmp_path: Path) -> None:
        output_root = tmp_path / "bid_ask_csv"
        output_path = output_root / "historical_1s" / "SOL_1s.csv"

        first = tmp_path / "first"
        first.write_text(_l2_archive_line(time_ms=BASE_TS, bid_px="150.0", ask_px="150.5") + "\n")
        second = tmp_path / "second"
        second.write_text(
            _l2_archive_line(time_ms=BASE_TS, bid_px="160.0", ask_px="160.5") + "\n"
        )

        import_decompressed_l2_file(first, "SOL", output_path, output_root=output_root)
        _, stats = import_decompressed_l2_file(
            second, "SOL", output_path, output_root=output_root
        )
        assert stats.duplicate_rows_removed == 1
        assert stats.rows_after_merge == 1

        rows = read_historical_bid_ask_csv(output_path)
        assert len(rows) == 1
        assert float(rows[0]["best_bid"]) == pytest.approx(160.0)
        assert float(rows[0]["best_ask"]) == pytest.approx(160.5)

    def test_rows_sorted_ascending_after_merge(self, tmp_path: Path) -> None:
        output_root = tmp_path / "bid_ask_csv"
        output_path = output_root / "historical_1s" / "SOL_1s.csv"

        first = tmp_path / "first"
        first.write_text(
            "\n".join(
                [
                    _l2_archive_line(time_ms=BASE_TS + 3000),
                    _l2_archive_line(time_ms=BASE_TS),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        second = tmp_path / "second"
        second.write_text(
            "\n".join(
                [
                    _l2_archive_line(time_ms=BASE_TS + 2000),
                    _l2_archive_line(time_ms=BASE_TS + 1000),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        import_decompressed_l2_file(first, "SOL", output_path, output_root=output_root)
        import_decompressed_l2_file(second, "SOL", output_path, output_root=output_root)

        rows = read_historical_bid_ask_csv(output_path)
        timestamps = [int(row["timestamp_ms"]) for row in rows]
        assert timestamps == sorted(timestamps)
        assert timestamps == [BASE_TS, BASE_TS + 1000, BASE_TS + 2000, BASE_TS + 3000]

    def test_overwrite_replaces_existing_csv(self, tmp_path: Path) -> None:
        output_root = tmp_path / "bid_ask_csv"
        output_path = output_root / "historical_1s" / "SOL_1s.csv"

        first = tmp_path / "first"
        first.write_text(
            "\n".join(
                [
                    _l2_archive_line(time_ms=BASE_TS),
                    _l2_archive_line(time_ms=BASE_TS + 1000),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        second = tmp_path / "second"
        second.write_text(_l2_archive_line(time_ms=BASE_TS + 5000) + "\n", encoding="utf-8")

        import_decompressed_l2_file(first, "SOL", output_path, output_root=output_root)
        _, stats = import_decompressed_l2_file(
            second,
            "SOL",
            output_path,
            output_root=output_root,
            overwrite=True,
        )
        assert stats.mode == "overwrite"
        assert stats.existing_rows_before == 0
        assert stats.new_rows_imported == 1
        assert stats.rows_after_merge == 1

        summary = summarize_historical_coverage(output_path)
        assert summary.row_count == 1
        assert summary.start_timestamp_ms == BASE_TS + 5000

    def test_schema_order_preserved_after_merge(self, tmp_path: Path) -> None:
        output_root = tmp_path / "bid_ask_csv"
        output_path = output_root / "historical_1s" / "SOL_1s.csv"

        first = tmp_path / "first"
        first.write_text(_l2_archive_line(time_ms=BASE_TS) + "\n")
        second = tmp_path / "second"
        second.write_text(_l2_archive_line(time_ms=BASE_TS + 1000) + "\n")

        import_decompressed_l2_file(first, "SOL", output_path, output_root=output_root)
        import_decompressed_l2_file(second, "SOL", output_path, output_root=output_root)

        header = output_path.read_text(encoding="utf-8").splitlines()[0]
        assert header.split(",") == list(BID_ASK_CSV_COLUMNS)

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
            mode="append_merge",
            existing_rows_before=2,
            new_rows_imported=3,
            rows_after_merge=5,
            duplicate_rows_removed=0,
            dedupe_key="timestamp_ms",
        )
        report_path = write_historical_import_report(report, output_root=output_root)
        assert report_path.exists()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["coin"] == "SOL"
        assert payload["source"] == "historical_archive_l2Book"
        assert payload["download_enabled"] is False
        assert payload["paper_only"] is True
        assert payload["execution_enabled"] is False
        assert payload["mode"] == "append_merge"
        assert payload["existing_rows_before"] == 2
        assert payload["new_rows_imported"] == 3
        assert payload["rows_after_merge"] == 5
        assert payload["duplicate_rows_removed"] == 0
        assert payload["dedupe_key"] == "timestamp_ms"

    def test_invalid_bid_ask_detection_still_works(self, tmp_path: Path) -> None:
        output_root = tmp_path / "bid_ask_csv"
        output_path = output_root / "historical_1s" / "SOL_1s.csv"
        rows = []
        for offset_ms in (0, 1000):
            snapshot = parse_l2_snapshot_line(
                _l2_archive_line(time_ms=BASE_TS + offset_ms)
            )
            assert snapshot is not None
            row = l2_snapshot_to_bid_ask_row(snapshot)
            assert row is not None
            rows.append(row)
        write_historical_bid_ask_csv(rows, output_path, output_root=output_root)

        bad_row = dict(rows[0])
        bad_row["best_bid"] = "200.0"
        bad_row["best_ask"] = "150.0"
        with output_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(",".join(str(bad_row[column]) for column in BID_ASK_CSV_COLUMNS) + "\n")

        summary = summarize_historical_coverage(output_path)
        assert summary.invalid_bid_ask_rows == 1
        assert summary.row_count == 3


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


class TestSummarizeCli:
    def test_summarize_cli_reads_local_csv_only(self, tmp_path: Path, capsys) -> None:
        output_root = tmp_path / "bid_ask_csv"
        output_path = output_root / "historical_1s" / "SOL_1s.csv"
        snapshot = parse_l2_snapshot_line(_l2_archive_line())
        assert snapshot is not None
        row = l2_snapshot_to_bid_ask_row(snapshot)
        assert row is not None
        write_historical_bid_ask_csv([row], output_path, output_root=output_root)

        exit_code = main(
            [
                "summarize",
                "--csv",
                str(output_path),
            ]
        )
        captured = capsys.readouterr().out

        assert exit_code == 0
        assert "row_count: 1" in captured
        assert f"start_timestamp_ms: {BASE_TS}" in captured
        assert f"end_timestamp_ms: {BASE_TS}" in captured
        assert "missing_seconds: 0" in captured
        assert "duplicate_timestamps: 0" in captured
        assert "invalid_bid_ask_rows: 0" in captured
        assert "historical_archive_l2Book" in captured

    def test_summarize_cli_with_coin_resolves_default_path(
        self, tmp_path: Path, capsys
    ) -> None:
        output_root = tmp_path / "bid_ask_csv"
        output_path = output_root / "historical_1s" / "SOL_1s.csv"
        snapshot = parse_l2_snapshot_line(_l2_archive_line())
        assert snapshot is not None
        row = l2_snapshot_to_bid_ask_row(snapshot)
        assert row is not None
        write_historical_bid_ask_csv([row], output_path, output_root=output_root)

        exit_code = main(
            [
                "summarize",
                "--coin",
                "SOL",
                "--output-root",
                str(output_root),
            ]
        )
        captured = capsys.readouterr().out

        assert exit_code == 0
        assert "row_count: 1" in captured
        assert str(output_path.resolve()) in captured


class TestImportCli:
    def test_import_cli_merge_output(self, tmp_path: Path, capsys) -> None:
        output_root = tmp_path / "bid_ask_csv"
        output_path = output_root / "historical_1s" / "SOL_1s.csv"
        hour_one = tmp_path / "hour1"
        hour_one.write_text(_l2_archive_line(time_ms=BASE_TS) + "\n")
        hour_two = tmp_path / "hour2"
        hour_two.write_text(_l2_archive_line(time_ms=BASE_TS + 3600_000) + "\n")

        assert (
            main(
                [
                    "import-local",
                    "--input",
                    str(hour_one),
                    "--coin",
                    "SOL",
                    "--output",
                    str(output_path),
                    "--output-root",
                    str(output_root),
                ]
            )
            == 0
        )
        exit_code = main(
            [
                "import-local",
                "--input",
                str(hour_two),
                "--coin",
                "SOL",
                "--output",
                str(output_path),
                "--output-root",
                str(output_root),
            ]
        )
        captured = capsys.readouterr().out

        assert exit_code == 0
        assert "Imported 1 new rows" in captured
        assert "Existing rows before: 1" in captured
        assert "Rows after merge: 2" in captured
        assert "Duplicate rows removed: 0" in captured
        assert str(output_path.resolve()) in captured

    def test_summarize_reports_merged_row_count(self, tmp_path: Path, capsys) -> None:
        output_root = tmp_path / "bid_ask_csv"
        output_path = output_root / "historical_1s" / "SOL_1s.csv"
        hour_one = tmp_path / "hour1"
        hour_one.write_text(_l2_archive_line(time_ms=BASE_TS) + "\n")
        hour_two = tmp_path / "hour2"
        hour_two.write_text(_l2_archive_line(time_ms=BASE_TS + 3600_000) + "\n")

        import_decompressed_l2_file(hour_one, "SOL", output_path, output_root=output_root)
        import_decompressed_l2_file(hour_two, "SOL", output_path, output_root=output_root)

        exit_code = main(
            [
                "summarize",
                "--csv",
                str(output_path),
            ]
        )
        captured = capsys.readouterr().out

        assert exit_code == 0
        assert "row_count: 2" in captured


class TestMergeHelpers:
    def test_merge_historical_bid_ask_rows_prefers_new_rows(self) -> None:
        existing = [{"timestamp_ms": "1000", "best_bid": "1.0"}]
        new = [{"timestamp_ms": "1000", "best_bid": "2.0"}]
        merged, removed = merge_historical_bid_ask_rows(existing, new)
        assert removed == 1
        assert merged[0]["best_bid"] == "2.0"
