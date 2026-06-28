"""Offline tests for the historical L2 backfill planner."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY
from historical_l2_backfill_planner import (
    ARCHIVE_WARNING,
    BackfillPlanConfig,
    NOTHING_DOWNLOADED_WARNING,
    build_l2_s3_path,
    build_local_decompressed_path,
    build_local_download_path,
    generate_backfill_manifest,
    iter_dates,
    main,
    validate_date_yyyymmdd,
    write_manifest_json,
    write_manifest_markdown,
)

PLANNER_FILE = Path(__file__).resolve().parent.parent / "historical_l2_backfill_planner.py"


class TestDateValidation:
    def test_validate_date_yyyymmdd_accepts_valid_date(self) -> None:
        assert validate_date_yyyymmdd("20230916") == "20230916"

    def test_validate_date_yyyymmdd_rejects_bad_format(self) -> None:
        with pytest.raises(ValueError, match="YYYYMMDD"):
            validate_date_yyyymmdd("2023-09-16")

    def test_validate_date_yyyymmdd_rejects_invalid_calendar_date(self) -> None:
        with pytest.raises(ValueError, match="invalid calendar date"):
            validate_date_yyyymmdd("20230230")


class TestDateIteration:
    def test_iter_dates_single_day(self) -> None:
        assert list(iter_dates("20230916", "20230916")) == ["20230916"]

    def test_iter_dates_inclusive_range(self) -> None:
        assert list(iter_dates("20230916", "20230918")) == [
            "20230916",
            "20230917",
            "20230918",
        ]

    def test_iter_dates_rejects_end_before_start(self) -> None:
        with pytest.raises(ValueError, match="before start_date"):
            list(iter_dates("20230918", "20230916"))


class TestPathBuilders:
    def test_build_l2_s3_path_format(self) -> None:
        path = build_l2_s3_path("20230916", 9, "SOL")
        assert path == (
            "s3://hyperliquid-archive/market_data/20230916/9/l2Book/SOL.lz4"
        )

    def test_build_local_download_path_format(self) -> None:
        path = build_local_download_path("20230916", 9, "SOL")
        assert path == "downloads/hyperliquid/20230916/9/SOL.lz4"

    def test_build_local_decompressed_path_format(self) -> None:
        path = build_local_decompressed_path("20230916", 9, "SOL")
        assert path == "downloads/hyperliquid/20230916/9/SOL.jsonl"


class TestManifestGeneration:
    def _config(self, tmp_path: Path, *, start: str, end: str) -> BackfillPlanConfig:
        return BackfillPlanConfig(
            start_date=start,
            end_date=end,
            coins=["BTC", "ETH", "SOL"],
            output_dir=tmp_path,
        )

    def test_one_day_three_coins_has_72_files(self, tmp_path: Path) -> None:
        manifest = generate_backfill_manifest(self._config(tmp_path, start="20230916", end="20230916"))
        assert manifest["total_days"] == 1
        assert manifest["total_hours"] == 24
        assert manifest["total_files"] == 72
        assert manifest["estimated_files_by_coin"] == {"BTC": 24, "ETH": 24, "SOL": 24}
        assert len(manifest["files"]) == 72

    def test_seven_day_three_coins_has_504_files(self, tmp_path: Path) -> None:
        manifest = generate_backfill_manifest(self._config(tmp_path, start="20230916", end="20230922"))
        assert manifest["total_days"] == 7
        assert manifest["total_hours"] == 168
        assert manifest["total_files"] == 504
        assert manifest["estimated_files_by_coin"] == {"BTC": 168, "ETH": 168, "SOL": 168}

    def test_manifest_entry_fields(self, tmp_path: Path) -> None:
        manifest = generate_backfill_manifest(self._config(tmp_path, start="20230916", end="20230916"))
        entry = manifest["files"][0]
        assert entry == {
            "date": "20230916",
            "hour": "0",
            "coin": "BTC",
            "s3_path": "s3://hyperliquid-archive/market_data/20230916/0/l2Book/BTC.lz4",
            "local_lz4_path": "downloads/hyperliquid/20230916/0/BTC.lz4",
            "local_jsonl_path": "downloads/hyperliquid/20230916/0/BTC.jsonl",
            "status": "planned",
        }

    def test_paper_only_and_execution_disabled(self, tmp_path: Path) -> None:
        manifest = generate_backfill_manifest(self._config(tmp_path, start="20230916", end="20230916"))
        assert manifest["paper_only"] is True
        assert manifest["execution_enabled"] is False
        assert manifest["source"] == "hyperliquid-archive"
        assert manifest["warning"] == ARCHIVE_WARNING

    def test_end_date_before_start_rejected(self, tmp_path: Path) -> None:
        config = BackfillPlanConfig(
            start_date="20230918",
            end_date="20230916",
            coins=["BTC"],
            output_dir=tmp_path,
        )
        with pytest.raises(ValueError, match="before start_date"):
            generate_backfill_manifest(config)


class TestManifestWriters:
    def test_write_manifest_json(self, tmp_path: Path) -> None:
        config = BackfillPlanConfig(
            start_date="20230916",
            end_date="20230916",
            coins=["SOL"],
            output_dir=tmp_path,
        )
        manifest = generate_backfill_manifest(config)
        json_path = write_manifest_json(manifest)

        assert json_path.exists()
        assert json_path.parent == tmp_path
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["total_files"] == 24
        assert loaded["coins"] == ["SOL"]

    def test_write_manifest_markdown(self, tmp_path: Path) -> None:
        config = BackfillPlanConfig(
            start_date="20230916",
            end_date="20230916",
            coins=["SOL"],
            output_dir=tmp_path,
        )
        manifest = generate_backfill_manifest(config)
        markdown_path = write_manifest_markdown(manifest)
        text = markdown_path.read_text(encoding="utf-8")

        assert markdown_path.exists()
        assert "## Summary" in text
        assert "## File counts" in text
        assert "## Example S3 paths" in text
        assert "## Local paths" in text
        assert "## Limitations" in text
        assert "## Next manual steps" in text
        assert "## Research-only disclaimer" in text
        assert NOTHING_DOWNLOADED_WARNING in text


class TestCli:
    def test_main_prints_summary(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(
            [
                "--start-date",
                "20230916",
                "--end-date",
                "20230916",
                "--coins",
                "BTC",
                "ETH",
                "SOL",
                "--output-dir",
                str(tmp_path),
            ]
        )
        captured = capsys.readouterr().out

        assert exit_code == 0
        assert "Total files planned: 72" in captured
        assert "20230916 to 20230916" in captured
        assert "BTC, ETH, SOL" in captured
        assert NOTHING_DOWNLOADED_WARNING in captured
        assert list(tmp_path.glob("historical_l2_backfill_*.json"))
        assert list(tmp_path.glob("historical_l2_backfill_*.md"))


class TestSafety:
    def test_no_network_calls(self) -> None:
        source = PLANNER_FILE.read_text(encoding="utf-8").lower()
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

    def test_no_subprocess_usage(self) -> None:
        source = PLANNER_FILE.read_text(encoding="utf-8").lower()
        assert "subprocess" not in source
        assert "os.system" not in source

    def test_no_wallet_signing_order_execution_code(self) -> None:
        source = PLANNER_FILE.read_text(encoding="utf-8")
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

    def test_constants_match_schema(self) -> None:
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False
