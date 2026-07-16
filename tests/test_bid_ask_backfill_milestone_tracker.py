"""Tests for bid_ask_backfill_milestone_tracker (offline, temp dirs, no network)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import bid_ask_backfill_milestone_tracker as tracker
from bid_ask_schema import (
    ASK_SIZE,
    BID_SIZE,
    EXECUTION_ENABLED,
    FORBIDDEN_MODULE_NAMES,
    FORBIDDEN_SOURCE_PATTERNS,
    PAPER_ONLY,
    timestamp_ms_to_iso,
)
from bid_ask_validator import bid_ask_csv_path

TRACKER_FILE = Path(__file__).resolve().parent.parent / "bid_ask_backfill_milestone_tracker.py"

BASE_MS = 1_700_000_000_000
THREE_DAY_MS = 3 * 24 * 60 * 60 * 1000


def _write_1s_csv(
    path: Path,
    *,
    coin: str = "BTC",
    rows: int,
    start_ms: int = BASE_MS,
    include_sizes: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "timestamp_ms",
        "timestamp_iso",
        "coin",
        "best_bid",
        "best_ask",
        "mid_price",
        "spread",
        "spread_bps",
        BID_SIZE,
        ASK_SIZE,
        "bbo_imbalance",
        "source_session_id",
        "source_channel",
    ]
    lines = [",".join(columns)]
    for index in range(rows):
        ts = start_ms + index * 1000
        iso = timestamp_ms_to_iso(ts)
        bid = 100.0 + index
        ask = 100.5 + index
        mid = (bid + ask) / 2.0
        spread = ask - bid
        spread_bps = spread / mid * 10000.0
        bid_size = "2.0" if include_sizes else ""
        ask_size = "1.0" if include_sizes else ""
        lines.append(
            f"{ts},{iso},{coin},{bid},{ask},{mid},{spread},{spread_bps},"
            f"{bid_size},{ask_size},0.0,session,bbo"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_resampled_csv(
    path: Path,
    *,
    coin: str = "BTC",
    interval_seconds: int,
    rows: int,
    start_ms: int = BASE_MS,
    include_sizes: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    step_ms = interval_seconds * 1000
    columns = [
        "start_timestamp_ms",
        "end_timestamp_ms",
        "timestamp_iso",
        "coin",
        "open_mid",
        "high_mid",
        "low_mid",
        "close_mid",
        "avg_bid",
        "avg_ask",
        "avg_mid",
        "avg_spread",
        "avg_spread_bps",
        "avg_bbo_imbalance",
        "row_count",
        "missing_seconds",
    ]
    if include_sizes:
        columns.extend([BID_SIZE, ASK_SIZE])

    lines = [",".join(columns)]
    for index in range(rows):
        start = start_ms + index * step_ms
        end = start + step_ms - 1
        iso = timestamp_ms_to_iso(start)
        mid = 100.0 + index
        row = [
            str(start),
            str(end),
            iso,
            coin,
            str(mid),
            str(mid + 0.5),
            str(mid - 0.5),
            str(mid),
            str(mid - 0.25),
            str(mid + 0.25),
            str(mid),
            "0.5",
            "5.0",
            "0.0",
            str(interval_seconds),
            "0",
        ]
        if include_sizes:
            row.extend(["2.0", "1.0"])
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_validation_report(
    data_root: Path,
    *,
    coin: str,
    interval: str,
    stamp: str,
    missing_periods_count: int = 42,
    passed: bool = True,
) -> Path:
    report_dir = data_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{coin}_{interval}_validation_{stamp}.json"
    payload = {
        "coin": coin,
        "interval": interval,
        "row_count": 100,
        "coverage_start": BASE_MS,
        "coverage_end": BASE_MS + 99_000,
        "duplicate_timestamp_count": 0,
        "missing_periods_count": missing_periods_count,
        "warnings": [f"{missing_periods_count} missing second(s) in coverage range"],
        "passed": passed,
        "generated_at": "2024-01-02T10:00:00Z",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "data"
    output_dir = tmp_path / "reports" / "backfill_milestones"
    return {
        "data_root": data_root,
        "output_dir": output_dir,
    }


class TestCoverage:
    def test_synthetic_csv_coverage_works(self, workspace: dict[str, Path]) -> None:
        data_root = workspace["data_root"]
        rows = THREE_DAY_MS // 1000

        for coin in ("BTC", "ETH"):
            _write_1s_csv(
                bid_ask_csv_path(coin, "1s", data_root, source="historical"),
                coin=coin,
                rows=rows,
            )
            _write_resampled_csv(
                bid_ask_csv_path(coin, "15s", data_root, source="historical"),
                coin=coin,
                interval_seconds=15,
                rows=rows // 15,
            )
            _write_resampled_csv(
                bid_ask_csv_path(coin, "1m", data_root, source="historical"),
                coin=coin,
                interval_seconds=60,
                rows=rows // 60,
            )

        report = tracker.build_backfill_milestone_report(data_root=data_root)
        btc_1s = report["coverage"]["BTC"]["1s"]

        assert btc_1s["row_count"] == rows
        assert btc_1s["first_timestamp_ms"] == BASE_MS
        assert btc_1s["latest_timestamp_ms"] == BASE_MS + (rows - 1) * 1000
        assert btc_1s["first_timestamp_iso"] == timestamp_ms_to_iso(BASE_MS)
        assert btc_1s["latest_timestamp_iso"] == timestamp_ms_to_iso(BASE_MS + (rows - 1) * 1000)
        assert btc_1s["coverage_hours"] == pytest.approx((rows - 1) / 3600, rel=1e-3)

    def test_missing_files_handled(self, workspace: dict[str, Path]) -> None:
        report = tracker.build_backfill_milestone_report(data_root=workspace["data_root"])

        for coin in ("BTC", "ETH"):
            for interval in ("1s", "15s", "1m"):
                entry = report["coverage"][coin][interval]
                assert entry["file_exists"] is False
                assert entry["row_count"] == 0
                assert entry["coverage_hours"] == 0.0
                assert entry["first_timestamp_ms"] is None
                assert entry["latest_timestamp_ms"] is None
                assert entry["first_timestamp_iso"] is None
                assert entry["latest_timestamp_iso"] is None
                assert entry["next_milestone"] == "3_days"

        assert any("CSV missing" in warning for warning in report["data_quality_warnings"])


class TestTimestampExport:
    def test_timestamp_ms_export(self, workspace: dict[str, Path]) -> None:
        data_root = workspace["data_root"]
        path = bid_ask_csv_path("BTC", "1s", data_root, source="historical")
        _write_1s_csv(path, rows=5, start_ms=BASE_MS)

        stats = tracker._analyze_csv(path, interval="1s")

        assert stats["first_timestamp_ms"] == BASE_MS
        assert stats["latest_timestamp_ms"] == BASE_MS + 4000
        assert stats["first_timestamp_iso"] == timestamp_ms_to_iso(BASE_MS)
        assert stats["latest_timestamp_iso"] == timestamp_ms_to_iso(BASE_MS + 4000)
        assert stats["coverage_hours"] == pytest.approx(0.001, rel=1e-3)

    def test_timestamp_iso_fallback_when_ms_missing(self, workspace: dict[str, Path]) -> None:
        data_root = workspace["data_root"]
        path = bid_ask_csv_path("BTC", "1s", data_root, source="historical")
        path.parent.mkdir(parents=True, exist_ok=True)
        first_iso = "2024-06-01T12:00:00.123000Z"
        latest_iso = "2024-06-01T12:00:04.123000Z"
        lines = [
            "timestamp_ms,timestamp_iso,coin,best_bid,best_ask,mid_price,spread,spread_bps",
            f",{first_iso},BTC,100,101,100.5,1,10",
            f",{latest_iso},BTC,101,102,101.5,1,10",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        stats = tracker._analyze_csv(path, interval="1s")

        assert stats["first_timestamp_iso"] == first_iso
        assert stats["latest_timestamp_iso"] == latest_iso
        assert stats["first_timestamp_ms"] == tracker._parse_iso_to_ms(first_iso)
        assert stats["latest_timestamp_ms"] == tracker._parse_iso_to_ms(latest_iso)
        assert stats["coverage_hours"] > 0.0

    def test_mixed_invalid_timestamps_ignored(self, workspace: dict[str, Path]) -> None:
        data_root = workspace["data_root"]
        path = bid_ask_csv_path("BTC", "1s", data_root, source="historical")
        path.parent.mkdir(parents=True, exist_ok=True)
        valid_ms = BASE_MS + 2000
        valid_iso = timestamp_ms_to_iso(valid_ms)
        lines = [
            "timestamp_ms,timestamp_iso,coin,best_bid,best_ask,mid_price,spread,spread_bps",
            ",,BTC,100,101,100.5,1,10",
            "NaN,not-an-iso,BTC,100,101,100.5,1,10",
            f" ,{valid_iso},BTC,100,101,100.5,1,10",
            f"{valid_ms},{valid_iso},BTC,101,102,101.5,1,10",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        stats = tracker._analyze_csv(path, interval="1s")

        assert stats["row_count"] == 4
        assert stats["first_timestamp_ms"] == valid_ms
        assert stats["latest_timestamp_ms"] == valid_ms
        assert stats["first_timestamp_iso"] == valid_iso
        assert stats["latest_timestamp_iso"] == valid_iso
        assert stats["coverage_hours"] == 0.0

    def test_missing_timestamp_columns(self, workspace: dict[str, Path]) -> None:
        data_root = workspace["data_root"]
        path = bid_ask_csv_path("BTC", "1s", data_root, source="historical")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "coin,best_bid,best_ask,mid_price,spread,spread_bps\nBTC,100,101,100.5,1,10\n",
            encoding="utf-8",
        )

        stats = tracker._analyze_csv(path, interval="1s")

        assert stats["row_count"] == 1
        assert stats["first_timestamp_ms"] is None
        assert stats["latest_timestamp_ms"] is None
        assert stats["first_timestamp_iso"] is None
        assert stats["latest_timestamp_iso"] is None
        assert stats["coverage_hours"] == 0.0

    def test_resampled_end_timestamp_ms_used_for_latest(self, workspace: dict[str, Path]) -> None:
        data_root = workspace["data_root"]
        path = bid_ask_csv_path("BTC", "15s", data_root, source="historical")
        _write_resampled_csv(path, interval_seconds=15, rows=3, start_ms=BASE_MS)

        stats = tracker._analyze_csv(path, interval="15s")

        assert stats["first_timestamp_ms"] == BASE_MS
        assert stats["latest_timestamp_ms"] == BASE_MS + 2 * 15_000 + 14_999
        assert stats["first_timestamp_iso"] == timestamp_ms_to_iso(BASE_MS)
        assert stats["latest_timestamp_iso"] == timestamp_ms_to_iso(BASE_MS + 2 * 15_000)
        assert stats["coverage_hours"] == pytest.approx(0.012, rel=1e-3)

    def test_coverage_hours_unchanged_with_iso_only_rows(self, workspace: dict[str, Path]) -> None:
        data_root = workspace["data_root"]
        path = bid_ask_csv_path("ETH", "1s", data_root, source="historical")
        path.parent.mkdir(parents=True, exist_ok=True)
        start_iso = "2024-01-01T00:00:00Z"
        end_iso = "2024-01-04T00:00:00Z"
        lines = [
            "timestamp_ms,timestamp_iso,coin,best_bid,best_ask,mid_price,spread,spread_bps",
            f",{start_iso},ETH,100,101,100.5,1,10",
            f",{end_iso},ETH,101,102,101.5,1,10",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        report = tracker.build_backfill_milestone_report(data_root=data_root)
        entry = report["coverage"]["ETH"]["1s"]

        assert entry["coverage_hours"] == pytest.approx(72.0, rel=1e-3)
        assert entry["first_timestamp_iso"] == start_iso
        assert entry["latest_timestamp_iso"] == end_iso


class TestMilestones:
    def test_milestone_percentages_computed_correctly(self, workspace: dict[str, Path]) -> None:
        data_root = workspace["data_root"]
        rows = (36 * 60 * 60) + 1
        _write_1s_csv(
            bid_ask_csv_path("BTC", "1s", data_root, source="historical"),
            rows=rows,
        )

        report = tracker.build_backfill_milestone_report(data_root=data_root)
        milestones = report["coverage"]["BTC"]["1s"]["milestone_progress"]["milestones"]

        assert milestones["3_days"]["percent_complete"] == pytest.approx(50.0, abs=0.1)
        assert milestones["7_days"]["percent_complete"] == pytest.approx(21.43, abs=0.1)
        assert milestones["3_days"]["met"] is False
        assert report["coverage"]["BTC"]["1s"]["next_milestone"] == "3_days"

    def test_all_targets_met_next_milestone(self, workspace: dict[str, Path]) -> None:
        data_root = workspace["data_root"]
        rows = (30 * 24 * 60 * 60) + 1
        _write_1s_csv(
            bid_ask_csv_path("BTC", "1s", data_root, source="historical"),
            rows=rows,
        )

        report = tracker.build_backfill_milestone_report(data_root=data_root)
        assert report["coverage"]["BTC"]["1s"]["next_milestone"] == "all_targets_met"


class TestDepthAndValidation:
    def test_bid_size_ask_size_detection_works(self, workspace: dict[str, Path]) -> None:
        data_root = workspace["data_root"]

        _write_1s_csv(
            bid_ask_csv_path("BTC", "1s", data_root, source="historical"),
            rows=5,
            include_sizes=True,
        )
        _write_1s_csv(
            bid_ask_csv_path("ETH", "1s", data_root, source="historical"),
            rows=5,
            include_sizes=False,
        )
        _write_resampled_csv(
            bid_ask_csv_path("BTC", "15s", data_root, source="historical"),
            interval_seconds=15,
            rows=3,
            include_sizes=True,
        )
        _write_resampled_csv(
            bid_ask_csv_path("ETH", "15s", data_root, source="historical"),
            interval_seconds=15,
            rows=3,
            include_sizes=False,
        )

        report = tracker.build_backfill_milestone_report(data_root=data_root)

        assert report["coverage"]["BTC"]["1s"]["bid_size_present"] is True
        assert report["coverage"]["BTC"]["1s"]["ask_size_present"] is True
        assert report["coverage"]["BTC"]["1s"]["microprice_features_expected"] is True

        assert report["coverage"]["ETH"]["1s"]["bid_size_present"] is False
        assert report["coverage"]["ETH"]["1s"]["microprice_features_expected"] is False

        assert report["coverage"]["BTC"]["15s"]["microprice_features_expected"] is True
        assert report["coverage"]["ETH"]["15s"]["microprice_features_expected"] is False
        assert any("ETH 15s" in gap for gap in report["depth_feature_status"]["gaps"])

    def test_latest_validator_warning_parsed_if_available(self, workspace: dict[str, Path]) -> None:
        data_root = workspace["data_root"]
        _write_1s_csv(
            bid_ask_csv_path("BTC", "1s", data_root, source="historical"),
            rows=10,
        )
        _write_validation_report(
            data_root,
            coin="BTC",
            interval="1s",
            stamp="20240101T100000Z",
            missing_periods_count=5,
        )
        _write_validation_report(
            data_root,
            coin="BTC",
            interval="1s",
            stamp="20240102T100000Z",
            missing_periods_count=99,
        )

        report = tracker.build_backfill_milestone_report(data_root=data_root)
        entry = report["coverage"]["BTC"]["1s"]

        assert entry["validator_report_available"] is True
        assert entry["missing_incomplete_warning_count"] == 99
        assert any("99 missing/incomplete" in warning for warning in report["data_quality_warnings"])


class TestOutputAndSafety:
    def test_safety_fields_included(self, workspace: dict[str, Path]) -> None:
        report = tracker.build_backfill_milestone_report(data_root=workspace["data_root"])

        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False
        assert report["safety_statement"]
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False

    def test_latest_name_files_written_as_regular_files(self, workspace: dict[str, Path]) -> None:
        tracker.build_and_write_milestone_report(
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
        )

        json_latest = workspace["output_dir"] / tracker.DEFAULT_LATEST_JSON_NAME
        md_latest = workspace["output_dir"] / tracker.DEFAULT_LATEST_MD_NAME
        assert json_latest.is_file()
        assert md_latest.is_file()
        assert not json_latest.is_symlink()
        assert not md_latest.is_symlink()
        assert json.loads(json_latest.read_text(encoding="utf-8"))
        assert "## 1. Safety Statement" in md_latest.read_text(encoding="utf-8")
        assert any(path.name.startswith("backfill_milestones_20") for path in workspace["output_dir"].glob("*.json"))


def _block_network_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network/subprocess call attempted")

    monkeypatch.setattr("subprocess.run", _blocked)
    monkeypatch.setattr("subprocess.Popen", _blocked)
    monkeypatch.setattr("boto3.client", _blocked)


class TestSafety:
    def test_no_forbidden_imports(self) -> None:
        source = TRACKER_FILE.read_text(encoding="utf-8")
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

    def test_no_network_calls_in_source(self) -> None:
        source = TRACKER_FILE.read_text(encoding="utf-8").lower()
        forbidden = (
            "requests.get",
            "requests.post",
            "urllib.request",
            "http.client",
            "aiohttp",
            "websocket",
            "boto3",
        )
        for pattern in forbidden:
            assert pattern not in source

    def test_no_subprocess_calls_in_source(self) -> None:
        source = TRACKER_FILE.read_text(encoding="utf-8").lower()
        forbidden = ("subprocess", "os.system", "popen")
        for pattern in forbidden:
            assert pattern not in source

    def test_no_network_subprocess_or_aws_at_runtime(
        self,
        workspace: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _block_network_subprocess(monkeypatch)
        tracker.build_and_write_milestone_report(
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
        )
