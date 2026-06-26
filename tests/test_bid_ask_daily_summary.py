"""Focused offline tests for bid/ask daily research summary."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from bid_ask_schema import (
    BID_ASK_CSV_COLUMNS,
    FORBIDDEN_MODULE_NAMES,
    FORBIDDEN_SOURCE_PATTERNS,
    build_bid_ask_row,
)
from bid_ask_daily_summary import (
    DEFAULT_OUTPUT_DIR,
    HIGH_MISSING_SECONDS_THRESHOLD,
    RESEARCH_ONLY_WARNING,
    build_daily_summary,
    discover_latest_csvs,
    load_latest_training_report,
    load_latest_validation_report,
    main,
    summarize_csv_file,
    write_summary_json,
    write_summary_markdown,
)
from bid_ask_resampler import RESAMPLED_COLUMNS
from bid_ask_validator import bid_ask_csv_path, reports_dir, save_validation_report
from bid_ask_validator import ValidationReport

SUMMARY_FILE = Path(__file__).resolve().parent.parent / "bid_ask_daily_summary.py"

BASE_MS = 1_718_000_040_000


def _write_1s_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(BID_ASK_CSV_COLUMNS)).to_csv(path, index=False)


def _write_resampled_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS)).to_csv(path, index=False)


def _make_1s_rows(*, coin: str = "BTC", count: int, start_ms: int = BASE_MS) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        best_bid = 100.0 + index * 0.1
        best_ask = best_bid + 1.0
        rows.append(
            build_bid_ask_row(
                timestamp_ms=start_ms + index * 1000,
                coin=coin,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_size=2.0,
                ask_size=1.0,
                source_session_id="test-session",
                source_channel="bbo",
            )
        )
    return rows


def _make_resampled_rows(
    *,
    coin: str = "BTC",
    count: int,
    interval_ms: int,
    start_ms: int = BASE_MS,
) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        start = start_ms + index * interval_ms
        end = start + interval_ms - 1
        mid = 100.0 + index
        rows.append(
            {
                "start_timestamp_ms": start,
                "end_timestamp_ms": end,
                "timestamp_iso": f"2024-06-10T00:00:{index:02d}Z",
                "coin": coin,
                "open_mid": mid,
                "high_mid": mid + 0.1,
                "low_mid": mid - 0.1,
                "close_mid": mid,
                "avg_bid": mid - 0.5,
                "avg_ask": mid + 0.5,
                "avg_mid": mid,
                "avg_spread": 1.0,
                "avg_spread_bps": 2.0,
                "avg_bbo_imbalance": 0.1,
                "row_count": interval_ms,
                "missing_seconds": 0,
            }
        )
    return rows


def _write_validation_report(
    data_root: Path,
    *,
    coin: str,
    interval: str,
    passed: bool = True,
    missing_periods_count: int = 0,
    stamp: str,
) -> Path:
    report = ValidationReport(
        coin=coin,
        interval=interval,
        row_count=10,
        coverage_start=BASE_MS,
        coverage_end=BASE_MS + 9_000,
        duplicate_timestamp_count=0,
        missing_periods_count=missing_periods_count,
        invalid_bid_ask_count=0,
        negative_spread_count=0,
        extreme_spread_count=0,
        warnings=[],
        passed=passed,
        input_path=str(bid_ask_csv_path(coin, interval, data_root)),
    )
    report_dir = reports_dir(data_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{coin}_{interval}_validation_{stamp}.json"
    report.report_path = str(report_path)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report_path


def _write_training_report(
    training_dir: Path,
    *,
    coin: str,
    generated_at: str,
    report_id: str,
) -> Path:
    training_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "coin": coin,
        "interval": "15s",
        "generated_at": generated_at,
        "report_id": report_id,
        "model_used": "majority_class",
        "evaluation": {
            "accuracy": 0.55,
            "baseline_accuracy": 0.50,
            "train_rows": 100,
            "test_rows": 40,
            "warnings": ["Research-only evaluation; does not prove profitability"],
        },
        "paper_only": True,
        "execution_enabled": False,
    }
    path = training_dir / f"{report_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "bid_ask_csv"


@pytest.fixture
def training_dir(tmp_path: Path) -> Path:
    return tmp_path / "bid_ask_training"


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "bid_ask_daily"


def _seed_complete_coin(
    data_root: Path,
    training_dir: Path,
    *,
    coin: str = "BTC",
    missing_seconds: int = 0,
) -> None:
    _write_1s_csv(bid_ask_csv_path(coin, "1s", data_root), _make_1s_rows(coin=coin, count=5))
    _write_resampled_csv(
        bid_ask_csv_path(coin, "15s", data_root),
        _make_resampled_rows(coin=coin, count=4, interval_ms=15_000),
    )
    _write_resampled_csv(
        bid_ask_csv_path(coin, "1m", data_root),
        _make_resampled_rows(coin=coin, count=3, interval_ms=60_000),
    )
    for interval in ("1s", "15s", "1m"):
        _write_validation_report(
            data_root,
            coin=coin,
            interval=interval,
            passed=True,
            missing_periods_count=missing_seconds if interval == "1s" else 0,
            stamp=f"20260625T12000{interval[0]}Z",
        )
    _write_training_report(
        training_dir,
        coin=coin,
        generated_at="2026-06-25T07:00:00+00:00",
        report_id="older-report",
    )
    _write_training_report(
        training_dir,
        coin=coin,
        generated_at="2026-06-25T08:00:00+00:00",
        report_id="newer-report",
    )


class TestDiscoverLatestCsvs:
    def test_discovers_existing_csvs(self, data_root: Path) -> None:
        _write_1s_csv(bid_ask_csv_path("BTC", "1s", data_root), _make_1s_rows(count=3))
        discovered = discover_latest_csvs(["BTC"], data_root=data_root)

        assert discovered["BTC"]["1s"] == bid_ask_csv_path("BTC", "1s", data_root)
        assert discovered["BTC"]["15s"] is None
        assert discovered["BTC"]["1m"] is None


class TestSummarizeCsvFile:
    def test_summarizes_row_count_and_timestamps(self, data_root: Path) -> None:
        path = bid_ask_csv_path("BTC", "1s", data_root)
        _write_1s_csv(path, _make_1s_rows(count=4))

        summary = summarize_csv_file(path)

        assert summary["row_count"] == 4
        assert summary["columns"] == list(BID_ASK_CSV_COLUMNS)
        assert summary["first_timestamp"] == BASE_MS
        assert summary["last_timestamp"] == BASE_MS + 3_000
        assert summary["file_size_bytes"] > 0


class TestLatestValidationReportSelection:
    def test_selects_latest_validation_report(self, data_root: Path) -> None:
        _write_validation_report(
            data_root,
            coin="BTC",
            interval="1s",
            passed=True,
            stamp="20260625T070000Z",
        )
        latest_path = _write_validation_report(
            data_root,
            coin="BTC",
            interval="1s",
            passed=False,
            stamp="20260625T080000Z",
        )

        report = load_latest_validation_report("BTC", "1s", data_root=data_root)

        assert report is not None
        assert report["report_path"] == str(latest_path)
        assert report["passed"] is False


class TestLatestTrainingReportSelection:
    def test_selects_latest_training_report(self, training_dir: Path) -> None:
        _write_training_report(
            training_dir,
            coin="ETH",
            generated_at="2026-06-24T10:00:00+00:00",
            report_id="old-eth",
        )
        _write_training_report(
            training_dir,
            coin="ETH",
            generated_at="2026-06-25T10:00:00+00:00",
            report_id="new-eth",
        )

        report = load_latest_training_report("ETH", training_reports_dir=training_dir)

        assert report is not None
        assert report["report_id"] == "new-eth"
        assert report["evaluation"]["accuracy"] == 0.55


class TestBuildDailySummary:
    def test_daily_summary_structure(
        self,
        data_root: Path,
        training_dir: Path,
    ) -> None:
        _seed_complete_coin(data_root, training_dir, coin="BTC")

        summary = build_daily_summary(
            ["BTC"],
            data_root=data_root,
            training_reports_dir=training_dir,
            include_backfill_tool_check=False,
        )

        assert summary["paper_only"] is True
        assert summary["execution_enabled"] is False
        assert summary["locally_collected_data_only"] is True
        assert len(summary["coins"]) == 1

        coin_summary = summary["coins"][0]
        assert coin_summary["coin"] == "BTC"
        assert coin_summary["csv_status"] == "complete"
        assert coin_summary["row_counts"]["1s"] == 5
        assert coin_summary["validation_passed"] is True
        assert coin_summary["training_accuracy"] == 0.55
        assert coin_summary["baseline_accuracy"] == 0.50
        assert coin_summary["train_rows"] == 100
        assert coin_summary["test_rows"] == 40
        assert "ready_for_review" in summary["overall_status"]

    def test_missing_files_handled_gracefully(
        self,
        data_root: Path,
        training_dir: Path,
    ) -> None:
        summary = build_daily_summary(
            ["SOL"],
            data_root=data_root,
            training_reports_dir=training_dir,
            include_backfill_tool_check=False,
        )

        coin_summary = summary["coins"][0]
        assert coin_summary["csv_status"] == "missing"
        assert coin_summary["validation_passed"] is False
        assert summary["overall_status"]["ready_for_review"] is False
        assert any("Missing 1s CSV" in warning for warning in coin_summary["warnings"])

    def test_high_missing_seconds_warning(
        self,
        data_root: Path,
        training_dir: Path,
    ) -> None:
        _seed_complete_coin(
            data_root,
            training_dir,
            coin="BTC",
            missing_seconds=HIGH_MISSING_SECONDS_THRESHOLD + 10,
        )

        summary = build_daily_summary(
            ["BTC"],
            data_root=data_root,
            training_reports_dir=training_dir,
            include_backfill_tool_check=False,
        )

        warnings = summary["coins"][0]["warnings"]
        assert any("high missing seconds" in warning.lower() for warning in warnings)


class TestReportWriters:
    def test_json_report_created(
        self,
        data_root: Path,
        training_dir: Path,
        output_dir: Path,
    ) -> None:
        _seed_complete_coin(data_root, training_dir)
        summary = build_daily_summary(
            ["BTC"],
            data_root=data_root,
            training_reports_dir=training_dir,
            include_backfill_tool_check=False,
        )

        path = write_summary_json(summary, output_dir=output_dir)

        assert path.is_file()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["paper_only"] is True
        assert payload["coins"][0]["coin"] == "BTC"

    def test_markdown_report_created(
        self,
        data_root: Path,
        training_dir: Path,
        output_dir: Path,
    ) -> None:
        _seed_complete_coin(data_root, training_dir)
        summary = build_daily_summary(
            ["BTC"],
            data_root=data_root,
            training_reports_dir=training_dir,
            include_backfill_tool_check=False,
        )

        path = write_summary_markdown(summary, output_dir=output_dir)
        text = path.read_text(encoding="utf-8")

        assert path.is_file()
        assert "# Bid/Ask Daily Research Summary" in text
        assert "## Coin Table" in text
        assert RESEARCH_ONLY_WARNING in text
        assert "bid_ask_daily_summary.py --coins BTC ETH SOL" in text


class TestNoProfitabilityClaim:
    def test_summary_never_labels_training_profitable(
        self,
        data_root: Path,
        training_dir: Path,
        output_dir: Path,
    ) -> None:
        _seed_complete_coin(data_root, training_dir)
        summary = build_daily_summary(
            ["BTC"],
            data_root=data_root,
            training_reports_dir=training_dir,
            include_backfill_tool_check=False,
        )
        json_path = write_summary_json(summary, output_dir=output_dir)
        markdown_path = write_summary_markdown(summary, output_dir=output_dir)

        combined = json.dumps(summary) + json_path.read_text() + markdown_path.read_text()
        lowered = combined.lower()
        assert "profitable" not in lowered or "does not prove profitability" in lowered
        assert RESEARCH_ONLY_WARNING in summary["overall_status"]["warnings"]


class TestCli:
    def test_main_prints_output_paths(
        self,
        data_root: Path,
        training_dir: Path,
        output_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _seed_complete_coin(data_root, training_dir)
        exit_code = main(
            [
                "--coins",
                "BTC",
                "--data-root",
                str(data_root),
                "--training-reports-dir",
                str(training_dir),
                "--output-dir",
                str(output_dir),
                "--skip-backfill-tool-check",
            ]
        )

        captured = capsys.readouterr().out
        assert exit_code == 0
        assert "json_report_path=" in captured
        assert "markdown_report_path=" in captured
        assert "ready_for_review=" in captured
        assert "warning:" in captured


class TestSafety:
    def test_no_wallet_signing_order_execution_code(self) -> None:
        source = SUMMARY_FILE.read_text(encoding="utf-8")
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

    def test_no_network_calls(self) -> None:
        source = SUMMARY_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {
            "requests",
            "urllib",
            "http",
            "socket",
            "websockets",
            "httpx",
            "aiohttp",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in forbidden
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in forbidden

    def test_no_subprocess_usage(self) -> None:
        source = SUMMARY_FILE.read_text(encoding="utf-8")
        assert "subprocess" not in source

    def test_default_output_dir_is_reports_bid_ask_daily(self) -> None:
        assert DEFAULT_OUTPUT_DIR == Path("reports/bid_ask_daily")
