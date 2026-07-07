"""Focused offline tests for bid/ask data health watchdog."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from bid_ask_data_health_watchdog import (
    DEFAULT_OUTPUT_DIR,
    build_safety_flags,
    main,
    run_data_health_check,
    write_health_json,
    write_health_markdown,
)
from bid_ask_schema import BID_ASK_CSV_COLUMNS, build_bid_ask_row
from bid_ask_resampler import RESAMPLED_COLUMNS
from bid_ask_validator import bid_ask_csv_path, reports_dir

WATCHDOG_FILE = Path(__file__).resolve().parent.parent / "bid_ask_data_health_watchdog.py"

BASE_MS = 1_718_000_040_000


def _write_1s_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(BID_ASK_CSV_COLUMNS)).to_csv(path, index=False)


def _make_1s_rows(
    *,
    coin: str = "BTC",
    count: int,
    start_ms: int = BASE_MS,
    duplicate_at: int | None = None,
    non_monotonic_at: int | None = None,
    missing_iso_at: int | None = None,
    inverted_at: int | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        timestamp_ms = start_ms + index * 1000
        if duplicate_at is not None and index == duplicate_at:
            timestamp_ms = start_ms + (duplicate_at - 1) * 1000
        if non_monotonic_at is not None and index == non_monotonic_at:
            timestamp_ms = start_ms + (non_monotonic_at - 2) * 1000

        best_bid = 100.0 + index * 0.1
        best_ask = best_bid + 1.0
        if inverted_at is not None and index == inverted_at:
            best_ask = best_bid - 0.5

        row = build_bid_ask_row(
            timestamp_ms=timestamp_ms,
            coin=coin,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_size=2.0,
            ask_size=1.0,
            source_session_id="test-session",
            source_channel="bbo",
        )
        if missing_iso_at is not None and index == missing_iso_at:
            row["timestamp_iso"] = ""
        rows.append(row)
    return rows


def _make_resampled_rows(
    *,
    coin: str = "BTC",
    count: int,
    start_ms: int = BASE_MS,
) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        start = start_ms + index * 15_000
        end = start + 14_999
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
                "row_count": 15,
                "missing_seconds": 0,
            }
        )
    return rows


def _write_resampled_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS)).to_csv(path, index=False)


def _write_validation_report(
    data_root: Path,
    *,
    coin: str,
    interval: str,
    passed: bool,
    stamp: str,
) -> Path:
    report_dir = reports_dir(data_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{coin}_{interval}_validation_{stamp}.json"
    payload = {
        "coin": coin,
        "interval": interval,
        "row_count": 10,
        "coverage_start": BASE_MS,
        "coverage_end": BASE_MS + 9_000,
        "duplicate_timestamp_count": 0,
        "missing_periods_count": 0,
        "invalid_bid_ask_count": 0,
        "negative_spread_count": 0,
        "extreme_spread_count": 0,
        "warnings": [],
        "passed": passed,
        "input_path": str(bid_ask_csv_path(coin, interval, data_root, source="historical")),
        "report_path": str(report_path),
        "paper_only": True,
        "generated_at": "2026-07-07T07:09:56.233870Z",
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def _write_candidate_report(
    reports_root: Path,
    *,
    stamp: str,
    filtered_paper_trades: int,
    cumulative_net_return_bps: float,
    status_pass: bool,
) -> Path:
    candidate_dir = reports_root / "bid_ask_candidate_validation"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    report_path = candidate_dir / f"candidate_validation_TEST_{stamp}.json"
    payload = {
        "candidate_name": "TEST",
        "run_timestamp": f"2026-07-06T{stamp[:2]}:{stamp[2:4]}:{stamp[4:6]}.000000Z",
        "filtered_paper_trades": filtered_paper_trades,
        "cumulative_net_return_bps": cumulative_net_return_bps,
        "validation_status": {"paper_candidate_pass": status_pass},
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def _write_phase_audit(
    reports_root: Path,
    *,
    total_files: int,
    freshness_by_coin: dict[str, str],
) -> Path:
    audit_dir = reports_root / "bid_ask_research_scheduler"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "phase3_audit_latest.json"
    payload = {
        "generated_at": "2026-07-07T07:16:59.406310Z",
        "downloads": {"total_files": total_files},
        "imported_historical_1s": {
            coin: {"freshness_status": status}
            for coin, status in freshness_by_coin.items()
        },
    }
    audit_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return audit_path


def _setup_healthy_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    reports_root = tmp_path / "reports"

    for coin in ("BTC", "ETH"):
        one_s_path = bid_ask_csv_path(coin, "1s", data_root, source="historical")
        _write_1s_csv(one_s_path, _make_1s_rows(coin=coin, count=20))

        fifteen_s_path = bid_ask_csv_path(coin, "15s", data_root, source="historical")
        _write_resampled_csv(
            fifteen_s_path,
            _make_resampled_rows(coin=coin, count=20, start_ms=BASE_MS),
        )

        for interval in ("1s", "15s"):
            _write_validation_report(
                data_root,
                coin=coin,
                interval=interval,
                passed=True,
                stamp="20260707T070956Z",
            )

    return data_root, reports_root


def test_healthy_data_reports_healthy_status(tmp_path: Path) -> None:
    data_root, reports_root = _setup_healthy_fixture(tmp_path)

    report = run_data_health_check(
        coins=["BTC", "ETH"],
        source="historical",
        data_root=data_root,
        reports_root=reports_root,
        max_stale_hours=999_999,
    )

    assert report["status"] == "healthy"
    assert report["issues"] == []
    assert report["recommended_commands"]
    assert report["safety_flags"]["read_only"] is True
    assert report["safety_flags"]["no_network_calls"] is True
    assert report["safety_flags"]["no_subprocess_calls"] is True


def test_duplicate_timestamps_are_critical(tmp_path: Path) -> None:
    data_root, reports_root = _setup_healthy_fixture(tmp_path)
    one_s_path = bid_ask_csv_path("BTC", "1s", data_root, source="historical")
    _write_1s_csv(one_s_path, _make_1s_rows(coin="BTC", count=10, duplicate_at=5))

    report = run_data_health_check(
        coins=["BTC"],
        source="historical",
        data_root=data_root,
        reports_root=reports_root,
        max_stale_hours=999_999,
    )

    assert report["status"] == "critical"
    assert any(issue["code"] == "duplicate_timestamps" for issue in report["issues"])


def test_non_monotonic_timestamps_are_critical(tmp_path: Path) -> None:
    data_root, reports_root = _setup_healthy_fixture(tmp_path)
    one_s_path = bid_ask_csv_path("BTC", "1s", data_root, source="historical")
    _write_1s_csv(one_s_path, _make_1s_rows(coin="BTC", count=10, non_monotonic_at=5))

    report = run_data_health_check(
        coins=["BTC"],
        source="historical",
        data_root=data_root,
        reports_root=reports_root,
        max_stale_hours=999_999,
    )

    assert report["status"] == "critical"
    assert any(issue["code"] == "non_monotonic_timestamps" for issue in report["issues"])


def test_missing_csv_is_critical(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    reports_root = tmp_path / "reports"

    report = run_data_health_check(
        coins=["BTC"],
        source="historical",
        data_root=data_root,
        reports_root=reports_root,
        max_stale_hours=999_999,
    )

    assert report["status"] == "critical"
    assert any(issue["code"] == "missing_csv" for issue in report["issues"])


def test_stale_15s_while_1s_advanced_is_warning(tmp_path: Path) -> None:
    data_root, reports_root = _setup_healthy_fixture(tmp_path)

    one_s_path = bid_ask_csv_path("BTC", "1s", data_root, source="historical")
    _write_1s_csv(one_s_path, _make_1s_rows(coin="BTC", count=30, start_ms=BASE_MS))

    fifteen_s_path = bid_ask_csv_path("BTC", "15s", data_root, source="historical")
    _write_resampled_csv(
        fifteen_s_path,
        _make_resampled_rows(coin="BTC", count=1, start_ms=BASE_MS),
    )

    report = run_data_health_check(
        coins=["BTC"],
        source="historical",
        data_root=data_root,
        reports_root=reports_root,
        max_stale_hours=999_999,
    )

    assert report["status"] == "warning"
    assert any(issue["code"] == "stale_15s_while_1s_advanced" for issue in report["issues"])


def test_validation_failure_is_warning(tmp_path: Path) -> None:
    data_root, reports_root = _setup_healthy_fixture(tmp_path)
    _write_validation_report(
        data_root,
        coin="BTC",
        interval="1s",
        passed=False,
        stamp="20260707T080000Z",
    )

    report = run_data_health_check(
        coins=["BTC"],
        source="historical",
        data_root=data_root,
        reports_root=reports_root,
        max_stale_hours=999_999,
    )

    assert report["status"] == "warning"
    assert any(issue["code"] == "validation_failed" for issue in report["issues"])


def test_recommendations_generated_for_issues(tmp_path: Path) -> None:
    data_root, reports_root = _setup_healthy_fixture(tmp_path)
    one_s_path = bid_ask_csv_path("BTC", "1s", data_root, source="historical")
    _write_1s_csv(one_s_path, _make_1s_rows(coin="BTC", count=10, duplicate_at=3))

    report = run_data_health_check(
        coins=["BTC"],
        source="historical",
        data_root=data_root,
        reports_root=reports_root,
        max_stale_hours=999_999,
    )

    commands = report["recommended_commands"]
    assert any("bid_ask_validator.py" in command for command in commands)


def test_safety_flags_included_in_report_and_writes(tmp_path: Path) -> None:
    data_root, reports_root = _setup_healthy_fixture(tmp_path)
    output_dir = tmp_path / "health_out"

    report = run_data_health_check(
        coins=["BTC"],
        source="historical",
        data_root=data_root,
        reports_root=reports_root,
        max_stale_hours=999_999,
    )
    flags = build_safety_flags()
    assert report["safety_flags"]["paper_only"] is True
    assert report["safety_flags"]["no_repairs_performed"] is True
    assert flags["research_only"] is True

    json_path = write_health_json(report, output_dir, "data_health_latest.json")
    markdown_path = write_health_markdown(report, output_dir, "data_health_latest.md")
    assert json_path.is_file()
    assert markdown_path.is_file()
    assert "Safety Flags" in markdown_path.read_text(encoding="utf-8")


def test_candidate_unchanged_warning(tmp_path: Path) -> None:
    data_root, reports_root = _setup_healthy_fixture(tmp_path)
    for stamp in ("181446", "181500", "181600"):
        _write_candidate_report(
            reports_root,
            stamp=stamp,
            filtered_paper_trades=5,
            cumulative_net_return_bps=3.8,
            status_pass=False,
        )

    report = run_data_health_check(
        coins=["BTC"],
        source="historical",
        data_root=data_root,
        reports_root=reports_root,
        max_stale_hours=999_999,
    )

    assert report["status"] == "warning"
    assert report["candidate_freshness"]["repeated_unchanged_warning"] is True
    assert any(issue["code"] == "candidate_result_unchanged" for issue in report["issues"])


def test_phase_audit_downloads_without_advancing_1s(tmp_path: Path) -> None:
    data_root, reports_root = _setup_healthy_fixture(tmp_path)
    _write_phase_audit(
        reports_root,
        total_files=50,
        freshness_by_coin={"BTC": "stale", "ETH": "advancing"},
    )

    report = run_data_health_check(
        coins=["BTC", "ETH"],
        source="historical",
        data_root=data_root,
        reports_root=reports_root,
        max_stale_hours=999_999,
    )

    assert report["status"] == "warning"
    assert any(issue["code"] == "downloads_exist_1s_not_advancing" for issue in report["issues"])
    assert any("autopilot-tick" in command for command in report["recommended_commands"])


def test_cli_check_writes_reports(tmp_path: Path) -> None:
    data_root, reports_root = _setup_healthy_fixture(tmp_path)
    output_dir = tmp_path / "health_out"

    exit_code = main(
        [
            "check",
            "--coins",
            "BTC",
            "--source",
            "historical",
            "--data-root",
            str(data_root),
            "--reports-root",
            str(reports_root),
            "--output-dir",
            str(output_dir),
            "--max-stale-hours",
            "999999",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "data_health_latest.json").is_file()
    assert (output_dir / "data_health_latest.md").is_file()


def test_watchdog_module_has_no_forbidden_patterns() -> None:
    source = WATCHDOG_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    forbidden_imports = {"subprocess", "boto3", "requests", "urllib", "socket", "web3"}
    assert forbidden_imports.isdisjoint(imported_roots)
    assert DEFAULT_OUTPUT_DIR == Path("reports/data_health")
