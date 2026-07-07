"""Tests for bid_ask_daily_progress_digest (offline, temp dirs, no network)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import bid_ask_daily_progress_digest as digest
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY

DIGEST_FILE = Path(__file__).resolve().parent.parent / "bid_ask_daily_progress_digest.py"

TEST_DATE = "20240102"
OLD_STAMP = "20240101T100000Z"
NEW_STAMP = "20240102T100000Z"


def _write_1s_csv(path: Path, coin: str, rows: int = 5, base_ts: int = 1_700_000_000_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "timestamp_ms,timestamp_iso,coin,best_bid,best_ask,mid_price,spread,spread_bps,"
        "bid_size,ask_size,bbo_imbalance,source_session_id,source_channel"
    ]
    for index in range(rows):
        ts = base_ts + index * 1000
        bid = 100.0 + index
        ask = 100.5 + index
        mid = (bid + ask) / 2.0
        spread = ask - bid
        spread_bps = spread / mid * 10000.0
        lines.append(
            f"{ts},2024-01-01T00:00:00Z,{coin},{bid},{ask},{mid},{spread},{spread_bps},"
            f"1.0,1.0,0.0,session,bbo"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_15s_csv(path: Path, coin: str, rows: int = 3, base_ts: int = 1_700_000_000_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "start_timestamp_ms,end_timestamp_ms,timestamp_iso,coin,open_mid,high_mid,low_mid,"
        "close_mid,avg_bid,avg_ask,avg_mid,avg_spread,avg_spread_bps,avg_bbo_imbalance,"
        "row_count,missing_seconds"
    ]
    for index in range(rows):
        start = base_ts + index * 15_000
        end = start + 14_999
        mid = 100.0 + index
        lines.append(
            f"{start},{end},2024-01-01T00:00:00Z,{coin},{mid},{mid + 0.5},{mid - 0.5},"
            f"{mid},{mid - 0.25},{mid + 0.25},{mid},0.5,5.0,0.0,15,0"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_validation_report(
    reports_dir: Path,
    coin: str,
    interval: str,
    stamp: str,
    *,
    passed: bool = True,
    row_count: int = 10,
    warnings: list[str] | None = None,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{coin}_{interval}_validation_{stamp}.json"
    payload = {
        "coin": coin,
        "interval": interval,
        "row_count": row_count,
        "coverage_start": 1_700_000_000_000,
        "coverage_end": 1_700_000_004_000,
        "duplicate_timestamp_count": 0,
        "missing_periods_count": 120,
        "warnings": warnings or ["120 missing second(s) in coverage range"],
        "passed": passed,
        "generated_at": "2024-01-02T10:00:00Z",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_candidate_validation(
    reports_dir: Path,
    stamp: str,
    *,
    candidate_name: str,
    cumulative_net_bps: float = 3.84,
    filtered_paper_trades: int = 5,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"candidate_validation_{candidate_name}_{stamp}.json"
    payload = {
        "candidate_name": candidate_name,
        "run_timestamp": "2024-01-02T10:00:00Z",
        "filtered_paper_trades": filtered_paper_trades,
        "cumulative_net_return_bps": cumulative_net_bps,
        "positive_after_costs": cumulative_net_bps > 0,
        "validation_status": {
            "insufficient_trades": filtered_paper_trades < 10,
            "not_market_ready": True,
            "paper_candidate_pass": False,
        },
        "stability_warnings": ["Concentrated in one time bucket"],
        "paper_only": True,
        "execution_enabled": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_phase3_audit(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2024-01-02T10:00:00Z",
        "paper_only": True,
        "execution_enabled": False,
        "real_execution_ready": False,
        "phase_config": {
            "phase_backfill_name": "phase3_btc_eth_recent",
            "target_days": 14,
            "estimated_real_days_to_complete": 4.5,
        },
        "scheduler_progress": {
            "latest_action": "no_import_progress",
            "files_downloaded_today": 12,
            "files_downloaded_new": 8,
            "chunks_completed_today": 2,
            "failed_runs_in_row": 1,
            "paused": False,
            "warnings": ["Historical CSVs did not advance after chunk cycle."],
            "errors": [],
        },
        "downloads": {
            "total_files": 42,
            "by_coin": {"BTC": 22, "ETH": 20},
        },
        "imported_historical_1s": {
            "BTC": {
                "row_count": 100,
                "first_timestamp_ms": 1_700_000_000_000,
                "last_timestamp_ms": 1_700_000_360_000,
                "last_timestamp": "2024-01-01T01:00:00Z",
                "coverage_hours": 1.0,
                "freshness_status": "advancing",
            },
            "ETH": {
                "row_count": 80,
                "first_timestamp_ms": 1_700_000_000_000,
                "last_timestamp_ms": 1_700_000_180_000,
                "last_timestamp": "2024-01-01T00:30:00Z",
                "coverage_hours": 0.5,
                "freshness_status": "advancing",
            },
        },
        "resampled_coverage": {
            "BTC": {"15s": {"row_count": 20}, "1m": {"row_count": 5}},
            "ETH": {"15s": {"row_count": 15}, "1m": {"row_count": 4}},
        },
        "candidate_freshness": {
            "repeated_unchanged_warning": True,
        },
        "next_recommended_action": "investigate importer path",
        "warnings": ["Candidate validation results unchanged across recent reports"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    reports_root = tmp_path / "reports"
    data_root = tmp_path / "data" / "bid_ask_csv"
    output_dir = tmp_path / "reports" / "daily_progress"

    _write_1s_csv(data_root / "historical_1s" / "BTC_1s.csv", "BTC", rows=10)
    _write_1s_csv(data_root / "historical_1s" / "ETH_1s.csv", "ETH", rows=8)
    _write_15s_csv(data_root / "historical_15s" / "BTC_15s.csv", "BTC", rows=4)
    _write_15s_csv(data_root / "historical_15s" / "ETH_15s.csv", "ETH", rows=3)

    validation_dir = data_root / "reports"
    _write_validation_report(validation_dir, "BTC", "1s", OLD_STAMP, row_count=8)
    _write_validation_report(
        validation_dir,
        "BTC",
        "1s",
        NEW_STAMP,
        row_count=10,
        passed=False,
        warnings=["10 missing second(s) in coverage range"],
    )
    _write_validation_report(validation_dir, "BTC", "15s", NEW_STAMP)
    _write_validation_report(validation_dir, "ETH", "1s", NEW_STAMP)

    candidate_dir = reports_root / "bid_ask_candidate_validation"
    _write_candidate_validation(
        candidate_dir,
        OLD_STAMP,
        candidate_name="BTC_DOWN_MICRO_H20_CANDIDATE_B",
        cumulative_net_bps=1.0,
        filtered_paper_trades=3,
    )
    _write_candidate_validation(
        candidate_dir,
        NEW_STAMP,
        candidate_name="BTC_DOWN_MICRO_H20_CANDIDATE_B",
        cumulative_net_bps=4.5,
        filtered_paper_trades=7,
    )
    _write_candidate_validation(
        candidate_dir,
        NEW_STAMP,
        candidate_name="BTC_DOWN_MICRO_H20_CANDIDATE_B_RELAXED",
        cumulative_net_bps=-50.0,
        filtered_paper_trades=25,
    )

    _write_phase3_audit(reports_root / "bid_ask_research_scheduler" / "phase3_audit_latest.json")

    return {
        "reports_root": reports_root,
        "data_root": data_root,
        "output_dir": output_dir,
    }


class TestBuildDailyProgressDigest:
    def test_digest_generated(self, workspace: dict[str, Path]) -> None:
        result = digest.build_and_write_digest(
            date_yyyymmdd=TEST_DATE,
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
        )

        json_dated = workspace["output_dir"] / f"daily_progress_{TEST_DATE}.json"
        md_dated = workspace["output_dir"] / f"daily_progress_{TEST_DATE}.md"
        json_latest = workspace["output_dir"] / digest.DEFAULT_LATEST_JSON_NAME
        md_latest = workspace["output_dir"] / digest.DEFAULT_LATEST_MD_NAME

        assert json_dated.is_file()
        assert md_dated.is_file()
        assert json_latest.is_file()
        assert md_latest.is_file()
        assert result["date"] == TEST_DATE
        assert result["what_changed_today"]["report_count"] >= 1
        assert "# Bid/Ask Daily Progress Digest" in md_dated.read_text(encoding="utf-8")

        payload = json.loads(json_dated.read_text(encoding="utf-8"))
        assert payload["generated_at"]
        assert payload["latest_timestamps"]["BTC"]["last_timestamp"]

    def test_missing_files_handled(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "daily_progress"
        result = digest.build_and_write_digest(
            date_yyyymmdd=TEST_DATE,
            reports_root=tmp_path / "missing_reports",
            data_root=tmp_path / "missing_data",
            output_dir=output_dir,
        )

        assert result["backfill_progress"]["available"] is False
        assert result["files_downloaded_today"]["available"] is False
        assert result["latest_candidate_result"]["available"] is False
        assert result["latest_relaxed_candidate_result"]["available"] is False
        assert (output_dir / digest.DEFAULT_LATEST_JSON_NAME).is_file()
        assert (output_dir / digest.DEFAULT_LATEST_MD_NAME).is_file()

    def test_candidate_summary_included(self, workspace: dict[str, Path]) -> None:
        result = digest.build_daily_progress_digest(
            date_yyyymmdd=TEST_DATE,
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
        )

        candidate = result["latest_candidate_result"]
        relaxed = result["latest_relaxed_candidate_result"]

        assert candidate["available"] is True
        assert candidate["candidate_name"] == "BTC_DOWN_MICRO_H20_CANDIDATE_B"
        assert candidate["filtered_paper_trades"] == 7
        assert candidate["cumulative_net_return_bps"] == 4.5

        assert relaxed["available"] is True
        assert "RELAXED" in relaxed["candidate_name"]
        assert relaxed["filtered_paper_trades"] == 25
        assert relaxed["cumulative_net_return_bps"] == -50.0

    def test_validation_warnings_included(self, workspace: dict[str, Path]) -> None:
        result = digest.build_daily_progress_digest(
            date_yyyymmdd=TEST_DATE,
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
        )

        btc_1s = result["validation_status"]["BTC"]["1s"]
        assert btc_1s["available"] is True
        assert btc_1s["passed"] is False
        assert btc_1s["warnings"]

        warning_text = " ".join(result["warnings"])
        assert "missing second" in warning_text.lower() or "10 missing" in warning_text
        assert any("Candidate validation results unchanged" in item for item in result["warnings"])

    def test_safety_flags_included(self, workspace: dict[str, Path]) -> None:
        result = digest.build_daily_progress_digest(
            date_yyyymmdd=TEST_DATE,
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
        )

        assert result["paper_only"] is True
        assert result["execution_enabled"] is False
        assert result["real_execution_ready"] is False
        assert result["safety_statement"]
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False

    def test_latest_name_files_written_as_regular_files(self, workspace: dict[str, Path]) -> None:
        digest.build_and_write_digest(
            date_yyyymmdd=TEST_DATE,
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
            latest_md_name="custom_latest.md",
            latest_json_name="custom_latest.json",
        )

        md_latest = workspace["output_dir"] / "custom_latest.md"
        json_latest = workspace["output_dir"] / "custom_latest.json"
        assert md_latest.is_file()
        assert json_latest.is_file()
        assert not md_latest.is_symlink()
        assert not json_latest.is_symlink()
        assert md_latest.read_text(encoding="utf-8")
        assert json.loads(json_latest.read_text(encoding="utf-8"))


class TestCli:
    def test_main_build_command(
        self,
        workspace: dict[str, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = digest.main(
            [
                "build",
                "--date",
                TEST_DATE,
                "--reports-root",
                str(workspace["reports_root"]),
                "--data-root",
                str(workspace["data_root"]),
                "--output-dir",
                str(workspace["output_dir"]),
            ]
        )

        captured = capsys.readouterr().out
        assert exit_code == 0
        assert f"date={TEST_DATE}" in captured
        assert "json_latest_path=" in captured
        assert "markdown_latest_path=" in captured
        assert "paper_only=true" in captured


def _block_network_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network/subprocess call attempted")

    monkeypatch.setattr("subprocess.run", _blocked)
    monkeypatch.setattr("subprocess.Popen", _blocked)
    monkeypatch.setattr("boto3.client", _blocked)


class TestSafety:
    def test_no_forbidden_imports(self) -> None:
        source = DIGEST_FILE.read_text(encoding="utf-8")
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
        source = DIGEST_FILE.read_text(encoding="utf-8").lower()
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
        source = DIGEST_FILE.read_text(encoding="utf-8").lower()
        forbidden = ("subprocess", "os.system", "popen")
        for pattern in forbidden:
            assert pattern not in source

    def test_no_network_subprocess_or_aws_at_runtime(
        self,
        workspace: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _block_network_subprocess(monkeypatch)
        digest.build_and_write_digest(
            date_yyyymmdd=TEST_DATE,
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
        )
