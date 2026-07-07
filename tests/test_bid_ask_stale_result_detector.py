"""Tests for bid_ask_stale_result_detector (offline temp dirs, no network)."""

from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import bid_ask_stale_result_detector as detector
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY

DETECTOR_FILE = Path(__file__).resolve().parent.parent / "bid_ask_stale_result_detector.py"
REFERENCE_TIME = datetime(2026, 7, 7, 12, 0, 0, tzinfo=timezone.utc)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
) -> Path:
    path = reports_dir / f"{coin}_{interval}_validation_{stamp}.json"
    _write_json(
        path,
        {
            "coin": coin,
            "interval": interval,
            "row_count": 10,
            "passed": True,
            "generated_at": "2026-07-07T00:00:00Z",
        },
    )
    return path


def _write_walk_forward_report(
    reports_dir: Path,
    coin: str,
    stamp: str,
    *,
    cumulative_net_bps: float = 3.84,
    accepted_trades: int = 5,
    csv_path: str | None = "reports/bid_ask_walk_forward/accepted_trades_BTC.csv",
    completed_windows: int = 3,
) -> Path:
    path = reports_dir / f"walk_forward_{coin}_{stamp}.json"
    _write_json(
        path,
        {
            "coins": [coin],
            "interval": "15s",
            "completed_windows": completed_windows,
            "cumulative_net_return_bps": cumulative_net_bps,
            "filtered_paper_trades": accepted_trades,
            "accepted_trades_csv_path": csv_path,
            "generated_at": "2026-07-07T00:00:00Z",
        },
    )
    return path


def _write_candidate_validation(
    reports_dir: Path,
    stamp: str,
    *,
    trades: int = 5,
    cumulative_net_bps: float = 3.84,
    status: str = "not_market_ready",
) -> Path:
    path = reports_dir / f"candidate_validation_BTC_DOWN_MICRO_H20_CANDIDATE_B_{stamp}.json"
    validation_status = {"paper_candidate_pass": True} if status == "paper_candidate_pass" else {
        "not_market_ready": True
    }
    _write_json(
        path,
        {
            "candidate_name": "BTC_DOWN_MICRO_H20_CANDIDATE_B",
            "run_timestamp": f"2026-07-07T{stamp[:2]}:{stamp[2:4]}:{stamp[4:6]}Z",
            "filtered_paper_trades": trades,
            "cumulative_net_return_bps": cumulative_net_bps,
            "validation_status": validation_status,
        },
    )
    return path


def _write_phase3_audit(
    path: Path,
    *,
    generated_at: str = "2026-07-07T11:30:00Z",
    one_s_last_ms: int = 1_700_000_360_000,
    fifteen_s_last_ms: int = 1_700_000_360_000,
) -> None:
    _write_json(
        path,
        {
            "generated_at": generated_at,
            "imported_historical_1s": {
                "BTC": {
                    "last_timestamp_ms": one_s_last_ms,
                    "freshness_status": "advancing",
                },
                "ETH": {
                    "last_timestamp_ms": one_s_last_ms,
                    "freshness_status": "advancing",
                },
            },
            "resampled_coverage": {
                "BTC": {"15s": {"last_timestamp_ms": fifteen_s_last_ms, "exists": True}},
                "ETH": {"15s": {"last_timestamp_ms": fifteen_s_last_ms, "exists": True}},
            },
        },
    )


def _write_dashboard_summary(path: Path, *, generated_at: str) -> None:
    _write_json(
        path,
        {
            "generated_at": generated_at,
            "paper_only": True,
            "execution_enabled": False,
        },
    )


def _write_scheduler_run(
    reports_root: Path,
    stamp: str,
    *,
    last_timestamp_ms: int,
) -> None:
    path = reports_root / "bid_ask_research_scheduler" / "runs" / f"bid_ask_scheduler_autopilot_{stamp}.json"
    coverage = {
        "BTC": {"last_timestamp_ms": last_timestamp_ms},
        "ETH": {"last_timestamp_ms": last_timestamp_ms},
    }
    _write_json(
        path,
        {
            "generated_at": "2026-07-07T10:00:00Z",
            "post_import_coverage": coverage,
        },
    )


def _build_healthy_workspace(tmp_path: Path) -> dict[str, Path]:
    reports_root = tmp_path / "reports"
    data_root = tmp_path / "data" / "bid_ask_csv"
    output_dir = tmp_path / "reports" / "stale_result_detector"

    _write_1s_csv(data_root / "historical_1s" / "BTC_1s.csv", "BTC", rows=10, base_ts=1_700_000_000_000)
    _write_1s_csv(data_root / "historical_1s" / "ETH_1s.csv", "ETH", rows=8, base_ts=1_700_000_000_000)
    _write_15s_csv(
        data_root / "historical_15s" / "BTC_15s.csv",
        "BTC",
        rows=4,
        base_ts=1_700_000_000_000,
    )
    _write_15s_csv(
        data_root / "historical_15s" / "ETH_15s.csv",
        "ETH",
        rows=3,
        base_ts=1_700_000_000_000,
    )

    validation_dir = data_root / "reports"
    for coin in ("BTC", "ETH"):
        for interval in ("1s", "15s"):
            _write_validation_report(validation_dir, coin, interval, "20260707T100000Z")

    walk_forward_dir = reports_root / "bid_ask_walk_forward"
    accepted_csv = walk_forward_dir / "accepted_trades_BTC_20260707T100000Z.csv"
    accepted_csv.parent.mkdir(parents=True, exist_ok=True)
    accepted_csv.write_text("timestamp_iso,coin\n2026-07-07T00:00:00Z,BTC\n", encoding="utf-8")
    _write_walk_forward_report(
        walk_forward_dir,
        "BTC",
        "20260707T090000Z",
        cumulative_net_bps=1.0,
        accepted_trades=3,
        csv_path=str(accepted_csv),
        completed_windows=2,
    )
    _write_walk_forward_report(
        walk_forward_dir,
        "BTC",
        "20260707T100000Z",
        cumulative_net_bps=4.5,
        accepted_trades=7,
        csv_path=str(accepted_csv),
        completed_windows=4,
    )

    candidate_dir = reports_root / "bid_ask_candidate_validation"
    _write_candidate_validation(candidate_dir, "20260707T090000Z", trades=3, cumulative_net_bps=1.0)
    _write_candidate_validation(candidate_dir, "20260707T100000Z", trades=7, cumulative_net_bps=4.5)

    _write_phase3_audit(
        reports_root / "bid_ask_research_scheduler" / "phase3_audit_latest.json",
        generated_at="2026-07-07T11:30:00Z",
        one_s_last_ms=1_700_000_009_000,
        fifteen_s_last_ms=1_700_000_009_000,
    )
    _write_dashboard_summary(
        reports_root / "dashboard_artifacts" / "dashboard_summary_latest.json",
        generated_at="2026-07-07T11:45:00Z",
    )
    _write_scheduler_run(
        reports_root,
        "20260707T080000Z",
        last_timestamp_ms=1_700_000_005_000,
    )

    return {
        "reports_root": reports_root,
        "data_root": data_root,
        "output_dir": output_dir,
    }


def _codes(report: dict) -> set[str]:
    return {item["code"] for item in report.get("stale_items", [])}


class TestStaleResultDetector:
    def test_detects_old_dashboard_artifact(self, tmp_path: Path) -> None:
        workspace = _build_healthy_workspace(tmp_path)
        _write_dashboard_summary(
            workspace["reports_root"] / "dashboard_artifacts" / "dashboard_summary_latest.json",
            generated_at="2026-07-07T00:00:00Z",
        )

        report = detector.build_stale_result_audit(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            max_artifact_age_hours=6,
            reference_time=REFERENCE_TIME,
        )

        assert "stale_dashboard_artifact" in _codes(report)

    def test_detects_repeated_candidate_results(self, tmp_path: Path) -> None:
        workspace = _build_healthy_workspace(tmp_path)
        candidate_dir = workspace["reports_root"] / "bid_ask_candidate_validation"
        for stamp in ("20260707T110000Z", "20260707T111000Z", "20260707T112000Z"):
            _write_candidate_validation(
                candidate_dir,
                stamp,
                trades=5,
                cumulative_net_bps=3.84,
                status="not_market_ready",
            )

        report = detector.build_stale_result_audit(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            candidate_repeat_threshold=3,
            reference_time=REFERENCE_TIME,
        )

        assert "candidate_result_unchanged" in _codes(report)

    def test_detects_missing_accepted_trades_csv_path(self, tmp_path: Path) -> None:
        workspace = _build_healthy_workspace(tmp_path)
        walk_forward_dir = workspace["reports_root"] / "bid_ask_walk_forward"
        _write_walk_forward_report(
            walk_forward_dir,
            "BTC",
            "20260707T120000Z",
            accepted_trades=9,
            csv_path="reports/bid_ask_walk_forward/missing_accepted_trades.csv",
        )

        report = detector.build_stale_result_audit(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            reference_time=REFERENCE_TIME,
        )

        assert "missing_accepted_trades_csv" in _codes(report)
        assert report["status"] == detector.STATUS_CRITICAL

    def test_detects_missing_reports(self, tmp_path: Path) -> None:
        reports_root = tmp_path / "reports"
        data_root = tmp_path / "data" / "bid_ask_csv"
        _write_dashboard_summary(
            reports_root / "dashboard_artifacts" / "dashboard_summary_latest.json",
            generated_at="2026-07-07T11:45:00Z",
        )
        _write_phase3_audit(
            reports_root / "bid_ask_research_scheduler" / "phase3_audit_latest.json",
            generated_at="2026-07-07T11:30:00Z",
        )

        report = detector.build_stale_result_audit(
            reports_root=reports_root,
            data_root=data_root,
            reference_time=REFERENCE_TIME,
        )

        codes = _codes(report)
        assert "missing_candidate_report" in codes
        assert "missing_validation_report" in codes

    def test_healthy_case(self, tmp_path: Path) -> None:
        workspace = _build_healthy_workspace(tmp_path)

        report = detector.run_stale_result_check(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
            reference_time=REFERENCE_TIME,
        )

        assert report["status"] == detector.STATUS_HEALTHY
        assert report["stale_items"] == []
        assert (workspace["output_dir"] / detector.DEFAULT_JSON_NAME).is_file()
        assert (workspace["output_dir"] / detector.DEFAULT_MARKDOWN_NAME).is_file()

    def test_includes_safety_flags(self, tmp_path: Path) -> None:
        workspace = _build_healthy_workspace(tmp_path)

        report = detector.build_stale_result_audit(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            reference_time=REFERENCE_TIME,
        )

        safety = report["safety_flags"]
        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert safety["paper_only"] is True
        assert safety["execution_enabled"] is False
        assert safety["read_only"] is True
        assert safety["no_network_calls"] is True
        assert safety["no_shell_commands"] is True
        assert safety["research_only"] is True
        assert safety["safety_statement"]
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False


def _block_network_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network/subprocess call attempted")

    monkeypatch.setattr("subprocess.run", _blocked)
    monkeypatch.setattr("subprocess.Popen", _blocked)
    monkeypatch.setattr("boto3.client", _blocked)


class TestSafety:
    def test_no_forbidden_imports(self) -> None:
        source = DETECTOR_FILE.read_text(encoding="utf-8")
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
        source = DETECTOR_FILE.read_text(encoding="utf-8").lower()
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
        source = DETECTOR_FILE.read_text(encoding="utf-8").lower()
        forbidden = ("subprocess", "os.system", "popen")
        for pattern in forbidden:
            assert pattern not in source

    def test_no_network_subprocess_or_aws_at_runtime(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        workspace = _build_healthy_workspace(tmp_path)
        _block_network_subprocess(monkeypatch)
        detector.run_stale_result_check(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
            reference_time=REFERENCE_TIME,
        )
