"""Tests for bid_ask_dashboard_artifact_refresher (offline, temp dirs, no network)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import bid_ask_dashboard_artifact_refresher as refresher
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY

REFRESHER_FILE = Path(__file__).resolve().parent.parent / "bid_ask_dashboard_artifact_refresher.py"


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
        "missing_periods_count": 0,
        "warnings": [],
        "passed": passed,
        "generated_at": "2024-01-01T00:00:00Z",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_walk_forward_report(
    reports_dir: Path,
    coin: str,
    stamp: str,
    *,
    cumulative_net_bps: float = 3.84,
    accepted_trades: int = 5,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"walk_forward_{coin}_{stamp}.json"
    payload = {
        "coins": [coin],
        "interval": "15s",
        "horizon": 20,
        "completed_windows": 3,
        "model_accuracy_mean": 0.55,
        "baseline_accuracy_mean": 0.50,
        "accuracy_edge_vs_baseline": 0.05,
        "cumulative_net_return_bps": cumulative_net_bps,
        "filtered_paper_trades": accepted_trades,
        "positive_after_costs": cumulative_net_bps > 0,
        "paper_only": True,
        "execution_enabled": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_candidate_validation(
    reports_dir: Path,
    stamp: str,
    *,
    csv_path: str | None = None,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"candidate_validation_BTC_DOWN_MICRO_H20_CANDIDATE_B_{stamp}.json"
    payload = {
        "candidate_name": "BTC_DOWN_MICRO_H20_CANDIDATE_B",
        "filtered_paper_trades": 5,
        "cumulative_net_return_bps": 3.84,
        "accepted_trades_csv_path": csv_path,
        "validation_status": {
            "insufficient_trades": True,
            "not_market_ready": True,
        },
        "paper_only": True,
        "execution_enabled": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_phase3_audit(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2024-01-01T00:00:00Z",
        "paper_only": True,
        "execution_enabled": False,
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
            "BTC": {"15s": {"row_count": 20}},
            "ETH": {"15s": {"row_count": 15}},
        },
        "downloads": {"total_files": 42},
        "scheduler_progress": {"latest_action": "chunk_completed"},
        "phase_config": {"phase_backfill_name": "phase3_btc_eth_recent"},
        "next_recommended_action": "continue backfill",
        "warnings": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    reports_root = tmp_path / "reports"
    data_root = tmp_path / "data" / "bid_ask_csv"
    output_dir = tmp_path / "reports" / "dashboard_artifacts"

    _write_1s_csv(data_root / "historical_1s" / "BTC_1s.csv", "BTC", rows=10)
    _write_1s_csv(data_root / "historical_1s" / "ETH_1s.csv", "ETH", rows=8)
    _write_15s_csv(data_root / "historical_15s" / "BTC_15s.csv", "BTC", rows=4)
    _write_15s_csv(data_root / "historical_15s" / "ETH_15s.csv", "ETH", rows=3)

    validation_dir = data_root / "reports"
    _write_validation_report(validation_dir, "BTC", "1s", "20240101T100000Z", row_count=10)
    _write_validation_report(validation_dir, "BTC", "1s", "20240101T120000Z", row_count=12)
    _write_validation_report(validation_dir, "BTC", "15s", "20240101T100000Z")
    _write_validation_report(validation_dir, "ETH", "1s", "20240101T100000Z")

    walk_forward_dir = reports_root / "bid_ask_walk_forward"
    _write_walk_forward_report(walk_forward_dir, "BTC", "20240101T100000Z", cumulative_net_bps=1.0)
    _write_walk_forward_report(walk_forward_dir, "BTC", "20240102T100000Z", cumulative_net_bps=4.5, accepted_trades=7)

    accepted_csv = walk_forward_dir / "accepted_trades_BTC_20240102T100000Z.csv"
    accepted_csv.write_text("timestamp_iso,coin\n2024-01-01T00:00:00Z,BTC\n", encoding="utf-8")

    candidate_dir = reports_root / "bid_ask_candidate_validation"
    _write_candidate_validation(
        candidate_dir,
        "20240102T100000Z",
        csv_path=str(accepted_csv),
    )

    _write_phase3_audit(reports_root / "bid_ask_research_scheduler" / "phase3_audit_latest.json")

    return {
        "reports_root": reports_root,
        "data_root": data_root,
        "output_dir": output_dir,
    }


class TestRefreshDashboardArtifacts:
    def test_writes_json_and_markdown(self, workspace: dict[str, Path]) -> None:
        summary = refresher.refresh_dashboard_artifacts(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
        )

        json_path = workspace["output_dir"] / refresher.DEFAULT_JSON_NAME
        md_path = workspace["output_dir"] / refresher.DEFAULT_MARKDOWN_NAME
        assert json_path.is_file()
        assert md_path.is_file()
        assert summary["output_paths"]["json"] == str(json_path)
        assert summary["output_paths"]["markdown"] == str(md_path)

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["generated_at"]
        assert "Phase Backfill" in md_path.read_text(encoding="utf-8")

    def test_handles_missing_files_gracefully(self, tmp_path: Path) -> None:
        summary = refresher.refresh_dashboard_artifacts(
            reports_root=tmp_path / "missing_reports",
            data_root=tmp_path / "missing_data",
            output_dir=tmp_path / "out",
        )

        assert summary["phase_backfill"]["available"] is False
        assert summary["latest_walk_forward"]["available"] is False
        assert summary["latest_candidate_validation"]["available"] is False
        assert summary["latest_accepted_trades_csv_path"] is None
        assert (tmp_path / "out" / refresher.DEFAULT_JSON_NAME).is_file()

    def test_includes_safety_flags(self, workspace: dict[str, Path]) -> None:
        summary = refresher.build_dashboard_summary(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
        )
        assert summary["paper_only"] is True
        assert summary["execution_enabled"] is False
        assert summary["real_execution_ready"] is False
        assert summary["market_ready"] is False
        assert summary["safety_statement"]
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False

    def test_summarizes_row_counts_and_latest_timestamps(self, workspace: dict[str, Path]) -> None:
        summary = refresher.build_dashboard_summary(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
        )
        btc = next(row for row in summary["data_coverage"]["coins"] if row["coin"] == "BTC")
        eth = next(row for row in summary["data_coverage"]["coins"] if row["coin"] == "ETH")

        assert btc["rows_1s"] == 100
        assert btc["rows_15s"] == 20
        assert btc["last_timestamp"] == "2024-01-01T01:00:00Z"
        assert eth["rows_1s"] == 80
        assert eth["last_timestamp"] == "2024-01-01T00:30:00Z"

        btc_1s = summary["validation"]["by_coin_interval"]["BTC"]["1s"]
        assert btc_1s["available"] is True
        assert btc_1s["row_count"] == 12

    def test_finds_latest_walk_forward_report(self, workspace: dict[str, Path]) -> None:
        summary = refresher.build_dashboard_summary(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
        )
        latest = summary["latest_walk_forward"]
        assert latest["stamp"] == "20240102T100000Z"
        assert latest["cumulative_net_bps"] == 4.5
        assert latest["accepted_trades"] == 7
        assert summary["latest_accepted_trades_csv_path"].endswith("accepted_trades_BTC_20240102T100000Z.csv")
        assert summary["next_recommended_action"] == "continue backfill"

    def test_candidate_b_includes_fallback_when_reports_missing(self, tmp_path: Path) -> None:
        summary = refresher.build_dashboard_summary(
            reports_root=tmp_path / "reports",
            data_root=tmp_path / "data",
        )
        scenarios = summary["candidate_b"]["scenarios"]
        assert len(scenarios) == 3
        assert all(row["source"] == "fallback" for row in scenarios)


def _block_network_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network/subprocess call attempted")

    monkeypatch.setattr("subprocess.run", _blocked)
    monkeypatch.setattr("subprocess.Popen", _blocked)
    monkeypatch.setattr("boto3.client", _blocked)


class TestSafety:
    def test_no_forbidden_imports(self) -> None:
        source = REFRESHER_FILE.read_text(encoding="utf-8")
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
        source = REFRESHER_FILE.read_text(encoding="utf-8").lower()
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
        source = REFRESHER_FILE.read_text(encoding="utf-8").lower()
        forbidden = ("subprocess", "os.system", "popen")
        for pattern in forbidden:
            assert pattern not in source

    def test_no_network_subprocess_or_aws_at_runtime(
        self,
        workspace: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _block_network_subprocess(monkeypatch)
        refresher.refresh_dashboard_artifacts(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
        )
