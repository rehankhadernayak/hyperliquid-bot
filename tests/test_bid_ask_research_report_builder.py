"""Tests for bid_ask_research_report_builder (offline, temp dirs, fake reports)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import bid_ask_research_report_builder as builder
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY

BUILDER_FILE = Path(__file__).resolve().parent.parent / "bid_ask_research_report_builder.py"


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


def _write_repair_report(
    reports_dir: Path,
    coin: str,
    interval: str,
    stamp: str,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{coin}_{interval}_repair_{stamp}.json"
    payload = {
        "coin": coin,
        "interval": interval,
        "rows_before": 100,
        "rows_after": 96,
        "mock_rows_removed": 4,
        "monotonic_after": True,
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
    min_confidence: float = 0.65,
    min_edge_bps: float = 2.0,
    trade_classes: list[str] | None = None,
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
        "win_rate_after_costs": 0.60,
        "positive_after_costs": cumulative_net_bps > 0,
        "min_confidence": min_confidence,
        "min_edge_bps": min_edge_bps,
        "trade_classes": trade_classes or ["DOWN"],
        "paper_only": True,
        "execution_enabled": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_candidate_validation(
    reports_dir: Path,
    stamp: str,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"candidate_validation_BTC_DOWN_MICRO_H20_CANDIDATE_B_{stamp}.json"
    payload = {
        "candidate_name": "BTC_DOWN_MICRO_H20_CANDIDATE_B",
        "filtered_paper_trades": 5,
        "cumulative_net_return_bps": 3.84,
        "validation_status": {
            "insufficient_trades": True,
            "not_market_ready": True,
            "unstable_concentrated": True,
        },
        "ready_for_more_research": True,
        "paper_only": True,
        "execution_enabled": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_training_report(training_dir: Path, coin: str, stamp: str) -> Path:
    training_dir.mkdir(parents=True, exist_ok=True)
    path = training_dir / f"{stamp}.json"
    payload = {
        "report_id": stamp,
        "coin": coin,
        "interval": "15s",
        "horizon": 20,
        "model_used": "RandomForestClassifier",
        "model_accuracy": 0.55,
        "baseline_accuracy": 0.50,
        "model_edge_vs_baseline": 0.05,
        "evaluation": {"train_rows": 1000, "test_rows": 250},
        "generated_at": "2024-01-01T00:00:00Z",
        "paper_only": True,
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
        "warnings": ["import stall detected"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_research_notes(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Notes\n\n## 10. Next Plan\n\n- Finish backfill\n- Inspect losing trades\n",
        encoding="utf-8",
    )


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    reports_root = tmp_path / "reports"
    data_root = tmp_path / "data" / "bid_ask_csv"
    output_dir = tmp_path / "reports" / "final_research_report"

    _write_1s_csv(data_root / "historical_1s" / "BTC_1s.csv", "BTC", rows=10)
    _write_1s_csv(data_root / "historical_1s" / "ETH_1s.csv", "ETH", rows=8)
    _write_15s_csv(data_root / "historical_15s" / "BTC_15s.csv", "BTC", rows=4)
    _write_15s_csv(data_root / "historical_15s" / "ETH_15s.csv", "ETH", rows=3)

    validation_dir = data_root / "reports"
    _write_validation_report(validation_dir, "BTC", "1s", "20240101T120000Z", row_count=12)
    _write_validation_report(validation_dir, "BTC", "15s", "20240101T100000Z")
    _write_validation_report(validation_dir, "ETH", "1s", "20240101T100000Z")
    _write_repair_report(validation_dir, "BTC", "1s", "20240101T100000Z")

    walk_forward_dir = reports_root / "bid_ask_walk_forward"
    _write_walk_forward_report(
        walk_forward_dir,
        "BTC",
        "20240102T100000Z",
        cumulative_net_bps=4.5,
        accepted_trades=7,
    )
    _write_walk_forward_report(
        walk_forward_dir,
        "BTC",
        "20260706T115051",
        cumulative_net_bps=3.84,
        accepted_trades=5,
    )
    _write_walk_forward_report(
        walk_forward_dir,
        "BTC",
        "20260706T142556",
        cumulative_net_bps=-101.82,
        accepted_trades=25,
        min_confidence=0.60,
        min_edge_bps=1.0,
    )
    _write_walk_forward_report(
        walk_forward_dir,
        "BTC",
        "20260706T120135",
        cumulative_net_bps=-267.95,
        accepted_trades=62,
        min_confidence=0.60,
        min_edge_bps=1.0,
        trade_classes=["DOWN", "UP"],
    )

    candidate_dir = reports_root / "bid_ask_candidate_validation"
    _write_candidate_validation(candidate_dir, "20240102T100000Z")

    training_dir = reports_root / "bid_ask_training"
    _write_training_report(training_dir, "BTC", "train_btc")
    _write_training_report(training_dir, "ETH", "train_eth")

    _write_phase3_audit(reports_root / "bid_ask_research_scheduler" / "phase3_audit_latest.json")
    _write_research_notes(reports_root / "research_notes" / "hyperliquid_bid_ask_research_notes.md")

    return {
        "reports_root": reports_root,
        "data_root": data_root,
        "output_dir": output_dir,
    }


class TestBuildResearchReport:
    def test_report_generated(self, workspace: dict[str, Path]) -> None:
        result = builder.build_research_report(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
        )

        latest_md = workspace["output_dir"] / builder.LATEST_MARKDOWN_NAME
        latest_json = workspace["output_dir"] / builder.LATEST_METADATA_NAME
        stamped_md = Path(result["stamped_markdown_report_path"])

        assert latest_md.is_file()
        assert latest_json.is_file()
        assert stamped_md.is_file()
        assert result["markdown_report_path"] == str(latest_md)
        assert result["section_count"] == len(builder.REPORT_SECTIONS)

    def test_required_sections_present(self, workspace: dict[str, Path]) -> None:
        result = builder.build_research_report(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
        )
        markdown = Path(result["markdown_report_path"]).read_text(encoding="utf-8")
        for section in builder.REPORT_SECTIONS:
            assert f"## {section}" in markdown

    def test_safety_wording_present(self, workspace: dict[str, Path]) -> None:
        result = builder.build_research_report(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
        )
        markdown = Path(result["markdown_report_path"]).read_text(encoding="utf-8")
        metadata = json.loads(
            (workspace["output_dir"] / builder.LATEST_METADATA_NAME).read_text(encoding="utf-8")
        )

        assert builder.PROFITABILITY_DISCLAIMER in markdown
        assert builder.MARKET_READINESS_DISCLAIMER in markdown
        assert "Real execution disabled" in markdown
        assert "not market ready" in markdown.lower()
        assert result["paper_only"] is True
        assert result["execution_enabled"] is False
        assert result["real_execution_ready"] is False
        assert metadata["market_ready"] is False

    def test_candidate_results_included(self, workspace: dict[str, Path]) -> None:
        result = builder.build_research_report(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
        )
        markdown = Path(result["markdown_report_path"]).read_text(encoding="utf-8")

        assert "Candidate B Results" in markdown
        assert "BTC_DOWN_MICRO_H20_CANDIDATE_B" in markdown
        assert "Relaxed Candidate Failure Analysis" in markdown
        assert "-101.82" in markdown or "-101.8" in markdown
        assert "Strict Candidate B" in markdown
        assert result["candidate_name"] == "BTC_DOWN_MICRO_H20_CANDIDATE_B"
        assert result["candidate_trades"] == 5

    def test_missing_data_handled(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "reports" / "final_research_report"
        result = builder.build_research_report(
            reports_root=tmp_path / "missing_reports",
            data_root=tmp_path / "missing_data",
            output_dir=output_dir,
        )
        markdown = Path(result["markdown_report_path"]).read_text(encoding="utf-8")

        assert result["section_count"] == len(builder.REPORT_SECTIONS)
        assert "Phase 3 audit report not available" in markdown
        assert "No walk-forward report found" in markdown or "report not available" in markdown
        assert result["imported_hours_mean"] == 0

    def test_metadata_json_written(self, workspace: dict[str, Path]) -> None:
        result = builder.build_research_report(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
        )
        metadata_path = workspace["output_dir"] / builder.LATEST_METADATA_NAME
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))

        assert payload["title"] == builder.DEFAULT_TITLE
        assert payload["sections"] == list(builder.REPORT_SECTIONS)
        assert payload["output_paths"]["markdown_latest"] == result["markdown_report_path"]
        assert payload["candidate_scenarios"]
        assert payload["blockers"]
        assert payload["profitability_disclaimer"] == builder.PROFITABILITY_DISCLAIMER


class TestNoProfitabilityClaim:
    def test_report_does_not_claim_profitability_or_market_readiness(
        self,
        workspace: dict[str, Path],
    ) -> None:
        result = builder.build_research_report(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
        )
        combined = (
            Path(result["markdown_report_path"]).read_text(encoding="utf-8")
            + json.dumps(result)
        ).lower()
        assert "do not prove future profitability" in combined
        assert "not market ready" in combined
        assert "market_ready" in combined
        assert "false" in combined


class TestCli:
    def test_main_build_command(self, workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = builder.main(
            [
                "build",
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
        assert "markdown_report_path=" in captured
        assert "metadata_report_path=" in captured
        assert "paper_only=True" in captured
        assert "real_execution_ready=False" in captured


def _block_network_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network/subprocess call attempted")

    monkeypatch.setattr("subprocess.run", _blocked)
    monkeypatch.setattr("subprocess.Popen", _blocked)
    monkeypatch.setattr("boto3.client", _blocked)


class TestSafety:
    def test_no_forbidden_imports(self) -> None:
        source = BUILDER_FILE.read_text(encoding="utf-8")
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
        source = BUILDER_FILE.read_text(encoding="utf-8").lower()
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
        source = BUILDER_FILE.read_text(encoding="utf-8").lower()
        forbidden = ("subprocess", "os.system", "popen")
        for pattern in forbidden:
            assert pattern not in source

    def test_output_dir_must_be_final_research_report(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="final_research_report"):
            builder.build_research_report(
                reports_root=tmp_path / "reports",
                data_root=tmp_path / "data",
                output_dir=tmp_path / "reports" / "other_dir",
            )

    def test_no_network_subprocess_or_aws_at_runtime(
        self,
        workspace: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _block_network_subprocess(monkeypatch)
        builder.build_research_report(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
        )
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False
