"""Focused offline tests for daily candidate validation runner."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY
from bid_ask_candidate_validator import (
    DEFAULT_CANDIDATE_NAME,
    DEFAULT_VALIDATION_REPORTS_DIR,
    build_candidate_config,
    build_candidate_validation_report,
    compute_validation_status,
    load_candidate_validation_reports,
    run_candidate_validation,
    summarize_candidate_validation_reports,
    write_candidate_validation_json,
    write_candidate_validation_markdown,
)

VALIDATOR_FILE = Path(__file__).resolve().parent.parent / "bid_ask_candidate_validator.py"


def _mock_walk_forward_report(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "json_report_path": "reports/bid_ask_walk_forward/walk_forward_BTC_test.json",
        "accepted_trades_csv_path": "reports/bid_ask_walk_forward/accepted_trades_BTC_test.csv",
        "beats_baseline_accuracy": True,
        "accuracy_edge_vs_baseline": 0.05,
        "filtered_paper_trades": 25,
        "cumulative_net_return_bps": 12.5,
        "average_net_return_bps": 0.5,
        "win_rate_after_costs": 0.62,
        "positive_after_costs": True,
        "ready_for_more_research": True,
        "stability_summary": {
            "concentration_warning": False,
            "sparse_stability_warning": False,
            "buckets_with_too_few_trades": 0,
        },
        "stability_warnings": [],
        "readiness": {
            "beats_baseline_accuracy": True,
            "positive_after_costs": True,
            "ready_for_more_research": True,
            "real_execution_ready": False,
        },
    }
    base.update(overrides)
    return base


def _passing_config(**overrides: Any) -> dict[str, Any]:
    config = build_candidate_config()
    config.update(overrides)
    return config


class TestRunWritesReports:
    def test_run_writes_json_and_markdown(self, tmp_path: Path) -> None:
        validation_dir = tmp_path / "validation"
        walk_forward_dir = tmp_path / "walk_forward"
        mock_runner = MagicMock(
            return_value=_mock_walk_forward_report(
                json_report_path=str(walk_forward_dir / "walk_forward_BTC_test.json"),
                accepted_trades_csv_path=str(walk_forward_dir / "accepted_trades_BTC_test.csv"),
            )
        )

        report = run_candidate_validation(
            validation_reports_dir=validation_dir,
            walk_forward_reports_dir=walk_forward_dir,
            run_timestamp="2026-07-03T08:00:00Z",
            walk_forward_runner=mock_runner,
        )

        mock_runner.assert_called_once()
        assert report["json_report_path"]
        assert report["markdown_report_path"]
        assert Path(report["json_report_path"]).is_file()
        assert Path(report["markdown_report_path"]).is_file()
        assert report["latest_walk_forward_report"] == str(walk_forward_dir / "walk_forward_BTC_test.json")
        assert report["accepted_trades_csv_path"] == str(
            walk_forward_dir / "accepted_trades_BTC_test.csv"
        )

        payload = json.loads(Path(report["json_report_path"]).read_text(encoding="utf-8"))
        assert payload["candidate_name"] == DEFAULT_CANDIDATE_NAME
        assert payload["paper_only"] is True
        assert payload["execution_enabled"] is False
        assert payload["real_execution_ready"] is False


class TestValidationStatusLogic:
    def test_paper_candidate_pass_when_all_checks_pass(self) -> None:
        status = compute_validation_status(
            beats_baseline_accuracy=True,
            positive_after_costs=True,
            filtered_paper_trades=25,
            stability_summary={"concentration_warning": False, "sparse_stability_warning": False},
            min_trades_required=20,
            require_positive=True,
        )
        assert status["paper_candidate_pass"] is True
        assert status["not_market_ready"] is False

    def test_insufficient_trades_triggers_not_market_ready(self) -> None:
        status = compute_validation_status(
            beats_baseline_accuracy=True,
            positive_after_costs=True,
            filtered_paper_trades=5,
            stability_summary={"concentration_warning": False, "sparse_stability_warning": False},
            min_trades_required=20,
            require_positive=True,
        )
        assert status["insufficient_trades"] is True
        assert status["not_market_ready"] is True
        assert status["paper_candidate_pass"] is False

    def test_concentration_warning_triggers_not_market_ready(self) -> None:
        status = compute_validation_status(
            beats_baseline_accuracy=True,
            positive_after_costs=True,
            filtered_paper_trades=25,
            stability_summary={"concentration_warning": True, "sparse_stability_warning": False},
            min_trades_required=20,
            require_positive=True,
        )
        assert status["unstable_concentrated"] is True
        assert status["not_market_ready"] is True
        assert status["paper_candidate_pass"] is False

    def test_sparse_warning_triggers_not_market_ready(self) -> None:
        status = compute_validation_status(
            beats_baseline_accuracy=True,
            positive_after_costs=True,
            filtered_paper_trades=25,
            stability_summary={"concentration_warning": False, "sparse_stability_warning": True},
            min_trades_required=20,
            require_positive=True,
        )
        assert status["unstable_sparse"] is True
        assert status["not_market_ready"] is True
        assert status["paper_candidate_pass"] is False

    def test_negative_after_costs_triggers_not_market_ready(self) -> None:
        status = compute_validation_status(
            beats_baseline_accuracy=True,
            positive_after_costs=False,
            filtered_paper_trades=25,
            stability_summary={"concentration_warning": False, "sparse_stability_warning": False},
            min_trades_required=20,
            require_positive=True,
        )
        assert status["negative_after_costs"] is True
        assert status["not_market_ready"] is True
        assert status["paper_candidate_pass"] is False

    def test_build_report_maps_walk_forward_fields(self) -> None:
        report = build_candidate_validation_report(
            _mock_walk_forward_report(filtered_paper_trades=30),
            config=_passing_config(),
            run_timestamp="2026-07-03T08:00:00Z",
        )
        assert report["validation_status"]["paper_candidate_pass"] is True
        assert report["filtered_paper_trades"] == 30
        assert report["real_execution_ready"] is False


class TestSummarize:
    def test_summarize_handles_no_reports(self, tmp_path: Path) -> None:
        summary = summarize_candidate_validation_reports(reports_dir=tmp_path / "missing")
        assert summary == "No candidate validation reports found."

    def test_summarize_prints_latest_and_trend(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "validation"
        reports_dir.mkdir(parents=True)

        for index, trades in enumerate((10, 25, 30)):
            report = build_candidate_validation_report(
                _mock_walk_forward_report(
                    filtered_paper_trades=trades,
                    cumulative_net_return_bps=float(index + 1),
                    average_net_return_bps=0.1 * (index + 1),
                    win_rate_after_costs=0.5 + index * 0.05,
                    accuracy_edge_vs_baseline=0.01 * (index + 1),
                    stability_summary={
                        "concentration_warning": index == 0,
                        "sparse_stability_warning": index == 0,
                    },
                ),
                config=_passing_config(min_trades_required=20),
                run_timestamp=f"2026-07-0{index + 1}T08:00:00Z",
            )
            write_candidate_validation_json(report, reports_dir=reports_dir)

        summary = summarize_candidate_validation_reports(reports_dir=reports_dir)
        assert "Latest report:" in summary
        assert "2026-07-03T08:00:00Z" in summary
        assert "Best cumulative_net_return_bps:" in summary
        assert "3.0000" in summary
        assert "Passes:" in summary
        assert "Trend (last 3 runs):" in summary
        assert "paper_candidate_pass" in summary
        assert "not_market_ready" in summary


class TestSafety:
    def test_safety_flags_remain_paper_only(self) -> None:
        report = build_candidate_validation_report(
            _mock_walk_forward_report(),
            config=_passing_config(),
            run_timestamp="2026-07-03T08:00:00Z",
        )
        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False

    def test_no_forbidden_imports(self) -> None:
        source = VALIDATOR_FILE.read_text(encoding="utf-8")
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
        source = VALIDATOR_FILE.read_text(encoding="utf-8").lower()
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

    def test_no_subprocess_calls(self) -> None:
        source = VALIDATOR_FILE.read_text(encoding="utf-8").lower()
        forbidden = ("subprocess", "os.system", "popen")
        for pattern in forbidden:
            assert pattern not in source

    def test_paper_only_constants(self) -> None:
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False
        assert DEFAULT_VALIDATION_REPORTS_DIR == Path("reports/bid_ask_candidate_validation")


class TestReportWriters:
    def test_markdown_includes_validation_status(self, tmp_path: Path) -> None:
        report = build_candidate_validation_report(
            _mock_walk_forward_report(),
            config=_passing_config(),
            run_timestamp="2026-07-03T08:00:00Z",
        )
        path = write_candidate_validation_markdown(report, reports_dir=tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "Validation Status" in text
        assert "paper_candidate_pass" in text

    def test_load_reports_sorted_by_timestamp(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "validation"
        for stamp in ("2026-07-01T08:00:00Z", "2026-07-02T08:00:00Z"):
            report = build_candidate_validation_report(
                _mock_walk_forward_report(),
                config=_passing_config(),
                run_timestamp=stamp,
            )
            write_candidate_validation_json(report, reports_dir=reports_dir)

        loaded = load_candidate_validation_reports(reports_dir=reports_dir)
        assert len(loaded) == 2
        assert loaded[0]["run_timestamp"] == "2026-07-01T08:00:00Z"
        assert loaded[1]["run_timestamp"] == "2026-07-02T08:00:00Z"
