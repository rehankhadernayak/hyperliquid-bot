"""Focused offline tests for daily candidate validation runner."""

from __future__ import annotations

import ast
import json
import pytest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY
from bid_ask_candidate_validator import (
    DEFAULT_CANDIDATE_NAME,
    DEFAULT_LAUNCHD_LABEL,
    DEFAULT_SLICE_MODES,
    DEFAULT_SOURCE,
    DEFAULT_VALIDATION_REPORTS_DIR,
    LAUNCHD_SETUP_MD,
    ROBUSTNESS_REPORT_PREFIX,
    _build_arg_parser,
    build_candidate_config,
    build_candidate_validation_report,
    build_robustness_from_prior_reports,
    build_row_slices,
    cmd_write_launchd_template,
    compute_robustness_summary,
    compute_validation_status,
    load_candidate_validation_reports,
    main,
    parse_slice_modes,
    run_candidate_validation,
    run_robustness_validation,
    summarize_candidate_validation_reports,
    write_candidate_validation_json,
    write_candidate_validation_markdown,
    write_robustness_json,
    write_robustness_markdown,
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


class TestSource:
    def test_build_candidate_config_defaults_to_historical(self) -> None:
        config = build_candidate_config()
        assert config["source"] == "historical"
        assert DEFAULT_SOURCE == "historical"

    def test_default_run_passes_historical_to_evaluator(self, tmp_path: Path) -> None:
        mock_runner = MagicMock(return_value=_mock_walk_forward_report())
        run_candidate_validation(
            validation_reports_dir=tmp_path / "validation",
            walk_forward_reports_dir=tmp_path / "walk_forward",
            walk_forward_runner=mock_runner,
        )
        mock_runner.assert_called_once()
        assert mock_runner.call_args.kwargs["source"] == "historical"

    def test_run_passes_combined_source_to_evaluator(self, tmp_path: Path) -> None:
        mock_runner = MagicMock(return_value=_mock_walk_forward_report())
        run_candidate_validation(
            source="combined",
            validation_reports_dir=tmp_path / "validation",
            walk_forward_reports_dir=tmp_path / "walk_forward",
            walk_forward_runner=mock_runner,
        )
        assert mock_runner.call_args.kwargs["source"] == "combined"

    def test_run_passes_local_source_to_evaluator(self, tmp_path: Path) -> None:
        mock_runner = MagicMock(return_value=_mock_walk_forward_report())
        run_candidate_validation(
            source="local",
            validation_reports_dir=tmp_path / "validation",
            walk_forward_reports_dir=tmp_path / "walk_forward",
            walk_forward_runner=mock_runner,
        )
        assert mock_runner.call_args.kwargs["source"] == "local"

    def test_cli_default_source_is_historical(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["run"])
        assert args.source == "historical"

    def test_cli_accepts_combined_source(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["run", "--source", "combined"])
        assert args.source == "combined"

    def test_cli_accepts_local_source(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["run", "--source", "local"])
        assert args.source == "local"

    def test_cli_rejects_invalid_source(self) -> None:
        parser = _build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--source", "invalid"])

    def test_json_report_config_includes_source(self, tmp_path: Path) -> None:
        mock_runner = MagicMock(return_value=_mock_walk_forward_report())
        report = run_candidate_validation(
            source="historical",
            validation_reports_dir=tmp_path / "validation",
            walk_forward_reports_dir=tmp_path / "walk_forward",
            run_timestamp="2026-07-03T08:00:00Z",
            walk_forward_runner=mock_runner,
        )
        payload = json.loads(Path(report["json_report_path"]).read_text(encoding="utf-8"))
        assert payload["config"]["source"] == "historical"

    def test_main_run_passes_source_to_evaluator(self, tmp_path: Path, monkeypatch) -> None:
        captured: dict[str, Any] = {}

        def fake_run_candidate_validation(**kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            report = build_candidate_validation_report(
                _mock_walk_forward_report(),
                config=build_candidate_config(source=kwargs["source"]),
                run_timestamp="2026-07-03T08:00:00Z",
            )
            write_candidate_validation_json(report, reports_dir=kwargs["validation_reports_dir"])
            write_candidate_validation_markdown(report, reports_dir=kwargs["validation_reports_dir"])
            return report

        monkeypatch.setattr(
            "bid_ask_candidate_validator.run_candidate_validation",
            fake_run_candidate_validation,
        )
        exit_code = main(
            [
                "run",
                "--source",
                "historical",
                "--data-root",
                str(tmp_path / "data"),
                "--reports-dir",
                str(tmp_path / "validation"),
                "--walk-forward-reports-dir",
                str(tmp_path / "walk_forward"),
            ]
        )
        assert exit_code == 0
        assert captured["source"] == "historical"

    def test_safety_flags_remain_paper_only_with_source(self, tmp_path: Path) -> None:
        mock_runner = MagicMock(return_value=_mock_walk_forward_report())
        report = run_candidate_validation(
            source="historical",
            validation_reports_dir=tmp_path / "validation",
            walk_forward_reports_dir=tmp_path / "walk_forward",
            walk_forward_runner=mock_runner,
        )
        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False


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
        assert "- source: historical" in summary
        assert "Best cumulative_net_return_bps:" in summary
        assert "3.0000" in summary
        assert "Passes:" in summary
        assert "Trend (last 3 runs):" in summary
        assert "timestamp | source | trades" in summary
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


class TestRobustnessValidation:
    def _run_with_mock_slices(
        self,
        tmp_path: Path,
        mock_runner,
        *,
        slice_modes: str = "recent_halves",
        total_rows: int = 1000,
    ) -> dict[str, Any]:
        return run_robustness_validation(
            validation_reports_dir=tmp_path / "validation",
            walk_forward_reports_dir=tmp_path / "walk_forward",
            run_timestamp="2026-07-03T09:00:00Z",
            slice_modes=slice_modes,
            total_rows=total_rows,
            walk_forward_runner=mock_runner,
        )

    def test_robustness_writes_json_and_markdown(self, tmp_path: Path) -> None:
        mock_runner = MagicMock(return_value=_mock_walk_forward_report())
        report = self._run_with_mock_slices(tmp_path, mock_runner, slice_modes="recent_halves")

        mock_runner.assert_called()
        assert report["json_report_path"]
        assert report["markdown_report_path"]
        assert Path(report["json_report_path"]).is_file()
        assert Path(report["markdown_report_path"]).is_file()
        assert ROBUSTNESS_REPORT_PREFIX in Path(report["json_report_path"]).name
        assert DEFAULT_CANDIDATE_NAME in Path(report["json_report_path"]).name

        payload = json.loads(Path(report["json_report_path"]).read_text(encoding="utf-8"))
        assert payload["evaluation_mode"] == "data_slicing"
        assert len(payload["slices"]) == 2
        assert payload["paper_only"] is True
        assert payload["execution_enabled"] is False
        assert payload["real_execution_ready"] is False

    def test_robustness_pass_false_when_any_slice_negative(self, tmp_path: Path) -> None:
        mock_runner = MagicMock(
            side_effect=[
                _mock_walk_forward_report(positive_after_costs=True),
                _mock_walk_forward_report(positive_after_costs=False, cumulative_net_return_bps=-1.0),
            ]
        )
        report = self._run_with_mock_slices(tmp_path, mock_runner, slice_modes="recent_halves")
        assert report["robustness_pass"] is False
        assert report["not_market_ready"] is True

    def test_robustness_pass_false_when_insufficient_trades(self, tmp_path: Path) -> None:
        mock_runner = MagicMock(
            side_effect=[
                _mock_walk_forward_report(filtered_paper_trades=25),
                _mock_walk_forward_report(filtered_paper_trades=5),
            ]
        )
        report = self._run_with_mock_slices(tmp_path, mock_runner, slice_modes="recent_halves")
        assert report["robustness_pass"] is False
        summary = report["robustness_summary"]
        assert summary["slices_with_enough_trades"] == 1

    def test_robustness_pass_false_when_stability_warnings(self, tmp_path: Path) -> None:
        mock_runner = MagicMock(
            side_effect=[
                _mock_walk_forward_report(),
                _mock_walk_forward_report(
                    stability_summary={
                        "concentration_warning": True,
                        "sparse_stability_warning": False,
                    },
                    stability_warnings=["concentrated returns"],
                ),
            ]
        )
        report = self._run_with_mock_slices(tmp_path, mock_runner, slice_modes="recent_halves")
        assert report["robustness_pass"] is False
        summary = report["robustness_summary"]
        assert summary["slices_without_stability_warnings"] == 1

    def test_robustness_pass_true_when_all_slices_pass(self, tmp_path: Path) -> None:
        mock_runner = MagicMock(return_value=_mock_walk_forward_report())
        report = self._run_with_mock_slices(
            tmp_path,
            mock_runner,
            slice_modes="recent_halves,recent_thirds",
            total_rows=900,
        )
        assert report["robustness_pass"] is True
        assert report["not_market_ready"] is False
        summary = report["robustness_summary"]
        assert summary["slices_tested"] == 5
        assert summary["slices_positive_after_costs"] == 5
        assert summary["slices_beating_baseline"] == 5
        assert summary["slices_with_enough_trades"] == 5
        assert summary["slices_without_stability_warnings"] == 5

    def test_robustness_handles_no_prior_reports_fallback(self, tmp_path: Path) -> None:
        def missing_data_loader(*_args, **_kwargs):
            raise FileNotFoundError("missing csv")

        report = run_robustness_validation(
            validation_reports_dir=tmp_path / "validation",
            walk_forward_reports_dir=tmp_path / "walk_forward",
            run_timestamp="2026-07-03T09:00:00Z",
            slice_modes="recent_halves",
            data_loader=missing_data_loader,
        )
        assert report["evaluation_mode"] == "trend_fallback"
        assert report["slices"] == []
        assert report["robustness_pass"] is False
        assert report["robustness_summary"]["slices_tested"] == 0

    def test_trend_fallback_uses_prior_reports(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "validation"
        reports_dir.mkdir(parents=True)
        for stamp in ("2026-07-01T08:00:00Z", "2026-07-02T08:00:00Z"):
            prior = build_candidate_validation_report(
                _mock_walk_forward_report(),
                config=_passing_config(),
                run_timestamp=stamp,
            )
            write_candidate_validation_json(prior, reports_dir=reports_dir)

        def missing_data_loader(*_args, **_kwargs):
            raise FileNotFoundError("missing csv")

        report = run_robustness_validation(
            validation_reports_dir=reports_dir,
            walk_forward_reports_dir=tmp_path / "walk_forward",
            run_timestamp="2026-07-03T09:00:00Z",
            slice_modes="recent_halves",
            data_loader=missing_data_loader,
        )
        assert report["evaluation_mode"] == "trend_fallback"
        assert len(report["slices"]) == 2
        assert report["robustness_pass"] is True

    def test_slice_modes_and_row_slices(self) -> None:
        modes = parse_slice_modes(DEFAULT_SLICE_MODES)
        assert modes == ["recent_halves", "recent_thirds", "expanding"]
        slices = build_row_slices(900, modes)
        assert len(slices) == 7
        names = {item["slice_name"] for item in slices}
        assert "half_first" in names
        assert "third_last" in names
        assert "expand_100" in names
        row_ranges = {(item["row_start"], item["row_end"]) for item in slices}
        assert len(row_ranges) == len(slices)

    def test_compute_robustness_summary_requires_all_slices(self) -> None:
        passing = {
            "positive_after_costs": True,
            "beats_baseline_accuracy": True,
            "validation_status": {
                "insufficient_trades": False,
                "unstable_concentrated": False,
                "unstable_sparse": False,
            },
        }
        assert compute_robustness_summary([passing, passing])["robustness_pass"] is True
        failing = dict(passing)
        failing["beats_baseline_accuracy"] = False
        assert compute_robustness_summary([passing, failing])["robustness_pass"] is False

    def test_cli_robustness_defaults(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["robustness"])
        assert args.source == "historical"
        assert args.slices == DEFAULT_SLICE_MODES

    def test_main_robustness_command(self, tmp_path: Path, monkeypatch) -> None:
        captured: dict[str, Any] = {}

        def fake_run_robustness_validation(**kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            report = {
                "candidate_name": DEFAULT_CANDIDATE_NAME,
                "evaluation_mode": "data_slicing",
                "robustness_summary": {"slices_tested": 2, "robustness_pass": True},
            }
            write_robustness_json(
                {
                    **report,
                    "run_timestamp": "2026-07-03T09:00:00Z",
                    "config": build_candidate_config(),
                    "slice_modes": ["recent_halves"],
                    "slices": [],
                    "robustness_pass": True,
                    "not_market_ready": False,
                    "paper_only": True,
                    "execution_enabled": False,
                    "real_execution_ready": False,
                },
                reports_dir=kwargs["validation_reports_dir"],
            )
            return report

        monkeypatch.setattr(
            "bid_ask_candidate_validator.run_robustness_validation",
            fake_run_robustness_validation,
        )
        exit_code = main(
            [
                "robustness",
                "--source",
                "historical",
                "--slices",
                "recent_halves",
                "--reports-dir",
                str(tmp_path / "validation"),
                "--data-root",
                str(tmp_path / "data"),
            ]
        )
        assert exit_code == 0
        assert captured["source"] == "historical"
        assert captured["slice_modes"] == "recent_halves"

    def test_build_robustness_from_prior_reports_filters_candidate(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "validation"
        reports_dir.mkdir(parents=True)
        other = build_candidate_validation_report(
            _mock_walk_forward_report(),
            config=_passing_config(candidate_name="OTHER"),
            run_timestamp="2026-07-01T08:00:00Z",
        )
        write_candidate_validation_json(other, reports_dir=reports_dir)
        slices = build_robustness_from_prior_reports(
            candidate_name=DEFAULT_CANDIDATE_NAME,
            reports_dir=reports_dir,
            config=_passing_config(),
        )
        assert slices == []


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
        assert "- source: historical" in text

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


class TestLaunchdTemplate:
    def test_writes_plist_and_setup(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "validation"
        plist_path, setup_path = cmd_write_launchd_template(
            reports_dir=reports_dir,
            project_root=tmp_path,
        )
        assert plist_path.name == f"{DEFAULT_LAUNCHD_LABEL}.plist"
        assert setup_path.name == LAUNCHD_SETUP_MD
        assert plist_path.is_file()
        assert setup_path.is_file()

    def test_start_calendar_interval_defaults(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "validation"
        plist_path, _ = cmd_write_launchd_template(
            reports_dir=reports_dir,
            project_root=tmp_path,
        )
        plist_text = plist_path.read_text(encoding="utf-8")
        assert "<key>StartCalendarInterval</key>" in plist_text
        assert "<key>Hour</key>" in plist_text
        assert "<integer>23</integer>" in plist_text
        assert "<key>Minute</key>" in plist_text
        assert "<integer>30</integer>" in plist_text

    def test_program_arguments_run_historical(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "validation"
        plist_path, _ = cmd_write_launchd_template(
            reports_dir=reports_dir,
            project_root=tmp_path,
        )
        plist_text = plist_path.read_text(encoding="utf-8")
        assert "bid_ask_candidate_validator.py" in plist_text
        assert "<string>run</string>" in plist_text
        assert "<string>--source</string>" in plist_text
        assert "<string>historical</string>" in plist_text

    def test_custom_hour_minute_source(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "validation"
        plist_path, _ = cmd_write_launchd_template(
            hour=6,
            minute=15,
            source="combined",
            reports_dir=reports_dir,
            project_root=tmp_path,
        )
        plist_text = plist_path.read_text(encoding="utf-8")
        assert "<integer>6</integer>" in plist_text
        assert "<integer>15</integer>" in plist_text
        assert "<string>combined</string>" in plist_text

    def test_custom_label_and_log_paths(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "validation"
        stdout_log = reports_dir / "custom.out.log"
        stderr_log = reports_dir / "custom.err.log"
        plist_path, _ = cmd_write_launchd_template(
            label="com.example.validator",
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            reports_dir=reports_dir,
            project_root=tmp_path,
        )
        plist_text = plist_path.read_text(encoding="utf-8")
        assert "<string>com.example.validator</string>" in plist_text
        assert str(stdout_log) in plist_text
        assert str(stderr_log) in plist_text

    def test_run_at_load_false_by_default(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "validation"
        plist_path, _ = cmd_write_launchd_template(
            reports_dir=reports_dir,
            project_root=tmp_path,
        )
        plist_text = plist_path.read_text(encoding="utf-8")
        assert "<key>RunAtLoad</key>" not in plist_text

    def test_run_at_load_true(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "validation"
        plist_path, _ = cmd_write_launchd_template(
            run_at_load=True,
            reports_dir=reports_dir,
            project_root=tmp_path,
        )
        plist_text = plist_path.read_text(encoding="utf-8")
        assert "<key>RunAtLoad</key>" in plist_text
        assert "<true/>" in plist_text

    def test_setup_includes_copy_load_tail_summarize(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "validation"
        plist_path, setup_path = cmd_write_launchd_template(
            reports_dir=reports_dir,
            project_root=tmp_path,
        )
        setup_text = setup_path.read_text(encoding="utf-8")
        assert "cp " in setup_text
        assert "~/Library/LaunchAgents/" in setup_text
        assert "launchctl unload" in setup_text
        assert "launchctl load" in setup_text
        assert "launchctl list | grep hyperliquid" in setup_text
        assert "tail -50" in setup_text
        assert "bid_ask_candidate_validator.py summarize" in setup_text
        assert str(plist_path) in setup_text
        assert "research-only" in setup_text.lower()
        assert "paper-only" in setup_text.lower()

    def test_does_not_auto_install_launchd(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "validation"
        launch_agents = tmp_path / "Library" / "LaunchAgents"
        plist_path, setup_path = cmd_write_launchd_template(
            reports_dir=reports_dir,
            project_root=tmp_path,
        )
        assert plist_path.parent == reports_dir
        assert setup_path.parent == reports_dir
        assert plist_path.is_file()
        assert not launch_agents.exists()

    def test_main_write_launchd_template(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "validation"
        exit_code = main(
            [
                "write-launchd-template",
                "--reports-dir",
                str(reports_dir),
                "--project-root",
                str(tmp_path),
            ]
        )
        assert exit_code == 0
        assert (reports_dir / f"{DEFAULT_LAUNCHD_LABEL}.plist").is_file()
        assert (reports_dir / LAUNCHD_SETUP_MD).is_file()

    def test_cli_custom_hour_minute(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(
            ["write-launchd-template", "--hour", "8", "--minute", "45", "--source", "local"]
        )
        assert args.hour == 8
        assert args.minute == 45
        assert args.source == "local"

    def test_safety_fields_unaffected_by_launchd_template(self, tmp_path: Path) -> None:
        report = build_candidate_validation_report(
            _mock_walk_forward_report(),
            config=_passing_config(),
            run_timestamp="2026-07-03T08:00:00Z",
        )
        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False
        cmd_write_launchd_template(reports_dir=tmp_path / "validation", project_root=tmp_path)
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False
