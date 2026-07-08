"""Focused offline tests for candidate grid runner."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY
from bid_ask_candidate_grid_runner import (
    DEFAULT_COINS,
    DEFAULT_HORIZONS,
    DEFAULT_MIN_CONFIDENCES,
    DEFAULT_MIN_TRADES_AFTER_REGIME_FILTER,
    DEFAULT_MIN_TRADES_REQUIRED,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRADE_CLASSES,
    CandidateGridConfig,
    _build_arg_parser,
    build_grid_combinations,
    build_grid_report,
    compute_paper_score,
    evaluate_grid_config,
    format_hours_label,
    format_top_n_summary,
    main,
    parse_hours_value,
    parse_max_cost_bps_value,
    parse_regime_filter_csv_values,
    rank_grid_results,
    run_candidate_grid,
    write_grid_report_csv,
    write_grid_report_json,
    write_grid_report_markdown,
)

GRID_RUNNER_FILE = Path(__file__).resolve().parent.parent / "bid_ask_candidate_grid_runner.py"


def _sample_config(**overrides: Any) -> CandidateGridConfig:
    base = CandidateGridConfig(
        coin="BTC",
        interval="15s",
        horizon=20,
        feature_set="microstructure",
        label_mode="direction",
        trade_class="DOWN",
        min_confidence=0.65,
        min_edge_bps=2.0,
        cost_multiplier=1.25,
        max_spread_bps=8.0,
        fee_bps=2.0,
        slippage_bps=1.0,
        source="historical",
    )
    if overrides:
        return CandidateGridConfig(**{**base.__dict__, **overrides})
    return base


def _mock_walk_forward_report(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "beats_baseline_accuracy": True,
        "accuracy_edge_vs_baseline": 0.05,
        "filtered_paper_trades": 25,
        "cumulative_net_return_bps": 12.5,
        "average_net_return_bps": 0.5,
        "win_rate_after_costs": 0.62,
        "positive_after_costs": True,
        "stability_summary": {
            "concentration_warning": False,
            "sparse_stability_warning": False,
        },
        "stability_warnings": [],
        "readiness": {
            "beats_baseline_accuracy": True,
            "positive_after_costs": True,
            "real_execution_ready": False,
        },
    }
    base.update(overrides)
    return base


def _result_from_metrics(config: CandidateGridConfig, **metrics: Any) -> dict[str, Any]:
    report = _mock_walk_forward_report(**metrics)
    score = compute_paper_score(
        {
            "positive_after_costs": report["positive_after_costs"],
            "beats_baseline_accuracy": report["beats_baseline_accuracy"],
            "filtered_paper_trades": report["filtered_paper_trades"],
            "average_net_return_bps": report["average_net_return_bps"],
            "stability_summary": report["stability_summary"],
            "stability_warnings": report["stability_warnings"],
        }
    )
    return {
        "config_id": config.config_id,
        "config": config.__dict__,
        **report,
        **score,
        "paper_only": True,
        "execution_enabled": False,
        "real_execution_ready": False,
    }


class TestGridCount:
    def test_default_grid_count(self) -> None:
        configs = build_grid_combinations()
        expected = (
            len(DEFAULT_COINS)
            * len(DEFAULT_HORIZONS)
            * len(DEFAULT_TRADE_CLASSES)
            * len(DEFAULT_MIN_CONFIDENCES)
            * 3  # min_edge_bps values
            * 3  # cost multipliers
            * 2  # max spread bps values
        )
        assert len(configs) == expected
        assert len(configs) == 648

    def test_small_grid_count(self) -> None:
        configs = build_grid_combinations(
            coins=("BTC",),
            horizons=(10,),
            trade_classes=("DOWN",),
            min_confidences=(0.55,),
            min_edge_bps_values=(0.0,),
            cost_multipliers=(1.0,),
            max_spread_bps_values=(8.0,),
        )
        assert len(configs) == 1


class TestMaxRunsCap:
    def test_max_runs_caps_evaluation(self) -> None:
        configs = build_grid_combinations(
            coins=("BTC",),
            horizons=(10, 20),
            trade_classes=("DOWN",),
            min_confidences=(0.55, 0.60),
            min_edge_bps_values=(0.0,),
            cost_multipliers=(1.0,),
            max_spread_bps_values=(8.0,),
        )
        assert len(configs) == 4
        mock_runner = MagicMock(return_value=_mock_walk_forward_report())
        output = run_candidate_grid(
            configs,
            max_runs=2,
            walk_forward_runner=mock_runner,
        )
        assert output["planned_count"] == 2
        assert output["completed_count"] == 2
        assert mock_runner.call_count == 2


class TestDryRun:
    def test_dry_run_does_not_evaluate(self) -> None:
        configs = build_grid_combinations(
            coins=("BTC",),
            horizons=(10,),
            trade_classes=("DOWN",),
            min_confidences=(0.55,),
            min_edge_bps_values=(0.0,),
            cost_multipliers=(1.0,),
            max_spread_bps_values=(8.0,),
        )
        mock_runner = MagicMock(return_value=_mock_walk_forward_report())
        output = run_candidate_grid(
            configs,
            dry_run=True,
            walk_forward_runner=mock_runner,
        )
        assert output["dry_run"] is True
        assert output["planned_count"] == 1
        assert output["results"] == []
        mock_runner.assert_not_called()

    def test_main_dry_run_prints_sample(self, capsys) -> None:
        exit_code = main(
            [
                "run",
                "--dry-run",
                "--coins",
                "BTC",
                "--horizons",
                "10",
                "--trade-classes",
                "DOWN",
                "--min-confidences",
                "0.55",
                "--min-edge-bps-values",
                "0",
                "--cost-multipliers",
                "1.0",
                "--max-spread-bps-values",
                "8",
            ]
        )
        captured = capsys.readouterr().out
        assert exit_code == 0
        assert "dry_run=true" in captured
        assert "planned_count=1" in captured
        assert "sample_configs" in captured


class TestRanking:
    def test_positive_beats_negative(self) -> None:
        config = _sample_config()
        positive = _result_from_metrics(config, positive_after_costs=True, average_net_return_bps=0.5)
        negative = _result_from_metrics(
            config,
            positive_after_costs=False,
            average_net_return_bps=-1.0,
            cumulative_net_return_bps=-5.0,
        )
        ranked = rank_grid_results([negative, positive])
        assert ranked[0]["positive_after_costs"] is True
        assert ranked[0]["paper_score"] > ranked[1]["paper_score"]

    def test_enough_trades_beats_sparse(self) -> None:
        config = _sample_config()
        enough = _result_from_metrics(config, filtered_paper_trades=30)
        sparse = _result_from_metrics(
            config,
            filtered_paper_trades=5,
            stability_summary={
                "concentration_warning": False,
                "sparse_stability_warning": False,
            },
        )
        ranked = rank_grid_results([sparse, enough])
        assert ranked[0]["filtered_paper_trades"] == 30
        assert ranked[0]["paper_score"] > ranked[1]["paper_score"]

    def test_warnings_penalized(self) -> None:
        config = _sample_config()
        clean = _result_from_metrics(config)
        warned = _result_from_metrics(
            config,
            stability_summary={
                "concentration_warning": True,
                "sparse_stability_warning": False,
            },
            stability_warnings=["concentrated returns"],
        )
        ranked = rank_grid_results([warned, clean])
        assert ranked[0]["stability_warning_count"] == 0
        assert ranked[1]["stability_warning_count"] >= 1
        assert ranked[0]["paper_score"] > ranked[1]["paper_score"]


class TestPaperScore:
    def test_insufficient_trades_penalty(self) -> None:
        score = compute_paper_score(
            {
                "positive_after_costs": True,
                "beats_baseline_accuracy": True,
                "filtered_paper_trades": 5,
                "average_net_return_bps": 1.0,
                "stability_summary": {},
                "stability_warnings": [],
            },
            min_trades_required=DEFAULT_MIN_TRADES_REQUIRED,
        )
        assert score["insufficient_trades"] is True
        assert score["insufficient_trades_penalty"] > 0
        assert score["real_execution_ready"] is False


class TestOutputsWritten:
    def test_writes_json_csv_markdown(self, tmp_path: Path) -> None:
        config = _sample_config()
        mock_runner = MagicMock(return_value=_mock_walk_forward_report())
        grid_output = run_candidate_grid(
            [config],
            walk_forward_runner=mock_runner,
        )
        report = build_grid_report(
            grid_output=grid_output,
            run_timestamp="2026-07-07T08:00:00Z",
            grid_params={"coins": ("BTC",)},
            top_n=5,
        )
        json_path = write_grid_report_json(report, output_dir=tmp_path)
        csv_path = write_grid_report_csv(report, output_dir=tmp_path)
        md_path = write_grid_report_markdown(report, output_dir=tmp_path)

        assert json_path.is_file()
        assert csv_path.is_file()
        assert md_path.is_file()
        assert str(tmp_path) in str(json_path)
        assert str(tmp_path) in str(csv_path)
        assert str(tmp_path) in str(md_path)

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["completed_count"] == 1
        assert payload["paper_only"] is True
        assert len(payload["ranked"]) == 1

        with csv_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["config_id"] == config.config_id

        markdown = md_path.read_text(encoding="utf-8")
        assert "Candidate Grid Report" in markdown
        assert config.config_id in markdown

    def test_main_run_writes_reports(self, tmp_path: Path, monkeypatch) -> None:
        def fake_run_candidate_grid(configs, **kwargs: Any) -> dict[str, Any]:
            config = _sample_config()
            result = _result_from_metrics(config)
            return {
                "dry_run": False,
                "planned_count": 1,
                "total_combinations": 1,
                "completed_count": 1,
                "failed_count": 0,
                "results": [result],
                "failures": [],
                "ranked": [result],
            }

        monkeypatch.setattr(
            "bid_ask_candidate_grid_runner.run_candidate_grid",
            fake_run_candidate_grid,
        )
        output_dir = tmp_path / "candidate_grid"
        exit_code = main(
            [
                "run",
                "--coins",
                "BTC",
                "--horizons",
                "20",
                "--trade-classes",
                "DOWN",
                "--min-confidences",
                "0.65",
                "--min-edge-bps-values",
                "2",
                "--cost-multipliers",
                "1.25",
                "--max-spread-bps-values",
                "8",
                "--output-dir",
                str(output_dir),
            ]
        )
        assert exit_code == 0
        assert list(output_dir.glob("candidate_grid_*.json"))
        assert list(output_dir.glob("candidate_grid_*.csv"))
        assert list(output_dir.glob("candidate_grid_*.md"))


class TestSafety:
    def test_safety_flags_in_report(self) -> None:
        config = _sample_config()
        result = _result_from_metrics(config)
        report = build_grid_report(
            grid_output={
                "dry_run": False,
                "planned_count": 1,
                "total_combinations": 1,
                "completed_count": 1,
                "failed_count": 0,
                "results": [result],
                "failures": [],
                "ranked": [result],
            },
            run_timestamp="2026-07-07T08:00:00Z",
            grid_params={},
        )
        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False
        assert result["real_execution_ready"] is False

    def test_evaluate_grid_config_uses_mock_runner(self) -> None:
        mock_runner = MagicMock(return_value=_mock_walk_forward_report())
        result = evaluate_grid_config(_sample_config(), walk_forward_runner=mock_runner)
        mock_runner.assert_called_once()
        assert mock_runner.call_args.kwargs["save_reports"] is False
        assert mock_runner.call_args.kwargs["save_accepted_trades"] is False
        assert result["paper_only"] is True
        assert result["real_execution_ready"] is False

    def test_no_forbidden_imports(self) -> None:
        source = GRID_RUNNER_FILE.read_text(encoding="utf-8")
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
        source = GRID_RUNNER_FILE.read_text(encoding="utf-8").lower()
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
        source = GRID_RUNNER_FILE.read_text(encoding="utf-8").lower()
        forbidden = ("subprocess", "os.system", "popen")
        for pattern in forbidden:
            assert pattern not in source

    def test_default_output_dir(self) -> None:
        assert DEFAULT_OUTPUT_DIR == Path("reports/candidate_grid")

    def test_paper_only_constants(self) -> None:
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False


class TestCLI:
    def test_cli_defaults(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["run"])
        assert args.source == "historical"
        assert args.interval == "15s"
        assert args.output_dir == DEFAULT_OUTPUT_DIR
        assert args.dry_run is False
        assert args.exclude_hours_values == "none"
        assert args.include_hours_values == "none"
        assert args.max_cost_bps_values == "none"
        assert args.min_trades_after_regime_filter == DEFAULT_MIN_TRADES_AFTER_REGIME_FILTER

    def test_format_top_n_summary(self) -> None:
        config = _sample_config()
        ranked = [
            _result_from_metrics(config, average_net_return_bps=1.0),
            _result_from_metrics(config, average_net_return_bps=0.1),
        ]
        summary = format_top_n_summary(ranked, top_n=2)
        assert "Top 2 configs" in summary
        assert config.config_id in summary


class TestRegimeFilterParsing:
    def test_parse_none(self) -> None:
        assert parse_hours_value("none") is None
        assert parse_max_cost_bps_value("none") is None
        assert parse_regime_filter_csv_values("none") == ("none",)

    def test_parse_3_plus_5(self) -> None:
        assert parse_hours_value("3+5") == (3, 5)
        assert format_hours_label((3, 5)) == "3+5"

    def test_parse_3_plus_5_plus_22(self) -> None:
        assert parse_hours_value("3+5+22") == (3, 5, 22)


class TestRegimeFilterGrid:
    def test_grid_count_expands_with_regime_filters(self) -> None:
        configs = build_grid_combinations(
            coins=("BTC",),
            horizons=(10,),
            trade_classes=("DOWN",),
            min_confidences=(0.55,),
            min_edge_bps_values=(0.0,),
            cost_multipliers=(1.0,),
            max_spread_bps_values=(8.0,),
            exclude_hours_values=("none", "5", "3+5"),
            include_hours_values=("none",),
            max_cost_bps_values=("none",),
        )
        assert len(configs) == 3
        exclude_sets = {config.exclude_hours for config in configs}
        assert exclude_sets == {None, (5,), (3, 5)}

    def test_evaluator_receives_exclude_hours(self) -> None:
        config = _sample_config(exclude_hours=(3, 5))
        mock_runner = MagicMock(return_value=_mock_walk_forward_report(
            regime_filters_enabled=True,
            trades_before_regime_filters=12,
            trades_after_regime_filters=6,
            trades_removed_by_regime_filters=6,
        ))

        def runner_with_regime_filters(**kwargs: Any) -> dict[str, Any]:
            return mock_runner(**kwargs)

        evaluate_grid_config(config, walk_forward_runner=runner_with_regime_filters)
        assert mock_runner.call_args.kwargs["exclude_hours"] == [3, 5]
        assert mock_runner.call_args.kwargs["include_hours"] is None
        assert mock_runner.call_args.kwargs["max_cost_bps"] is None

    def test_evaluator_receives_max_cost_bps(self) -> None:
        config = _sample_config(max_cost_bps=5.0)
        mock_runner = MagicMock(return_value=_mock_walk_forward_report(
            regime_filters_enabled=True,
            trades_before_regime_filters=10,
            trades_after_regime_filters=8,
            trades_removed_by_regime_filters=2,
        ))

        def runner_with_regime_filters(**kwargs: Any) -> dict[str, Any]:
            return mock_runner(**kwargs)

        evaluate_grid_config(config, walk_forward_runner=runner_with_regime_filters)
        assert mock_runner.call_args.kwargs["max_cost_bps"] == 5.0

    def test_output_columns_include_regime_filter_fields(self, tmp_path: Path) -> None:
        config = _sample_config(exclude_hours=(3, 5), max_cost_bps=5.0)
        mock_runner = MagicMock(
            return_value=_mock_walk_forward_report(
                regime_filters_enabled=True,
                trades_before_regime_filters=12,
                trades_after_regime_filters=6,
                trades_removed_by_regime_filters=6,
            )
        )

        def runner_with_regime_filters(**kwargs: Any) -> dict[str, Any]:
            return mock_runner(**kwargs)

        result = evaluate_grid_config(config, walk_forward_runner=runner_with_regime_filters)
        grid_output = {
            "dry_run": False,
            "planned_count": 1,
            "total_combinations": 1,
            "completed_count": 1,
            "failed_count": 0,
            "results": [result],
            "failures": [],
            "ranked": [result],
        }
        report = build_grid_report(
            grid_output=grid_output,
            run_timestamp="2026-07-08T08:00:00Z",
            grid_params={},
        )
        csv_path = write_grid_report_csv(report, output_dir=tmp_path)
        with csv_path.open(encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        assert row["exclude_hours"] == "3+5"
        assert row["max_cost_bps"] == "5.0"
        assert row["trades_before_regime_filters"] == "12"
        assert row["trades_after_regime_filters"] == "6"
        assert row["trades_removed_by_regime_filters"] == "6"
        assert row["regime_filters_enabled"] == "True"
        assert row["exploratory_overfit_warning"] == "True"

    def test_overfit_warning_true_when_filter_used(self) -> None:
        config = _sample_config(exclude_hours=(5,))
        mock_runner = MagicMock(return_value=_mock_walk_forward_report())
        result = evaluate_grid_config(config, walk_forward_runner=mock_runner)
        assert result["exploratory_overfit_warning"] is True
        assert result["regime_filters_enabled"] is True

    def test_no_overfit_warning_when_no_filter(self) -> None:
        config = _sample_config()
        mock_runner = MagicMock(return_value=_mock_walk_forward_report())
        result = evaluate_grid_config(config, walk_forward_runner=mock_runner)
        assert result["exploratory_overfit_warning"] is False

    def test_min_trades_penalty_applies(self) -> None:
        score = compute_paper_score(
            {
                "positive_after_costs": True,
                "beats_baseline_accuracy": True,
                "filtered_paper_trades": 25,
                "average_net_return_bps": 1.0,
                "stability_summary": {},
                "stability_warnings": [],
                "regime_filters_enabled": True,
                "trades_after_regime_filters": 3,
            },
            min_trades_after_regime_filter=DEFAULT_MIN_TRADES_AFTER_REGIME_FILTER,
        )
        assert score["insufficient_regime_trades"] is True
        assert score["insufficient_regime_trades_penalty"] > 0

    def test_markdown_labels_exploratory_configs(self, tmp_path: Path) -> None:
        config = _sample_config(exclude_hours=(5,))
        result = evaluate_grid_config(
            config,
            walk_forward_runner=MagicMock(return_value=_mock_walk_forward_report()),
        )
        report = build_grid_report(
            grid_output={
                "dry_run": False,
                "planned_count": 1,
                "total_combinations": 1,
                "completed_count": 1,
                "failed_count": 0,
                "results": [result],
                "failures": [],
                "ranked": [result],
            },
            run_timestamp="2026-07-08T08:00:00Z",
            grid_params={},
            top_n=1,
        )
        md_path = write_grid_report_markdown(report, output_dir=tmp_path)
        markdown = md_path.read_text(encoding="utf-8")
        assert "Exploratory" in markdown
        assert "EXPLORATORY REGIME FILTER WARNING" in markdown
        assert "overfit-prone" in markdown

    def test_ranking_uses_post_filter_metrics_when_regime_filters_enabled(self) -> None:
        baseline_config = _sample_config()
        filtered_config = _sample_config(exclude_hours=(5,))

        baseline_result = _result_from_metrics(
            baseline_config,
            filtered_paper_trades=10,
            cumulative_net_return_bps=20.0,
            average_net_return_bps=2.0,
            positive_after_costs=True,
            regime_filters_enabled=False,
        )
        filtered_result = _result_from_metrics(
            filtered_config,
            filtered_paper_trades=8,
            cumulative_net_return_bps=30.0,
            average_net_return_bps=3.75,
            positive_after_costs=True,
            regime_filters_enabled=True,
            trades_before_regime_filters=10,
            trades_after_regime_filters=8,
            trades_removed_by_regime_filters=2,
            cumulative_net_return_before_regime_filters=20.0,
            cumulative_net_return_after_regime_filters=30.0,
        )

        ranked = rank_grid_results([baseline_result, filtered_result])
        assert ranked[0]["config_id"] == filtered_config.config_id
        assert ranked[0]["cumulative_net_return_bps"] == pytest.approx(30.0)
        assert ranked[0]["filtered_paper_trades"] == 8
