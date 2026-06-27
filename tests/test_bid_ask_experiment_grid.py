"""Focused offline tests for bid/ask experiment grid."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bid_ask_experiment_grid import (
    DEFAULT_GRID_REPORTS_DIR,
    ExperimentConfig,
    build_grid_report,
    default_experiment_grid,
    main,
    rank_experiments,
    run_experiment_grid,
    run_single_experiment,
    write_grid_report_json,
    write_grid_report_markdown,
)
from bid_ask_resampler import RESAMPLED_COLUMNS, resampled_csv_path
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY

GRID_FILE = Path(__file__).resolve().parent.parent / "bid_ask_experiment_grid.py"

BASE_MS = 1_718_000_040_000


def _make_resampled_rows(
    *,
    coin: str = "BTC",
    count: int,
    start_ms: int = BASE_MS,
    step_ms: int = 15_000,
    mid_start: float = 100.0,
    mid_step: float = 0.5,
) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        start = start_ms + index * step_ms
        end = start + step_ms - 1
        mid = mid_start + index * mid_step
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
                "avg_spread_bps": 2.0 + (index % 3) * 0.1,
                "avg_bbo_imbalance": 0.2 + index * 0.01,
                "row_count": 15,
                "missing_seconds": 0,
            }
        )
    return rows


def _write_resampled_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS)).to_csv(path, index=False)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "bid_ask_csv"


def _seed_coin_csv(data_root: Path, coin: str, interval: str, count: int = 40) -> None:
    step_ms = 15_000 if interval == "15s" else 60_000
    path = resampled_csv_path(coin, interval, data_root)
    _write_resampled_csv(path, _make_resampled_rows(coin=coin, count=count, step_ms=step_ms))


def _fake_training_runner(**kwargs: Any) -> dict[str, Any]:
    return {
        "report": {
            "coin": kwargs["coin"],
            "interval": kwargs["interval"],
            "horizon": kwargs["horizon"],
            "threshold_bps": kwargs["threshold_bps"],
            "model_used": "FakeModel",
            "model_accuracy": 0.55,
            "baseline_accuracy": 0.40,
            "model_edge_vs_baseline": 0.15,
            "model_artifact_path": None,
            "warnings": ["Research-only evaluation; does not prove profitability"],
        },
        "evaluation": {
            "train_rows": 28,
            "test_rows": 12,
        },
    }


class TestDefaultGrid:
    def test_default_experiment_grid(self) -> None:
        grid = default_experiment_grid()
        assert len(grid) == 5
        assert grid[0] == ExperimentConfig(
            coin="BTC",
            interval="15s",
            horizon=1,
            threshold_bps=1.0,
            notes="BTC 15s short-horizon tight threshold",
        )
        assert grid[1].coin == "BTC"
        assert grid[1].interval == "1m"
        assert grid[1].horizon == 1
        assert grid[1].threshold_bps == 3.0
        assert grid[2] == ExperimentConfig(
            coin="BTC",
            interval="1m",
            horizon=5,
            threshold_bps=5.0,
            notes="BTC 1m medium-horizon wider threshold",
        )
        assert grid[3].coin == "ETH"
        assert grid[4].coin == "SOL"


class TestSingleExperiment:
    def test_run_single_experiment_result_format(self, data_root: Path, tmp_path: Path) -> None:
        _seed_coin_csv(data_root, "BTC", "15s")
        config = ExperimentConfig(
            coin="BTC",
            interval="15s",
            horizon=1,
            threshold_bps=1.0,
            notes="test note",
        )
        result = run_single_experiment(
            config,
            data_root=data_root,
            reports_dir=tmp_path / "reports",
            models_dir=tmp_path / "models",
        )
        expected_keys = {
            "experiment_id",
            "coin",
            "interval",
            "horizon",
            "threshold_bps",
            "train_rows",
            "test_rows",
            "model_used",
            "model_accuracy",
            "baseline_accuracy",
            "model_edge_vs_baseline",
            "model_artifact_path",
            "warnings",
        }
        assert expected_keys.issubset(result.keys())
        assert result["coin"] == "BTC"
        assert result["interval"] == "15s"
        assert result["test_rows"] > 0
        assert result["paper_only"] is True
        assert result["execution_enabled"] is False
        assert any("Config note: test note" in warning for warning in result["warnings"])


class TestGridExecution:
    def test_grid_continues_after_failure(self) -> None:
        configs = [
            ExperimentConfig(coin="BTC", interval="15s", horizon=1, threshold_bps=1.0),
            ExperimentConfig(coin="MISSING", interval="15s", horizon=1, threshold_bps=1.0),
            ExperimentConfig(coin="ETH", interval="1m", horizon=5, threshold_bps=5.0),
        ]

        def runner(**kwargs: Any) -> dict[str, Any]:
            coin = kwargs["coin"]
            if coin == "MISSING":
                raise FileNotFoundError("Training CSV not found")
            return _fake_training_runner(**kwargs)

        output = run_experiment_grid(configs, training_runner=runner)
        assert len(output["results"]) == 2
        assert len(output["failures"]) == 1
        assert output["failures"][0]["coin"] == "MISSING"
        assert "Training CSV not found" in output["failures"][0]["error"]


class TestRanking:
    def test_rank_experiments_prefers_edge_over_accuracy(self) -> None:
        results = [
            {
                "coin": "BTC",
                "interval": "1m",
                "horizon": 1,
                "threshold_bps": 3.0,
                "model_edge_vs_baseline": 0.05,
                "model_accuracy": 0.90,
                "test_rows": 100,
            },
            {
                "coin": "ETH",
                "interval": "1m",
                "horizon": 5,
                "threshold_bps": 5.0,
                "model_edge_vs_baseline": 0.10,
                "model_accuracy": 0.60,
                "test_rows": 100,
            },
        ]
        ranked = rank_experiments(results)
        assert ranked[0]["coin"] == "ETH"
        assert ranked[1]["coin"] == "BTC"


class TestReports:
    def test_write_grid_report_json(self, tmp_path: Path) -> None:
        configs = default_experiment_grid()[:1]
        results = [
            {
                "experiment_id": "abc123",
                "coin": "BTC",
                "interval": "15s",
                "horizon": 1,
                "threshold_bps": 1.0,
                "train_rows": 28,
                "test_rows": 12,
                "model_used": "majority_class",
                "model_accuracy": 0.5,
                "baseline_accuracy": 0.4,
                "model_edge_vs_baseline": 0.1,
                "model_artifact_path": None,
                "warnings": ["Research-only evaluation; does not prove profitability"],
            }
        ]
        report = build_grid_report(configs=configs, results=results, failures=[])
        output_dir = tmp_path / "grid_reports"
        path = write_grid_report_json(report, output_dir=output_dir)
        assert path.is_file()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["completed_count"] == 1
        assert loaded["paper_only"] is True
        assert loaded["execution_enabled"] is False
        assert loaded["best_result"]["coin"] == "BTC"

    def test_write_grid_report_markdown(self, tmp_path: Path) -> None:
        configs = default_experiment_grid()[:1]
        results = [
            {
                "experiment_id": "abc123",
                "coin": "BTC",
                "interval": "15s",
                "horizon": 1,
                "threshold_bps": 1.0,
                "train_rows": 28,
                "test_rows": 12,
                "model_used": "majority_class",
                "model_accuracy": 0.5,
                "baseline_accuracy": 0.4,
                "model_edge_vs_baseline": 0.1,
                "model_artifact_path": None,
                "warnings": ["Research-only evaluation; does not prove profitability"],
            }
        ]
        failures = [
            {
                "coin": "SOL",
                "interval": "1m",
                "horizon": 5,
                "threshold_bps": 5.0,
                "error": "Training CSV not found",
            }
        ]
        report = build_grid_report(configs=configs, results=results, failures=failures)
        output_dir = tmp_path / "grid_reports"
        path = write_grid_report_markdown(report, output_dir=output_dir)
        text = path.read_text(encoding="utf-8")
        assert path.is_file()
        assert "## Experiment table" in text
        assert "## Best result" in text
        assert "## Failed experiments" in text
        assert "## Warnings" in text
        assert "## Research interpretation" in text
        assert "## Paper-only disclaimer" in text
        assert "Paper-only. No trades placed." in text
        assert "Training CSV not found" in text


class TestCli:
    def test_main_default_smoke(
        self,
        data_root: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        for config in default_experiment_grid():
            _seed_coin_csv(data_root, config.coin, config.interval)

        exit_code = main(
            [
                "--default",
                "--data-root",
                str(data_root),
                "--reports-dir",
                str(tmp_path / "training_reports"),
                "--models-dir",
                str(tmp_path / "models"),
                "--output-dir",
                str(tmp_path / "grid_reports"),
            ]
        )
        captured = capsys.readouterr().out
        assert exit_code == 0
        assert "experiments=5" in captured
        assert "json_report_path=" in captured
        assert "markdown_report_path=" in captured
        assert "best_config=" in captured
        assert "Paper-only. No trades placed." in captured

    def test_main_single_experiment(
        self,
        data_root: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _seed_coin_csv(data_root, "BTC", "1m")
        exit_code = main(
            [
                "--coin",
                "BTC",
                "--interval",
                "1m",
                "--horizon",
                "5",
                "--threshold-bps",
                "5",
                "--data-root",
                str(data_root),
                "--reports-dir",
                str(tmp_path / "training_reports"),
                "--models-dir",
                str(tmp_path / "models"),
                "--output-dir",
                str(tmp_path / "grid_reports"),
            ]
        )
        captured = capsys.readouterr().out
        assert exit_code == 0
        assert "experiments=1" in captured
        assert "best_config=BTC 1m h=5 thr=5.0bps" in captured


class TestSafety:
    def test_no_wallet_signing_order_execution_code(self) -> None:
        source = GRID_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        lowered = source.lower()
        assert "requests" not in lowered
        assert "urllib" not in lowered
        assert "socket" not in lowered
        assert "httpx" not in lowered
        assert "subprocess" not in lowered
        for pattern in FORBIDDEN_SOURCE_PATTERNS:
            assert pattern not in lowered
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES

    def test_no_network_calls(self) -> None:
        source = GRID_FILE.read_text(encoding="utf-8").lower()
        forbidden = (
            "requests.get",
            "requests.post",
            "urllib.request",
            "http.client",
            "aiohttp",
            "websocket",
        )
        for pattern in forbidden:
            assert pattern not in source

    def test_no_subprocess_calls(self) -> None:
        source = GRID_FILE.read_text(encoding="utf-8").lower()
        assert "subprocess" not in source
        assert "os.system" not in source
        assert "popen" not in source

    def test_paper_only_constants(self) -> None:
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False

    def test_default_output_dir(self) -> None:
        assert DEFAULT_GRID_REPORTS_DIR == Path("reports/bid_ask_experiments")

    def test_grid_report_flags(self, tmp_path: Path) -> None:
        report = build_grid_report(
            configs=default_experiment_grid()[:1],
            results=[],
            failures=[],
        )
        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
