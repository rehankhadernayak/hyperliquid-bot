"""Focused offline tests for walk-forward paper evaluation."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from bid_ask_resampler import RESAMPLED_COLUMNS, resampled_csv_path
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY
from bid_ask_walk_forward_evaluator import (
    DEFAULT_REPORTS_DIR,
    SAFETY_NOTE,
    _build_arg_parser,
    build_walk_forward_report,
    evaluate_coin_walk_forward,
    generate_walk_forward_windows,
    longest_losing_streak,
    main,
    max_drawdown_bps,
    resolve_coins,
    run_walk_forward_evaluation,
    simulate_paper_signal_returns,
    write_walk_forward_json,
    write_walk_forward_markdown,
)

EVALUATOR_FILE = Path(__file__).resolve().parent.parent / "bid_ask_walk_forward_evaluator.py"

BASE_MS = 1_718_000_040_000


def _make_resampled_rows(
    *,
    coin: str = "BTC",
    count: int,
    start_ms: int = BASE_MS,
    step_ms: int = 15_000,
    mid_start: float = 100.0,
    mid_step: float = 0.5,
    spread_bps: float = 2.0,
    imbalance: float = 0.2,
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
                "timestamp_iso": f"2024-06-10T00:00:{index % 60:02d}Z",
                "coin": coin,
                "open_mid": mid,
                "high_mid": mid + 0.1,
                "low_mid": mid - 0.1,
                "close_mid": mid,
                "avg_bid": mid - 0.5,
                "avg_ask": mid + 0.5,
                "avg_mid": mid,
                "avg_spread": 1.0,
                "avg_spread_bps": spread_bps + (index % 3) * 0.1,
                "avg_bbo_imbalance": imbalance + index * 0.01,
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


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    return tmp_path / "walk_forward_reports"


class TestGenerateWalkForwardWindows:
    def test_chronological_windows_no_shuffle(self) -> None:
        windows = generate_walk_forward_windows(
            100,
            train_bars=30,
            test_bars=10,
            step_bars=10,
        )
        assert len(windows) == 7
        assert windows[0] == (0, 30, 30, 40)
        assert windows[1] == (10, 40, 40, 50)
        for train_start, train_end, test_start, test_end in windows:
            assert train_end == test_start
            assert train_end - train_start == 30
            assert test_end - test_start == 10
            assert train_start < test_start

    def test_too_few_rows_produces_no_windows(self) -> None:
        windows = generate_walk_forward_windows(
            20,
            train_bars=15,
            test_bars=10,
            step_bars=5,
        )
        assert windows == []


class TestCostSimulation:
    def test_fee_slippage_spread_reduce_net_returns(self) -> None:
        row_no_spread = pd.Series({"future_return_bps": 10.0})
        row_with_spread = pd.Series({"future_return_bps": 10.0, "avg_spread_bps": 2.0})
        predictions = ["UP"]
        confidences = [1.0]

        no_cost = simulate_paper_signal_returns(
            pd.DataFrame([row_no_spread]),
            predictions,
            confidences,
            fee_bps=0.0,
            slippage_bps=0.0,
            default_spread_bps=0.0,
            min_confidence=0.0,
        )
        with_cost = simulate_paper_signal_returns(
            pd.DataFrame([row_with_spread]),
            predictions,
            confidences,
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            min_confidence=0.0,
        )

        assert no_cost["net_returns_bps"][0] == pytest.approx(10.0)
        assert with_cost["net_returns_bps"][0] == pytest.approx(10.0 - 5.0)

    def test_flat_signal_zero_return(self) -> None:
        row = pd.Series({"future_return_bps": 10.0, "avg_spread_bps": 2.0})
        test_df = pd.DataFrame([row])
        result = simulate_paper_signal_returns(
            test_df,
            ["FLAT"],
            [1.0],
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            min_confidence=0.0,
        )
        assert result["net_returns_bps"][0] == 0.0
        assert result["total_paper_trades"] == 0

    def test_down_signal_inverse_return(self) -> None:
        row = pd.Series({"future_return_bps": 8.0, "avg_spread_bps": 1.0})
        test_df = pd.DataFrame([row])
        result = simulate_paper_signal_returns(
            test_df,
            ["DOWN"],
            [1.0],
            fee_bps=1.0,
            slippage_bps=1.0,
            default_spread_bps=1.0,
            min_confidence=0.0,
        )
        assert result["gross_returns_bps"][0] == pytest.approx(-8.0)
        assert result["net_returns_bps"][0] == pytest.approx(-8.0 - 3.0)


class TestDrawdownAndStreak:
    def test_max_drawdown(self) -> None:
        returns = [5.0, -10.0, 3.0, -2.0]
        assert max_drawdown_bps(returns) == pytest.approx(10.0)

    def test_longest_losing_streak(self) -> None:
        returns = [1.0, -1.0, -2.0, 3.0, -1.0]
        assert longest_losing_streak(returns) == 2


class TestEvaluateCoinWalkForward:
    def test_model_vs_baseline_metrics_exist(self, data_root: Path, reports_dir: Path) -> None:
        path = resampled_csv_path("BTC", "15s", data_root, source="historical")
        _write_resampled_csv(path, _make_resampled_rows(count=80, mid_step=1.0))

        result = evaluate_coin_walk_forward(
            coin="BTC",
            interval="15s",
            source="historical",
            horizon=3,
            threshold_bps=0.5,
            train_bars=30,
            test_bars=10,
            step_bars=10,
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            min_confidence=0.0,
            data_root=data_root,
            save_window_models=False,
            models_dir=reports_dir / "models",
        )

        assert result["completed_windows"] >= 1
        assert "model_accuracy_mean" in result
        assert "baseline_accuracy_mean" in result
        assert "accuracy_edge_vs_baseline" in result
        assert "total_paper_signals" in result
        assert "cumulative_net_return_bps" in result
        window = result["windows"][0]
        assert window["status"] == "completed"
        assert "model_accuracy" in window
        assert "baseline_accuracy" in window

    def test_too_few_rows_skips_safely(self, data_root: Path, reports_dir: Path) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=10))

        result = evaluate_coin_walk_forward(
            coin="BTC",
            interval="15s",
            source="local",
            horizon=1,
            threshold_bps=1.0,
            train_bars=50,
            test_bars=20,
            step_bars=10,
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            min_confidence=0.0,
            data_root=data_root,
            save_window_models=False,
            models_dir=reports_dir / "models",
        )

        assert result["total_windows"] == 0
        assert result["completed_windows"] == 0
        assert any("too_few" in warning for warning in result["warnings"])

    def test_data_loaded_in_chronological_order(self, data_root: Path, reports_dir: Path) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        rows = _make_resampled_rows(count=50)
        rows_reversed = list(reversed(rows))
        _write_resampled_csv(path, rows_reversed)

        result = evaluate_coin_walk_forward(
            coin="BTC",
            interval="15s",
            source="local",
            horizon=1,
            threshold_bps=1.0,
            train_bars=20,
            test_bars=10,
            step_bars=10,
            fee_bps=0.0,
            slippage_bps=0.0,
            default_spread_bps=0.0,
            min_confidence=0.0,
            data_root=data_root,
            save_window_models=False,
            models_dir=reports_dir / "models",
        )

        assert result["completed_windows"] >= 1
        first_window = next(item for item in result["windows"] if item["status"] == "completed")
        assert first_window["train_start"] == 0


class TestRunWalkForwardEvaluation:
    def test_multiple_coins_combined_report(self, data_root: Path, reports_dir: Path) -> None:
        for coin in ("BTC", "ETH"):
            path = resampled_csv_path(coin, "15s", data_root, source="historical")
            _write_resampled_csv(path, _make_resampled_rows(coin=coin, count=70, mid_step=0.8))

        report = run_walk_forward_evaluation(
            coins=["BTC", "ETH"],
            interval="15s",
            source="historical",
            horizon=2,
            threshold_bps=0.5,
            train_bars=25,
            test_bars=10,
            step_bars=10,
            fee_bps=1.0,
            slippage_bps=1.0,
            default_spread_bps=1.0,
            min_confidence=0.0,
            data_root=data_root,
            reports_dir=reports_dir,
            save_reports=True,
        )

        assert report["coins"] == ["BTC", "ETH"]
        assert "BTC" in report["per_coin"]
        assert "ETH" in report["per_coin"]
        assert len(report["per_coin_details"]) == 2
        assert Path(report["json_report_path"]).is_file()
        assert Path(report["markdown_report_path"]).is_file()

    def test_readiness_flags(self, data_root: Path, reports_dir: Path) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=80, mid_step=2.0))

        report = run_walk_forward_evaluation(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=2,
            threshold_bps=0.1,
            train_bars=25,
            test_bars=10,
            step_bars=10,
            fee_bps=0.0,
            slippage_bps=0.0,
            default_spread_bps=0.0,
            min_confidence=0.0,
            data_root=data_root,
            reports_dir=reports_dir,
        )

        readiness = report["readiness"]
        assert "beats_baseline_accuracy" in readiness
        assert "positive_after_costs" in readiness
        assert "enough_windows" in readiness
        assert "enough_trades" in readiness
        assert "ready_for_more_research" in readiness
        assert readiness["real_execution_ready"] is False

    def test_paper_only_and_execution_disabled(self, data_root: Path, reports_dir: Path) -> None:
        path = resampled_csv_path("SOL", "15s", data_root, source="combined")
        _write_resampled_csv(path, _make_resampled_rows(coin="SOL", count=60))

        report = run_walk_forward_evaluation(
            coins=["SOL"],
            interval="15s",
            source="combined",
            horizon=1,
            train_bars=20,
            test_bars=10,
            step_bars=10,
            data_root=data_root,
            reports_dir=reports_dir,
        )

        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False

    def test_reports_written_to_temp_dir(self, data_root: Path, reports_dir: Path) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=55))

        report = {
            "generated_at": "2024-06-01T00:00:00+00:00",
            "coins": ["BTC"],
            "interval": "15s",
            "source": "local",
            "readiness": {"real_execution_ready": False},
            "warnings": [SAFETY_NOTE],
            "paper_only": True,
            "execution_enabled": False,
            "real_execution_ready": False,
        }
        report.update(
            build_walk_forward_report(
                coins=["BTC"],
                interval="15s",
                source="local",
                horizon=1,
                threshold_bps=1.0,
                train_bars=20,
                test_bars=10,
                step_bars=10,
                fee_bps=1.0,
                slippage_bps=1.0,
                default_spread_bps=1.0,
                min_confidence=0.0,
                per_coin_results=[],
                data_root=data_root,
            )
        )

        json_path = write_walk_forward_json(report, reports_dir=reports_dir)
        md_path = write_walk_forward_markdown(report, reports_dir=reports_dir)
        assert json_path.parent == reports_dir
        assert md_path.parent == reports_dir
        assert json_path.is_file()
        assert md_path.is_file()
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["paper_only"] is True
        assert loaded["execution_enabled"] is False


class TestCLI:
    def test_cli_parses_source_coin_coins_interval_horizon(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--source",
                "historical",
                "--coin",
                "BTC",
                "--coins",
                "ETH",
                "SOL",
                "--interval",
                "15s",
                "--horizon",
                "5",
            ]
        )
        assert args.source == "historical"
        assert args.coin == "BTC"
        assert args.coins == ["ETH", "SOL"]
        assert args.interval == "15s"
        assert args.horizon == 5

    def test_resolve_coins_deduplicates(self) -> None:
        assert resolve_coins(coin="BTC", coins=["ETH", "BTC"]) == ["ETH", "BTC"]

    def test_main_writes_reports(self, data_root: Path, reports_dir: Path, tmp_path: Path) -> None:
        path = resampled_csv_path("BTC", "15s", data_root, source="historical")
        _write_resampled_csv(path, _make_resampled_rows(count=65))

        exit_code = main(
            [
                "--source",
                "historical",
                "--coin",
                "BTC",
                "--interval",
                "15s",
                "--horizon",
                "3",
                "--train-bars",
                "25",
                "--test-bars",
                "10",
                "--step-bars",
                "10",
                "--fee-bps",
                "2",
                "--slippage-bps",
                "1",
                "--min-confidence",
                "0",
                "--data-root",
                str(data_root),
                "--reports-dir",
                str(reports_dir),
            ]
        )
        assert exit_code == 0
        assert any(reports_dir.glob("walk_forward_*.json"))
        assert any(reports_dir.glob("walk_forward_*.md"))


class TestSafety:
    def test_no_forbidden_imports(self) -> None:
        source = EVALUATOR_FILE.read_text(encoding="utf-8")
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
        source = EVALUATOR_FILE.read_text(encoding="utf-8").lower()
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
        source = EVALUATOR_FILE.read_text(encoding="utf-8").lower()
        forbidden = ("subprocess", "os.system", "popen")
        for pattern in forbidden:
            assert pattern not in source

    def test_paper_only_constants(self) -> None:
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False
        assert DEFAULT_REPORTS_DIR == Path("reports/bid_ask_walk_forward")
