"""Focused offline tests for bid/ask movement training."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from bid_ask_resampler import RESAMPLED_COLUMNS, resampled_csv_path
from bid_ask_schema import FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS
from bid_ask_train import (
    EXECUTION_ENABLED,
    FEATURE_COLUMNS,
    PAPER_ONLY,
    MajorityClassBaseline,
    RuleBaseline,
    _build_arg_parser,
    chronological_train_test_split,
    create_features,
    create_labels,
    evaluate_predictions,
    load_training_data,
    main,
    run_training_pipeline,
    save_training_report,
    train_baseline_model,
    train_optional_sklearn_model,
)
TRAIN_FILE = Path(__file__).resolve().parent.parent / "bid_ask_train.py"

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
                "avg_spread_bps": spread_bps + (index % 3) * 0.1,
                "avg_bbo_imbalance": imbalance + index * 0.01,
                "row_count": 15,
                "missing_seconds": 0,
            }
        )
    return rows


def _write_resampled_csv(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS)).to_csv(path, index=False)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "bid_ask_csv"


@pytest.fixture
def training_csv(data_root: Path) -> Path:
    rows = _make_resampled_rows(count=40)
    path = resampled_csv_path("BTC", "15s", data_root)
    _write_resampled_csv(path, rows)
    return path


class TestLoadTrainingData:
    def test_load_training_data(self, data_root: Path, training_csv: Path) -> None:
        df = load_training_data("BTC", "15s", data_root)
        assert len(df) == 40
        assert list(df.columns) == list(RESAMPLED_COLUMNS)
        assert df["start_timestamp_ms"].is_monotonic_increasing


class TestCreateFeatures:
    def test_create_features(self, data_root: Path, training_csv: Path) -> None:
        df = load_training_data("BTC", "15s", data_root)
        featured = create_features(df)

        for column in FEATURE_COLUMNS:
            assert column in featured.columns

        row = featured.iloc[10]
        expected_return = (
            (row["close_mid"] - featured.iloc[9]["close_mid"])
            / featured.iloc[9]["close_mid"]
            * 10_000.0
        )
        assert row["mid_return_1"] == pytest.approx(expected_return)
        assert row["spread_change_1"] == pytest.approx(
            row["avg_spread_bps"] - featured.iloc[9]["avg_spread_bps"]
        )
        assert row["bbo_imbalance_change_1"] == pytest.approx(
            row["avg_bbo_imbalance"] - featured.iloc[9]["avg_bbo_imbalance"]
        )


class TestCreateLabels:
    def test_create_labels(self, data_root: Path, training_csv: Path) -> None:
        df = create_features(load_training_data("BTC", "15s", data_root))
        labeled = create_labels(df, horizon=1, threshold_bps=1.0)

        assert "future_return_bps" in labeled.columns
        assert "label" in labeled.columns

        for index in range(len(labeled) - 1):
            current_mid = labeled.iloc[index]["close_mid"]
            future_mid = labeled.iloc[index + 1]["close_mid"]
            expected = (future_mid - current_mid) / current_mid * 10_000.0
            assert labeled.iloc[index]["future_return_bps"] == pytest.approx(expected)

            if expected > 1.0:
                assert labeled.iloc[index]["label"] == "UP"
            elif expected < -1.0:
                assert labeled.iloc[index]["label"] == "DOWN"
            else:
                assert labeled.iloc[index]["label"] == "FLAT"


class TestChronologicalSplit:
    def test_chronological_train_test_split(
        self, data_root: Path, training_csv: Path
    ) -> None:
        df = load_training_data("BTC", "15s", data_root)
        train_df, test_df = chronological_train_test_split(df, train_fraction=0.70)

        assert len(train_df) == 28
        assert len(test_df) == 12
        assert train_df["start_timestamp_ms"].max() < test_df["start_timestamp_ms"].min()


class TestNoFutureLeakage:
    def test_no_future_leakage_in_features(
        self, data_root: Path, training_csv: Path
    ) -> None:
        df = load_training_data("BTC", "15s", data_root)
        featured = create_features(df)
        labeled = create_labels(featured, horizon=1, threshold_bps=1.0)

        row_index = 10
        original = labeled.iloc[row_index].copy()
        mutated = labeled.copy()
        for future_index in range(row_index + 1, len(mutated)):
            mutated.loc[future_index, "close_mid"] = (
                mutated.loc[future_index, "close_mid"] * 10
            )

        recomputed = create_features(mutated.iloc[: row_index + 1].copy())
        for column in FEATURE_COLUMNS:
            assert recomputed.iloc[row_index][column] == pytest.approx(
                create_features(labeled.iloc[: row_index + 1].copy()).iloc[row_index][
                    column
                ]
            )

        assert original["label"] == labeled.iloc[row_index]["label"]


class TestBaselineTraining:
    def test_train_baseline_model(self, data_root: Path, training_csv: Path) -> None:
        df = create_labels(
            create_features(load_training_data("BTC", "15s", data_root)),
            horizon=1,
            threshold_bps=1.0,
        ).dropna(subset=list(FEATURE_COLUMNS) + ["label"])
        train_df, _ = chronological_train_test_split(df, train_fraction=0.70)

        baseline = train_baseline_model(train_df)
        majority: MajorityClassBaseline = baseline["majority_model"]
        rule: RuleBaseline = baseline["rule_model"]

        assert majority.majority_class in {"UP", "DOWN", "FLAT"}
        assert len(majority.predict(train_df)) == len(train_df)
        assert len(rule.predict(train_df)) == len(train_df)


class TestEvaluationReport:
    def test_evaluate_predictions(self, data_root: Path, training_csv: Path) -> None:
        df = create_labels(
            create_features(load_training_data("BTC", "15s", data_root)),
            horizon=1,
            threshold_bps=1.0,
        ).dropna(subset=list(FEATURE_COLUMNS) + ["label"])
        train_df, test_df = chronological_train_test_split(df, train_fraction=0.70)
        baseline = train_baseline_model(train_df)
        predictions = baseline["majority_model"].predict(test_df)
        evaluation = evaluate_predictions(
            train_df=train_df,
            test_df=test_df,
            predictions=predictions,
            baseline_predictions=predictions,
        )

        assert "accuracy" in evaluation
        assert "baseline_accuracy" in evaluation
        assert "confusion_matrix" in evaluation
        assert set(evaluation["label_counts"]["test"]) == {"UP", "DOWN", "FLAT"}
        assert evaluation["train_rows"] == len(train_df)
        assert evaluation["test_rows"] == len(test_df)
        assert evaluation["features_used"] == list(FEATURE_COLUMNS)
        assert any(
            "does not prove profitability" in warning
            for warning in evaluation["warnings"]
        )


class TestWarnings:
    def test_small_dataset_warning(self, data_root: Path) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=10))
        df = create_labels(
            create_features(load_training_data("BTC", "15s", data_root)),
            horizon=1,
            threshold_bps=1.0,
        ).dropna(subset=list(FEATURE_COLUMNS) + ["label"])
        train_df, test_df = chronological_train_test_split(df, train_fraction=0.70)
        baseline = train_baseline_model(train_df)
        predictions = baseline["majority_model"].predict(test_df)
        evaluation = evaluate_predictions(
            train_df=train_df,
            test_df=test_df,
            predictions=predictions,
            baseline_predictions=predictions,
        )
        assert any("Low row count" in warning for warning in evaluation["warnings"])

    def test_class_imbalance_warning(self, data_root: Path) -> None:
        rows: list[dict] = []
        for index in range(30):
            mid = 100.0 if index < 27 else 100.0 + (index - 26) * 5.0
            rows.append(_make_resampled_rows(count=1, mid_start=mid, mid_step=0.0)[0])
            rows[-1]["start_timestamp_ms"] = BASE_MS + index * 15_000
            rows[-1]["end_timestamp_ms"] = rows[-1]["start_timestamp_ms"] + 14_999
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, rows)

        df = create_labels(
            create_features(load_training_data("BTC", "15s", data_root)),
            horizon=1,
            threshold_bps=1.0,
        ).dropna(subset=list(FEATURE_COLUMNS) + ["label"])
        train_df, test_df = chronological_train_test_split(df, train_fraction=0.70)
        baseline = train_baseline_model(train_df)
        predictions = baseline["majority_model"].predict(test_df)
        evaluation = evaluate_predictions(
            train_df=train_df,
            test_df=test_df,
            predictions=predictions,
            baseline_predictions=predictions,
        )
        assert any("Class imbalance" in warning for warning in evaluation["warnings"])


class TestReportCreation:
    def test_save_training_report(self, tmp_path: Path) -> None:
        report = {
            "coin": "BTC",
            "interval": "15s",
            "evaluation": {"accuracy": 0.5},
        }
        reports_dir = tmp_path / "reports"
        path = save_training_report(report, reports_dir=reports_dir)
        assert path.is_file()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["coin"] == "BTC"
        assert "report_id" in loaded

    def test_run_training_pipeline(self, data_root: Path, tmp_path: Path) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=40))
        result = run_training_pipeline(
            coin="BTC",
            interval="15s",
            horizon=1,
            data_root=data_root,
            reports_dir=tmp_path / "reports",
            models_dir=tmp_path / "models",
        )
        assert result["report"]["coin"] == "BTC"
        assert result["evaluation"]["test_rows"] > 0
        assert result["report_path"] is not None
        assert result["report_path"].is_file()


class TestSklearnOptional:
    def test_train_optional_sklearn_model(self, data_root: Path, training_csv: Path) -> None:
        df = create_labels(
            create_features(load_training_data("BTC", "15s", data_root)),
            horizon=1,
            threshold_bps=1.0,
        ).dropna(subset=list(FEATURE_COLUMNS) + ["label"])
        train_df, _ = chronological_train_test_split(df, train_fraction=0.70)
        result = train_optional_sklearn_model(train_df)
        assert "available" in result
        if not result["available"]:
            assert "sklearn not installed" in (result.get("message") or "")


class TestCommandParser:
    def test_argparse_smoke(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--coin", "BTC", "--interval", "15s", "--horizon", "1"])
        assert args.coin == "BTC"
        assert args.interval == "15s"
        assert args.horizon == 1

    def test_main_with_temp_data(self, data_root: Path, tmp_path: Path, capsys) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=40))
        exit_code = main(
            [
                "--coin",
                "BTC",
                "--interval",
                "15s",
                "--horizon",
                "1",
                "--data-root",
                str(data_root),
                "--reports-dir",
                str(tmp_path / "reports"),
            ]
        )
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "train_rows=" in captured.out
        assert "accuracy=" in captured.out


class TestSafety:
    def test_no_wallet_signing_order_execution_code(self) -> None:
        source = TRAIN_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        lowered = source.lower()
        assert "requests" not in lowered
        assert "urllib" not in lowered
        assert "socket" not in lowered
        assert "httpx" not in lowered
        for pattern in FORBIDDEN_SOURCE_PATTERNS:
            assert pattern not in lowered
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES

    def test_paper_only_constants(self) -> None:
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False

    def test_no_network_calls(self) -> None:
        source = TRAIN_FILE.read_text(encoding="utf-8").lower()
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
