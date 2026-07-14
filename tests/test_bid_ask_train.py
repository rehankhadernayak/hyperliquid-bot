"""Focused offline tests for bid/ask movement training."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from bid_ask_resampler import RESAMPLED_COLUMNS, resampled_csv_path
from bid_ask_schema import FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS
from bid_ask_model_io import build_artifact_payload, load_model_artifact
from bid_ask_features import (
    CANDIDATE_C_CORE_COLUMNS,
    CANDIDATE_E_DEPTH_AWARE_COLUMNS,
    assert_no_future_feature_columns,
)
from bid_ask_train import (
    EXECUTION_ENABLED,
    FEATURE_COLUMNS,
    LABELS,
    PAPER_ONLY,
    MajorityClassBaseline,
    RuleBaseline,
    _build_arg_parser,
    build_training_report,
    chronological_train_test_split,
    create_features,
    create_labels,
    create_training_features,
    evaluate_predictions,
    get_training_feature_columns,
    load_training_data,
    main,
    run_training_pipeline,
    save_training_model_artifact,
    save_training_report,
    train_baseline_model,
    train_optional_sklearn_model,
)


class FakeModel:
    def predict(self, X):
        return ["UP"] * len(X)

    def predict_proba(self, X):
        return [[0.2, 0.3, 0.5]] * len(X)


class FakeScaler:
    def fit_transform(self, X):
        return X

    def transform(self, X):
        return X


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
    include_sizes: bool = False,
    bid_size: float = 2.0,
    ask_size: float = 1.0,
) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        start = start_ms + index * step_ms
        end = start + step_ms - 1
        mid = mid_start + index * mid_step
        row = {
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
        if include_sizes:
            row["bid_size"] = bid_size + index * 0.01
            row["ask_size"] = ask_size + index * 0.01
        rows.append(row)
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

    def test_load_historical_training_data(self, data_root: Path) -> None:
        path = resampled_csv_path("SOL", "1m", data_root, source="historical")
        _write_resampled_csv(path, _make_resampled_rows(coin="SOL", count=20, step_ms=60_000))
        df = load_training_data("SOL", "1m", data_root, source="historical")
        assert len(df) == 20
        assert path == data_root / "historical_1m" / "SOL_1m.csv"


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


class TestCandidateCCoreFeatures:
    def test_create_training_features_candidate_c_core_columns(self) -> None:
        rows = _make_resampled_rows(count=40)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        featured, meta = create_training_features(
            raw_df,
            feature_set="candidate_c_core",
            interval="15s",
        )

        assert meta["feature_set"] == "candidate_c_core"
        assert meta["feature_columns"] == list(CANDIDATE_C_CORE_COLUMNS)
        assert meta["feature_count"] == len(CANDIDATE_C_CORE_COLUMNS)
        for column in CANDIDATE_C_CORE_COLUMNS:
            assert column in featured.columns

    def test_no_future_return_columns_in_candidate_c_core_features(self) -> None:
        rows = _make_resampled_rows(count=30)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        _, meta = create_training_features(raw_df, feature_set="candidate_c_core", interval="15s")
        assert_no_future_feature_columns(meta["feature_columns"])
        assert "future_return_bps" not in meta["feature_columns"]

    def test_candidate_c_core_rolling_features_are_causal(self) -> None:
        rows = _make_resampled_rows(count=60, mid_start=100.0, mid_step=1.0)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        featured, _ = create_training_features(raw_df, feature_set="candidate_c_core", interval="15s")

        altered = raw_df.copy()
        altered.loc[altered.index[-1], "close_mid"] = (
            altered.loc[altered.index[-1], "close_mid"] * 2
        )
        altered_featured, _ = create_training_features(
            altered,
            feature_set="candidate_c_core",
            interval="15s",
        )

        for column in CANDIDATE_C_CORE_COLUMNS:
            pd.testing.assert_series_equal(
                featured[column].iloc[:-1],
                altered_featured[column].iloc[:-1],
                check_names=False,
            )

    def test_get_training_feature_columns(self) -> None:
        assert get_training_feature_columns("basic") == FEATURE_COLUMNS
        assert get_training_feature_columns("candidate_c_core") == CANDIDATE_C_CORE_COLUMNS

    def test_cli_accepts_candidate_c_core_feature_set(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(
            ["--coin", "BTC", "--interval", "15s", "--feature-set", "candidate_c_core"]
        )
        assert args.feature_set == "candidate_c_core"

    def test_training_report_includes_candidate_c_core_metadata(
        self, data_root: Path, tmp_path: Path
    ) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=40))
        result = run_training_pipeline(
            coin="BTC",
            interval="15s",
            horizon=1,
            data_root=data_root,
            reports_dir=tmp_path / "reports",
            models_dir=tmp_path / "models",
            feature_set="candidate_c_core",
        )
        report = result["report"]
        assert report["feature_set"] == "candidate_c_core"
        assert report["feature_columns"] == list(CANDIDATE_C_CORE_COLUMNS)
        assert report["feature_count"] == len(CANDIDATE_C_CORE_COLUMNS)
        assert result["evaluation"]["features_used"] == list(CANDIDATE_C_CORE_COLUMNS)


class TestCandidateEDepthAwareFeatures:
    def test_create_training_features_candidate_e_depth_aware_columns(self) -> None:
        rows = _make_resampled_rows(count=40, include_sizes=True)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        featured, meta = create_training_features(
            raw_df,
            feature_set="candidate_e_depth_aware",
            interval="15s",
        )

        assert meta["feature_set"] == "candidate_e_depth_aware"
        assert meta["feature_columns"] == list(CANDIDATE_E_DEPTH_AWARE_COLUMNS)
        assert meta["feature_count"] == len(CANDIDATE_E_DEPTH_AWARE_COLUMNS)
        for column in CANDIDATE_E_DEPTH_AWARE_COLUMNS:
            assert column in featured.columns

    def test_candidate_e_depth_aware_fails_without_size_columns(self) -> None:
        rows = _make_resampled_rows(count=30)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))

        with pytest.raises(ValueError, match="bid_size/ask_size"):
            create_training_features(
                raw_df,
                feature_set="candidate_e_depth_aware",
                interval="15s",
            )

    def test_no_future_return_columns_in_candidate_e_depth_aware_features(self) -> None:
        rows = _make_resampled_rows(count=30, include_sizes=True)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        _, meta = create_training_features(
            raw_df,
            feature_set="candidate_e_depth_aware",
            interval="15s",
        )
        assert_no_future_feature_columns(meta["feature_columns"])
        assert "future_return_bps" not in meta["feature_columns"]

    def test_candidate_e_depth_aware_rolling_features_are_causal(self) -> None:
        rows = _make_resampled_rows(count=60, mid_start=100.0, mid_step=1.0, include_sizes=True)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        featured, _ = create_training_features(
            raw_df,
            feature_set="candidate_e_depth_aware",
            interval="15s",
        )

        altered = raw_df.copy()
        altered.loc[altered.index[-1], "close_mid"] = (
            altered.loc[altered.index[-1], "close_mid"] * 2
        )
        altered_featured, _ = create_training_features(
            altered,
            feature_set="candidate_e_depth_aware",
            interval="15s",
        )

        for column in CANDIDATE_E_DEPTH_AWARE_COLUMNS:
            pd.testing.assert_series_equal(
                featured[column].iloc[:-1],
                altered_featured[column].iloc[:-1],
                check_names=False,
            )

    def test_get_training_feature_columns_candidate_e(self) -> None:
        assert get_training_feature_columns("candidate_e_depth_aware") == CANDIDATE_E_DEPTH_AWARE_COLUMNS

    def test_cli_accepts_candidate_e_depth_aware_feature_set(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--coin",
                "BTC",
                "--interval",
                "15s",
                "--feature-set",
                "candidate_e_depth_aware",
            ]
        )
        assert args.feature_set == "candidate_e_depth_aware"

    def test_training_report_includes_candidate_e_depth_aware_metadata(
        self, data_root: Path, tmp_path: Path
    ) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=40, include_sizes=True))
        result = run_training_pipeline(
            coin="BTC",
            interval="15s",
            horizon=1,
            data_root=data_root,
            reports_dir=tmp_path / "reports",
            models_dir=tmp_path / "models",
            feature_set="candidate_e_depth_aware",
        )
        report = result["report"]
        assert report["feature_set"] == "candidate_e_depth_aware"
        assert report["feature_columns"] == list(CANDIDATE_E_DEPTH_AWARE_COLUMNS)
        assert report["feature_count"] == len(CANDIDATE_E_DEPTH_AWARE_COLUMNS)
        assert result["evaluation"]["features_used"] == list(CANDIDATE_E_DEPTH_AWARE_COLUMNS)
        assert report["paper_only"] is True
        assert report["execution_enabled"] is False


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


class TestLabelDistributionReporting:
    @staticmethod
    def _labeled_split(
        data_root: Path,
        *,
        rows: list[dict],
        train_fraction: float = 0.70,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, rows)
        df = create_labels(
            create_features(load_training_data("BTC", "15s", data_root)),
            horizon=1,
            threshold_bps=1.0,
        ).dropna(subset=list(FEATURE_COLUMNS) + ["label"])
        return chronological_train_test_split(df, train_fraction=train_fraction)

    @staticmethod
    def _build_report_from_split(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict:
        baseline = train_baseline_model(train_df)
        predictions = baseline["majority_model"].predict(test_df)
        evaluation = evaluate_predictions(
            train_df=train_df,
            test_df=test_df,
            predictions=predictions,
            baseline_predictions=predictions,
        )
        return build_training_report(
            coin="BTC",
            interval="15s",
            horizon=1,
            threshold_bps=1.0,
            train_fraction=0.7,
            source_path="data/test.csv",
            evaluation=evaluation,
            baseline=baseline,
            sklearn_result={"available": False, "model": None},
            model_used="majority_class",
        )

    def test_label_counts_exists(self, data_root: Path) -> None:
        train_df, test_df = self._labeled_split(
            data_root, rows=_make_resampled_rows(count=40)
        )
        report = self._build_report_from_split(train_df, test_df)
        assert set(report["label_counts"]) == {"UP", "DOWN", "FLAT"}
        assert sum(report["label_counts"].values()) == len(train_df) + len(test_df)

    def test_label_percentages_exists(self, data_root: Path) -> None:
        train_df, test_df = self._labeled_split(
            data_root, rows=_make_resampled_rows(count=40)
        )
        report = self._build_report_from_split(train_df, test_df)
        assert set(report["label_percentages"]) == {"UP", "DOWN", "FLAT"}
        assert sum(report["label_percentages"].values()) == pytest.approx(1.0)

    def test_train_test_label_counts_exist(self, data_root: Path) -> None:
        train_df, test_df = self._labeled_split(
            data_root, rows=_make_resampled_rows(count=40)
        )
        report = self._build_report_from_split(train_df, test_df)
        assert set(report["train_label_counts"]) == {"UP", "DOWN", "FLAT"}
        assert set(report["test_label_counts"]) == {"UP", "DOWN", "FLAT"}
        assert sum(report["train_label_counts"].values()) == len(train_df)
        assert sum(report["test_label_counts"].values()) == len(test_df)

    def test_missing_class_appears_as_zero(self, data_root: Path) -> None:
        rows: list[dict] = []
        for index in range(30):
            mid = 100.0 if index < 27 else 100.0 + (index - 26) * 5.0
            rows.append(_make_resampled_rows(count=1, mid_start=mid, mid_step=0.0)[0])
            rows[-1]["start_timestamp_ms"] = BASE_MS + index * 15_000
            rows[-1]["end_timestamp_ms"] = rows[-1]["start_timestamp_ms"] + 14_999
        train_df, test_df = self._labeled_split(data_root, rows=rows)
        report = self._build_report_from_split(train_df, test_df)
        for field in (
            "label_counts",
            "label_percentages",
            "train_label_counts",
            "test_label_counts",
            "train_label_percentages",
            "test_label_percentages",
        ):
            assert set(report[field]) == {"UP", "DOWN", "FLAT"}
            assert all(isinstance(report[field][label], (int, float)) for label in LABELS)

    def test_under_10_percent_warning(self, data_root: Path) -> None:
        rows: list[dict] = []
        for index in range(30):
            mid = 100.0 if index < 27 else 100.0 + (index - 26) * 5.0
            rows.append(_make_resampled_rows(count=1, mid_start=mid, mid_step=0.0)[0])
            rows[-1]["start_timestamp_ms"] = BASE_MS + index * 15_000
            rows[-1]["end_timestamp_ms"] = rows[-1]["start_timestamp_ms"] + 14_999
        train_df, test_df = self._labeled_split(data_root, rows=rows)
        report = self._build_report_from_split(train_df, test_df)
        assert any(
            "Class imbalance in train" in warning and "under 10%" in warning
            for warning in report["warnings"]
        )

    def test_over_70_percent_warning(self, data_root: Path) -> None:
        rows = _make_resampled_rows(count=30, mid_start=100.0, mid_step=0.0)
        train_df, test_df = self._labeled_split(data_root, rows=rows)
        report = self._build_report_from_split(train_df, test_df)
        assert any(
            "Class dominance in train" in warning and "over 70%" in warning
            for warning in report["warnings"]
        )

    def test_existing_accuracy_and_artifact_fields_still_exist(
        self, data_root: Path, tmp_path: Path
    ) -> None:
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
        report = result["report"]
        assert "model_accuracy" in report
        assert "baseline_accuracy" in report
        assert "model_edge_vs_baseline" in report
        assert "model_artifact_path" in report
        assert "model_used" in report
        assert report["paper_only"] is True
        assert report["execution_enabled"] is False

    def test_no_order_execution_wallet_code(self) -> None:
        source = TRAIN_FILE.read_text(encoding="utf-8")
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

    def test_no_subprocess_calls(self) -> None:
        source = TRAIN_FILE.read_text(encoding="utf-8").lower()
        forbidden = ("subprocess", "os.system", "popen")
        for pattern in forbidden:
            assert pattern not in source


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

    def test_save_training_report_with_fake_model_artifact(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        reports_dir = tmp_path / "reports"
        sklearn_result = {
            "available": True,
            "model": FakeModel(),
            "scaler": FakeScaler(),
            "model_type": "FakeModel",
        }
        artifact_payload = build_artifact_payload(
            model=sklearn_result["model"],
            scaler=sklearn_result["scaler"],
            feature_names=FEATURE_COLUMNS,
            coin="BTC",
            interval="15s",
            horizon=1,
            threshold_bps=1.0,
            train_rows=28,
            test_rows=12,
            baseline_accuracy=0.4,
            model_accuracy=0.6,
            model_type="FakeModel",
        )
        report = {"coin": "BTC", "interval": "15s"}
        path = save_training_report(
            report,
            reports_dir=reports_dir,
            sklearn_result=sklearn_result,
            models_dir=models_dir,
            artifact_payload=artifact_payload,
            source="local",
        )
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert "model_artifact_path" in loaded
        assert Path(loaded["model_artifact_path"]).is_file()
        artifact = load_model_artifact(loaded["model_artifact_path"])
        assert artifact is not None
        assert artifact["paper_only"] is True
        assert artifact["execution_enabled"] is False

    def test_historical_artifact_name_does_not_overwrite_local(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        artifact_id = "testartifact123"
        payload = build_artifact_payload(
            model=FakeModel(),
            scaler=FakeScaler(),
            feature_names=FEATURE_COLUMNS,
            coin="SOL",
            interval="1m",
            horizon=5,
            threshold_bps=1.0,
            train_rows=10,
            test_rows=5,
            baseline_accuracy=0.4,
            model_accuracy=0.6,
            model_type="FakeModel",
        )
        payload["source"] = "historical"
        payload["data_dir"] = "data/bid_ask_csv/historical_1m"

        local_path = save_training_model_artifact(
            payload,
            models_dir=models_dir,
            artifact_id=artifact_id,
            source="local",
        )
        historical_path = save_training_model_artifact(
            payload,
            models_dir=models_dir,
            artifact_id=artifact_id,
            source="historical",
        )

        assert local_path.name == f"SOL_1m_{artifact_id}.joblib"
        assert historical_path.name == f"SOL_historical_1m_{artifact_id}.joblib"
        assert local_path != historical_path
        assert local_path.is_file()
        assert historical_path.is_file()

    def test_model_edge_vs_baseline_in_report(self) -> None:
        evaluation = {
            "accuracy": 0.6,
            "baseline_accuracy": 0.4,
            "warnings": ["Research-only evaluation; does not prove profitability"],
            "train_rows": 28,
            "test_rows": 12,
        }
        report = build_training_report(
            coin="BTC",
            interval="15s",
            horizon=1,
            threshold_bps=1.0,
            train_fraction=0.7,
            source_path="data/test.csv",
            evaluation=evaluation,
            baseline={"majority_class": "UP"},
            sklearn_result={"available": True, "model": FakeModel(), "model_type": "FakeModel"},
            model_used="FakeModel",
        )
        assert report["model_edge_vs_baseline"] == pytest.approx(0.2)
        assert report["model_accuracy"] == pytest.approx(0.6)
        assert report["baseline_accuracy"] == pytest.approx(0.4)
        assert report["model_artifact_path"] is None

    def test_model_not_beating_baseline_warning(self) -> None:
        evaluation = {
            "accuracy": 0.3,
            "baseline_accuracy": 0.5,
            "warnings": [],
            "train_rows": 28,
            "test_rows": 12,
        }
        report = build_training_report(
            coin="BTC",
            interval="15s",
            horizon=1,
            threshold_bps=1.0,
            train_fraction=0.7,
            source_path="data/test.csv",
            evaluation=evaluation,
            baseline={"majority_class": "UP"},
            sklearn_result={"available": True, "model": FakeModel(), "model_type": "FakeModel"},
            model_used="FakeModel",
        )
        assert any("does not exceed baseline" in warning for warning in report["warnings"])

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
        assert result["report"]["source"] == "local"
        assert result["report"]["data_dir"] == str(data_root / "15s")
        assert result["report"]["paper_only"] is True
        assert result["report"]["execution_enabled"] is False
        assert result["evaluation"]["test_rows"] > 0
        assert result["report_path"] is not None
        assert result["report_path"].is_file()
        assert "model_edge_vs_baseline" in result["report"]
        loaded = json.loads(result["report_path"].read_text(encoding="utf-8"))
        if loaded.get("model_artifact_path"):
            assert Path(loaded["model_artifact_path"]).is_file()

    def test_run_training_pipeline_historical_source(self, data_root: Path, tmp_path: Path) -> None:
        path = resampled_csv_path("SOL", "1m", data_root, source="historical")
        _write_resampled_csv(path, _make_resampled_rows(coin="SOL", count=40, step_ms=60_000))
        result = run_training_pipeline(
            coin="SOL",
            interval="1m",
            horizon=5,
            data_root=data_root,
            reports_dir=tmp_path / "reports",
            models_dir=tmp_path / "models",
            source="historical",
        )
        report = result["report"]
        assert report["source"] == "historical"
        assert report["data_dir"] == str(data_root / "historical_1m")
        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        if report.get("model_artifact_path"):
            assert "historical_1m" in Path(report["model_artifact_path"]).name


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
        assert args.source == "local"

    def test_argparse_historical_source(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(
            ["--coin", "SOL", "--interval", "1m", "--horizon", "5", "--source", "historical"]
        )
        assert args.source == "historical"

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
