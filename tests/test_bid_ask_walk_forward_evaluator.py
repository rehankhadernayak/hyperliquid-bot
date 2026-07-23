"""Focused offline tests for walk-forward paper evaluation."""

from __future__ import annotations

import ast
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from bid_ask_resampler import RESAMPLED_COLUMNS, resampled_csv_path
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY
from bid_ask_features import (
    ACCEPTED_TRADES_CANDIDATE_F_DIAGNOSTIC_COLUMNS,
    CANDIDATE_C_CORE_COLUMNS,
    CANDIDATE_E_DEPTH_AWARE_COLUMNS,
    CANDIDATE_E_DEPTH_AWARE_EXTRA_COLUMNS,
    CANDIDATE_F_REGIME_DEPTH_COLUMNS,
    CANDIDATE_F_REGIME_DEPTH_EXTRA_COLUMNS,
    REGIME_BREAKDOWN_COLUMNS,
    assert_no_future_feature_columns,
)
from bid_ask_train import FEATURE_COLUMNS, create_features, create_labels
from bid_ask_walk_forward_evaluator import (
    ACCEPTED_TRADES_CSV_COLUMNS,
    ALL_WINDOWS_SKIPPED_ONE_CLASS_REJECTION,
    DEFAULT_ACCEPTED_TRADES_LIMIT,
    DEFAULT_DIAGNOSTICS_DETAIL,
    DEFAULT_FEATURE_SET,
    DEFAULT_LABEL_COST_MULTIPLIER,
    DEFAULT_LABEL_MIN_EDGE_BPS,
    DEFAULT_LABEL_MODE,
    DEFAULT_MIN_TRADES,
    DEFAULT_MIN_WINDOWS,
    DEFAULT_REPORTS_DIR,
    DEFAULT_STABILITY_MIN_TRADES_PER_BUCKET,
    MICROSTRUCTURE_EXTRA_COLUMNS,
    ONE_CLASS_TRAINING_WINDOW_REASON,
    REGIME_OVERFIT_WARNING,
    SAFETY_NOTE,
    TradeFilterConfig,
    _build_arg_parser,
    _training_label_class_counts,
    aggregate_accepted_trade_metrics,
    aggregate_class_trade_diagnostics,
    aggregate_coin_class_trade_diagnostics,
    aggregate_stability_bucket_metrics,
    apply_diagnostics_detail,
    apply_regime_filter_metrics,
    apply_stability_readiness,
    build_feature_metadata,
    build_stability_warnings,
    build_trade_filter,
    build_walk_forward_report,
    compute_label_distribution,
    compute_readiness_flags,
    compute_regime_breakdown_diagnostics,
    compute_stability_diagnostics,
    compute_stability_summary,
    cost_aware_label_warnings,
    create_candidate_d_labels,
    create_cost_aware_labels,
    create_walk_forward_features,
    create_walk_forward_labels,
    evaluate_coin_walk_forward,
    evaluate_trade_filter,
    evaluate_walk_forward_window,
    filter_accepted_trades_by_regime,
    finalize_accepted_trades_csv,
    generate_walk_forward_windows,
    get_walk_forward_feature_columns,
    is_candidate_d_label_mode,
    longest_losing_streak,
    main,
    max_drawdown_bps,
    parse_feature_set,
    parse_hours_list,
    parse_label_mode,
    parse_trade_classes,
    build_candidate_d_label_diagnostics,
    candidate_d_supports_walk_forward_training,
    compute_candidate_d_class_distribution,
    compute_candidate_d_estimated_cost_bps,
    binary_tradeable_direction,
    binary_tradeable_label_warnings,
    is_binary_tradeable_label_mode,
    is_candidate_d_diagnostics_only,
    regime_filters_enabled,
    resolve_accepted_trades_csv_columns,
    resolve_coins,
    run_walk_forward_evaluation,
    simulate_binary_tradeable_paper_returns,
    simulate_paper_signal_returns,
    summarize_label_required_edge_bps,
    train_walk_forward_sklearn_model,
    walk_forward_feature_matrix,
    walk_forward_rows_ready_for_model,
    write_walk_forward_json,
    write_walk_forward_markdown,
    write_accepted_trades_csv,
    _trade_record_utc_hour,
)

EVALUATOR_FILE = Path(__file__).resolve().parent.parent / "bid_ask_walk_forward_evaluator.py"

FORBIDDEN_ACCEPTED_TRADES_COLUMNS = frozenset(
    {
        "order_id",
        "wallet",
        "private_key",
        "signature",
        "execution",
        "leverage",
        "margin",
        "deposit",
        "withdrawal",
    }
)

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
        if include_sizes:
            row["bid_size"] = bid_size + index * 0.01
            row["ask_size"] = ask_size + index * 0.01
        rows.append(row)
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


READINESS_FIELDS = (
    "beats_baseline_accuracy",
    "positive_after_costs",
    "enough_windows",
    "enough_trades",
    "ready_for_more_research",
    "real_execution_ready",
)

FILTER_METRIC_FIELDS = (
    "raw_predictions",
    "filtered_paper_trades",
    "blocked_by_confidence",
    "blocked_by_cost",
    "blocked_by_spread",
    "blocked_by_volatility",
    "blocked_by_trade_class",
    "trade_filter_summary",
    "average_cost_bps",
    "average_required_edge_bps",
)


def _permissive_filter(**overrides: object) -> TradeFilterConfig:
    defaults: dict[str, object] = {
        "min_confidence": 0.0,
        "threshold_bps": 0.1,
        "min_edge_bps": 0.0,
        "horizon_return_scale_bps": 1000.0,
    }
    defaults.update(overrides)
    return build_trade_filter(**defaults)  # type: ignore[arg-type]


def _assert_readiness_booleans(report: dict, *, prefix: str = "") -> None:
    for field in READINESS_FIELDS:
        key = f"{prefix}{field}" if prefix else field
        value = report[field]
        assert value is not None, field
        assert isinstance(value, bool), field
    nested = report["readiness"]
    for field in READINESS_FIELDS:
        assert nested[field] is not None, field
        assert isinstance(nested[field], bool), field
        assert nested[field] == report[field], field


def _make_labeled_feature_rows(
    *,
    count: int,
    future_returns_bps: list[float] | None = None,
    spread_bps: float = 2.0,
    mid_start: float = 100.0,
) -> pd.DataFrame:
    rows = _make_resampled_rows(count=count, mid_start=mid_start, spread_bps=spread_bps)
    featured = create_features(pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS)))
    if future_returns_bps is not None:
        featured["future_return_bps"] = future_returns_bps + [float("nan")] * (
            len(featured) - len(future_returns_bps)
        )
    return featured


class TestFeatureSets:
    def test_basic_feature_set_matches_existing_columns(self) -> None:
        rows = _make_resampled_rows(count=40)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        featured, meta = create_walk_forward_features(raw_df, feature_set="basic")
        expected = create_features(raw_df)

        assert meta["feature_set"] == "basic"
        assert meta["feature_columns"] == list(FEATURE_COLUMNS)
        assert meta["feature_count"] == len(FEATURE_COLUMNS)
        assert meta["skipped_microstructure_features"] == []
        assert meta["feature_warning"] is None
        for column in FEATURE_COLUMNS:
            assert featured[column].equals(expected[column])

    def test_microstructure_adds_more_features_than_basic(self) -> None:
        rows = _make_resampled_rows(count=80)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        basic_meta = create_walk_forward_features(raw_df, feature_set="basic")[1]
        micro_df, micro_meta = create_walk_forward_features(raw_df, feature_set="microstructure")

        assert micro_meta["feature_set"] == "microstructure"
        assert micro_meta["feature_count"] > basic_meta["feature_count"]
        assert set(FEATURE_COLUMNS).issubset(micro_meta["feature_columns"])
        assert any(
            column in micro_meta["feature_columns"]
            for column in MICROSTRUCTURE_EXTRA_COLUMNS
        )
        for column in micro_meta["feature_columns"]:
            assert column in micro_df.columns

    def test_no_future_leakage_from_negative_shifts(self) -> None:
        rows = _make_resampled_rows(count=60, mid_start=100.0, mid_step=1.0)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        featured, _ = create_walk_forward_features(raw_df, feature_set="microstructure")

        # Perturbing only the final row should not change earlier feature values.
        altered = raw_df.copy()
        altered.loc[altered.index[-1], "close_mid"] = altered.loc[altered.index[-1], "close_mid"] * 2
        altered_featured, _ = create_walk_forward_features(altered, feature_set="microstructure")

        compare_columns = [
            column
            for column in featured.columns
            if column in MICROSTRUCTURE_EXTRA_COLUMNS or column in FEATURE_COLUMNS
        ]
        for column in compare_columns:
            pd.testing.assert_series_equal(
                featured[column].iloc[:-1],
                altered_featured[column].iloc[:-1],
                check_names=False,
            )

    def test_missing_spread_and_imbalance_columns_are_skipped_safely(self) -> None:
        rows = _make_resampled_rows(count=30)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS)).drop(
            columns=["avg_spread_bps", "avg_bbo_imbalance"]
        )
        featured, meta = create_walk_forward_features(raw_df, feature_set="microstructure")

        assert meta["feature_set"] == "microstructure"
        assert meta["skipped_microstructure_features"]
        assert "spread_bps" in meta["skipped_microstructure_features"]
        assert "bbo_imbalance" in meta["skipped_microstructure_features"]
        assert "return_1" in meta["feature_columns"]
        assert "return_5" in featured.columns

    def test_report_includes_feature_metadata(self, data_root: Path) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=80, mid_step=1.5))

        report = run_walk_forward_evaluation(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=2,
            threshold_bps=0.1,
            train_bars=25,
            test_bars=10,
            step_bars=10,
            feature_set="microstructure",
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            save_reports=False,
            min_windows=1,
            min_trades=1,
        )

        assert report["feature_set"] == "microstructure"
        assert report["feature_columns"]
        assert report["feature_count"] == len(report["feature_columns"])
        assert isinstance(report["skipped_microstructure_features"], list)

    def test_skipped_feature_list_populated_when_columns_missing(self) -> None:
        rows = _make_resampled_rows(count=20)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS)).drop(columns=["avg_spread_bps"])
        _, meta = create_walk_forward_features(raw_df, feature_set="microstructure")

        assert "spread_bps" in meta["skipped_microstructure_features"]
        assert "rolling_spread_mean_5" in meta["skipped_microstructure_features"]

    def test_feature_warning_when_too_few_features(self) -> None:
        metadata = build_feature_metadata(
            feature_set="microstructure",
            feature_columns=FEATURE_COLUMNS,
            skipped_microstructure_features=list(MICROSTRUCTURE_EXTRA_COLUMNS),
        )
        assert metadata["feature_warning"] is not None
        assert "too_few_features" in metadata["feature_warning"]

    def test_microstructure_walk_forward_safety_flags(self, data_root: Path) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=70, mid_step=1.0))

        report = run_walk_forward_evaluation(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=2,
            train_bars=25,
            test_bars=10,
            step_bars=10,
            feature_set="microstructure",
            data_root=data_root,
            save_reports=False,
            min_windows=1,
            min_trades=1,
        )

        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False

    def test_cli_parses_feature_set(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--feature-set", "microstructure"])
        assert args.feature_set == "microstructure"

    def test_cli_feature_set_default(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.feature_set == DEFAULT_FEATURE_SET

    def test_parse_feature_set(self) -> None:
        assert parse_feature_set("basic") == "basic"
        assert parse_feature_set("microstructure") == "microstructure"
        assert parse_feature_set("candidate_c_core") == "candidate_c_core"
        assert parse_feature_set("candidate_e_depth_aware") == "candidate_e_depth_aware"
        assert parse_feature_set("candidate_f_regime_depth") == "candidate_f_regime_depth"
        with pytest.raises(ValueError):
            parse_feature_set("invalid")

    def test_get_walk_forward_feature_columns(self) -> None:
        assert get_walk_forward_feature_columns("basic") == FEATURE_COLUMNS
        assert get_walk_forward_feature_columns("candidate_c_core") == CANDIDATE_C_CORE_COLUMNS
        micro_columns = get_walk_forward_feature_columns("microstructure")
        assert micro_columns[: len(FEATURE_COLUMNS)] == FEATURE_COLUMNS
        assert len(micro_columns) == len(FEATURE_COLUMNS) + len(MICROSTRUCTURE_EXTRA_COLUMNS)

    def test_walk_forward_feature_matrix_uses_selected_columns(self) -> None:
        rows = _make_resampled_rows(count=10)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        featured, meta = create_walk_forward_features(raw_df, feature_set="microstructure")
        matrix = walk_forward_feature_matrix(featured, tuple(meta["feature_columns"]))
        assert list(matrix.columns) == meta["feature_columns"]

    def test_candidate_c_core_feature_set_creates_expected_columns(self) -> None:
        rows = _make_resampled_rows(count=80)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        featured, meta = create_walk_forward_features(
            raw_df,
            feature_set="candidate_c_core",
            interval="15s",
        )

        assert meta["feature_set"] == "candidate_c_core"
        assert meta["feature_columns"] == list(CANDIDATE_C_CORE_COLUMNS)
        assert meta["feature_count"] == len(CANDIDATE_C_CORE_COLUMNS)
        for column in CANDIDATE_C_CORE_COLUMNS:
            assert column in featured.columns

    def test_candidate_c_core_excludes_future_return_columns(self) -> None:
        rows = _make_resampled_rows(count=40)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        _, meta = create_walk_forward_features(raw_df, feature_set="candidate_c_core", interval="15s")
        assert_no_future_feature_columns(meta["feature_columns"])

    def test_candidate_c_core_rolling_features_are_causal(self) -> None:
        rows = _make_resampled_rows(count=60, mid_start=100.0, mid_step=1.0)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        featured, _ = create_walk_forward_features(
            raw_df,
            feature_set="candidate_c_core",
            interval="15s",
        )

        altered = raw_df.copy()
        altered.loc[altered.index[-1], "close_mid"] = (
            altered.loc[altered.index[-1], "close_mid"] * 2
        )
        altered_featured, _ = create_walk_forward_features(
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

    def test_cli_parses_candidate_c_core_feature_set(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--feature-set", "candidate_c_core"])
        assert args.feature_set == "candidate_c_core"

    def test_parse_feature_set_candidate_c_core(self) -> None:
        assert parse_feature_set("candidate_c_core") == "candidate_c_core"

    def test_get_walk_forward_feature_columns_candidate_c_core(self) -> None:
        assert get_walk_forward_feature_columns("candidate_c_core") == CANDIDATE_C_CORE_COLUMNS

    def test_candidate_c_core_walk_forward_runs_on_synthetic_data(self, data_root: Path) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=80, mid_step=1.5))

        report = run_walk_forward_evaluation(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=2,
            threshold_bps=0.1,
            train_bars=25,
            test_bars=10,
            step_bars=10,
            feature_set="candidate_c_core",
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            save_reports=False,
            min_windows=1,
            min_trades=1,
        )

        assert report["feature_set"] == "candidate_c_core"
        assert report["feature_columns"] == list(CANDIDATE_C_CORE_COLUMNS)
        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False


class TestCandidateEDepthAwareFeatures:
    def test_candidate_e_depth_aware_feature_set_creates_expected_columns(self) -> None:
        rows = _make_resampled_rows(count=80, include_sizes=True)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        featured, meta = create_walk_forward_features(
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
        rows = _make_resampled_rows(count=40)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))

        with pytest.raises(ValueError, match="bid_size/ask_size"):
            create_walk_forward_features(
                raw_df,
                feature_set="candidate_e_depth_aware",
                interval="15s",
            )

    def test_candidate_e_depth_aware_excludes_future_return_columns(self) -> None:
        rows = _make_resampled_rows(count=40, include_sizes=True)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        _, meta = create_walk_forward_features(
            raw_df,
            feature_set="candidate_e_depth_aware",
            interval="15s",
        )
        assert_no_future_feature_columns(meta["feature_columns"])

    def test_candidate_e_depth_aware_rolling_features_are_causal(self) -> None:
        rows = _make_resampled_rows(count=60, mid_start=100.0, mid_step=1.0, include_sizes=True)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        featured, _ = create_walk_forward_features(
            raw_df,
            feature_set="candidate_e_depth_aware",
            interval="15s",
        )

        altered = raw_df.copy()
        altered.loc[altered.index[-1], "close_mid"] = (
            altered.loc[altered.index[-1], "close_mid"] * 2
        )
        altered_featured, _ = create_walk_forward_features(
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

    def test_candidate_e_depth_aware_microprice_formula(self) -> None:
        rows = _make_resampled_rows(count=5, include_sizes=True, bid_size=2.0, ask_size=1.0)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        raw_df.loc[0, "avg_bid"] = 99.0
        raw_df.loc[0, "avg_ask"] = 101.0
        raw_df.loc[0, "close_mid"] = 100.0
        featured, _ = create_walk_forward_features(
            raw_df,
            feature_set="candidate_e_depth_aware",
            interval="15s",
        )

        expected_microprice = (101.0 * 2.0 + 99.0 * 1.0) / 3.0
        assert featured.iloc[0]["microprice"] == pytest.approx(expected_microprice)
        expected_dev_bps = (expected_microprice - 100.0) / 100.0 * 10_000.0
        assert featured.iloc[0]["microprice_mid_deviation_bps"] == pytest.approx(expected_dev_bps)
        assert featured.iloc[0]["total_top_size"] == pytest.approx(3.0)
        assert featured.iloc[0]["size_imbalance"] == pytest.approx(1.0 / 3.0)

    def test_cli_parses_candidate_e_depth_aware_feature_set(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--feature-set", "candidate_e_depth_aware"])
        assert args.feature_set == "candidate_e_depth_aware"

    def test_get_walk_forward_feature_columns_candidate_e_depth_aware(self) -> None:
        assert (
            get_walk_forward_feature_columns("candidate_e_depth_aware")
            == CANDIDATE_E_DEPTH_AWARE_COLUMNS
        )
        assert CANDIDATE_E_DEPTH_AWARE_EXTRA_COLUMNS == (
            "bid_size",
            "ask_size",
            "total_top_size",
            "size_imbalance",
            "microprice",
            "microprice_mid_deviation_bps",
            "rolling_pressure_change",
        )

    def test_candidate_e_depth_aware_walk_forward_runs_on_synthetic_data(
        self, data_root: Path
    ) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=80, mid_step=1.5, include_sizes=True))

        report = run_walk_forward_evaluation(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=2,
            threshold_bps=0.1,
            train_bars=25,
            test_bars=10,
            step_bars=10,
            feature_set="candidate_e_depth_aware",
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            save_reports=False,
            min_windows=1,
            min_trades=1,
        )

        assert report["feature_set"] == "candidate_e_depth_aware"
        assert report["feature_columns"] == list(CANDIDATE_E_DEPTH_AWARE_COLUMNS)
        assert report["feature_count"] == len(CANDIDATE_E_DEPTH_AWARE_COLUMNS)
        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False


class TestCandidateFRegimeDepthFeatures:
    def test_candidate_f_regime_depth_feature_set_creates_expected_columns(self) -> None:
        rows = _make_resampled_rows(count=300, include_sizes=True)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        featured, meta = create_walk_forward_features(
            raw_df,
            feature_set="candidate_f_regime_depth",
            interval="15s",
        )

        assert meta["feature_set"] == "candidate_f_regime_depth"
        assert meta["feature_columns"] == list(CANDIDATE_F_REGIME_DEPTH_COLUMNS)
        assert meta["feature_count"] == len(CANDIDATE_F_REGIME_DEPTH_COLUMNS)
        for column in CANDIDATE_E_DEPTH_AWARE_COLUMNS:
            assert column in featured.columns
        for column in CANDIDATE_F_REGIME_DEPTH_EXTRA_COLUMNS:
            assert column in featured.columns

    def test_candidate_f_regime_depth_fails_without_size_columns(self) -> None:
        rows = _make_resampled_rows(count=40)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))

        with pytest.raises(ValueError, match="bid_size/ask_size"):
            create_walk_forward_features(
                raw_df,
                feature_set="candidate_f_regime_depth",
                interval="15s",
            )

    def test_candidate_f_regime_depth_excludes_future_return_columns(self) -> None:
        rows = _make_resampled_rows(count=300, include_sizes=True)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        _, meta = create_walk_forward_features(
            raw_df,
            feature_set="candidate_f_regime_depth",
            interval="15s",
        )
        assert_no_future_feature_columns(meta["feature_columns"])

    def test_candidate_f_regime_depth_rolling_features_are_causal(self) -> None:
        rows = _make_resampled_rows(count=300, mid_start=100.0, mid_step=1.0, include_sizes=True)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        featured, _ = create_walk_forward_features(
            raw_df,
            feature_set="candidate_f_regime_depth",
            interval="15s",
        )

        altered = raw_df.copy()
        altered.loc[altered.index[-1], "close_mid"] = (
            altered.loc[altered.index[-1], "close_mid"] * 2
        )
        altered_featured, _ = create_walk_forward_features(
            altered,
            feature_set="candidate_f_regime_depth",
            interval="15s",
        )

        for column in CANDIDATE_F_REGIME_DEPTH_COLUMNS:
            pd.testing.assert_series_equal(
                featured[column].iloc[:-1],
                altered_featured[column].iloc[:-1],
                check_names=False,
            )

    def test_cli_parses_candidate_f_regime_depth_feature_set(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--feature-set", "candidate_f_regime_depth"])
        assert args.feature_set == "candidate_f_regime_depth"

    def test_get_walk_forward_feature_columns_candidate_f_regime_depth(self) -> None:
        assert (
            get_walk_forward_feature_columns("candidate_f_regime_depth")
            == CANDIDATE_F_REGIME_DEPTH_COLUMNS
        )

    def test_regime_breakdown_diagnostics(self) -> None:
        records = [
            {
                "net_return_bps": 5.0,
                "cost_bps": 2.0,
                "confidence": 0.8,
                "low_volatility_regime": 1.0,
                "high_volatility_regime": 0.0,
                "tight_spread_regime": 1.0,
                "wide_spread_regime": 0.0,
            },
            {
                "net_return_bps": -3.0,
                "cost_bps": 2.0,
                "confidence": 0.7,
                "low_volatility_regime": 1.0,
                "high_volatility_regime": 0.0,
                "tight_spread_regime": 0.0,
                "wide_spread_regime": 1.0,
            },
            {
                "net_return_bps": 2.0,
                "cost_bps": 2.0,
                "confidence": 0.6,
                "low_volatility_regime": 0.0,
                "high_volatility_regime": 1.0,
                "tight_spread_regime": 0.0,
                "wide_spread_regime": 0.0,
            },
        ]
        diagnostics = compute_regime_breakdown_diagnostics(records)

        assert set(diagnostics.keys()) == set(REGIME_BREAKDOWN_COLUMNS)
        assert diagnostics["low_volatility_regime"]["trades"] == 2
        assert diagnostics["low_volatility_regime"]["cumulative_net_return_bps"] == pytest.approx(2.0)
        assert diagnostics["low_volatility_regime"]["average_net_return_bps"] == pytest.approx(1.0)
        assert diagnostics["low_volatility_regime"]["win_rate_after_costs"] == pytest.approx(0.5)
        assert diagnostics["high_volatility_regime"]["trades"] == 1
        assert diagnostics["high_volatility_regime"]["cumulative_net_return_bps"] == pytest.approx(2.0)
        assert diagnostics["tight_spread_regime"]["trades"] == 1
        assert diagnostics["wide_spread_regime"]["trades"] == 1

    def test_candidate_f_regime_depth_walk_forward_runs_on_synthetic_data(
        self, data_root: Path
    ) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(
            path, _make_resampled_rows(count=300, mid_step=1.5, include_sizes=True)
        )

        report = run_walk_forward_evaluation(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=2,
            threshold_bps=0.1,
            train_bars=25,
            test_bars=10,
            step_bars=10,
            feature_set="candidate_f_regime_depth",
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            save_reports=False,
            min_windows=1,
            min_trades=1,
        )

        assert report["feature_set"] == "candidate_f_regime_depth"
        assert report["feature_columns"] == list(CANDIDATE_F_REGIME_DEPTH_COLUMNS)
        assert report["feature_count"] == len(CANDIDATE_F_REGIME_DEPTH_COLUMNS)
        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False


class TestCostAwareLabels:
    def test_direction_mode_matches_create_labels(self) -> None:
        featured = _make_labeled_feature_rows(count=12, mid_start=100.0)
        labeled, meta = create_walk_forward_labels(
            featured,
            label_mode="direction",
            horizon=2,
            threshold_bps=1.0,
        )
        expected = create_labels(featured, horizon=2, threshold_bps=1.0)

        assert labeled["label"].tolist() == expected["label"].tolist()
        assert meta["label_mode"] == "direction"
        assert meta["label_cost_multiplier"] == DEFAULT_LABEL_COST_MULTIPLIER
        assert meta["label_min_edge_bps"] == DEFAULT_LABEL_MIN_EDGE_BPS
        assert meta["label_required_edge_summary"]["average_label_required_edge_bps"] == 1.0

    def test_cost_aware_converts_small_moves_to_flat(self) -> None:
        featured = _make_labeled_feature_rows(
            count=6,
            spread_bps=2.0,
            mid_start=100.0,
        )
        featured = create_features(
            pd.DataFrame(
                _make_resampled_rows(count=6, mid_start=100.0, mid_step=0.001, spread_bps=2.0),
                columns=list(RESAMPLED_COLUMNS),
            )
        )
        labeled, _ = create_walk_forward_labels(
            featured,
            label_mode="cost_aware",
            horizon=1,
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            label_cost_multiplier=1.0,
            label_min_edge_bps=0.0,
        )
        ready = labeled.dropna(subset=["future_return_bps"]).head(3)

        # Required edge is fee + slippage + spread ~= 5 bps; small moves become FLAT.
        assert (ready["label"] == "FLAT").all()

    def test_cost_aware_keeps_large_up_and_down_moves(self) -> None:
        rows = _make_resampled_rows(count=6, mid_start=100.0, mid_step=0.0, spread_bps=2.0)
        mids = [100.0, 110.0, 90.0, 120.0, 80.0, 100.0]
        for index, mid in enumerate(mids):
            rows[index]["close_mid"] = mid
            rows[index]["open_mid"] = mid
            rows[index]["high_mid"] = mid + 0.1
            rows[index]["low_mid"] = mid - 0.1
            rows[index]["avg_mid"] = mid
        featured = create_features(pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS)))
        labeled, meta = create_walk_forward_labels(
            featured,
            label_mode="cost_aware",
            horizon=1,
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            label_cost_multiplier=1.0,
            label_min_edge_bps=0.0,
        )
        labels = labeled["label"].dropna().tolist()

        assert "UP" in labels
        assert "DOWN" in labels
        assert meta["label_distribution_by_mode"]["UP"]["count"] >= 1
        assert meta["label_distribution_by_mode"]["DOWN"]["count"] >= 1

    def test_label_cost_multiplier_increases_required_edge(self) -> None:
        rows = _make_resampled_rows(count=3, mid_start=100.0, mid_step=0.8, spread_bps=2.0)
        featured = create_features(pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS)))

        base = create_cost_aware_labels(
            featured,
            horizon=1,
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            label_cost_multiplier=1.0,
        )
        scaled = create_cost_aware_labels(
            featured,
            horizon=1,
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            label_cost_multiplier=3.0,
        )

        assert (
            scaled["label_required_edge_bps"].iloc[0]
            > base["label_required_edge_bps"].iloc[0]
        )
        assert base["label"].tolist().count("UP") >= scaled["label"].tolist().count("UP")

    def test_label_min_edge_bps_increases_required_edge(self) -> None:
        rows = _make_resampled_rows(count=3, mid_start=100.0, mid_step=1.0, spread_bps=2.0)
        featured = create_features(pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS)))

        no_min = create_cost_aware_labels(
            featured,
            horizon=1,
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            label_min_edge_bps=0.0,
        )
        with_min = create_cost_aware_labels(
            featured,
            horizon=1,
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            label_min_edge_bps=20.0,
        )

        assert (
            with_min["label_required_edge_bps"].iloc[0]
            > no_min["label_required_edge_bps"].iloc[0]
        )
        assert with_min["label"].tolist().count("UP") <= no_min["label"].tolist().count("UP")

    def test_label_summary_fields_in_report(self, data_root: Path) -> None:
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
            label_mode="cost_aware",
            label_cost_multiplier=1.5,
            label_min_edge_bps=1.0,
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            save_reports=False,
            min_windows=1,
            min_trades=1,
        )

        assert report["label_mode"] == "cost_aware"
        assert report["label_cost_multiplier"] == 1.5
        assert report["label_min_edge_bps"] == 1.0
        summary = report["label_required_edge_summary"]
        assert "average_label_required_edge_bps" in summary
        assert "min_label_required_edge_bps" in summary
        assert "max_label_required_edge_bps" in summary
        assert summary["min_label_required_edge_bps"] <= summary["average_label_required_edge_bps"]
        assert summary["average_label_required_edge_bps"] <= summary["max_label_required_edge_bps"]
        for label in ("UP", "DOWN", "FLAT"):
            assert label in report["label_distribution_by_mode"]

    def test_markdown_includes_label_fields(self, data_root: Path, reports_dir: Path) -> None:
        report = build_walk_forward_report(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=1,
            threshold_bps=1.0,
            train_bars=20,
            test_bars=10,
            step_bars=10,
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            trade_filter=_permissive_filter(),
            per_coin_results=[
                {
                    "coin": "BTC",
                    "status": "completed",
                    "completed_windows": 1,
                    "total_windows": 1,
                    "skipped_windows": 0,
                    "model_accuracy_mean": 0.5,
                    "baseline_accuracy_mean": 0.5,
                    "accuracy_edge_vs_baseline": 0.0,
                    "label_mode": "cost_aware",
                    "label_cost_multiplier": 1.0,
                    "label_min_edge_bps": 0.0,
                    "label_required_edge_summary": summarize_label_required_edge_bps([5.0, 6.0]),
                    "label_distribution_by_mode": compute_label_distribution(["UP", "FLAT", "FLAT"]),
                    "warnings": [],
                    "windows": [],
                }
            ],
            data_root=data_root,
            label_mode="cost_aware",
        )
        md_path = write_walk_forward_markdown(report, reports_dir=reports_dir)
        md_text = md_path.read_text(encoding="utf-8")
        assert "## Labels" in md_text
        assert "label_mode: cost_aware" in md_text
        assert "average_label_required_edge_bps" in md_text
        assert "label UP:" in md_text

    def test_cost_aware_labels_all_flat_warning(self) -> None:
        warnings = cost_aware_label_warnings(["FLAT"] * 20, label_mode="cost_aware")
        assert any("cost_aware_labels_all_flat" in warning for warning in warnings)

    def test_cost_aware_labels_sparse_warning(self) -> None:
        labels = ["FLAT"] * 19 + ["UP"]
        warnings = cost_aware_label_warnings(labels, label_mode="cost_aware")
        assert any("cost_aware_labels_sparse" in warning for warning in warnings)

    def test_direction_mode_has_no_cost_aware_warnings(self) -> None:
        warnings = cost_aware_label_warnings(["FLAT"] * 10, label_mode="direction")
        assert warnings == []

    def test_class_imbalance_warnings_still_apply(self, data_root: Path) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=80, mid_step=0.01, spread_bps=0.1))

        report = run_walk_forward_evaluation(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=2,
            threshold_bps=0.1,
            train_bars=25,
            test_bars=10,
            step_bars=10,
            label_mode="cost_aware",
            label_cost_multiplier=2.0,
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            save_reports=False,
            min_windows=1,
            min_trades=1,
        )
        assert any("class_imbalance" in warning for warning in report["warnings"])

    def test_cost_aware_walk_forward_safety_flags(self, data_root: Path) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=60, mid_step=1.0))

        report = run_walk_forward_evaluation(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=1,
            train_bars=20,
            test_bars=10,
            step_bars=10,
            label_mode="cost_aware",
            data_root=data_root,
            save_reports=False,
            min_windows=1,
            min_trades=1,
        )
        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False

    def test_cli_parses_label_options(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--label-mode",
                "cost_aware",
                "--label-cost-multiplier",
                "1.25",
                "--label-min-edge-bps",
                "2.0",
            ]
        )
        assert args.label_mode == "cost_aware"
        assert args.label_cost_multiplier == 1.25
        assert args.label_min_edge_bps == 2.0

    def test_cli_label_defaults(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.label_mode == DEFAULT_LABEL_MODE
        assert args.label_cost_multiplier == DEFAULT_LABEL_COST_MULTIPLIER
        assert args.label_min_edge_bps == DEFAULT_LABEL_MIN_EDGE_BPS

    def test_parse_label_mode(self) -> None:
        assert parse_label_mode("direction") == "direction"
        assert parse_label_mode("cost_aware") == "cost_aware"
        assert parse_label_mode("cost_aware_3class") == "cost_aware_3class"
        assert parse_label_mode("binary_tradeable_down") == "binary_tradeable_down"
        assert parse_label_mode("binary_tradeable_up") == "binary_tradeable_up"
        with pytest.raises(ValueError):
            parse_label_mode("invalid")


def _make_candidate_d_featured(
    mids: list[float],
    *,
    spread_bps: float | None = 2.0,
) -> pd.DataFrame:
    rows: list[dict] = []
    for index, mid in enumerate(mids):
        row = {
            "start_timestamp_ms": BASE_MS + index * 15_000,
            "end_timestamp_ms": BASE_MS + index * 15_000 + 14_999,
            "timestamp_iso": f"2024-06-10T00:00:{index:02d}Z",
            "coin": "BTC",
            "open_mid": mid,
            "high_mid": mid + 0.1,
            "low_mid": mid - 0.1,
            "close_mid": mid,
            "avg_bid": mid - 0.5,
            "avg_ask": mid + 0.5,
            "avg_mid": mid,
            "avg_spread": 1.0,
            "avg_bbo_imbalance": 0.2,
            "row_count": 15,
            "missing_seconds": 0,
        }
        if spread_bps is not None:
            row["avg_spread_bps"] = spread_bps
        rows.append(row)

    columns = list(RESAMPLED_COLUMNS)
    if spread_bps is None:
        columns = [column for column in columns if column != "avg_spread_bps"]
    return create_features(pd.DataFrame(rows, columns=columns))


class TestCandidateDLabels:
    FEE_BPS = 2.0
    SLIPPAGE_BPS = 1.0
    SPREAD_BPS = 2.0
    THRESHOLD_BPS = 1.0

    def test_cost_aware_3class_labels_edges(self) -> None:
        featured = _make_candidate_d_featured(
            [100.0, 100.08, 99.92, 99.925, 100.0],
            spread_bps=self.SPREAD_BPS,
        )
        labeled, meta = create_walk_forward_labels(
            featured,
            label_mode="cost_aware_3class",
            horizon=1,
            threshold_bps=self.THRESHOLD_BPS,
            fee_bps=self.FEE_BPS,
            slippage_bps=self.SLIPPAGE_BPS,
        )
        labels = labeled["label"].tolist()

        assert labels[0] == "UP_EDGE"
        assert labels[1] == "DOWN_EDGE"
        assert labels[2] == "NO_TRADE"
        assert meta["label_mode"] == "cost_aware_3class"
        assert meta["estimated_cost_formula"] == "fee_bps + slippage_bps + spread_bps"

    def test_binary_tradeable_down_labels(self) -> None:
        featured = _make_candidate_d_featured(
            [100.0, 99.90, 100.02, 100.0],
            spread_bps=self.SPREAD_BPS,
        )
        labeled, _ = create_walk_forward_labels(
            featured,
            label_mode="binary_tradeable_down",
            horizon=1,
            threshold_bps=self.THRESHOLD_BPS,
            fee_bps=self.FEE_BPS,
            slippage_bps=self.SLIPPAGE_BPS,
        )
        labels = labeled["label"].tolist()

        assert labels[0] == "1"
        assert labels[1] == "0"
        assert labels[2] == "0"

    def test_binary_tradeable_up_labels(self) -> None:
        featured = _make_candidate_d_featured(
            [100.0, 100.08, 100.02, 100.0],
            spread_bps=self.SPREAD_BPS,
        )
        labeled, _ = create_walk_forward_labels(
            featured,
            label_mode="binary_tradeable_up",
            horizon=1,
            threshold_bps=self.THRESHOLD_BPS,
            fee_bps=self.FEE_BPS,
            slippage_bps=self.SLIPPAGE_BPS,
        )
        labels = labeled["label"].tolist()

        assert labels[0] == "1"
        assert labels[1] == "0"
        assert labels[2] == "0"

    def test_missing_spread_bps_uses_fee_plus_slippage_with_warning(self) -> None:
        raw_df = pd.DataFrame({"close_mid": [100.0, 100.08, 100.0]})
        estimated, formula, warnings = compute_candidate_d_estimated_cost_bps(
            raw_df,
            fee_bps=self.FEE_BPS,
            slippage_bps=self.SLIPPAGE_BPS,
        )

        assert formula == "fee_bps + slippage_bps"
        assert any("candidate_d_missing_spread_bps" in warning for warning in warnings)
        assert all(
            value == pytest.approx(self.FEE_BPS + self.SLIPPAGE_BPS)
            for value in estimated.tolist()
        )

        labeled, label_warnings = create_candidate_d_labels(
            raw_df,
            label_mode="cost_aware_3class",
            horizon=1,
            threshold_bps=self.THRESHOLD_BPS,
            fee_bps=self.FEE_BPS,
            slippage_bps=self.SLIPPAGE_BPS,
        )
        assert any("candidate_d_missing_spread_bps" in warning for warning in label_warnings)
        assert labeled.attrs["candidate_d_estimated_cost_formula"] == "fee_bps + slippage_bps"
        assert labeled["estimated_cost_bps"].iloc[0] == pytest.approx(
            self.FEE_BPS + self.SLIPPAGE_BPS
        )

    def test_label_diagnostics_include_class_counts_and_fractions(self) -> None:
        featured = _make_candidate_d_featured(
            [100.0, 100.08, 99.92, 99.925, 100.0],
            spread_bps=self.SPREAD_BPS,
        )
        labeled, _ = create_walk_forward_labels(
            featured,
            label_mode="cost_aware_3class",
            horizon=1,
            threshold_bps=self.THRESHOLD_BPS,
            fee_bps=self.FEE_BPS,
            slippage_bps=self.SLIPPAGE_BPS,
        )
        ready = labeled.dropna(subset=["future_return_bps"]).iloc[:-1]
        diagnostics = build_candidate_d_label_diagnostics(
            ready,
            label_mode="cost_aware_3class",
            horizon_bars=1,
            threshold_bps=self.THRESHOLD_BPS,
            estimated_cost_formula="fee_bps + slippage_bps + spread_bps",
        )

        assert diagnostics["class_counts"]["UP_EDGE"] >= 1
        assert diagnostics["class_counts"]["DOWN_EDGE"] >= 1
        assert diagnostics["class_counts"]["NO_TRADE"] >= 1
        assert sum(diagnostics["class_fractions"].values()) == pytest.approx(1.0)
        assert diagnostics["no_trade_fraction"] == pytest.approx(
            diagnostics["class_fractions"]["NO_TRADE"]
        )
        assert diagnostics["edge_fraction"] == pytest.approx(
            diagnostics["class_fractions"]["UP_EDGE"]
            + diagnostics["class_fractions"]["DOWN_EDGE"]
        )
        assert diagnostics["false_edge_rate_if_predictions_available"] is None
        assert diagnostics["walk_forward_training_supported"] is False

    def test_binary_label_diagnostics_support_walk_forward_training(self) -> None:
        featured = _make_candidate_d_featured(
            [100.0, 100.08, 99.90, 100.0],
            spread_bps=self.SPREAD_BPS,
        )
        labeled, _ = create_walk_forward_labels(
            featured,
            label_mode="binary_tradeable_up",
            horizon=1,
            threshold_bps=self.THRESHOLD_BPS,
            fee_bps=self.FEE_BPS,
            slippage_bps=self.SLIPPAGE_BPS,
        )
        ready = labeled.dropna(subset=["future_return_bps"]).iloc[:-1]
        diagnostics = build_candidate_d_label_diagnostics(
            ready,
            label_mode="binary_tradeable_up",
            horizon_bars=1,
            threshold_bps=self.THRESHOLD_BPS,
            estimated_cost_formula="fee_bps + slippage_bps + spread_bps",
        )
        assert diagnostics["walk_forward_training_supported"] is True
        assert candidate_d_supports_walk_forward_training("binary_tradeable_up") is True
        assert candidate_d_supports_walk_forward_training("binary_tradeable_down") is True
        assert candidate_d_supports_walk_forward_training("cost_aware_3class") is False
        assert is_candidate_d_diagnostics_only("cost_aware_3class") is True
        assert is_binary_tradeable_label_mode("binary_tradeable_down") is True

    def test_run_walk_forward_evaluation_diagnostics_only(self, data_root: Path) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=40, mid_step=1.0))

        report = run_walk_forward_evaluation(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=2,
            threshold_bps=1.0,
            train_bars=20,
            test_bars=10,
            step_bars=10,
            label_mode="cost_aware_3class",
            data_root=data_root,
            save_reports=False,
            min_windows=1,
            min_trades=1,
        )

        assert report["label_mode"] == "cost_aware_3class"
        assert "candidate_d_label_diagnostics" in report
        diagnostics = report["candidate_d_label_diagnostics"]
        assert "class_counts" in diagnostics
        assert "class_fractions" in diagnostics
        assert "no_trade_fraction" in diagnostics
        assert "edge_fraction" in diagnostics
        assert any(
            "candidate_d_diagnostics_only" in warning for warning in report["warnings"]
        )
        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False

    def test_markdown_includes_candidate_d_section(self, data_root: Path, reports_dir: Path) -> None:
        report = build_walk_forward_report(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=1,
            threshold_bps=1.0,
            train_bars=20,
            test_bars=10,
            step_bars=10,
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            trade_filter=_permissive_filter(),
            per_coin_results=[
                {
                    "coin": "BTC",
                    "status": "diagnostics_only",
                    "completed_windows": 0,
                    "total_windows": 0,
                    "skipped_windows": 0,
                    "model_accuracy_mean": 0.0,
                    "baseline_accuracy_mean": 0.0,
                    "accuracy_edge_vs_baseline": 0.0,
                    "label_mode": "cost_aware_3class",
                    "label_distribution_by_mode": compute_candidate_d_class_distribution(
                        ["UP_EDGE", "NO_TRADE", "DOWN_EDGE"],
                        label_mode="cost_aware_3class",
                    ),
                    "label_required_edge_summary": summarize_label_required_edge_bps([6.0, 6.0]),
                    "candidate_d_label_diagnostics": build_candidate_d_label_diagnostics(
                        pd.DataFrame(
                            {
                                "label": ["UP_EDGE", "NO_TRADE", "DOWN_EDGE"],
                                "future_return_bps": [8.0, 1.0, -8.0],
                                "estimated_cost_bps": [5.0, 5.0, 5.0],
                            }
                        ),
                        label_mode="cost_aware_3class",
                        horizon_bars=1,
                        threshold_bps=1.0,
                        estimated_cost_formula="fee_bps + slippage_bps + spread_bps",
                    ),
                    "warnings": [],
                    "windows": [],
                }
            ],
            data_root=data_root,
            label_mode="cost_aware_3class",
        )
        md_path = write_walk_forward_markdown(report, reports_dir=reports_dir)
        md_text = md_path.read_text(encoding="utf-8")
        assert "## Candidate D Cost-Aware Label Diagnostics" in md_text
        assert "no_trade_fraction:" in md_text

    def test_no_future_leakage_in_feature_columns(self) -> None:
        rows = _make_resampled_rows(count=40, mid_step=1.0)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        _, meta = create_walk_forward_features(raw_df, feature_set="basic", interval="15s")
        assert_no_future_feature_columns(meta["feature_columns"])

    def test_cli_parses_candidate_d_label_modes(self) -> None:
        parser = _build_arg_parser()
        for mode in (
            "cost_aware_3class",
            "binary_tradeable_down",
            "binary_tradeable_up",
        ):
            args = parser.parse_args(["--label-mode", mode])
            assert args.label_mode == mode
            assert is_candidate_d_label_mode(mode)


class TestBinaryTradeableWalkForward:
    def test_binary_tradeable_down_trains_and_evaluates(self, data_root: Path) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=80, mid_step=2.5))

        report = run_walk_forward_evaluation(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=2,
            threshold_bps=0.1,
            train_bars=25,
            test_bars=10,
            step_bars=10,
            label_mode="binary_tradeable_down",
            trade_classes=frozenset({"DOWN"}),
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            save_reports=False,
            min_windows=1,
            min_trades=1,
        )

        assert report["label_mode"] == "binary_tradeable_down"
        assert report["completed_windows"] >= 1
        assert "candidate_d_label_diagnostics" in report
        assert report["candidate_d_label_diagnostics"]["walk_forward_training_supported"] is True
        assert not any(
            "candidate_d_diagnostics_only" in warning for warning in report["warnings"]
        )
        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False

    def test_binary_tradeable_up_trains_and_evaluates(self, data_root: Path) -> None:
        path = resampled_csv_path("ETH", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=80, mid_step=2.5, coin="ETH"))

        report = run_walk_forward_evaluation(
            coins=["ETH"],
            interval="15s",
            source="local",
            horizon=2,
            threshold_bps=0.1,
            train_bars=25,
            test_bars=10,
            step_bars=10,
            label_mode="binary_tradeable_up",
            trade_classes=frozenset({"UP"}),
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            save_reports=False,
            min_windows=1,
            min_trades=1,
        )

        assert report["label_mode"] == "binary_tradeable_up"
        assert report["completed_windows"] >= 1
        assert report["candidate_d_label_diagnostics"]["edge_fraction"] >= 0.0
        assert report["positive_after_costs"] in (True, False)

    def test_accepted_trade_net_return_formula_down_and_up(self) -> None:
        row = pd.Series(
            {
                "future_return_bps": 10.0,
                "avg_spread_bps": 2.0,
                "estimated_cost_bps": 5.0,
                "label": "1",
            }
        )

        down = simulate_binary_tradeable_paper_returns(
            pd.DataFrame([row]),
            ["1"],
            [1.0],
            label_mode="binary_tradeable_down",
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            trade_filter=TradeFilterConfig(
                min_confidence=0.0,
                min_edge_bps=0.0,
                trade_classes=frozenset({"DOWN"}),
                cost_multiplier=0.0,
                horizon_return_scale_bps=1000.0,
                threshold_bps=0.0,
            ),
        )
        up = simulate_binary_tradeable_paper_returns(
            pd.DataFrame([row]),
            ["1"],
            [1.0],
            label_mode="binary_tradeable_up",
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            trade_filter=TradeFilterConfig(
                min_confidence=0.0,
                min_edge_bps=0.0,
                trade_classes=frozenset({"UP"}),
                cost_multiplier=0.0,
                horizon_return_scale_bps=1000.0,
                threshold_bps=0.0,
            ),
        )

        assert down["net_returns_bps"][0] == pytest.approx(-10.0 - 5.0)
        assert up["net_returns_bps"][0] == pytest.approx(10.0 - 5.0)

    def test_binary_tradeable_positive_class_sparse_warning(self) -> None:
        warnings = binary_tradeable_label_warnings(
            ["0"] * 19 + ["1"],
            split_name="train",
            label_mode="binary_tradeable_down",
        )
        assert any("binary_tradeable_positive_class_sparse" in warning for warning in warnings)

    def test_accepted_trades_csv_includes_candidate_d_fields(
        self,
        data_root: Path,
        reports_dir: Path,
    ) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=80, mid_step=3.0))

        report = run_walk_forward_evaluation(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=2,
            threshold_bps=0.1,
            train_bars=25,
            test_bars=10,
            step_bars=10,
            label_mode="binary_tradeable_down",
            trade_classes=frozenset({"DOWN"}),
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            reports_dir=reports_dir,
            save_reports=False,
            min_windows=1,
            min_trades=0,
        )
        finalize_accepted_trades_csv(
            report,
            list(report.get("accepted_trade_records", [])),
            reports_dir=reports_dir,
            save_accepted_trades=True,
        )
        csv_path = report.get("accepted_trades_csv_path")
        if csv_path is None:
            pytest.skip("no accepted trades in synthetic run")
        saved = pd.read_csv(csv_path)
        for column in (
            "label_mode",
            "predicted_edge_probability",
            "true_edge_label",
            "estimated_cost_bps",
            "future_return_bps",
            "net_return_bps",
        ):
            assert column in saved.columns
        if not saved.empty:
            assert saved["label_mode"].iloc[0] == "binary_tradeable_down"


def _make_mixed_flat_then_trend_rows(
    *,
    flat_count: int = 30,
    trend_count: int = 50,
    trend_step: float = 10.0,
    include_sizes: bool = False,
) -> list[dict]:
    flat_rows = _make_resampled_rows(
        count=flat_count,
        mid_step=0.0,
        mid_start=100.0,
        include_sizes=include_sizes,
    )
    trend_start_ms = flat_rows[-1]["end_timestamp_ms"] + 1
    trend_rows = _make_resampled_rows(
        count=trend_count,
        mid_step=trend_step,
        mid_start=100.0,
        start_ms=trend_start_ms,
        include_sizes=include_sizes,
    )
    return flat_rows + trend_rows


class TestOneClassTrainingWindows:
    @patch("bid_ask_walk_forward_evaluator._SKLEARN_AVAILABLE", True)
    @patch("bid_ask_walk_forward_evaluator.train_walk_forward_sklearn_model")
    def test_one_class_training_window_skipped_and_does_not_call_fit(
        self,
        mock_train,
        data_root: Path,
        reports_dir: Path,
    ) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=60, mid_step=0.0, include_sizes=True))

        result = evaluate_coin_walk_forward(
            coin="BTC",
            interval="15s",
            source="local",
            horizon=2,
            threshold_bps=50.0,
            train_bars=20,
            test_bars=10,
            step_bars=10,
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            trade_filter=_permissive_filter(),
            data_root=data_root,
            save_window_models=False,
            models_dir=reports_dir / "models",
            label_mode="binary_tradeable_up",
            feature_set="candidate_f_regime_depth",
        )

        skipped = [
            window
            for window in result["windows"]
            if window.get("reason") == ONE_CLASS_TRAINING_WINDOW_REASON
        ]
        assert skipped, "expected at least one one-class skipped window"
        assert skipped[0]["status"] == "skipped"
        assert skipped[0]["class_counts"] == {"0": 20}
        mock_train.assert_not_called()

    @patch("bid_ask_walk_forward_evaluator._SKLEARN_AVAILABLE", True)
    def test_all_windows_one_class_writes_rejected_report_not_crash(
        self,
        data_root: Path,
        reports_dir: Path,
    ) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=60, mid_step=0.0, include_sizes=True))

        report = run_walk_forward_evaluation(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=2,
            threshold_bps=50.0,
            train_bars=20,
            test_bars=10,
            step_bars=10,
            label_mode="binary_tradeable_up",
            feature_set="candidate_f_regime_depth",
            trade_classes=frozenset({"UP"}),
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            reports_dir=reports_dir,
            save_reports=True,
            min_windows=1,
            min_trades=0,
        )

        assert report["status"] == "rejected"
        assert report["rejection_reason"] == ALL_WINDOWS_SKIPPED_ONE_CLASS_REJECTION
        assert report["completed_windows"] == 0
        assert report["one_class_training_windows_count"] >= 1
        assert report["skipped_window_reasons"][ONE_CLASS_TRAINING_WINDOW_REASON] >= 1
        assert report["accepted_trades_count"] == 0
        assert report["cumulative_net_return_bps"] == 0.0
        assert report["average_net_return_bps"] == 0.0
        assert report["win_rate"] is None
        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False
        assert Path(report["json_report_path"]).is_file()
        assert any(
            "one_class_training_windows" in warning for warning in report["warnings"]
        )

    @patch("bid_ask_walk_forward_evaluator._SKLEARN_AVAILABLE", True)
    @patch("bid_ask_walk_forward_evaluator.train_walk_forward_sklearn_model")
    def test_mixed_valid_and_skipped_windows_continue(
        self,
        mock_train,
        data_root: Path,
        reports_dir: Path,
    ) -> None:
        mock_train.return_value = {
            "available": True,
            "model": None,
            "scaler": None,
            "model_type": "mock",
            "message": None,
        }
        path = resampled_csv_path("ETH", "15s", data_root)
        _write_resampled_csv(
            path,
            _make_mixed_flat_then_trend_rows(
                flat_count=25,
                trend_count=55,
                trend_step=8.0,
                include_sizes=True,
            ),
        )

        report = run_walk_forward_evaluation(
            coins=["ETH"],
            interval="15s",
            source="local",
            horizon=2,
            threshold_bps=0.5,
            train_bars=20,
            test_bars=10,
            step_bars=10,
            label_mode="binary_tradeable_up",
            feature_set="candidate_e_depth_aware",
            trade_classes=frozenset({"UP"}),
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            save_reports=False,
            min_windows=1,
            min_trades=0,
        )

        assert report["completed_windows"] >= 1
        assert report["one_class_training_windows_count"] >= 1
        assert report["skipped_window_reasons"][ONE_CLASS_TRAINING_WINDOW_REASON] >= 1
        assert report["status"] != "rejected"

    @patch("bid_ask_walk_forward_evaluator._SKLEARN_AVAILABLE", True)
    def test_train_walk_forward_sklearn_model_reports_class_counts_for_one_class(
        self,
    ) -> None:
        pytest.importorskip("sklearn")
        train_df = pd.DataFrame(
            {
                "label": ["0"] * 10,
                "future_return_bps": [0.0] * 10,
                "return_1": [0.0] * 10,
                "return_15s_bps": [0.0] * 10,
                "spread_bps": [2.0] * 10,
                "avg_spread_bps": [2.0] * 10,
                "volatility_bps": [1.0] * 10,
                "avg_bbo_imbalance": [0.1] * 10,
            }
        )
        feature_columns = tuple(
            column for column in FEATURE_COLUMNS if column in train_df.columns
        )
        with patch("sklearn.ensemble.RandomForestClassifier.fit") as mock_fit:
            result = train_walk_forward_sklearn_model(train_df, feature_columns)
        mock_fit.assert_not_called()
        assert result["reason"] == ONE_CLASS_TRAINING_WINDOW_REASON
        assert result["class_counts"] == {"0": 10}
        assert result["model"] is None

    @patch("bid_ask_walk_forward_evaluator._SKLEARN_AVAILABLE", True)
    def test_direction_label_mode_one_class_window_skipped(
        self,
        data_root: Path,
        reports_dir: Path,
    ) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=60, mid_step=0.0, include_sizes=True))

        result = evaluate_coin_walk_forward(
            coin="BTC",
            interval="15s",
            source="local",
            horizon=2,
            threshold_bps=1000.0,
            train_bars=20,
            test_bars=10,
            step_bars=10,
            fee_bps=0.0,
            slippage_bps=0.0,
            default_spread_bps=0.0,
            trade_filter=_permissive_filter(),
            data_root=data_root,
            save_window_models=False,
            models_dir=reports_dir / "models",
            label_mode="direction",
        )

        skipped = [
            window
            for window in result["windows"]
            if window.get("reason") == ONE_CLASS_TRAINING_WINDOW_REASON
        ]
        assert skipped
        assert _training_label_class_counts(["FLAT"] * 5) == {"FLAT": 5}


class TestComputeReadinessFlags:
    def test_negative_results_all_false_except_real_execution(self) -> None:
        flags = compute_readiness_flags(
            accuracy_edge_vs_baseline=-0.01,
            cumulative_net_return_bps=-5.0,
            completed_windows=1,
            total_paper_trades=10,
            min_windows=5,
            min_trades=100,
        )
        assert flags["beats_baseline_accuracy"] is False
        assert flags["positive_after_costs"] is False
        assert flags["enough_windows"] is False
        assert flags["enough_trades"] is False
        assert flags["ready_for_more_research"] is False
        assert flags["real_execution_ready"] is False

    def test_positive_after_costs_when_net_return_positive(self) -> None:
        flags = compute_readiness_flags(
            accuracy_edge_vs_baseline=0.05,
            cumulative_net_return_bps=12.5,
            completed_windows=5,
            total_paper_trades=100,
            min_windows=5,
            min_trades=100,
        )
        assert flags["positive_after_costs"] is True
        assert flags["beats_baseline_accuracy"] is True
        assert flags["enough_windows"] is True
        assert flags["enough_trades"] is True
        assert flags["ready_for_more_research"] is True
        assert flags["real_execution_ready"] is False

    def test_ready_for_more_research_requires_windows_and_trades_only(self) -> None:
        flags = compute_readiness_flags(
            accuracy_edge_vs_baseline=-0.1,
            cumulative_net_return_bps=-1.0,
            completed_windows=10,
            total_paper_trades=200,
            min_windows=5,
            min_trades=100,
        )
        assert flags["ready_for_more_research"] is True
        assert flags["beats_baseline_accuracy"] is False
        assert flags["positive_after_costs"] is False


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
        confidence_edges = [1.0]
        trade_filter = _permissive_filter()

        no_cost = simulate_paper_signal_returns(
            pd.DataFrame([row_no_spread]),
            predictions,
            confidences,
            confidence_edges,
            fee_bps=0.0,
            slippage_bps=0.0,
            default_spread_bps=0.0,
            trade_filter=trade_filter,
        )
        with_cost = simulate_paper_signal_returns(
            pd.DataFrame([row_with_spread]),
            predictions,
            confidences,
            confidence_edges,
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            trade_filter=trade_filter,
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
            [1.0],
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            trade_filter=_permissive_filter(),
        )
        assert result["net_returns_bps"][0] == 0.0
        assert result["total_paper_trades"] == 0
        assert result["raw_predictions"] == 0

    def test_down_signal_inverse_return(self) -> None:
        row = pd.Series({"future_return_bps": 8.0, "avg_spread_bps": 1.0})
        test_df = pd.DataFrame([row])
        result = simulate_paper_signal_returns(
            test_df,
            ["DOWN"],
            [1.0],
            [1.0],
            fee_bps=1.0,
            slippage_bps=1.0,
            default_spread_bps=1.0,
            trade_filter=_permissive_filter(),
        )
        assert result["gross_returns_bps"][0] == pytest.approx(-8.0)
        assert result["net_returns_bps"][0] == pytest.approx(-8.0 - 3.0)


class TestCostAwareTradeFiltering:
    def test_high_min_edge_reduces_trades(self) -> None:
        rows = [
            {"future_return_bps": 20.0, "avg_spread_bps": 2.0},
            {"future_return_bps": 15.0, "avg_spread_bps": 2.0},
        ]
        test_df = pd.DataFrame(rows)
        loose = simulate_paper_signal_returns(
            test_df,
            ["UP", "UP"],
            [0.9, 0.9],
            [0.8, 0.8],
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            trade_filter=_permissive_filter(min_edge_bps=0.0, horizon_return_scale_bps=50.0),
        )
        strict = simulate_paper_signal_returns(
            test_df,
            ["UP", "UP"],
            [0.9, 0.9],
            [0.8, 0.8],
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            trade_filter=_permissive_filter(min_edge_bps=100.0, horizon_return_scale_bps=50.0),
        )
        assert loose["filtered_paper_trades"] >= strict["filtered_paper_trades"]
        assert strict["blocked_by_cost"] >= 1

    def test_max_spread_blocks_high_spread_rows(self) -> None:
        rows = [
            {"future_return_bps": 20.0, "avg_spread_bps": 1.0},
            {"future_return_bps": 20.0, "avg_spread_bps": 10.0},
        ]
        test_df = pd.DataFrame(rows)
        result = simulate_paper_signal_returns(
            test_df,
            ["UP", "UP"],
            [1.0, 1.0],
            [1.0, 1.0],
            fee_bps=0.0,
            slippage_bps=0.0,
            default_spread_bps=2.0,
            trade_filter=_permissive_filter(max_spread_bps=2.0),
        )
        assert result["blocked_by_spread"] == 1
        assert result["filtered_paper_trades"] == 1

    def test_cost_multiplier_increases_required_edge(self) -> None:
        row = pd.Series({"future_return_bps": 20.0, "avg_spread_bps": 2.0})
        test_df = pd.DataFrame([row])
        base = simulate_paper_signal_returns(
            test_df,
            ["UP"],
            [0.6],
            [0.5],
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            trade_filter=_permissive_filter(
                cost_multiplier=1.0,
                horizon_return_scale_bps=20.0,
            ),
        )
        scaled = simulate_paper_signal_returns(
            test_df,
            ["UP"],
            [0.6],
            [0.5],
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            trade_filter=_permissive_filter(
                cost_multiplier=5.0,
                horizon_return_scale_bps=20.0,
            ),
        )
        assert base["average_required_edge_bps"] < scaled["average_required_edge_bps"]
        assert scaled["filtered_paper_trades"] <= base["filtered_paper_trades"]

    def test_filtered_paper_trades_lte_raw_predictions(self) -> None:
        rows = [
            {"future_return_bps": 10.0, "avg_spread_bps": 2.0},
            {"future_return_bps": 10.0, "avg_spread_bps": 2.0},
            {"future_return_bps": 0.0, "avg_spread_bps": 2.0},
        ]
        test_df = pd.DataFrame(rows)
        result = simulate_paper_signal_returns(
            test_df,
            ["UP", "DOWN", "FLAT"],
            [1.0, 0.2, 1.0],
            [0.9, 0.9, 0.9],
            fee_bps=1.0,
            slippage_bps=1.0,
            default_spread_bps=1.0,
            trade_filter=build_trade_filter(
                min_confidence=0.5,
                threshold_bps=1.0,
                horizon_return_scale_bps=100.0,
            ),
        )
        assert result["filtered_paper_trades"] <= result["raw_predictions"]
        assert result["raw_predictions"] == 2

    def test_blocked_counters_add_up_reasonably(self) -> None:
        rows = [
            {"future_return_bps": 10.0, "avg_spread_bps": 2.0},
            {"future_return_bps": 10.0, "avg_spread_bps": 2.0},
            {"future_return_bps": 10.0, "avg_spread_bps": 8.0},
        ]
        test_df = pd.DataFrame(rows)
        result = simulate_paper_signal_returns(
            test_df,
            ["UP", "UP", "UP"],
            [0.2, 0.9, 0.9],
            [0.9, 0.1, 0.9],
            fee_bps=2.0,
            slippage_bps=1.0,
            default_spread_bps=2.0,
            trade_filter=build_trade_filter(
                min_confidence=0.5,
                threshold_bps=1.0,
                max_spread_bps=3.0,
                min_edge_bps=50.0,
                horizon_return_scale_bps=10.0,
            ),
        )
        blocked_total = (
            result["blocked_by_confidence"]
            + result["blocked_by_cost"]
            + result["blocked_by_spread"]
            + result["blocked_by_volatility"]
            + result["blocked_by_trade_class"]
        )
        assert blocked_total == result["raw_predictions"] - result["filtered_paper_trades"]

    def test_evaluate_trade_filter_trade_class(self) -> None:
        row = pd.Series({"future_return_bps": 10.0, "avg_spread_bps": 1.0})
        allowed, reason, _, _ = evaluate_trade_filter(
            prediction="DOWN",
            confidence=1.0,
            confidence_edge=1.0,
            row=row,
            fee_bps=0.0,
            slippage_bps=0.0,
            default_spread_bps=1.0,
            trade_filter=build_trade_filter(
                min_confidence=0.0,
                threshold_bps=0.1,
                trade_classes=frozenset({"UP"}),
                horizon_return_scale_bps=100.0,
            ),
        )
        assert allowed is False
        assert reason == "trade_class"


class TestDrawdownAndStreak:
    def test_max_drawdown(self) -> None:
        returns = [5.0, -10.0, 3.0, -2.0]
        assert max_drawdown_bps(returns) == pytest.approx(10.0)

    def test_longest_losing_streak(self) -> None:
        returns = [1.0, -1.0, -2.0, 3.0, -1.0]
        assert longest_losing_streak(returns) == 2


class TestAcceptedTradeNetReturnAggregation:
    def test_average_equals_cumulative_over_trade_count(self) -> None:
        accepted = [10.0, -4.0, 6.0, -2.0]
        metrics = aggregate_accepted_trade_metrics(accepted)

        assert metrics["filtered_paper_trades"] == 4
        assert metrics["cumulative_net_return_bps"] == pytest.approx(10.0)
        assert metrics["average_net_return_bps"] == pytest.approx(2.5)
        assert metrics["expected_average_from_cumulative"] == pytest.approx(2.5)
        assert metrics["average_net_return_difference_abs"] == pytest.approx(0.0)
        assert metrics["net_return_consistency_check"] is True

    def test_positive_average_cannot_coexist_with_negative_cumulative(self) -> None:
        accepted = [5.0, -3.0, 2.0, -10.0]
        metrics = aggregate_accepted_trade_metrics(accepted)

        assert metrics["cumulative_net_return_bps"] < 0
        assert metrics["average_net_return_bps"] < 0
        assert metrics["positive_after_costs"] is False
        assert metrics["net_return_consistency_check"] is True

    def test_zero_trades_produce_zero_return_metrics(self) -> None:
        metrics = aggregate_accepted_trade_metrics([])

        assert metrics["filtered_paper_trades"] == 0
        assert metrics["cumulative_net_return_bps"] == 0.0
        assert metrics["average_net_return_bps"] == 0.0
        assert metrics["win_rate_after_costs"] == 0.0
        assert metrics["max_drawdown_bps"] == 0.0
        assert metrics["longest_losing_streak"] == 0
        assert metrics["positive_after_costs"] is False
        assert metrics["net_return_consistency_check"] is True

    def test_win_rate_uses_only_accepted_filtered_trades(self) -> None:
        accepted = [3.0, -1.0, 2.0, -4.0]
        metrics = aggregate_accepted_trade_metrics(accepted)

        assert metrics["win_rate_after_costs"] == pytest.approx(0.5)

    def test_drawdown_uses_accepted_trade_cumulative_curve(self) -> None:
        accepted = [5.0, -10.0, 3.0, -2.0]
        metrics = aggregate_accepted_trade_metrics(accepted)

        assert metrics["max_drawdown_bps"] == pytest.approx(max_drawdown_bps(accepted))
        assert metrics["longest_losing_streak"] == longest_losing_streak(accepted)

    def test_report_aggregation_uses_single_accepted_trade_list(
        self, data_root: Path, reports_dir: Path
    ) -> None:
        coin_a = {
            "coin": "BTC",
            "status": "completed",
            "completed_windows": 1,
            "total_windows": 1,
            "skipped_windows": 0,
            "model_accuracy_mean": 0.6,
            "baseline_accuracy_mean": 0.5,
            "accuracy_edge_vs_baseline": 0.1,
            "total_paper_signals": 2,
            "raw_predictions": 2,
            "filtered_paper_trades": 2,
            "total_paper_trades": 2,
            "accepted_trade_net_returns_bps": [100.0, 100.0],
            "trade_returns_bps": [100.0, 100.0],
            "cumulative_net_return_bps": 200.0,
            "average_net_return_bps": 100.0,
            "average_gross_return_bps": 100.0,
            "average_cost_bps": 0.0,
            "average_required_edge_bps": 0.0,
            "blocked_by_confidence": 0,
            "blocked_by_cost": 0,
            "blocked_by_spread": 0,
            "blocked_by_volatility": 0,
            "blocked_by_trade_class": 0,
            "warnings": [],
            "windows": [],
        }
        coin_b = {
            **coin_a,
            "coin": "ETH",
            "accepted_trade_net_returns_bps": [-0.5] * 1000,
            "trade_returns_bps": [-0.5] * 1000,
            "filtered_paper_trades": 1000,
            "total_paper_trades": 1000,
            "raw_predictions": 1000,
            "total_paper_signals": 1000,
            "cumulative_net_return_bps": -500.0,
            "average_net_return_bps": -0.5,
        }

        report = build_walk_forward_report(
            coins=["BTC", "ETH"],
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
            trade_filter=_permissive_filter(),
            per_coin_results=[coin_a, coin_b],
            data_root=data_root,
        )

        assert report["filtered_paper_trades"] == 1002
        assert report["cumulative_net_return_bps"] == pytest.approx(-300.0)
        assert report["average_net_return_bps"] == pytest.approx(-300.0 / 1002)
        assert report["expected_average_from_cumulative"] == pytest.approx(-300.0 / 1002)
        assert report["net_return_consistency_check"] is True
        assert not (
            report["average_net_return_bps"] > 0 and report["cumulative_net_return_bps"] < 0
        )

    def test_consistency_check_passes_on_normal_reports(
        self, data_root: Path, reports_dir: Path
    ) -> None:
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
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            reports_dir=reports_dir,
            save_reports=False,
            min_windows=1,
            min_trades=1,
        )

        assert report["net_return_consistency_check"] is True
        assert report["average_net_return_difference_abs"] == pytest.approx(0.0)
        if report["filtered_paper_trades"] > 0:
            assert report["average_net_return_bps"] == pytest.approx(
                report["cumulative_net_return_bps"] / report["filtered_paper_trades"]
            )

    def test_markdown_includes_consistency_fields(
        self, data_root: Path, reports_dir: Path
    ) -> None:
        report = build_walk_forward_report(
            coins=["BTC"],
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
            trade_filter=_permissive_filter(),
            per_coin_results=[
                {
                    "coin": "BTC",
                    "status": "completed",
                    "completed_windows": 1,
                    "total_windows": 1,
                    "skipped_windows": 0,
                    "model_accuracy_mean": 0.5,
                    "baseline_accuracy_mean": 0.5,
                    "accuracy_edge_vs_baseline": 0.0,
                    "total_paper_signals": 3,
                    "raw_predictions": 3,
                    "filtered_paper_trades": 3,
                    "total_paper_trades": 3,
                    "accepted_trade_net_returns_bps": [2.0, -1.0, 3.0],
                    "trade_returns_bps": [2.0, -1.0, 3.0],
                    "cumulative_net_return_bps": 4.0,
                    "average_net_return_bps": 4.0 / 3,
                    "average_gross_return_bps": 2.0,
                    "average_cost_bps": 0.0,
                    "average_required_edge_bps": 0.0,
                    "blocked_by_confidence": 0,
                    "blocked_by_cost": 0,
                    "blocked_by_spread": 0,
                    "blocked_by_volatility": 0,
                    "blocked_by_trade_class": 0,
                    "warnings": [],
                    "windows": [],
                }
            ],
            data_root=data_root,
        )
        md_path = write_walk_forward_markdown(report, reports_dir=reports_dir)
        md_text = md_path.read_text(encoding="utf-8")
        assert "filtered_paper_trades (accepted)" in md_text
        assert "net_return_consistency_check" in md_text
        assert "cumulative_net_return_bps" in md_text
        assert "average_net_return_bps" in md_text

    def test_paper_only_and_execution_disabled(self, data_root: Path, reports_dir: Path) -> None:
        report = build_walk_forward_report(
            coins=["SOL"],
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
            trade_filter=_permissive_filter(),
            per_coin_results=[],
            data_root=data_root,
        )
        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False


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
            trade_filter=_permissive_filter(),
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
        for field in FILTER_METRIC_FIELDS:
            assert field in result
        assert result["filtered_paper_trades"] <= result["raw_predictions"]
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
            trade_filter=_permissive_filter(),
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
            trade_filter=_permissive_filter(),
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
            horizon_return_scale_bps=1000.0,
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
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            reports_dir=reports_dir,
            min_windows=2,
            min_trades=1,
        )

        _assert_readiness_booleans(report)
        assert report["real_execution_ready"] is False

    def test_negative_results_produce_false_readiness_flags(
        self, data_root: Path, reports_dir: Path
    ) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=50, mid_step=0.01))

        report = run_walk_forward_evaluation(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=2,
            threshold_bps=1.0,
            train_bars=20,
            test_bars=10,
            step_bars=10,
            fee_bps=10.0,
            slippage_bps=10.0,
            default_spread_bps=10.0,
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            reports_dir=reports_dir,
            min_windows=100,
            min_trades=10_000,
        )

        _assert_readiness_booleans(report)
        assert report["beats_baseline_accuracy"] is False
        assert report["positive_after_costs"] is False
        assert report["enough_windows"] is False
        assert report["enough_trades"] is False
        assert report["ready_for_more_research"] is False

    def test_positive_synthetic_can_produce_positive_after_costs(
        self, data_root: Path, reports_dir: Path
    ) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(
            path,
            _make_resampled_rows(count=120, mid_step=5.0, spread_bps=0.1),
        )

        report = run_walk_forward_evaluation(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=2,
            threshold_bps=0.1,
            train_bars=30,
            test_bars=15,
            step_bars=15,
            fee_bps=0.0,
            slippage_bps=0.0,
            default_spread_bps=0.0,
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            reports_dir=reports_dir,
            min_windows=1,
            min_trades=1,
        )

        _assert_readiness_booleans(report)
        assert report["positive_after_costs"] is True
        assert report["cumulative_net_return_bps"] > 0

    def test_json_report_top_level_readiness_not_none(
        self, data_root: Path, reports_dir: Path
    ) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=60))

        report = run_walk_forward_evaluation(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=1,
            train_bars=20,
            test_bars=10,
            step_bars=10,
            data_root=data_root,
            reports_dir=reports_dir,
        )
        loaded = json.loads(Path(report["json_report_path"]).read_text(encoding="utf-8"))
        _assert_readiness_booleans(loaded)
        assert loaded["paper_only"] is True
        assert loaded["execution_enabled"] is False

    def test_json_report_includes_filter_metrics(
        self, data_root: Path, reports_dir: Path
    ) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=60))

        report = run_walk_forward_evaluation(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=1,
            train_bars=20,
            test_bars=10,
            step_bars=10,
            data_root=data_root,
            reports_dir=reports_dir,
            horizon_return_scale_bps=1000.0,
            min_confidence=0.0,
        )
        for field in FILTER_METRIC_FIELDS:
            assert field in report
        loaded = json.loads(Path(report["json_report_path"]).read_text(encoding="utf-8"))
        for field in FILTER_METRIC_FIELDS:
            assert field in loaded

    def test_filtered_returns_use_only_accepted_trades(
        self, data_root: Path, reports_dir: Path
    ) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=80, mid_step=3.0, spread_bps=0.5))

        permissive = run_walk_forward_evaluation(
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
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            reports_dir=reports_dir,
            save_reports=False,
            min_windows=1,
            min_trades=1,
        )
        strict = run_walk_forward_evaluation(
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
            min_edge_bps=500.0,
            horizon_return_scale_bps=10.0,
            data_root=data_root,
            reports_dir=reports_dir,
            save_reports=False,
            min_windows=1,
            min_trades=1,
        )
        assert strict["filtered_paper_trades"] <= permissive["filtered_paper_trades"]
        assert strict["total_paper_trades"] == strict["filtered_paper_trades"]

    def test_enough_trades_uses_filtered_paper_trades(
        self, data_root: Path, reports_dir: Path
    ) -> None:
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
            min_confidence=0.0,
            min_edge_bps=500.0,
            horizon_return_scale_bps=10.0,
            data_root=data_root,
            reports_dir=reports_dir,
            save_reports=False,
            min_windows=1,
            min_trades=1000,
        )
        assert report["filtered_paper_trades"] < 1000
        assert report["enough_trades"] is False
        assert report["readiness"]["enough_trades"] is False

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

        report = build_walk_forward_report(
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
            trade_filter=_permissive_filter(),
            per_coin_results=[],
            data_root=data_root,
        )
        _assert_readiness_booleans(report)

        json_path = write_walk_forward_json(report, reports_dir=reports_dir)
        md_path = write_walk_forward_markdown(report, reports_dir=reports_dir)
        assert json_path.parent == reports_dir
        assert md_path.parent == reports_dir
        assert json_path.is_file()
        assert md_path.is_file()
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["paper_only"] is True
        assert loaded["execution_enabled"] is False
        _assert_readiness_booleans(loaded)
        md_text = md_path.read_text(encoding="utf-8")
        for field in READINESS_FIELDS:
            assert field in md_text


class TestCLI:
    def test_cli_parses_trade_filter_options(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--min-edge-bps",
                "2.5",
                "--max-spread-bps",
                "3.0",
                "--min-volatility-bps",
                "0.5",
                "--max-volatility-bps",
                "10.0",
                "--trade-classes",
                "UP",
                "--cost-multiplier",
                "1.5",
                "--horizon-return-scale-bps",
                "12.0",
                "--min-absolute-future-label-bps",
                "1.5",
            ]
        )
        assert args.min_edge_bps == 2.5
        assert args.max_spread_bps == 3.0
        assert args.min_volatility_bps == 0.5
        assert args.max_volatility_bps == 10.0
        assert args.trade_classes == "UP"
        assert args.cost_multiplier == 1.5
        assert args.horizon_return_scale_bps == 12.0
        assert args.min_absolute_future_label_bps == 1.5

    def test_parse_trade_classes(self) -> None:
        assert parse_trade_classes("UP,DOWN") == frozenset({"UP", "DOWN"})
        assert parse_trade_classes("UP") == frozenset({"UP"})

    def test_cli_parses_min_windows_and_min_trades(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--min-windows", "3", "--min-trades", "50"])
        assert args.min_windows == 3
        assert args.min_trades == 50

    def test_cli_min_windows_min_trades_defaults(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.min_windows == DEFAULT_MIN_WINDOWS
        assert args.min_trades == DEFAULT_MIN_TRADES

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


class TestTradeDiagnostics:
    def test_simulate_tracks_blocked_and_accepted_by_class(self) -> None:
        rows = [
            {"future_return_bps": 10.0, "avg_spread_bps": 2.0, "label": "UP"},
            {"future_return_bps": -8.0, "avg_spread_bps": 2.0, "label": "DOWN"},
            {"future_return_bps": 5.0, "avg_spread_bps": 2.0, "label": "FLAT"},
        ]
        test_df = pd.DataFrame(rows)
        result = simulate_paper_signal_returns(
            test_df,
            ["UP", "DOWN", "FLAT"],
            [1.0, 0.2, 1.0],
            [1.0, 1.0, 1.0],
            fee_bps=1.0,
            slippage_bps=1.0,
            default_spread_bps=1.0,
            trade_filter=build_trade_filter(
                min_confidence=0.5,
                threshold_bps=1.0,
                horizon_return_scale_bps=100.0,
            ),
            coin="BTC",
            window_index=0,
        )

        assert result["accepted_by_class"]["UP"] == 1
        assert result["accepted_by_class"]["DOWN"] == 0
        assert result["accepted_by_class"]["FLAT"] == 0
        assert result["blocked_by_class"]["DOWN"] == 1
        assert result["blocked_by_class"]["FLAT"] == 1
        assert "class_trade_diagnostics" in result
        assert result["class_trade_diagnostics"]["UP"]["trades"] == 1
        assert result["class_trade_diagnostics"]["DOWN"]["trades"] == 0
        assert result["class_trade_diagnostics"]["FLAT"]["trades"] == 0

    def test_aggregate_class_trade_diagnostics_separates_up_and_down(self) -> None:
        records = [
            {
                "predicted_class": "UP",
                "net_return_bps": 5.0,
                "cost_bps": 2.0,
                "confidence": 0.9,
            },
            {
                "predicted_class": "UP",
                "net_return_bps": -1.0,
                "cost_bps": 2.0,
                "confidence": 0.8,
            },
            {
                "predicted_class": "DOWN",
                "net_return_bps": 3.0,
                "cost_bps": 2.5,
                "confidence": 0.7,
            },
        ]
        diagnostics = aggregate_class_trade_diagnostics(records)

        assert diagnostics["UP"]["trades"] == 2
        assert diagnostics["UP"]["cumulative_net_return_bps"] == pytest.approx(4.0)
        assert diagnostics["UP"]["average_net_return_bps"] == pytest.approx(2.0)
        assert diagnostics["UP"]["win_rate_after_costs"] == pytest.approx(0.5)
        assert diagnostics["DOWN"]["trades"] == 1
        assert diagnostics["DOWN"]["cumulative_net_return_bps"] == pytest.approx(3.0)
        assert diagnostics["FLAT"]["trades"] == 0

    def test_aggregate_coin_class_trade_diagnostics_separates_coins(self) -> None:
        records = [
            {
                "coin": "BTC",
                "predicted_class": "UP",
                "net_return_bps": 4.0,
                "cost_bps": 1.0,
                "confidence": 0.9,
            },
            {
                "coin": "ETH",
                "predicted_class": "DOWN",
                "net_return_bps": -2.0,
                "cost_bps": 1.0,
                "confidence": 0.8,
            },
            {
                "coin": "SOL",
                "predicted_class": "DOWN",
                "net_return_bps": 1.0,
                "cost_bps": 1.0,
                "confidence": 0.75,
            },
        ]
        diagnostics = aggregate_coin_class_trade_diagnostics(
            records,
            coins=["BTC", "ETH", "SOL"],
        )

        assert set(diagnostics) == {"BTC", "ETH", "SOL"}
        assert diagnostics["BTC"]["UP"]["trades"] == 1
        assert diagnostics["ETH"]["DOWN"]["trades"] == 1
        assert diagnostics["SOL"]["DOWN"]["trades"] == 1
        assert diagnostics["BTC"]["DOWN"]["trades"] == 0

    def test_walk_forward_report_includes_trade_diagnostics(
        self, data_root: Path, reports_dir: Path
    ) -> None:
        for coin in ("BTC", "ETH", "SOL"):
            path = resampled_csv_path(coin, "15s", data_root)
            _write_resampled_csv(
                path,
                _make_resampled_rows(coin=coin, count=80, mid_step=1.5),
            )

        report = run_walk_forward_evaluation(
            coins=["BTC", "ETH", "SOL"],
            interval="15s",
            source="local",
            horizon=2,
            threshold_bps=0.1,
            train_bars=25,
            test_bars=10,
            step_bars=10,
            fee_bps=1.0,
            slippage_bps=1.0,
            default_spread_bps=1.0,
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            reports_dir=reports_dir,
            save_reports=False,
            min_windows=1,
            min_trades=1,
        )

        assert "class_trade_diagnostics" in report
        assert "coin_class_trade_diagnostics" in report
        assert "blocked_by_class" in report
        assert "accepted_by_class" in report
        for label in ("UP", "DOWN", "FLAT"):
            assert label in report["class_trade_diagnostics"]
            assert label in report["blocked_by_class"]
            assert label in report["accepted_by_class"]
        for coin in ("BTC", "ETH", "SOL"):
            assert coin in report["coin_class_trade_diagnostics"]

        md_path = write_walk_forward_markdown(report, reports_dir=reports_dir)
        md_text = md_path.read_text(encoding="utf-8")
        assert "## Trade Diagnostics" in md_text
        assert "### Coin-class trade diagnostics" in md_text
        assert "accepted UP:" in md_text

    def test_save_accepted_trades_writes_paper_only_csv(
        self, data_root: Path, reports_dir: Path
    ) -> None:
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
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            reports_dir=reports_dir,
            save_reports=False,
            min_windows=1,
            min_trades=1,
        )

        csv_path = Path(report["accepted_trades_csv_path"])
        assert csv_path.is_file()
        assert csv_path.parent == reports_dir
        assert report["accepted_trades_count"] > 0
        saved = pd.read_csv(csv_path)
        assert list(saved.columns) == list(ACCEPTED_TRADES_CSV_COLUMNS)
        for column in FORBIDDEN_ACCEPTED_TRADES_COLUMNS:
            assert column not in saved.columns
        if len(saved) > 0:
            assert saved["predicted_class"].isin(["UP", "DOWN"]).any()

    def test_direct_evaluator_writes_accepted_trades_csv_by_default(
        self, data_root: Path, reports_dir: Path
    ) -> None:
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
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            reports_dir=reports_dir,
            save_reports=True,
            min_windows=1,
            min_trades=1,
        )

        assert report["filtered_paper_trades"] > 0
        assert report["accepted_trades_csv_path"] is not None
        assert report["accepted_trades_count"] > 0
        assert Path(report["accepted_trades_csv_path"]).is_file()

        loaded = json.loads(Path(report["json_report_path"]).read_text(encoding="utf-8"))
        assert loaded["accepted_trades_csv_path"] == report["accepted_trades_csv_path"]
        assert loaded["accepted_trades_count"] == report["accepted_trades_count"]
        assert loaded["paper_only"] is True
        assert loaded["execution_enabled"] is False
        assert loaded["real_execution_ready"] is False

    def test_no_accepted_trades_sets_null_path_and_zero_count(
        self, data_root: Path, reports_dir: Path
    ) -> None:
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
            min_confidence=0.0,
            min_edge_bps=500.0,
            horizon_return_scale_bps=10.0,
            data_root=data_root,
            reports_dir=reports_dir,
            save_reports=True,
            min_windows=1,
            min_trades=1,
        )

        assert report["filtered_paper_trades"] == 0
        assert report["accepted_trades_csv_path"] is None
        assert report["accepted_trades_count"] == 0

        loaded = json.loads(Path(report["json_report_path"]).read_text(encoding="utf-8"))
        assert loaded["accepted_trades_csv_path"] is None
        assert loaded["accepted_trades_count"] == 0

    def test_finalize_accepted_trades_csv_writes_required_columns(
        self, reports_dir: Path
    ) -> None:
        report = {
            "generated_at": "2024-06-10T00:00:00+00:00",
            "coins": ["BTC"],
            "filtered_paper_trades": 1,
        }
        records = [_synthetic_accepted_trade()]

        finalize_accepted_trades_csv(
            report,
            records,
            reports_dir=reports_dir,
            save_accepted_trades=True,
        )

        csv_path = Path(report["accepted_trades_csv_path"])
        assert csv_path.is_file()
        assert report["accepted_trades_count"] == 1
        saved = pd.read_csv(csv_path)
        assert list(saved.columns) == list(ACCEPTED_TRADES_CSV_COLUMNS)

    def test_finalize_accepted_trades_csv_includes_candidate_f_diagnostics_when_present(
        self, reports_dir: Path
    ) -> None:
        report = {
            "generated_at": "2024-06-10T00:00:00+00:00",
            "coins": ["BTC"],
            "filtered_paper_trades": 1,
        }
        records = [
            {
                **_synthetic_accepted_trade(),
                "realized_volatility_5m_bps": 12.5,
                "realized_volatility_15m_bps": 10.0,
                "volatility_ratio_5m_15m": 1.25,
                "spread_volatility_interaction": 25.0,
                "microprice_volatility_interaction": 3.5,
                "imbalance_volatility_interaction": 0.8,
                "low_volatility_regime": 1.0,
                "high_volatility_regime": 0.0,
                "tight_spread_regime": 1.0,
                "wide_spread_regime": 0.0,
            }
        ]

        finalize_accepted_trades_csv(
            report,
            records,
            reports_dir=reports_dir,
            save_accepted_trades=True,
        )

        saved = pd.read_csv(Path(report["accepted_trades_csv_path"]))
        expected_columns = ACCEPTED_TRADES_CSV_COLUMNS + ACCEPTED_TRADES_CANDIDATE_F_DIAGNOSTIC_COLUMNS
        assert list(saved.columns) == list(expected_columns)
        assert saved.loc[0, "realized_volatility_5m_bps"] == pytest.approx(12.5)
        assert saved.loc[0, "low_volatility_regime"] == pytest.approx(1.0)

    def test_resolve_accepted_trades_csv_columns_backward_compatible(self) -> None:
        base_only = [_synthetic_accepted_trade()]
        assert resolve_accepted_trades_csv_columns(base_only) == ACCEPTED_TRADES_CSV_COLUMNS

        with_diagnostics = [{**_synthetic_accepted_trade(), "low_volatility_regime": 1.0}]
        assert resolve_accepted_trades_csv_columns(with_diagnostics) == (
            ACCEPTED_TRADES_CSV_COLUMNS + ("low_volatility_regime",)
        )

    def test_write_accepted_trades_csv_exports_candidate_f_diagnostics(
        self, reports_dir: Path
    ) -> None:
        report = {
            "generated_at": "2024-06-10T00:00:00+00:00",
            "coins": ["BTC"],
        }
        records = [
            {
                **_synthetic_accepted_trade(),
                **{
                    column: float(index + 1)
                    for index, column in enumerate(ACCEPTED_TRADES_CANDIDATE_F_DIAGNOSTIC_COLUMNS)
                },
            }
        ]

        csv_path = write_accepted_trades_csv(
            records,
            reports_dir=reports_dir,
            report=report,
        )
        assert csv_path is not None
        saved = pd.read_csv(csv_path)
        assert list(saved.columns) == list(
            ACCEPTED_TRADES_CSV_COLUMNS + ACCEPTED_TRADES_CANDIDATE_F_DIAGNOSTIC_COLUMNS
        )
        assert saved.loc[0, "realized_volatility_5m_bps"] == pytest.approx(1.0)
        assert saved.loc[0, "wide_spread_regime"] == pytest.approx(10.0)

    def test_diagnostics_detail_basic_strips_trade_records(self) -> None:
        report = {
            "generated_at": "2024-06-10T00:00:00+00:00",
            "accepted_trade_records": [{"coin": "BTC", "net_return_bps": 1.0}],
            "per_coin_details": [
                {
                    "accepted_trade_records": [{"coin": "BTC"}],
                    "windows": [{"accepted_trade_records": [{"coin": "BTC"}]}],
                }
            ],
        }
        trimmed = apply_diagnostics_detail(
            report,
            diagnostics_detail="basic",
            accepted_trades_limit=100,
        )
        assert trimmed["diagnostics_detail"] == "basic"
        assert trimmed["accepted_trade_records_count"] == 1
        assert "accepted_trade_records" not in trimmed
        assert "accepted_trade_records" not in trimmed["per_coin_details"][0]

    def test_diagnostics_detail_full_keeps_limited_records(self) -> None:
        records = [{"coin": "BTC", "net_return_bps": float(index)} for index in range(5)]
        report = {"generated_at": "2024-06-10T00:00:00+00:00", "accepted_trade_records": records}
        full = apply_diagnostics_detail(
            report,
            diagnostics_detail="full",
            accepted_trades_limit=3,
        )
        assert full["diagnostics_detail"] == "full"
        assert len(full["accepted_trade_records"]) == 3

    def test_cli_parses_diagnostics_options(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--diagnostics-detail",
                "full",
                "--save-accepted-trades",
                "--accepted-trades-limit",
                "250",
            ]
        )
        assert args.diagnostics_detail == "full"
        assert args.save_accepted_trades is True
        assert args.accepted_trades_limit == 250

    def test_cli_diagnostics_defaults(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.diagnostics_detail == DEFAULT_DIAGNOSTICS_DETAIL
        assert args.save_accepted_trades is True
        assert args.accepted_trades_limit == DEFAULT_ACCEPTED_TRADES_LIMIT

    def test_trade_diagnostics_safety_flags_preserved(
        self, data_root: Path, reports_dir: Path
    ) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=60))

        report = run_walk_forward_evaluation(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=1,
            train_bars=20,
            test_bars=10,
            step_bars=10,
            data_root=data_root,
            reports_dir=reports_dir,
            save_reports=True,
            min_windows=1,
            min_trades=1,
        )

        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False
        loaded = json.loads(Path(report["json_report_path"]).read_text(encoding="utf-8"))
        assert loaded["paper_only"] is True
        assert loaded["execution_enabled"] is False
        assert "class_trade_diagnostics" in loaded


def _synthetic_accepted_trade(
    *,
    coin: str = "BTC",
    predicted_class: str = "DOWN",
    net_return_bps: float = 3.0,
    cost_bps: float = 2.0,
    confidence: float = 0.9,
    timestamp_iso: str = "2024-06-10T06:00:00Z",
    timestamp_ms: int | None = None,
    window_index: int | None = 0,
) -> dict:
    return {
        "coin": coin,
        "window_index": window_index,
        "row_index": 0,
        "timestamp_iso": timestamp_iso,
        "timestamp_ms": timestamp_ms,
        "predicted_class": predicted_class,
        "actual_label": predicted_class,
        "confidence": confidence,
        "future_return_bps": abs(net_return_bps),
        "gross_return_bps": abs(net_return_bps),
        "cost_bps": cost_bps,
        "net_return_bps": net_return_bps,
        "spread_bps": 2.0,
    }


class TestRegimeFilters:
    def test_no_regime_filters_preserves_existing_behavior(self, data_root: Path) -> None:
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
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            save_reports=False,
            min_windows=1,
            min_trades=1,
        )

        assert report["regime_filters_enabled"] is False
        assert report["exploratory_overfit_warning"] is False
        assert report["trades_before_regime_filters"] == report["filtered_paper_trades"]
        assert report["trades_after_regime_filters"] == report["filtered_paper_trades"]
        assert report["trades_removed_by_regime_filters"] == 0
        assert report["cumulative_net_return_before_regime_filters"] == pytest.approx(
            report["cumulative_net_return_bps"]
        )
        assert report["cumulative_net_return_after_regime_filters"] == pytest.approx(
            report["cumulative_net_return_bps"]
        )

    def test_exclude_hours_removes_expected_trades(self) -> None:
        records = [
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T05:00:00Z", net_return_bps=5.0),
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T06:00:00Z", net_return_bps=4.0),
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T07:00:00Z", net_return_bps=3.0),
        ]
        filtered = filter_accepted_trades_by_regime(records, exclude_hours=[5])

        assert len(filtered) == 2
        assert all(_trade_record_utc_hour(record) != 5 for record in filtered)

    def test_include_hours_keeps_expected_trades(self) -> None:
        records = [
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T05:00:00Z", net_return_bps=5.0),
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T06:00:00Z", net_return_bps=4.0),
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T07:00:00Z", net_return_bps=3.0),
        ]
        filtered = filter_accepted_trades_by_regime(records, include_hours=[6, 7])

        assert len(filtered) == 2
        assert {_trade_record_utc_hour(record) for record in filtered} == {6, 7}

    def test_include_then_exclude_order_works(self) -> None:
        records = [
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T05:00:00Z", net_return_bps=5.0),
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T06:00:00Z", net_return_bps=4.0),
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T07:00:00Z", net_return_bps=3.0),
        ]
        filtered = filter_accepted_trades_by_regime(
            records,
            include_hours=[5, 6, 7],
            exclude_hours=[5],
        )

        assert len(filtered) == 2
        assert {_trade_record_utc_hour(record) for record in filtered} == {6, 7}

    def test_max_cost_bps_removes_high_cost_trades(self) -> None:
        records = [
            _synthetic_accepted_trade(cost_bps=3.0, net_return_bps=5.0),
            _synthetic_accepted_trade(cost_bps=7.0, net_return_bps=4.0),
            _synthetic_accepted_trade(cost_bps=5.0, net_return_bps=3.0),
        ]
        filtered = filter_accepted_trades_by_regime(records, max_cost_bps=5.0)

        assert len(filtered) == 2
        assert all(float(record["cost_bps"]) <= 5.0 for record in filtered)

    def test_min_cost_bps_removes_low_cost_trades(self) -> None:
        records = [
            _synthetic_accepted_trade(cost_bps=3.0, net_return_bps=5.0),
            _synthetic_accepted_trade(cost_bps=7.0, net_return_bps=4.0),
            _synthetic_accepted_trade(cost_bps=5.0, net_return_bps=3.0),
        ]
        filtered = filter_accepted_trades_by_regime(records, min_cost_bps=5.0)

        assert len(filtered) == 2
        assert all(float(record["cost_bps"]) >= 5.0 for record in filtered)

    def test_missing_timestamp_iso_falls_back_to_timestamp_ms(self) -> None:
        ms_at_hour_5 = int(datetime.fromisoformat("2024-06-10T05:30:00+00:00").timestamp() * 1000)
        record = _synthetic_accepted_trade(
            timestamp_iso="",
            timestamp_ms=ms_at_hour_5,
            net_return_bps=2.0,
        )
        assert _trade_record_utc_hour(record) == 5

        filtered = filter_accepted_trades_by_regime([record], exclude_hours=[5])
        assert filtered == []

    def test_accepted_trades_csv_contains_post_filter_trades(
        self, data_root: Path, reports_dir: Path
    ) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        rows = _make_resampled_rows(count=80, mid_step=2.0)
        for index, row in enumerate(rows):
            row["timestamp_iso"] = f"2024-06-10T{index % 24:02d}:00:{index % 60:02d}Z"
        _write_resampled_csv(path, rows)

        report = run_walk_forward_evaluation(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=2,
            threshold_bps=0.1,
            train_bars=25,
            test_bars=10,
            step_bars=10,
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            reports_dir=reports_dir,
            save_reports=False,
            min_windows=1,
            min_trades=1,
            exclude_hours=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        )

        assert report["trades_after_regime_filters"] == 0
        assert report["accepted_trades_csv_path"] is None
        assert report["accepted_trades_count"] == 0

        report_keep = run_walk_forward_evaluation(
            coins=["BTC"],
            interval="15s",
            source="local",
            horizon=2,
            threshold_bps=0.1,
            train_bars=25,
            test_bars=10,
            step_bars=10,
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            reports_dir=reports_dir,
            save_reports=False,
            min_windows=1,
            min_trades=1,
            include_hours=[6],
        )
        if report_keep["trades_after_regime_filters"] > 0:
            csv_path = Path(report_keep["accepted_trades_csv_path"])
            saved = pd.read_csv(csv_path)
            assert len(saved) == report_keep["trades_after_regime_filters"]
            assert len(saved) <= report_keep["trades_before_regime_filters"]

    def test_json_includes_before_after_metrics(
        self, data_root: Path, reports_dir: Path
    ) -> None:
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
            min_confidence=0.0,
            horizon_return_scale_bps=1000.0,
            data_root=data_root,
            reports_dir=reports_dir,
            save_reports=True,
            min_windows=1,
            min_trades=1,
            exclude_hours=[5],
        )
        loaded = json.loads(Path(report["json_report_path"]).read_text(encoding="utf-8"))

        for field in (
            "regime_filters_enabled",
            "exploratory_overfit_warning",
            "exclude_hours",
            "include_hours",
            "max_cost_bps",
            "min_cost_bps",
            "trades_before_regime_filters",
            "trades_after_regime_filters",
            "trades_removed_by_regime_filters",
            "cumulative_net_return_before_regime_filters",
            "cumulative_net_return_after_regime_filters",
            "average_net_return_before_regime_filters",
            "average_net_return_after_regime_filters",
            "win_rate_before_regime_filters",
            "win_rate_after_regime_filters",
        ):
            assert field in loaded

    def test_overfit_warning_true_when_any_regime_filter_active(self) -> None:
        report = apply_regime_filter_metrics(
            {
                "coins": ["BTC"],
                "accepted_trade_records": [
                    _synthetic_accepted_trade(timestamp_iso="2024-06-10T05:00:00Z"),
                    _synthetic_accepted_trade(timestamp_iso="2024-06-10T06:00:00Z"),
                ],
            },
            exclude_hours=[5],
        )
        assert report["regime_filters_enabled"] is True
        assert report["exploratory_overfit_warning"] is True

    def test_no_overfit_warning_when_inactive(self) -> None:
        report = apply_regime_filter_metrics(
            {
                "coins": ["BTC"],
                "accepted_trade_records": [
                    _synthetic_accepted_trade(timestamp_iso="2024-06-10T05:00:00Z"),
                ],
            },
        )
        assert report["regime_filters_enabled"] is False
        assert report["exploratory_overfit_warning"] is False

    def test_markdown_includes_overfit_warning(self, reports_dir: Path) -> None:
        report = apply_regime_filter_metrics(
            {
                "generated_at": "2024-06-10T00:00:00+00:00",
                "coins": ["BTC"],
                "accepted_trade_records": [
                    _synthetic_accepted_trade(timestamp_iso="2024-06-10T05:00:00Z"),
                    _synthetic_accepted_trade(timestamp_iso="2024-06-10T06:00:00Z"),
                ],
            },
            exclude_hours=[5],
        )
        md_path = write_walk_forward_markdown(report, reports_dir=reports_dir)
        md_text = md_path.read_text(encoding="utf-8")
        assert "## Regime Filters" in md_text
        assert REGIME_OVERFIT_WARNING in md_text

    def test_cli_parses_regime_filter_options(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--exclude-hours",
                "3,5",
                "--include-hours",
                "6,7",
                "--max-cost-bps",
                "6.0",
                "--min-cost-bps",
                "2.0",
            ]
        )
        assert args.exclude_hours == "3,5"
        assert args.include_hours == "6,7"
        assert args.max_cost_bps == 6.0
        assert args.min_cost_bps == 2.0

    def test_parse_hours_list(self) -> None:
        assert parse_hours_list("3,5") == [3, 5]
        assert parse_hours_list("") is None
        assert parse_hours_list(None) is None
        with pytest.raises(ValueError):
            parse_hours_list("25")

    def test_regime_filters_enabled(self) -> None:
        assert regime_filters_enabled(exclude_hours=[5]) is True
        assert regime_filters_enabled(max_cost_bps=5.0) is True
        assert regime_filters_enabled(min_cost_bps=2.0) is True
        assert regime_filters_enabled() is False

    def test_regime_filter_recomputes_top_level_metrics(self) -> None:
        records = [
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T05:00:00Z", net_return_bps=-10.0),
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T06:00:00Z", net_return_bps=4.0),
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T07:00:00Z", net_return_bps=3.0),
        ]
        report = apply_regime_filter_metrics(
            {
                "coins": ["BTC"],
                "completed_windows": 5,
                "min_windows": 1,
                "min_trades": 1,
                "accuracy_edge_vs_baseline": 0.02,
                "accepted_trade_records": records,
                "filtered_paper_trades": 3,
                "cumulative_net_return_bps": -3.0,
                "average_net_return_bps": -1.0,
                "win_rate_after_costs": 2 / 3,
                "positive_after_costs": False,
                "warnings": ["negative_after_costs"],
                "readiness": {
                    "beats_baseline_accuracy": True,
                    "positive_after_costs": False,
                    "enough_windows": True,
                    "enough_trades": True,
                    "ready_for_more_research": True,
                    "real_execution_ready": False,
                },
            },
            exclude_hours=[5],
        )

        assert report["trades_before_regime_filters"] == 3
        assert report["trades_after_regime_filters"] == 2
        assert report["filtered_paper_trades"] == 2
        assert report["cumulative_net_return_before_regime_filters"] == pytest.approx(-3.0)
        assert report["cumulative_net_return_after_regime_filters"] == pytest.approx(7.0)
        assert report["cumulative_net_return_bps"] == pytest.approx(7.0)
        assert report["average_net_return_bps"] == pytest.approx(3.5)
        assert report["positive_after_costs"] is True
        assert report["readiness"]["positive_after_costs"] is True
        assert "negative_after_costs" not in report["warnings"]

    def test_cumulative_net_return_bps_equals_post_filter_sum(self) -> None:
        records = [
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T05:00:00Z", net_return_bps=1.0),
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T06:00:00Z", net_return_bps=2.0),
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T07:00:00Z", net_return_bps=3.0),
        ]
        report = apply_regime_filter_metrics(
            {"coins": ["BTC"], "accepted_trade_records": records},
            exclude_hours=[5],
        )
        post_filter_sum = sum(
            float(record["net_return_bps"]) for record in report["accepted_trade_records"]
        )

        assert report["cumulative_net_return_bps"] == pytest.approx(post_filter_sum)
        assert report["cumulative_net_return_after_regime_filters"] == pytest.approx(post_filter_sum)
        assert report["cumulative_net_return_before_regime_filters"] == pytest.approx(6.0)

    def test_no_trades_after_regime_filters_sets_zero_metrics_and_warning(self) -> None:
        records = [
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T05:00:00Z", net_return_bps=-8.0),
        ]
        report = apply_regime_filter_metrics(
            {
                "coins": ["BTC"],
                "accepted_trade_records": records,
                "filtered_paper_trades": 1,
                "cumulative_net_return_bps": -8.0,
                "positive_after_costs": False,
                "warnings": ["negative_after_costs"],
            },
            exclude_hours=[5],
        )

        assert report["trades_after_regime_filters"] == 0
        assert report["filtered_paper_trades"] == 0
        assert report["cumulative_net_return_bps"] == 0.0
        assert report["average_net_return_bps"] == 0.0
        assert report["win_rate_after_costs"] == 0.0
        assert report["positive_after_costs"] is False
        assert "no_trades_after_regime_filters" in report["warnings"]
        assert "negative_after_costs" not in report["warnings"]

    def test_markdown_shows_before_and_after_regime_sections(self, reports_dir: Path) -> None:
        report = apply_regime_filter_metrics(
            {
                "generated_at": "2024-06-10T00:00:00+00:00",
                "coins": ["BTC"],
                "accepted_trade_records": [
                    _synthetic_accepted_trade(timestamp_iso="2024-06-10T05:00:00Z"),
                    _synthetic_accepted_trade(timestamp_iso="2024-06-10T06:00:00Z"),
                ],
            },
            exclude_hours=[5],
        )
        md_path = write_walk_forward_markdown(report, reports_dir=reports_dir)
        md_text = md_path.read_text(encoding="utf-8")
        assert "### Before regime filters" in md_text
        assert "### After regime filters" in md_text


class TestStabilityValidation:
    def test_bucket_metrics_are_correct(self) -> None:
        records = [
            _synthetic_accepted_trade(net_return_bps=4.0, cost_bps=1.0, confidence=0.8),
            _synthetic_accepted_trade(net_return_bps=-2.0, cost_bps=3.0, confidence=0.6),
        ]
        metrics = aggregate_stability_bucket_metrics(records)

        assert metrics["trades"] == 2
        assert metrics["cumulative_net_return_bps"] == pytest.approx(2.0)
        assert metrics["average_net_return_bps"] == pytest.approx(1.0)
        assert metrics["win_rate_after_costs"] == pytest.approx(0.5)
        assert metrics["average_cost_bps"] == pytest.approx(2.0)
        assert metrics["average_confidence"] == pytest.approx(0.7)
        assert metrics["positive_after_costs"] is True

    def test_stability_by_hour_date_and_window(self) -> None:
        records = [
            _synthetic_accepted_trade(
                timestamp_iso="2024-06-10T06:00:00Z",
                window_index=0,
                net_return_bps=5.0,
            ),
            _synthetic_accepted_trade(
                timestamp_iso="2024-06-10T06:15:00Z",
                window_index=0,
                net_return_bps=4.0,
            ),
            _synthetic_accepted_trade(
                timestamp_iso="2024-06-10T07:00:00Z",
                window_index=1,
                net_return_bps=-1.0,
            ),
        ]
        diagnostics = compute_stability_diagnostics(records, coins=["BTC"])

        assert diagnostics["by_hour"]["06"]["trades"] == 2
        assert diagnostics["by_hour"]["07"]["trades"] == 1
        assert diagnostics["by_date"]["2024-06-10"]["trades"] == 3
        assert diagnostics["by_window_index"]["0"]["trades"] == 2
        assert diagnostics["by_window_index"]["1"]["trades"] == 1
        assert diagnostics["by_coin"]["BTC"]["trades"] == 3
        assert diagnostics["by_predicted_class"]["DOWN"]["trades"] == 3
        assert diagnostics["by_coin_predicted_class"]["BTC:DOWN"]["trades"] == 3

    def test_concentration_warning_triggers(self) -> None:
        records = [
            _synthetic_accepted_trade(
                timestamp_iso="2024-06-10T06:00:00Z",
                window_index=0,
                net_return_bps=8.0,
            ),
            _synthetic_accepted_trade(
                timestamp_iso="2024-06-10T07:00:00Z",
                window_index=1,
                net_return_bps=1.0,
            ),
        ]
        diagnostics = compute_stability_diagnostics(records, coins=["BTC"])
        summary = compute_stability_summary(
            diagnostics,
            min_trades_per_bucket=3,
        )

        assert summary["concentration_warning"] is True
        assert any(
            "stability_concentration" in warning
            for warning in build_stability_warnings(summary, min_trades_per_bucket=3)
        )

    def test_sparse_warning_triggers(self) -> None:
        records = [
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T06:00:00Z", net_return_bps=2.0),
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T07:00:00Z", net_return_bps=1.0),
        ]
        diagnostics = compute_stability_diagnostics(records, coins=["BTC"])
        summary = compute_stability_summary(
            diagnostics,
            min_trades_per_bucket=3,
        )

        assert summary["sparse_stability_warning"] is True
        assert summary["buckets_with_too_few_trades"] > 0
        assert any(
            "stability_sparse_buckets" in warning
            for warning in build_stability_warnings(summary, min_trades_per_bucket=3)
        )

    def test_handles_missing_timestamp_iso_safely(self) -> None:
        records = [
            _synthetic_accepted_trade(timestamp_iso="", net_return_bps=2.0),
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T08:00:00Z", net_return_bps=1.0),
        ]
        diagnostics = compute_stability_diagnostics(records, coins=["BTC"])

        assert diagnostics["by_hour"] == {"08": diagnostics["by_hour"]["08"]}
        assert diagnostics["by_date"] == {"2024-06-10": diagnostics["by_date"]["2024-06-10"]}
        assert diagnostics["by_coin"]["BTC"]["trades"] == 2

    def test_handles_zero_trades_safely(self) -> None:
        diagnostics = compute_stability_diagnostics([], coins=["BTC"])
        summary = compute_stability_summary(diagnostics, min_trades_per_bucket=3)

        assert diagnostics["by_coin"]["BTC"]["trades"] == 0
        assert summary["profitable_bucket_fraction"] == 0.0
        assert summary["concentration_warning"] is False
        assert summary["sparse_stability_warning"] is False
        assert build_stability_warnings(summary, min_trades_per_bucket=3) == []

    def test_stability_reduces_ready_for_more_research(
        self, data_root: Path, reports_dir: Path
    ) -> None:
        readiness = compute_readiness_flags(
            accuracy_edge_vs_baseline=0.1,
            cumulative_net_return_bps=10.0,
            completed_windows=10,
            total_paper_trades=200,
            min_windows=5,
            min_trades=100,
        )
        adjusted = apply_stability_readiness(
            readiness,
            {
                "concentration_warning": True,
                "sparse_stability_warning": False,
            },
        )

        assert readiness["ready_for_more_research"] is True
        assert adjusted["ready_for_more_research"] is False
        assert adjusted["real_execution_ready"] is False

    def test_json_report_includes_stability_fields_when_enabled(
        self, data_root: Path, reports_dir: Path
    ) -> None:
        records = [
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T06:00:00Z", net_return_bps=3.0),
            _synthetic_accepted_trade(timestamp_iso="2024-06-10T07:00:00Z", net_return_bps=2.0),
        ]
        report = build_walk_forward_report(
            coins=["BTC"],
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
            trade_filter=_permissive_filter(),
            per_coin_results=[
                {
                    "coin": "BTC",
                    "status": "completed",
                    "completed_windows": 1,
                    "total_windows": 1,
                    "skipped_windows": 0,
                    "model_accuracy_mean": 0.6,
                    "baseline_accuracy_mean": 0.5,
                    "accuracy_edge_vs_baseline": 0.1,
                    "total_paper_signals": 2,
                    "raw_predictions": 2,
                    "filtered_paper_trades": 2,
                    "total_paper_trades": 2,
                    "accepted_trade_net_returns_bps": [3.0, 2.0],
                    "trade_returns_bps": [3.0, 2.0],
                    "cumulative_net_return_bps": 5.0,
                    "average_net_return_bps": 2.5,
                    "average_gross_return_bps": 2.5,
                    "average_cost_bps": 0.0,
                    "average_required_edge_bps": 0.0,
                    "blocked_by_confidence": 0,
                    "blocked_by_cost": 0,
                    "blocked_by_spread": 0,
                    "blocked_by_volatility": 0,
                    "blocked_by_trade_class": 0,
                    "accepted_trade_records": records,
                    "warnings": [],
                    "windows": [],
                }
            ],
            data_root=data_root,
            min_windows=1,
            min_trades=1,
            stability_report_enabled=True,
            stability_min_trades_per_bucket=3,
        )
        json_path = write_walk_forward_json(report, reports_dir=reports_dir)
        md_path = write_walk_forward_markdown(report, reports_dir=reports_dir)
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        md_text = md_path.read_text(encoding="utf-8")

        assert report["stability_report_enabled"] is True
        assert report["stability_min_trades_per_bucket"] == 3
        assert "stability_diagnostics" in report
        assert "stability_summary" in report
        assert "stability_warnings" in report
        assert loaded["stability_report_enabled"] is True
        assert "stability_diagnostics" in loaded
        assert "stability_summary" in loaded
        assert "## Stability Validation" in md_text
        assert "concentration_warning" in md_text

    def test_stability_fields_absent_when_disabled(
        self, data_root: Path, reports_dir: Path
    ) -> None:
        report = build_walk_forward_report(
            coins=["BTC"],
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
            trade_filter=_permissive_filter(),
            per_coin_results=[],
            data_root=data_root,
        )
        json_path = write_walk_forward_json(report, reports_dir=reports_dir)
        md_path = write_walk_forward_markdown(report, reports_dir=reports_dir)
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        md_text = md_path.read_text(encoding="utf-8")

        assert report["stability_report_enabled"] is False
        assert "stability_diagnostics" not in report
        assert "stability_summary" not in report
        assert "stability_warnings" not in report
        assert loaded["stability_report_enabled"] is False
        assert "stability_diagnostics" not in loaded
        assert "## Stability Validation" not in md_text

    def test_stability_safety_flags_preserved(
        self, data_root: Path, reports_dir: Path
    ) -> None:
        report = build_walk_forward_report(
            coins=["BTC"],
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
            trade_filter=_permissive_filter(),
            per_coin_results=[
                {
                    "coin": "BTC",
                    "status": "completed",
                    "completed_windows": 1,
                    "total_windows": 1,
                    "skipped_windows": 0,
                    "model_accuracy_mean": 0.6,
                    "baseline_accuracy_mean": 0.5,
                    "accuracy_edge_vs_baseline": 0.1,
                    "total_paper_signals": 1,
                    "raw_predictions": 1,
                    "filtered_paper_trades": 1,
                    "total_paper_trades": 1,
                    "accepted_trade_net_returns_bps": [2.0],
                    "trade_returns_bps": [2.0],
                    "cumulative_net_return_bps": 2.0,
                    "average_net_return_bps": 2.0,
                    "average_gross_return_bps": 2.0,
                    "average_cost_bps": 0.0,
                    "average_required_edge_bps": 0.0,
                    "blocked_by_confidence": 0,
                    "blocked_by_cost": 0,
                    "blocked_by_spread": 0,
                    "blocked_by_volatility": 0,
                    "blocked_by_trade_class": 0,
                    "accepted_trade_records": [
                        _synthetic_accepted_trade(net_return_bps=2.0),
                    ],
                    "warnings": [],
                    "windows": [],
                }
            ],
            data_root=data_root,
            stability_report_enabled=True,
        )

        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False
        assert report["readiness"]["real_execution_ready"] is False

    def test_cli_parses_stability_options(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--stability-report",
                "--stability-min-trades-per-bucket",
                "5",
            ]
        )
        assert args.stability_report is True
        assert args.stability_min_trades_per_bucket == 5

    def test_cli_stability_defaults(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.stability_report is False
        assert args.stability_min_trades_per_bucket == DEFAULT_STABILITY_MIN_TRADES_PER_BUCKET


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
