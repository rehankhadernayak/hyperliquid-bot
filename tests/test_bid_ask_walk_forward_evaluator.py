"""Focused offline tests for walk-forward paper evaluation."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from bid_ask_resampler import RESAMPLED_COLUMNS, resampled_csv_path
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY
from bid_ask_train import FEATURE_COLUMNS, create_features, create_labels
from bid_ask_walk_forward_evaluator import (
    ACCEPTED_TRADES_CSV_COLUMNS,
    DEFAULT_ACCEPTED_TRADES_LIMIT,
    DEFAULT_DIAGNOSTICS_DETAIL,
    DEFAULT_FEATURE_SET,
    DEFAULT_LABEL_COST_MULTIPLIER,
    DEFAULT_LABEL_MIN_EDGE_BPS,
    DEFAULT_LABEL_MODE,
    DEFAULT_MIN_TRADES,
    DEFAULT_MIN_WINDOWS,
    DEFAULT_REPORTS_DIR,
    MICROSTRUCTURE_EXTRA_COLUMNS,
    SAFETY_NOTE,
    TradeFilterConfig,
    _build_arg_parser,
    aggregate_accepted_trade_metrics,
    aggregate_class_trade_diagnostics,
    aggregate_coin_class_trade_diagnostics,
    apply_diagnostics_detail,
    build_feature_metadata,
    build_trade_filter,
    build_walk_forward_report,
    compute_label_distribution,
    compute_readiness_flags,
    cost_aware_label_warnings,
    create_cost_aware_labels,
    create_walk_forward_features,
    create_walk_forward_labels,
    evaluate_coin_walk_forward,
    evaluate_trade_filter,
    generate_walk_forward_windows,
    get_walk_forward_feature_columns,
    longest_losing_streak,
    main,
    max_drawdown_bps,
    parse_feature_set,
    parse_label_mode,
    parse_trade_classes,
    resolve_coins,
    run_walk_forward_evaluation,
    simulate_paper_signal_returns,
    summarize_label_required_edge_bps,
    walk_forward_feature_matrix,
    write_walk_forward_json,
    write_walk_forward_markdown,
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
        with pytest.raises(ValueError):
            parse_feature_set("invalid")

    def test_get_walk_forward_feature_columns(self) -> None:
        assert get_walk_forward_feature_columns("basic") == FEATURE_COLUMNS
        micro_columns = get_walk_forward_feature_columns("microstructure")
        assert micro_columns[: len(FEATURE_COLUMNS)] == FEATURE_COLUMNS
        assert len(micro_columns) == len(FEATURE_COLUMNS) + len(MICROSTRUCTURE_EXTRA_COLUMNS)

    def test_walk_forward_feature_matrix_uses_selected_columns(self) -> None:
        rows = _make_resampled_rows(count=10)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))
        featured, meta = create_walk_forward_features(raw_df, feature_set="microstructure")
        matrix = walk_forward_feature_matrix(featured, tuple(meta["feature_columns"]))
        assert list(matrix.columns) == meta["feature_columns"]


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
        with pytest.raises(ValueError):
            parse_label_mode("invalid")


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
            save_accepted_trades=True,
            min_windows=1,
            min_trades=1,
        )

        csv_path = Path(report["accepted_trades_csv_path"])
        assert csv_path.is_file()
        assert csv_path.parent == reports_dir
        saved = pd.read_csv(csv_path)
        assert list(saved.columns) == list(ACCEPTED_TRADES_CSV_COLUMNS)
        for column in FORBIDDEN_ACCEPTED_TRADES_COLUMNS:
            assert column not in saved.columns
        if len(saved) > 0:
            assert saved["predicted_class"].isin(["UP", "DOWN"]).any()

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
        assert args.save_accepted_trades is False
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
            save_accepted_trades=True,
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
