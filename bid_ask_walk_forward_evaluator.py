"""Walk-forward paper evaluation of bid/ask models with realistic cost simulation.

Research-only and paper-only. No orders, wallets, or network calls.
"""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bid_ask_model_io import build_artifact_payload, save_model_artifact
from bid_ask_resampler import normalize_source, resolve_bid_ask_data_dir, resampled_csv_path
from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT
from bid_ask_train import (
    FEATURE_COLUMNS,
    LABELS,
    MID_PRICE_COLUMN,
    MajorityClassBaseline,
    RuleBaseline,
    _SKLEARN_AVAILABLE,
    _accuracy,
    create_features,
    create_labels,
    load_training_data,
    train_baseline_model,
)

SAFETY_NOTE = (
    "Walk-forward evaluation is research-only and paper-only. "
    "Simulated returns do not prove profitability. No real orders are placed."
)

DEFAULT_DATA_ROOT = Path("data/bid_ask_csv")
DEFAULT_REPORTS_DIR = Path("reports/bid_ask_walk_forward")
DEFAULT_MODELS_DIR = Path("models/bid_ask_walk_forward")

SUPPORTED_INTERVALS = frozenset({"15s", "1m"})
SUPPORTED_SOURCES = frozenset({"local", "historical", "combined"})

DEFAULT_TRAIN_BARS = 500
DEFAULT_TEST_BARS = 100
DEFAULT_STEP_BARS = 100
DEFAULT_FEE_BPS = 2.0
DEFAULT_SLIPPAGE_BPS = 1.0
DEFAULT_SPREAD_BPS = 2.0
DEFAULT_MIN_CONFIDENCE = 0.4
DEFAULT_THRESHOLD_BPS = 1.0
DEFAULT_HORIZON = 5
DEFAULT_MIN_EDGE_BPS = 0.0
DEFAULT_COST_MULTIPLIER = 1.0
DEFAULT_HORIZON_RETURN_SCALE_BPS = 10.0
DEFAULT_LABEL_MODE = "direction"
DEFAULT_LABEL_COST_MULTIPLIER = 1.0
DEFAULT_LABEL_MIN_EDGE_BPS = 0.0
DEFAULT_TRADE_CLASSES = frozenset({"UP", "DOWN"})
DEFAULT_DIAGNOSTICS_DETAIL = "basic"
DEFAULT_ACCEPTED_TRADES_LIMIT = 1000
DEFAULT_STABILITY_MIN_TRADES_PER_BUCKET = 3
STABILITY_CONCENTRATION_FRACTION = 0.70

SUPPORTED_LABEL_MODES = frozenset({"direction", "cost_aware"})
SUPPORTED_FEATURE_SETS = frozenset({"basic", "microstructure"})
DEFAULT_FEATURE_SET = "basic"

MICROSTRUCTURE_EXTRA_COLUMNS = (
    "return_1",
    "return_3",
    "return_5",
    "return_10",
    "return_20",
    "rolling_return_mean_5",
    "rolling_return_mean_10",
    "rolling_return_std_5",
    "rolling_return_std_10",
    "rolling_return_zscore_10",
    "momentum_ratio_5_20",
    "spread_bps",
    "rolling_spread_mean_5",
    "rolling_spread_mean_20",
    "rolling_spread_std_20",
    "spread_zscore_20",
    "spread_percentile_50",
    "is_tight_spread_regime",
    "is_wide_spread_regime",
    "bbo_imbalance",
    "imbalance_change_1",
    "imbalance_mean_5",
    "imbalance_mean_20",
    "imbalance_std_20",
    "imbalance_zscore_20",
    "imbalance_persistence_5",
    "imbalance_persistence_10",
    "volatility_5",
    "volatility_20",
    "volatility_ratio_5_20",
    "is_low_volatility_regime",
    "is_high_volatility_regime",
    "imbalance_x_return_5",
    "spread_x_volatility_20",
    "tight_spread_x_imbalance",
    "wide_spread_x_volatility",
)

MIN_MICROSTRUCTURE_FEATURE_COUNT = len(FEATURE_COLUMNS) + 1

ACCEPTED_TRADES_CSV_COLUMNS = (
    "coin",
    "window_index",
    "row_index",
    "timestamp_iso",
    "predicted_class",
    "actual_label",
    "confidence",
    "future_return_bps",
    "gross_return_bps",
    "cost_bps",
    "net_return_bps",
    "spread_bps",
)

MIN_ROWS_WARNING = 50
DEFAULT_MIN_WINDOWS = 5
DEFAULT_MIN_TRADES = 100
MIN_CLASS_FRACTION = 0.10
MAX_CLASS_FRACTION = 0.70
MISSING_DATA_FRACTION_WARNING = 0.05


def resolve_coins(*, coin: str | None, coins: list[str] | None) -> list[str]:
    """Merge --coin and --coins into a deduplicated uppercase list."""
    resolved: list[str] = []
    if coins:
        resolved.extend(str(value).upper() for value in coins)
    if coin:
        resolved.append(str(coin).upper())
    if not resolved:
        resolved.append("BTC")
    seen: set[str] = set()
    ordered: list[str] = []
    for value in resolved:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def generate_walk_forward_windows(
    n_rows: int,
    *,
    train_bars: int,
    test_bars: int,
    step_bars: int,
) -> list[tuple[int, int, int, int]]:
    """Return chronological (train_start, train_end, test_start, test_end) index ranges."""
    if train_bars < 1 or test_bars < 1 or step_bars < 1:
        raise ValueError("train_bars, test_bars, and step_bars must be >= 1")

    windows: list[tuple[int, int, int, int]] = []
    start = 0
    while True:
        train_start = start
        train_end = start + train_bars
        test_start = train_end
        test_end = test_start + test_bars
        if test_end > n_rows:
            break
        windows.append((train_start, train_end, test_start, test_end))
        start += step_bars
    return windows


@dataclass(frozen=True)
class TradeFilterConfig:
    """Cost-aware paper-trade filter settings."""

    min_edge_bps: float = DEFAULT_MIN_EDGE_BPS
    max_spread_bps: float | None = None
    min_volatility_bps: float | None = None
    max_volatility_bps: float | None = None
    trade_classes: frozenset[str] = DEFAULT_TRADE_CLASSES
    cost_multiplier: float = DEFAULT_COST_MULTIPLIER
    horizon_return_scale_bps: float = DEFAULT_HORIZON_RETURN_SCALE_BPS
    min_absolute_future_label_bps: float | None = None
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    threshold_bps: float = DEFAULT_THRESHOLD_BPS


def parse_trade_classes(value: str) -> frozenset[str]:
    """Parse comma-separated trade classes, e.g. UP,DOWN."""
    classes = frozenset(part.strip().upper() for part in value.split(",") if part.strip())
    if not classes:
        return DEFAULT_TRADE_CLASSES
    return classes


def parse_label_mode(value: str) -> str:
    """Parse label mode: direction (default) or cost_aware."""
    mode = str(value).strip().lower()
    if mode not in SUPPORTED_LABEL_MODES:
        supported = ", ".join(sorted(SUPPORTED_LABEL_MODES))
        raise ValueError(f"Unsupported label_mode {value!r}; use one of: {supported}")
    return mode


def parse_feature_set(value: str) -> str:
    """Parse feature set: basic (default) or microstructure."""
    feature_set = str(value).strip().lower()
    if feature_set not in SUPPORTED_FEATURE_SETS:
        supported = ", ".join(sorted(SUPPORTED_FEATURE_SETS))
        raise ValueError(f"Unsupported feature_set {value!r}; use one of: {supported}")
    return feature_set


def resolve_mid_price_column(df: pd.DataFrame) -> str | None:
    """Return the best available mid-price column name."""
    for column in ("close_mid", "avg_mid_price", "avg_mid"):
        if column in df.columns:
            return column
    return None


def resolve_spread_column(df: pd.DataFrame) -> str | None:
    """Return the best available spread column name."""
    for column in ("avg_spread_bps", "spread_bps"):
        if column in df.columns:
            return column
    return None


def resolve_imbalance_column(df: pd.DataFrame) -> str | None:
    """Return the best available BBO imbalance column name."""
    for column in ("avg_bbo_imbalance", "bbo_imbalance"):
        if column in df.columns:
            return column
    return None


def get_walk_forward_feature_columns(feature_set: str) -> tuple[str, ...]:
    """Return ordered feature columns for the requested feature set."""
    mode = parse_feature_set(feature_set)
    if mode == "basic":
        return FEATURE_COLUMNS
    return FEATURE_COLUMNS + MICROSTRUCTURE_EXTRA_COLUMNS


def _safe_fill_series(series: pd.Series) -> pd.Series:
    """Fill NaNs from rolling calculations without using future rows."""
    return series.ffill().bfill().fillna(0.0)


def _return_bps(mid: pd.Series, lag: int) -> pd.Series:
    prior = mid.shift(lag).replace(0, np.nan)
    return (mid - mid.shift(lag)) / prior * 10_000.0


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def _imbalance_persistence(imbalance: pd.Series, window: int) -> pd.Series:
    sign = np.sign(imbalance.fillna(0.0))
    same_as_prior = (sign == sign.shift(1)).astype(float)
    return same_as_prior.rolling(window=window, min_periods=1).sum()


def _add_microstructure_features(
    df: pd.DataFrame,
    *,
    missing_sources: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add microstructure features beyond the basic set."""
    out = df.copy()
    skipped: list[str] = []
    missing = missing_sources or set()
    mid_column = resolve_mid_price_column(out)
    spread_column = None if "avg_spread_bps" in missing else resolve_spread_column(out)
    imbalance_column = None if "avg_bbo_imbalance" in missing else resolve_imbalance_column(out)

    if mid_column is None:
        skipped.extend(
            [
                "return_1",
                "return_3",
                "return_5",
                "return_10",
                "return_20",
                "rolling_return_mean_5",
                "rolling_return_mean_10",
                "rolling_return_std_5",
                "rolling_return_std_10",
                "rolling_return_zscore_10",
                "momentum_ratio_5_20",
                "volatility_5",
                "volatility_20",
                "volatility_ratio_5_20",
                "is_low_volatility_regime",
                "is_high_volatility_regime",
                "imbalance_x_return_5",
            ]
        )
    else:
        mid = out[mid_column].astype(float)
        out["return_1"] = _return_bps(mid, 1)
        out["return_3"] = _return_bps(mid, 3)
        out["return_5"] = _return_bps(mid, 5)
        out["return_10"] = _return_bps(mid, 10)
        out["return_20"] = _return_bps(mid, 20)

        return_1 = out["return_1"]
        out["rolling_return_mean_5"] = return_1.rolling(window=5, min_periods=5).mean()
        out["rolling_return_mean_10"] = return_1.rolling(window=10, min_periods=10).mean()
        out["rolling_return_std_5"] = return_1.rolling(window=5, min_periods=5).std()
        out["rolling_return_std_10"] = return_1.rolling(window=10, min_periods=10).std()
        out["rolling_return_zscore_10"] = _rolling_zscore(return_1, 10)
        out["momentum_ratio_5_20"] = out["rolling_return_mean_5"] / out[
            "rolling_return_mean_10"
        ].replace(0, np.nan)

        out["volatility_5"] = return_1.rolling(window=5, min_periods=5).std()
        out["volatility_20"] = return_1.rolling(window=20, min_periods=20).std()
        out["volatility_ratio_5_20"] = out["volatility_5"] / out["volatility_20"].replace(
            0, np.nan
        )
        vol_median = out["volatility_20"].rolling(window=50, min_periods=20).median()
        out["is_low_volatility_regime"] = (out["volatility_20"] < vol_median).astype(float)
        out["is_high_volatility_regime"] = (out["volatility_20"] > vol_median).astype(float)

        for column in (
            "return_1",
            "return_3",
            "return_5",
            "return_10",
            "return_20",
            "rolling_return_mean_5",
            "rolling_return_mean_10",
            "rolling_return_std_5",
            "rolling_return_std_10",
            "rolling_return_zscore_10",
            "momentum_ratio_5_20",
            "volatility_5",
            "volatility_20",
            "volatility_ratio_5_20",
            "is_low_volatility_regime",
            "is_high_volatility_regime",
        ):
            out[column] = _safe_fill_series(out[column])

    if spread_column is None:
        skipped.extend(
            [
                "spread_bps",
                "rolling_spread_mean_5",
                "rolling_spread_mean_20",
                "rolling_spread_std_20",
                "spread_zscore_20",
                "spread_percentile_50",
                "is_tight_spread_regime",
                "is_wide_spread_regime",
                "spread_x_volatility_20",
                "wide_spread_x_volatility",
            ]
        )
    else:
        spread = out[spread_column].astype(float)
        out["spread_bps"] = spread
        out["rolling_spread_mean_5"] = spread.rolling(window=5, min_periods=5).mean()
        out["rolling_spread_mean_20"] = spread.rolling(window=20, min_periods=20).mean()
        out["rolling_spread_std_20"] = spread.rolling(window=20, min_periods=20).std()
        out["spread_zscore_20"] = _rolling_zscore(spread, 20)
        out["spread_percentile_50"] = spread.rolling(window=50, min_periods=50).apply(
            lambda values: float(pd.Series(values).rank(pct=True).iloc[-1]),
            raw=False,
        )
        spread_mean_20 = out["rolling_spread_mean_20"]
        spread_std_20 = out["rolling_spread_std_20"]
        out["is_tight_spread_regime"] = (
            spread < (spread_mean_20 - spread_std_20)
        ).astype(float)
        out["is_wide_spread_regime"] = (
            spread > (spread_mean_20 + spread_std_20)
        ).astype(float)

        for column in (
            "spread_bps",
            "rolling_spread_mean_5",
            "rolling_spread_mean_20",
            "rolling_spread_std_20",
            "spread_zscore_20",
            "spread_percentile_50",
            "is_tight_spread_regime",
            "is_wide_spread_regime",
        ):
            out[column] = _safe_fill_series(out[column])

    if imbalance_column is None:
        skipped.extend(
            [
                "bbo_imbalance",
                "imbalance_change_1",
                "imbalance_mean_5",
                "imbalance_mean_20",
                "imbalance_std_20",
                "imbalance_zscore_20",
                "imbalance_persistence_5",
                "imbalance_persistence_10",
                "imbalance_x_return_5",
                "tight_spread_x_imbalance",
            ]
        )
    else:
        imbalance = out[imbalance_column].astype(float)
        out["bbo_imbalance"] = imbalance
        out["imbalance_change_1"] = imbalance.diff(1)
        out["imbalance_mean_5"] = imbalance.rolling(window=5, min_periods=5).mean()
        out["imbalance_mean_20"] = imbalance.rolling(window=20, min_periods=20).mean()
        out["imbalance_std_20"] = imbalance.rolling(window=20, min_periods=20).std()
        out["imbalance_zscore_20"] = _rolling_zscore(imbalance, 20)
        out["imbalance_persistence_5"] = _imbalance_persistence(imbalance, 5)
        out["imbalance_persistence_10"] = _imbalance_persistence(imbalance, 10)

        for column in (
            "bbo_imbalance",
            "imbalance_change_1",
            "imbalance_mean_5",
            "imbalance_mean_20",
            "imbalance_std_20",
            "imbalance_zscore_20",
            "imbalance_persistence_5",
            "imbalance_persistence_10",
        ):
            out[column] = _safe_fill_series(out[column])

    if "return_5" in out.columns and "bbo_imbalance" in out.columns:
        out["imbalance_x_return_5"] = _safe_fill_series(
            out["bbo_imbalance"] * out["return_5"]
        )
    elif "imbalance_x_return_5" not in skipped:
        skipped.append("imbalance_x_return_5")

    if "spread_bps" in out.columns and "volatility_20" in out.columns:
        out["spread_x_volatility_20"] = _safe_fill_series(
            out["spread_bps"] * out["volatility_20"]
        )
    elif "spread_x_volatility_20" not in skipped:
        skipped.append("spread_x_volatility_20")

    if "is_tight_spread_regime" in out.columns and "bbo_imbalance" in out.columns:
        out["tight_spread_x_imbalance"] = _safe_fill_series(
            out["is_tight_spread_regime"] * out["bbo_imbalance"]
        )
    elif "tight_spread_x_imbalance" not in skipped:
        skipped.append("tight_spread_x_imbalance")

    if "is_wide_spread_regime" in out.columns and "volatility_20" in out.columns:
        out["wide_spread_x_volatility"] = _safe_fill_series(
            out["is_wide_spread_regime"] * out["volatility_20"]
        )
    elif "wide_spread_x_volatility" not in skipped:
        skipped.append("wide_spread_x_volatility")

    return out, sorted(set(skipped))


def build_feature_metadata(
    *,
    feature_set: str,
    feature_columns: tuple[str, ...],
    skipped_microstructure_features: list[str] | None = None,
) -> dict[str, Any]:
    """Build report metadata for the active feature set."""
    mode = parse_feature_set(feature_set)
    columns = list(feature_columns)
    skipped = list(skipped_microstructure_features or [])
    metadata: dict[str, Any] = {
        "feature_set": mode,
        "feature_columns": columns,
        "feature_count": len(columns),
        "skipped_microstructure_features": skipped,
        "feature_warning": None,
    }
    if mode == "microstructure" and len(columns) < MIN_MICROSTRUCTURE_FEATURE_COUNT:
        metadata["feature_warning"] = (
            f"too_few_features: only {len(columns)} columns available "
            f"(expected at least {MIN_MICROSTRUCTURE_FEATURE_COUNT})"
        )
    return metadata


def _prepare_source_columns_for_basic_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, set[str]]:
    """Ensure create_features inputs exist; return df and missing source column names."""
    out = df.copy()
    missing_sources: set[str] = set()

    spread_column = resolve_spread_column(out)
    if spread_column is None:
        out["avg_spread_bps"] = 0.0
        missing_sources.add("avg_spread_bps")
    elif spread_column != "avg_spread_bps":
        out["avg_spread_bps"] = out[spread_column].astype(float)

    imbalance_column = resolve_imbalance_column(out)
    if imbalance_column is None:
        out["avg_bbo_imbalance"] = 0.0
        missing_sources.add("avg_bbo_imbalance")
    elif imbalance_column != "avg_bbo_imbalance":
        out["avg_bbo_imbalance"] = out[imbalance_column].astype(float)

    return out, missing_sources


def _basic_features_blocked_by_missing_sources(missing_sources: set[str]) -> set[str]:
    blocked = set(missing_sources)
    if "avg_spread_bps" in missing_sources:
        blocked.add("spread_change_1")
    if "avg_bbo_imbalance" in missing_sources:
        blocked.add("bbo_imbalance_change_1")
    return blocked


def create_walk_forward_features(
    df: pd.DataFrame,
    *,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create model features for walk-forward evaluation."""
    mode = parse_feature_set(feature_set)
    prepared_df, missing_sources = _prepare_source_columns_for_basic_features(df)
    featured_df = create_features(prepared_df)
    skipped: list[str] = sorted(_basic_features_blocked_by_missing_sources(missing_sources))

    if mode == "microstructure":
        featured_df, micro_skipped = _add_microstructure_features(
            featured_df,
            missing_sources=missing_sources,
        )
        skipped = sorted(set(skipped) | set(micro_skipped))

    blocked_basic = _basic_features_blocked_by_missing_sources(missing_sources)
    requested_columns = get_walk_forward_feature_columns(mode)
    feature_columns = tuple(
        column
        for column in requested_columns
        if column in featured_df.columns and column not in blocked_basic
    )
    metadata = build_feature_metadata(
        feature_set=mode,
        feature_columns=feature_columns,
        skipped_microstructure_features=skipped if mode == "microstructure" else [],
    )
    return featured_df, metadata


def walk_forward_feature_matrix(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Extract the active feature matrix, filling safe defaults for sparse columns."""
    available = [column for column in feature_columns if column in df.columns]
    matrix = df[available].copy()
    for column in ("avg_bbo_imbalance", "bbo_imbalance", "bbo_imbalance_change_1", "imbalance_change_1"):
        if column in matrix.columns:
            matrix[column] = matrix[column].fillna(0.0)
    return matrix.astype(float)


def walk_forward_rows_ready_for_model(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Drop rows that are missing required labels, returns, or active features."""
    mid_column = resolve_mid_price_column(df) or MID_PRICE_COLUMN
    required = list(feature_columns) + ["label", "future_return_bps", mid_column]
    required = [column for column in required if column in df.columns]
    return df.dropna(subset=required).reset_index(drop=True)


def train_walk_forward_sklearn_model(
    train_df: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> dict[str, Any]:
    """Train sklearn classifier on the active walk-forward feature set."""
    if not _SKLEARN_AVAILABLE:
        return {
            "available": False,
            "model": None,
            "message": "sklearn not installed; using baseline models only",
        }

    X_train = walk_forward_feature_matrix(train_df, feature_columns)
    y_train = train_df["label"].tolist()
    if len(X_train) == 0:
        return {
            "available": True,
            "model": None,
            "message": "no training rows available for sklearn model",
        }

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    if len(set(y_train)) < 2:
        model: Any = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            class_weight="balanced_subsample",
        )

    model.fit(X_scaled, y_train)
    return {
        "available": True,
        "model": model,
        "scaler": scaler,
        "model_type": type(model).__name__,
        "message": None,
    }


def predict_walk_forward_sklearn(
    sklearn_result: dict[str, Any],
    test_df: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> list[str] | None:
    """Predict labels with the active walk-forward feature set."""
    model = sklearn_result.get("model")
    scaler = sklearn_result.get("scaler")
    if model is None or scaler is None:
        return None
    X_test = walk_forward_feature_matrix(test_df, feature_columns)
    if len(X_test) == 0:
        return []
    predictions = model.predict(scaler.transform(X_test))
    return [str(label) for label in predictions]

def compute_label_distribution(labels: list[str]) -> dict[str, dict[str, float | int]]:
    """Return UP/DOWN/FLAT counts and fractions for report payloads."""
    total = len(labels)
    counts = {label: 0 for label in LABELS}
    for label in labels:
        normalized = str(label).upper()
        if normalized in counts:
            counts[normalized] += 1
    if total == 0:
        return {label: {"count": 0, "fraction": 0.0} for label in LABELS}
    return {
        label: {"count": counts[label], "fraction": counts[label] / total}
        for label in LABELS
    }


def summarize_label_required_edge_bps(values: pd.Series | list[float]) -> dict[str, float]:
    """Summarize per-row label required edge values in bps."""
    series = pd.Series(values, dtype=float)
    if series.empty:
        return {
            "average_label_required_edge_bps": 0.0,
            "min_label_required_edge_bps": 0.0,
            "max_label_required_edge_bps": 0.0,
        }
    return {
        "average_label_required_edge_bps": float(series.mean()),
        "min_label_required_edge_bps": float(series.min()),
        "max_label_required_edge_bps": float(series.max()),
    }


def create_cost_aware_labels(
    df: pd.DataFrame,
    *,
    horizon: int,
    fee_bps: float,
    slippage_bps: float,
    default_spread_bps: float,
    label_cost_multiplier: float = DEFAULT_LABEL_COST_MULTIPLIER,
    label_min_edge_bps: float = DEFAULT_LABEL_MIN_EDGE_BPS,
) -> pd.DataFrame:
    """Add future_return_bps and cost-aware UP/DOWN/FLAT labels (labels only, not execution)."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    out = df.copy()
    mid = out["close_mid"].astype(float)
    future_mid = mid.shift(-horizon)
    out["future_return_bps"] = (future_mid - mid) / mid.replace(0, np.nan) * 10_000.0

    spread_bps = out.apply(lambda row: _spread_cost_bps(row, default_spread_bps), axis=1)
    estimated_cost_bps = fee_bps + slippage_bps + spread_bps
    label_required_edge_bps = estimated_cost_bps * label_cost_multiplier + label_min_edge_bps
    out["label_required_edge_bps"] = label_required_edge_bps

    future_return = out["future_return_bps"]
    labels = np.full(len(out), "FLAT", dtype=object)
    valid = future_return.notna()
    labels[valid & (future_return > label_required_edge_bps)] = "UP"
    labels[valid & (future_return < -label_required_edge_bps)] = "DOWN"
    out["label"] = labels
    return out


def create_walk_forward_labels(
    df: pd.DataFrame,
    *,
    label_mode: str = DEFAULT_LABEL_MODE,
    horizon: int = DEFAULT_HORIZON,
    threshold_bps: float = DEFAULT_THRESHOLD_BPS,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    default_spread_bps: float = DEFAULT_SPREAD_BPS,
    label_cost_multiplier: float = DEFAULT_LABEL_COST_MULTIPLIER,
    label_min_edge_bps: float = DEFAULT_LABEL_MIN_EDGE_BPS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create training labels for walk-forward evaluation."""
    mode = parse_label_mode(label_mode)
    if mode == "direction":
        labeled_df = create_labels(df, horizon=horizon, threshold_bps=threshold_bps)
        distribution = compute_label_distribution(labeled_df["label"].tolist())
        summary = {
            "average_label_required_edge_bps": float(threshold_bps),
            "min_label_required_edge_bps": float(threshold_bps),
            "max_label_required_edge_bps": float(threshold_bps),
        }
        return labeled_df, {
            "label_mode": mode,
            "label_cost_multiplier": label_cost_multiplier,
            "label_min_edge_bps": label_min_edge_bps,
            "label_required_edge_summary": summary,
            "label_distribution_by_mode": distribution,
        }

    labeled_df = create_cost_aware_labels(
        df,
        horizon=horizon,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        default_spread_bps=default_spread_bps,
        label_cost_multiplier=label_cost_multiplier,
        label_min_edge_bps=label_min_edge_bps,
    )
    distribution = compute_label_distribution(labeled_df["label"].tolist())
    summary = summarize_label_required_edge_bps(labeled_df["label_required_edge_bps"])
    return labeled_df, {
        "label_mode": mode,
        "label_cost_multiplier": label_cost_multiplier,
        "label_min_edge_bps": label_min_edge_bps,
        "label_required_edge_summary": summary,
        "label_distribution_by_mode": distribution,
    }


def cost_aware_label_warnings(labels: list[str], *, label_mode: str) -> list[str]:
    """Return warnings when cost-aware labels are sparse or entirely FLAT."""
    if parse_label_mode(label_mode) != "cost_aware":
        return []

    warnings: list[str] = []
    total = len(labels)
    if total == 0:
        return warnings

    up_down_count = sum(1 for label in labels if label in ("UP", "DOWN"))
    if up_down_count == 0:
        warnings.append(
            "cost_aware_labels_all_flat: no UP or DOWN labels after cost-aware labeling"
        )
        return warnings

    up_down_fraction = up_down_count / total
    if up_down_fraction < MIN_CLASS_FRACTION:
        warnings.append(
            f"cost_aware_labels_sparse: UP+DOWN labels are {up_down_fraction:.1%} "
            f"(under {MIN_CLASS_FRACTION:.0%})"
        )
    return warnings


def _spread_cost_bps(row: pd.Series, default_spread_bps: float) -> float:
    if "avg_spread_bps" in row.index and pd.notna(row["avg_spread_bps"]):
        return float(row["avg_spread_bps"])
    if "spread_bps" in row.index and pd.notna(row["spread_bps"]):
        return float(row["spread_bps"])
    return float(default_spread_bps)


def _volatility_bps(row: pd.Series) -> float | None:
    if "rolling_volatility_5" in row.index and pd.notna(row["rolling_volatility_5"]):
        return float(row["rolling_volatility_5"])
    return None


def _estimate_cost_bps(
    row: pd.Series,
    *,
    fee_bps: float,
    slippage_bps: float,
    default_spread_bps: float,
) -> float:
    spread_cost = _spread_cost_bps(row, default_spread_bps)
    return fee_bps + slippage_bps + spread_cost


def _prediction_confidence(
    *,
    sklearn_result: dict[str, Any],
    model_predictions: list[str],
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    model_used: str,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> list[float]:
    model = sklearn_result.get("model")
    scaler = sklearn_result.get("scaler")
    if model is not None and scaler is not None and hasattr(model, "predict_proba"):
        X_test = walk_forward_feature_matrix(test_df, feature_columns)
        if len(X_test) == 0:
            return []
        confidences: list[float] = []
        for proba in model.predict_proba(scaler.transform(X_test)):
            confidences.append(float(max(proba)))
        return confidences

    if model_used == MajorityClassBaseline.model_type:
        labels = train_df["label"].tolist()
        if not labels:
            return [0.0] * len(model_predictions)
        majority_fraction = labels.count(
            str(pd.Series(labels).value_counts().idxmax())
        ) / len(labels)
        return [majority_fraction] * len(model_predictions)

    if model_used == RuleBaseline.model_type:
        returns = test_df["mid_return_1"].astype(float).abs()
        max_return = float(returns.max()) if len(returns) else 1.0
        if max_return <= 0:
            return [0.5] * len(model_predictions)
        return [min(1.0, float(value) / max_return) for value in returns]

    return [1.0] * len(model_predictions)


def _confidence_edge_from_proba(proba: np.ndarray) -> float:
    sorted_probs = np.sort(proba)[::-1]
    if len(sorted_probs) < 2:
        return float(sorted_probs[0]) if len(sorted_probs) == 1 else 0.0
    return float(sorted_probs[0] - sorted_probs[1])


def _prediction_confidence_edges(
    *,
    sklearn_result: dict[str, Any],
    model_predictions: list[str],
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    model_used: str,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> list[float]:
    model = sklearn_result.get("model")
    scaler = sklearn_result.get("scaler")
    if model is not None and scaler is not None and hasattr(model, "predict_proba"):
        X_test = walk_forward_feature_matrix(test_df, feature_columns)
        if len(X_test) == 0:
            return []
        return [
            _confidence_edge_from_proba(proba)
            for proba in model.predict_proba(scaler.transform(X_test))
        ]

    confidences = _prediction_confidence(
        sklearn_result=sklearn_result,
        model_predictions=model_predictions,
        test_df=test_df,
        train_df=train_df,
        model_used=model_used,
        feature_columns=feature_columns,
    )
    n_classes = len(LABELS)
    return [
        max(0.0, confidence - (1.0 - confidence) / max(1, n_classes - 1))
        for confidence in confidences
    ]


def _expected_edge_proxy_bps(
    confidence_edge: float,
    *,
    horizon_return_scale_bps: float,
    threshold_bps: float,
    min_absolute_future_label_bps: float | None,
) -> float:
    proxy = confidence_edge * horizon_return_scale_bps
    if min_absolute_future_label_bps is not None:
        return max(proxy, threshold_bps)
    return proxy


def evaluate_trade_filter(
    *,
    prediction: str,
    confidence: float,
    confidence_edge: float,
    row: pd.Series,
    fee_bps: float,
    slippage_bps: float,
    default_spread_bps: float,
    trade_filter: TradeFilterConfig,
) -> tuple[bool, str, float, float]:
    """Return (allowed, block_reason, total_cost_bps, required_edge_bps)."""
    spread_bps = _spread_cost_bps(row, default_spread_bps)
    total_cost_bps = fee_bps + slippage_bps + spread_bps
    required_edge_bps = total_cost_bps * trade_filter.cost_multiplier + trade_filter.min_edge_bps
    expected_edge_proxy_bps = _expected_edge_proxy_bps(
        confidence_edge,
        horizon_return_scale_bps=trade_filter.horizon_return_scale_bps,
        threshold_bps=trade_filter.threshold_bps,
        min_absolute_future_label_bps=trade_filter.min_absolute_future_label_bps,
    )

    if prediction == "FLAT":
        return False, "flat", total_cost_bps, required_edge_bps

    if prediction not in trade_filter.trade_classes:
        return False, "trade_class", total_cost_bps, required_edge_bps

    if confidence < trade_filter.min_confidence:
        return False, "confidence", total_cost_bps, required_edge_bps

    if trade_filter.max_spread_bps is not None and spread_bps > trade_filter.max_spread_bps:
        return False, "spread", total_cost_bps, required_edge_bps

    volatility_bps = _volatility_bps(row)
    if trade_filter.min_volatility_bps is not None:
        if volatility_bps is None or volatility_bps < trade_filter.min_volatility_bps:
            return False, "volatility", total_cost_bps, required_edge_bps
    if trade_filter.max_volatility_bps is not None:
        if volatility_bps is None or volatility_bps > trade_filter.max_volatility_bps:
            return False, "volatility", total_cost_bps, required_edge_bps

    if trade_filter.min_absolute_future_label_bps is not None:
        label_edge = max(expected_edge_proxy_bps, trade_filter.threshold_bps)
        if label_edge <= trade_filter.min_absolute_future_label_bps:
            return False, "cost", total_cost_bps, required_edge_bps

    if expected_edge_proxy_bps <= required_edge_bps:
        return False, "cost", total_cost_bps, required_edge_bps

    return True, "", total_cost_bps, required_edge_bps


def empty_class_count() -> dict[str, int]:
    """Return zero counts for UP/DOWN/FLAT prediction classes."""
    return {label: 0 for label in LABELS}


def empty_class_trade_metrics() -> dict[str, Any]:
    """Return zeroed trade metrics for one predicted class."""
    return {
        "trades": 0,
        "cumulative_net_return_bps": 0.0,
        "average_net_return_bps": 0.0,
        "win_rate_after_costs": 0.0,
        "average_cost_bps": 0.0,
        "average_confidence": 0.0,
    }


def empty_class_trade_diagnostics() -> dict[str, dict[str, Any]]:
    """Return empty per-class trade diagnostics for UP/DOWN/FLAT."""
    return {label: empty_class_trade_metrics() for label in LABELS}


def _normalize_predicted_class(prediction: str) -> str:
    normalized = str(prediction).upper()
    if normalized in LABELS:
        return normalized
    return "FLAT"


def aggregate_class_trade_diagnostics(
    accepted_trade_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute per-class metrics from accepted filtered paper trade records."""
    diagnostics = empty_class_trade_diagnostics()
    grouped: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
    for record in accepted_trade_records:
        predicted_class = _normalize_predicted_class(record.get("predicted_class", "FLAT"))
        grouped[predicted_class].append(record)

    for label in LABELS:
        records = grouped[label]
        if not records:
            continue
        net_returns = [float(record["net_return_bps"]) for record in records]
        costs = [float(record["cost_bps"]) for record in records]
        confidences = [float(record["confidence"]) for record in records]
        trades = len(records)
        cumulative_net_return_bps = float(sum(net_returns))
        wins = sum(1 for value in net_returns if value > 0)
        diagnostics[label] = {
            "trades": trades,
            "cumulative_net_return_bps": cumulative_net_return_bps,
            "average_net_return_bps": cumulative_net_return_bps / trades,
            "win_rate_after_costs": wins / trades,
            "average_cost_bps": float(np.mean(costs)),
            "average_confidence": float(np.mean(confidences)),
        }
    return diagnostics


def aggregate_coin_class_trade_diagnostics(
    accepted_trade_records: list[dict[str, Any]],
    *,
    coins: list[str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute per-coin, per-class metrics from accepted filtered paper trade records."""
    coin_set = {coin.upper() for coin in coins} if coins else None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in accepted_trade_records:
        coin = str(record.get("coin", "UNKNOWN")).upper()
        if coin_set is not None and coin not in coin_set:
            continue
        grouped.setdefault(coin, []).append(record)

    if coin_set is not None:
        for coin in sorted(coin_set):
            grouped.setdefault(coin, [])

    return {
        coin: aggregate_class_trade_diagnostics(records)
        for coin, records in sorted(grouped.items())
    }


def empty_stability_bucket_metrics() -> dict[str, Any]:
    """Return zeroed stability metrics for one diagnostic bucket."""
    return {
        "trades": 0,
        "cumulative_net_return_bps": 0.0,
        "average_net_return_bps": 0.0,
        "win_rate_after_costs": 0.0,
        "average_cost_bps": 0.0,
        "average_confidence": 0.0,
        "positive_after_costs": False,
    }


def aggregate_stability_bucket_metrics(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute stability metrics for one bucket of accepted paper trades."""
    if not records:
        return empty_stability_bucket_metrics()

    net_returns = [float(record["net_return_bps"]) for record in records]
    costs = [float(record["cost_bps"]) for record in records]
    confidences = [float(record["confidence"]) for record in records]
    trades = len(records)
    cumulative_net_return_bps = float(sum(net_returns))
    wins = sum(1 for value in net_returns if value > 0)
    return {
        "trades": trades,
        "cumulative_net_return_bps": cumulative_net_return_bps,
        "average_net_return_bps": cumulative_net_return_bps / trades,
        "win_rate_after_costs": wins / trades,
        "average_cost_bps": float(np.mean(costs)),
        "average_confidence": float(np.mean(confidences)),
        "positive_after_costs": cumulative_net_return_bps > 0,
    }


def _parse_timestamp_iso(value: object) -> datetime | None:
    """Parse timestamp_iso safely without external dependencies."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _stability_group_records(
    accepted_trade_records: list[dict[str, Any]],
    *,
    key_fn: Any,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in accepted_trade_records:
        key = key_fn(record)
        if key is None:
            continue
        grouped.setdefault(str(key), []).append(record)
    return grouped


def compute_stability_diagnostics(
    accepted_trade_records: list[dict[str, Any]],
    *,
    coins: list[str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute stability diagnostics grouped by coin, class, time, and window."""
    by_coin = _stability_group_records(
        accepted_trade_records,
        key_fn=lambda record: str(record.get("coin", "UNKNOWN")).upper(),
    )
    if coins:
        for coin in sorted({value.upper() for value in coins}):
            by_coin.setdefault(coin, [])

    by_predicted_class = _stability_group_records(
        accepted_trade_records,
        key_fn=lambda record: _normalize_predicted_class(record.get("predicted_class", "FLAT")),
    )
    for label in LABELS:
        by_predicted_class.setdefault(label, [])

    by_coin_predicted_class = _stability_group_records(
        accepted_trade_records,
        key_fn=lambda record: (
            f"{str(record.get('coin', 'UNKNOWN')).upper()}:"
            f"{_normalize_predicted_class(record.get('predicted_class', 'FLAT'))}"
        ),
    )

    by_hour = _stability_group_records(
        accepted_trade_records,
        key_fn=lambda record: (
            f"{parsed.hour:02d}"
            if (parsed := _parse_timestamp_iso(record.get("timestamp_iso"))) is not None
            else None
        ),
    )
    by_date = _stability_group_records(
        accepted_trade_records,
        key_fn=lambda record: (
            parsed.date().isoformat()
            if (parsed := _parse_timestamp_iso(record.get("timestamp_iso"))) is not None
            else None
        ),
    )
    by_window_index = _stability_group_records(
        accepted_trade_records,
        key_fn=lambda record: (
            str(record["window_index"])
            if record.get("window_index") is not None
            else None
        ),
    )

    def _metrics_map(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
        return {
            key: aggregate_stability_bucket_metrics(records)
            for key, records in sorted(grouped.items())
        }

    return {
        "by_coin": _metrics_map(by_coin),
        "by_predicted_class": _metrics_map(by_predicted_class),
        "by_coin_predicted_class": _metrics_map(by_coin_predicted_class),
        "by_hour": _metrics_map(by_hour),
        "by_date": _metrics_map(by_date),
        "by_window_index": _metrics_map(by_window_index),
    }


def _iter_stability_buckets(
    stability_diagnostics: dict[str, dict[str, dict[str, Any]]],
) -> list[tuple[str, str, dict[str, Any]]]:
    buckets: list[tuple[str, str, dict[str, Any]]] = []
    for group_name, group in stability_diagnostics.items():
        for bucket_key, metrics in group.items():
            buckets.append((group_name, bucket_key, metrics))
    return buckets


def _dimension_has_concentration(
    group: dict[str, dict[str, Any]],
    *,
    concentration_fraction: float = STABILITY_CONCENTRATION_FRACTION,
) -> bool:
    positive_values = [
        float(metrics["cumulative_net_return_bps"])
        for metrics in group.values()
        if metrics.get("trades", 0) > 0
        and float(metrics.get("cumulative_net_return_bps", 0.0)) > 0
    ]
    if not positive_values:
        return False
    total_positive = float(sum(positive_values))
    if total_positive <= 0:
        return False
    largest_positive = max(positive_values)
    return largest_positive / total_positive > concentration_fraction


def compute_stability_summary(
    stability_diagnostics: dict[str, dict[str, dict[str, Any]]],
    *,
    min_trades_per_bucket: int,
) -> dict[str, Any]:
    """Summarize stability diagnostics across all buckets."""
    buckets = _iter_stability_buckets(stability_diagnostics)
    active_buckets = [
        metrics for _, _, metrics in buckets if int(metrics.get("trades", 0)) > 0
    ]

    if not active_buckets:
        return {
            "profitable_bucket_fraction": 0.0,
            "unprofitable_bucket_fraction": 0.0,
            "buckets_with_too_few_trades": 0,
            "largest_positive_bucket_net_bps": 0.0,
            "largest_negative_bucket_net_bps": 0.0,
            "concentration_warning": False,
            "sparse_stability_warning": False,
        }

    profitable = sum(1 for metrics in active_buckets if metrics.get("positive_after_costs"))
    total_active = len(active_buckets)
    sparse_buckets = sum(
        1
        for metrics in active_buckets
        if 0 < int(metrics.get("trades", 0)) < min_trades_per_bucket
    )
    cumulative_values = [
        float(metrics.get("cumulative_net_return_bps", 0.0)) for metrics in active_buckets
    ]
    positive_values = [value for value in cumulative_values if value > 0]
    negative_values = [value for value in cumulative_values if value < 0]

    concentration_groups = ("by_hour", "by_date", "by_window_index")
    concentration_warning = any(
        _dimension_has_concentration(stability_diagnostics.get(group_name, {}))
        for group_name in concentration_groups
        if stability_diagnostics.get(group_name)
    )

    return {
        "profitable_bucket_fraction": profitable / total_active,
        "unprofitable_bucket_fraction": (total_active - profitable) / total_active,
        "buckets_with_too_few_trades": sparse_buckets,
        "largest_positive_bucket_net_bps": max(positive_values) if positive_values else 0.0,
        "largest_negative_bucket_net_bps": min(negative_values) if negative_values else 0.0,
        "concentration_warning": concentration_warning,
        "sparse_stability_warning": sparse_buckets > 0,
    }


def build_stability_warnings(
    stability_summary: dict[str, Any],
    *,
    min_trades_per_bucket: int,
) -> list[str]:
    """Build human-readable stability warnings from summary metrics."""
    warnings: list[str] = []
    if stability_summary.get("concentration_warning"):
        warnings.append(
            "stability_concentration: one time/window bucket contributes more than "
            f"{STABILITY_CONCENTRATION_FRACTION:.0%} of total positive net bps"
        )
    if stability_summary.get("sparse_stability_warning"):
        warnings.append(
            "stability_sparse_buckets: "
            f"{stability_summary.get('buckets_with_too_few_trades', 0)} bucket(s) have "
            f"fewer than {min_trades_per_bucket} trades"
        )
    return warnings


def apply_stability_readiness(
    readiness: dict[str, bool],
    stability_summary: dict[str, Any],
) -> dict[str, bool]:
    """Adjust readiness when stability validation fails."""
    updated = dict(readiness)
    if stability_summary.get("concentration_warning") or stability_summary.get(
        "sparse_stability_warning"
    ):
        updated["ready_for_more_research"] = False
    updated["real_execution_ready"] = False
    return updated


def _format_stability_bucket_lines(
    *,
    heading: str,
    buckets: dict[str, dict[str, Any]],
) -> list[str]:
    lines = [heading, ""]
    if not buckets:
        lines.append("- no buckets")
        lines.append("")
        return lines

    for bucket_key, metrics in sorted(buckets.items()):
        lines.extend(
            [
                f"### {bucket_key}",
                "",
                f"- trades: {metrics.get('trades', 0)}",
                f"- cumulative_net_return_bps: {metrics.get('cumulative_net_return_bps', 0.0):.4f}",
                f"- average_net_return_bps: {metrics.get('average_net_return_bps', 0.0):.4f}",
                f"- win_rate_after_costs: {metrics.get('win_rate_after_costs', 0.0):.4f}",
                f"- average_cost_bps: {metrics.get('average_cost_bps', 0.0):.4f}",
                f"- average_confidence: {metrics.get('average_confidence', 0.0):.4f}",
                f"- positive_after_costs: {str(metrics.get('positive_after_costs', False)).lower()}",
                "",
            ]
        )
    return lines


def simulate_paper_signal_returns(
    test_df: pd.DataFrame,
    predictions: list[str],
    confidences: list[float],
    confidence_edges: list[float],
    *,
    fee_bps: float,
    slippage_bps: float,
    default_spread_bps: float,
    trade_filter: TradeFilterConfig,
    coin: str | None = None,
    window_index: int | None = None,
) -> dict[str, Any]:
    """Simulate paper-only cost-adjusted returns with cost-aware filtering."""
    gross_returns: list[float] = []
    net_returns: list[float] = []
    trade_returns: list[float] = []
    accepted_trade_records: list[dict[str, Any]] = []
    raw_predictions = 0
    filtered_paper_trades = 0
    blocked_by_confidence = 0
    blocked_by_cost = 0
    blocked_by_spread = 0
    blocked_by_volatility = 0
    blocked_by_trade_class = 0
    blocked_by_class = empty_class_count()
    accepted_by_class = empty_class_count()
    cost_samples: list[float] = []
    required_edge_samples: list[float] = []

    for index, (_, row) in enumerate(test_df.iterrows()):
        prediction = predictions[index] if index < len(predictions) else "FLAT"
        predicted_class = _normalize_predicted_class(prediction)
        confidence = float(confidences[index]) if index < len(confidences) else 0.0
        confidence_edge = (
            float(confidence_edges[index]) if index < len(confidence_edges) else 0.0
        )
        future_return = float(row["future_return_bps"])
        spread_bps = _spread_cost_bps(row, default_spread_bps)
        actual_label = str(row["label"]) if "label" in row.index and pd.notna(row["label"]) else "FLAT"

        allowed, block_reason, total_cost_bps, required_edge_bps = evaluate_trade_filter(
            prediction=prediction,
            confidence=confidence,
            confidence_edge=confidence_edge,
            row=row,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            default_spread_bps=default_spread_bps,
            trade_filter=trade_filter,
        )

        if prediction in ("UP", "DOWN"):
            raw_predictions += 1
            cost_samples.append(total_cost_bps)
            required_edge_samples.append(required_edge_bps)

        if not allowed:
            blocked_by_class[predicted_class] += 1
            if block_reason == "confidence":
                blocked_by_confidence += 1
            elif block_reason == "cost":
                blocked_by_cost += 1
            elif block_reason == "spread":
                blocked_by_spread += 1
            elif block_reason == "volatility":
                blocked_by_volatility += 1
            elif block_reason == "trade_class":
                blocked_by_trade_class += 1
            gross_returns.append(0.0)
            net_returns.append(0.0)
            continue

        if prediction == "UP":
            gross = future_return
        else:
            gross = -future_return

        net = gross - total_cost_bps
        gross_returns.append(gross)
        net_returns.append(net)
        filtered_paper_trades += 1
        trade_returns.append(net)
        accepted_by_class[predicted_class] += 1
        accepted_trade_records.append(
            {
                "coin": str(coin).upper() if coin else "UNKNOWN",
                "window_index": window_index,
                "row_index": index,
                "timestamp_iso": (
                    str(row["timestamp_iso"])
                    if "timestamp_iso" in row.index and pd.notna(row["timestamp_iso"])
                    else ""
                ),
                "predicted_class": predicted_class,
                "actual_label": actual_label,
                "confidence": confidence,
                "future_return_bps": future_return,
                "gross_return_bps": gross,
                "cost_bps": total_cost_bps,
                "net_return_bps": net,
                "spread_bps": spread_bps,
            }
        )

    trade_filter_summary = trade_filter_to_summary(trade_filter)
    class_trade_diagnostics = aggregate_class_trade_diagnostics(accepted_trade_records)

    return {
        "gross_returns_bps": gross_returns,
        "net_returns_bps": net_returns,
        "accepted_trade_net_returns_bps": trade_returns,
        "trade_returns_bps": trade_returns,
        "accepted_trade_records": accepted_trade_records,
        "class_trade_diagnostics": class_trade_diagnostics,
        "blocked_by_class": blocked_by_class,
        "accepted_by_class": accepted_by_class,
        "raw_predictions": raw_predictions,
        "filtered_paper_trades": filtered_paper_trades,
        "blocked_by_confidence": blocked_by_confidence,
        "blocked_by_cost": blocked_by_cost,
        "blocked_by_spread": blocked_by_spread,
        "blocked_by_volatility": blocked_by_volatility,
        "blocked_by_trade_class": blocked_by_trade_class,
        "trade_filter_summary": trade_filter_summary,
        "average_cost_bps": float(np.mean(cost_samples)) if cost_samples else 0.0,
        "average_required_edge_bps": (
            float(np.mean(required_edge_samples)) if required_edge_samples else 0.0
        ),
        "total_paper_signals": raw_predictions,
        "total_paper_trades": filtered_paper_trades,
    }


def trade_filter_to_summary(trade_filter: TradeFilterConfig) -> dict[str, Any]:
    """Serialize trade filter settings for reports."""
    return {
        "min_edge_bps": trade_filter.min_edge_bps,
        "max_spread_bps": trade_filter.max_spread_bps,
        "min_volatility_bps": trade_filter.min_volatility_bps,
        "max_volatility_bps": trade_filter.max_volatility_bps,
        "trade_classes": sorted(trade_filter.trade_classes),
        "cost_multiplier": trade_filter.cost_multiplier,
        "horizon_return_scale_bps": trade_filter.horizon_return_scale_bps,
        "min_absolute_future_label_bps": trade_filter.min_absolute_future_label_bps,
        "min_confidence": trade_filter.min_confidence,
        "threshold_bps": trade_filter.threshold_bps,
    }


def build_trade_filter(
    *,
    min_confidence: float,
    threshold_bps: float,
    min_edge_bps: float = DEFAULT_MIN_EDGE_BPS,
    max_spread_bps: float | None = None,
    min_volatility_bps: float | None = None,
    max_volatility_bps: float | None = None,
    trade_classes: frozenset[str] | None = None,
    cost_multiplier: float = DEFAULT_COST_MULTIPLIER,
    horizon_return_scale_bps: float = DEFAULT_HORIZON_RETURN_SCALE_BPS,
    min_absolute_future_label_bps: float | None = None,
) -> TradeFilterConfig:
    return TradeFilterConfig(
        min_edge_bps=min_edge_bps,
        max_spread_bps=max_spread_bps,
        min_volatility_bps=min_volatility_bps,
        max_volatility_bps=max_volatility_bps,
        trade_classes=trade_classes or DEFAULT_TRADE_CLASSES,
        cost_multiplier=cost_multiplier,
        horizon_return_scale_bps=horizon_return_scale_bps,
        min_absolute_future_label_bps=min_absolute_future_label_bps,
        min_confidence=min_confidence,
        threshold_bps=threshold_bps,
    )


NET_RETURN_CONSISTENCY_TOLERANCE_BPS = 1e-9


def aggregate_accepted_trade_metrics(
    accepted_trade_net_returns_bps: list[float],
) -> dict[str, Any]:
    """Compute net-return metrics from one accepted filtered-trade return list."""
    filtered_paper_trades = len(accepted_trade_net_returns_bps)
    if filtered_paper_trades == 0:
        return {
            "accepted_trade_net_returns_bps": [],
            "filtered_paper_trades": 0,
            "cumulative_net_return_bps": 0.0,
            "average_net_return_bps": 0.0,
            "win_rate_after_costs": 0.0,
            "max_drawdown_bps": 0.0,
            "longest_losing_streak": 0,
            "positive_after_costs": False,
            "expected_average_from_cumulative": 0.0,
            "average_net_return_difference_abs": 0.0,
            "net_return_consistency_check": True,
        }

    cumulative_net_return_bps = float(sum(accepted_trade_net_returns_bps))
    average_net_return_bps = cumulative_net_return_bps / filtered_paper_trades
    expected_average_from_cumulative = cumulative_net_return_bps / filtered_paper_trades
    average_net_return_difference_abs = abs(
        average_net_return_bps - expected_average_from_cumulative
    )
    net_return_consistency_check = (
        average_net_return_difference_abs <= NET_RETURN_CONSISTENCY_TOLERANCE_BPS
    )
    wins = sum(1 for value in accepted_trade_net_returns_bps if value > 0)

    return {
        "accepted_trade_net_returns_bps": list(accepted_trade_net_returns_bps),
        "filtered_paper_trades": filtered_paper_trades,
        "cumulative_net_return_bps": cumulative_net_return_bps,
        "average_net_return_bps": average_net_return_bps,
        "win_rate_after_costs": wins / filtered_paper_trades,
        "max_drawdown_bps": max_drawdown_bps(accepted_trade_net_returns_bps),
        "longest_losing_streak": longest_losing_streak(accepted_trade_net_returns_bps),
        "positive_after_costs": cumulative_net_return_bps > 0,
        "expected_average_from_cumulative": expected_average_from_cumulative,
        "average_net_return_difference_abs": average_net_return_difference_abs,
        "net_return_consistency_check": net_return_consistency_check,
    }


def max_drawdown_bps(net_returns: list[float]) -> float:
    """Peak-to-trough drawdown on cumulative net returns."""
    if not net_returns:
        return 0.0
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in net_returns:
        cumulative += value
        if cumulative > peak:
            peak = cumulative
        drawdown = peak - cumulative
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def longest_losing_streak(net_returns: list[float]) -> int:
    """Longest run of consecutive negative net returns."""
    max_streak = 0
    current = 0
    for value in net_returns:
        if value < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _no_trade_predictions(count: int) -> list[str]:
    return ["FLAT"] * count


def compute_readiness_flags(
    *,
    accuracy_edge_vs_baseline: float,
    cumulative_net_return_bps: float,
    completed_windows: int,
    total_paper_trades: int,
    min_windows: int,
    min_trades: int,
) -> dict[str, bool]:
    """Return top-level walk-forward readiness booleans (never None)."""
    enough_windows = completed_windows >= min_windows
    enough_trades = total_paper_trades >= min_trades
    return {
        "beats_baseline_accuracy": accuracy_edge_vs_baseline > 0,
        "positive_after_costs": cumulative_net_return_bps > 0,
        "enough_windows": enough_windows,
        "enough_trades": enough_trades,
        "ready_for_more_research": enough_windows and enough_trades,
        "real_execution_ready": False,
    }


def _label_imbalance_warnings(labels: list[str], split_name: str) -> list[str]:
    warnings: list[str] = []
    total = len(labels)
    if total == 0:
        return warnings
    counts = pd.Series(labels).value_counts()
    for label in LABELS:
        fraction = counts.get(label, 0) / total
        if fraction < MIN_CLASS_FRACTION:
            warnings.append(
                f"class_imbalance: {split_name} {label} is {fraction:.1%} "
                f"(under {MIN_CLASS_FRACTION:.0%})"
            )
        if fraction > MAX_CLASS_FRACTION:
            warnings.append(
                f"class_imbalance: {split_name} {label} is {fraction:.1%} "
                f"(over {MAX_CLASS_FRACTION:.0%})"
            )
    return warnings


def _missing_data_warning(df: pd.DataFrame) -> str | None:
    check_columns = [
        column
        for column in ("avg_spread_bps", "spread_bps", "close_mid", "avg_bbo_imbalance")
        if column in df.columns
    ]
    if not check_columns or df.empty:
        return None
    missing_fraction = float(df[check_columns].isna().any(axis=1).mean())
    if missing_fraction >= MISSING_DATA_FRACTION_WARNING:
        return (
            f"high_missing_data: {missing_fraction:.1%} of rows have missing "
            f"values in {check_columns}"
        )
    return None


def _merge_class_counts(*counts: dict[str, int]) -> dict[str, int]:
    merged = empty_class_count()
    for count_map in counts:
        for label in LABELS:
            merged[label] += int(count_map.get(label, 0))
    return merged


def _format_class_trade_diagnostics_section(
    class_trade_diagnostics: dict[str, dict[str, Any]],
    *,
    heading: str,
) -> list[str]:
    lines = [heading, ""]
    for label in LABELS:
        metrics = class_trade_diagnostics.get(label, empty_class_trade_metrics())
        lines.extend(
            [
                f"### {label}",
                "",
                f"- trades: {metrics.get('trades', 0)}",
                f"- cumulative_net_return_bps: {metrics.get('cumulative_net_return_bps', 0.0):.4f}",
                f"- average_net_return_bps: {metrics.get('average_net_return_bps', 0.0):.4f}",
                f"- win_rate_after_costs: {metrics.get('win_rate_after_costs', 0.0):.4f}",
                f"- average_cost_bps: {metrics.get('average_cost_bps', 0.0):.4f}",
                f"- average_confidence: {metrics.get('average_confidence', 0.0):.4f}",
                "",
            ]
        )
    return lines


def write_accepted_trades_csv(
    accepted_trade_records: list[dict[str, Any]],
    *,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    report: dict[str, Any],
    limit: int = DEFAULT_ACCEPTED_TRADES_LIMIT,
) -> Path | None:
    """Write accepted filtered paper trades to CSV (research-only diagnostics)."""
    if not accepted_trade_records:
        return None

    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = _report_stamp(report)
    coins_label = "_".join(report.get("coins", ["UNKNOWN"]))
    output_path = reports_dir / f"accepted_trades_{coins_label}_{stamp}.csv"

    limited_records = accepted_trade_records[: max(0, limit)]
    rows = [
        {column: record.get(column, "") for column in ACCEPTED_TRADES_CSV_COLUMNS}
        for record in limited_records
    ]
    pd.DataFrame(rows, columns=list(ACCEPTED_TRADES_CSV_COLUMNS)).to_csv(
        output_path, index=False
    )
    report["accepted_trades_csv_path"] = str(output_path)
    report["accepted_trades_count"] = len(limited_records)
    return output_path


def finalize_accepted_trades_csv(
    report: dict[str, Any],
    accepted_trade_records: list[dict[str, Any]],
    *,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    save_accepted_trades: bool = True,
    accepted_trades_limit: int = DEFAULT_ACCEPTED_TRADES_LIMIT,
) -> None:
    """Persist accepted filtered paper trades CSV and report metadata."""
    filtered_paper_trades = int(report.get("filtered_paper_trades", 0))
    if (
        save_accepted_trades
        and filtered_paper_trades > 0
        and accepted_trade_records
    ):
        write_accepted_trades_csv(
            accepted_trade_records,
            reports_dir=reports_dir,
            report=report,
            limit=accepted_trades_limit,
        )
        return

    report["accepted_trades_csv_path"] = None
    report["accepted_trades_count"] = 0


def evaluate_walk_forward_window(
    model_df: pd.DataFrame,
    window: tuple[int, int, int, int],
    *,
    window_index: int,
    fee_bps: float,
    slippage_bps: float,
    default_spread_bps: float,
    trade_filter: TradeFilterConfig,
    save_window_models: bool,
    models_dir: Path,
    coin: str,
    interval: str,
    horizon: int,
    threshold_bps: float,
    source: str,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> dict[str, Any]:
    """Train on one chronological window and evaluate on the following test slice."""
    train_start, train_end, test_start, test_end = window
    train_df = model_df.iloc[train_start:train_end].copy().reset_index(drop=True)
    test_df = model_df.iloc[test_start:test_end].copy().reset_index(drop=True)

    if len(train_df) < 5 or len(test_df) < 1:
        return {
            "window_index": window_index,
            "status": "skipped",
            "reason": "too_few_rows",
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        }

    baseline = train_baseline_model(train_df)
    sklearn_result = train_walk_forward_sklearn_model(train_df, feature_columns)

    majority_model: MajorityClassBaseline = baseline["majority_model"]
    rule_model: RuleBaseline = baseline["rule_model"]
    baseline_predictions = majority_model.predict(test_df)
    no_trade_predictions = _no_trade_predictions(len(test_df))
    momentum_predictions = rule_model.predict(walk_forward_feature_matrix(test_df, feature_columns))
    sklearn_predictions = predict_walk_forward_sklearn(
        sklearn_result,
        test_df,
        feature_columns,
    )

    if sklearn_predictions is not None:
        model_predictions = sklearn_predictions
        model_used = sklearn_result.get("model_type") or "sklearn"
    else:
        model_predictions = baseline_predictions
        model_used = majority_model.model_type

    y_true = test_df["label"].tolist()
    confidences = _prediction_confidence(
        sklearn_result=sklearn_result,
        model_predictions=model_predictions,
        test_df=test_df,
        train_df=train_df,
        model_used=model_used,
        feature_columns=feature_columns,
    )
    confidence_edges = _prediction_confidence_edges(
        sklearn_result=sklearn_result,
        model_predictions=model_predictions,
        test_df=test_df,
        train_df=train_df,
        model_used=model_used,
        feature_columns=feature_columns,
    )
    paper_simulation = simulate_paper_signal_returns(
        test_df,
        model_predictions,
        confidences,
        confidence_edges,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        default_spread_bps=default_spread_bps,
        trade_filter=trade_filter,
        coin=coin,
        window_index=window_index,
    )
    accepted_trade_net_returns_bps = list(
        paper_simulation["accepted_trade_net_returns_bps"]
    )
    trade_metrics = aggregate_accepted_trade_metrics(accepted_trade_net_returns_bps)

    model_artifact_path: str | None = None
    if save_window_models and sklearn_result.get("model") is not None:
        artifact_id = f"wf_{window_index}_{uuid.uuid4().hex[:8]}"
        payload = build_artifact_payload(
            model=sklearn_result["model"],
            scaler=sklearn_result.get("scaler"),
            feature_names=list(feature_columns),
            coin=coin,
            interval=interval,
            horizon=horizon,
            threshold_bps=threshold_bps,
            train_rows=len(train_df),
            test_rows=len(test_df),
            baseline_accuracy=_accuracy(y_true, baseline_predictions),
            model_accuracy=_accuracy(y_true, model_predictions),
            model_type=str(sklearn_result.get("model_type") or model_used),
        )
        payload["source"] = normalize_source(source)
        payload["walk_forward_window_index"] = window_index
        saved_path = save_model_artifact(payload, models_dir=models_dir, artifact_id=artifact_id)
        model_artifact_path = str(saved_path)

    return {
        "window_index": window_index,
        "status": "completed",
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "model_used": model_used,
        "model_accuracy": _accuracy(y_true, model_predictions),
        "baseline_accuracy": _accuracy(y_true, baseline_predictions),
        "no_trade_baseline_accuracy": _accuracy(y_true, no_trade_predictions),
        "momentum_baseline_accuracy": _accuracy(y_true, momentum_predictions),
        "prediction_counts": {
            label: model_predictions.count(label) for label in LABELS
        },
        "baseline_prediction_counts": {
            label: baseline_predictions.count(label) for label in LABELS
        },
        "total_paper_signals": paper_simulation["total_paper_signals"],
        "total_paper_trades": paper_simulation["total_paper_trades"],
        "raw_predictions": paper_simulation["raw_predictions"],
        "filtered_paper_trades": paper_simulation["filtered_paper_trades"],
        "blocked_by_confidence": paper_simulation["blocked_by_confidence"],
        "blocked_by_cost": paper_simulation["blocked_by_cost"],
        "blocked_by_spread": paper_simulation["blocked_by_spread"],
        "blocked_by_volatility": paper_simulation["blocked_by_volatility"],
        "blocked_by_trade_class": paper_simulation["blocked_by_trade_class"],
        "blocked_by_class": paper_simulation["blocked_by_class"],
        "accepted_by_class": paper_simulation["accepted_by_class"],
        "class_trade_diagnostics": paper_simulation["class_trade_diagnostics"],
        "accepted_trade_records": paper_simulation["accepted_trade_records"],
        "trade_filter_summary": paper_simulation["trade_filter_summary"],
        "average_cost_bps": paper_simulation["average_cost_bps"],
        "average_required_edge_bps": paper_simulation["average_required_edge_bps"],
        "average_gross_return_bps": (
            float(np.mean(paper_simulation["gross_returns_bps"]))
            if paper_simulation["gross_returns_bps"]
            else 0.0
        ),
        "accepted_trade_net_returns_bps": accepted_trade_net_returns_bps,
        "average_net_return_bps": trade_metrics["average_net_return_bps"],
        "cumulative_net_return_bps": trade_metrics["cumulative_net_return_bps"],
        "win_rate_after_costs": trade_metrics["win_rate_after_costs"],
        "longest_losing_streak": trade_metrics["longest_losing_streak"],
        "max_drawdown_bps": trade_metrics["max_drawdown_bps"],
        "positive_after_costs": trade_metrics["positive_after_costs"],
        "expected_average_from_cumulative": trade_metrics["expected_average_from_cumulative"],
        "average_net_return_difference_abs": trade_metrics["average_net_return_difference_abs"],
        "net_return_consistency_check": trade_metrics["net_return_consistency_check"],
        "trade_returns_bps": accepted_trade_net_returns_bps,
        "model_artifact_path": model_artifact_path,
        "warnings": _label_imbalance_warnings(y_true, "test"),
    }


def evaluate_coin_walk_forward(
    *,
    coin: str,
    interval: str,
    source: str,
    horizon: int,
    threshold_bps: float,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    fee_bps: float,
    slippage_bps: float,
    default_spread_bps: float,
    trade_filter: TradeFilterConfig,
    data_root: Path,
    save_window_models: bool,
    models_dir: Path,
    max_rows: int | None = None,
    min_windows: int = DEFAULT_MIN_WINDOWS,
    min_trades: int = DEFAULT_MIN_TRADES,
    label_mode: str = DEFAULT_LABEL_MODE,
    label_cost_multiplier: float = DEFAULT_LABEL_COST_MULTIPLIER,
    label_min_edge_bps: float = DEFAULT_LABEL_MIN_EDGE_BPS,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> dict[str, Any]:
    """Run walk-forward evaluation for one coin."""
    normalized_source = normalize_source(source)
    warnings: list[str] = [SAFETY_NOTE]

    try:
        raw_df = load_training_data(coin, interval, data_root, source=normalized_source)
    except FileNotFoundError:
        readiness = compute_readiness_flags(
            accuracy_edge_vs_baseline=0.0,
            cumulative_net_return_bps=0.0,
            completed_windows=0,
            total_paper_trades=0,
            min_windows=min_windows,
            min_trades=min_trades,
        )
        return {
            "coin": coin,
            "status": "skipped",
            "reason": "csv_not_found",
            "warnings": warnings + [f"too_few_rows: CSV not found for {coin}"],
            "windows": [],
            "completed_windows": 0,
            "total_paper_trades": 0,
            "accuracy_edge_vs_baseline": 0.0,
            "cumulative_net_return_bps": 0.0,
            **readiness,
            "readiness": readiness,
            **build_feature_metadata(
                feature_set=feature_set,
                feature_columns=get_walk_forward_feature_columns(feature_set),
            ),
        }

    if max_rows is not None and max_rows > 0:
        raw_df = raw_df.iloc[:max_rows].copy()

    missing_warning = _missing_data_warning(raw_df)
    if missing_warning:
        warnings.append(missing_warning)

    featured_df, feature_meta = create_walk_forward_features(
        raw_df,
        feature_set=feature_set,
    )
    feature_columns = tuple(feature_meta["feature_columns"])
    labeled_df, label_meta = create_walk_forward_labels(
        featured_df,
        label_mode=label_mode,
        horizon=horizon,
        threshold_bps=threshold_bps,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        default_spread_bps=default_spread_bps,
        label_cost_multiplier=label_cost_multiplier,
        label_min_edge_bps=label_min_edge_bps,
    )
    model_df = walk_forward_rows_ready_for_model(labeled_df, feature_columns)

    if len(model_df) < MIN_ROWS_WARNING:
        warnings.append(f"too_few_rows: only {len(model_df)} model-ready rows for {coin}")

    windows = generate_walk_forward_windows(
        len(model_df),
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=step_bars,
    )

    window_results: list[dict[str, Any]] = []
    for window_index, window in enumerate(windows):
        result = evaluate_walk_forward_window(
            model_df,
            window,
            window_index=window_index,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            default_spread_bps=default_spread_bps,
            trade_filter=trade_filter,
            save_window_models=save_window_models,
            models_dir=models_dir,
            coin=coin,
            interval=interval,
            horizon=horizon,
            threshold_bps=threshold_bps,
            source=normalized_source,
            feature_columns=feature_columns,
        )
        window_results.append(result)

    completed = [item for item in window_results if item.get("status") == "completed"]
    skipped = [item for item in window_results if item.get("status") != "completed"]

    model_accuracies = [float(item["model_accuracy"]) for item in completed]
    baseline_accuracies = [float(item["baseline_accuracy"]) for item in completed]
    model_accuracy_mean = float(np.mean(model_accuracies)) if model_accuracies else 0.0
    baseline_accuracy_mean = float(np.mean(baseline_accuracies)) if baseline_accuracies else 0.0

    total_paper_signals = sum(int(item.get("total_paper_signals", 0)) for item in completed)
    total_paper_trades = sum(int(item.get("filtered_paper_trades", 0)) for item in completed)
    raw_predictions = sum(int(item.get("raw_predictions", 0)) for item in completed)
    filtered_paper_trades = total_paper_trades
    blocked_by_confidence = sum(int(item.get("blocked_by_confidence", 0)) for item in completed)
    blocked_by_cost = sum(int(item.get("blocked_by_cost", 0)) for item in completed)
    blocked_by_spread = sum(int(item.get("blocked_by_spread", 0)) for item in completed)
    blocked_by_volatility = sum(int(item.get("blocked_by_volatility", 0)) for item in completed)
    blocked_by_trade_class = sum(int(item.get("blocked_by_trade_class", 0)) for item in completed)

    cost_values = [
        float(item.get("average_cost_bps", 0.0))
        for item in completed
        if item.get("raw_predictions", 0) > 0
    ]
    required_edge_values = [
        float(item.get("average_required_edge_bps", 0.0))
        for item in completed
        if item.get("raw_predictions", 0) > 0
    ]
    average_cost_bps = float(np.mean(cost_values)) if cost_values else 0.0
    average_required_edge_bps = float(np.mean(required_edge_values)) if required_edge_values else 0.0

    accepted_trade_net_returns_bps: list[float] = []
    accepted_trade_records: list[dict[str, Any]] = []
    blocked_by_class = empty_class_count()
    accepted_by_class = empty_class_count()
    for window in completed:
        accepted_trade_net_returns_bps.extend(
            float(value)
            for value in window.get(
                "accepted_trade_net_returns_bps",
                window.get("trade_returns_bps", []),
            )
        )
        accepted_trade_records.extend(window.get("accepted_trade_records", []))
        blocked_by_class = _merge_class_counts(
            blocked_by_class, window.get("blocked_by_class", empty_class_count())
        )
        accepted_by_class = _merge_class_counts(
            accepted_by_class, window.get("accepted_by_class", empty_class_count())
        )

    trade_metrics = aggregate_accepted_trade_metrics(accepted_trade_net_returns_bps)
    class_trade_diagnostics = aggregate_class_trade_diagnostics(accepted_trade_records)
    coin_class_trade_diagnostics = {
        coin: class_trade_diagnostics,
    }
    cumulative_net_return_bps = trade_metrics["cumulative_net_return_bps"]
    average_net_return_bps = trade_metrics["average_net_return_bps"]
    win_rate_after_costs = trade_metrics["win_rate_after_costs"]
    average_gross_return_bps = (
        float(np.mean([item.get("average_gross_return_bps", 0.0) for item in completed]))
        if completed
        else 0.0
    )

    if not trade_metrics["net_return_consistency_check"]:
        warnings.append("net_return_aggregation_inconsistent")

    if len(windows) < min_windows:
        warnings.append(
            f"too_few_windows: only {len(windows)} possible windows "
            f"(need {min_windows})"
        )
    accuracy_edge_vs_baseline = model_accuracy_mean - baseline_accuracy_mean
    if accuracy_edge_vs_baseline <= 0 and completed:
        warnings.append("model_not_better_than_baseline")
    if cumulative_net_return_bps <= 0 and total_paper_trades > 0:
        warnings.append("negative_after_costs")

    train_labels = model_df["label"].tolist() if not model_df.empty else []
    warnings.extend(_label_imbalance_warnings(train_labels, "all"))
    warnings.extend(cost_aware_label_warnings(train_labels, label_mode=label_mode))
    if feature_meta.get("feature_warning"):
        warnings.append(str(feature_meta["feature_warning"]))

    readiness = compute_readiness_flags(
        accuracy_edge_vs_baseline=accuracy_edge_vs_baseline,
        cumulative_net_return_bps=cumulative_net_return_bps,
        completed_windows=len(completed),
        total_paper_trades=filtered_paper_trades,
        min_windows=min_windows,
        min_trades=min_trades,
    )

    source_path = str(resampled_csv_path(coin, interval, data_root, source=normalized_source))

    return {
        "coin": coin,
        "status": "completed" if completed else "skipped",
        "source_path": source_path,
        "model_ready_rows": len(model_df),
        "total_windows": len(windows),
        "completed_windows": len(completed),
        "skipped_windows": len(skipped),
        "model_accuracy_mean": model_accuracy_mean,
        "baseline_accuracy_mean": baseline_accuracy_mean,
        "accuracy_edge_vs_baseline": accuracy_edge_vs_baseline,
        "total_paper_signals": total_paper_signals,
        "total_paper_trades": total_paper_trades,
        "raw_predictions": raw_predictions,
        "filtered_paper_trades": filtered_paper_trades,
        "blocked_by_confidence": blocked_by_confidence,
        "blocked_by_cost": blocked_by_cost,
        "blocked_by_spread": blocked_by_spread,
        "blocked_by_volatility": blocked_by_volatility,
        "blocked_by_trade_class": blocked_by_trade_class,
        "blocked_by_class": blocked_by_class,
        "accepted_by_class": accepted_by_class,
        "class_trade_diagnostics": class_trade_diagnostics,
        "coin_class_trade_diagnostics": coin_class_trade_diagnostics,
        "accepted_trade_records": accepted_trade_records,
        "trade_filter_summary": trade_filter_to_summary(trade_filter),
        "average_cost_bps": average_cost_bps,
        "average_required_edge_bps": average_required_edge_bps,
        "average_gross_return_bps": average_gross_return_bps,
        "accepted_trade_net_returns_bps": accepted_trade_net_returns_bps,
        "average_net_return_bps": average_net_return_bps,
        "cumulative_net_return_bps": cumulative_net_return_bps,
        "max_drawdown_bps": trade_metrics["max_drawdown_bps"],
        "win_rate_after_costs": win_rate_after_costs,
        "longest_losing_streak": trade_metrics["longest_losing_streak"],
        "expected_average_from_cumulative": trade_metrics["expected_average_from_cumulative"],
        "average_net_return_difference_abs": trade_metrics["average_net_return_difference_abs"],
        "net_return_consistency_check": trade_metrics["net_return_consistency_check"],
        **readiness,
        "readiness": readiness,
        "windows": window_results,
        "warnings": warnings,
        **label_meta,
        **feature_meta,
    }


def _aggregate_label_distribution(
    per_coin_results: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Merge per-coin label distributions into one aggregate distribution."""
    totals = {label: 0 for label in LABELS}
    for item in per_coin_results:
        distribution = item.get("label_distribution_by_mode", {})
        for label in LABELS:
            label_stats = distribution.get(label, {})
            totals[label] += int(label_stats.get("count", 0))
    total_rows = sum(totals.values())
    if total_rows == 0:
        return {label: {"count": 0, "fraction": 0.0} for label in LABELS}
    return {
        label: {"count": totals[label], "fraction": totals[label] / total_rows}
        for label in LABELS
    }


def _aggregate_label_required_edge_summary(
    per_coin_results: list[dict[str, Any]],
    *,
    label_mode: str,
    threshold_bps: float,
) -> dict[str, float]:
    """Merge per-coin label required-edge summaries for the combined report."""
    if parse_label_mode(label_mode) == "direction":
        return {
            "average_label_required_edge_bps": float(threshold_bps),
            "min_label_required_edge_bps": float(threshold_bps),
            "max_label_required_edge_bps": float(threshold_bps),
        }

    summaries = [
        item["label_required_edge_summary"]
        for item in per_coin_results
        if item.get("label_required_edge_summary")
    ]
    if not summaries:
        return summarize_label_required_edge_bps([])

    averages = [float(item["average_label_required_edge_bps"]) for item in summaries]
    mins = [float(item["min_label_required_edge_bps"]) for item in summaries]
    maxes = [float(item["max_label_required_edge_bps"]) for item in summaries]
    return {
        "average_label_required_edge_bps": float(np.mean(averages)),
        "min_label_required_edge_bps": float(min(mins)),
        "max_label_required_edge_bps": float(max(maxes)),
    }


def build_walk_forward_report(
    *,
    coins: list[str],
    interval: str,
    source: str,
    horizon: int,
    threshold_bps: float,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    fee_bps: float,
    slippage_bps: float,
    default_spread_bps: float,
    trade_filter: TradeFilterConfig,
    per_coin_results: list[dict[str, Any]],
    data_root: Path,
    min_windows: int = DEFAULT_MIN_WINDOWS,
    min_trades: int = DEFAULT_MIN_TRADES,
    label_mode: str = DEFAULT_LABEL_MODE,
    label_cost_multiplier: float = DEFAULT_LABEL_COST_MULTIPLIER,
    label_min_edge_bps: float = DEFAULT_LABEL_MIN_EDGE_BPS,
    feature_set: str = DEFAULT_FEATURE_SET,
    feature_columns: tuple[str, ...] | None = None,
    skipped_microstructure_features: list[str] | None = None,
    feature_warning: str | None = None,
    stability_report_enabled: bool = False,
    stability_min_trades_per_bucket: int = DEFAULT_STABILITY_MIN_TRADES_PER_BUCKET,
) -> dict[str, Any]:
    """Aggregate per-coin walk-forward results into one combined report."""
    normalized_source = normalize_source(source)
    data_dir = resolve_bid_ask_data_dir(interval, normalized_source, data_root)

    completed_coins = [item for item in per_coin_results if item.get("completed_windows", 0) > 0]
    warnings: list[str] = [SAFETY_NOTE]

    total_windows = sum(int(item.get("total_windows", 0)) for item in per_coin_results)
    completed_windows = sum(int(item.get("completed_windows", 0)) for item in per_coin_results)
    skipped_windows = sum(int(item.get("skipped_windows", 0)) for item in per_coin_results)
    total_paper_signals = sum(int(item.get("total_paper_signals", 0)) for item in per_coin_results)
    raw_predictions = sum(int(item.get("raw_predictions", 0)) for item in per_coin_results)
    filtered_paper_trades = sum(int(item.get("filtered_paper_trades", 0)) for item in per_coin_results)
    total_paper_trades = filtered_paper_trades
    blocked_by_confidence = sum(int(item.get("blocked_by_confidence", 0)) for item in per_coin_results)
    blocked_by_cost = sum(int(item.get("blocked_by_cost", 0)) for item in per_coin_results)
    blocked_by_spread = sum(int(item.get("blocked_by_spread", 0)) for item in per_coin_results)
    blocked_by_volatility = sum(int(item.get("blocked_by_volatility", 0)) for item in per_coin_results)
    blocked_by_trade_class = sum(int(item.get("blocked_by_trade_class", 0)) for item in per_coin_results)

    cost_values = [
        float(item.get("average_cost_bps", 0.0))
        for item in completed_coins
        if item.get("raw_predictions", 0) > 0
    ]
    required_edge_values = [
        float(item.get("average_required_edge_bps", 0.0))
        for item in completed_coins
        if item.get("raw_predictions", 0) > 0
    ]
    average_cost_bps = float(np.mean(cost_values)) if cost_values else 0.0
    average_required_edge_bps = float(np.mean(required_edge_values)) if required_edge_values else 0.0

    model_accuracy_values = [
        float(item["model_accuracy_mean"])
        for item in completed_coins
        if item.get("model_accuracy_mean") is not None
    ]
    baseline_accuracy_values = [
        float(item["baseline_accuracy_mean"])
        for item in completed_coins
        if item.get("baseline_accuracy_mean") is not None
    ]
    model_accuracy_mean = float(np.mean(model_accuracy_values)) if model_accuracy_values else 0.0
    baseline_accuracy_mean = (
        float(np.mean(baseline_accuracy_values)) if baseline_accuracy_values else 0.0
    )

    accepted_trade_net_returns_bps: list[float] = []
    accepted_trade_records: list[dict[str, Any]] = []
    blocked_by_class = empty_class_count()
    accepted_by_class = empty_class_count()
    for item in completed_coins:
        accepted_trade_net_returns_bps.extend(
            float(value)
            for value in item.get(
                "accepted_trade_net_returns_bps",
                item.get("trade_returns_bps", []),
            )
        )
        accepted_trade_records.extend(item.get("accepted_trade_records", []))
        blocked_by_class = _merge_class_counts(
            blocked_by_class, item.get("blocked_by_class", empty_class_count())
        )
        accepted_by_class = _merge_class_counts(
            accepted_by_class, item.get("accepted_by_class", empty_class_count())
        )

    trade_metrics = aggregate_accepted_trade_metrics(accepted_trade_net_returns_bps)
    class_trade_diagnostics = aggregate_class_trade_diagnostics(accepted_trade_records)
    coin_class_trade_diagnostics = aggregate_coin_class_trade_diagnostics(
        accepted_trade_records,
        coins=coins,
    )
    cumulative_net_return_bps = trade_metrics["cumulative_net_return_bps"]
    average_net_return_bps = trade_metrics["average_net_return_bps"]
    win_rate_after_costs = trade_metrics["win_rate_after_costs"]
    gross_values = [
        float(item.get("average_gross_return_bps", 0.0))
        for item in completed_coins
        if item.get("total_paper_trades", 0) > 0
    ]

    for item in per_coin_results:
        warnings.extend(item.get("warnings", []))

    if total_windows < min_windows:
        warnings.append(
            f"too_few_windows: aggregate total_windows={total_windows} "
            f"(need {min_windows})"
        )
    accuracy_edge_vs_baseline = model_accuracy_mean - baseline_accuracy_mean
    if accuracy_edge_vs_baseline <= 0 and completed_coins:
        warnings.append("model_not_better_than_baseline")
    if cumulative_net_return_bps <= 0 and total_paper_trades > 0:
        warnings.append("negative_after_costs")
    if not trade_metrics["net_return_consistency_check"]:
        warnings.append("net_return_aggregation_inconsistent")

    readiness = compute_readiness_flags(
        accuracy_edge_vs_baseline=accuracy_edge_vs_baseline,
        cumulative_net_return_bps=cumulative_net_return_bps,
        completed_windows=completed_windows,
        total_paper_trades=filtered_paper_trades,
        min_windows=min_windows,
        min_trades=min_trades,
    )

    stability_diagnostics: dict[str, dict[str, dict[str, Any]]] | None = None
    stability_summary: dict[str, Any] | None = None
    stability_warnings: list[str] = []
    if stability_report_enabled:
        stability_diagnostics = compute_stability_diagnostics(
            accepted_trade_records,
            coins=coins,
        )
        stability_summary = compute_stability_summary(
            stability_diagnostics,
            min_trades_per_bucket=stability_min_trades_per_bucket,
        )
        stability_warnings = build_stability_warnings(
            stability_summary,
            min_trades_per_bucket=stability_min_trades_per_bucket,
        )
        readiness = apply_stability_readiness(readiness, stability_summary)
        warnings.extend(stability_warnings)

    per_coin_summary = {
        item["coin"]: {
            "status": item.get("status"),
            "completed_windows": item.get("completed_windows", 0),
            "model_accuracy_mean": item.get("model_accuracy_mean", 0.0),
            "baseline_accuracy_mean": item.get("baseline_accuracy_mean", 0.0),
            "accuracy_edge_vs_baseline": item.get("accuracy_edge_vs_baseline", 0.0),
            "raw_predictions": item.get("raw_predictions", 0),
            "filtered_paper_trades": item.get("filtered_paper_trades", 0),
            "total_paper_trades": item.get("filtered_paper_trades", 0),
            "cumulative_net_return_bps": item.get("cumulative_net_return_bps", 0.0),
            "readiness": item.get("readiness") or compute_readiness_flags(
                accuracy_edge_vs_baseline=float(item.get("accuracy_edge_vs_baseline", 0.0)),
                cumulative_net_return_bps=float(item.get("cumulative_net_return_bps", 0.0)),
                completed_windows=int(item.get("completed_windows", 0)),
                total_paper_trades=int(item.get("filtered_paper_trades", 0)),
                min_windows=min_windows,
                min_trades=min_trades,
            ),
        }
        for item in per_coin_results
    }

    label_distribution_by_mode = _aggregate_label_distribution(per_coin_results)
    label_required_edge_summary = _aggregate_label_required_edge_summary(
        per_coin_results,
        label_mode=label_mode,
        threshold_bps=threshold_bps,
    )

    resolved_feature_columns = tuple(
        feature_columns
        or per_coin_results[0].get("feature_columns", get_walk_forward_feature_columns(feature_set))
        if per_coin_results
        else get_walk_forward_feature_columns(feature_set)
    )
    resolved_skipped = list(
        skipped_microstructure_features
        or (per_coin_results[0].get("skipped_microstructure_features", []) if per_coin_results else [])
    )
    feature_metadata = build_feature_metadata(
        feature_set=feature_set,
        feature_columns=resolved_feature_columns,
        skipped_microstructure_features=resolved_skipped,
    )
    if feature_warning:
        feature_metadata["feature_warning"] = feature_warning

    report: dict[str, Any] = {
        "report_id": uuid.uuid4().hex,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coins": coins,
        "interval": interval,
        "source": normalized_source,
        "data_dir": str(data_dir),
        "data_root": str(data_root),
        "horizon": horizon,
        "threshold_bps": threshold_bps,
        "label_mode": parse_label_mode(label_mode),
        "label_cost_multiplier": label_cost_multiplier,
        "label_min_edge_bps": label_min_edge_bps,
        "label_required_edge_summary": label_required_edge_summary,
        "label_distribution_by_mode": label_distribution_by_mode,
        "train_bars": train_bars,
        "test_bars": test_bars,
        "step_bars": step_bars,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "default_spread_bps": default_spread_bps,
        "min_confidence": trade_filter.min_confidence,
        "min_edge_bps": trade_filter.min_edge_bps,
        "max_spread_bps": trade_filter.max_spread_bps,
        "min_volatility_bps": trade_filter.min_volatility_bps,
        "max_volatility_bps": trade_filter.max_volatility_bps,
        "trade_classes": sorted(trade_filter.trade_classes),
        "cost_multiplier": trade_filter.cost_multiplier,
        "horizon_return_scale_bps": trade_filter.horizon_return_scale_bps,
        "min_absolute_future_label_bps": trade_filter.min_absolute_future_label_bps,
        "trade_filter_summary": trade_filter_to_summary(trade_filter),
        "min_windows": min_windows,
        "min_trades": min_trades,
        "total_windows": total_windows,
        "completed_windows": completed_windows,
        "skipped_windows": skipped_windows,
        "model_accuracy_mean": model_accuracy_mean,
        "baseline_accuracy_mean": baseline_accuracy_mean,
        "accuracy_edge_vs_baseline": accuracy_edge_vs_baseline,
        "total_paper_signals": total_paper_signals,
        "total_paper_trades": total_paper_trades,
        "raw_predictions": raw_predictions,
        "filtered_paper_trades": filtered_paper_trades,
        "blocked_by_confidence": blocked_by_confidence,
        "blocked_by_cost": blocked_by_cost,
        "blocked_by_spread": blocked_by_spread,
        "blocked_by_volatility": blocked_by_volatility,
        "blocked_by_trade_class": blocked_by_trade_class,
        "blocked_by_class": blocked_by_class,
        "accepted_by_class": accepted_by_class,
        "class_trade_diagnostics": class_trade_diagnostics,
        "coin_class_trade_diagnostics": coin_class_trade_diagnostics,
        "accepted_trade_records": accepted_trade_records,
        "average_cost_bps": average_cost_bps,
        "average_required_edge_bps": average_required_edge_bps,
        "accepted_trade_net_returns_bps": accepted_trade_net_returns_bps,
        "average_gross_return_bps": float(np.mean(gross_values)) if gross_values else 0.0,
        "average_net_return_bps": average_net_return_bps,
        "cumulative_net_return_bps": cumulative_net_return_bps,
        "max_drawdown_bps": trade_metrics["max_drawdown_bps"],
        "win_rate_after_costs": win_rate_after_costs,
        "longest_losing_streak": trade_metrics["longest_losing_streak"],
        "expected_average_from_cumulative": trade_metrics["expected_average_from_cumulative"],
        "average_net_return_difference_abs": trade_metrics["average_net_return_difference_abs"],
        "net_return_consistency_check": trade_metrics["net_return_consistency_check"],
        "per_coin": per_coin_summary,
        "per_coin_details": per_coin_results,
        **readiness,
        "readiness": readiness,
        "warnings": list(dict.fromkeys(warnings)),
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "safety_statement": SAFETY_STATEMENT,
        "safety_note": SAFETY_NOTE,
        **feature_metadata,
    }

    report["stability_report_enabled"] = stability_report_enabled
    if stability_report_enabled and stability_diagnostics is not None and stability_summary is not None:
        report["stability_min_trades_per_bucket"] = stability_min_trades_per_bucket
        report["stability_diagnostics"] = stability_diagnostics
        report["stability_summary"] = stability_summary
        report["stability_warnings"] = stability_warnings
        report.update(readiness)
        report["readiness"] = readiness

    return report


def _report_stamp(report: dict[str, Any]) -> str:
    generated_at = str(report.get("generated_at", ""))
    if generated_at:
        return generated_at.replace(":", "").replace("+", "").replace("-", "")[:15]
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_walk_forward_json(
    report: dict[str, Any],
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> Path:
    """Write timestamped JSON walk-forward report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = _report_stamp(report)
    coins_label = "_".join(report.get("coins", ["UNKNOWN"]))
    output_path = reports_dir / f"walk_forward_{coins_label}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    report["json_report_path"] = str(output_path)
    return output_path


def write_walk_forward_markdown(
    report: dict[str, Any],
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> Path:
    """Write timestamped Markdown walk-forward report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = _report_stamp(report)
    coins_label = "_".join(report.get("coins", ["UNKNOWN"]))
    output_path = reports_dir / f"walk_forward_{coins_label}_{stamp}.md"

    readiness = report.get("readiness") or {}
    beats_baseline = report.get("beats_baseline_accuracy", readiness.get("beats_baseline_accuracy", False))
    positive_after_costs = report.get("positive_after_costs", readiness.get("positive_after_costs", False))
    enough_windows = report.get("enough_windows", readiness.get("enough_windows", False))
    enough_trades = report.get("enough_trades", readiness.get("enough_trades", False))
    ready_for_more_research = report.get(
        "ready_for_more_research", readiness.get("ready_for_more_research", False)
    )
    real_execution_ready = report.get("real_execution_ready", readiness.get("real_execution_ready", False))

    lines = [
        "# Walk-Forward Paper Evaluation",
        "",
        f"Generated at: {report.get('generated_at', 'n/a')}",
        f"Coins: {', '.join(report.get('coins', []))}",
        f"Interval: {report.get('interval')} | Source: {report.get('source')}",
        f"Horizon: {report.get('horizon')} bars | Threshold: {report.get('threshold_bps')} bps",
        "",
        "## Safety",
        "",
        f"- paper_only: {str(report.get('paper_only', True)).lower()}",
        f"- execution_enabled: {str(report.get('execution_enabled', False)).lower()}",
        f"- real_execution_ready: {str(real_execution_ready).lower()}",
        "",
        "## Labels",
        "",
        f"- label_mode: {report.get('label_mode', DEFAULT_LABEL_MODE)}",
        f"- label_cost_multiplier: {report.get('label_cost_multiplier', DEFAULT_LABEL_COST_MULTIPLIER)}",
        f"- label_min_edge_bps: {report.get('label_min_edge_bps', DEFAULT_LABEL_MIN_EDGE_BPS)}",
    ]

    label_summary = report.get("label_required_edge_summary", {})
    lines.extend(
        [
            f"- average_label_required_edge_bps: "
            f"{label_summary.get('average_label_required_edge_bps', 0.0):.4f}",
            f"- min_label_required_edge_bps: "
            f"{label_summary.get('min_label_required_edge_bps', 0.0):.4f}",
            f"- max_label_required_edge_bps: "
            f"{label_summary.get('max_label_required_edge_bps', 0.0):.4f}",
        ]
    )
    label_distribution = report.get("label_distribution_by_mode", {})
    for label in LABELS:
        stats = label_distribution.get(label, {})
        lines.append(
            f"- label {label}: count={stats.get('count', 0)} "
            f"fraction={stats.get('fraction', 0.0):.4f}"
        )

    lines.extend(
        [
            "",
            "## Aggregate metrics",
            "",
            f"- total_windows: {report.get('total_windows', 0)}",
            f"- completed_windows: {report.get('completed_windows', 0)}",
            f"- skipped_windows: {report.get('skipped_windows', 0)}",
            f"- model_accuracy_mean: {report.get('model_accuracy_mean', 0.0):.4f}",
            f"- baseline_accuracy_mean: {report.get('baseline_accuracy_mean', 0.0):.4f}",
            f"- accuracy_edge_vs_baseline: {report.get('accuracy_edge_vs_baseline', 0.0):.4f}",
            f"- total_paper_signals: {report.get('total_paper_signals', 0)}",
            f"- total_paper_trades: {report.get('total_paper_trades', 0)}",
            f"- raw_predictions: {report.get('raw_predictions', 0)}",
            f"- filtered_paper_trades: {report.get('filtered_paper_trades', 0)}",
            f"- blocked_by_confidence: {report.get('blocked_by_confidence', 0)}",
            f"- blocked_by_cost: {report.get('blocked_by_cost', 0)}",
            f"- blocked_by_spread: {report.get('blocked_by_spread', 0)}",
            f"- blocked_by_volatility: {report.get('blocked_by_volatility', 0)}",
            f"- blocked_by_trade_class: {report.get('blocked_by_trade_class', 0)}",
            f"- average_cost_bps: {report.get('average_cost_bps', 0.0):.4f}",
            f"- average_required_edge_bps: {report.get('average_required_edge_bps', 0.0):.4f}",
            f"- average_gross_return_bps: {report.get('average_gross_return_bps', 0.0):.4f}",
            f"- filtered_paper_trades (accepted): {report.get('filtered_paper_trades', 0)}",
            f"- cumulative_net_return_bps: {report.get('cumulative_net_return_bps', 0.0):.4f}",
            f"- average_net_return_bps: {report.get('average_net_return_bps', 0.0):.4f}",
            f"- expected_average_from_cumulative: {report.get('expected_average_from_cumulative', 0.0):.4f}",
            f"- average_net_return_difference_abs: {report.get('average_net_return_difference_abs', 0.0):.6f}",
            f"- net_return_consistency_check: {str(report.get('net_return_consistency_check', True)).lower()}",
            f"- max_drawdown_bps: {report.get('max_drawdown_bps', 0.0):.4f}",
            f"- win_rate_after_costs: {report.get('win_rate_after_costs', 0.0):.4f}",
            f"- longest_losing_streak: {report.get('longest_losing_streak', 0)}",
            "",
            "## Readiness",
            "",
            f"- beats_baseline_accuracy: {beats_baseline}",
            f"- positive_after_costs: {positive_after_costs}",
            f"- enough_windows: {enough_windows}",
            f"- enough_trades: {enough_trades}",
            f"- ready_for_more_research: {ready_for_more_research}",
            f"- real_execution_ready: {real_execution_ready}",
            "",
            "## Model vs baseline after costs",
            "",
        ]
    )

    if beats_baseline and positive_after_costs:
        lines.append(
            "Model accuracy beats baseline and cumulative net returns are positive "
            "after simulated costs (paper-only)."
        )
    elif beats_baseline:
        lines.append(
            "Model accuracy beats baseline, but cumulative net returns are not "
            "positive after simulated costs."
        )
    else:
        lines.append(
            "Model does not clearly beat baseline on accuracy and/or net returns "
            "after simulated costs."
        )

    lines.extend(
        [
            "",
            "## Per-coin summary",
            "",
            "| Coin | Windows | Model acc | Baseline acc | Edge | Trades | Net bps |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for coin, summary in sorted(report.get("per_coin", {}).items()):
        lines.append(
            f"| {coin} | {summary.get('completed_windows', 0)} | "
            f"{summary.get('model_accuracy_mean', 0.0):.4f} | "
            f"{summary.get('baseline_accuracy_mean', 0.0):.4f} | "
            f"{summary.get('accuracy_edge_vs_baseline', 0.0):.4f} | "
            f"{summary.get('filtered_paper_trades', summary.get('total_paper_trades', 0))} | "
            f"{summary.get('cumulative_net_return_bps', 0.0):.4f} |"
        )

    lines.extend(
        _format_class_trade_diagnostics_section(
            report.get("class_trade_diagnostics", empty_class_trade_diagnostics()),
            heading="## Trade Diagnostics",
        )
    )

    blocked_by_class = report.get("blocked_by_class", empty_class_count())
    accepted_by_class = report.get("accepted_by_class", empty_class_count())
    lines.extend(
        [
            "### Accepted vs blocked by class",
            "",
            f"- accepted UP: {accepted_by_class.get('UP', 0)}",
            f"- accepted DOWN: {accepted_by_class.get('DOWN', 0)}",
            f"- accepted FLAT: {accepted_by_class.get('FLAT', 0)}",
            f"- blocked UP: {blocked_by_class.get('UP', 0)}",
            f"- blocked DOWN: {blocked_by_class.get('DOWN', 0)}",
            f"- blocked FLAT: {blocked_by_class.get('FLAT', 0)}",
            "",
            "### Coin-class trade diagnostics",
            "",
        ]
    )
    coin_class_trade_diagnostics = report.get(
        "coin_class_trade_diagnostics", {}
    )
    for coin in sorted(coin_class_trade_diagnostics):
        lines.extend(
            _format_class_trade_diagnostics_section(
                coin_class_trade_diagnostics[coin],
                heading=f"#### {coin}",
            )
        )

    if report.get("stability_report_enabled"):
        stability_summary = report.get("stability_summary", {})
        lines.extend(
            [
                "",
                "## Stability Validation",
                "",
                f"- stability_report_enabled: true",
                f"- stability_min_trades_per_bucket: "
                f"{report.get('stability_min_trades_per_bucket', DEFAULT_STABILITY_MIN_TRADES_PER_BUCKET)}",
                f"- profitable_bucket_fraction: "
                f"{stability_summary.get('profitable_bucket_fraction', 0.0):.4f}",
                f"- unprofitable_bucket_fraction: "
                f"{stability_summary.get('unprofitable_bucket_fraction', 0.0):.4f}",
                f"- buckets_with_too_few_trades: "
                f"{stability_summary.get('buckets_with_too_few_trades', 0)}",
                f"- largest_positive_bucket_net_bps: "
                f"{stability_summary.get('largest_positive_bucket_net_bps', 0.0):.4f}",
                f"- largest_negative_bucket_net_bps: "
                f"{stability_summary.get('largest_negative_bucket_net_bps', 0.0):.4f}",
                f"- concentration_warning: "
                f"{str(stability_summary.get('concentration_warning', False)).lower()}",
                f"- sparse_stability_warning: "
                f"{str(stability_summary.get('sparse_stability_warning', False)).lower()}",
            ]
        )
        stability_warnings = report.get("stability_warnings", [])
        lines.extend(["", "### Stability warnings", ""])
        if stability_warnings:
            for warning in stability_warnings:
                lines.append(f"- {warning}")
        else:
            lines.append("- none")

        stability_diagnostics = report.get("stability_diagnostics", {})
        for group_name in (
            "by_coin",
            "by_predicted_class",
            "by_coin_predicted_class",
            "by_hour",
            "by_date",
            "by_window_index",
        ):
            lines.extend(
                _format_stability_bucket_lines(
                    heading=f"### {group_name}",
                    buckets=stability_diagnostics.get(group_name, {}),
                )
            )

    lines.extend(["", "## Warnings", ""])
    warning_items = report.get("warnings", [])
    if not warning_items:
        lines.append("No warnings.")
    else:
        for warning in warning_items:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Research-only disclaimer",
            "",
            report.get("safety_note", SAFETY_NOTE),
            "",
            report.get("safety_statement", SAFETY_STATEMENT),
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["markdown_report_path"] = str(output_path)
    return output_path


def apply_diagnostics_detail(
    report: dict[str, Any],
    *,
    diagnostics_detail: str = DEFAULT_DIAGNOSTICS_DETAIL,
    accepted_trades_limit: int = DEFAULT_ACCEPTED_TRADES_LIMIT,
) -> dict[str, Any]:
    """Trim report payload according to diagnostics detail level."""
    detail = diagnostics_detail.lower()
    report["diagnostics_detail"] = detail
    accepted_trade_records = list(report.get("accepted_trade_records", []))
    report["accepted_trade_records_count"] = len(accepted_trade_records)

    if detail == "full":
        report["accepted_trade_records"] = accepted_trade_records[:accepted_trades_limit]
    else:
        report.pop("accepted_trade_records", None)
        for coin_detail in report.get("per_coin_details", []):
            coin_detail.pop("accepted_trade_records", None)
            for window in coin_detail.get("windows", []):
                window.pop("accepted_trade_records", None)
    return report


def run_walk_forward_evaluation(
    *,
    coins: list[str],
    interval: str,
    source: str = "local",
    horizon: int = DEFAULT_HORIZON,
    threshold_bps: float = DEFAULT_THRESHOLD_BPS,
    train_bars: int = DEFAULT_TRAIN_BARS,
    test_bars: int = DEFAULT_TEST_BARS,
    step_bars: int = DEFAULT_STEP_BARS,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    default_spread_bps: float = DEFAULT_SPREAD_BPS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_edge_bps: float = DEFAULT_MIN_EDGE_BPS,
    max_spread_bps: float | None = None,
    min_volatility_bps: float | None = None,
    max_volatility_bps: float | None = None,
    trade_classes: frozenset[str] | None = None,
    cost_multiplier: float = DEFAULT_COST_MULTIPLIER,
    horizon_return_scale_bps: float = DEFAULT_HORIZON_RETURN_SCALE_BPS,
    min_absolute_future_label_bps: float | None = None,
    label_mode: str = DEFAULT_LABEL_MODE,
    label_cost_multiplier: float = DEFAULT_LABEL_COST_MULTIPLIER,
    label_min_edge_bps: float = DEFAULT_LABEL_MIN_EDGE_BPS,
    feature_set: str = DEFAULT_FEATURE_SET,
    data_root: Path = DEFAULT_DATA_ROOT,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    save_window_models: bool = False,
    max_rows: int | None = None,
    save_reports: bool = True,
    min_windows: int = DEFAULT_MIN_WINDOWS,
    min_trades: int = DEFAULT_MIN_TRADES,
    diagnostics_detail: str = DEFAULT_DIAGNOSTICS_DETAIL,
    save_accepted_trades: bool = True,
    accepted_trades_limit: int = DEFAULT_ACCEPTED_TRADES_LIMIT,
    stability_report_enabled: bool = False,
    stability_min_trades_per_bucket: int = DEFAULT_STABILITY_MIN_TRADES_PER_BUCKET,
) -> dict[str, Any]:
    """Run walk-forward evaluation for one or more coins and write reports."""
    trade_filter = build_trade_filter(
        min_confidence=min_confidence,
        threshold_bps=threshold_bps,
        min_edge_bps=min_edge_bps,
        max_spread_bps=max_spread_bps,
        min_volatility_bps=min_volatility_bps,
        max_volatility_bps=max_volatility_bps,
        trade_classes=trade_classes,
        cost_multiplier=cost_multiplier,
        horizon_return_scale_bps=horizon_return_scale_bps,
        min_absolute_future_label_bps=min_absolute_future_label_bps,
    )
    per_coin_results = [
        evaluate_coin_walk_forward(
            coin=coin,
            interval=interval,
            source=source,
            horizon=horizon,
            threshold_bps=threshold_bps,
            train_bars=train_bars,
            test_bars=test_bars,
            step_bars=step_bars,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            default_spread_bps=default_spread_bps,
            trade_filter=trade_filter,
            data_root=data_root,
            save_window_models=save_window_models,
            models_dir=models_dir,
            max_rows=max_rows,
            min_windows=min_windows,
            min_trades=min_trades,
            label_mode=label_mode,
            label_cost_multiplier=label_cost_multiplier,
            label_min_edge_bps=label_min_edge_bps,
            feature_set=feature_set,
        )
        for coin in coins
    ]

    aggregate_feature_columns = tuple(
        per_coin_results[0]["feature_columns"]
        if per_coin_results and per_coin_results[0].get("feature_columns")
        else get_walk_forward_feature_columns(feature_set)
    )
    aggregate_skipped = sorted(
        {
            feature_name
            for item in per_coin_results
            for feature_name in item.get("skipped_microstructure_features", [])
        }
    )
    aggregate_feature_warning = next(
        (
            str(item["feature_warning"])
            for item in per_coin_results
            if item.get("feature_warning")
        ),
        None,
    )

    report = build_walk_forward_report(
        coins=coins,
        interval=interval,
        source=source,
        horizon=horizon,
        threshold_bps=threshold_bps,
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=step_bars,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        default_spread_bps=default_spread_bps,
        trade_filter=trade_filter,
        per_coin_results=per_coin_results,
        data_root=data_root,
        min_windows=min_windows,
        min_trades=min_trades,
        label_mode=label_mode,
        label_cost_multiplier=label_cost_multiplier,
        label_min_edge_bps=label_min_edge_bps,
        feature_set=feature_set,
        feature_columns=aggregate_feature_columns,
        skipped_microstructure_features=aggregate_skipped,
        feature_warning=aggregate_feature_warning,
        stability_report_enabled=stability_report_enabled,
        stability_min_trades_per_bucket=stability_min_trades_per_bucket,
    )

    all_accepted_trade_records = list(report.get("accepted_trade_records", []))
    report = apply_diagnostics_detail(
        report,
        diagnostics_detail=diagnostics_detail,
        accepted_trades_limit=accepted_trades_limit,
    )

    finalize_accepted_trades_csv(
        report,
        all_accepted_trade_records,
        reports_dir=reports_dir,
        save_accepted_trades=save_accepted_trades,
        accepted_trades_limit=accepted_trades_limit,
    )

    if save_reports:
        write_walk_forward_json(report, reports_dir=reports_dir)
        write_walk_forward_markdown(report, reports_dir=reports_dir)

    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Walk-forward paper evaluation of bid/ask models with cost simulation "
            "(research-only, paper-only)."
        )
    )
    parser.add_argument("--coin", help="Single coin symbol, e.g. BTC")
    parser.add_argument(
        "--coins",
        nargs="+",
        help="Multiple coin symbols, e.g. BTC ETH SOL",
    )
    parser.add_argument(
        "--interval",
        default="15s",
        choices=sorted(SUPPORTED_INTERVALS),
        help="Resampled interval (default: 15s)",
    )
    parser.add_argument(
        "--source",
        default="local",
        choices=sorted(SUPPORTED_SOURCES),
        help="Data source: local, historical, or combined (default: local)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"Label horizon in bars (default: {DEFAULT_HORIZON})",
    )
    parser.add_argument(
        "--threshold-bps",
        type=float,
        default=DEFAULT_THRESHOLD_BPS,
        help=f"UP/DOWN label threshold in bps for direction mode (default: {DEFAULT_THRESHOLD_BPS})",
    )
    parser.add_argument(
        "--label-mode",
        default=DEFAULT_LABEL_MODE,
        choices=sorted(SUPPORTED_LABEL_MODES),
        help="Label mode: direction (threshold) or cost_aware (default: direction)",
    )
    parser.add_argument(
        "--label-cost-multiplier",
        type=float,
        default=DEFAULT_LABEL_COST_MULTIPLIER,
        help=(
            "Multiplier on estimated row cost when creating cost_aware labels "
            f"(default: {DEFAULT_LABEL_COST_MULTIPLIER})"
        ),
    )
    parser.add_argument(
        "--label-min-edge-bps",
        type=float,
        default=DEFAULT_LABEL_MIN_EDGE_BPS,
        help=(
            "Minimum edge above estimated row cost for cost_aware labels "
            f"(default: {DEFAULT_LABEL_MIN_EDGE_BPS})"
        ),
    )
    parser.add_argument(
        "--feature-set",
        default=DEFAULT_FEATURE_SET,
        choices=sorted(SUPPORTED_FEATURE_SETS),
        help="Feature set: basic (default) or microstructure",
    )
    parser.add_argument(
        "--train-bars",
        type=int,
        default=DEFAULT_TRAIN_BARS,
        help=f"Training window size in bars (default: {DEFAULT_TRAIN_BARS})",
    )
    parser.add_argument(
        "--test-bars",
        type=int,
        default=DEFAULT_TEST_BARS,
        help=f"Test window size in bars (default: {DEFAULT_TEST_BARS})",
    )
    parser.add_argument(
        "--step-bars",
        type=int,
        default=DEFAULT_STEP_BARS,
        help=f"Step size between windows in bars (default: {DEFAULT_STEP_BARS})",
    )
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=DEFAULT_FEE_BPS,
        help=f"Simulated fee in bps (default: {DEFAULT_FEE_BPS})",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=DEFAULT_SLIPPAGE_BPS,
        help=f"Simulated slippage in bps (default: {DEFAULT_SLIPPAGE_BPS})",
    )
    parser.add_argument(
        "--default-spread-bps",
        type=float,
        default=DEFAULT_SPREAD_BPS,
        help=f"Fallback spread cost in bps (default: {DEFAULT_SPREAD_BPS})",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=f"Minimum model confidence for paper signals (default: {DEFAULT_MIN_CONFIDENCE})",
    )
    parser.add_argument(
        "--min-edge-bps",
        type=float,
        default=DEFAULT_MIN_EDGE_BPS,
        help=f"Minimum edge above costs in bps (default: {DEFAULT_MIN_EDGE_BPS})",
    )
    parser.add_argument(
        "--max-spread-bps",
        type=float,
        default=None,
        help="Maximum spread in bps to allow paper trades (default: no limit)",
    )
    parser.add_argument(
        "--min-volatility-bps",
        type=float,
        default=None,
        help="Minimum rolling volatility in bps (default: no limit)",
    )
    parser.add_argument(
        "--max-volatility-bps",
        type=float,
        default=None,
        help="Maximum rolling volatility in bps (default: no limit)",
    )
    parser.add_argument(
        "--trade-classes",
        default="UP,DOWN",
        help="Comma-separated classes eligible for paper trades (default: UP,DOWN)",
    )
    parser.add_argument(
        "--cost-multiplier",
        type=float,
        default=DEFAULT_COST_MULTIPLIER,
        help=f"Multiplier on total cost when computing required edge (default: {DEFAULT_COST_MULTIPLIER})",
    )
    parser.add_argument(
        "--horizon-return-scale-bps",
        type=float,
        default=DEFAULT_HORIZON_RETURN_SCALE_BPS,
        help=(
            "Scale factor for confidence-edge proxy in bps "
            f"(default: {DEFAULT_HORIZON_RETURN_SCALE_BPS})"
        ),
    )
    parser.add_argument(
        "--min-absolute-future-label-bps",
        type=float,
        default=None,
        help="Alternative filter: minimum label/proxy edge in bps (default: disabled)",
    )
    parser.add_argument(
        "--min-windows",
        type=int,
        default=DEFAULT_MIN_WINDOWS,
        help=f"Minimum completed windows for readiness (default: {DEFAULT_MIN_WINDOWS})",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=DEFAULT_MIN_TRADES,
        help=f"Minimum paper trades for readiness (default: {DEFAULT_MIN_TRADES})",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing bid/ask CSV files",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory for walk-forward JSON and Markdown reports",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory for optional per-window model artifacts",
    )
    parser.add_argument(
        "--save-window-models",
        action="store_true",
        help="Save sklearn model artifact for each completed window",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows loaded per coin (research/debug)",
    )
    parser.add_argument(
        "--diagnostics-detail",
        default=DEFAULT_DIAGNOSTICS_DETAIL,
        choices=("basic", "full"),
        help="Diagnostics payload detail level (default: basic)",
    )
    parser.add_argument(
        "--save-accepted-trades",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save accepted filtered paper trades CSV when trades exist "
            "(default: true)"
        ),
    )
    parser.add_argument(
        "--accepted-trades-limit",
        type=int,
        default=DEFAULT_ACCEPTED_TRADES_LIMIT,
        help=f"Max accepted trades in CSV/full diagnostics (default: {DEFAULT_ACCEPTED_TRADES_LIMIT})",
    )
    parser.add_argument(
        "--stability-report",
        action="store_true",
        help="Include stability diagnostics from accepted paper trades (default: disabled)",
    )
    parser.add_argument(
        "--stability-min-trades-per-bucket",
        type=int,
        default=DEFAULT_STABILITY_MIN_TRADES_PER_BUCKET,
        help=(
            "Minimum trades per stability bucket before treating bucket as reliable "
            f"(default: {DEFAULT_STABILITY_MIN_TRADES_PER_BUCKET})"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    coins = resolve_coins(coin=args.coin, coins=args.coins)

    report = run_walk_forward_evaluation(
        coins=coins,
        interval=args.interval,
        source=args.source,
        horizon=args.horizon,
        threshold_bps=args.threshold_bps,
        label_mode=args.label_mode,
        label_cost_multiplier=args.label_cost_multiplier,
        label_min_edge_bps=args.label_min_edge_bps,
        feature_set=args.feature_set,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        default_spread_bps=args.default_spread_bps,
        min_confidence=args.min_confidence,
        min_edge_bps=args.min_edge_bps,
        max_spread_bps=args.max_spread_bps,
        min_volatility_bps=args.min_volatility_bps,
        max_volatility_bps=args.max_volatility_bps,
        trade_classes=parse_trade_classes(args.trade_classes),
        cost_multiplier=args.cost_multiplier,
        horizon_return_scale_bps=args.horizon_return_scale_bps,
        min_absolute_future_label_bps=args.min_absolute_future_label_bps,
        data_root=args.data_root,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        save_window_models=args.save_window_models,
        max_rows=args.max_rows,
        min_windows=args.min_windows,
        min_trades=args.min_trades,
        diagnostics_detail=args.diagnostics_detail,
        save_accepted_trades=args.save_accepted_trades,
        accepted_trades_limit=args.accepted_trades_limit,
        stability_report_enabled=args.stability_report,
        stability_min_trades_per_bucket=args.stability_min_trades_per_bucket,
    )

    readiness = report["readiness"]
    print(
        f"coins={','.join(coins)} interval={args.interval} source={report['source']} "
        f"horizon={args.horizon}"
    )
    print(
        f"windows={report['completed_windows']}/{report['total_windows']} "
        f"model_accuracy_mean={report['model_accuracy_mean']:.4f} "
        f"baseline_accuracy_mean={report['baseline_accuracy_mean']:.4f}"
    )
    print(
        f"cumulative_net_return_bps={report['cumulative_net_return_bps']:.4f} "
        f"beats_baseline={readiness['beats_baseline_accuracy']} "
        f"positive_after_costs={readiness['positive_after_costs']}"
    )
    if report.get("json_report_path"):
        print(f"json_report_path={report['json_report_path']}")
    if report.get("markdown_report_path"):
        print(f"markdown_report_path={report['markdown_report_path']}")
    if report.get("accepted_trades_csv_path"):
        print(f"accepted_trades_csv_path={report['accepted_trades_csv_path']}")
    print(f"accepted_trades_count={report.get('accepted_trades_count', 0)}")
    for warning in report.get("warnings", []):
        if warning != SAFETY_NOTE:
            print(f"warning: {warning}")
    print(SAFETY_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
