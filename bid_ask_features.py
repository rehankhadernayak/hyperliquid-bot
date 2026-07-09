"""Shared bid/ask feature engineering for training and walk-forward evaluation.

All feature sets are causal: rolling and lag features use only current/past rows.
Research-only and paper-only.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from bid_ask_resampler import INTERVAL_SECONDS

SUPPORTED_FEATURE_SETS = frozenset({"basic", "microstructure", "candidate_c_core"})

MID_PRICE_COLUMN = "close_mid"

CANDIDATE_C_CORE_COLUMNS = (
    "bbo_imbalance",
    "imbalance_mean_20",
    "imbalance_std_20",
    "imbalance_zscore_20",
    "imbalance_change_1",
    "spread_bps",
    "rolling_spread_std_20",
    "spread_zscore_20",
    "return_15s_bps",
    "return_30s_bps",
    "return_60s_bps",
    "short_term_drawdown_bps",
    "short_term_upmove_bps",
    "utc_hour",
    "hour_sin",
    "hour_cos",
    "is_weekend",
)

FUTURE_FEATURE_COLUMNS = frozenset(
    {
        "future_return_bps",
        "label",
    }
)

MIN_CANDIDATE_C_CORE_FEATURE_COUNT = 5


def parse_feature_set(value: str) -> str:
    """Parse feature set name."""
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


def _return_bps(mid: pd.Series, lag: int) -> pd.Series:
    prior = mid.shift(lag).replace(0, np.nan)
    return (mid - mid.shift(lag)) / prior * 10_000.0


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def _bars_for_seconds(interval: str, seconds: int) -> int:
    interval_seconds = INTERVAL_SECONDS[interval]
    return max(1, int(round(seconds / interval_seconds)))


def _safe_fill_series(series: pd.Series) -> pd.Series:
    """Fill NaNs from rolling calculations without using future rows."""
    return series.ffill().bfill().fillna(0.0)


def create_candidate_c_core_features(
    df: pd.DataFrame,
    *,
    interval: str = "15s",
) -> tuple[pd.DataFrame, list[str]]:
    """Build candidate_c_core features from resampled bid/ask bars."""
    if df.empty:
        return df.copy(), []

    out = df.copy()
    skipped: list[str] = []

    mid_column = resolve_mid_price_column(out)
    if mid_column is None:
        raise ValueError(
            "Missing mid price column (close_mid/avg_mid); "
            "required for candidate_c_core return features"
        )
    mid = out[mid_column].astype(float)

    spread_column = resolve_spread_column(out)
    if spread_column is None:
        skipped.extend(
            [
                "spread_bps",
                "rolling_spread_std_20",
                "spread_zscore_20",
            ]
        )
    else:
        spread = out[spread_column].astype(float)
        out["spread_bps"] = spread
        out["rolling_spread_std_20"] = spread.rolling(window=20, min_periods=20).std()
        out["spread_zscore_20"] = _rolling_zscore(spread, 20)

    imbalance_column = resolve_imbalance_column(out)
    if imbalance_column is None:
        skipped.extend(
            [
                "bbo_imbalance",
                "imbalance_mean_20",
                "imbalance_std_20",
                "imbalance_zscore_20",
                "imbalance_change_1",
            ]
        )
    else:
        imbalance = out[imbalance_column].astype(float)
        out["bbo_imbalance"] = imbalance
        out["imbalance_mean_20"] = imbalance.rolling(window=20, min_periods=20).mean()
        out["imbalance_std_20"] = imbalance.rolling(window=20, min_periods=20).std()
        out["imbalance_zscore_20"] = _rolling_zscore(imbalance, 20)
        out["imbalance_change_1"] = imbalance.diff(1)

    lag_15 = _bars_for_seconds(interval, 15)
    lag_30 = _bars_for_seconds(interval, 30)
    lag_60 = _bars_for_seconds(interval, 60)
    return_1 = _return_bps(mid, 1)
    out["return_15s_bps"] = _return_bps(mid, lag_15)
    out["return_30s_bps"] = _return_bps(mid, lag_30)
    out["return_60s_bps"] = _return_bps(mid, lag_60)
    out["short_term_drawdown_bps"] = return_1.rolling(window=10, min_periods=10).min()
    out["short_term_upmove_bps"] = return_1.rolling(window=10, min_periods=10).max()

    if "start_timestamp_ms" not in out.columns:
        raise ValueError(
            "Missing start_timestamp_ms; required for candidate_c_core time features"
        )
    ts = out["start_timestamp_ms"].astype("int64")
    dt = pd.to_datetime(ts, unit="ms", utc=True)
    out["utc_hour"] = dt.dt.hour.astype(float)
    out["hour_sin"] = np.sin(2.0 * math.pi * out["utc_hour"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * math.pi * out["utc_hour"] / 24.0)
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)

    for column in CANDIDATE_C_CORE_COLUMNS:
        if column in out.columns:
            out[column] = _safe_fill_series(out[column])

    return out, sorted(set(skipped))


def build_candidate_c_core_metadata(
    *,
    feature_columns: tuple[str, ...],
    skipped_candidate_c_core_features: list[str] | None = None,
) -> dict[str, Any]:
    """Build report metadata for candidate_c_core feature selection."""
    columns = list(feature_columns)
    skipped = list(skipped_candidate_c_core_features or [])
    metadata: dict[str, Any] = {
        "feature_set": "candidate_c_core",
        "feature_columns": columns,
        "feature_count": len(columns),
        "skipped_microstructure_features": [],
        "skipped_candidate_c_core_features": skipped,
        "feature_warning": None,
    }
    if len(columns) < MIN_CANDIDATE_C_CORE_FEATURE_COUNT:
        metadata["feature_warning"] = (
            f"too_few_features: only {len(columns)} columns available "
            f"(expected at least {MIN_CANDIDATE_C_CORE_FEATURE_COUNT})"
        )
    return metadata


def assert_no_future_feature_columns(feature_columns: list[str] | tuple[str, ...]) -> None:
    """Raise if any forbidden future-looking columns are selected as features."""
    forbidden = [column for column in feature_columns if column in FUTURE_FEATURE_COLUMNS]
    if forbidden:
        raise ValueError(f"Future-looking columns cannot be features: {forbidden}")
    for column in feature_columns:
        if column.startswith("future_return"):
            raise ValueError(f"Future-looking columns cannot be features: {column}")
