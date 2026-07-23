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

SUPPORTED_FEATURE_SETS = frozenset(
    {
        "basic",
        "microstructure",
        "candidate_c_core",
        "candidate_e_depth_aware",
        "candidate_f_regime_depth",
    }
)

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

CANDIDATE_E_DEPTH_AWARE_EXTRA_COLUMNS = (
    "bid_size",
    "ask_size",
    "total_top_size",
    "size_imbalance",
    "microprice",
    "microprice_mid_deviation_bps",
    "rolling_pressure_change",
)

CANDIDATE_E_DEPTH_AWARE_COLUMNS = CANDIDATE_C_CORE_COLUMNS + CANDIDATE_E_DEPTH_AWARE_EXTRA_COLUMNS

CANDIDATE_F_REGIME_DEPTH_EXTRA_COLUMNS = (
    "realized_volatility_5m_bps",
    "realized_volatility_15m_bps",
    "volatility_ratio_5m_15m",
    "spread_volatility_interaction",
    "microprice_volatility_interaction",
    "imbalance_volatility_interaction",
    "low_volatility_regime",
    "high_volatility_regime",
    "tight_spread_regime",
    "wide_spread_regime",
)

CANDIDATE_F_REGIME_DEPTH_COLUMNS = (
    CANDIDATE_E_DEPTH_AWARE_COLUMNS + CANDIDATE_F_REGIME_DEPTH_EXTRA_COLUMNS
)

REGIME_QUANTILE_WINDOW = 240
REGIME_QUANTILE_MIN_PERIODS = 120
REALIZED_VOL_5M_WINDOW = 20
REALIZED_VOL_15M_WINDOW = 60

REGIME_BREAKDOWN_COLUMNS = (
    "low_volatility_regime",
    "high_volatility_regime",
    "tight_spread_regime",
    "wide_spread_regime",
)

VOLATILITY_REGIME_COLUMNS = (
    "low_volatility_regime",
    "high_volatility_regime",
)

SPREAD_REGIME_COLUMNS = (
    "tight_spread_regime",
    "wide_spread_regime",
)

ACCEPTED_TRADES_CANDIDATE_F_DIAGNOSTIC_COLUMNS = CANDIDATE_F_REGIME_DEPTH_EXTRA_COLUMNS

FUTURE_FEATURE_COLUMNS = frozenset(
    {
        "future_return_bps",
        "label",
    }
)

MIN_CANDIDATE_C_CORE_FEATURE_COUNT = 5
MIN_CANDIDATE_E_DEPTH_AWARE_FEATURE_COUNT = 5
MIN_CANDIDATE_F_REGIME_DEPTH_FEATURE_COUNT = 5
REQUIRED_DEPTH_SIZE_COLUMNS = ("bid_size", "ask_size")


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


def _require_depth_size_columns(df: pd.DataFrame, *, feature_set: str = "candidate_e_depth_aware") -> None:
    """Raise if bid_size/ask_size are unavailable for depth-aware features."""
    missing = [column for column in REQUIRED_DEPTH_SIZE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Missing bid_size/ask_size columns required for "
            f"{feature_set} depth features: {missing}"
        )

    bid_size = df["bid_size"]
    ask_size = df["ask_size"]
    if bid_size.isna().all() or ask_size.isna().all():
        raise ValueError(
            "bid_size/ask_size contain no usable data; "
            f"{feature_set} requires resampled depth columns"
        )


def _require_candidate_f_base_columns(df: pd.DataFrame) -> None:
    """Raise if spread/imbalance columns are unavailable for Candidate F regime features."""
    _require_depth_size_columns(df, feature_set="candidate_f_regime_depth")

    if resolve_spread_column(df) is None:
        raise ValueError(
            "Missing spread column (avg_spread_bps/spread_bps) required for "
            "candidate_f_regime_depth spread and regime features"
        )
    if resolve_imbalance_column(df) is None:
        raise ValueError(
            "Missing imbalance column (avg_bbo_imbalance/bbo_imbalance) required for "
            "candidate_f_regime_depth imbalance interaction features"
        )


def _resolve_bid_ask_price_columns(df: pd.DataFrame) -> tuple[str, str]:
    bid_column = "avg_bid" if "avg_bid" in df.columns else "best_bid"
    ask_column = "avg_ask" if "avg_ask" in df.columns else "best_ask"
    missing = [column for column in (bid_column, ask_column) if column not in df.columns]
    if missing:
        raise ValueError(
            "Missing bid/ask price columns required for "
            f"candidate_e_depth_aware microprice features: {missing}"
        )
    return bid_column, ask_column


def _compute_microprice(
    bid: pd.Series,
    ask: pd.Series,
    bid_size: pd.Series,
    ask_size: pd.Series,
) -> pd.Series:
    total = bid_size.astype(float) + ask_size.astype(float)
    numerator = ask.astype(float) * bid_size.astype(float) + bid.astype(float) * ask_size.astype(float)
    return numerator / total.replace(0, np.nan)


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


def create_candidate_e_depth_aware_features(
    df: pd.DataFrame,
    *,
    interval: str = "15s",
) -> tuple[pd.DataFrame, list[str]]:
    """Build candidate_e_depth_aware features from resampled bid/ask bars."""
    if df.empty:
        return df.copy(), []

    _require_depth_size_columns(df)
    out, skipped = create_candidate_c_core_features(df, interval=interval)

    bid_size = out["bid_size"].astype(float)
    ask_size = out["ask_size"].astype(float)
    out["bid_size"] = bid_size
    out["ask_size"] = ask_size
    out["total_top_size"] = bid_size + ask_size
    total = out["total_top_size"].replace(0, np.nan)
    out["size_imbalance"] = (bid_size - ask_size) / total

    bid_column, ask_column = _resolve_bid_ask_price_columns(out)
    mid_column = resolve_mid_price_column(out)
    if mid_column is None:
        raise ValueError(
            "Missing mid price column (close_mid/avg_mid); "
            "required for candidate_e_depth_aware microprice features"
        )
    mid = out[mid_column].astype(float)
    out["microprice"] = _compute_microprice(
        out[bid_column],
        out[ask_column],
        bid_size,
        ask_size,
    )
    out["microprice_mid_deviation_bps"] = (
        (out["microprice"] - mid) / mid.replace(0, np.nan) * 10_000.0
    )
    out["rolling_pressure_change"] = out["microprice_mid_deviation_bps"].diff(5)

    for column in CANDIDATE_E_DEPTH_AWARE_COLUMNS:
        if column in out.columns:
            out[column] = _safe_fill_series(out[column])

    return out, sorted(set(skipped))


def _rolling_quantile(series: pd.Series, quantile: float, window: int, min_periods: int) -> pd.Series:
    return series.rolling(window=window, min_periods=min_periods).quantile(quantile)


def create_candidate_f_regime_depth_features(
    df: pd.DataFrame,
    *,
    interval: str = "15s",
) -> tuple[pd.DataFrame, list[str]]:
    """Build candidate_f_regime_depth features from resampled bid/ask bars."""
    if df.empty:
        return df.copy(), []

    _require_candidate_f_base_columns(df)
    out, skipped = create_candidate_e_depth_aware_features(df, interval=interval)

    mid_column = resolve_mid_price_column(out)
    if mid_column is None:
        raise ValueError(
            "Missing mid price column (close_mid/avg_mid); "
            "required for candidate_f_regime_depth volatility features"
        )
    mid = out[mid_column].astype(float)
    return_1 = _return_bps(mid, 1)

    out["realized_volatility_5m_bps"] = return_1.rolling(
        window=REALIZED_VOL_5M_WINDOW,
        min_periods=REALIZED_VOL_5M_WINDOW,
    ).std()
    out["realized_volatility_15m_bps"] = return_1.rolling(
        window=REALIZED_VOL_15M_WINDOW,
        min_periods=REALIZED_VOL_15M_WINDOW,
    ).std()
    out["volatility_ratio_5m_15m"] = out["realized_volatility_5m_bps"] / out[
        "realized_volatility_15m_bps"
    ].replace(0, np.nan)

    if "spread_bps" not in out.columns:
        raise ValueError(
            "Missing spread_bps required for candidate_f_regime_depth interaction/regime features"
        )
    if "bbo_imbalance" not in out.columns:
        raise ValueError(
            "Missing bbo_imbalance required for candidate_f_regime_depth interaction features"
        )
    if "microprice_mid_deviation_bps" not in out.columns:
        raise ValueError(
            "Missing microprice_mid_deviation_bps required for "
            "candidate_f_regime_depth interaction features"
        )

    spread = out["spread_bps"].astype(float)
    vol_5m = out["realized_volatility_5m_bps"]
    out["spread_volatility_interaction"] = spread * vol_5m
    out["microprice_volatility_interaction"] = out["microprice_mid_deviation_bps"] * vol_5m
    out["imbalance_volatility_interaction"] = out["bbo_imbalance"] * vol_5m

    vol_q25 = _rolling_quantile(
        vol_5m,
        0.25,
        REGIME_QUANTILE_WINDOW,
        REGIME_QUANTILE_MIN_PERIODS,
    )
    vol_q75 = _rolling_quantile(
        vol_5m,
        0.75,
        REGIME_QUANTILE_WINDOW,
        REGIME_QUANTILE_MIN_PERIODS,
    )
    spread_q25 = _rolling_quantile(
        spread,
        0.25,
        REGIME_QUANTILE_WINDOW,
        REGIME_QUANTILE_MIN_PERIODS,
    )
    spread_q75 = _rolling_quantile(
        spread,
        0.75,
        REGIME_QUANTILE_WINDOW,
        REGIME_QUANTILE_MIN_PERIODS,
    )

    out["low_volatility_regime"] = (vol_5m < vol_q25).astype(float)
    out["high_volatility_regime"] = (vol_5m > vol_q75).astype(float)
    out["tight_spread_regime"] = (spread < spread_q25).astype(float)
    out["wide_spread_regime"] = (spread > spread_q75).astype(float)

    for column in CANDIDATE_F_REGIME_DEPTH_COLUMNS:
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


def build_candidate_e_depth_aware_metadata(
    *,
    feature_columns: tuple[str, ...],
    skipped_candidate_c_core_features: list[str] | None = None,
) -> dict[str, Any]:
    """Build report metadata for candidate_e_depth_aware feature selection."""
    columns = list(feature_columns)
    skipped = list(skipped_candidate_c_core_features or [])
    metadata: dict[str, Any] = {
        "feature_set": "candidate_e_depth_aware",
        "feature_columns": columns,
        "feature_count": len(columns),
        "skipped_microstructure_features": [],
        "skipped_candidate_c_core_features": skipped,
        "feature_warning": None,
    }
    if len(columns) < MIN_CANDIDATE_E_DEPTH_AWARE_FEATURE_COUNT:
        metadata["feature_warning"] = (
            f"too_few_features: only {len(columns)} columns available "
            f"(expected at least {MIN_CANDIDATE_E_DEPTH_AWARE_FEATURE_COUNT})"
        )
    return metadata


def build_candidate_f_regime_depth_metadata(
    *,
    feature_columns: tuple[str, ...],
    skipped_candidate_c_core_features: list[str] | None = None,
) -> dict[str, Any]:
    """Build report metadata for candidate_f_regime_depth feature selection."""
    columns = list(feature_columns)
    skipped = list(skipped_candidate_c_core_features or [])
    metadata: dict[str, Any] = {
        "feature_set": "candidate_f_regime_depth",
        "feature_columns": columns,
        "feature_count": len(columns),
        "skipped_microstructure_features": [],
        "skipped_candidate_c_core_features": skipped,
        "feature_warning": None,
    }
    if len(columns) < MIN_CANDIDATE_F_REGIME_DEPTH_FEATURE_COUNT:
        metadata["feature_warning"] = (
            f"too_few_features: only {len(columns)} columns available "
            f"(expected at least {MIN_CANDIDATE_F_REGIME_DEPTH_FEATURE_COUNT})"
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
