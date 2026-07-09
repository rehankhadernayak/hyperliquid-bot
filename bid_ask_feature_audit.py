"""Read-only Candidate C feature audit for resampled bid/ask CSVs.

Paper-only and research-only. Reads local resampled CSV files and writes audit
reports under reports/feature_audit/. No execution, orders, wallets, AWS, network,
or shell commands.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bid_ask_resampler import (
    INTERVAL_SECONDS,
    REQUIRED_RESAMPLED_COLUMNS,
    RESAMPLED_COLUMNS,
    normalize_source,
    resampled_csv_path,
)
from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT

DEFAULT_DATA_ROOT = Path("data/bid_ask_csv")
DEFAULT_OUTPUT_DIR = Path("reports/feature_audit")
DEFAULT_REPORTS_ROOT = Path("reports")
DEFAULT_WALK_FORWARD_DIR = Path("reports/bid_ask_walk_forward")

SUPPORTED_INTERVALS = frozenset({"15s", "1m"})
DEFAULT_HORIZONS = (1, 5, 10, 20)
DEFAULT_THRESHOLD_BPS = 1.0
DEFAULT_FEE_BPS = 2.0
DEFAULT_SLIPPAGE_BPS = 1.0
DEFAULT_WARMUP_BARS = 20
QUANTILE_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)

MID_PRICE_COLUMN = "close_mid"

FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "spread_cost_regime": (
        "spread_bps",
        "rolling_spread_mean_20",
        "rolling_spread_std_20",
        "spread_zscore_20",
        "estimated_cost_bps",
    ),
    "order_book_imbalance": (
        "bbo_imbalance",
        "imbalance_mean_20",
        "imbalance_std_20",
        "imbalance_zscore_20",
        "imbalance_change_1",
    ),
    "microprice_pressure": (
        "microprice",
        "microprice_mid_deviation_bps",
        "rolling_pressure_change",
    ),
    "volatility_momentum": (
        "return_15s_bps",
        "return_30s_bps",
        "return_60s_bps",
        "realized_volatility_20",
        "short_term_drawdown_bps",
        "short_term_upmove_bps",
    ),
    "time_session": (
        "utc_hour",
        "hour_sin",
        "hour_cos",
        "is_weekend",
    ),
    "liquidity_size": (
        "bid_size",
        "ask_size",
        "total_top_size",
        "size_imbalance",
    ),
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _report_stamp(generated_at: str | None = None) -> str:
    raw = generated_at or _utc_now_iso()
    return raw.replace("-", "").replace(":", "").replace(".", "")


def build_safety_flags() -> dict[str, Any]:
    return {
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "real_execution_ready": False,
        "read_only": True,
        "no_repairs_performed": True,
        "no_network_calls": True,
        "no_subprocess_calls": True,
        "research_only": True,
        "safety_statement": SAFETY_STATEMENT,
    }


def load_resampled_csv(
    coin: str,
    interval: str,
    data_root: Path,
    *,
    source: str,
) -> tuple[pd.DataFrame, Path]:
    if interval not in SUPPORTED_INTERVALS:
        supported = ", ".join(sorted(SUPPORTED_INTERVALS))
        raise ValueError(f"Unsupported interval {interval!r}; use one of: {supported}")

    normalized_source = normalize_source(source)
    path = resampled_csv_path(coin, interval, data_root, source=normalized_source)
    if not path.is_file():
        raise FileNotFoundError(f"Resampled CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_RESAMPLED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df.sort_values("start_timestamp_ms").reset_index(drop=True)
    return df, path


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


def _compute_microprice(
    bid: pd.Series,
    ask: pd.Series,
    bid_size: pd.Series,
    ask_size: pd.Series,
) -> pd.Series:
    total = bid_size.astype(float) + ask_size.astype(float)
    numerator = ask.astype(float) * bid_size.astype(float) + bid.astype(float) * ask_size.astype(float)
    return numerator / total.replace(0, np.nan)


def compute_candidate_features(
    df: pd.DataFrame,
    *,
    interval: str,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> tuple[pd.DataFrame, list[str]]:
    """Build Candidate C audit features from a resampled bid/ask frame."""
    if df.empty:
        return df.copy(), []

    out = df.copy()
    skipped: list[str] = []
    mid = out[MID_PRICE_COLUMN].astype(float)

    if "avg_spread_bps" in out.columns:
        spread = out["avg_spread_bps"].astype(float)
        out["spread_bps"] = spread
        out["rolling_spread_mean_20"] = spread.rolling(window=20, min_periods=20).mean()
        out["rolling_spread_std_20"] = spread.rolling(window=20, min_periods=20).std()
        out["spread_zscore_20"] = _rolling_zscore(spread, 20)
        out["estimated_cost_bps"] = fee_bps + slippage_bps + spread
    else:
        skipped.extend(FEATURE_FAMILIES["spread_cost_regime"])

    if "avg_bbo_imbalance" in out.columns:
        imbalance = out["avg_bbo_imbalance"].astype(float)
        out["bbo_imbalance"] = imbalance
        out["imbalance_mean_20"] = imbalance.rolling(window=20, min_periods=20).mean()
        out["imbalance_std_20"] = imbalance.rolling(window=20, min_periods=20).std()
        out["imbalance_zscore_20"] = _rolling_zscore(imbalance, 20)
        out["imbalance_change_1"] = imbalance.diff(1)
    else:
        skipped.extend(FEATURE_FAMILIES["order_book_imbalance"])

    size_columns = {"bid_size", "ask_size"}
    bid_col = "avg_bid" if "avg_bid" in out.columns else "best_bid"
    ask_col = "avg_ask" if "avg_ask" in out.columns else "best_ask"
    if size_columns.issubset(out.columns) and bid_col in out.columns and ask_col in out.columns:
        bid_size = out["bid_size"].astype(float)
        ask_size = out["ask_size"].astype(float)
        out["microprice"] = _compute_microprice(out[bid_col], out[ask_col], bid_size, ask_size)
        out["microprice_mid_deviation_bps"] = (
            (out["microprice"] - mid) / mid.replace(0, np.nan) * 10_000.0
        )
        out["rolling_pressure_change"] = out["microprice_mid_deviation_bps"].diff(5)
    else:
        skipped.extend(FEATURE_FAMILIES["microprice_pressure"])

    lag_15 = _bars_for_seconds(interval, 15)
    lag_30 = _bars_for_seconds(interval, 30)
    lag_60 = _bars_for_seconds(interval, 60)
    return_1 = _return_bps(mid, 1)
    out["return_15s_bps"] = _return_bps(mid, lag_15)
    out["return_30s_bps"] = _return_bps(mid, lag_30)
    out["return_60s_bps"] = _return_bps(mid, lag_60)
    out["realized_volatility_20"] = return_1.rolling(window=20, min_periods=20).std()
    out["short_term_drawdown_bps"] = return_1.rolling(window=10, min_periods=10).min()
    out["short_term_upmove_bps"] = return_1.rolling(window=10, min_periods=10).max()

    ts = out["start_timestamp_ms"].astype("int64")
    dt = pd.to_datetime(ts, unit="ms", utc=True)
    out["utc_hour"] = dt.dt.hour.astype(float)
    out["hour_sin"] = np.sin(2.0 * math.pi * out["utc_hour"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * math.pi * out["utc_hour"] / 24.0)
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)

    if size_columns.issubset(out.columns):
        bid_size = out["bid_size"].astype(float)
        ask_size = out["ask_size"].astype(float)
        out["total_top_size"] = bid_size + ask_size
        total = out["total_top_size"].replace(0, np.nan)
        out["size_imbalance"] = (bid_size - ask_size) / total
    else:
        skipped.extend(FEATURE_FAMILIES["liquidity_size"])

    return out, sorted(set(skipped))


def add_future_returns(
    df: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    out = df.copy()
    mid = out[MID_PRICE_COLUMN].astype(float)
    for horizon in horizons:
        future_mid = mid.shift(-horizon)
        column = f"future_return_h{horizon}_bps"
        out[column] = (future_mid - mid) / mid.replace(0, np.nan) * 10_000.0
    out["future_return_bps"] = out.get(
        f"future_return_h{horizons[0]}_bps",
        pd.Series(np.nan, index=out.index),
    )
    return out


def add_labels(
    df: pd.DataFrame,
    *,
    horizon: int = 1,
    threshold_bps: float = DEFAULT_THRESHOLD_BPS,
) -> pd.DataFrame:
    out = df.copy()
    column = f"future_return_h{horizon}_bps"
    if column not in out.columns:
        out = add_future_returns(out, horizons=(horizon,))
    future = out[column]
    labels = np.full(len(out), "FLAT", dtype=object)
    labels[future > threshold_bps] = "UP"
    labels[future < -threshold_bps] = "DOWN"
    out["label"] = labels
    return out


def _finite_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _safe_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def compute_feature_stats(series: pd.Series) -> dict[str, Any]:
    values = _finite_series(series)
    total = len(values)
    missing = int(values.isna().sum())
    valid = values.dropna()
    payload: dict[str, Any] = {
        "missingness": _safe_float(missing / total) if total else None,
        "count": int(valid.count()),
        "min": _safe_float(valid.min()) if not valid.empty else None,
        "median": _safe_float(valid.median()) if not valid.empty else None,
        "max": _safe_float(valid.max()) if not valid.empty else None,
        "mean": _safe_float(valid.mean()) if not valid.empty else None,
        "std": _safe_float(valid.std()) if not valid.empty else None,
        "quantiles": {},
    }
    for level in QUANTILE_LEVELS:
        key = f"p{int(level * 100):02d}"
        payload["quantiles"][key] = (
            _safe_float(valid.quantile(level)) if not valid.empty else None
        )
    return payload


def compute_feature_correlations(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    warmup_bars: int = DEFAULT_WARMUP_BARS,
) -> dict[str, dict[str, float | None]]:
    work = df.iloc[warmup_bars:].copy() if warmup_bars > 0 else df.copy()
    correlations: dict[str, dict[str, float | None]] = {}
    for feature in feature_columns:
        if feature not in work.columns:
            continue
        feature_values = _finite_series(work[feature])
        entry: dict[str, float | None] = {}
        for horizon in horizons:
            target_column = f"future_return_h{horizon}_bps"
            if target_column not in work.columns:
                continue
            target = _finite_series(work[target_column])
            aligned = pd.concat([feature_values, target], axis=1).dropna()
            if len(aligned) < 3:
                entry[f"h{horizon}"] = None
                continue
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            entry[f"h{horizon}"] = _safe_float(corr)
        correlations[feature] = entry
    return correlations


def _tercile_labels(series: pd.Series) -> pd.Series:
    values = _finite_series(series)
    if values.dropna().empty:
        return pd.Series(["unknown"] * len(series), index=series.index)
    try:
        buckets = pd.qcut(values, q=3, labels=["low", "mid", "high"], duplicates="drop")
    except ValueError:
        return pd.Series(["unknown"] * len(series), index=series.index)
    return buckets.astype(str).fillna("unknown")


def _distribution_summary(series: pd.Series) -> dict[str, float | None]:
    values = _finite_series(series).dropna()
    if values.empty:
        return {"count": 0, "mean": None, "std": None, "median": None}
    return {
        "count": int(values.count()),
        "mean": _safe_float(values.mean()),
        "std": _safe_float(values.std()),
        "median": _safe_float(values.median()),
    }


def _class_balance(labels: pd.Series) -> dict[str, float | None]:
    total = len(labels)
    if total == 0:
        return {}
    counts = labels.value_counts(normalize=True)
    return {str(label): _safe_float(float(counts.get(label, 0.0))) for label in ("UP", "DOWN", "FLAT")}


def build_label_diagnostics(
    df: pd.DataFrame,
    *,
    horizon: int = 1,
    warmup_bars: int = DEFAULT_WARMUP_BARS,
) -> dict[str, Any]:
    work = df.iloc[warmup_bars:].copy() if warmup_bars > 0 else df.copy()
    target_column = f"future_return_h{horizon}_bps"
    if target_column not in work.columns:
        work = add_future_returns(work, horizons=(horizon,))
    if "label" not in work.columns:
        work = add_labels(work, horizon=horizon)

    diagnostics: dict[str, Any] = {
        "horizon_bars": horizon,
        "overall_future_return": _distribution_summary(work[target_column]),
        "overall_class_balance": _class_balance(work["label"]),
        "by_utc_hour": {},
        "by_spread_tercile": {},
        "by_imbalance_tercile": {},
    }

    for hour, group in work.groupby("utc_hour", dropna=True):
        diagnostics["by_utc_hour"][str(int(hour))] = {
            "future_return": _distribution_summary(group[target_column]),
            "class_balance": _class_balance(group["label"]),
            "count": int(len(group)),
        }

    if "spread_bps" in work.columns:
        work = work.copy()
        work["_spread_tercile"] = _tercile_labels(work["spread_bps"])
        for bucket, group in work.groupby("_spread_tercile", dropna=True):
            diagnostics["by_spread_tercile"][str(bucket)] = {
                "future_return": _distribution_summary(group[target_column]),
                "class_balance": _class_balance(group["label"]),
                "count": int(len(group)),
            }

    if "bbo_imbalance" in work.columns:
        work = work.copy()
        work["_imbalance_tercile"] = _tercile_labels(work["bbo_imbalance"])
        for bucket, group in work.groupby("_imbalance_tercile", dropna=True):
            diagnostics["by_imbalance_tercile"][str(bucket)] = {
                "future_return": _distribution_summary(group[target_column]),
                "class_balance": _class_balance(group["label"]),
                "count": int(len(group)),
            }

    return diagnostics


def _find_latest_accepted_trades_csv(
    coin: str,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
) -> Path | None:
    walk_forward_dir = reports_root / DEFAULT_WALK_FORWARD_DIR.name
    if not walk_forward_dir.is_dir():
        walk_forward_dir = DEFAULT_WALK_FORWARD_DIR
    if not walk_forward_dir.is_dir():
        return None

    candidates = sorted(walk_forward_dir.glob(f"accepted_trades_{coin}_*.csv"))
    if not candidates:
        candidates = sorted(walk_forward_dir.glob(f"accepted_trades_*{coin}*.csv"))
    if not candidates:
        return None
    return candidates[-1]


def build_accepted_trades_diagnostics(
    csv_path: Path | None,
) -> dict[str, Any] | None:
    if csv_path is None or not csv_path.is_file():
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return {"csv_path": str(csv_path), "trade_count": 0, "by_spread_tercile": {}, "by_hour": {}}

    net_column = "net_return_bps" if "net_return_bps" in df.columns else None
    spread_column = "spread_bps" if "spread_bps" in df.columns else None
    hour_column = None
    if "timestamp_iso" in df.columns:
        hours = pd.to_datetime(df["timestamp_iso"], utc=True, errors="coerce").dt.hour
        df = df.copy()
        df["_utc_hour"] = hours
        hour_column = "_utc_hour"

    payload: dict[str, Any] = {
        "csv_path": str(csv_path),
        "trade_count": int(len(df)),
        "net_return_summary": _distribution_summary(df[net_column]) if net_column else None,
        "by_spread_tercile": {},
        "by_hour": {},
    }

    if spread_column and net_column:
        df = df.copy()
        df["_spread_tercile"] = _tercile_labels(df[spread_column])
        for bucket, group in df.groupby("_spread_tercile", dropna=True):
            payload["by_spread_tercile"][str(bucket)] = {
                "count": int(len(group)),
                "net_return": _distribution_summary(group[net_column]),
            }

    if hour_column and net_column:
        for hour, group in df.groupby(hour_column, dropna=True):
            payload["by_hour"][str(int(hour))] = {
                "count": int(len(group)),
                "net_return": _distribution_summary(group[net_column]),
            }

    return payload


def _all_audit_feature_columns() -> list[str]:
    columns: list[str] = []
    for family_columns in FEATURE_FAMILIES.values():
        columns.extend(family_columns)
    deduped: list[str] = []
    for column in columns:
        if column not in deduped:
            deduped.append(column)
    return deduped


def build_feature_audit_report(
    *,
    coin: str,
    interval: str,
    source: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    threshold_bps: float = DEFAULT_THRESHOLD_BPS,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    warmup_bars: int = DEFAULT_WARMUP_BARS,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or _utc_now_iso()
    raw_df, csv_path = load_resampled_csv(coin, interval, data_root, source=source)
    featured_df, skipped_features = compute_candidate_features(
        raw_df,
        interval=interval,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    featured_df = add_future_returns(featured_df, horizons=horizons)
    featured_df = add_labels(featured_df, horizon=horizons[0], threshold_bps=threshold_bps)

    audit_columns = [
        column
        for column in _all_audit_feature_columns()
        if column in featured_df.columns and column not in skipped_features
    ]

    feature_stats = {
        column: compute_feature_stats(featured_df[column]) for column in audit_columns
    }
    feature_correlations = compute_feature_correlations(
        featured_df,
        audit_columns,
        horizons=horizons,
        warmup_bars=warmup_bars,
    )

    accepted_trades_path = _find_latest_accepted_trades_csv(coin, reports_root=reports_root)
    accepted_trades = build_accepted_trades_diagnostics(accepted_trades_path)

    start_ts = int(raw_df["start_timestamp_ms"].iloc[0]) if not raw_df.empty else None
    end_ts = int(raw_df["start_timestamp_ms"].iloc[-1]) if not raw_df.empty else None

    return {
        "generated_at": generated,
        "coin": coin,
        "interval": interval,
        "source": normalize_source(source),
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "real_execution_ready": False,
        "safety_statement": SAFETY_STATEMENT,
        "safety_flags": build_safety_flags(),
        "input_csv_path": str(csv_path),
        "row_count": int(len(raw_df)),
        "time_range": {
            "start_timestamp_ms": start_ts,
            "end_timestamp_ms": end_ts,
        },
        "horizons_bars": list(horizons),
        "threshold_bps": threshold_bps,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "warmup_bars": warmup_bars,
        "feature_families": {
            family: [column for column in columns if column in audit_columns]
            for family, columns in FEATURE_FAMILIES.items()
        },
        "skipped_features": skipped_features,
        "feature_stats": feature_stats,
        "feature_correlations": feature_correlations,
        "label_diagnostics": build_label_diagnostics(
            featured_df,
            horizon=horizons[0],
            warmup_bars=warmup_bars,
        ),
        "accepted_trades_diagnostics": accepted_trades,
    }


def write_feature_audit_json(
    report: dict[str, Any],
    output_dir: Path,
    *,
    json_name: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / json_name
    output_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    report["json_report_path"] = str(output_path)
    return output_path


def write_feature_audit_markdown(
    report: dict[str, Any],
    output_dir: Path,
    *,
    markdown_name: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / markdown_name

    lines = [
        "# Bid/Ask Candidate C Feature Audit",
        "",
        f"- generated_at: {report.get('generated_at')}",
        f"- coin: {report.get('coin')}",
        f"- interval: {report.get('interval')}",
        f"- source: {report.get('source')}",
        f"- row_count: {report.get('row_count')}",
        f"- paper_only: {report.get('paper_only')}",
        f"- execution_enabled: {report.get('execution_enabled')}",
        f"- real_execution_ready: {report.get('real_execution_ready')}",
        "",
        "## Safety",
        "",
        str(report.get("safety_statement", "")),
        "",
        "## Input",
        "",
        f"- csv: `{report.get('input_csv_path')}`",
        f"- horizons (bars): {report.get('horizons_bars')}",
        f"- warmup_bars: {report.get('warmup_bars')}",
        "",
        "## Skipped Features",
        "",
    ]

    skipped = report.get("skipped_features") or []
    if skipped:
        for feature in skipped:
            lines.append(f"- {feature}")
    else:
        lines.append("- None")

    lines.extend(["", "## Feature Summary", ""])
    for feature, stats in (report.get("feature_stats") or {}).items():
        lines.append(f"### {feature}")
        lines.append(f"- missingness: {stats.get('missingness')}")
        lines.append(f"- min / median / max: {stats.get('min')} / {stats.get('median')} / {stats.get('max')}")
        quantiles = stats.get("quantiles") or {}
        if quantiles:
            q_text = ", ".join(f"{key}={value}" for key, value in quantiles.items())
            lines.append(f"- quantiles: {q_text}")
        correlations = (report.get("feature_correlations") or {}).get(feature) or {}
        if correlations:
            corr_text = ", ".join(f"{key}={value}" for key, value in correlations.items())
            lines.append(f"- correlations: {corr_text}")
        lines.append("")

    lines.extend(["## Label Diagnostics", ""])
    label_diag = report.get("label_diagnostics") or {}
    overall = label_diag.get("overall_future_return") or {}
    lines.append(
        f"- overall future return: count={overall.get('count')} "
        f"mean={overall.get('mean')} std={overall.get('std')}"
    )
    balance = label_diag.get("overall_class_balance") or {}
    if balance:
        balance_text = ", ".join(f"{key}={value}" for key, value in balance.items())
        lines.append(f"- class balance: {balance_text}")

    accepted = report.get("accepted_trades_diagnostics")
    lines.extend(["", "## Accepted Trades Diagnostics", ""])
    if accepted:
        lines.append(f"- csv: `{accepted.get('csv_path')}`")
        lines.append(f"- trade_count: {accepted.get('trade_count')}")
        net_summary = accepted.get("net_return_summary") or {}
        lines.append(
            f"- net return: count={net_summary.get('count')} "
            f"mean={net_summary.get('mean')} std={net_summary.get('std')}"
        )
    else:
        lines.append("- No accepted-trades CSV found")

    lines.extend(["", "## Safety Flags", ""])
    for key, value in (report.get("safety_flags") or {}).items():
        lines.append(f"- {key}: {value}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["markdown_report_path"] = str(output_path)
    return output_path


def run_feature_audit(
    *,
    coin: str,
    interval: str,
    source: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    threshold_bps: float = DEFAULT_THRESHOLD_BPS,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    warmup_bars: int = DEFAULT_WARMUP_BARS,
    json_name: str | None = None,
    markdown_name: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or _utc_now_iso()
    stamp = _report_stamp(generated)
    coin_upper = coin.upper()
    json_filename = json_name or f"feature_audit_{coin_upper}_{interval}_{stamp}.json"
    md_filename = markdown_name or f"feature_audit_{coin_upper}_{interval}_{stamp}.md"

    report = build_feature_audit_report(
        coin=coin_upper,
        interval=interval,
        source=source,
        data_root=data_root,
        reports_root=reports_root,
        horizons=horizons,
        threshold_bps=threshold_bps,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        warmup_bars=warmup_bars,
        generated_at=generated,
    )

    json_path = write_feature_audit_json(report, output_dir, json_name=json_filename)
    md_path = write_feature_audit_markdown(report, output_dir, markdown_name=md_filename)
    report["output_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Candidate C features on resampled bid/ask CSVs "
            "(paper-only, research-only, read-only)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Run feature audit and write reports")
    audit.add_argument("--coin", required=True, help="Coin symbol, e.g. BTC or ETH")
    audit.add_argument(
        "--source",
        default="historical",
        help="Data source: local, historical, or combined (default: historical)",
    )
    audit.add_argument(
        "--interval",
        default="15s",
        choices=sorted(SUPPORTED_INTERVALS),
        help="Resampled interval (default: 15s)",
    )
    audit.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    audit.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    audit.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    audit.add_argument(
        "--horizons",
        default=",".join(str(value) for value in DEFAULT_HORIZONS),
        help="Comma-separated forward return horizons in bars (default: 1,5,10,20)",
    )
    audit.add_argument("--threshold-bps", type=float, default=DEFAULT_THRESHOLD_BPS)
    audit.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    audit.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    audit.add_argument("--warmup-bars", type=int, default=DEFAULT_WARMUP_BARS)
    audit.add_argument("--json-name", default=None)
    audit.add_argument("--markdown-name", default=None)
    return parser


def _parse_horizons(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("At least one horizon is required")
    if any(value < 1 for value in values):
        raise ValueError("Horizons must be >= 1")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "audit":
        report = run_feature_audit(
            coin=args.coin,
            interval=args.interval,
            source=args.source,
            data_root=args.data_root,
            reports_root=args.reports_root,
            output_dir=args.output_dir,
            horizons=_parse_horizons(args.horizons),
            threshold_bps=args.threshold_bps,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            warmup_bars=args.warmup_bars,
            json_name=args.json_name,
            markdown_name=args.markdown_name,
        )
        print(json.dumps(report["output_paths"], indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
