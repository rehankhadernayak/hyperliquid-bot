"""Resample 1-second bid/ask CSV files into 15-second and 1-minute bars."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bid_ask_schema import (
    BEST_ASK,
    BEST_BID,
    BBO_IMBALANCE,
    BID_ASK_CSV_COLUMNS,
    COIN,
    MID_PRICE,
    PAPER_ONLY,
    SAFETY_STATEMENT,
    SPREAD,
    SPREAD_BPS,
    TIMESTAMP_ISO,
    TIMESTAMP_MS,
    timestamp_ms_to_iso,
)

SUPPORTED_INTERVALS = frozenset({"15s", "1m"})
INTERVAL_SECONDS = {"15s": 15, "1m": 60}

DEFAULT_DATA_ROOT = Path("data/bid_ask_csv")

RESAMPLED_COLUMNS = (
    "start_timestamp_ms",
    "end_timestamp_ms",
    TIMESTAMP_ISO,
    COIN,
    "open_mid",
    "high_mid",
    "low_mid",
    "close_mid",
    "avg_bid",
    "avg_ask",
    "avg_mid",
    "avg_spread",
    "avg_spread_bps",
    "avg_bbo_imbalance",
    "row_count",
    "missing_seconds",
)


def _validate_interval(interval: str) -> int:
    if interval not in SUPPORTED_INTERVALS:
        supported = ", ".join(sorted(SUPPORTED_INTERVALS))
        raise ValueError(f"Unsupported interval {interval!r}; use one of: {supported}")
    return INTERVAL_SECONDS[interval]


def bid_ask_1s_path(coin: str, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    return data_root / "1s" / f"{coin}_1s.csv"


def resampled_csv_path(
    coin: str,
    interval: str,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Path:
    _validate_interval(interval)
    return data_root / interval / f"{coin}_{interval}.csv"


def reports_dir(data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    return data_root / "reports"


def load_1s_csv(coin: str, data_root: Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    path = bid_ask_1s_path(coin, data_root)
    if not path.is_file():
        raise FileNotFoundError(f"1s CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [column for column in BID_ASK_CSV_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    return df.sort_values(TIMESTAMP_MS).reset_index(drop=True)


def resample_bid_ask(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    interval_seconds = _validate_interval(interval)
    interval_ms = interval_seconds * 1000

    if df.empty:
        return pd.DataFrame(columns=list(RESAMPLED_COLUMNS))

    work = df.sort_values(TIMESTAMP_MS).copy()
    work["_bucket_start"] = (work[TIMESTAMP_MS] // interval_ms) * interval_ms

    rows: list[dict[str, Any]] = []
    for bucket_start, group in work.groupby("_bucket_start", sort=True):
        start_ms = int(bucket_start)
        end_ms = start_ms + interval_ms - 1
        row_count = len(group)
        imbalance = group[BBO_IMBALANCE]
        avg_imbalance = (
            float(imbalance.mean(skipna=True)) if imbalance.notna().any() else None
        )

        rows.append(
            {
                "start_timestamp_ms": start_ms,
                "end_timestamp_ms": end_ms,
                TIMESTAMP_ISO: timestamp_ms_to_iso(start_ms),
                COIN: str(group[COIN].iloc[0]),
                "open_mid": float(group[MID_PRICE].iloc[0]),
                "high_mid": float(group[MID_PRICE].max()),
                "low_mid": float(group[MID_PRICE].min()),
                "close_mid": float(group[MID_PRICE].iloc[-1]),
                "avg_bid": float(group[BEST_BID].mean()),
                "avg_ask": float(group[BEST_ASK].mean()),
                "avg_mid": float(group[MID_PRICE].mean()),
                "avg_spread": float(group[SPREAD].mean()),
                "avg_spread_bps": float(group[SPREAD_BPS].mean()),
                "avg_bbo_imbalance": avg_imbalance,
                "row_count": row_count,
                "missing_seconds": interval_seconds - row_count,
            }
        )

    return pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))


def write_resampled_csv(
    df: pd.DataFrame,
    coin: str,
    interval: str,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Path:
    path = resampled_csv_path(coin, interval, data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_resampling_report(
    *,
    coin: str,
    interval: str,
    source_path: Path,
    output_path: Path,
    input_rows: int,
    output_rows: int,
    total_missing_seconds: int,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Path:
    report_dir = reports_dir(data_root)
    report_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"{coin}_{interval}_resample_{stamp}.json"

    report = {
        "coin": coin,
        "interval": interval,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "total_missing_seconds": total_missing_seconds,
        "generated_at": generated_at,
        "paper_only": PAPER_ONLY,
        "safety_statement": SAFETY_STATEMENT,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def resample_coin(
    coin: str,
    intervals: tuple[str, ...] = ("15s", "1m"),
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, dict[str, Path]]:
    source_df = load_1s_csv(coin, data_root)
    source_path = bid_ask_1s_path(coin, data_root)

    results: dict[str, dict[str, Path]] = {}
    for interval in intervals:
        resampled = resample_bid_ask(source_df, interval)
        output_path = write_resampled_csv(resampled, coin, interval, data_root)
        total_missing = (
            int(resampled["missing_seconds"].sum()) if not resampled.empty else 0
        )
        report_path = write_resampling_report(
            coin=coin,
            interval=interval,
            source_path=source_path,
            output_path=output_path,
            input_rows=len(source_df),
            output_rows=len(resampled),
            total_missing_seconds=total_missing,
            data_root=data_root,
        )
        results[interval] = {
            "output_path": output_path,
            "report_path": report_path,
        }
    return results
