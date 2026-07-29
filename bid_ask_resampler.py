"""Resample 1-second bid/ask CSV files into 15-second and 1-minute bars."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bid_ask_schema import (
    ASK_SIZE,
    BEST_ASK,
    BEST_BID,
    BBO_IMBALANCE,
    BID_ASK_CSV_COLUMNS,
    BID_SIZE,
    COIN,
    MID_PRICE,
    PAPER_ONLY,
    SAFETY_STATEMENT,
    SOURCE_CHANNEL,
    SOURCE_SESSION_ID,
    SPREAD,
    SPREAD_BPS,
    TIMESTAMP_ISO,
    TIMESTAMP_MS,
    timestamp_ms_to_iso,
)

SUPPORTED_INTERVALS = frozenset({"15s", "1m"})
SUPPORTED_INPUT_INTERVALS = frozenset({"1s"})
SUPPORTED_SOURCES = frozenset({"local", "historical", "combined"})
COMBINED_SOURCE_ALIASES = frozenset({"combined"})
INTERVAL_SECONDS = {"15s": 15, "1m": 60}

DEFAULT_DATA_ROOT = Path("data/bid_ask_csv")
HISTORICAL_SOURCE_ALIASES = frozenset({"historical", "exchange", "all-time", "alltime"})

REQUIRED_RESAMPLED_COLUMNS = (
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

# Top-of-book sizes use last-valid 1s tick within each bar (close-of-bar snapshot).
OPTIONAL_RESAMPLED_COLUMNS = (BID_SIZE, ASK_SIZE)

RESAMPLED_COLUMNS = REQUIRED_RESAMPLED_COLUMNS + OPTIONAL_RESAMPLED_COLUMNS

CRITICAL_NUMERIC_1S_COLUMNS = (
    TIMESTAMP_MS,
    BEST_BID,
    BEST_ASK,
    MID_PRICE,
    SPREAD,
    SPREAD_BPS,
)

OPTIONAL_NUMERIC_1S_COLUMNS = (BID_SIZE, ASK_SIZE, BBO_IMBALANCE)

METADATA_STRING_COLUMNS = (
    TIMESTAMP_ISO,
    COIN,
    SOURCE_SESSION_ID,
    SOURCE_CHANNEL,
)


def sanitize_1s_dataframe(
    df: pd.DataFrame,
    *,
    source_label: str = "1s input",
) -> tuple[pd.DataFrame, int]:
    """Coerce numeric fields and drop rows that cannot be resampled."""
    work = df.copy()
    missing = [column for column in BID_ASK_CSV_COLUMNS if column not in work.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in CRITICAL_NUMERIC_1S_COLUMNS + OPTIONAL_NUMERIC_1S_COLUMNS:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    numeric_columns = [
        column
        for column in (*CRITICAL_NUMERIC_1S_COLUMNS, *OPTIONAL_NUMERIC_1S_COLUMNS)
        if column in work.columns
    ]
    if numeric_columns:
        work[numeric_columns] = work[numeric_columns].replace(
            [float("inf"), float("-inf")],
            pd.NA,
        )

    original_row_count = len(work)
    work = work.dropna(subset=list(CRITICAL_NUMERIC_1S_COLUMNS))
    work = work.loc[work[TIMESTAMP_MS] > 0]
    removed_row_count = original_row_count - len(work)
    if removed_row_count:
        warnings.warn(
            f"{source_label}: removed {removed_row_count} malformed row(s) "
            "with non-numeric, non-finite, missing, or non-positive timestamp values",
            RuntimeWarning,
            stacklevel=2,
        )

    return work.reset_index(drop=True), removed_row_count


def _last_valid_value(series: pd.Series) -> float | None:
    """Return the last non-null numeric value in a series, or None when absent."""
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])


def _validate_interval(interval: str) -> int:
    if interval not in SUPPORTED_INTERVALS:
        supported = ", ".join(sorted(SUPPORTED_INTERVALS))
        raise ValueError(f"Unsupported interval {interval!r}; use one of: {supported}")
    return INTERVAL_SECONDS[interval]


def normalize_source(source: str) -> str:
    normalized = source.strip().lower()
    if normalized in HISTORICAL_SOURCE_ALIASES:
        return "historical"
    if normalized == "local":
        return "local"
    if normalized in COMBINED_SOURCE_ALIASES:
        return "combined"
    supported = ", ".join(sorted(SUPPORTED_SOURCES))
    raise ValueError(f"Unsupported source {source!r}; use one of: {supported}")


def resolve_bid_ask_data_dir(
    interval: str,
    source: str = "local",
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Path:
    """Return the CSV directory for an interval and data source."""
    if interval not in SUPPORTED_INPUT_INTERVALS | SUPPORTED_INTERVALS:
        supported = ", ".join(sorted(SUPPORTED_INPUT_INTERVALS | SUPPORTED_INTERVALS))
        raise ValueError(f"Unsupported interval {interval!r}; use one of: {supported}")

    normalized_source = normalize_source(source)
    if normalized_source == "local":
        subdir = interval
    elif normalized_source == "combined":
        subdir = f"combined_{interval}"
    else:
        subdir = f"historical_{interval}"
    return data_root / subdir


def bid_ask_1s_path(
    coin: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    source: str = "local",
) -> Path:
    return resolve_bid_ask_data_dir("1s", source, data_root) / f"{coin}_1s.csv"


def resampled_csv_path(
    coin: str,
    interval: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    source: str = "local",
) -> Path:
    _validate_interval(interval)
    return resolve_bid_ask_data_dir(interval, source, data_root) / f"{coin}_{interval}.csv"


def reports_dir(data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    return data_root / "reports"


def load_1s_csv(
    coin: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    source: str = "local",
) -> pd.DataFrame:
    path = bid_ask_1s_path(coin, data_root, source=source)
    if not path.is_file():
        raise FileNotFoundError(f"1s CSV not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    missing = [column for column in BID_ASK_CSV_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df, _removed = sanitize_1s_dataframe(df, source_label=str(path))
    return df.sort_values(TIMESTAMP_MS).reset_index(drop=True)


def resample_bid_ask(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    interval_seconds = _validate_interval(interval)
    interval_ms = interval_seconds * 1000

    if df.empty:
        return pd.DataFrame(columns=list(RESAMPLED_COLUMNS))

    work, _removed = sanitize_1s_dataframe(df, source_label="1s resample input")
    if work.empty:
        return pd.DataFrame(columns=list(RESAMPLED_COLUMNS))

    work = work.sort_values(TIMESTAMP_MS).copy()
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

        row: dict[str, Any] = {
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
        if BID_SIZE in group.columns:
            row[BID_SIZE] = _last_valid_value(group[BID_SIZE])
        if ASK_SIZE in group.columns:
            row[ASK_SIZE] = _last_valid_value(group[ASK_SIZE])
        rows.append(row)

    return pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))


def source_1s_timestamp_bounds(df: pd.DataFrame) -> tuple[int | None, int | None]:
    """Return min/max timestamp_ms from sanitized 1s input."""
    if df.empty or TIMESTAMP_MS not in df.columns:
        return None, None
    work, _removed = sanitize_1s_dataframe(df, source_label="1s bounds")
    if work.empty:
        return None, None
    timestamps = work[TIMESTAMP_MS].astype("int64")
    return int(timestamps.min()), int(timestamps.max())


def last_resampled_bucket_start(max_timestamp_ms: int, interval: str) -> int:
    """Return the bucket start ms for the bar containing the final 1s observation."""
    interval_ms = _validate_interval(interval) * 1000
    return (int(max_timestamp_ms) // interval_ms) * interval_ms


def trim_resampled_to_source_bounds(
    df: pd.DataFrame,
    max_timestamp_ms: int,
    interval: str,
) -> pd.DataFrame:
    """Drop resampled bars that start after the bucket containing the final 1s tick."""
    if df.empty:
        return df
    last_bucket_start = last_resampled_bucket_start(max_timestamp_ms, interval)
    trimmed = df.loc[df["start_timestamp_ms"].astype("int64") <= last_bucket_start]
    return trimmed.reset_index(drop=True)


def assert_resampled_aligned_with_source(
    resampled: pd.DataFrame,
    *,
    source_min_ms: int | None,
    source_max_ms: int | None,
    interval: str,
) -> None:
    """Raise when resampled output extends beyond the current 1s source bounds."""
    if resampled.empty:
        if source_max_ms is not None:
            raise ValueError(
                f"Resampled {interval} output is empty but 1s source has data "
                f"(max_timestamp_ms={source_max_ms})"
            )
        return
    if source_max_ms is None:
        return

    last_bucket_start = last_resampled_bucket_start(source_max_ms, interval)
    output_last_start = int(resampled["start_timestamp_ms"].astype("int64").max())
    if output_last_start > last_bucket_start:
        raise ValueError(
            f"Resampled {interval} last bar start {output_last_start} exceeds "
            f"final 1s bucket start {last_bucket_start} "
            f"(source_max_timestamp_ms={source_max_ms})"
        )

    if source_min_ms is not None:
        output_first_start = int(resampled["start_timestamp_ms"].astype("int64").min())
        first_bucket_start = last_resampled_bucket_start(source_min_ms, interval)
        if output_first_start < first_bucket_start:
            raise ValueError(
                f"Resampled {interval} first bar start {output_first_start} precedes "
                f"first 1s bucket start {first_bucket_start}"
            )


def write_resampled_csv(
    df: pd.DataFrame,
    coin: str,
    interval: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    source: str = "local",
) -> Path:
    path = resampled_csv_path(coin, interval, data_root, source=source)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
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
    discarded_malformed_rows: int = 0,
    source_min_timestamp_ms: int | None = None,
    source_max_timestamp_ms: int | None = None,
    output_max_start_timestamp_ms: int | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    source: str = "local",
    data_dir: Path | None = None,
) -> Path:
    report_dir = reports_dir(data_root)
    report_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"{coin}_{interval}_resample_{stamp}.json"

    report = {
        "coin": coin,
        "interval": interval,
        "source": normalize_source(source),
        "data_dir": str(data_dir or resolve_bid_ask_data_dir(interval, source, data_root)),
        "source_path": str(source_path),
        "output_path": str(output_path),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "total_missing_seconds": total_missing_seconds,
        "discarded_malformed_rows": discarded_malformed_rows,
        "source_min_timestamp_ms": source_min_timestamp_ms,
        "source_max_timestamp_ms": source_max_timestamp_ms,
        "output_max_start_timestamp_ms": output_max_start_timestamp_ms,
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
    *,
    source: str = "local",
) -> dict[str, dict[str, Path]]:
    raw_df = pd.read_csv(
        bid_ask_1s_path(coin, data_root, source=source),
        low_memory=False,
    )
    missing = [column for column in BID_ASK_CSV_COLUMNS if column not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    source_df, discarded_rows = sanitize_1s_dataframe(
        raw_df,
        source_label=str(bid_ask_1s_path(coin, data_root, source=source)),
    )
    source_df = source_df.sort_values(TIMESTAMP_MS).reset_index(drop=True)
    source_path = bid_ask_1s_path(coin, data_root, source=source)
    source_min_ms, source_max_ms = source_1s_timestamp_bounds(source_df)

    results: dict[str, dict[str, Path]] = {}
    for interval in intervals:
        resampled = resample_bid_ask(source_df, interval)
        if source_max_ms is not None:
            resampled = trim_resampled_to_source_bounds(resampled, source_max_ms, interval)
        assert_resampled_aligned_with_source(
            resampled,
            source_min_ms=source_min_ms,
            source_max_ms=source_max_ms,
            interval=interval,
        )
        output_path = write_resampled_csv(
            resampled,
            coin,
            interval,
            data_root,
            source=source,
        )
        total_missing = (
            int(resampled["missing_seconds"].sum()) if not resampled.empty else 0
        )
        output_max_start = (
            int(resampled["start_timestamp_ms"].astype("int64").max())
            if not resampled.empty
            else None
        )
        report_path = write_resampling_report(
            coin=coin,
            interval=interval,
            source_path=source_path,
            output_path=output_path,
            input_rows=len(raw_df),
            output_rows=len(resampled),
            total_missing_seconds=total_missing,
            discarded_malformed_rows=discarded_rows,
            source_min_timestamp_ms=source_min_ms,
            source_max_timestamp_ms=source_max_ms,
            output_max_start_timestamp_ms=output_max_start,
            data_root=data_root,
            source=source,
            data_dir=resolve_bid_ask_data_dir(interval, source, data_root),
        )
        results[interval] = {
            "output_path": output_path,
            "report_path": report_path,
        }
    return results


def _resolve_target_intervals(to_interval: str) -> tuple[str, ...]:
    if to_interval == "all":
        return ("15s", "1m")
    if to_interval not in SUPPORTED_INTERVALS:
        supported = ", ".join(sorted(SUPPORTED_INTERVALS | {"all"}))
        raise ValueError(f"Unsupported to-interval {to_interval!r}; use one of: {supported}")
    return (to_interval,)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resample 1-second bid/ask CSV files into 15-second and 1-minute bars "
            "(read-only, research-only, paper-only)."
        )
    )
    parser.add_argument("--coins", nargs="+", required=True, help="Coin symbols, e.g. BTC ETH")
    parser.add_argument(
        "--from-interval",
        default="1s",
        choices=sorted(SUPPORTED_INPUT_INTERVALS),
        help="Input interval (default: 1s)",
    )
    parser.add_argument(
        "--to-interval",
        required=True,
        help="Output interval: 15s, 1m, or all",
    )
    parser.add_argument(
        "--source",
        default="local",
        choices=sorted(SUPPORTED_SOURCES),
        help="Data source: local recorder CSVs or historical archive CSVs (default: local)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for bid/ask CSV files (default: data/bid_ask_csv)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.from_interval != "1s":
        print(f"error=Only 1s input is supported; got {args.from_interval!r}", file=sys.stderr)
        return 2

    try:
        intervals = _resolve_target_intervals(args.to_interval)
    except ValueError as exc:
        print(f"error={exc}", file=sys.stderr)
        return 2

    source = normalize_source(args.source)
    coins = [coin.strip().upper() for coin in args.coins if coin.strip()]
    exit_code = 0
    for coin in coins:
        try:
            results = resample_coin(
                coin,
                intervals=intervals,
                data_root=args.data_root,
                source=source,
            )
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            print(f"coin={coin} error={exc}")
            exit_code = 1
            continue

        for interval, paths in results.items():
            print(f"coin={coin} interval={interval} source={source}")
            print(f"output_path={paths['output_path']}")
            print(f"report_path={paths['report_path']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
