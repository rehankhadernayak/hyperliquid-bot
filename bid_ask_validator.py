"""Read-only validation for bid/ask CSV datasets (1s, 15s, 1m)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bid_ask_resampler import (
    DEFAULT_DATA_ROOT,
    normalize_source,
    resolve_bid_ask_data_dir,
)
from bid_ask_schema import (
    BID_ASK_CSV_COLUMNS,
    BEST_ASK,
    BEST_BID,
    COIN,
    PAPER_ONLY,
    SAFETY_STATEMENT,
    SPREAD,
    SPREAD_BPS,
    TIMESTAMP_MS,
)

SUPPORTED_INTERVALS = frozenset({"1s", "15s", "1m"})
SUPPORTED_SOURCES = frozenset({"local", "historical", "combined"})
INTERVAL_SECONDS = {"1s": 1, "15s": 15, "1m": 60}

RESAMPLED_COLUMNS = (
    "start_timestamp_ms",
    "end_timestamp_ms",
    "timestamp_iso",
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

EXTREME_SPREAD_BPS_THRESHOLD = 100.0


@dataclass
class ValidationReport:
    coin: str
    interval: str
    row_count: int
    coverage_start: int | None
    coverage_end: int | None
    duplicate_timestamp_count: int
    missing_periods_count: int
    invalid_bid_ask_count: int
    negative_spread_count: int
    extreme_spread_count: int
    warnings: list[str] = field(default_factory=list)
    passed: bool = False
    input_path: str = ""
    report_path: str = ""
    paper_only: bool = PAPER_ONLY
    safety_statement: str = SAFETY_STATEMENT
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_interval(interval: str) -> None:
    if interval not in SUPPORTED_INTERVALS:
        supported = ", ".join(sorted(SUPPORTED_INTERVALS))
        raise ValueError(f"Unsupported interval {interval!r}; use one of: {supported}")


def bid_ask_csv_path(
    coin: str,
    interval: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    source: str = "local",
) -> Path:
    _validate_interval(interval)
    data_dir = resolve_bid_ask_data_dir(interval, source, data_root)
    if interval == "1s":
        return data_dir / f"{coin}_1s.csv"
    return data_dir / f"{coin}_{interval}.csv"


def reports_dir(data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    return data_root / "reports"


def load_bid_ask_csv_for_validation(
    coin: str,
    interval: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    source: str = "local",
) -> pd.DataFrame:
    """Load a bid/ask CSV for validation without modifying it."""
    _validate_interval(interval)
    path = bid_ask_csv_path(coin, interval, data_root, source=source)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    if interval == "1s":
        required = list(BID_ASK_CSV_COLUMNS)
    else:
        required = list(RESAMPLED_COLUMNS)

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    return df


def _duplicate_timestamp_count(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    counts = df[column].value_counts()
    return int(sum(count - 1 for count in counts if count > 1))


def _count_non_monotonic(df: pd.DataFrame, column: str) -> int:
    if len(df) < 2 or column not in df.columns:
        return 0
    values = df[column].tolist()
    return sum(1 for index in range(1, len(values)) if values[index] <= values[index - 1])


def _count_missing_seconds_1s(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    timestamps = df[TIMESTAMP_MS].astype("int64")
    coverage_start = int(timestamps.min())
    coverage_end = int(timestamps.max())
    start_second = coverage_start // 1000
    end_second = coverage_end // 1000
    unique_seconds = timestamps // 1000
    expected_seconds = end_second - start_second + 1
    return max(0, expected_seconds - unique_seconds.nunique())


def _count_missing_resampled_bars(df: pd.DataFrame, interval: str) -> int:
    if df.empty:
        return 0
    interval_ms = INTERVAL_SECONDS[interval] * 1000
    missing_from_gaps = 0
    starts = df["start_timestamp_ms"].astype("int64").tolist()
    for index in range(1, len(starts)):
        expected = starts[index - 1] + interval_ms
        if starts[index] > expected:
            missing_from_gaps += int((starts[index] - expected) // interval_ms)

    incomplete = 0
    if "missing_seconds" in df.columns:
        incomplete += int((df["missing_seconds"] > 0).sum())
    if "row_count" in df.columns:
        incomplete += int((df["row_count"] <= 0).sum())

    return missing_from_gaps + incomplete


def validate_bid_ask_dataframe(
    df: pd.DataFrame,
    interval: str,
    *,
    coin: str = "",
    extreme_spread_bps_threshold: float = EXTREME_SPREAD_BPS_THRESHOLD,
) -> ValidationReport:
    """Validate a bid/ask dataframe and return a structured report."""
    _validate_interval(interval)

    row_count = len(df)
    if row_count == 0:
        warnings: list[str] = ["CSV contains no rows"]
        return ValidationReport(
            coin=coin,
            interval=interval,
            row_count=0,
            coverage_start=None,
            coverage_end=None,
            duplicate_timestamp_count=0,
            missing_periods_count=0,
            invalid_bid_ask_count=0,
            negative_spread_count=0,
            extreme_spread_count=0,
            warnings=warnings,
            passed=False,
        )

    if interval == "1s":
        return _validate_1s_dataframe(
            df,
            coin=coin,
            interval=interval,
            extreme_spread_bps_threshold=extreme_spread_bps_threshold,
        )
    return _validate_resampled_dataframe(df, coin=coin, interval=interval)


def _validate_1s_dataframe(
    df: pd.DataFrame,
    *,
    coin: str,
    interval: str,
    extreme_spread_bps_threshold: float,
) -> ValidationReport:
    warnings: list[str] = []
    hard_fail = False

    row_count = len(df)
    timestamps = df[TIMESTAMP_MS].astype("int64")
    coverage_start = int(timestamps.min())
    coverage_end = int(timestamps.max())

    invalid_bid_ask_count = int((df[BEST_ASK] < df[BEST_BID]).sum())
    if invalid_bid_ask_count:
        hard_fail = True
        warnings.append(f"{invalid_bid_ask_count} row(s) have best_ask < best_bid")

    negative_spread_count = int((df[SPREAD] < 0).sum())
    if negative_spread_count:
        hard_fail = True
        warnings.append(f"{negative_spread_count} row(s) have negative spread")

    negative_spread_bps_count = int((df[SPREAD_BPS] < 0).sum())
    if negative_spread_bps_count:
        hard_fail = True
        warnings.append(f"{negative_spread_bps_count} row(s) have negative spread_bps")

    duplicate_timestamp_count = _duplicate_timestamp_count(df, TIMESTAMP_MS)
    if duplicate_timestamp_count:
        hard_fail = True
        warnings.append(
            f"{duplicate_timestamp_count} duplicate timestamp_ms value(s) detected"
        )

    non_monotonic = _count_non_monotonic(df, TIMESTAMP_MS)
    if non_monotonic:
        hard_fail = True
        warnings.append(f"{non_monotonic} non-monotonic timestamp_ms transition(s) detected")

    missing_periods_count = _count_missing_seconds_1s(df)
    if missing_periods_count:
        warnings.append(f"{missing_periods_count} missing second(s) in coverage range")

    extreme_mask = df[SPREAD_BPS] >= extreme_spread_bps_threshold
    extreme_spread_count = int(extreme_mask.sum())
    if extreme_spread_count:
        warnings.append(
            f"{extreme_spread_count} row(s) exceed extreme spread threshold "
            f"({extreme_spread_bps_threshold} bps)"
        )

    return ValidationReport(
        coin=coin,
        interval=interval,
        row_count=row_count,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        duplicate_timestamp_count=duplicate_timestamp_count,
        missing_periods_count=missing_periods_count,
        invalid_bid_ask_count=invalid_bid_ask_count,
        negative_spread_count=negative_spread_count + negative_spread_bps_count,
        extreme_spread_count=extreme_spread_count,
        warnings=warnings,
        passed=not hard_fail,
    )


def _validate_resampled_dataframe(
    df: pd.DataFrame,
    *,
    coin: str,
    interval: str,
) -> ValidationReport:
    warnings: list[str] = []
    hard_fail = False

    row_count = len(df)
    starts = df["start_timestamp_ms"].astype("int64")
    coverage_start = int(starts.min())
    coverage_end = int(df["end_timestamp_ms"].astype("int64").max())

    invalid_ohlc_count = int((df["high_mid"] < df["low_mid"]).sum())
    if invalid_ohlc_count:
        hard_fail = True
        warnings.append(f"{invalid_ohlc_count} bar(s) have high_mid < low_mid")

    empty_bars = int((df["row_count"] <= 0).sum())
    if empty_bars:
        hard_fail = True
        warnings.append(f"{empty_bars} bar(s) have row_count <= 0")

    negative_avg_spread = int((df["avg_spread"] < 0).sum())
    if negative_avg_spread:
        hard_fail = True
        warnings.append(f"{negative_avg_spread} bar(s) have negative avg_spread")

    negative_avg_spread_bps = int((df["avg_spread_bps"] < 0).sum())
    if negative_avg_spread_bps:
        hard_fail = True
        warnings.append(f"{negative_avg_spread_bps} bar(s) have negative avg_spread_bps")

    duplicate_timestamp_count = _duplicate_timestamp_count(df, "start_timestamp_ms")
    if duplicate_timestamp_count:
        hard_fail = True
        warnings.append(
            f"{duplicate_timestamp_count} duplicate start_timestamp_ms value(s) detected"
        )

    non_monotonic = _count_non_monotonic(df, "start_timestamp_ms")
    if non_monotonic:
        hard_fail = True
        warnings.append(
            f"{non_monotonic} non-monotonic start_timestamp_ms transition(s) detected"
        )

    missing_periods_count = _count_missing_resampled_bars(df, interval)
    if missing_periods_count:
        warnings.append(f"{missing_periods_count} missing or incomplete bar(s) detected")

    return ValidationReport(
        coin=coin,
        interval=interval,
        row_count=row_count,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        duplicate_timestamp_count=duplicate_timestamp_count,
        missing_periods_count=missing_periods_count,
        invalid_bid_ask_count=invalid_ohlc_count + empty_bars,
        negative_spread_count=negative_avg_spread + negative_avg_spread_bps,
        extreme_spread_count=0,
        warnings=warnings,
        passed=not hard_fail,
    )


def save_validation_report(
    report: ValidationReport,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Path:
    """Write a JSON validation report under data/bid_ask_csv/reports/."""
    report_dir = reports_dir(data_root)
    report_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"{report.coin}_{report.interval}_validation_{stamp}.json"
    report.report_path = str(report_path)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report_path


def validate_coin_interval(
    coin: str,
    interval: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    extreme_spread_bps_threshold: float = EXTREME_SPREAD_BPS_THRESHOLD,
    *,
    source: str = "local",
) -> ValidationReport:
    """Load, validate, and persist a validation report for one coin/interval."""
    input_path = bid_ask_csv_path(coin, interval, data_root, source=source)
    df = load_bid_ask_csv_for_validation(coin, interval, data_root, source=source)
    report = validate_bid_ask_dataframe(
        df,
        interval,
        coin=coin,
        extreme_spread_bps_threshold=extreme_spread_bps_threshold,
    )
    report.input_path = str(input_path)
    save_validation_report(report, data_root=data_root)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate bid/ask CSV datasets (read-only, research-only, paper-only)."
        )
    )
    parser.add_argument("--coins", nargs="+", required=True, help="Coin symbols, e.g. BTC ETH")
    parser.add_argument(
        "--interval",
        required=True,
        choices=sorted(SUPPORTED_INTERVALS),
        help="CSV interval to validate (1s, 15s, or 1m)",
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
    source = normalize_source(args.source)
    coins = [coin.strip().upper() for coin in args.coins if coin.strip()]
    exit_code = 0

    for coin in coins:
        try:
            report = validate_coin_interval(
                coin,
                args.interval,
                data_root=args.data_root,
                source=source,
            )
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            print(f"coin={coin} interval={args.interval} source={source} error={exc}")
            exit_code = 1
            continue

        print(f"coin={coin} interval={args.interval} source={source}")
        print(f"data_dir={resolve_bid_ask_data_dir(args.interval, source, args.data_root)}")
        print(f"input_path={report.input_path}")
        print(f"report_path={report.report_path}")
        print(f"passed={report.passed}")
        print(f"row_count={report.row_count}")
        for warning in report.warnings:
            print(f"warning: {warning}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
