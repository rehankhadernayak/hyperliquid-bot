"""Safe local importer skeleton for Hyperliquid historical L2 archive snapshots.

Reads decompressed archive files only. Does not download from S3 or call exchange APIs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bid_ask_schema import (
    BID_ASK_CSV_COLUMNS,
    BEST_ASK,
    BEST_BID,
    PAPER_ONLY,
    SAFETY_STATEMENT,
    TIMESTAMP_MS,
    build_bid_ask_row,
    validate_bid_ask,
)
from historical_backfill_research import (
    ARCHIVE_BUCKET,
    ARCHIVE_MARKET_DATA_PREFIX,
    DEFAULT_L2_DATATYPE,
    DOWNLOAD_ENABLED,
    EXECUTION_ENABLED,
)

DEFAULT_OUTPUT_ROOT = Path("data/bid_ask_csv")
HISTORICAL_1S_SUBDIR = "historical_1s"
REPORTS_SUBDIR = "reports"

SOURCE_SESSION_ID_HISTORICAL = "historical_archive"
SOURCE_CHANNEL_HISTORICAL = "historical_l2Book"
COVERAGE_SOURCE = "historical_archive_l2Book"


@dataclass
class ImportStats:
    rows_written: int = 0
    malformed_lines_skipped: int = 0
    invalid_rows_skipped: int = 0


@dataclass
class HistoricalCoverageSummary:
    row_count: int
    start_timestamp_ms: int | None
    end_timestamp_ms: int | None
    missing_seconds: int
    duplicate_timestamps: int
    invalid_bid_ask_rows: int
    source: str = COVERAGE_SOURCE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HistoricalImportReport:
    coin: str
    input_path: str
    output_path: str
    row_count: int
    coverage_start_ms: int | None
    coverage_end_ms: int | None
    missing_seconds: int
    duplicate_timestamps: int
    invalid_bid_ask_rows: int
    malformed_lines_skipped: int
    invalid_rows_skipped: int
    source: str = COVERAGE_SOURCE
    paper_only: bool = PAPER_ONLY
    download_enabled: bool = DOWNLOAD_ENABLED
    safety_statement: str = SAFETY_STATEMENT
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_date(date: str | int) -> str:
    text = str(date).strip()
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"date must be YYYYMMDD, got {date!r}")
    return text


def _normalize_hour(hour: int | str) -> str:
    value = int(hour)
    if value < 0 or value > 23:
        raise ValueError(f"hour must be 0-23, got {hour!r}")
    return str(value)


def _normalize_coin(coin: str) -> str:
    text = coin.strip().upper()
    if not text:
        raise ValueError("coin must be non-empty")
    return text


def build_archive_s3_path(
    date: str | int,
    hour: int | str,
    coin: str,
    datatype: str = DEFAULT_L2_DATATYPE,
) -> str:
    """Return the expected Hyperliquid archive S3 path (no download)."""
    normalized_date = _normalize_date(date)
    normalized_hour = _normalize_hour(hour)
    normalized_coin = _normalize_coin(coin)
    datatype_name = datatype.strip() or DEFAULT_L2_DATATYPE
    return (
        f"s3://{ARCHIVE_BUCKET}/{ARCHIVE_MARKET_DATA_PREFIX}/"
        f"{normalized_date}/{normalized_hour}/{datatype_name}/{normalized_coin}.lz4"
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_l2_data(payload: dict[str, Any]) -> dict[str, Any] | None:
    if "levels" in payload:
        return payload
    data = payload.get("data")
    if isinstance(data, dict) and "levels" in data:
        return data
    raw = payload.get("raw")
    if isinstance(raw, dict):
        return _extract_l2_data(raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped.replace("'", '"'))
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(stripped)
                except (SyntaxError, ValueError):
                    return None
            if isinstance(parsed, dict):
                return _extract_l2_data(parsed)
    return None


def _top_of_book_from_levels(levels: list[Any]) -> tuple[float, float, float | None, float | None] | None:
    if len(levels) < 2:
        return None
    bid_levels = levels[0] if isinstance(levels[0], list) else []
    ask_levels = levels[1] if isinstance(levels[1], list) else []
    if not bid_levels or not ask_levels:
        return None
    bid = bid_levels[0] if isinstance(bid_levels[0], dict) else {}
    ask = ask_levels[0] if isinstance(ask_levels[0], dict) else {}
    bid_px = _safe_float(bid.get("px"))
    ask_px = _safe_float(ask.get("px"))
    if bid_px is None or ask_px is None:
        return None
    return bid_px, ask_px, _safe_float(bid.get("sz")), _safe_float(ask.get("sz"))


def parse_l2_snapshot_line(line: str) -> dict[str, Any] | None:
    """Parse one JSON/object line from a decompressed L2 archive file."""
    stripped = line.strip()
    if not stripped:
        return None

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    data = _extract_l2_data(payload)
    if data is None:
        return None

    levels = data.get("levels")
    if not isinstance(levels, list):
        return None

    top = _top_of_book_from_levels(levels)
    if top is None:
        return None

    best_bid, best_ask, bid_size, ask_size = top
    exchange_ts = data.get("time")
    if exchange_ts is None:
        exchange_ts = payload.get("time")
    if exchange_ts is None:
        return None

    try:
        timestamp_ms = int(exchange_ts)
    except (TypeError, ValueError):
        return None

    coin = str(data.get("coin", payload.get("coin", ""))).strip()
    if not coin:
        return None

    return {
        "timestamp_ms": timestamp_ms,
        "coin": coin,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "source_channel": SOURCE_CHANNEL_HISTORICAL,
    }


def l2_snapshot_to_bid_ask_row(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    """Convert parsed L2 top-of-book into the standard bid/ask CSV schema."""
    try:
        return build_bid_ask_row(
            timestamp_ms=int(snapshot["timestamp_ms"]),
            coin=str(snapshot["coin"]),
            best_bid=float(snapshot["best_bid"]),
            best_ask=float(snapshot["best_ask"]),
            bid_size=snapshot.get("bid_size"),
            ask_size=snapshot.get("ask_size"),
            source_session_id=SOURCE_SESSION_ID_HISTORICAL,
            source_channel=SOURCE_CHANNEL_HISTORICAL,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _validate_historical_output_path(output_path: Path, output_root: Path) -> None:
    allowed_dir = (output_root / HISTORICAL_1S_SUBDIR).resolve()
    resolved = output_path.resolve()
    if resolved != allowed_dir and allowed_dir not in resolved.parents:
        raise ValueError(f"Output path must be under {allowed_dir}")


def write_historical_bid_ask_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Write historical bid/ask rows under data/bid_ask_csv/historical_1s/."""
    _validate_historical_output_path(output_path, output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BID_ASK_CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in BID_ASK_CSV_COLUMNS})
    return output_path


def import_decompressed_l2_file(
    input_path: Path | str,
    coin: str,
    output_path: Path | str,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[Path, ImportStats]:
    """Read a decompressed local archive file and write historical bid/ask CSV."""
    input_file = Path(input_path)
    output_file = Path(output_path)
    normalized_coin = _normalize_coin(coin)

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    rows: list[dict[str, Any]] = []
    stats = ImportStats()

    with input_file.open(encoding="utf-8") as handle:
        for line in handle:
            snapshot = parse_l2_snapshot_line(line)
            if snapshot is None:
                if line.strip():
                    stats.malformed_lines_skipped += 1
                continue

            if snapshot["coin"].upper() != normalized_coin:
                stats.invalid_rows_skipped += 1
                continue

            row = l2_snapshot_to_bid_ask_row(snapshot)
            if row is None:
                stats.invalid_rows_skipped += 1
                continue

            rows.append(row)
            stats.rows_written += 1

    rows.sort(key=lambda item: int(item[TIMESTAMP_MS]))
    write_historical_bid_ask_csv(rows, output_file, output_root=output_root)
    return output_file, stats


def summarize_historical_coverage(csv_path: Path | str) -> HistoricalCoverageSummary:
    """Summarize coverage metrics for a historical bid/ask CSV."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    row_count = 0
    timestamps: list[int] = []
    duplicate_timestamps = 0
    invalid_bid_ask_rows = 0
    seen: dict[int, int] = {}

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            try:
                timestamp_ms = int(row[TIMESTAMP_MS])
                best_bid = float(row[BEST_BID])
                best_ask = float(row[BEST_ASK])
                validate_bid_ask(best_bid, best_ask)
            except (KeyError, TypeError, ValueError):
                invalid_bid_ask_rows += 1
                continue

            timestamps.append(timestamp_ms)
            seen[timestamp_ms] = seen.get(timestamp_ms, 0) + 1

    duplicate_timestamps = sum(count - 1 for count in seen.values() if count > 1)

    if not timestamps:
        return HistoricalCoverageSummary(
            row_count=row_count,
            start_timestamp_ms=None,
            end_timestamp_ms=None,
            missing_seconds=0,
            duplicate_timestamps=duplicate_timestamps,
            invalid_bid_ask_rows=invalid_bid_ask_rows,
        )

    timestamps.sort()
    start_timestamp_ms = timestamps[0]
    end_timestamp_ms = timestamps[-1]
    seconds_present = {timestamp // 1000 for timestamp in timestamps}
    start_second = start_timestamp_ms // 1000
    end_second = end_timestamp_ms // 1000
    expected_seconds = end_second - start_second + 1
    missing_seconds = max(0, expected_seconds - len(seconds_present))

    return HistoricalCoverageSummary(
        row_count=row_count,
        start_timestamp_ms=start_timestamp_ms,
        end_timestamp_ms=end_timestamp_ms,
        missing_seconds=missing_seconds,
        duplicate_timestamps=duplicate_timestamps,
        invalid_bid_ask_rows=invalid_bid_ask_rows,
    )


def write_historical_import_report(
    report: HistoricalImportReport,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Save a JSON import report under data/bid_ask_csv/reports/."""
    reports_dir = output_root / REPORTS_SUBDIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = reports_dir / f"historical_{report.coin}_{timestamp}.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report_path


def default_historical_output_path(
    coin: str,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    return output_root / HISTORICAL_1S_SUBDIR / f"{_normalize_coin(coin)}_1s.csv"


def _cmd_inspect_path(args: argparse.Namespace) -> int:
    path = build_archive_s3_path(args.date, args.hour, args.coin, datatype=args.datatype)
    print(path)
    return 0


def _cmd_import_local(args: argparse.Namespace) -> int:
    coin = _normalize_coin(args.coin)
    output_path = (
        Path(args.output)
        if args.output
        else default_historical_output_path(coin, output_root=Path(args.output_root))
    )
    output_root = Path(args.output_root)

    csv_path, stats = import_decompressed_l2_file(
        args.input,
        coin,
        output_path,
        output_root=output_root,
    )
    coverage = summarize_historical_coverage(csv_path)
    report = HistoricalImportReport(
        coin=coin,
        input_path=str(Path(args.input).resolve()),
        output_path=str(csv_path.resolve()),
        row_count=coverage.row_count,
        coverage_start_ms=coverage.start_timestamp_ms,
        coverage_end_ms=coverage.end_timestamp_ms,
        missing_seconds=coverage.missing_seconds,
        duplicate_timestamps=coverage.duplicate_timestamps,
        invalid_bid_ask_rows=coverage.invalid_bid_ask_rows,
        malformed_lines_skipped=stats.malformed_lines_skipped,
        invalid_rows_skipped=stats.invalid_rows_skipped,
    )
    report_path = write_historical_import_report(report, output_root=output_root)

    print(f"Imported {coverage.row_count} rows to {csv_path}")
    print(f"Coverage report: {report_path}")
    print(SAFETY_STATEMENT)
    print("No archive data was downloaded.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safe local Hyperliquid historical L2 importer (research-only, no downloads)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-path",
        help="Print expected archive S3 path without downloading",
    )
    inspect_parser.add_argument("--date", required=True, help="Archive date as YYYYMMDD")
    inspect_parser.add_argument("--hour", required=True, type=int, help="Hour 0-23")
    inspect_parser.add_argument("--coin", required=True, help="Coin symbol, e.g. SOL")
    inspect_parser.add_argument(
        "--datatype",
        default=DEFAULT_L2_DATATYPE,
        help="Archive datatype folder (default: l2Book)",
    )
    inspect_parser.set_defaults(func=_cmd_inspect_path)

    import_parser = subparsers.add_parser(
        "import-local",
        help="Import a decompressed local archive file only",
    )
    import_parser.add_argument(
        "--input",
        required=True,
        help="Path to decompressed local archive file",
    )
    import_parser.add_argument("--coin", required=True, help="Coin symbol, e.g. SOL")
    import_parser.add_argument(
        "--output",
        help="Optional output CSV path under data/bid_ask_csv/historical_1s/",
    )
    import_parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Bid/ask output root (default: data/bid_ask_csv)",
    )
    import_parser.set_defaults(func=_cmd_import_local)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
