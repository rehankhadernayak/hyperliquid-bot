"""Safe local importer skeleton for Hyperliquid historical L2 archive snapshots.

Reads decompressed archive files only. Does not download from S3 or call exchange APIs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bid_ask_resampler import normalize_source, resolve_bid_ask_data_dir
from bid_ask_schema import (
    ASK_SIZE,
    BID_ASK_CSV_COLUMNS,
    BEST_ASK,
    BEST_BID,
    BID_SIZE,
    COIN,
    PAPER_ONLY,
    SAFETY_STATEMENT,
    SOURCE_CHANNEL,
    SOURCE_SESSION_ID,
    TIMESTAMP_ISO,
    TIMESTAMP_MS,
    build_bid_ask_row,
    configure_csv_field_size_limit,
    timestamp_ms_to_iso,
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
DEFAULT_BACKUP_DIR = DEFAULT_OUTPUT_ROOT / "backups"
HISTORICAL_1S_SUBDIR = "historical_1s"
REPORTS_SUBDIR = "reports"
DEFAULT_MOCK_BID = 100.0
DEFAULT_MOCK_ASK = 100.1

SOURCE_SESSION_ID_HISTORICAL = "historical_archive"
SOURCE_CHANNEL_HISTORICAL = "historical_l2Book"
COVERAGE_SOURCE = "historical_archive_l2Book"


@dataclass
class ImportStats:
    rows_written: int = 0
    malformed_lines_skipped: int = 0
    invalid_rows_skipped: int = 0
    mode: str = "overwrite"
    existing_rows_before: int = 0
    new_rows_imported: int = 0
    rows_after_merge: int = 0
    duplicate_rows_removed: int = 0


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
    mode: str = "append_merge"
    existing_rows_before: int = 0
    new_rows_imported: int = 0
    rows_after_merge: int = 0
    duplicate_rows_removed: int = 0
    dedupe_key: str = TIMESTAMP_MS
    source: str = COVERAGE_SOURCE
    paper_only: bool = PAPER_ONLY
    execution_enabled: bool = EXECUTION_ENABLED
    download_enabled: bool = DOWNLOAD_ENABLED
    safety_statement: str = SAFETY_STATEMENT
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HistoricalRepairReport:
    coin: str
    source: str
    interval: str
    path: str
    dry_run: bool
    backup_path: str | None
    rows_before: int
    rows_after_basic_filters: int
    duplicate_rows_removed: int
    mock_rows_removed: int
    rows_after: int
    timestamp_iso_recomputed_count: int
    first_timestamp_iso_before: str | None
    last_timestamp_iso_before: str | None
    first_timestamp_iso_after: str | None
    last_timestamp_iso_after: str | None
    duplicate_timestamps_after: int
    monotonic_after: bool
    paper_only: bool = PAPER_ONLY
    execution_enabled: bool = EXECUTION_ENABLED
    real_execution_ready: bool = False
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


def read_historical_bid_ask_csv(csv_path: Path | str) -> list[dict[str, Any]]:
    """Read historical bid/ask rows from an existing CSV."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    rows: list[dict[str, Any]] = []
    configure_csv_field_size_limit()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def merge_historical_bid_ask_rows(
    existing_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    *,
    dedupe_key: str = TIMESTAMP_MS,
) -> tuple[list[dict[str, Any]], int]:
    """Concatenate rows, dedupe by key keeping the latest row, and sort ascending."""
    if dedupe_key != TIMESTAMP_MS:
        raise ValueError(f"Unsupported dedupe_key {dedupe_key!r}; only timestamp_ms is supported")

    combined = list(existing_rows) + list(new_rows)
    if not combined:
        return [], 0

    unique_keys = {str(row[dedupe_key]) for row in combined}
    duplicates_removed = max(0, len(combined) - len(unique_keys))

    deduped: dict[str, dict[str, Any]] = {}
    for row in combined:
        key = str(row[dedupe_key])
        deduped[key] = row

    merged = sorted(deduped.values(), key=lambda item: int(item[dedupe_key]))
    return merged, duplicates_removed


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
    overwrite: bool = False,
    dedupe_key: str = TIMESTAMP_MS,
) -> tuple[Path, ImportStats]:
    """Read a decompressed local archive file and write historical bid/ask CSV."""
    input_file = Path(input_path)
    output_file = Path(output_path)
    normalized_coin = _normalize_coin(coin)

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    new_rows: list[dict[str, Any]] = []
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

            new_rows.append(row)

    stats.new_rows_imported = len(new_rows)
    new_rows.sort(key=lambda item: int(item[TIMESTAMP_MS]))

    if overwrite or not output_file.is_file():
        stats.mode = "overwrite"
        stats.existing_rows_before = 0
        final_rows = new_rows
        stats.duplicate_rows_removed = 0
    else:
        stats.mode = "append_merge"
        existing_rows = read_historical_bid_ask_csv(output_file)
        stats.existing_rows_before = len(existing_rows)
        final_rows, stats.duplicate_rows_removed = merge_historical_bid_ask_rows(
            existing_rows,
            new_rows,
            dedupe_key=dedupe_key,
        )

    stats.rows_after_merge = len(final_rows)
    stats.rows_written = stats.rows_after_merge
    write_historical_bid_ask_csv(final_rows, output_file, output_root=output_root)
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

    configure_csv_field_size_limit()
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


def resolve_repair_csv_path(
    coin: str,
    *,
    interval: str = "1s",
    source: str = "historical",
    data_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Return the bid/ask CSV path for repair given source and interval."""
    normalized_coin = _normalize_coin(coin)
    normalized_source = normalize_source(source)
    data_dir = resolve_bid_ask_data_dir(interval, normalized_source, data_root)
    if interval == "1s":
        return data_dir / f"{normalized_coin}_1s.csv"
    return data_dir / f"{normalized_coin}_{interval}.csv"


def _validate_repair_output_path(output_path: Path, data_root: Path, interval: str, source: str) -> None:
    allowed_dir = resolve_bid_ask_data_dir(interval, normalize_source(source), data_root).resolve()
    resolved = output_path.resolve()
    if resolved != allowed_dir and allowed_dir not in resolved.parents:
        raise ValueError(f"Output path must be under {allowed_dir}")


def _first_last_timestamp_iso(rows: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    if not rows:
        return None, None
    iso_values = [str(row.get(TIMESTAMP_ISO, "")) for row in rows]
    return iso_values[0], iso_values[-1]


def _count_duplicate_timestamps(rows: list[dict[str, Any]]) -> int:
    seen: dict[str, int] = {}
    for row in rows:
        key = str(row[TIMESTAMP_MS])
        seen[key] = seen.get(key, 0) + 1
    return sum(count - 1 for count in seen.values() if count > 1)


def _is_monotonic_timestamps(rows: list[dict[str, Any]]) -> bool:
    if len(rows) < 2:
        return True
    timestamps = [int(row[TIMESTAMP_MS]) for row in rows]
    return all(timestamps[index] > timestamps[index - 1] for index in range(1, len(timestamps)))


def _row_passes_basic_filters(raw_row: dict[str, Any], coin: str) -> bool:
    timestamp_ms = _safe_float(raw_row.get(TIMESTAMP_MS))
    if timestamp_ms is None:
        return False
    if timestamp_ms != int(timestamp_ms):
        return False

    best_bid = _safe_float(raw_row.get(BEST_BID))
    best_ask = _safe_float(raw_row.get(BEST_ASK))
    if best_bid is None or best_ask is None:
        return False
    if best_bid <= 0 or best_ask <= 0:
        return False
    if best_ask < best_bid:
        return False

    row_coin = str(raw_row.get(COIN, "")).strip().upper()
    if row_coin != coin:
        return False

    return True


def _recompute_repaired_row(
    raw_row: dict[str, Any],
    *,
    timestamp_ms: int,
    coin: str,
    best_bid: float,
    best_ask: float,
) -> dict[str, Any]:
    bid_size = _safe_float(raw_row.get(BID_SIZE))
    ask_size = _safe_float(raw_row.get(ASK_SIZE))
    source_session_id = str(raw_row.get(SOURCE_SESSION_ID, SOURCE_SESSION_ID_HISTORICAL))
    source_channel = str(raw_row.get(SOURCE_CHANNEL, SOURCE_CHANNEL_HISTORICAL))
    return build_bid_ask_row(
        timestamp_ms=timestamp_ms,
        coin=coin,
        best_bid=best_bid,
        best_ask=best_ask,
        bid_size=bid_size,
        ask_size=ask_size,
        source_session_id=source_session_id,
        source_channel=source_channel,
    )


def _is_mock_row(row: dict[str, Any], mock_bid: float, mock_ask: float) -> bool:
    best_bid = _safe_float(row.get(BEST_BID))
    best_ask = _safe_float(row.get(BEST_ASK))
    if best_bid is None or best_ask is None:
        return False
    return best_bid == mock_bid and best_ask == mock_ask


def write_repaired_bid_ask_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    data_root: Path = DEFAULT_OUTPUT_ROOT,
    interval: str = "1s",
    source: str = "historical",
) -> Path:
    """Write repaired bid/ask rows to the resolved source/interval CSV path."""
    _validate_repair_output_path(output_path, data_root, interval, source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BID_ASK_CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in BID_ASK_CSV_COLUMNS})
    return output_path


def write_historical_repair_report(
    report: HistoricalRepairReport,
    *,
    data_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Save a JSON repair report under data/bid_ask_csv/reports/."""
    reports_dir = data_root / REPORTS_SUBDIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = reports_dir / f"{report.coin}_{report.interval}_repair_{timestamp}.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report_path


def repair_historical_bid_ask_csv(
    coin: str,
    *,
    interval: str = "1s",
    source: str = "historical",
    data_root: Path = DEFAULT_OUTPUT_ROOT,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    dry_run: bool = False,
    drop_mock_rows: bool = False,
    mock_bid: float = DEFAULT_MOCK_BID,
    mock_ask: float = DEFAULT_MOCK_ASK,
    csv_path: Path | None = None,
) -> tuple[HistoricalRepairReport, Path]:
    """Back up, clean, dedupe, sort, and rewrite a historical bid/ask CSV."""
    normalized_coin = _normalize_coin(coin)
    normalized_source = normalize_source(source)
    output_path = csv_path or resolve_repair_csv_path(
        normalized_coin,
        interval=interval,
        source=normalized_source,
        data_root=data_root,
    )

    if not output_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {output_path}")

    raw_rows = read_historical_bid_ask_csv(output_path)
    rows_before = len(raw_rows)
    first_iso_before, last_iso_before = _first_last_timestamp_iso(raw_rows)

    filtered_rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if not _row_passes_basic_filters(raw_row, normalized_coin):
            continue
        timestamp_ms = int(float(raw_row[TIMESTAMP_MS]))
        best_bid = float(raw_row[BEST_BID])
        best_ask = float(raw_row[BEST_ASK])
        filtered_rows.append(
            _recompute_repaired_row(
                raw_row,
                timestamp_ms=timestamp_ms,
                coin=normalized_coin,
                best_bid=best_bid,
                best_ask=best_ask,
            )
        )

    rows_after_basic_filters = len(filtered_rows)
    timestamp_iso_recomputed_count = rows_after_basic_filters

    after_mock_rows = filtered_rows
    mock_rows_removed = 0
    if drop_mock_rows:
        kept: list[dict[str, Any]] = []
        for row in filtered_rows:
            if _is_mock_row(row, mock_bid, mock_ask):
                mock_rows_removed += 1
            else:
                kept.append(row)
        after_mock_rows = kept

    deduped: dict[int, dict[str, Any]] = {}
    for row in after_mock_rows:
        deduped[int(row[TIMESTAMP_MS])] = row

    duplicate_rows_removed = len(after_mock_rows) - len(deduped)
    final_rows = sorted(deduped.values(), key=lambda item: int(item[TIMESTAMP_MS]))

    first_iso_after = timestamp_ms_to_iso(int(final_rows[0][TIMESTAMP_MS])) if final_rows else None
    last_iso_after = timestamp_ms_to_iso(int(final_rows[-1][TIMESTAMP_MS])) if final_rows else None

    backup_path: Path | None = None
    if not dry_run:
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = backup_dir / f"{normalized_coin}_{interval}_repair_{timestamp}.csv"
        shutil.copy2(output_path, backup_path)
        write_repaired_bid_ask_csv(
            final_rows,
            output_path,
            data_root=data_root,
            interval=interval,
            source=normalized_source,
        )

    report = HistoricalRepairReport(
        coin=normalized_coin,
        source=normalized_source,
        interval=interval,
        path=str(output_path.resolve()),
        dry_run=dry_run,
        backup_path=str(backup_path.resolve()) if backup_path is not None else None,
        rows_before=rows_before,
        rows_after_basic_filters=rows_after_basic_filters,
        duplicate_rows_removed=duplicate_rows_removed,
        mock_rows_removed=mock_rows_removed,
        rows_after=len(final_rows),
        timestamp_iso_recomputed_count=timestamp_iso_recomputed_count,
        first_timestamp_iso_before=first_iso_before,
        last_timestamp_iso_before=last_iso_before,
        first_timestamp_iso_after=first_iso_after,
        last_timestamp_iso_after=last_iso_after,
        duplicate_timestamps_after=_count_duplicate_timestamps(final_rows),
        monotonic_after=_is_monotonic_timestamps(final_rows),
    )
    report_path = write_historical_repair_report(report, data_root=data_root)
    return report, report_path


def _cmd_inspect_path(args: argparse.Namespace) -> int:
    path = build_archive_s3_path(args.date, args.hour, args.coin, datatype=args.datatype)
    print(path)
    return 0


def _flatten_coins(coin_groups: list[list[str]]) -> list[str]:
    coins: list[str] = []
    for group in coin_groups:
        for coin in group:
            normalized = coin.strip().upper()
            if normalized:
                coins.append(normalized)
    return coins


def _cmd_repair_csv(args: argparse.Namespace) -> int:
    coins = _flatten_coins(args.coin)
    if not coins:
        print("At least one --coin is required", file=sys.stderr)
        return 1

    data_root = Path(args.data_root)
    backup_dir = Path(args.backup_dir)
    drop_mock_rows = bool(args.drop_mock_rows)
    exit_code = 0

    for coin in coins:
        try:
            report, report_path = repair_historical_bid_ask_csv(
                coin,
                interval=args.interval,
                source=args.source,
                data_root=data_root,
                backup_dir=backup_dir,
                dry_run=args.dry_run,
                drop_mock_rows=drop_mock_rows,
                mock_bid=args.mock_bid,
                mock_ask=args.mock_ask,
            )
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            print(f"coin={coin} error={exc}", file=sys.stderr)
            exit_code = 1
            continue

        print(f"coin={report.coin} source={report.source} interval={report.interval}")
        print(f"path={report.path}")
        print(f"dry_run={report.dry_run}")
        print(f"rows_before={report.rows_before}")
        print(f"rows_after={report.rows_after}")
        print(f"duplicate_rows_removed={report.duplicate_rows_removed}")
        print(f"mock_rows_removed={report.mock_rows_removed}")
        print(f"monotonic_after={report.monotonic_after}")
        print(f"duplicate_timestamps_after={report.duplicate_timestamps_after}")
        if report.backup_path:
            print(f"backup_path={report.backup_path}")
        print(f"report_path={report_path}")
        print(SAFETY_STATEMENT)

    return exit_code


def _cmd_summarize(args: argparse.Namespace) -> int:
    if args.csv:
        csv_path = Path(args.csv)
    elif args.coin:
        csv_path = default_historical_output_path(
            args.coin,
            output_root=Path(args.output_root),
        )
    else:
        print("Either --csv or --coin is required", file=sys.stderr)
        return 1

    coverage = summarize_historical_coverage(csv_path)
    print(f"csv_path: {csv_path.resolve()}")
    print(f"row_count: {coverage.row_count}")
    print(f"start_timestamp_ms: {coverage.start_timestamp_ms}")
    print(f"end_timestamp_ms: {coverage.end_timestamp_ms}")
    print(f"missing_seconds: {coverage.missing_seconds}")
    print(f"duplicate_timestamps: {coverage.duplicate_timestamps}")
    print(f"invalid_bid_ask_rows: {coverage.invalid_bid_ask_rows}")
    print(f"source: {coverage.source}")
    print(SAFETY_STATEMENT)
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
        overwrite=args.overwrite,
        dedupe_key=args.dedupe_key,
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
        mode=stats.mode,
        existing_rows_before=stats.existing_rows_before,
        new_rows_imported=stats.new_rows_imported,
        rows_after_merge=stats.rows_after_merge,
        duplicate_rows_removed=stats.duplicate_rows_removed,
        dedupe_key=args.dedupe_key,
    )
    report_path = write_historical_import_report(report, output_root=output_root)

    print(f"Imported {stats.new_rows_imported} new rows")
    print(f"Existing rows before: {stats.existing_rows_before}")
    print(f"Rows after merge: {stats.rows_after_merge}")
    print(f"Duplicate rows removed: {stats.duplicate_rows_removed}")
    print(f"Output: {csv_path.resolve()}")
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
    import_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output CSV with only the imported file",
    )
    import_parser.add_argument(
        "--dedupe-key",
        default=TIMESTAMP_MS,
        help="Column used to deduplicate rows on merge (default: timestamp_ms)",
    )
    import_parser.set_defaults(func=_cmd_import_local)

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize coverage for a local historical bid/ask CSV only",
    )
    summarize_parser.add_argument(
        "--csv",
        help="Path to historical bid/ask CSV under data/bid_ask_csv/historical_1s/",
    )
    summarize_parser.add_argument("--coin", help="Coin symbol; uses default historical CSV path")
    summarize_parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Bid/ask output root (default: data/bid_ask_csv)",
    )
    summarize_parser.set_defaults(func=_cmd_summarize)

    repair_parser = subparsers.add_parser(
        "repair-csv",
        help="Back up, clean, dedupe, and rewrite historical bid/ask CSVs (local only)",
    )
    repair_parser.add_argument(
        "--coin",
        nargs="+",
        action="append",
        required=True,
        help="Coin symbol(s), e.g. --coin BTC ETH or --coin BTC --coin ETH",
    )
    repair_parser.add_argument(
        "--interval",
        default="1s",
        help="CSV interval (default: 1s)",
    )
    repair_parser.add_argument(
        "--source",
        default="historical",
        help="Data source: historical or local (default: historical)",
    )
    repair_parser.add_argument(
        "--data-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Bid/ask data root (default: data/bid_ask_csv)",
    )
    repair_parser.add_argument(
        "--backup-dir",
        default=str(DEFAULT_BACKUP_DIR),
        help="Backup directory (default: data/bid_ask_csv/backups)",
    )
    repair_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and report without writing backup or modifying CSV",
    )
    mock_group = repair_parser.add_mutually_exclusive_group()
    mock_group.add_argument(
        "--drop-mock-rows",
        action="store_true",
        help="Drop rows matching mock bid/ask defaults (100.0/100.1)",
    )
    mock_group.add_argument(
        "--keep-mock-rows",
        action="store_true",
        help="Keep mock-looking rows (default)",
    )
    repair_parser.add_argument(
        "--mock-bid",
        type=float,
        default=DEFAULT_MOCK_BID,
        help="Mock bid price used with --drop-mock-rows (default: 100.0)",
    )
    repair_parser.add_argument(
        "--mock-ask",
        type=float,
        default=DEFAULT_MOCK_ASK,
        help="Mock ask price used with --drop-mock-rows (default: 100.1)",
    )
    repair_parser.set_defaults(func=_cmd_repair_csv)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
