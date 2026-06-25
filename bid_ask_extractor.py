"""Read-only local bid/ask extraction from Hyperliquid market data files."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from bid_ask_schema import (
    BID_ASK_CSV_COLUMNS,
    FORBIDDEN_MODULE_NAMES,
    FORBIDDEN_SOURCE_PATTERNS,
    PAPER_ONLY,
    SAFETY_STATEMENT,
    build_bid_ask_row,
)

DEFAULT_MARKET_STREAM_DIR = Path("data/market_stream")
DEFAULT_OUTPUT_ROOT = Path("data/bid_ask_csv")
CSV_1S_SUBDIR = "1s"
REPORTS_SUBDIR = "reports"

RAW_BBO_CHANNELS = frozenset({"bbo"})
RAW_L2_CHANNELS = frozenset({"l2Book"})
NORMALIZED_BBO_EVENT_TYPES = frozenset({"bbo"})
NORMALIZED_L2_EVENT_TYPES = frozenset({"l2_book"})

_PARQUET_AVAILABLE = False
try:
    import pyarrow.parquet as pq

    _PARQUET_AVAILABLE = True
except ImportError:
    pq = None  # type: ignore[assignment]


@dataclass
class BidAskSource:
    path: Path
    source_kind: str
    channel: str
    coin: str | None = None


@dataclass
class ExtractStats:
    rows_extracted: int = 0
    invalid_rows_skipped: int = 0
    malformed_lines_skipped: int = 0


@dataclass
class SamplingResult:
    rows: list[dict[str, Any]]
    missing_seconds: int
    duplicate_timestamps: int
    coverage_start_ms: int | None
    coverage_end_ms: int | None


@dataclass
class ExtractionReport:
    coin: str
    source_files: list[str]
    output_path: str
    row_count: int
    coverage_start_ms: int | None
    coverage_end_ms: int | None
    missing_seconds: int
    duplicate_timestamps: int
    invalid_rows_skipped: int
    locally_collected_data_only: bool = True
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coin_from_partition(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("coin="):
            return part.split("=", 1)[1]
    return None


def _event_type_from_partition(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("event_type="):
            return part.split("=", 1)[1]
    return None


def discover_local_bid_ask_sources(
    market_stream_dir: Path,
    *,
    coins: tuple[str, ...] | None = None,
) -> list[BidAskSource]:
    """Find local normalized or raw files that contain BBO/bid-ask data."""
    sources: list[BidAskSource] = []
    if not market_stream_dir.exists():
        return sources

    raw_dir = market_stream_dir / "raw"
    if raw_dir.exists():
        for jsonl_path in sorted(raw_dir.rglob("*.jsonl")):
            sources.append(
                BidAskSource(
                    path=jsonl_path,
                    source_kind="raw_jsonl",
                    channel="bbo",
                )
            )

    normalized_dir = market_stream_dir / "normalized"
    if normalized_dir.exists():
        for parquet_path in sorted(normalized_dir.rglob("*.parquet")):
            event_type = _event_type_from_partition(parquet_path)
            if event_type not in NORMALIZED_BBO_EVENT_TYPES | NORMALIZED_L2_EVENT_TYPES:
                continue
            coin = _coin_from_partition(parquet_path)
            if coins and coin and coin not in coins:
                continue
            channel = "bbo" if event_type == "bbo" else "l2Book"
            sources.append(
                BidAskSource(
                    path=parquet_path,
                    source_kind="normalized_parquet",
                    channel=channel,
                    coin=coin,
                )
            )

        for jsonl_path in sorted(normalized_dir.rglob("*.jsonl")):
            event_type = _event_type_from_partition(jsonl_path)
            if event_type not in NORMALIZED_BBO_EVENT_TYPES | NORMALIZED_L2_EVENT_TYPES:
                continue
            coin = _coin_from_partition(jsonl_path)
            if coins and coin and coin not in coins:
                continue
            channel = "bbo" if event_type == "bbo" else "l2Book"
            sources.append(
                BidAskSource(
                    path=jsonl_path,
                    source_kind="normalized_jsonl",
                    channel=channel,
                    coin=coin,
                )
            )

    return sources


def _parse_bbo_payload(
    data: dict[str, Any],
    *,
    timestamp_ms: int,
    source_session_id: str,
    source_channel: str,
) -> dict[str, Any] | None:
    bbo = data.get("bbo", [])
    if len(bbo) < 2:
        return None
    bid = bbo[0] if isinstance(bbo[0], dict) else {}
    ask = bbo[1] if isinstance(bbo[1], dict) else {}
    bid_px = _safe_float(bid.get("px"))
    ask_px = _safe_float(ask.get("px"))
    if bid_px is None or ask_px is None:
        return None
    bid_sz = _safe_float(bid.get("sz"))
    ask_sz = _safe_float(ask.get("sz"))
    coin = str(data.get("coin", ""))
    exchange_ts = data.get("time")
    row_ts = int(exchange_ts) if exchange_ts is not None else timestamp_ms
    try:
        return build_bid_ask_row(
            timestamp_ms=row_ts,
            coin=coin,
            best_bid=bid_px,
            best_ask=ask_px,
            bid_size=bid_sz,
            ask_size=ask_sz,
            source_session_id=source_session_id,
            source_channel=source_channel,
        )
    except ValueError:
        return None


def _parse_l2_payload(
    data: dict[str, Any],
    *,
    timestamp_ms: int,
    source_session_id: str,
    source_channel: str,
) -> dict[str, Any] | None:
    levels = data.get("levels", [])
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
    bid_sz = _safe_float(bid.get("sz"))
    ask_sz = _safe_float(ask.get("sz"))
    coin = str(data.get("coin", ""))
    exchange_ts = data.get("time")
    row_ts = int(exchange_ts) if exchange_ts is not None else timestamp_ms
    try:
        return build_bid_ask_row(
            timestamp_ms=row_ts,
            coin=coin,
            best_bid=bid_px,
            best_ask=ask_px,
            bid_size=bid_sz,
            ask_size=ask_sz,
            source_session_id=source_session_id,
            source_channel=source_channel,
        )
    except ValueError:
        return None


def _iter_raw_records(path: Path) -> Iterator[tuple[dict[str, Any] | None, str | None]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped), None
            except json.JSONDecodeError as exc:
                yield None, str(exc)


def extract_bid_ask_rows_from_raw_jsonl(
    path: Path,
    *,
    coins: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], ExtractStats]:
    """Parse raw JSONL BBO messages safely."""
    rows: list[dict[str, Any]] = []
    stats = ExtractStats()

    for record, error in _iter_raw_records(path):
        if error or record is None:
            stats.malformed_lines_skipped += 1
            continue

        channel = record.get("channel", "")
        if channel not in RAW_BBO_CHANNELS | RAW_L2_CHANNELS:
            continue

        raw_payload = record.get("raw_payload", {})
        if not isinstance(raw_payload, dict):
            stats.invalid_rows_skipped += 1
            continue

        data = raw_payload.get("data", raw_payload)
        if not isinstance(data, dict):
            stats.invalid_rows_skipped += 1
            continue

        coin = str(data.get("coin", ""))
        if coins and coin not in coins:
            continue

        source_session_id = str(record.get("recorder_session_id", ""))
        received_ms = int(record.get("received_timestamp_ms") or 0)
        exchange_ms = record.get("exchange_timestamp_ms")
        fallback_ts = int(exchange_ms) if exchange_ms is not None else received_ms

        if channel in RAW_BBO_CHANNELS:
            row = _parse_bbo_payload(
                data,
                timestamp_ms=fallback_ts,
                source_session_id=source_session_id,
                source_channel="bbo",
            )
        else:
            row = _parse_l2_payload(
                data,
                timestamp_ms=fallback_ts,
                source_session_id=source_session_id,
                source_channel="l2Book",
            )

        if row is None:
            stats.invalid_rows_skipped += 1
            continue

        rows.append(row)
        stats.rows_extracted += 1

    return rows, stats


def _normalized_row_to_bid_ask(
    row: dict[str, Any],
    *,
    source_channel: str,
) -> dict[str, Any] | None:
    coin = str(row.get("coin", ""))
    best_bid = _safe_float(row.get("best_bid_price"))
    best_ask = _safe_float(row.get("best_ask_price"))
    if best_bid is None or best_ask is None:
        return None

    bid_size = _safe_float(row.get("best_bid_size"))
    ask_size = _safe_float(row.get("best_ask_size"))
    if bid_size is None and ask_size is None and source_channel == "l2Book":
        bids_raw = row.get("bids")
        asks_raw = row.get("asks")
        if isinstance(bids_raw, str):
            try:
                bids = json.loads(bids_raw)
            except json.JSONDecodeError:
                bids = []
        elif isinstance(bids_raw, list):
            bids = bids_raw
        else:
            bids = []
        if isinstance(asks_raw, str):
            try:
                asks = json.loads(asks_raw)
            except json.JSONDecodeError:
                asks = []
        elif isinstance(asks_raw, list):
            asks = asks_raw
        else:
            asks = []
        if bids:
            bid_size = _safe_float(bids[0][1] if isinstance(bids[0], (list, tuple)) else None)
        if asks:
            ask_size = _safe_float(asks[0][1] if isinstance(asks[0], (list, tuple)) else None)

    exchange_ts = row.get("exchange_timestamp_ms")
    received_ts = row.get("received_timestamp_ms")
    if exchange_ts is not None:
        timestamp_ms = int(exchange_ts)
    elif received_ts is not None:
        timestamp_ms = int(received_ts)
    else:
        return None

    source_session_id = str(row.get("source_session_id", row.get("recorder_session_id", "")))

    try:
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
    except ValueError:
        return None


def extract_bid_ask_rows_from_normalized(
    path: Path,
    *,
    coins: tuple[str, ...] | None = None,
    source_channel: str = "bbo",
) -> tuple[list[dict[str, Any]], ExtractStats]:
    """Parse normalized BBO parquet/JSONL if available."""
    rows: list[dict[str, Any]] = []
    stats = ExtractStats()

    if path.suffix == ".parquet":
        if not _PARQUET_AVAILABLE or pq is None:
            return rows, stats
        try:
            table = pq.read_table(str(path))
            raw_rows = table.to_pylist()
        except Exception:
            stats.malformed_lines_skipped += 1
            return rows, stats
    else:
        raw_rows = []
        for record, error in _iter_raw_records(path):
            if error or record is None:
                stats.malformed_lines_skipped += 1
                continue
            raw_rows.append(record)

    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            stats.invalid_rows_skipped += 1
            continue
        coin = str(raw_row.get("coin", ""))
        if coins and coin not in coins:
            continue
        row = _normalized_row_to_bid_ask(raw_row, source_channel=source_channel)
        if row is None:
            stats.invalid_rows_skipped += 1
            continue
        rows.append(row)
        stats.rows_extracted += 1

    return rows, stats


def sample_to_1s(
    rows: list[dict[str, Any]],
    *,
    forward_fill_max_gap_seconds: int | None = None,
) -> SamplingResult:
    """Convert tick-level BBO data into 1-second rows (last observation per second)."""
    if not rows:
        return SamplingResult(
            rows=[],
            missing_seconds=0,
            duplicate_timestamps=0,
            coverage_start_ms=None,
            coverage_end_ms=None,
        )

    sorted_rows = sorted(rows, key=lambda row: int(row["timestamp_ms"]))
    timestamp_counts: dict[int, int] = {}
    for row in sorted_rows:
        ts = int(row["timestamp_ms"])
        timestamp_counts[ts] = timestamp_counts.get(ts, 0) + 1
    duplicate_timestamps = sum(count - 1 for count in timestamp_counts.values() if count > 1)

    by_second: dict[int, dict[str, Any]] = {}
    for row in sorted_rows:
        second = int(row["timestamp_ms"]) // 1000
        by_second[second] = row

    if forward_fill_max_gap_seconds is not None and by_second:
        filled: dict[int, dict[str, Any]] = {}
        seconds_sorted = sorted(by_second)
        last_row: dict[str, Any] | None = None
        for second in range(seconds_sorted[0], seconds_sorted[-1] + 1):
            if second in by_second:
                last_row = by_second[second]
                filled[second] = last_row
                continue
            if last_row is None:
                continue
            gap = second - (int(last_row["timestamp_ms"]) // 1000)
            if gap <= forward_fill_max_gap_seconds:
                filled[second] = dict(last_row)
                filled[second]["timestamp_ms"] = second * 1000
                filled[second]["timestamp_iso"] = datetime.fromtimestamp(
                    second, tz=timezone.utc
                ).isoformat().replace("+00:00", "Z")
        by_second = filled

    sampled_rows = [by_second[second] for second in sorted(by_second)]
    coverage_start_ms = int(sampled_rows[0]["timestamp_ms"]) if sampled_rows else None
    coverage_end_ms = int(sampled_rows[-1]["timestamp_ms"]) if sampled_rows else None

    missing_seconds = 0
    if sampled_rows and coverage_start_ms is not None and coverage_end_ms is not None:
        start_second = coverage_start_ms // 1000
        end_second = coverage_end_ms // 1000
        expected_seconds = end_second - start_second + 1
        missing_seconds = max(0, expected_seconds - len(by_second))

    return SamplingResult(
        rows=sampled_rows,
        missing_seconds=missing_seconds,
        duplicate_timestamps=duplicate_timestamps,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
    )


def _validate_output_path(output_path: Path, output_root: Path) -> None:
    allowed_dir = (output_root / CSV_1S_SUBDIR).resolve()
    resolved = output_path.resolve()
    if resolved != allowed_dir and allowed_dir not in resolved.parents:
        raise ValueError(f"Output path must be under {allowed_dir}")


def write_bid_ask_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Write output only under data/bid_ask_csv/1s/."""
    _validate_output_path(output_path, output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BID_ASK_CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in BID_ASK_CSV_COLUMNS})
    return output_path


def save_extraction_report(
    report: ExtractionReport,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Save JSON extraction report under data/bid_ask_csv/reports/."""
    reports_dir = output_root / REPORTS_SUBDIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = reports_dir / f"{report.coin}_{timestamp}.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report_path


def extract_rows_from_source(
    source: BidAskSource,
    *,
    coins: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], ExtractStats]:
    if source.source_kind == "raw_jsonl":
        return extract_bid_ask_rows_from_raw_jsonl(source.path, coins=coins)
    return extract_bid_ask_rows_from_normalized(
        source.path,
        coins=coins,
        source_channel=source.channel,
    )


def extract_coin_bid_ask_1s(
    coin: str,
    *,
    market_stream_dir: Path = DEFAULT_MARKET_STREAM_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    forward_fill_max_gap_seconds: int | None = None,
) -> ExtractionReport:
    """Extract 1-second bid/ask CSV and report for one coin from local data."""
    sources = discover_local_bid_ask_sources(market_stream_dir, coins=(coin,))
    used_files: list[str] = []
    all_rows: list[dict[str, Any]] = []
    invalid_rows_skipped = 0

    for source in sources:
        rows, stats = extract_rows_from_source(source, coins=(coin,))
        if not rows:
            continue
        used_files.append(str(source.path))
        all_rows.extend(rows)
        invalid_rows_skipped += stats.invalid_rows_skipped + stats.malformed_lines_skipped

    sampling = sample_to_1s(
        all_rows,
        forward_fill_max_gap_seconds=forward_fill_max_gap_seconds,
    )
    output_path = output_root / CSV_1S_SUBDIR / f"{coin}_1s.csv"
    write_bid_ask_csv(sampling.rows, output_path, output_root=output_root)

    report = ExtractionReport(
        coin=coin,
        source_files=sorted(set(used_files)),
        output_path=str(output_path),
        row_count=len(sampling.rows),
        coverage_start_ms=sampling.coverage_start_ms,
        coverage_end_ms=sampling.coverage_end_ms,
        missing_seconds=sampling.missing_seconds,
        duplicate_timestamps=sampling.duplicate_timestamps,
        invalid_rows_skipped=invalid_rows_skipped,
        locally_collected_data_only=True,
    )
    save_extraction_report(report, output_root=output_root)
    return report
