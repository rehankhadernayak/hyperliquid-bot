"""Read-only local market data archive index for dashboard charts.

Scans raw preview, normalized parquet, and optional archive directories.
Does not rewrite, normalize, or modify any market data files.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dashboard_data

ALL_LOCAL_LABEL = (
    "All local data — this is not full exchange history unless historical backfill has been imported."
)

MISSING_HISTORY_WARNING = (
    "Only locally collected data is available for this range. "
    "Historical backfill has not been imported yet."
)

BACKFILL_PLACEHOLDER_TITLE = "Historical backfill: not imported"
BACKFILL_PLACEHOLDER_BODY = (
    "Local recorder data exists only for the dates collected by websocket_recorder.py. "
    "6M/All exchange history requires a separate read-only historical backfill step. "
    "Backfill is not used for live trading. "
    "Backfill data must be labeled separately from WebSocket-recorded data."
)

RAW_PREVIEW_TAIL_BYTES = 4_096
ARCHIVE_DIR_NAME = "archive"


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_jsonl_first_timestamp(path: Path) -> int | None:
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                ts = _safe_int(record.get("received_timestamp_ms") or record.get("exchange_timestamp_ms"))
                if ts is not None and ts > 0:
                    return ts
                break
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return None


def _read_jsonl_last_timestamp(path: Path) -> int | None:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size <= 0:
        return None

    read_size = min(size, RAW_PREVIEW_TAIL_BYTES)
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, size - read_size))
            chunk = handle.read(read_size)
    except OSError:
        return None

    text = chunk.decode("utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    if size > read_size and lines:
        lines = lines[1:]

    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = _safe_int(record.get("received_timestamp_ms") or record.get("exchange_timestamp_ms"))
        if ts is not None and ts > 0:
            return ts
    return None


def _scan_raw_preview(stream_dir: Path) -> dict[str, Any]:
    raw_dir = stream_dir / "raw"
    if not raw_dir.exists():
        return {
            "available": False,
            "earliest_timestamp_ms": None,
            "latest_timestamp_ms": None,
            "file_count": 0,
            "row_count_estimate": None,
            "channels": [],
            "sessions": [],
        }

    try:
        files = sorted(
            (path for path in raw_dir.rglob("*.jsonl") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
        )
    except OSError:
        files = []
    files = files[-50:]

    if not files:
        return {
            "available": False,
            "earliest_timestamp_ms": None,
            "latest_timestamp_ms": None,
            "file_count": 0,
            "row_count_estimate": None,
            "channels": [],
            "sessions": [],
        }

    earliest = _read_jsonl_first_timestamp(files[0])
    latest = _read_jsonl_last_timestamp(files[-1])
    sessions: set[str] = set()
    channels: set[str] = set()
    for path in files[:20]:
        try:
            with path.open(encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    if idx >= 50:
                        break
                    stripped = line.strip()
                    if not stripped:
                        continue
                    record = json.loads(stripped)
                    session = record.get("recorder_session_id")
                    if session:
                        sessions.add(str(session))
                    channel = record.get("channel")
                    if channel:
                        channels.add(str(channel))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue

    return {
        "available": True,
        "earliest_timestamp_ms": earliest,
        "latest_timestamp_ms": latest,
        "file_count": len(files),
        "row_count_estimate": None,
        "channels": sorted(channels),
        "sessions": sorted(sessions),
    }


def _parse_partition_date(name: str) -> datetime | None:
    if not name.startswith("date="):
        return None
    text = name.split("=", 1)[1]
    try:
        return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _scan_normalized(stream_dir: Path) -> dict[str, Any]:
    normalized_dir = stream_dir / "normalized"
    if not normalized_dir.exists():
        return {
            "available": False,
            "earliest_timestamp_ms": None,
            "latest_timestamp_ms": None,
            "file_count": 0,
            "row_count_estimate": 0,
            "coins": [],
            "event_types": [],
        }

    coins: set[str] = set()
    event_types: set[str] = set()
    dates: list[datetime] = []
    file_count = 0
    row_count = 0
    metadata_reads = 0
    max_metadata_reads = 32

    try:
        type_dirs = list(normalized_dir.glob("event_type=*"))
    except OSError:
        type_dirs = []

    for type_dir in type_dirs:
        event_type = type_dir.name.split("=", 1)[-1]
        event_types.add(event_type)
        try:
            date_dirs = list(type_dir.glob("date=*"))
        except OSError:
            continue
        for date_dir in date_dirs:
            parsed = _parse_partition_date(date_dir.name)
            if parsed is not None:
                dates.append(parsed)
            try:
                coin_dirs = list(date_dir.glob("coin=*"))
            except OSError:
                continue
            for coin_dir in coin_dirs:
                coins.add(coin_dir.name.split("=", 1)[-1])
                try:
                    parquet_files = list(coin_dir.glob("*.parquet"))
                except OSError:
                    continue
                file_count += len(parquet_files)
                for parquet_path in parquet_files:
                    if metadata_reads >= max_metadata_reads:
                        continue
                    row_count += _parquet_row_count(parquet_path)
                    metadata_reads += 1

    earliest_ms = None
    latest_ms = None
    if dates:
        earliest_ms = int(min(dates).timestamp() * 1000)
        latest_day = max(dates)
        latest_ms = int((latest_day.timestamp() + 86400) * 1000)

    return {
        "available": file_count > 0,
        "earliest_timestamp_ms": earliest_ms,
        "latest_timestamp_ms": latest_ms,
        "file_count": file_count,
        "row_count_estimate": row_count,
        "coins": sorted(coins),
        "event_types": sorted(event_types),
    }


def _parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        metadata = pq.read_metadata(path)
        return metadata.num_rows
    except Exception:
        return 0


def _scan_archive_dir(stream_dir: Path) -> dict[str, Any]:
    archive_dir = stream_dir / ARCHIVE_DIR_NAME
    if not archive_dir.exists():
        return {
            "available": False,
            "earliest_timestamp_ms": None,
            "latest_timestamp_ms": None,
            "file_count": 0,
            "row_count_estimate": 0,
        }

    try:
        files = sorted(
            (path for path in archive_dir.rglob("*") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
        )
    except OSError:
        files = []

    earliest = None
    latest = None
    row_count = 0
    for path in files:
        if path.suffix == ".jsonl":
            first_ts = _read_jsonl_first_timestamp(path)
            last_ts = _read_jsonl_last_timestamp(path)
            if first_ts is not None:
                earliest = first_ts if earliest is None else min(earliest, first_ts)
            if last_ts is not None:
                latest = last_ts if latest is None else max(latest, latest)
        elif path.suffix == ".parquet":
            row_count += _parquet_row_count(path)

    return {
        "available": bool(files),
        "earliest_timestamp_ms": earliest,
        "latest_timestamp_ms": latest,
        "file_count": len(files),
        "row_count_estimate": row_count,
    }


def _load_manifest_sessions(stream_dir: Path) -> list[str]:
    manifests_dir = stream_dir / "manifests"
    if not manifests_dir.exists():
        return []
    sessions: list[str] = []
    try:
        for path in manifests_dir.glob("*.json"):
            if path.name.endswith(".checkpoint.json"):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            session_id = payload.get("recorder_session_id")
            if session_id:
                sessions.append(str(session_id))
    except OSError:
        return []
    return sorted(set(sessions))


def build_market_archive_index(*, stream_dir: Path | None = None) -> dict[str, Any]:
    """Build a lightweight summary of locally stored market data (read-only)."""
    root = stream_dir or dashboard_data.MARKET_STREAM_DIR
    raw = _scan_raw_preview(root)
    normalized = _scan_normalized(root)
    archive = _scan_archive_dir(root)
    sessions = _load_manifest_sessions(root)

    timestamps = [
        ts
        for ts in (
            raw.get("earliest_timestamp_ms"),
            normalized.get("earliest_timestamp_ms"),
            archive.get("earliest_timestamp_ms"),
        )
        if ts is not None
    ]
    latest_timestamps = [
        ts
        for ts in (
            raw.get("latest_timestamp_ms"),
            normalized.get("latest_timestamp_ms"),
            archive.get("latest_timestamp_ms"),
        )
        if ts is not None
    ]

    data_sources: list[str] = []
    if raw.get("available"):
        data_sources.append("raw_preview")
    if normalized.get("available"):
        data_sources.append("normalized")
    if archive.get("available"):
        data_sources.append("archive")

    coins = sorted(set(normalized.get("coins") or []))
    channels = sorted(set(raw.get("channels") or []))

    return {
        "stream_dir": str(root),
        "earliest_timestamp_ms": min(timestamps) if timestamps else None,
        "latest_timestamp_ms": max(latest_timestamps) if latest_timestamps else None,
        "coins": coins,
        "channels": channels,
        "data_sources": data_sources,
        "sessions": sessions or raw.get("sessions") or [],
        "raw_preview": raw,
        "normalized": normalized,
        "archive": archive,
        "raw_preview_rows": raw.get("row_count_estimate"),
        "normalized_rows": normalized.get("row_count_estimate", 0),
        "archive_rows": archive.get("row_count_estimate", 0),
        "all_local_label": ALL_LOCAL_LABEL,
        "missing_history_warning": MISSING_HISTORY_WARNING,
    }


def resolve_chart_range_bounds(
    range_key: str,
    *,
    archive_index: dict[str, Any] | None = None,
    reference_time_ms: int | None = None,
    prefer_latest_local_reference: bool = False,
) -> dict[str, Any]:
    """Map a chart range key to requested start/end timestamps and mode."""
    from dashboard_charts import CHART_RANGE_OPTIONS

    archive_index = archive_index or build_market_archive_index()
    spec = CHART_RANGE_OPTIONS.get(range_key, CHART_RANGE_OPTIONS["Live 5m"])
    mode = spec["mode"]
    window_seconds = spec.get("seconds")

    if reference_time_ms is None:
        import time

        reference_time_ms = int(time.time() * 1000)

    earliest_local = archive_index.get("earliest_timestamp_ms")
    latest_local = archive_index.get("latest_timestamp_ms") or reference_time_ms

    if prefer_latest_local_reference and latest_local is not None:
        reference_time_ms = latest_local
    elif (
        latest_local is not None
        and latest_local < reference_time_ms
        and mode == "live"
    ):
        reference_time_ms = latest_local

    if mode == "all_local":
        requested_start_ms = earliest_local
        requested_end_ms = latest_local
        window_seconds = None
        if requested_start_ms is not None and requested_end_ms is not None:
            window_seconds = max(1, (requested_end_ms - requested_start_ms) // 1000)
    elif window_seconds is not None:
        requested_end_ms = latest_local if mode == "historical" else reference_time_ms
        requested_start_ms = requested_end_ms - (window_seconds * 1000)
    else:
        requested_start_ms = earliest_local
        requested_end_ms = reference_time_ms

    local_span_ms = None
    if earliest_local is not None and latest_local is not None:
        local_span_ms = max(0, latest_local - earliest_local)

    requested_span_ms = None
    if requested_start_ms is not None and requested_end_ms is not None:
        requested_span_ms = max(0, requested_end_ms - requested_start_ms)

    local_coverage_percent = None
    missing_history_warning = None
    if (
        mode in ("historical", "all_local")
        and requested_span_ms
        and local_span_ms is not None
        and local_span_ms < requested_span_ms
    ):
        local_coverage_percent = round(min(100.0, (local_span_ms / requested_span_ms) * 100.0), 1)
        missing_history_warning = MISSING_HISTORY_WARNING
    elif (
        mode in ("historical", "all_local")
        and requested_start_ms is not None
        and earliest_local is not None
        and requested_start_ms < earliest_local
    ):
        if requested_span_ms:
            covered = max(0, latest_local - earliest_local)
            local_coverage_percent = round(min(100.0, (covered / requested_span_ms) * 100.0), 1)
        missing_history_warning = MISSING_HISTORY_WARNING

    effective_start_ms = requested_start_ms
    if earliest_local is not None:
        if effective_start_ms is None:
            effective_start_ms = earliest_local
        elif effective_start_ms < earliest_local:
            effective_start_ms = earliest_local

    effective_end_ms = requested_end_ms
    if latest_local is not None and effective_end_ms is not None:
        effective_end_ms = min(effective_end_ms, latest_local)

    if (
        effective_start_ms is not None
        and effective_end_ms is not None
        and effective_start_ms > effective_end_ms
    ):
        if window_seconds:
            effective_start_ms = effective_end_ms - (window_seconds * 1000)
        elif earliest_local is not None:
            effective_start_ms = earliest_local

    range_empty = False
    if effective_start_ms is not None and effective_end_ms is not None:
        range_empty = effective_start_ms >= effective_end_ms

    return {
        "range_key": range_key,
        "mode": mode,
        "window_seconds": window_seconds or 0,
        "requested_range_start_ms": requested_start_ms,
        "requested_range_end_ms": requested_end_ms,
        "effective_start_ms": effective_start_ms,
        "effective_end_ms": effective_end_ms,
        "earliest_local_timestamp_ms": earliest_local,
        "latest_local_timestamp_ms": latest_local,
        "local_coverage_percent": local_coverage_percent,
        "missing_history_warning": missing_history_warning,
        "is_live_range": mode == "live",
        "is_all_local": mode == "all_local",
        "all_local_label": ALL_LOCAL_LABEL if mode == "all_local" else None,
        "prefer_latest_local_reference": prefer_latest_local_reference,
        "reference_time_ms": reference_time_ms,
        "selected_range_empty": range_empty,
    }
