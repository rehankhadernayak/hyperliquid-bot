"""Deterministic replay engine for recorded Hyperliquid WebSocket sessions."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from market_data.quality_models import QualityLabel
from market_data.replay_clock import ReplayClock
from market_data.replay_cursor import ReplayCursor
from market_data.replay_events import (
    MarketDataReplayConsumer,
    NullConsumer,
    ReplayContext,
    ReplayEvent,
    ReplaySummary,
    build_replay_event,
    channel_for_event_type,
    row_to_event,
)
from market_data.replay_filters import ReplayFilters, apply_filters
from market_data.replay_manifest import SessionManifest, load_session_manifest
from market_data.replay_ordering import (
    analyze_ordering,
    assign_replay_sequences,
    primary_time_for_event,
    sort_events_deterministically,
)
from market_data.replay_report import build_replay_report, write_replay_report


class ReplayError(Exception):
    """Raised when replay cannot proceed safely."""


@dataclass
class LoadedSession:
    manifest: SessionManifest
    quality_report: dict[str, Any] | None
    quality_report_path: Path | None
    raw_files_used: list[str] = field(default_factory=list)
    normalized_files_used: list[str] = field(default_factory=list)
    source_rows_read: int = 0


def _parse_event_type_from_path(path: Path) -> str | None:
    match = re.search(r"event_type=([^/]+)", str(path).replace("\\", "/"))
    return match.group(1) if match else None


def load_quality_report_for_session(
    session_id: str,
    quality_reports_dir: Path,
) -> tuple[dict[str, Any] | None, Path | None]:
    path = quality_reports_dir / f"{session_id}.json"
    if not path.exists():
        return None, None
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle), path
    except (json.JSONDecodeError, OSError):
        return None, path


def discover_normalized_files(
    manifest: SessionManifest,
    market_stream_dir: Path,
    repo_root: Path | None = None,
) -> list[Path]:
    """Discover normalized files from manifest or partition scan."""
    files: list[Path] = []

    for entry in manifest.normalized_files:
        if entry.exists and entry.readable:
            files.append(entry.path)

    if files:
        return sorted(set(files))

    norm_dir = market_stream_dir / "normalized"
    if not norm_dir.exists():
        return []

    for path in sorted(norm_dir.rglob("*.parquet")):
        files.append(path)
    for path in sorted(norm_dir.rglob("*.jsonl")):
        files.append(path)
    return sorted(set(files))


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        # Read single file directly — avoid hive-partition dataset inference on path.
        return pq.ParquetFile(str(path)).read().to_pylist()
    except Exception:
        return []


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except OSError:
        pass
    return rows


def load_normalized_events(
    files: list[Path],
    replay_session_id: str,
    source_session_id: str,
) -> tuple[list[ReplayEvent], int]:
    """Load events from normalized Parquet/JSONL files."""
    records: list[ReplayEvent] = []
    rows_read = 0
    seq = 0

    for path in files:
        event_type = _parse_event_type_from_path(path)
        if not event_type:
            continue

        if path.suffix == ".parquet":
            rows = _read_parquet_rows(path)
        elif path.suffix == ".jsonl":
            rows = _read_jsonl_rows(path)
        else:
            continue

        for row_index, row in enumerate(rows):
            rows_read += 1
            try:
                event = row_to_event(dict(row), event_type)
            except (TypeError, ValueError, KeyError):
                continue

            primary_time, primary_source = primary_time_for_event(event, event_type)
            seq += 1
            records.append(
                build_replay_event(
                    event=event,
                    event_type=event_type,
                    replay_session_id=replay_session_id,
                    source_recorder_session_id=source_session_id,
                    source_file=str(path),
                    source_row_index=row_index,
                    replay_sequence=seq,
                    primary_time_ms=primary_time,
                    primary_time_source=primary_source,
                )
            )

    return records, rows_read


def validate_raw_files(manifest: SessionManifest) -> list[str]:
    """Validate raw files exist and are readable (fallback/validation only)."""
    warnings: list[str] = []
    for entry in manifest.raw_files:
        if not entry.exists:
            warnings.append(f"Raw file missing: {entry.path}")
        elif not entry.readable:
            warnings.append(f"Raw file unreadable: {entry.path}")
    return warnings


def check_quality_gate(
    quality_report: dict[str, Any] | None,
    allow_degraded: bool,
) -> None:
    """Block replay for UNUSABLE sessions unless allow_degraded is set."""
    if quality_report is None:
        return

    label = quality_report.get("quality_label")
    if label == QualityLabel.UNUSABLE.value and not allow_degraded:
        raise ReplayError(
            f"Quality report marks session UNUSABLE. "
            f"Use --allow-degraded to replay anyway."
        )


def inherited_quality_warnings(quality_report: dict[str, Any] | None) -> list[str]:
    if not quality_report:
        return []
    warnings: list[str] = []
    for item in quality_report.get("data_integrity_warnings") or []:
        if isinstance(item, dict):
            warnings.append(item.get("message", str(item)))
        else:
            warnings.append(str(item))
    gap_total = (quality_report.get("gap_summary") or {}).get("total_gaps", 0)
    if gap_total:
        warnings.append(f"Quality report recorded {gap_total} gap(s)")
    return warnings[:20]


@dataclass
class ReplayResult:
    report: dict[str, Any]
    report_paths: dict[str, Path]
    events: list[ReplayEvent]
    cursor: ReplayCursor


class ReplayEngine:
    """Deterministic replay engine for recorded WebSocket market data."""

    def __init__(
        self,
        market_stream_dir: Path,
        repo_root: Path | None = None,
        test_mode: bool = False,
    ) -> None:
        self._market_stream_dir = market_stream_dir
        self._repo_root = repo_root
        self._test_mode = test_mode

    def load_session(
        self,
        session_id: str,
        allow_degraded: bool = False,
    ) -> LoadedSession:
        manifest = load_session_manifest(session_id, self._market_stream_dir, self._repo_root)
        if not manifest.exists:
            raise ReplayError(manifest.load_error or f"Manifest not found for {session_id}")

        quality_dir = self._market_stream_dir / "quality_reports"
        quality_report, quality_path = load_quality_report_for_session(session_id, quality_dir)
        check_quality_gate(quality_report, allow_degraded)

        norm_files = discover_normalized_files(manifest, self._market_stream_dir, self._repo_root)
        raw_warnings = validate_raw_files(manifest)

        return LoadedSession(
            manifest=manifest,
            quality_report=quality_report,
            quality_report_path=quality_path,
            raw_files_used=[str(e.path) for e in manifest.raw_files if e.exists],
            normalized_files_used=[str(p) for p in norm_files],
            source_rows_read=0,
        )

    def prepare_events(
        self,
        session: LoadedSession,
        filters: ReplayFilters,
        replay_session_id: str | None = None,
    ) -> tuple[list[ReplayEvent], int, LoadedSession]:
        replay_id = replay_session_id or uuid.uuid4().hex
        session_id = session.manifest.session_id

        norm_paths = [Path(p) for p in session.normalized_files_used]
        events, rows_read = load_normalized_events(norm_paths, replay_id, session_id)
        session.source_rows_read = rows_read

        if not events:
            raise ReplayError("No replayable events found in normalized files")

        sorted_events = sort_events_deterministically(events)
        sequenced = assign_replay_sequences(sorted_events)

        quality_label = (session.quality_report or {}).get("quality_label")
        filtered, skipped = apply_filters(sequenced, filters, quality_label)
        filtered = assign_replay_sequences(filtered)

        return filtered, skipped, session

    def run(
        self,
        session_id: str,
        *,
        filters: ReplayFilters | None = None,
        speed: float = 1.0,
        dry_run: bool = True,
        allow_degraded: bool = False,
        output_dir: Path | None = None,
        consumer: MarketDataReplayConsumer | None = None,
        replay_session_id: str | None = None,
    ) -> ReplayResult:
        filters = filters or ReplayFilters()
        replay_id = replay_session_id or uuid.uuid4().hex
        start_wall = time.time()

        session = self.load_session(session_id, allow_degraded=allow_degraded)
        events, skipped, session = self.prepare_events(session, filters, replay_id)

        ordering_analysis = analyze_ordering(events)
        inherited = inherited_quality_warnings(session.quality_report)

        if dry_run:
            consumer_impl: MarketDataReplayConsumer = NullConsumer()
            emitted = 0
        else:
            consumer_impl = consumer or NullConsumer()
            emitted = self._emit_events(events, consumer_impl, replay_id, session_id, filters, speed)

        end_wall = time.time()
        duration_ms = int((end_wall - start_wall) * 1000)

        generated_at = datetime.now(timezone.utc).isoformat()
        quality_label = (session.quality_report or {}).get("quality_label")
        quality_path = str(session.quality_report_path) if session.quality_report_path else None

        first_time = events[0].primary_time_ms if events else None
        last_time = events[-1].primary_time_ms if events else None

        report = build_replay_report(
            replay_session_id=replay_id,
            source_recorder_session_id=session_id,
            quality_report_path=quality_path,
            quality_label=quality_label,
            generated_at=generated_at,
            replay_start_ms=first_time,
            replay_end_ms=last_time,
            filters=filters.to_dict(),
            speed=speed,
            dry_run=dry_run,
            events=events,
            skipped_events=skipped,
            ordering_analysis=ordering_analysis,
            inherited_quality_warnings=inherited,
            source_files_read=session.normalized_files_used,
            source_rows_read=session.source_rows_read,
            consumer_type=type(consumer_impl).__name__,
            emitted_count=emitted if not dry_run else 0,
            replay_duration_ms=duration_ms,
            allow_degraded=allow_degraded,
        )

        report_paths: dict[str, Path] = {}
        if output_dir is not None:
            report_paths = write_replay_report(report, output_dir)

        cursor = ReplayCursor(events)
        return ReplayResult(report=report, report_paths=report_paths, events=events, cursor=cursor)

    def _emit_events(
        self,
        events: list[ReplayEvent],
        consumer: MarketDataReplayConsumer,
        replay_session_id: str,
        source_session_id: str,
        filters: ReplayFilters,
        speed: float,
    ) -> int:
        context = ReplayContext(
            replay_session_id=replay_session_id,
            source_recorder_session_id=source_session_id,
            dry_run=False,
            speed=speed,
            filters=filters.to_dict(),
        )
        clock = ReplayClock(
            speed=speed,
            start_timestamp_ms=filters.start_timestamp_ms,
            end_timestamp_ms=filters.end_timestamp_ms,
            max_events=filters.max_events,
            test_mode=self._test_mode,
        )

        consumer.on_start(context)
        emitted = 0

        if events:
            clock.start(events[0].primary_time_ms)

        for event in events:
            if not clock.should_emit(event.primary_time_ms):
                continue
            clock.wait_until(event.primary_time_ms)
            emitted_ms = clock.replay_wall_time_ms()
            emitted_event = event.with_emitted(emitted_ms)
            consumer.on_event(emitted_event)
            clock.record_emission()
            emitted += 1

        summary = ReplaySummary(
            total_events=len(events),
            emitted_count=emitted,
            skipped_events=0,
            stream_hash=report_stream_hash(events),
            first_event_time_ms=events[0].primary_time_ms if events else None,
            last_event_time_ms=events[-1].primary_time_ms if events else None,
            replay_duration_ms=clock.replay_wall_time_ms() - (clock._replay_wall_start_ms or 0),
            dry_run=False,
        )
        consumer.on_finish(summary)
        return emitted


def report_stream_hash(events: list[ReplayEvent]) -> str:
    from market_data.replay_events import compute_stream_hash

    return compute_stream_hash(events)
