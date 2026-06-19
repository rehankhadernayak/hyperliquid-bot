"""Replay report generation (JSON and markdown)."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from market_data.replay_events import compute_stream_hash
from market_data.replay_ordering import ORDERING_RULE, OrderingAnalysis

REPLAY_REPORT_SCHEMA_VERSION = "1.0.0"


def build_replay_report(
    *,
    replay_session_id: str,
    source_recorder_session_id: str,
    quality_report_path: str | None,
    quality_label: str | None,
    generated_at: str,
    replay_start_ms: int | None,
    replay_end_ms: int | None,
    filters: dict[str, Any],
    speed: float,
    dry_run: bool,
    events: list[Any],
    skipped_events: int,
    ordering_analysis: OrderingAnalysis,
    inherited_quality_warnings: list[str],
    source_files_read: list[str],
    source_rows_read: int,
    consumer_type: str,
    emitted_count: int,
    replay_duration_ms: int,
    allow_degraded: bool,
) -> dict[str, Any]:
    by_channel: Counter[str] = Counter()
    by_coin: Counter[str] = Counter()
    first_time: int | None = None
    last_time: int | None = None

    for event in events:
        ch = event.channel or event.original_event_type
        by_channel[ch] += 1
        coin = event.coin or "global"
        by_coin[coin] += 1
        if first_time is None or event.primary_time_ms < first_time:
            first_time = event.primary_time_ms
        if last_time is None or event.primary_time_ms > last_time:
            last_time = event.primary_time_ms

    stream_hash = compute_stream_hash(events)

    return {
        "schema_version": REPLAY_REPORT_SCHEMA_VERSION,
        "replay_session_id": replay_session_id,
        "source_recorder_session_id": source_recorder_session_id,
        "quality_report_path": quality_report_path,
        "quality_label": quality_label,
        "generated_at": generated_at,
        "replay_start_ms": replay_start_ms,
        "replay_end_ms": replay_end_ms,
        "filters": filters,
        "ordering_rule": ORDERING_RULE,
        "speed": speed,
        "dry_run": dry_run,
        "allow_degraded": allow_degraded,
        "event_counts_by_channel": dict(sorted(by_channel.items())),
        "event_counts_by_coin": dict(sorted(by_coin.items())),
        "first_event_time_ms": first_time,
        "last_event_time_ms": last_time,
        "total_events": len(events),
        "skipped_events": skipped_events,
        "missing_timestamp_counts": ordering_analysis.missing_timestamp_counts,
        "events_ordered_by_exchange_timestamp": ordering_analysis.events_ordered_by_exchange_timestamp,
        "events_ordered_by_received_timestamp": ordering_analysis.events_ordered_by_received_timestamp,
        "tie_breaker_usage": ordering_analysis.tie_breaker_usage,
        "ordering_warnings": ordering_analysis.ordering_warnings,
        "data_quality_warnings_inherited": inherited_quality_warnings,
        "source_files_read": source_files_read,
        "source_rows_read": source_rows_read,
        "consumer_type": consumer_type,
        "emitted_count": emitted_count,
        "replay_duration_ms": replay_duration_ms,
        "stream_hash": stream_hash,
        "paper_only": True,
        "execution_enabled": False,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Market Data Replay Report",
        "",
        f"**Replay session:** `{report.get('replay_session_id', '—')}`",
        f"**Source recorder session:** `{report.get('source_recorder_session_id', '—')}`",
        f"**Generated at:** {report.get('generated_at', '—')}",
        "",
        "## Summary",
        "",
        f"- **Quality label:** {report.get('quality_label', '—')}",
        f"- **Total events:** {report.get('total_events', 0)}",
        f"- **Emitted count:** {report.get('emitted_count', 0)}",
        f"- **Skipped events:** {report.get('skipped_events', 0)}",
        f"- **Dry run:** {report.get('dry_run', True)}",
        f"- **Speed:** {report.get('speed', 1.0)}",
        f"- **Stream hash:** `{report.get('stream_hash', '—')}`",
        "",
        "## Filters",
        "",
        "```json",
        json.dumps(report.get("filters") or {}, indent=2),
        "```",
        "",
        "## Event counts by channel",
        "",
    ]

    for ch, count in sorted((report.get("event_counts_by_channel") or {}).items()):
        lines.append(f"- **{ch}:** {count}")

    lines.extend(["", "## Event counts by coin", ""])
    for coin, count in sorted((report.get("event_counts_by_coin") or {}).items()):
        lines.append(f"- **{coin}:** {count}")

    lines.extend(
        [
            "",
            "## Ordering rule",
            "",
            report.get("ordering_rule", ORDERING_RULE),
            "",
            "## Warnings",
            "",
        ]
    )

    ordering_warnings = report.get("ordering_warnings") or []
    inherited = report.get("data_quality_warnings_inherited") or []
    if not ordering_warnings and not inherited:
        lines.append("_No warnings._")
    else:
        for warning in ordering_warnings:
            lines.append(f"- {warning}")
        for warning in inherited:
            lines.append(f"- _(inherited)_ {warning}")

    lines.extend(
        [
            "",
            "## Safety",
            "",
            "This replay is **read-only** and **paper-only**. "
            "No wallets, signing, orders, or execution endpoints are used. "
            "Replay output is not connected to live trading decisions.",
            "",
        ]
    )
    return "\n".join(lines)


def write_replay_report(
    report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_id = report["replay_session_id"]
    json_path = output_dir / f"{replay_id}.json"
    md_path = output_dir / f"{replay_id}.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def list_replay_reports(replays_dir: Path) -> list[dict[str, Any]]:
    """List replay report JSON files sorted newest first."""
    if not replays_dir.exists():
        return []

    reports: list[dict[str, Any]] = []
    for path in sorted(replays_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            reports.append(
                {
                    "replay_session_id": data.get("replay_session_id", path.stem),
                    "source_recorder_session_id": data.get("source_recorder_session_id"),
                    "generated_at": data.get("generated_at"),
                    "total_events": data.get("total_events", 0),
                    "stream_hash": data.get("stream_hash"),
                    "dry_run": data.get("dry_run", True),
                    "quality_label": data.get("quality_label"),
                    "warning_count": len(data.get("ordering_warnings") or [])
                    + len(data.get("data_quality_warnings_inherited") or []),
                    "event_counts_by_channel": data.get("event_counts_by_channel") or {},
                    "event_counts_by_coin": data.get("event_counts_by_coin") or {},
                    "json_path": str(path),
                    "markdown_path": str(path.with_suffix(".md")),
                }
            )
        except (json.JSONDecodeError, OSError):
            continue
    return reports
