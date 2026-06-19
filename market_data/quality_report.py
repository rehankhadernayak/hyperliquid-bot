"""Build and write market data quality reports."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from market_data.quality_checks import (
    assign_overall_label,
    build_recommendations,
    build_usability_flags,
    inspect_session,
)
from market_data.quality_models import (
    QUALITY_REPORT_SCHEMA_VERSION,
    CHANNEL_TO_EVENT_TYPE,
    QualityLabel,
)
from market_data.replay_manifest import SessionManifest, load_session_manifest


def generate_report_id() -> str:
    return uuid.uuid4().hex


def build_quality_report(
    session_id: str,
    market_stream_dir: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Inspect a recorder session and return a quality report dict."""
    manifest = load_session_manifest(session_id, market_stream_dir, repo_root)
    state = inspect_session(manifest)
    overall = assign_overall_label(state, manifest)
    usability = build_usability_flags(state, manifest, overall)
    recommendations = build_recommendations(state, manifest, overall)

    manifest_data = manifest.manifest_data or {}
    files_inspected = []
    file_sizes: dict[str, int] = {}
    row_counts: dict[str, int] = {}

    for entry in manifest.raw_files + manifest.normalized_files:
        path_str = str(entry.path)
        files_inspected.append(path_str)
        file_sizes[path_str] = entry.size_bytes
        if entry.row_count is not None:
            row_counts[path_str] = entry.row_count

    channel_summaries = {
        ch: summary.to_dict()
        for ch, summary in sorted(state.channel_summaries.items())
    }
    coin_summaries = {
        coin: summary.to_dict()
        for coin, summary in sorted(state.coin_summaries.items())
    }
    latency_summaries = {
        ch: summary.to_dict()
        for ch, summary in sorted(state.latency_by_channel.items())
    }

    bbo_spread_stats: dict[str, dict[str, float | None]] = {}
    for coin, spreads in state.bbo_spread_bps.items():
        if spreads:
            sorted_spreads = sorted(spreads)
            mid_idx = len(sorted_spreads) // 2
            p95_idx = int(len(sorted_spreads) * 0.95)
            bbo_spread_stats[coin] = {
                "median_spread_bps": sorted_spreads[mid_idx],
                "p95_spread_bps": sorted_spreads[min(p95_idx, len(sorted_spreads) - 1)],
            }

    gap_summary = {
        "total_gaps": len(state.gaps),
        "reconnect_gaps": sum(1 for g in state.gaps if g.gap_type == "RECONNECT_WINDOW"),
        "stale_channel_gaps": sum(1 for g in state.gaps if g.gap_type == "CHANNEL_STALENESS"),
        "candle_gaps": sum(1 for g in state.gaps if g.gap_type == "MISSING_CANDLE_PERIOD"),
        "backward_timestamp_events": max(
            state.backward_exchange_count,
            int((manifest.gap_statistics or {}).get("backward_timestamps") or 0),
        ),
        "gaps": [g.to_dict() for g in state.gaps],
    }

    raw_manifest = manifest.raw_row_count_manifest
    norm_manifest = manifest.normalized_row_count_manifest
    expansion_ratio = (
        state.normalized_inspected_count / state.raw_inspected_count
        if state.raw_inspected_count > 0
        else None
    )

    consistency = {
        "raw_count_manifest": raw_manifest,
        "raw_count_inspected": state.raw_inspected_count,
        "normalized_count_manifest": norm_manifest,
        "normalized_count_inspected": state.normalized_inspected_count,
        "normalized_to_raw_expansion_ratio": round(expansion_ratio, 4) if expansion_ratio else None,
        "duplicates_excluded_normalized": int(manifest_data.get("duplicate_count") or 0),
        "malformed_events": int(manifest_data.get("malformed_count") or 0) + state.raw_malformed_count,
        "note": (
            "Normalized events can exceed raw events when one raw message expands "
            "into multiple coin-specific rows (e.g. allMids, batched trades)"
        ),
    }

    warning_count = len(state.warnings)
    severity_counts = {}
    for w in state.warnings:
        severity_counts[w.severity] = severity_counts.get(w.severity, 0) + 1

    return {
        "schema_version": QUALITY_REPORT_SCHEMA_VERSION,
        "report_id": generate_report_id(),
        "recorder_session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest_path": str(manifest.manifest_path) if manifest.manifest_path else None,
        "files_inspected": files_inspected,
        "file_sizes": file_sizes,
        "row_counts": row_counts,
        "session_metadata": {
            "start_time_ms": manifest.start_time_ms,
            "end_time_ms": manifest.end_time_ms,
            "network": manifest_data.get("network"),
            "coins": manifest.coins,
            "shutdown_reason": manifest_data.get("shutdown_reason"),
            "reconnect_count": manifest_data.get("reconnect_count"),
            "degraded": manifest_data.get("degraded"),
        },
        "channel_summaries": channel_summaries,
        "coin_summaries": coin_summaries,
        "latency_summaries": latency_summaries,
        "bbo_spread_stats": bbo_spread_stats,
        "gap_summary": gap_summary,
        "raw_normalized_consistency": consistency,
        "data_integrity_warnings": [w.to_dict() for w in state.warnings],
        "warning_count": warning_count,
        "severity_counts": severity_counts,
        "quality_label": overall.value,
        "channel_quality_labels": {
            ch: s.get("quality_label") for ch, s in channel_summaries.items()
        },
        "usability_flags": usability,
        "recommendations": recommendations,
        "paper_only": True,
        "execution_enabled": False,
        "channel_to_event_type": CHANNEL_TO_EVENT_TYPE,
    }


def write_quality_report(
    report: dict[str, Any],
    output_dir: Path,
    write_json: bool = True,
    write_markdown: bool = True,
) -> dict[str, Path]:
    """Write JSON and/or markdown quality reports. Returns paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = report["recorder_session_id"]
    written: dict[str, Path] = {}

    if write_json:
        json_path = output_dir / f"{session_id}.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        written["json"] = json_path

    if write_markdown:
        md_path = output_dir / f"{session_id}.md"
        md_path.write_text(render_markdown_report(report), encoding="utf-8")
        written["markdown"] = md_path

    return written


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render a human-readable markdown quality summary."""
    session_id = report.get("recorder_session_id", "unknown")
    label = report.get("quality_label", "UNKNOWN")
    generated = report.get("generated_at", "")

    verdict_map = {
        QualityLabel.GOOD_FOR_RESEARCH.value: (
            "Session data is reliable for research and offline analysis."
        ),
        QualityLabel.USABLE_WITH_WARNINGS.value: (
            "Session data is usable but review warnings before drawing conclusions."
        ),
        QualityLabel.DEGRADED.value: (
            "Session has significant quality issues; use raw data cautiously."
        ),
        QualityLabel.UNUSABLE.value: (
            "Session is not suitable for research; re-record required."
        ),
    }
    verdict = verdict_map.get(label, "Quality could not be determined.")

    lines = [
        "# Market Data Quality Report",
        "",
        f"**Session ID:** `{session_id}`",
        f"**Overall quality:** `{label}`",
        f"**Generated:** {generated}",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Key metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    consistency = report.get("raw_normalized_consistency") or {}
    gap_summary = report.get("gap_summary") or {}
    meta = report.get("session_metadata") or {}

    lines.extend(
        [
            f"| Raw events (inspected) | {consistency.get('raw_count_inspected', '—')} |",
            f"| Normalized events (inspected) | {consistency.get('normalized_count_inspected', '—')} |",
            f"| Reconnects | {meta.get('reconnect_count', '—')} |",
            f"| Warnings | {report.get('warning_count', 0)} |",
            f"| Total gaps | {gap_summary.get('total_gaps', 0)} |",
            f"| Expansion ratio | {consistency.get('normalized_to_raw_expansion_ratio', '—')} |",
            "",
            "## Channel quality",
            "",
            "| Channel | Raw | Normalized | Label |",
            "|---------|-----|------------|-------|",
        ]
    )

    for ch, summary in sorted((report.get("channel_summaries") or {}).items()):
        lines.append(
            f"| {ch} | {summary.get('raw_event_count', 0)} | "
            f"{summary.get('normalized_event_count', 0)} | "
            f"`{summary.get('quality_label', '—')}` |"
        )

    lines.extend(["", "## Latency by channel", ""])
    lines.append("| Channel | Count | Median | p95 | p99 | Notes |")
    lines.append("|---------|-------|--------|-----|-----|-------|")

    for ch, lat in sorted((report.get("latency_summaries") or {}).items()):
        if not lat.get("latency_available", True):
            lines.append(f"| {ch} | — | — | — | — | {lat.get('note', 'unavailable')} |")
            continue
        note = lat.get("note") or ("informational" if lat.get("informational_only") else "")
        lines.append(
            f"| {ch} | {lat.get('count', 0)} | "
            f"{_fmt_ms(lat.get('median_ms'))} | {_fmt_ms(lat.get('p95_ms'))} | "
            f"{_fmt_ms(lat.get('p99_ms'))} | {note} |"
        )

    warnings = report.get("data_integrity_warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for w in warnings[:50]:
            loc = ""
            if w.get("channel"):
                loc = f" [{w['channel']}"
                if w.get("coin"):
                    loc += f"/{w['coin']}"
                loc += "]"
            lines.append(f"- **{w.get('severity')}**{loc}: {w.get('message')}")
        if len(warnings) > 50:
            lines.append(f"- … and {len(warnings) - 50} more warnings")

    lines.extend(["", "## Usability flags", ""])
    for flag_name, flag_data in sorted((report.get("usability_flags") or {}).items()):
        usable = flag_data.get("usable", False)
        reason = flag_data.get("reason", "")
        lines.append(f"- **{flag_name}:** {'yes' if usable else 'no'} — {reason}")

    recs = report.get("recommendations") or []
    if recs:
        lines.extend(["", "## Recommendations", ""])
        for rec in recs:
            lines.append(f"- {rec}")

    lines.extend(
        [
            "",
            "## Safety statement",
            "",
            "This quality report is **read-only** and **research-only**. "
            "`paper_only: true`, `execution_enabled: false`. "
            "No wallets, signing, or exchange actions. "
            "Quality assessment does not prove strategy profitability.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.0f} ms"
