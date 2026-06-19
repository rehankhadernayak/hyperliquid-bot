"""JSON and markdown report generation for feature research."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from feature_research.feature_config import FEATURE_RESEARCH_SCHEMA_VERSION
from feature_research.feature_safety import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT


def new_feature_research_id() -> str:
    return uuid.uuid4().hex


def list_feature_research_reports(output_dir: Path) -> list[dict[str, Any]]:
    if not output_dir.exists():
        return []
    reports: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.json"), reverse=True):
        if path.parent.name == "datasets":
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            data.setdefault("feature_research_id", path.stem)
            data.setdefault("json_path", str(path))
            md_path = path.with_suffix(".md")
            if md_path.exists():
                data.setdefault("markdown_path", str(md_path))
            reports.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    reports.sort(key=lambda r: r.get("generated_at", ""), reverse=True)
    return reports


def build_feature_research_report(
    *,
    feature_research_id: str,
    replay_session_id: str,
    source_recorder_session_id: str,
    quality_report_path: str | None,
    quality_label: str | None,
    feature_config: dict[str, Any],
    horizons: list[int],
    coins: list[str],
    channels: list[str],
    row_count: int,
    feature_count: int,
    label_count: int,
    missing_data_summary: dict[str, Any],
    leakage_check: dict[str, Any],
    feature_analysis: list[dict[str, Any]],
    top_possible_signals: list[dict[str, Any]],
    rejected_signals: list[dict[str, Any]],
    warnings: list[str],
    limitations: list[str],
    recommendations: list[str],
    deterministic_dataset_hash: str,
    stream_hash: str,
    dataset_path: str | None,
    dataset_format: str | None,
    dry_run: bool,
    generated_at: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": FEATURE_RESEARCH_SCHEMA_VERSION,
        "feature_research_id": feature_research_id,
        "replay_session_id": replay_session_id,
        "source_recorder_session_id": source_recorder_session_id,
        "quality_report_path": quality_report_path,
        "quality_label": quality_label,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "feature_config": feature_config,
        "horizons": horizons,
        "coins": coins,
        "channels": channels,
        "row_count": row_count,
        "feature_count": feature_count,
        "label_count": label_count,
        "missing_data_summary": missing_data_summary,
        "leakage_check": leakage_check,
        "feature_analysis": feature_analysis,
        "top_possible_signals": top_possible_signals,
        "rejected_signals": rejected_signals,
        "warnings": warnings,
        "limitations": limitations,
        "recommendations": recommendations,
        "deterministic_dataset_hash": deterministic_dataset_hash,
        "stream_hash": stream_hash,
        "dataset_path": dataset_path,
        "dataset_format": dataset_format,
        "dry_run": dry_run,
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "safety_statement": SAFETY_STATEMENT,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Feature Research Report",
        "",
        f"**Feature Research ID:** `{report['feature_research_id']}`",
        f"**Replay Session:** `{report.get('replay_session_id', 'n/a')}`",
        f"**Source Recorder Session:** `{report.get('source_recorder_session_id', 'n/a')}`",
        "",
        "## Data Quality",
        "",
        f"- Quality label: **{report.get('quality_label') or 'unknown'}**",
        f"- Quality report: `{report.get('quality_report_path') or 'n/a'}`",
        "",
        "## Safety Statement",
        "",
        report.get("safety_statement", SAFETY_STATEMENT),
        "",
        f"- Paper only: `{report.get('paper_only')}`",
        f"- Execution enabled: `{report.get('execution_enabled')}`",
        "",
        "## Summary",
        "",
        f"- Row count: **{report.get('row_count', 0)}**",
        f"- Feature count: **{report.get('feature_count', 0)}**",
        f"- Label count: **{report.get('label_count', 0)}**",
        f"- Horizons (seconds): {report.get('horizons', [])}",
        f"- Coins: {', '.join(report.get('coins', []))}",
        f"- Channels: {', '.join(report.get('channels', []))}",
        f"- Deterministic hash: `{report.get('deterministic_dataset_hash', '')}`",
        "",
    ]

    leakage = report.get("leakage_check") or {}
    lines.extend(
        [
            "## Leakage Check",
            "",
            f"- Passed: **{'yes' if leakage.get('passed') else 'no'}**",
        ]
    )
    for check in leakage.get("checks", []):
        lines.append(f"- {check.get('name')}: {check.get('status')}")
    for warning in leakage.get("warnings", []):
        lines.append(f"- Warning: {warning}")
    lines.append("")

    missing = report.get("missing_data_summary") or {}
    lines.extend(["## Missing Data", ""])
    lines.append(f"- Total rows: {missing.get('total_rows', 0)}")
    for name, info in (missing.get("features") or {}).items():
        rate = info.get("missing_rate", 0)
        if rate > 0.05:
            lines.append(f"- `{name}`: {info.get('missing_count')} missing ({rate:.1%})")
    lines.append("")

    possible = report.get("top_possible_signals") or []
    lines.extend(["## Top Possible Signals", ""])
    if possible:
        for sig in possible[:10]:
            lines.append(
                f"- **{sig.get('feature')}** @ {sig.get('horizon_seconds')}s — "
                f"corr={sig.get('correlation')}, IC={sig.get('information_coefficient')} — "
                f"{sig.get('reason')}"
            )
    else:
        lines.append("No features reached POSSIBLE_SIGNAL_NEEDS_MORE_DATA threshold.")
    lines.append("")

    rejected = report.get("rejected_signals") or []
    lines.extend(["## Weak / Rejected Signals", ""])
    if rejected:
        for sig in rejected[:10]:
            lines.append(
                f"- {sig.get('feature')} @ {sig.get('horizon_seconds')}s — "
                f"{sig.get('candidate_signal')}: {sig.get('reason')}"
            )
    else:
        lines.append("None flagged as rejected or unstable.")
    lines.append("")

    lines.extend(["## Limitations", ""])
    for lim in report.get("limitations") or []:
        lines.append(f"- {lim}")
    lines.append("")

    lines.extend(["## Recommendations", ""])
    for rec in report.get("recommendations") or []:
        lines.append(f"- {rec}")
    lines.append("")

    lines.extend(
        [
            "## Next Strategy Ideas",
            "",
            "Feature research does not prove profitability. Use findings to prioritize "
            "hypotheses for controlled event backtests with explicit risk limits.",
            "",
        ]
    )

    if report.get("warnings"):
        lines.extend(["## Warnings", ""])
        for w in report["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)


def write_feature_research_report(
    report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    research_id = report["feature_research_id"]
    json_path = output_dir / f"{research_id}.json"
    md_path = output_dir / f"{research_id}.md"

    report = dict(report)
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(md_path)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
