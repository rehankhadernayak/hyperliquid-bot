"""Standalone research report builder for Hyperliquid bid/ask ML pipeline.

Paper-only and research-only. Reads local JSON/CSV/Markdown artifacts and writes
a consolidated research report under reports/final_research_report/. No execution,
no orders, no wallets, no AWS, no network, no shell commands.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bid_ask_daily_summary import (
    DEFAULT_TRAINING_REPORTS_DIR,
    load_latest_training_report,
)
from bid_ask_dashboard_artifact_refresher import (
    CANDIDATE_B_RELAXED_25_NAME,
    CANDIDATE_B_RELAXED_62_NAME,
    CANDIDATE_B_STRICT_NAME,
    DEFAULT_DATA_ROOT,
    DEFAULT_REPORTS_ROOT,
    MILESTONES,
    RESEARCH_COINS,
    build_dashboard_summary,
)
from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT

DEFAULT_OUTPUT_DIR = Path("reports/final_research_report")
DEFAULT_TITLE = "Hyperliquid Bid/Ask ML Research Report"
DEFAULT_FORMAT = "markdown"

LATEST_MARKDOWN_NAME = "hyperliquid_bid_ask_research_report_latest.md"
LATEST_METADATA_NAME = "hyperliquid_bid_ask_research_report_latest.json"
STAMPED_MARKDOWN_PREFIX = "hyperliquid_bid_ask_research_report_"
STAMPED_METADATA_PREFIX = "hyperliquid_bid_ask_research_report_"

RESEARCH_NOTES_REL = Path("research_notes/hyperliquid_bid_ask_research_notes.md")
DASHBOARD_SUMMARY_REL = Path("dashboard_artifacts/dashboard_summary_latest.json")

PROFITABILITY_DISCLAIMER = (
    "Simulated returns do not prove future profitability. "
    "Positive backtest snippets can be noise, overfitting, or too few trades."
)
MARKET_READINESS_DISCLAIMER = (
    "This system is not market ready. Real execution is disabled."
)
RESEARCH_ONLY_NOTE = "Paper-only research. Does not prove profitability."

REPORT_SECTIONS: tuple[str, ...] = (
    "Abstract",
    "Project Goal",
    "Safety Boundary",
    "System Architecture",
    "Data Pipeline",
    "Data Quality and Repair",
    "Feature Engineering",
    "Modeling Method",
    "Walk-Forward Evaluation",
    "Candidate B Results",
    "Relaxed Candidate Failure Analysis",
    "Milestone Journey",
    "Current Blockers",
    "Next Plan",
    "Limitations",
    "Market-Readiness Checklist",
    "Appendix: Key Commands and Report Paths",
)

KEY_COMMANDS: tuple[str, ...] = (
    "python3 bid_ask_research_report_builder.py build",
    "python3 bid_ask_data_pipeline.py validate --coin BTC --interval 1s",
    "python3 bid_ask_train.py --coin BTC --interval 15s",
    "python3 bid_ask_walk_forward_evaluator.py --coin BTC",
    "python3 bid_ask_candidate_validator.py validate",
    "python3 bid_ask_daily_summary.py --coins BTC ETH",
    "python3 bid_ask_dashboard_artifact_refresher.py refresh",
    "python3 bid_ask_research_scheduler.py phase3-audit",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _validate_output_dir(output_dir: Path) -> Path:
    if output_dir.name != "final_research_report":
        raise ValueError(
            "Output must be written under a directory named final_research_report "
            f"(got {output_dir})"
        )
    return output_dir


def _parse_report_stamp(path: Path) -> str:
    stem = path.stem
    if "_repair_" in stem:
        return stem.rsplit("_repair_", 1)[-1]
    if "_validation_" in stem:
        return stem.rsplit("_validation_", 1)[-1]
    if stem.startswith("robustness_"):
        return stem.rsplit("_", 1)[-1]
    return stem


def _latest_repair_reports(data_root: Path) -> list[dict[str, Any]]:
    reports_dir = data_root / "reports"
    if not reports_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(reports_dir.glob("*_repair_*.json"), key=_parse_report_stamp, reverse=True):
        payload = _load_json(path)
        if payload is None:
            continue
        entry = dict(payload)
        entry["_report_path"] = str(path)
        rows.append(entry)
    return rows[:6]


def _latest_robustness_report(reports_root: Path) -> dict[str, Any] | None:
    candidate_dir = reports_root / "bid_ask_candidate_validation"
    if not candidate_dir.is_dir():
        return None
    for path in sorted(candidate_dir.glob("robustness_*.json"), key=_parse_report_stamp, reverse=True):
        payload = _load_json(path)
        if payload is not None:
            payload = dict(payload)
            payload["_report_path"] = str(path)
            return payload
    return None


def _load_research_notes(reports_root: Path) -> str | None:
    path = reports_root / RESEARCH_NOTES_REL
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _load_dashboard_artifact(reports_root: Path) -> dict[str, Any] | None:
    return _load_json(reports_root / DASHBOARD_SUMMARY_REL)


def _extract_notes_section(notes: str | None, heading: str) -> str | None:
    if not notes:
        return None
    pattern = rf"^##\s+\d+\.\s+{re.escape(heading)}\s*$"
    match = re.search(pattern, notes, flags=re.MULTILINE | re.IGNORECASE)
    if not match:
        return None
    start = match.end()
    next_heading = re.search(r"^##\s+\d+\.", notes[start:], flags=re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(notes)
    return notes[start:end].strip()


def _walk_forward_by_name(reports_root: Path, filename: str) -> dict[str, Any] | None:
    path = reports_root / "bid_ask_walk_forward" / filename
    payload = _load_json(path)
    if payload is None:
        return None
    payload = dict(payload)
    payload["_report_path"] = str(path)
    return payload


def _training_summary(reports_root: Path) -> dict[str, Any]:
    training_dir = reports_root / "bid_ask_training"
    if not training_dir.is_dir():
        training_dir = DEFAULT_TRAINING_REPORTS_DIR
    by_coin: dict[str, Any] = {}
    for coin in RESEARCH_COINS:
        report = load_latest_training_report(coin, training_dir)
        if report is None:
            by_coin[coin] = {"available": False}
            continue
        evaluation = report.get("evaluation") or {}
        by_coin[coin] = {
            "available": True,
            "report_id": report.get("report_id"),
            "interval": report.get("interval"),
            "horizon": report.get("horizon"),
            "model_used": report.get("model_used"),
            "model_accuracy": report.get("model_accuracy"),
            "baseline_accuracy": report.get("baseline_accuracy"),
            "model_edge_vs_baseline": report.get("model_edge_vs_baseline"),
            "train_rows": evaluation.get("train_rows"),
            "test_rows": evaluation.get("test_rows"),
            "generated_at": report.get("generated_at"),
        }
    return {"by_coin": by_coin}


def _collect_blockers(
    dashboard: dict[str, Any],
    candidate: dict[str, Any] | None,
    repair_reports: list[dict[str, Any]],
) -> list[str]:
    blockers: list[str] = []

    phase = dashboard.get("phase_backfill") or {}
    for warning in phase.get("warnings") or []:
        blockers.append(str(warning))

    validation_status = (candidate or {}).get("validation_status") or {}
    if validation_status.get("insufficient_trades"):
        blockers.append(
            "Candidate B has insufficient paper trades for statistical confidence."
        )
    if validation_status.get("negative_after_costs"):
        blockers.append("Latest candidate validation shows negative returns after costs.")
    if validation_status.get("unstable_concentrated"):
        blockers.append("Stability check: returns concentrated in one time bucket.")
    if validation_status.get("unstable_sparse"):
        blockers.append("Stability check: sparse trade buckets flagged.")

    imported_hours = (dashboard.get("data_coverage") or {}).get("imported_hours_mean") or 0
    target_hours = (dashboard.get("data_coverage") or {}).get("target_hours") or 336
    if imported_hours < target_hours * 0.5:
        blockers.append(
            f"Historical backfill incomplete ({imported_hours:.1f}h imported vs "
            f"{target_hours}h target)."
        )

    if not repair_reports:
        blockers.append("No repair reports found — data quality repair history unavailable.")

    if not blockers:
        blockers.append(
            "Primary blocker: find a candidate that increases trade count while staying "
            "positive after realistic costs."
        )
    return blockers


def _default_next_plan() -> list[str]:
    return [
        "Finish BTC/ETH 14-day data sprint and resolve import planning stalls.",
        "Inspect losing trades from relaxed Candidate B scenarios.",
        "Test horizons 10 / 15 / 20 / 30 / 40.",
        "Compare BTC vs ETH on the same feature set.",
        "Build a candidate grid (confidence × edge × class × horizon).",
        "Freeze one promising config only if it survives scaling and stability checks.",
        "Run an out-of-sample paper test on held-out dates (not started).",
    ]


def _market_readiness_checklist(
    dashboard: dict[str, Any],
    candidate: dict[str, Any] | None,
    robustness: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    validation_status = (candidate or {}).get("validation_status") or {}
    latest_wf = dashboard.get("latest_walk_forward") or {}
    items = [
        {
            "item": "Real execution disabled",
            "status": "pass",
            "detail": "execution_enabled=false, real_execution_ready=false",
        },
        {
            "item": "Paper-only mode enforced",
            "status": "pass" if dashboard.get("paper_only") else "fail",
            "detail": f"paper_only={dashboard.get('paper_only')}",
        },
        {
            "item": "Sufficient paper trades (≥20–30)",
            "status": "fail",
            "detail": f"filtered_paper_trades={latest_wf.get('accepted_trades')}",
        },
        {
            "item": "Positive after costs in simulation",
            "status": (
                "fail"
                if validation_status.get("negative_after_costs")
                or latest_wf.get("positive_after_costs") is False
                else "unknown"
            ),
            "detail": PROFITABILITY_DISCLAIMER,
        },
        {
            "item": "Stability checks pass",
            "status": (
                "fail"
                if validation_status.get("unstable_concentrated")
                or validation_status.get("unstable_sparse")
                else "unknown"
            ),
            "detail": "Concentration and sparse-bucket warnings under review.",
        },
        {
            "item": "Robustness evaluation pass",
            "status": "fail" if robustness and not robustness.get("robustness_pass") else "unknown",
            "detail": (
                str(robustness.get("robustness_summary"))
                if robustness
                else "No robustness report available."
            ),
        },
        {
            "item": "Out-of-sample paper test complete",
            "status": "fail",
            "detail": "Not started.",
        },
        {
            "item": "Market ready",
            "status": "fail",
            "detail": MARKET_READINESS_DISCLAIMER,
        },
    ]
    return items


def build_research_report_context(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    title: str = DEFAULT_TITLE,
) -> dict[str, Any]:
    """Gather all local artifacts needed for the research report."""
    dashboard = build_dashboard_summary(reports_root=reports_root, data_root=data_root)
    dashboard_artifact = _load_dashboard_artifact(reports_root)
    research_notes = _load_research_notes(reports_root)
    repair_reports = _latest_repair_reports(data_root)
    robustness = _latest_robustness_report(reports_root)
    training = _training_summary(reports_root)

    candidate_validation = dashboard.get("latest_candidate_validation") or {}
    strict_wf = _walk_forward_by_name(reports_root, CANDIDATE_B_STRICT_NAME)
    relaxed_25_wf = _walk_forward_by_name(reports_root, CANDIDATE_B_RELAXED_25_NAME)
    relaxed_62_wf = _walk_forward_by_name(reports_root, CANDIDATE_B_RELAXED_62_NAME)

    blockers = _collect_blockers(
        dashboard,
        candidate_validation if candidate_validation.get("candidate_name") else None,
        repair_reports,
    )

    next_plan_text = _extract_notes_section(research_notes, "Next Plan")
    next_plan = (
        [line.strip("- ").strip() for line in next_plan_text.splitlines() if line.strip().startswith("-")]
        if next_plan_text
        else _default_next_plan()
    )

    return {
        "generated_at": _utc_now_iso(),
        "title": title,
        "format": DEFAULT_FORMAT,
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "real_execution_ready": False,
        "market_ready": False,
        "safety_statement": SAFETY_STATEMENT,
        "profitability_disclaimer": PROFITABILITY_DISCLAIMER,
        "market_readiness_disclaimer": MARKET_READINESS_DISCLAIMER,
        "research_only_note": RESEARCH_ONLY_NOTE,
        "sections": list(REPORT_SECTIONS),
        "dashboard": dashboard,
        "dashboard_artifact_available": dashboard_artifact is not None,
        "research_notes_available": research_notes is not None,
        "research_notes_excerpt": (research_notes[:2000] + "...") if research_notes and len(research_notes) > 2000 else research_notes,
        "repair_reports": repair_reports,
        "robustness": robustness,
        "training": training,
        "candidate_b": dashboard.get("candidate_b") or {},
        "candidate_validation": candidate_validation,
        "walk_forward": {
            "latest": dashboard.get("latest_walk_forward") or {},
            "strict": strict_wf,
            "relaxed_25": relaxed_25_wf,
            "relaxed_62": relaxed_62_wf,
        },
        "milestones": dashboard.get("milestones") or [],
        "blockers": blockers,
        "next_plan": next_plan,
        "market_readiness_checklist": _market_readiness_checklist(
            dashboard,
            candidate_validation if candidate_validation.get("candidate_name") else None,
            robustness,
        ),
        "key_commands": list(KEY_COMMANDS),
        "source_paths": {
            **(dashboard.get("source_paths") or {}),
            "research_notes": str(reports_root / RESEARCH_NOTES_REL),
            "dashboard_artifact": str(reports_root / DASHBOARD_SUMMARY_REL),
            "training_dir": str(reports_root / "bid_ask_training"),
            "repair_reports_dir": str(data_root / "reports"),
            "output_dir": str(DEFAULT_OUTPUT_DIR),
        },
    }


def _section_heading(title: str) -> str:
    return f"## {title}"


def _fmt_metric(value: Any, *, suffix: str = "", none_text: str = "n/a") -> str:
    if value is None:
        return none_text
    if isinstance(value, float):
        return f"{value:+.2f}{suffix}" if suffix else f"{value:.4f}"
    return f"{value}{suffix}"


def render_research_report_markdown(context: dict[str, Any]) -> str:
    """Render the full research report as Markdown."""
    dashboard = context.get("dashboard") or {}
    phase = dashboard.get("phase_backfill") or {}
    validation = dashboard.get("validation") or {}
    latest_wf = context.get("walk_forward", {}).get("latest") or {}
    strict_wf = context.get("walk_forward", {}).get("strict")
    relaxed_25 = context.get("walk_forward", {}).get("relaxed_25")
    relaxed_62 = context.get("walk_forward", {}).get("relaxed_62")
    candidate_validation = context.get("candidate_validation") or {}
    candidate_scenarios = (context.get("candidate_b") or {}).get("scenarios") or []
    training = context.get("training") or {}

    lines: list[str] = [
        f"# {context.get('title', DEFAULT_TITLE)}",
        "",
        f"- generated_at: {context.get('generated_at')}",
        f"- paper_only: {context.get('paper_only')}",
        f"- execution_enabled: {context.get('execution_enabled')}",
        f"- real_execution_ready: {context.get('real_execution_ready')}",
        f"- market_ready: {context.get('market_ready')}",
        "",
        _section_heading("Abstract"),
        "",
        "This report consolidates local bid/ask ML research artifacts into a single "
        "paper-only summary. It describes data pipeline status, model evaluation, "
        "Candidate B walk-forward results, and known blockers.",
        "",
        PROFITABILITY_DISCLAIMER,
        "",
        MARKET_READINESS_DISCLAIMER,
        "",
        _section_heading("Project Goal"),
        "",
        "Build a paper-only research pipeline on Hyperliquid BTC/ETH bid/ask "
        "microstructure data: extract and backfill historical L2, repair and validate "
        "CSVs, resample to 15s/1m, train classifiers, walk-forward test, filter "
        "candidates with realistic costs, and log paper decisions without real orders.",
        "",
        "Presentation goal: demonstrate a serious research system, not a fake profitable bot.",
        "",
        _section_heading("Safety Boundary"),
        "",
        "| Flag | Value |",
        "|------|-------|",
        f"| paper_only | {context.get('paper_only')} |",
        f"| execution_enabled | {context.get('execution_enabled')} |",
        f"| real_execution_ready | {context.get('real_execution_ready')} |",
        f"| market_ready | {context.get('market_ready')} |",
        "",
        str(context.get("safety_statement", "")),
        "",
        PROFITABILITY_DISCLAIMER,
        "",
        _section_heading("System Architecture"),
        "",
        "```",
        "Hyperliquid historical L2 archives",
        "        │",
        "        ▼",
        "Bid/Ask importer / extractor",
        "        │",
        "        ▼",
        "     1s CSV (historical_1s/)",
        "        │",
        "        ▼",
        "  15s / 1m resampler",
        "        │",
        "        ▼",
        "    Validator (+ repair pass)",
        "        │",
        "        ▼",
        "  Feature builder (~44 microstructure features)",
        "        │",
        "        ▼",
        "  Model training (RandomForestClassifier)",
        "        │",
        "        ▼",
        "Walk-forward evaluator",
        "        │",
        "        ▼",
        "Candidate validator (costs, stability, min trades)",
        "        │",
        "        ▼",
        "Paper decision dashboard (read-only)",
        "```",
        "",
        "Key scripts: `bid_ask_research_scheduler.py`, `bid_ask_data_pipeline.py`, "
        "`bid_ask_walk_forward_evaluator.py`, `bid_ask_candidate_validator.py`, "
        "`bid_ask_research_dashboard.py`.",
        "",
        _section_heading("Data Pipeline"),
        "",
    ]

    if phase.get("available"):
        lines.extend(
            [
                f"- phase_backfill: {phase.get('phase_backfill_name') or 'phase3'}",
                f"- imported_hours_mean: {phase.get('imported_hours_mean')}",
                f"- target_hours: {phase.get('target_hours')}",
                f"- progress_fraction: {phase.get('progress_fraction')}",
                f"- downloads_total_files: {phase.get('downloads_total_files')}",
                f"- scheduler_latest_action: {phase.get('scheduler_latest_action')}",
            ]
        )
    else:
        lines.append("- Phase 3 audit report not available.")

    lines.extend(["", "### Coverage by coin", ""])
    for coin_row in (dashboard.get("data_coverage") or {}).get("coins") or []:
        lines.append(
            f"- {coin_row.get('coin')}: 1s rows={coin_row.get('rows_1s')} "
            f"15s rows={coin_row.get('rows_15s')} "
            f"coverage_hours={coin_row.get('coverage_hours')} "
            f"latest={coin_row.get('last_timestamp') or 'n/a'}"
        )

    lines.extend(["", _section_heading("Data Quality and Repair"), ""])
    by_coin = validation.get("by_coin_interval") or {}
    for coin, intervals in by_coin.items():
        lines.append(f"### {coin}")
        for interval, details in intervals.items():
            if not details.get("available"):
                lines.append(f"- {interval}: validation report not available")
                continue
            lines.append(
                f"- {interval}: passed={details.get('passed')} rows={details.get('row_count')} "
                f"missing_periods={details.get('missing_periods_count')} "
                f"duplicates={details.get('duplicate_timestamp_count')}"
            )
        lines.append("")

    repair_reports = context.get("repair_reports") or []
    if repair_reports:
        lines.append("### Recent repair reports")
        for report in repair_reports:
            lines.append(
                f"- {report.get('coin')} {report.get('interval')}: "
                f"mock_rows_removed={report.get('mock_rows_removed')} "
                f"rows_before={report.get('rows_before')} rows_after={report.get('rows_after')} "
                f"monotonic_after={report.get('monotonic_after')}"
            )
    else:
        lines.append("- No repair reports found locally.")

    lines.extend(["", _section_heading("Feature Engineering"), ""])
    lines.extend(
        [
            "- Microstructure feature set (~44 features) built from 15s resampled bars.",
            "- Features include spread, BBO imbalance, rolling momentum, and volatility proxies.",
            "- Feature builder runs offline against local CSVs only.",
            "- No live market data feeds are required for report generation.",
        ]
    )

    lines.extend(["", _section_heading("Modeling Method"), ""])
    for coin, details in (training.get("by_coin") or {}).items():
        if not details.get("available"):
            lines.append(f"- {coin}: no training report available")
            continue
        lines.append(
            f"- {coin}: model={details.get('model_used')} interval={details.get('interval')} "
            f"horizon={details.get('horizon')} accuracy={_fmt_metric(details.get('model_accuracy'))} "
            f"baseline={_fmt_metric(details.get('baseline_accuracy'))} "
            f"edge={_fmt_metric(details.get('model_edge_vs_baseline'))} "
            f"train_rows={details.get('train_rows')} test_rows={details.get('test_rows')}"
        )
    lines.append("")
    lines.append(
        "Classifier: RandomForestClassifier on microstructure features. "
        "Evaluation compares model accuracy against a majority-class baseline."
    )

    lines.extend(["", _section_heading("Walk-Forward Evaluation"), ""])
    if latest_wf.get("report_path"):
        lines.extend(
            [
                f"- latest report: {latest_wf.get('report_path')}",
                f"- coin: {latest_wf.get('coin')}",
                f"- interval: {latest_wf.get('interval')}",
                f"- horizon: {latest_wf.get('horizon')}",
                f"- windows: {latest_wf.get('windows')}",
                f"- model_accuracy: {_fmt_metric(latest_wf.get('model_accuracy'))}",
                f"- baseline_accuracy: {_fmt_metric(latest_wf.get('baseline_accuracy'))}",
                f"- accuracy_edge: {_fmt_metric(latest_wf.get('accuracy_edge'))}",
                f"- filtered_paper_trades: {latest_wf.get('accepted_trades')}",
                f"- cumulative_net_bps (simulated): {_fmt_metric(latest_wf.get('cumulative_net_bps'), suffix=' bps')}",
                f"- positive_after_costs: {latest_wf.get('positive_after_costs')}",
            ]
        )
    else:
        lines.append("- No walk-forward report found.")

    lines.extend(["", _section_heading("Candidate B Results"), ""])
    lines.append(
        "Candidate B (`BTC_DOWN_MICRO_H20_CANDIDATE_B`): strict filters confidence ≥ 0.65, "
        "edge ≥ 2 bps, DOWN class only. Results below are simulated paper trades only."
    )
    lines.append("")
    if candidate_validation.get("candidate_name"):
        lines.extend(
            [
                f"- candidate_name: {candidate_validation.get('candidate_name')}",
                f"- filtered_paper_trades: {candidate_validation.get('filtered_paper_trades')}",
                f"- cumulative_net_bps (simulated): "
                f"{_fmt_metric(candidate_validation.get('cumulative_net_return_bps'), suffix=' bps')}",
                f"- validation_status: {json.dumps(candidate_validation.get('validation_status') or {})}",
                f"- ready_for_more_research: {candidate_validation.get('ready_for_more_research')}",
            ]
        )
    else:
        lines.append("- Latest candidate validation report not available.")

    lines.append("")
    lines.append("### Scenario comparison")
    for row in candidate_scenarios:
        lines.append(
            f"- {row.get('config')}: trades={row.get('trades')} "
            f"net_bps={row.get('cumulative_net_bps')} source={row.get('source')}"
        )

    if strict_wf:
        lines.append("")
        lines.append(
            f"Strict walk-forward detail: trades={strict_wf.get('filtered_paper_trades')} "
            f"net_bps={_fmt_metric(strict_wf.get('cumulative_net_return_bps'), suffix=' bps')} "
            f"win_rate={_fmt_metric(strict_wf.get('win_rate_after_costs'))}"
        )

    lines.extend(["", _section_heading("Relaxed Candidate Failure Analysis"), ""])
    lines.append(
        "Relaxing Candidate B filters increases trade count but simulated post-cost "
        "returns turn sharply negative. This is not evidence of a working strategy "
        "at higher frequency."
    )
    lines.append("")
    for label, payload in (
        ("Relaxed 25-trade (DOWN only)", relaxed_25),
        ("Relaxed 62-trade (DOWN + UP)", relaxed_62),
    ):
        if payload:
            lines.append(
                f"- {label}: trades={payload.get('filtered_paper_trades')} "
                f"cumulative_net_bps={_fmt_metric(payload.get('cumulative_net_return_bps'), suffix=' bps')} "
                f"win_rate={_fmt_metric(payload.get('win_rate_after_costs'))} "
                f"positive_after_costs={payload.get('positive_after_costs')}"
            )
        else:
            fallback = next(
                (row for row in candidate_scenarios if label.split()[1].startswith(row.get("config", "").split()[1][:3])),
                None,
            )
            if fallback:
                lines.append(
                    f"- {label}: trades={fallback.get('trades')} "
                    f"cumulative_net_bps={fallback.get('cumulative_net_bps')} (fallback summary)"
                )
            else:
                lines.append(f"- {label}: report not available")

    lines.extend(["", _section_heading("Milestone Journey"), ""])
    milestones = context.get("milestones") or []
    if milestones:
        lines.append("| Milestone | Progress | Status |")
        lines.append("|-----------|----------|--------|")
        for milestone in milestones:
            lines.append(
                f"| {milestone.get('name')} | {milestone.get('progress_pct')}% | "
                f"{milestone.get('status')} |"
            )
    else:
        lines.append("| Milestone | Progress | Status |")
        lines.append("|-----------|----------|--------|")
        for name, pct, status in MILESTONES:
            lines.append(f"| {name} | {pct}% | {status} |")

    lines.extend(["", _section_heading("Current Blockers"), ""])
    for blocker in context.get("blockers") or []:
        lines.append(f"- {blocker}")

    lines.extend(["", _section_heading("Next Plan"), ""])
    for item in context.get("next_plan") or []:
        lines.append(f"- {item}")

    lines.extend(["", _section_heading("Limitations"), ""])
    limitations = [
        RESEARCH_ONLY_NOTE,
        PROFITABILITY_DISCLAIMER,
        "Small sample sizes (e.g., 5 strict Candidate B trades) are not statistically meaningful.",
        "Historical coverage may be incomplete; validation can pass with large gap warnings.",
        "Cost model (fee + slippage + multiplier) is a simulation assumption, not live fill data.",
        "Model accuracy edge over baseline does not translate directly to tradable edge after costs.",
        MARKET_READINESS_DISCLAIMER,
    ]
    for limitation in limitations:
        lines.append(f"- {limitation}")

    lines.extend(["", _section_heading("Market-Readiness Checklist"), ""])
    lines.append(MARKET_READINESS_DISCLAIMER)
    lines.append("")
    lines.append("| Check | Status | Detail |")
    lines.append("|-------|--------|--------|")
    for row in context.get("market_readiness_checklist") or []:
        lines.append(f"| {row.get('item')} | {row.get('status')} | {row.get('detail')} |")

    lines.extend(["", _section_heading("Appendix: Key Commands and Report Paths"), ""])
    lines.append("### Key commands")
    for command in context.get("key_commands") or []:
        lines.append(f"- `{command}`")
    lines.append("")
    lines.append("### Source report paths")
    for key, value in (context.get("source_paths") or {}).items():
        lines.append(f"- {key}: `{value}`")

    lines.append("")
    lines.append(f"*Reminder: {RESEARCH_ONLY_NOTE} {PROFITABILITY_DISCLAIMER}*")

    return "\n".join(lines) + "\n"


def build_metadata_summary(context: dict[str, Any], *, stamp: str, output_paths: dict[str, str]) -> dict[str, Any]:
    dashboard = context.get("dashboard") or {}
    latest_wf = context.get("walk_forward", {}).get("latest") or {}
    candidate_validation = context.get("candidate_validation") or {}
    return {
        "generated_at": context.get("generated_at"),
        "stamp": stamp,
        "title": context.get("title"),
        "format": context.get("format"),
        "paper_only": context.get("paper_only"),
        "execution_enabled": context.get("execution_enabled"),
        "real_execution_ready": context.get("real_execution_ready"),
        "market_ready": context.get("market_ready"),
        "safety_statement": context.get("safety_statement"),
        "profitability_disclaimer": context.get("profitability_disclaimer"),
        "market_readiness_disclaimer": context.get("market_readiness_disclaimer"),
        "sections": context.get("sections"),
        "section_count": len(context.get("sections") or []),
        "research_notes_available": context.get("research_notes_available"),
        "dashboard_artifact_available": context.get("dashboard_artifact_available"),
        "imported_hours_mean": (dashboard.get("data_coverage") or {}).get("imported_hours_mean"),
        "latest_walk_forward_trades": latest_wf.get("accepted_trades"),
        "latest_walk_forward_net_bps": latest_wf.get("cumulative_net_bps"),
        "candidate_name": candidate_validation.get("candidate_name"),
        "candidate_trades": candidate_validation.get("filtered_paper_trades"),
        "candidate_net_bps": candidate_validation.get("cumulative_net_return_bps"),
        "candidate_scenarios": (context.get("candidate_b") or {}).get("scenarios"),
        "blockers": context.get("blockers"),
        "next_plan": context.get("next_plan"),
        "milestones": context.get("milestones"),
        "market_readiness_checklist": context.get("market_readiness_checklist"),
        "source_paths": context.get("source_paths"),
        "output_paths": output_paths,
    }


def write_research_report(
    context: dict[str, Any],
    output_dir: Path,
    *,
    stamp: str | None = None,
) -> dict[str, Any]:
    """Write latest and stamped Markdown plus JSON metadata summary."""
    output_dir = _validate_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or _utc_stamp()

    latest_md_path = output_dir / LATEST_MARKDOWN_NAME
    stamped_md_path = output_dir / f"{STAMPED_MARKDOWN_PREFIX}{stamp}.md"
    latest_json_path = output_dir / LATEST_METADATA_NAME
    stamped_json_path = output_dir / f"{STAMPED_METADATA_PREFIX}{stamp}.json"

    markdown = render_research_report_markdown(context)
    latest_md_path.write_text(markdown, encoding="utf-8")
    stamped_md_path.write_text(markdown, encoding="utf-8")

    output_paths = {
        "markdown_latest": str(latest_md_path),
        "markdown_stamped": str(stamped_md_path),
        "metadata_latest": str(latest_json_path),
        "metadata_stamped": str(stamped_json_path),
    }
    metadata = build_metadata_summary(context, stamp=stamp, output_paths=output_paths)
    metadata_json = json.dumps(metadata, indent=2, sort_keys=False) + "\n"
    latest_json_path.write_text(metadata_json, encoding="utf-8")
    stamped_json_path.write_text(metadata_json, encoding="utf-8")

    return {
        **metadata,
        "markdown_report_path": str(latest_md_path),
        "stamped_markdown_report_path": str(stamped_md_path),
    }


def build_research_report(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    title: str = DEFAULT_TITLE,
    report_format: str = DEFAULT_FORMAT,
) -> dict[str, Any]:
    if report_format != "markdown":
        raise ValueError(f"Unsupported format: {report_format!r}")

    context = build_research_report_context(
        reports_root=reports_root,
        data_root=data_root,
        title=title,
    )
    return write_research_report(context, output_dir)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a consolidated bid/ask ML research report from local artifacts "
            "(paper-only, research-only)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build research report Markdown and metadata JSON")
    build.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    build.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build.add_argument("--title", default=DEFAULT_TITLE)
    build.add_argument("--format", dest="report_format", default=DEFAULT_FORMAT, choices=["markdown"])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        result = build_research_report(
            reports_root=args.reports_root,
            data_root=args.data_root,
            output_dir=args.output_dir,
            title=args.title,
            report_format=args.report_format,
        )
        print(f"markdown_report_path={result['markdown_report_path']}")
        print(f"metadata_report_path={result['output_paths']['metadata_latest']}")
        print(f"sections={result.get('section_count')}")
        print(f"paper_only={result.get('paper_only')}")
        print(f"real_execution_ready={result.get('real_execution_ready')}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
