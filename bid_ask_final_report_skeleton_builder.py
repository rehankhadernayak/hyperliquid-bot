"""Final research report skeleton builder for bid/ask ML research.

Paper-only and research-only. Reads local JSON/CSV/Markdown artifacts and writes
a gradually fillable report skeleton under reports/final_research_report/. No
execution, orders, wallets, AWS, network, or shell commands.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bid_ask_research_status_builder import (
    CANDIDATE_TIMELINE,
    DEFAULT_DATA_ROOT,
    DEFAULT_REPORTS_ROOT,
    RESEARCH_COINS,
    build_research_status,
)
from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT

DEFAULT_OUTPUT_DIR = Path("reports/final_research_report")

DEFAULT_TITLE = "Hyperliquid Bid/Ask ML Research — Final Report (Skeleton)"
DEFAULT_LATEST_JSON_NAME = "final_report_skeleton_latest.json"
DEFAULT_LATEST_MD_NAME = "final_report_skeleton_latest.md"
STAMPED_JSON_PREFIX = "final_report_skeleton_"
STAMPED_MD_PREFIX = "final_report_skeleton_"

RESEARCH_STATUS_REL = Path("research_status/research_status_latest.json")
CANDIDATE_LEDGER_REL = Path("candidate_ledger/candidate_ledger.json")
DASHBOARD_SUMMARY_REL = Path("dashboard_artifacts/dashboard_summary_latest.json")
DEMO_PACKET_REL = Path("demo_mode/demo_packet_latest.json")
PHASE3_AUDIT_REL = Path("bid_ask_research_scheduler/phase3_audit_latest.json")

RESEARCH_ONLY_NOTE = (
    "Research-only skeleton. Metrics do not prove profitability. "
    "Real trading remains disabled."
)
PROFITABILITY_DISCLAIMER = (
    "Simulated returns do not prove future profitability. "
    "Positive backtest snippets can be noise, overfitting, or too few trades."
)

REPORT_SECTIONS: tuple[str, ...] = (
    "Title and Safety Disclaimer",
    "Abstract",
    "Research Question",
    "System Overview",
    "Data Pipeline",
    "Historical Backfill and Validation",
    "Feature Engineering",
    "Labeling and Evaluation",
    "Candidate Timeline",
    "Current Results Summary",
    "Failure Analysis",
    "Robustness Checks",
    "Dashboard and Demo Artifacts",
    "Limitations",
    "Next Steps",
    "Appendix: Commands and Artifact Paths",
)

KEY_COMMANDS: tuple[str, ...] = (
    "python3 bid_ask_final_report_skeleton_builder.py build",
    "python3 bid_ask_research_status_builder.py build",
    "python3 bid_ask_candidate_ledger.py update",
    "python3 bid_ask_accepted_trades_analyzer.py analyze --coin BTC",
    "python3 bid_ask_feature_audit.py audit --coin BTC --interval 15s",
    "python3 bid_ask_dashboard_artifact_refresher.py refresh",
    "python3 bid_ask_demo_mode.py prepare",
    "python3 bid_ask_research_report_builder.py build",
    "python3 bid_ask_data_pipeline.py validate --coin BTC --interval 1s",
    "python3 bid_ask_walk_forward_evaluator.py --coin BTC",
)

DRAFT_MARKER = "Draft — To be updated"
PARTIAL_MARKER = "Partial — data available, narrative incomplete"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _validate_output_dir(output_dir: Path) -> Path:
    if output_dir.name != "final_research_report":
        raise ValueError(
            "Output directory must be named final_research_report "
            f"(got {output_dir})"
        )
    return output_dir


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _section_status(*, available: bool, has_detail: bool = False) -> str:
    if available and has_detail:
        return "partial"
    if available:
        return "draft"
    return "draft"


def _status_line(status: str) -> str:
    if status == "partial":
        return f"**Status: {PARTIAL_MARKER}**"
    return f"**Status: {DRAFT_MARKER}**"


def _load_research_status(reports_root: Path) -> dict[str, Any]:
    path = reports_root / RESEARCH_STATUS_REL
    payload = _load_json(path)
    if payload is not None:
        return {
            "available": True,
            "path": str(path),
            "payload": payload,
        }
    return {
        "available": False,
        "path": str(path),
        "payload": None,
    }


def _load_candidate_ledger(reports_root: Path) -> dict[str, Any]:
    path = reports_root / CANDIDATE_LEDGER_REL
    payload = _load_json(path)
    if payload is None:
        return {
            "available": False,
            "path": str(path),
            "row_count": 0,
        }
    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        rows = []
    return {
        "available": True,
        "path": str(path),
        "row_count": payload.get("row_count", len(rows)),
        "generated_at": payload.get("generated_at"),
        "summary": payload.get("summary"),
    }


def _load_dashboard_summary(reports_root: Path) -> dict[str, Any]:
    path = reports_root / DASHBOARD_SUMMARY_REL
    payload = _load_json(path)
    if payload is None:
        return {"available": False, "path": str(path)}
    return {
        "available": True,
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "phase_backfill": payload.get("phase_backfill"),
        "milestones": payload.get("milestones"),
        "next_recommended_action": payload.get("next_recommended_action"),
    }


def _load_demo_packet(reports_root: Path) -> dict[str, Any]:
    path = reports_root / DEMO_PACKET_REL
    payload = _load_json(path)
    if payload is None:
        return {"available": False, "path": str(path)}
    return {
        "available": True,
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "warnings": list(payload.get("warnings") or []),
        "what_not_to_claim": list(payload.get("what_not_to_claim") or []),
    }


def _latest_feature_audits(reports_root: Path) -> list[dict[str, Any]]:
    audit_dir = reports_root / "feature_audit"
    if not audit_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(audit_dir.glob("feature_audit_*.json")):
        payload = _load_json(path)
        if payload is None:
            continue
        rows.append(
            {
                "path": str(path),
                "coin": payload.get("coin"),
                "interval": payload.get("interval"),
                "row_count": payload.get("row_count"),
                "feature_families": list((payload.get("feature_families") or {}).keys()),
                "generated_at": payload.get("generated_at"),
            }
        )
    return rows[-6:]


def _resolve_research_context(
    *,
    reports_root: Path,
    data_root: Path,
    research_status_wrapper: dict[str, Any],
) -> dict[str, Any]:
    if research_status_wrapper.get("available"):
        payload = research_status_wrapper["payload"]
        assert isinstance(payload, dict)
        return {
            "source": "research_status_latest",
            "data_status": payload.get("data_status") or {},
            "ledger_summary": payload.get("ledger_summary") or {},
            "accepted_trades_analysis": payload.get("accepted_trades_analysis") or {},
            "candidate_timeline": payload.get("candidate_timeline") or list(CANDIDATE_TIMELINE),
            "blockers": list(payload.get("blockers") or []),
            "next_actions": list(payload.get("next_actions") or []),
        }

    status = build_research_status(reports_root=reports_root, data_root=data_root)
    return {
        "source": "rebuilt_from_artifacts",
        "data_status": status.get("data_status") or {},
        "ledger_summary": status.get("ledger_summary") or {},
        "accepted_trades_analysis": status.get("accepted_trades_analysis") or {},
        "candidate_timeline": status.get("candidate_timeline") or list(CANDIDATE_TIMELINE),
        "blockers": list(status.get("blockers") or []),
        "next_actions": list(status.get("next_actions") or []),
    }


def _data_row_counts(data_status: dict[str, Any]) -> dict[str, dict[str, int | None]]:
    counts: dict[str, dict[str, int | None]] = {}
    coins = data_status.get("coins") or {}
    for coin in RESEARCH_COINS:
        coin_data = coins.get(coin) or {}
        counts[coin] = {}
        for interval in ("1s", "15s", "1m"):
            row = coin_data.get(interval) or {}
            counts[coin][interval] = row.get("row_count")
    return counts


def _build_section_metadata(
    *,
    research_status_wrapper: dict[str, Any],
    ledger: dict[str, Any],
    dashboard: dict[str, Any],
    demo_packet: dict[str, Any],
    feature_audits: list[dict[str, Any]],
    context: dict[str, Any],
    phase3: dict[str, Any] | None,
) -> dict[str, str]:
    data_status = context.get("data_status") or {}
    has_data_rows = any(
        (coin_data.get(interval) or {}).get("row_count")
        for coin_data in (data_status.get("coins") or {}).values()
        for interval in ("1s", "15s", "1m")
    )
    ledger_summary = context.get("ledger_summary") or {}
    accepted = context.get("accepted_trades_analysis") or {}

    return {
        "title_and_safety_disclaimer": "partial",
        "abstract": "draft",
        "research_question": "draft",
        "system_overview": "draft",
        "data_pipeline": _section_status(available=bool(has_data_rows), has_detail=bool(data_status)),
        "historical_backfill_and_validation": _section_status(
            available=phase3 is not None or bool(data_status.get("warnings") is not None),
            has_detail=phase3 is not None,
        ),
        "feature_engineering": _section_status(
            available=bool(feature_audits),
            has_detail=bool(feature_audits),
        ),
        "labeling_and_evaluation": _section_status(
            available=accepted.get("available") is True,
            has_detail=accepted.get("available") is True,
        ),
        "candidate_timeline": _section_status(
            available=True,
            has_detail=bool(context.get("candidate_timeline")),
        ),
        "current_results_summary": _section_status(
            available=ledger.get("available") or ledger_summary.get("available"),
            has_detail=ledger_summary.get("available") is True,
        ),
        "failure_analysis": _section_status(
            available=bool(context.get("blockers")),
            has_detail=bool(context.get("blockers")),
        ),
        "robustness_checks": _section_status(
            available=ledger.get("available"),
            has_detail=ledger.get("row_count", 0) > 0,
        ),
        "dashboard_and_demo_artifacts": _section_status(
            available=dashboard.get("available") or demo_packet.get("available"),
            has_detail=dashboard.get("available") is True,
        ),
        "limitations": "partial",
        "next_steps": _section_status(
            available=bool(context.get("next_actions")),
            has_detail=bool(context.get("next_actions")),
        ),
        "appendix": "partial",
        "research_status_input": "partial" if research_status_wrapper.get("available") else "draft",
    }


def build_final_report_skeleton(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    title: str = DEFAULT_TITLE,
) -> dict[str, Any]:
    """Build the final report skeleton payload from local artifacts."""
    research_status_wrapper = _load_research_status(reports_root)
    ledger = _load_candidate_ledger(reports_root)
    dashboard = _load_dashboard_summary(reports_root)
    demo_packet = _load_demo_packet(reports_root)
    feature_audits = _latest_feature_audits(reports_root)
    phase3 = _load_json(reports_root / PHASE3_AUDIT_REL)

    context = _resolve_research_context(
        reports_root=reports_root,
        data_root=data_root,
        research_status_wrapper=research_status_wrapper,
    )
    data_row_counts = _data_row_counts(context["data_status"])
    section_status = _build_section_metadata(
        research_status_wrapper=research_status_wrapper,
        ledger=ledger,
        dashboard=dashboard,
        demo_packet=demo_packet,
        feature_audits=feature_audits,
        context=context,
        phase3=phase3,
    )

    return {
        "generated_at": _utc_now_iso(),
        "title": title,
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "real_execution_ready": False,
        "safety_statement": SAFETY_STATEMENT,
        "research_only_note": RESEARCH_ONLY_NOTE,
        "profitability_disclaimer": PROFITABILITY_DISCLAIMER,
        "does_not_claim_profitability": True,
        "sections": list(REPORT_SECTIONS),
        "section_status": section_status,
        "safety": {
            "paper_only": PAPER_ONLY,
            "execution_enabled": EXECUTION_ENABLED,
            "real_execution_ready": False,
            "safety_statement": SAFETY_STATEMENT,
        },
        "inputs": {
            "reports_root": str(reports_root),
            "data_root": str(data_root),
            "research_status": research_status_wrapper,
            "candidate_ledger": ledger,
            "dashboard_summary": dashboard,
            "demo_packet": demo_packet,
            "feature_audits": feature_audits,
            "phase3_available": phase3 is not None,
            "context_source": context["source"],
        },
        "data_row_counts": data_row_counts,
        "data_status": context["data_status"],
        "candidate_timeline": context["candidate_timeline"],
        "ledger_summary": context["ledger_summary"],
        "accepted_trades_analysis": context["accepted_trades_analysis"],
        "blockers": context["blockers"],
        "next_actions": context["next_actions"],
        "key_commands": list(KEY_COMMANDS),
        "artifact_paths": {
            "research_status": str(reports_root / RESEARCH_STATUS_REL),
            "candidate_ledger": str(reports_root / CANDIDATE_LEDGER_REL),
            "dashboard_summary": str(reports_root / DASHBOARD_SUMMARY_REL),
            "demo_packet": str(reports_root / DEMO_PACKET_REL),
            "phase3_audit": str(reports_root / PHASE3_AUDIT_REL),
            "feature_audit_dir": str(reports_root / "feature_audit"),
            "accepted_trades_analysis_dir": str(reports_root / "accepted_trades_analysis"),
            "data_root": str(data_root),
        },
    }


def _format_data_row_counts(data_row_counts: dict[str, dict[str, int | None]]) -> list[str]:
    lines: list[str] = []
    for coin in RESEARCH_COINS:
        intervals = data_row_counts.get(coin) or {}
        parts = []
        for interval in ("1s", "15s", "1m"):
            count = intervals.get(interval)
            parts.append(f"{interval}={count if count is not None else 'n/a'}")
        lines.append(f"- {coin}: {', '.join(parts)}")
    return lines


def _format_candidate_timeline(timeline: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for entry in timeline:
        candidate = entry.get("candidate")
        lines.append(f"### Candidate {candidate}: {entry.get('name')}")
        lines.append("")
        lines.append(f"- Status: {entry.get('status')}")
        lines.append(f"- Summary: {entry.get('summary')}")
        lines.append("")
    return lines


def write_final_report_skeleton_markdown(
    skeleton: dict[str, Any],
    output_dir: Path,
    *,
    filename: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    section_status = skeleton.get("section_status") or {}

    lines = [
        f"# {skeleton.get('title', DEFAULT_TITLE)}",
        "",
        f"Generated at: {skeleton['generated_at']}",
        "",
        "## 1. Title and Safety Disclaimer",
        "",
        _status_line(section_status.get("title_and_safety_disclaimer", "partial")),
        "",
        f"- paper_only: {str(skeleton['paper_only']).lower()}",
        f"- execution_enabled: {str(skeleton['execution_enabled']).lower()}",
        f"- real_execution_ready: {str(skeleton['real_execution_ready']).lower()}",
        "",
        skeleton.get("safety_statement", SAFETY_STATEMENT),
        "",
        skeleton.get("research_only_note", RESEARCH_ONLY_NOTE),
        "",
        skeleton.get("profitability_disclaimer", PROFITABILITY_DISCLAIMER),
        "",
        "## 2. Abstract",
        "",
        _status_line(section_status.get("abstract", "draft")),
        "",
        "- High-level summary of the bid/ask ML research program.",
        "- To be updated once walk-forward and candidate validation narratives are finalized.",
        "- Current artifacts indicate mixed diagnostic progress with no robust positive edge.",
        "",
        "## 3. Research Question",
        "",
        _status_line(section_status.get("research_question", "draft")),
        "",
        "- Can short-horizon bid/ask microstructure features predict cost-aware, tradeable edge on BTC/ETH?",
        "- Sub-questions: which label modes, feature families, and horizons survive walk-forward costs?",
        "",
        "## 4. System Overview",
        "",
        _status_line(section_status.get("system_overview", "draft")),
        "",
        "- Local CSV collection → validation/resample → feature audit → training → walk-forward → ledger.",
        "- Paper-only evaluation; no live execution path.",
        "",
        "## 5. Data Pipeline",
        "",
        _status_line(section_status.get("data_pipeline", "draft")),
        "",
        "**Latest row counts**",
        "",
    ]
    lines.extend(_format_data_row_counts(skeleton.get("data_row_counts") or {}))
    lines.extend(
        [
            "",
            "- Pipeline stages: 1s BBO capture, 15s/1m resample, validator reports, optional repair.",
            "- See data_root and validation artifacts in Appendix.",
            "",
            "## 6. Historical Backfill and Validation",
            "",
            _status_line(section_status.get("historical_backfill_and_validation", "draft")),
            "",
        ]
    )

    inputs = skeleton.get("inputs") or {}
    if inputs.get("phase3_available"):
        lines.append("- Phase 3 audit report available — import/resample coverage embedded in dashboard.")
    else:
        lines.append("- Phase 3 audit report not available — backfill metrics pending.")

    data_status = (skeleton.get("inputs") or {}).get("research_status", {})
    context_source = inputs.get("context_source", "unknown")
    lines.append(f"- Context source: {context_source}")
    if data_status.get("available"):
        lines.append(f"- Research status loaded from: {data_status.get('path')}")
    else:
        lines.append("- Research status not built — using direct artifact scan where possible.")

    data_warnings = list((skeleton.get("data_status") or {}).get("warnings") or [])
    if not data_warnings:
        for coin in RESEARCH_COINS:
            coin_rows = (skeleton.get("data_row_counts") or {}).get(coin) or {}
            if any(count in (None, 0) for count in coin_rows.values()):
                data_warnings.append(f"{coin}: incomplete interval coverage")

    if data_warnings:
        lines.extend(["", "**Validator / coverage warnings**", ""])
        lines.extend(f"- {warning}" for warning in data_warnings[:8])
    lines.append("")

    lines.extend(
        [
            "## 7. Feature Engineering",
            "",
            _status_line(section_status.get("feature_engineering", "draft")),
            "",
            "### Candidate C core",
            "",
            "- Spread/cost/regime and microstructure feature families from candidate_c_core.",
            "- Feature audit reports summarize missingness, quantiles, and label correlations.",
            "",
            "### Candidate E depth-aware",
            "",
            "- Depth/size/imbalance features requiring bid_size and ask_size columns.",
            "- ETH UP H20 showed sparse positive signal; robustness not met.",
            "",
        ]
    )

    feature_audits = inputs.get("feature_audits") or []
    if feature_audits:
        lines.extend(["**Latest feature audit artifacts**", ""])
        for row in feature_audits:
            families = ", ".join(row.get("feature_families") or []) or "n/a"
            lines.append(
                f"- {row.get('coin')} {row.get('interval')}: rows={row.get('row_count')}, "
                f"families={families}"
            )
        lines.append("")
    else:
        lines.extend(["- No feature audit artifacts found.", ""])

    lines.extend(
        [
            "## 8. Labeling and Evaluation",
            "",
            _status_line(section_status.get("labeling_and_evaluation", "draft")),
            "",
            "### Direction labels",
            "",
            "- UP/DOWN/FLAT classification against mid-price horizon moves.",
            "",
            "### Cost-aware labels",
            "",
            "- Candidate D framing: labels net of fee/slippage assumptions.",
            "- Useful for diagnostics; not sufficient alone for predictive edge.",
            "",
            "### Binary tradeable-edge labels",
            "",
            "- Tradeable UP/DOWN after costs; used in walk-forward paper filters.",
            "",
        ]
    )

    accepted = skeleton.get("accepted_trades_analysis") or {}
    if accepted.get("available"):
        lines.extend(["**Accepted-trades analysis (local artifacts only)**", ""])
        for coin in RESEARCH_COINS:
            row = (accepted.get("by_coin") or {}).get(coin) or {}
            if not row.get("available"):
                lines.append(f"- {coin}: no analysis report")
                continue
            lines.append(
                f"- {coin}: trades={row.get('trade_count')}, "
                f"cumulative_net={row.get('cumulative_net_return_bps')} bps "
                f"(research metric — not proof of profitability)"
            )
        lines.append("")
    else:
        lines.extend(["- Accepted-trades analysis not available.", ""])

    lines.extend(["## 9. Candidate Timeline", ""])
    lines.append(_status_line(section_status.get("candidate_timeline", "draft")))
    lines.append("")
    lines.extend(_format_candidate_timeline(skeleton.get("candidate_timeline") or []))

    lines.extend(
        [
            "## 10. Current Results Summary",
            "",
            _status_line(section_status.get("current_results_summary", "draft")),
            "",
        ]
    )

    ledger = inputs.get("candidate_ledger") or {}
    ledger_summary = skeleton.get("ledger_summary") or {}
    lines.append(f"- Candidate ledger rows: {ledger.get('row_count', 0)}")
    if ledger.get("available"):
        lines.append(f"- Ledger path: {ledger.get('path')}")
    else:
        lines.append("- Candidate ledger not available.")

    top_scores = ledger_summary.get("top_by_paper_score") or []
    if top_scores:
        lines.extend(["", "**Top grid configs by paper_score (if present)**", ""])
        for row in top_scores[:5]:
            sparse = " **SPARSE (<10 trades)**" if row.get("sparse_warning") else ""
            positive = row.get("positive_after_costs")
            lines.append(
                f"- {row.get('config_id') or row.get('coin')}: "
                f"score={row.get('paper_score')}, trades={row.get('trade_count')}, "
                f"positive_after_costs={positive}{sparse}"
            )
    else:
        lines.append("- No grid ranking available — results summary incomplete.")

    lines.extend(
        [
            "",
            "- No market-ready edge claimed. Positive snippets may be sparse or rejected.",
            "",
            "## 11. Failure Analysis",
            "",
            _status_line(section_status.get("failure_analysis", "draft")),
            "",
        ]
    )
    for blocker in skeleton.get("blockers") or []:
        lines.append(f"- {blocker}")
    if not skeleton.get("blockers"):
        lines.append("- Failure analysis pending — blockers not yet derived.")
    lines.append("")

    lines.extend(
        [
            "## 12. Robustness Checks",
            "",
            _status_line(section_status.get("robustness_checks", "draft")),
            "",
            "- Minimum-trade threshold (10 accepted paper trades) applied in candidate grid.",
            "- Sparse positives flagged; depth-aware Candidate E failed robustness despite sparse gains.",
            "- Walk-forward and robustness JSON reports referenced via candidate ledger.",
            "",
            "## 13. Dashboard and Demo Artifacts",
            "",
            _status_line(section_status.get("dashboard_and_demo_artifacts", "draft")),
            "",
        ]
    )

    dashboard = inputs.get("dashboard_summary") or {}
    demo = inputs.get("demo_packet") or {}
    if dashboard.get("available"):
        lines.append(f"- Dashboard summary: {dashboard.get('path')}")
        action = dashboard.get("next_recommended_action")
        if action:
            lines.append(f"- Next recommended action: {action}")
    else:
        lines.append("- Dashboard summary not built yet.")

    if demo.get("available"):
        lines.append(f"- Demo packet: {demo.get('path')}")
        for item in (demo.get("what_not_to_claim") or [])[:4]:
            lines.append(f"  - Do not claim: {item}")
    else:
        lines.append("- Demo packet not built yet.")
    lines.append("")

    lines.extend(
        [
            "## 14. Limitations",
            "",
            _status_line(section_status.get("limitations", "partial")),
            "",
            "- Limited historical depth vs. production market regimes.",
            "- Paper fills and cost assumptions may not match live microstructure.",
            "- Multiple candidate experiments remain negative or sparse after costs.",
            "- Skeleton sections marked Draft until backfill and validation mature.",
            "",
            "## 15. Next Steps",
            "",
            _status_line(section_status.get("next_steps", "draft")),
            "",
        ]
    )
    for action in skeleton.get("next_actions") or []:
        lines.append(f"- {action}")
    if not skeleton.get("next_actions"):
        lines.append("- Next steps pending — run research status builder after more data lands.")
    lines.append("")

    lines.extend(
        [
            "## 16. Appendix: Commands and Artifact Paths",
            "",
            _status_line(section_status.get("appendix", "partial")),
            "",
            "### Key commands",
            "",
        ]
    )
    for command in skeleton.get("key_commands") or KEY_COMMANDS:
        lines.append(f"- `{command}`")
    lines.extend(["", "### Artifact paths", ""])
    for key, path in (skeleton.get("artifact_paths") or {}).items():
        lines.append(f"- {key}: `{path}`")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_final_report_skeleton_json(
    skeleton: dict[str, Any],
    output_dir: Path,
    *,
    filename: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    output_path.write_text(json.dumps(skeleton, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def build_and_write_final_report_skeleton(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    title: str = DEFAULT_TITLE,
    latest_json_name: str = DEFAULT_LATEST_JSON_NAME,
    latest_md_name: str = DEFAULT_LATEST_MD_NAME,
) -> dict[str, Any]:
    output_dir = _validate_output_dir(output_dir)
    skeleton = build_final_report_skeleton(
        reports_root=reports_root,
        data_root=data_root,
        title=title,
    )
    stamp = _utc_stamp()

    json_stamped = write_final_report_skeleton_json(
        skeleton,
        output_dir,
        filename=f"{STAMPED_JSON_PREFIX}{stamp}.json",
    )
    md_stamped = write_final_report_skeleton_markdown(
        skeleton,
        output_dir,
        filename=f"{STAMPED_MD_PREFIX}{stamp}.md",
    )
    json_latest = write_final_report_skeleton_json(skeleton, output_dir, filename=latest_json_name)
    md_latest = write_final_report_skeleton_markdown(skeleton, output_dir, filename=latest_md_name)

    skeleton["output_paths"] = {
        "json_latest": str(json_latest),
        "markdown_latest": str(md_latest),
        "json_stamped": str(json_stamped),
        "markdown_stamped": str(md_stamped),
        "stamp": stamp,
    }
    return skeleton


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build final research report skeleton (paper-only, read-only inputs).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build",
        help="Read local artifacts and write final report skeleton files.",
    )
    build_parser.add_argument(
        "--reports-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help="Root directory for research reports.",
    )
    build_parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for bid/ask CSV data and validation reports.",
    )
    build_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for final report skeleton output files.",
    )
    build_parser.add_argument(
        "--title",
        type=str,
        default=DEFAULT_TITLE,
        help="Report title for the skeleton markdown.",
    )

    args = parser.parse_args(argv)

    if args.command != "build":
        parser.error(f"Unknown command: {args.command}")
        return 2

    skeleton = build_and_write_final_report_skeleton(
        reports_root=args.reports_root,
        data_root=args.data_root,
        output_dir=args.output_dir,
        title=args.title,
    )
    paths = skeleton["output_paths"]
    print(f"json_latest_path={paths['json_latest']}")
    print(f"markdown_latest_path={paths['markdown_latest']}")
    print(f"json_stamped_path={paths['json_stamped']}")
    print(f"markdown_stamped_path={paths['markdown_stamped']}")
    print(f"paper_only={str(skeleton['paper_only']).lower()}")
    print(f"execution_enabled={str(skeleton['execution_enabled']).lower()}")
    print(f"real_execution_ready={str(skeleton['real_execution_ready']).lower()}")
    print(f"does_not_claim_profitability={str(skeleton['does_not_claim_profitability']).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
