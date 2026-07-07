"""Standalone demo mode packet preparer for bid/ask research presentations.

Paper-only and research-only. Reads local JSON/CSV artifacts and writes a
presentation packet under reports/demo_mode/. No orders, wallets, keys,
network, AWS, downloads, or subprocess calls.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT
from bid_ask_validator import bid_ask_csv_path, reports_dir

DEFAULT_REPORTS_ROOT = Path("reports")
DEFAULT_DATA_ROOT = Path("data/bid_ask_csv")
DEFAULT_OUTPUT_DIR = Path("reports/demo_mode")

PHASE3_AUDIT_REL = Path("bid_ask_research_scheduler/phase3_audit_latest.json")
DAILY_SUMMARY_GLOB = "bid_ask_daily/bid_ask_daily_summary_*.json"
WALK_FORWARD_DIR_REL = Path("bid_ask_walk_forward")
CANDIDATE_VALIDATION_DIR_REL = Path("bid_ask_candidate_validation")

RESEARCH_COINS = ("BTC", "ETH")
HISTORICAL_SOURCE = "historical"
PHASE3_TARGET_HOURS = 14 * 24
DASHBOARD_LAUNCH_COMMAND = "streamlit run bid_ask_research_dashboard.py"

PROJECT_EXPLANATION_30S = (
    "This is a paper-only Hyperliquid bid/ask microstructure research lab. "
    "We ingest local L2-derived CSVs, validate and resample them, train "
    "directional classifiers, and evaluate candidates with a walk-forward "
    "paper backtest. The dashboard summarizes pipeline progress, data quality, "
    "and candidate trade-offs — without placing real orders or touching wallets."
)

ARCHITECTURE_TEXT = (
    "Hyperliquid S3 / WebSocket → Raw L2 Data → Bid/Ask Importer → 1s CSV → "
    "15s / 1m Resampler → Validator → Feature Builder → Walk-Forward Evaluator → "
    "Candidate Validator → Paper Decision Dashboard"
)

MILESTONES: tuple[tuple[str, int, str], ...] = (
    ("Data Pipeline Works", 80, "Historical 1s CSVs advancing for BTC and ETH."),
    ("Paper Backtest Engine", 85, "Walk-forward evaluator producing paper trade reports."),
    ("First Candidate Found", 40, "Candidate B is positive but on very few trades."),
    ("Candidate Scaling", 20, "Relaxing filters adds trades but destroys post-cost returns."),
    ("Robustness", 15, "Limited windows; concentrated time buckets flagged."),
    ("Out-of-Sample Paper Test", 0, "Not started — waiting for a frozen candidate."),
    ("Paper Operations", 70, "Paper decision viewer and validation loop active."),
    ("Market Ready", 0, "Real execution disabled — not market ready."),
)

NEXT_RESEARCH_PLAN = [
    "Finish BTC/ETH 14-day sprint",
    "Inspect losing trades",
    "Test horizons 10 / 15 / 20 / 30 / 40",
    "Compare BTC vs ETH",
    "Build candidate grid",
    "Freeze promising config",
    "Out-of-sample test",
]

WHAT_NOT_TO_CLAIM = [
    "Do not claim the strategy is profitable or market ready.",
    "Do not claim paper backtest returns will transfer to live trading.",
    "Do not claim model accuracy alone proves edge after costs.",
    "Do not claim Candidate B is validated at scale — only 5 strict trades so far.",
    "Do not claim AWS backfill or live execution is enabled in this demo.",
    "Do not claim missing data gaps have been fully repaired.",
]

LIKELY_QA: tuple[tuple[str, str], ...] = (
    (
        "Is this live trading?",
        "No. Paper-only research. Real execution is disabled and no orders are placed.",
    ),
    (
        "Why only five Candidate B trades?",
        "Strict confidence and cost filters accept very few signals. Relaxing filters "
        "adds trades but cumulative net return turns sharply negative.",
    ),
    (
        "Does positive cumulative net bps mean profit?",
        "No. It is a simulated paper metric on a tiny sample. Research-only; "
        "does not prove profitability.",
    ),
    (
        "What data does the dashboard use?",
        "Only local CSV and JSON artifacts already on disk — no network downloads during the demo.",
    ),
    (
        "When will this be market ready?",
        "Not yet. Milestones for out-of-sample paper testing and market readiness are incomplete.",
    ),
    (
        "Can I run heavy training during the demo?",
        "No. Use this preparer beforehand; the demo reads precomputed reports only.",
    ),
)

DEMO_SCRIPT_STEPS: tuple[tuple[str, str], ...] = (
    ("0:00", "Open with safety boundary: paper-only, execution disabled, not market ready."),
    ("0:30", "State the 30-second project explanation — local bid/ask ML research pipeline."),
    ("1:00", "Show KPI cards: BTC/ETH row counts, latest timestamps, Candidate B status."),
    ("1:30", "Walk through data collection progress and Phase 3 14-day sprint bar."),
    ("2:00", "Review data quality badges for BTC/ETH 1s and 15s validation."),
    ("2:30", "Show walk-forward summary: model vs baseline accuracy, paper trade count."),
    ("3:00", "Explain Candidate B: strict config, +3.84 bps on 5 trades; relaxed configs fail."),
    ("3:30", "Open paper decision viewer — emphasize PAPER_ACCEPTED, not real fills."),
    ("4:00", "Scroll milestone journey and architecture pipeline diagram."),
    ("4:30", "Cover next research plan and what not to claim."),
    ("4:50", "Close with Q&A; remind audience this is research-only."),
)

DEMO_CHECKLIST = [
    "Run demo packet preparer before the meeting (this tool).",
    "Confirm phase3_audit_latest.json and walk-forward reports exist locally.",
    "Do not start resample, train, or backfill jobs during the presentation.",
    "Launch dashboard only when ready to present (command is in the packet).",
    "Keep safety banner visible: paper-only, execution disabled.",
    "Prepare answers for low trade count and missing-data caveats.",
    "Do not promise profitability, market readiness, or live trading.",
]

CANDIDATE_B_FALLBACK = {
    "config": "Strict Candidate B",
    "trades": 5,
    "cumulative_net_bps": 3.84,
    "status": "not_market_ready",
}


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _parse_report_stamp(path: Path) -> str:
    stem = path.stem
    if stem.startswith("walk_forward_"):
        return stem.rsplit("_", 1)[-1]
    if "_validation_" in stem:
        return stem.rsplit("_validation_", 1)[-1]
    if stem.startswith("bid_ask_daily_summary_"):
        return stem.removeprefix("bid_ask_daily_summary_")
    if stem.startswith("candidate_validation_"):
        return stem.rsplit("_", 1)[-1]
    return stem


def _latest_json(reports_root: Path, pattern: str) -> dict[str, Any] | None:
    matches = sorted(reports_root.glob(pattern), key=lambda path: path.name)
    for path in reversed(matches):
        payload = _load_json(path)
        if payload is not None:
            payload["_report_path"] = str(path)
            return payload
    return None


def _latest_walk_forward(reports_root: Path, coin: str) -> dict[str, Any] | None:
    wf_dir = reports_root / WALK_FORWARD_DIR_REL
    if not wf_dir.is_dir():
        return None
    pattern = f"walk_forward_{coin}_*.json"
    matches = sorted(wf_dir.glob(pattern), key=_parse_report_stamp, reverse=True)
    for path in matches:
        payload = _load_json(path)
        if payload is not None:
            payload["_report_path"] = str(path)
            return payload
    return None


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open(encoding="utf-8") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except OSError:
        return 0


def _coin_coverage(coin: str, phase3: dict[str, Any] | None, data_root: Path) -> dict[str, Any]:
    imported = (phase3 or {}).get("imported_historical_1s") or {}
    coin_data = imported.get(coin) or {}
    resampled = ((phase3 or {}).get("resampled_coverage") or {}).get(coin) or {}
    rows_1s = coin_data.get("row_count") or _count_csv_rows(
        bid_ask_csv_path(coin, "1s", data_root, source=HISTORICAL_SOURCE)
    )
    rows_15s = (resampled.get("15s") or {}).get("row_count") or _count_csv_rows(
        bid_ask_csv_path(coin, "15s", data_root, source=HISTORICAL_SOURCE)
    )
    return {
        "coin": coin,
        "rows_1s": rows_1s,
        "rows_15s": rows_15s,
        "last_timestamp": coin_data.get("last_timestamp"),
        "coverage_hours": coin_data.get("coverage_hours"),
    }


def validate_output_dir(output_dir: Path) -> Path:
    """Ensure writes stay under a demo_mode directory."""
    if output_dir.name != "demo_mode":
        raise ValueError(
            f"output_dir must end with demo_mode (got {output_dir}). "
            "Only writes under reports/demo_mode/ are permitted."
        )
    return output_dir


def discover_artifacts(
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    """Collect latest local artifact references without network or subprocess calls."""
    phase3_path = reports_root / PHASE3_AUDIT_REL
    phase3 = _load_json(phase3_path)
    if phase3 is not None:
        phase3["_report_path"] = str(phase3_path)

    daily_summary = _latest_json(reports_root, DAILY_SUMMARY_GLOB)
    candidate_dir = reports_root / CANDIDATE_VALIDATION_DIR_REL
    candidate = None
    if candidate_dir.is_dir():
        matches = sorted(
            candidate_dir.glob("candidate_validation_*.json"),
            key=_parse_report_stamp,
            reverse=True,
        )
        for path in matches:
            candidate = _load_json(path)
            if candidate is not None:
                candidate["_report_path"] = str(path)
                break

    walk_forward: dict[str, dict[str, Any] | None] = {}
    for coin in RESEARCH_COINS:
        walk_forward[coin] = _latest_walk_forward(reports_root, coin)

    validation_reports: dict[str, dict[str, Any] | None] = {}
    report_dir = reports_dir(data_root)
    if report_dir.is_dir():
        for coin in RESEARCH_COINS:
            for interval in ("1s", "15s"):
                pattern = f"{coin}_{interval}_validation_*.json"
                matches = sorted(report_dir.glob(pattern), key=_parse_report_stamp, reverse=True)
                key = f"{coin}_{interval}"
                validation_reports[key] = None
                for path in matches:
                    payload = _load_json(path)
                    if payload is not None:
                        payload["_report_path"] = str(path)
                        validation_reports[key] = payload
                        break

    return {
        "phase3_audit": phase3,
        "daily_summary": daily_summary,
        "candidate_validation": candidate,
        "walk_forward": walk_forward,
        "validation_reports": validation_reports,
    }


def build_current_kpis(
    artifacts: dict[str, Any],
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    """Build KPI snapshot from latest local artifacts."""
    phase3 = artifacts.get("phase3_audit")
    candidate = artifacts.get("candidate_validation")
    btc = _coin_coverage("BTC", phase3, data_root)
    eth = _coin_coverage("ETH", phase3, data_root)

    downloads_total = (phase3 or {}).get("downloads", {}).get("total_files")
    imported_hours = 0.0
    if phase3:
        imported = phase3.get("imported_historical_1s") or {}
        hours = [
            float((imported.get(coin) or {}).get("coverage_hours") or 0)
            for coin in RESEARCH_COINS
        ]
        hours = [value for value in hours if value > 0]
        imported_hours = sum(hours) / len(hours) if hours else 0.0

    cand_trades = (candidate or {}).get("filtered_paper_trades")
    cand_bps = (candidate or {}).get("cumulative_net_return_bps")
    wf_btc = (artifacts.get("walk_forward") or {}).get("BTC") or {}
    if cand_trades is None:
        cand_trades = wf_btc.get("filtered_paper_trades") or wf_btc.get("accepted_trades_count")
    if cand_bps is None:
        cand_bps = wf_btc.get("cumulative_net_return_bps")

    if cand_trades is None:
        cand_trades = CANDIDATE_B_FALLBACK["trades"]
    if cand_bps is None:
        cand_bps = CANDIDATE_B_FALLBACK["cumulative_net_bps"]

    status = f"{cand_trades} trades, {float(cand_bps):+.2f} bps (not market ready)"

    daily = artifacts.get("daily_summary") or {}
    overall = daily.get("overall_status") or {}

    return {
        "btc_1s_rows": btc["rows_1s"],
        "eth_1s_rows": eth["rows_1s"],
        "btc_15s_rows": btc["rows_15s"],
        "eth_15s_rows": eth["rows_15s"],
        "btc_last_timestamp": btc.get("last_timestamp"),
        "eth_last_timestamp": eth.get("last_timestamp"),
        "backfill_downloads_total": downloads_total,
        "phase3_imported_hours": round(imported_hours, 3),
        "phase3_target_hours": PHASE3_TARGET_HOURS,
        "candidate_b_trades": cand_trades,
        "candidate_b_cumulative_net_bps": cand_bps,
        "candidate_b_status": status,
        "daily_summary_ready_for_review": overall.get("ready_for_review"),
        "walk_forward_btc_windows": wf_btc.get("completed_windows"),
        "walk_forward_btc_model_accuracy": wf_btc.get("model_accuracy_mean"),
        "walk_forward_btc_baseline_accuracy": wf_btc.get("baseline_accuracy_mean"),
    }


def build_candidate_b_explanation(artifacts: dict[str, Any]) -> dict[str, Any]:
    """Summarize Candidate B trade-off story from local reports."""
    candidate = artifacts.get("candidate_validation") or {}
    wf_btc = (artifacts.get("walk_forward") or {}).get("BTC") or {}
    phase3 = artifacts.get("phase3_audit") or {}
    freshness = phase3.get("candidate_freshness") or {}

    trades = (
        candidate.get("filtered_paper_trades")
        or wf_btc.get("filtered_paper_trades")
        or wf_btc.get("accepted_trades_count")
        or CANDIDATE_B_FALLBACK["trades"]
    )
    cumulative_bps = (
        candidate.get("cumulative_net_return_bps")
        or wf_btc.get("cumulative_net_return_bps")
        or CANDIDATE_B_FALLBACK["cumulative_net_bps"]
    )

    return {
        "name": "BTC DOWN MICRO H20 CANDIDATE B (strict)",
        "trades": trades,
        "cumulative_net_bps": cumulative_bps,
        "status": candidate.get("status") or freshness.get("status") or "not_market_ready",
        "summary": (
            "Strict Candidate B filters for high-confidence DOWN micro signals. "
            f"It shows +{float(cumulative_bps):.2f} bps cumulative net return on "
            f"{trades} paper trades — positive but not statistically meaningful. "
            "Relaxing filters increases trade count but destroys post-cost performance."
        ),
        "main_takeaway": (
            "Relaxing filters increases trade count but destroys performance. "
            "The current blocker is finding a candidate that scales."
        ),
        "source_reports": {
            "candidate_validation": candidate.get("_report_path"),
            "walk_forward_btc": wf_btc.get("_report_path"),
        },
    }


def build_safety_boundary() -> dict[str, Any]:
    return {
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "real_execution_ready": False,
        "market_ready": False,
        "statement": SAFETY_STATEMENT,
        "demo_rules": [
            "Paper-only and research-only presentation.",
            "No orders, wallets, private keys, or signing.",
            "No AWS, network downloads, or subprocess calls from this preparer.",
            "Dashboard launch command is provided but not executed by this tool.",
            "Simulated returns are not profitability proof.",
        ],
    }


def build_milestone_journey() -> list[dict[str, Any]]:
    return [
        {"name": name, "progress_pct": pct, "status": status}
        for name, pct, status in MILESTONES
    ]


def build_demo_script(include_script: bool) -> list[dict[str, str]] | None:
    if not include_script:
        return None
    return [{"time": time_mark, "action": action} for time_mark, action in DEMO_SCRIPT_STEPS]


def build_demo_checklist(include_checklist: bool) -> list[str] | None:
    if not include_checklist:
        return None
    return list(DEMO_CHECKLIST)


def build_demo_packet(
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    include_script: bool = True,
    include_checklist: bool = True,
) -> dict[str, Any]:
    """Assemble the full demo packet from local artifacts."""
    artifacts = discover_artifacts(reports_root=reports_root, data_root=data_root)
    kpis = build_current_kpis(artifacts, data_root=data_root)

    artifact_paths = {
        key: (value or {}).get("_report_path") if isinstance(value, dict) else None
        for key, value in {
            "phase3_audit": artifacts.get("phase3_audit"),
            "daily_summary": artifacts.get("daily_summary"),
            "candidate_validation": artifacts.get("candidate_validation"),
            "walk_forward_btc": (artifacts.get("walk_forward") or {}).get("BTC"),
            "walk_forward_eth": (artifacts.get("walk_forward") or {}).get("ETH"),
        }.items()
    }

    warnings: list[str] = [
        "Research-only. Does not prove profitability.",
        "Demo packet uses locally collected artifacts only.",
    ]
    if artifacts.get("phase3_audit") is None:
        warnings.append("Phase 3 audit report not found — KPIs may use CSV fallbacks.")
    if (artifacts.get("walk_forward") or {}).get("BTC") is None:
        warnings.append("Latest BTC walk-forward report not found — Candidate B uses fallback values.")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": "prepare",
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "safety_statement": SAFETY_STATEMENT,
        "project_explanation_30s": PROJECT_EXPLANATION_30S,
        "demo_script_5min": build_demo_script(include_script),
        "demo_checklist": build_demo_checklist(include_checklist),
        "dashboard_launch_command": DASHBOARD_LAUNCH_COMMAND,
        "current_kpis": kpis,
        "candidate_b": build_candidate_b_explanation(artifacts),
        "safety_boundary": build_safety_boundary(),
        "milestone_journey": build_milestone_journey(),
        "architecture_text": ARCHITECTURE_TEXT,
        "likely_questions_and_answers": [
            {"question": question, "answer": answer} for question, answer in LIKELY_QA
        ],
        "what_not_to_claim": list(WHAT_NOT_TO_CLAIM),
        "next_research_plan": list(NEXT_RESEARCH_PLAN),
        "artifact_paths": artifact_paths,
        "warnings": warnings,
    }


def write_demo_packet_json(
    packet: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    validate_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "demo_packet_latest.json"
    output_path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    packet["json_report_path"] = str(output_path)
    return output_path


def write_demo_packet_markdown(
    packet: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    validate_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "demo_packet_latest.md"

    lines = [
        "# Bid/Ask Demo Mode Packet",
        "",
        f"- generated_at: {packet['generated_at']}",
        f"- paper_only: {packet['paper_only']}",
        f"- execution_enabled: {packet['execution_enabled']}",
        "",
        "## 30-Second Project Explanation",
        "",
        packet["project_explanation_30s"],
        "",
    ]

    if packet.get("demo_script_5min"):
        lines.extend(["## 5-Minute Demo Script", ""])
        for step in packet["demo_script_5min"]:
            lines.append(f"- **{step['time']}** — {step['action']}")
        lines.append("")

    lines.extend(
        [
            "## Dashboard Launch Command",
            "",
            f"```bash",
            f"{packet['dashboard_launch_command']}",
            f"```",
            "",
            "_Run manually before the demo; this preparer does not launch it._",
            "",
            "## Current KPIs",
            "",
        ]
    )
    for key, value in packet["current_kpis"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Candidate B Explanation", ""])
    candidate = packet["candidate_b"]
    lines.append(f"- name: {candidate['name']}")
    lines.append(f"- trades: {candidate['trades']}")
    lines.append(f"- cumulative_net_bps: {candidate['cumulative_net_bps']}")
    lines.append(f"- status: {candidate['status']}")
    lines.append(f"- summary: {candidate['summary']}")
    lines.append(f"- main_takeaway: {candidate['main_takeaway']}")
    lines.append("")

    lines.extend(["## Safety Boundary", ""])
    safety = packet["safety_boundary"]
    lines.append(f"- paper_only: {safety['paper_only']}")
    lines.append(f"- execution_enabled: {safety['execution_enabled']}")
    lines.append(f"- real_execution_ready: {safety['real_execution_ready']}")
    lines.append(f"- market_ready: {safety['market_ready']}")
    lines.append(f"- statement: {safety['statement']}")
    for rule in safety["demo_rules"]:
        lines.append(f"- {rule}")
    lines.append("")

    lines.extend(["## Milestone Journey", ""])
    for milestone in packet["milestone_journey"]:
        lines.append(
            f"- **{milestone['name']}** ({milestone['progress_pct']}%): {milestone['status']}"
        )
    lines.append("")

    lines.extend(["## Architecture", "", packet["architecture_text"], ""])

    lines.extend(["## Likely Questions and Answers", ""])
    for item in packet["likely_questions_and_answers"]:
        lines.append(f"**Q:** {item['question']}")
        lines.append(f"**A:** {item['answer']}")
        lines.append("")

    lines.extend(["## What Not to Claim", ""])
    for item in packet["what_not_to_claim"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.extend(["## Next Research Plan", ""])
    for item in packet["next_research_plan"]:
        lines.append(f"- {item}")
    lines.append("")

    if packet.get("demo_checklist"):
        lines.extend(["## Demo Checklist", ""])
        for item in packet["demo_checklist"]:
            lines.append(f"- [ ] {item}")
        lines.append("")

    lines.extend(["## Warnings", ""])
    for warning in packet["warnings"]:
        lines.append(f"- {warning}")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    packet["markdown_report_path"] = str(output_path)
    return output_path


def prepare_demo_mode(
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    include_script: bool = True,
    include_checklist: bool = True,
) -> dict[str, Any]:
    """Prepare demo packet files under output_dir."""
    validate_output_dir(output_dir)
    packet = build_demo_packet(
        reports_root=reports_root,
        data_root=data_root,
        include_script=include_script,
        include_checklist=include_checklist,
    )
    json_path = write_demo_packet_json(packet, output_dir=output_dir)
    markdown_path = write_demo_packet_markdown(packet, output_dir=output_dir)
    packet["output_dir"] = str(output_dir)
    packet["json_report_path"] = str(json_path)
    packet["markdown_report_path"] = str(markdown_path)
    return packet


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a read-only bid/ask demo presentation packet from local artifacts "
            "(paper-only, no subprocesses, no network)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Generate demo_packet_latest.json and demo_packet_latest.md",
    )
    prepare.add_argument(
        "--reports-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help="Root directory for research reports (default: reports)",
    )
    prepare.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for bid/ask CSV data (default: data/bid_ask_csv)",
    )
    prepare.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for demo packet files (default: reports/demo_mode)",
    )
    prepare.add_argument(
        "--include-script",
        type=_parse_bool,
        default=True,
        help="Include 5-minute demo script (default: true)",
    )
    prepare.add_argument(
        "--include-checklist",
        type=_parse_bool,
        default=True,
        help="Include demo checklist (default: true)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command != "prepare":
        parser.error(f"unsupported command: {args.command}")

    packet = prepare_demo_mode(
        reports_root=args.reports_root,
        data_root=args.data_root,
        output_dir=args.output_dir,
        include_script=args.include_script,
        include_checklist=args.include_checklist,
    )
    print(f"json_report_path={packet['json_report_path']}")
    print(f"markdown_report_path={packet['markdown_report_path']}")
    print(f"dashboard_launch_command={packet['dashboard_launch_command']}")
    for warning in packet["warnings"]:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
