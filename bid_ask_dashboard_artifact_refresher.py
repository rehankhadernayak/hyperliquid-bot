"""Standalone read-only refresher for bid/ask research dashboard artifacts.

Paper-only and research-only. Reads local JSON/CSV reports and writes a dashboard
summary JSON and Markdown under reports/dashboard_artifacts/. No execution, no AWS,
no shell commands, no network calls.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT
from bid_ask_validator import bid_ask_csv_path

DEFAULT_REPORTS_ROOT = Path("reports")
DEFAULT_DATA_ROOT = Path("data/bid_ask_csv")
DEFAULT_OUTPUT_DIR = Path("reports/dashboard_artifacts")
DEFAULT_JSON_NAME = "dashboard_summary_latest.json"
DEFAULT_MARKDOWN_NAME = "dashboard_summary_latest.md"

RESEARCH_COINS = ("BTC", "ETH")
HISTORICAL_SOURCE = "historical"
PHASE3_TARGET_HOURS = 14 * 24

PHASE3_AUDIT_REL = Path("bid_ask_research_scheduler/phase3_audit_latest.json")
WALK_FORWARD_REL = Path("bid_ask_walk_forward")
CANDIDATE_VALIDATION_REL = Path("bid_ask_candidate_validation")

CANDIDATE_B_STRICT_NAME = "walk_forward_BTC_20260706T115051.json"
CANDIDATE_B_RELAXED_25_NAME = "walk_forward_BTC_20260706T142556.json"
CANDIDATE_B_RELAXED_62_NAME = "walk_forward_BTC_20260706T120135.json"

CANDIDATE_B_FALLBACKS: tuple[dict[str, Any], ...] = (
    {"config": "Strict Candidate B", "trades": 5, "cumulative_net_bps": 3.84},
    {"config": "Relaxed 25-trade", "trades": 25, "cumulative_net_bps": -101.82},
    {"config": "Relaxed 62-trade", "trades": 62, "cumulative_net_bps": -267.95},
)

MILESTONES: tuple[tuple[str, int, str], ...] = (
    ("Data Pipeline Works", 80, "Historical 1s CSVs advancing for BTC and ETH."),
    ("Paper Backtest Engine", 85, "Walk-forward evaluator producing paper trade reports."),
    ("First Candidate Found", 40, "Candidate B is positive (+3.84 bps) but only 5 trades."),
    ("Candidate Scaling", 20, "Relaxing filters adds trades but destroys post-cost returns."),
    ("Robustness", 15, "Limited windows; concentrated time buckets flagged."),
    ("Out-of-Sample Paper Test", 0, "Not started — waiting for a frozen candidate."),
    ("Paper Operations", 70, "Paper decision viewer and validation loop active."),
    ("Market Ready", 0, "Real execution disabled — not market ready."),
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
    if stem.startswith("candidate_validation_"):
        return stem.rsplit("_", 1)[-1]
    return stem


def _fmt_ts_ms(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return sum(1 for _ in reader)
    except OSError:
        return 0


def _csv_timestamp_bounds(path: Path, *, interval: str) -> tuple[int | None, int | None]:
    if not path.is_file():
        return None, None
    column = "timestamp_ms" if interval == "1s" else "start_timestamp_ms"
    first_ts: int | None = None
    last_ts: int | None = None
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw = row.get(column)
                if not raw:
                    continue
                value = int(float(raw))
                if first_ts is None:
                    first_ts = value
                last_ts = value
    except (OSError, ValueError):
        return None, None
    return first_ts, last_ts


def _phase3_audit_path(reports_root: Path) -> Path:
    return reports_root / PHASE3_AUDIT_REL


def _walk_forward_dir(reports_root: Path) -> Path:
    return reports_root / WALK_FORWARD_REL


def _candidate_validation_dir(reports_root: Path) -> Path:
    return reports_root / CANDIDATE_VALIDATION_REL


def _latest_validation_report(
    coin: str,
    interval: str,
    data_root: Path,
) -> dict[str, Any] | None:
    reports_dir = data_root / "reports"
    if not reports_dir.is_dir():
        return None
    pattern = f"{coin}_{interval}_validation_*.json"
    candidates = sorted(reports_dir.glob(pattern), key=_parse_report_stamp, reverse=True)
    for path in candidates:
        payload = _load_json(path)
        if payload is not None:
            payload = dict(payload)
            payload["_report_path"] = str(path)
            return payload
    return None


def _collect_validation_reports(data_root: Path) -> list[dict[str, Any]]:
    reports_dir = data_root / "reports"
    if not reports_dir.is_dir():
        return []
    reports: list[dict[str, Any]] = []
    for path in sorted(reports_dir.glob("*validation*.json"), key=_parse_report_stamp, reverse=True):
        payload = _load_json(path)
        if payload is None:
            continue
        entry = dict(payload)
        entry["_report_path"] = str(path)
        entry["_stamp"] = _parse_report_stamp(path)
        reports.append(entry)
    return reports


def _latest_walk_forward_report(
    reports_root: Path,
    *,
    coin: str | None = None,
) -> dict[str, Any] | None:
    walk_forward_dir = _walk_forward_dir(reports_root)
    if not walk_forward_dir.is_dir():
        return None
    pattern = f"walk_forward_{coin}_*.json" if coin else "walk_forward_*.json"
    candidates = sorted(walk_forward_dir.glob(pattern), key=_parse_report_stamp, reverse=True)
    for path in candidates:
        payload = _load_json(path)
        if payload is not None:
            payload = dict(payload)
            payload["_report_path"] = str(path)
            payload["_report_name"] = path.name
            payload["_stamp"] = _parse_report_stamp(path)
            return payload
    return None


def _walk_forward_summary_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    coins = payload.get("coins") or []
    return {
        "report_path": payload.get("_report_path"),
        "report_name": payload.get("_report_name"),
        "stamp": payload.get("_stamp"),
        "coin": coins[0] if coins else None,
        "interval": payload.get("interval"),
        "horizon": payload.get("horizon"),
        "windows": payload.get("completed_windows"),
        "model_accuracy": payload.get("model_accuracy_mean"),
        "baseline_accuracy": payload.get("baseline_accuracy_mean"),
        "accuracy_edge": payload.get("accuracy_edge_vs_baseline"),
        "cumulative_net_bps": payload.get("cumulative_net_return_bps"),
        "accepted_trades": payload.get("filtered_paper_trades") or payload.get("accepted_trades_count"),
        "positive_after_costs": payload.get("positive_after_costs"),
        "min_confidence": payload.get("min_confidence"),
        "min_edge_bps": payload.get("min_edge_bps"),
    }


def _latest_candidate_validation(reports_root: Path) -> dict[str, Any] | None:
    candidate_dir = _candidate_validation_dir(reports_root)
    if not candidate_dir.is_dir():
        return None
    candidates = sorted(
        candidate_dir.glob("candidate_validation_*.json"),
        key=_parse_report_stamp,
        reverse=True,
    )
    for path in candidates:
        payload = _load_json(path)
        if payload is not None:
            payload = dict(payload)
            payload["_report_path"] = str(path)
            return payload
    return None


def _coin_coverage(
    coin: str,
    phase3: dict[str, Any] | None,
    data_root: Path,
) -> dict[str, Any]:
    imported = (phase3 or {}).get("imported_historical_1s") or {}
    coin_data = imported.get(coin) or {}
    resampled = ((phase3 or {}).get("resampled_coverage") or {}).get(coin) or {}

    path_1s = bid_ask_csv_path(coin, "1s", data_root, source=HISTORICAL_SOURCE)
    path_15s = bid_ask_csv_path(coin, "15s", data_root, source=HISTORICAL_SOURCE)

    rows_1s = coin_data.get("row_count") or _count_csv_rows(path_1s)
    rows_15s = (resampled.get("15s") or {}).get("row_count") or _count_csv_rows(path_15s)

    first_ms = coin_data.get("first_timestamp_ms")
    last_ms = coin_data.get("last_timestamp_ms")
    if first_ms is None or last_ms is None:
        csv_first, csv_last = _csv_timestamp_bounds(path_1s, interval="1s")
        first_ms = first_ms or csv_first
        last_ms = last_ms or csv_last

    return {
        "coin": coin,
        "path_1s": str(path_1s),
        "path_15s": str(path_15s),
        "rows_1s": rows_1s,
        "rows_15s": rows_15s,
        "first_timestamp_ms": first_ms,
        "last_timestamp_ms": last_ms,
        "last_timestamp": coin_data.get("last_timestamp") or _fmt_ts_ms(last_ms),
        "coverage_hours": coin_data.get("coverage_hours"),
        "freshness_status": coin_data.get("freshness_status"),
    }


def _phase3_imported_hours(phase3: dict[str, Any] | None) -> float:
    if not phase3:
        return 0.0
    imported = phase3.get("imported_historical_1s") or {}
    hours = [float((imported.get(coin) or {}).get("coverage_hours") or 0) for coin in RESEARCH_COINS]
    hours = [value for value in hours if value > 0]
    return sum(hours) / len(hours) if hours else 0.0


def _summarize_phase_backfill(phase3: dict[str, Any] | None) -> dict[str, Any]:
    if phase3 is None:
        return {
            "available": False,
            "audit_path": None,
            "imported_hours_mean": 0.0,
            "target_hours": PHASE3_TARGET_HOURS,
            "progress_fraction": 0.0,
            "downloads_total_files": None,
            "scheduler_latest_action": None,
            "warnings": [],
        }

    imported_hours = _phase3_imported_hours(phase3)
    progress = min(imported_hours / PHASE3_TARGET_HOURS, 1.0) if PHASE3_TARGET_HOURS else 0.0
    scheduler = phase3.get("scheduler_progress") or {}
    downloads = phase3.get("downloads") or {}
    phase_config = phase3.get("phase_config") or {}

    return {
        "available": True,
        "audit_path": str(PHASE3_AUDIT_REL),
        "generated_at": phase3.get("generated_at"),
        "baseline_date": phase3.get("baseline_date"),
        "imported_hours_mean": round(imported_hours, 3),
        "target_hours": PHASE3_TARGET_HOURS,
        "progress_fraction": round(progress, 4),
        "downloads_total_files": downloads.get("total_files"),
        "scheduler_latest_action": scheduler.get("latest_action"),
        "phase_backfill_name": phase_config.get("phase_backfill_name"),
        "focused_coins": phase_config.get("focused_coins") or list(RESEARCH_COINS),
        "warnings": phase3.get("warnings") or [],
    }


def _summarize_data_coverage(
    phase3: dict[str, Any] | None,
    data_root: Path,
) -> dict[str, Any]:
    coins = [_coin_coverage(coin, phase3, data_root) for coin in RESEARCH_COINS]
    return {
        "source": HISTORICAL_SOURCE,
        "coins": coins,
        "imported_hours_mean": round(_phase3_imported_hours(phase3), 3),
        "target_hours": PHASE3_TARGET_HOURS,
    }


def _summarize_validation(data_root: Path) -> dict[str, Any]:
    by_coin_interval: dict[str, dict[str, Any]] = {}
    for coin in RESEARCH_COINS:
        by_coin_interval[coin] = {}
        for interval in ("1s", "15s"):
            report = _latest_validation_report(coin, interval, data_root)
            if report is None:
                by_coin_interval[coin][interval] = {"available": False}
                continue
            by_coin_interval[coin][interval] = {
                "available": True,
                "report_path": report.get("_report_path"),
                "passed": report.get("passed"),
                "row_count": report.get("row_count"),
                "coverage_start": report.get("coverage_start"),
                "coverage_end": report.get("coverage_end"),
                "duplicate_timestamp_count": report.get("duplicate_timestamp_count"),
                "missing_periods_count": report.get("missing_periods_count"),
                "warnings": report.get("warnings") or [],
                "generated_at": report.get("generated_at"),
            }

    all_reports = _collect_validation_reports(data_root)
    return {
        "by_coin_interval": by_coin_interval,
        "report_count": len(all_reports),
        "latest_reports": all_reports[:6],
    }


def _walk_forward_summary_from_path(path: Path) -> dict[str, Any] | None:
    payload = _load_json(path)
    if payload is None:
        return None
    payload = dict(payload)
    payload["_report_path"] = str(path)
    payload["_report_name"] = path.name
    payload["_stamp"] = _parse_report_stamp(path)
    return _walk_forward_summary_from_payload(payload)


def _candidate_b_summary(reports_root: Path) -> dict[str, Any]:
    walk_forward_dir = _walk_forward_dir(reports_root)
    scenarios = (
        ("Strict Candidate B", CANDIDATE_B_STRICT_NAME, CANDIDATE_B_FALLBACKS[0]),
        ("Relaxed 25-trade", CANDIDATE_B_RELAXED_25_NAME, CANDIDATE_B_FALLBACKS[1]),
        ("Relaxed 62-trade", CANDIDATE_B_RELAXED_62_NAME, CANDIDATE_B_FALLBACKS[2]),
    )
    rows: list[dict[str, Any]] = []
    for label, filename, fallback in scenarios:
        path = walk_forward_dir / filename
        summary = _walk_forward_summary_from_path(path)
        if summary:
            rows.append(
                {
                    "config": label,
                    "trades": summary.get("accepted_trades") or fallback["trades"],
                    "cumulative_net_bps": summary.get("cumulative_net_bps") or fallback["cumulative_net_bps"],
                    "report_path": summary.get("report_path"),
                    "source": "report",
                }
            )
        else:
            rows.append(
                {
                    "config": label,
                    "trades": fallback["trades"],
                    "cumulative_net_bps": fallback["cumulative_net_bps"],
                    "report_path": None,
                    "source": "fallback",
                }
            )
    return {"scenarios": rows}


def _latest_accepted_trades_csv(
    reports_root: Path,
    candidate: dict[str, Any] | None,
) -> str | None:
    if candidate:
        csv_path = candidate.get("accepted_trades_csv_path")
        if csv_path and Path(csv_path).is_file():
            return str(csv_path)
    walk_forward_dir = _walk_forward_dir(reports_root)
    if not walk_forward_dir.is_dir():
        return None
    csv_files = sorted(
        walk_forward_dir.glob("accepted_trades_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return str(csv_files[0]) if csv_files else None


def _milestone_progress(
    phase3: dict[str, Any] | None,
    walk_forward: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    milestones: list[dict[str, Any]] = []
    imported_hours = _phase3_imported_hours(phase3)
    data_pipeline_pct = min(100, int(round((imported_hours / PHASE3_TARGET_HOURS) * 100))) if phase3 else 0

    for name, default_pct, default_status in MILESTONES:
        pct = default_pct
        status = default_status
        if name == "Data Pipeline Works" and phase3 is not None:
            pct = max(default_pct, data_pipeline_pct)
            status = f"{imported_hours:.1f}h imported / {PHASE3_TARGET_HOURS}h target."
        elif name == "Paper Backtest Engine" and walk_forward is not None:
            pct = max(default_pct, 85)
            status = f"Latest walk-forward: {walk_forward.get('_report_name', 'report available')}."
        elif name == "First Candidate Found" and candidate is not None:
            trades = candidate.get("filtered_paper_trades")
            bps = candidate.get("cumulative_net_return_bps")
            if trades is not None and bps is not None:
                status = f"Latest candidate: {trades} trades, {bps:+.2f} bps (paper-only)."
        elif name == "Paper Operations" and candidate is not None:
            pct = max(default_pct, 70)
            status = "Candidate validation reports available locally."
        milestones.append({"name": name, "progress_pct": pct, "status": status})
    return milestones


def _next_recommended_action(
    phase3: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    validation: dict[str, Any],
) -> str:
    if phase3 and phase3.get("next_recommended_action"):
        return str(phase3["next_recommended_action"])

    failed: list[str] = []
    for coin, intervals in validation.get("by_coin_interval", {}).items():
        for interval, details in intervals.items():
            if details.get("available") and details.get("passed") is False:
                failed.append(f"{coin} {interval}")

    if failed:
        return f"Review validation failures for: {', '.join(failed)}."

    if candidate is None:
        return "Run candidate validation after the next walk-forward evaluation."

    validation_status = candidate.get("validation_status") or {}
    if validation_status.get("insufficient_trades"):
        return "Candidate B has insufficient trades — continue data backfill and re-evaluate."

    if validation_status.get("unstable_concentrated") or validation_status.get("unstable_sparse"):
        return "Investigate stability warnings before scaling the candidate."

    return "Continue Phase 3 backfill and inspect latest walk-forward diagnostics."


def build_dashboard_summary(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    """Build a read-only dashboard summary from local artifacts."""
    phase3_path = _phase3_audit_path(reports_root)
    phase3 = _load_json(phase3_path)
    candidate = _latest_candidate_validation(reports_root)
    latest_walk_forward = _latest_walk_forward_report(reports_root)
    validation = _summarize_validation(data_root)

    summary: dict[str, Any] = {
        "generated_at": _utc_now_iso(),
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "real_execution_ready": False,
        "market_ready": False,
        "safety_statement": SAFETY_STATEMENT,
        "phase_backfill": _summarize_phase_backfill(phase3),
        "data_coverage": _summarize_data_coverage(phase3, data_root),
        "validation": validation,
        "latest_walk_forward": (
            _walk_forward_summary_from_payload(latest_walk_forward)
            if latest_walk_forward
            else {"available": False}
        ),
        "latest_walk_forward_by_coin": {
            coin: (
                _walk_forward_summary_from_payload(report)
                if (report := _latest_walk_forward_report(reports_root, coin=coin))
                else {"available": False}
            )
            for coin in RESEARCH_COINS
        },
        "candidate_b": _candidate_b_summary(reports_root),
        "latest_candidate_validation": (
            {
                "report_path": candidate.get("_report_path"),
                "candidate_name": candidate.get("candidate_name"),
                "filtered_paper_trades": candidate.get("filtered_paper_trades"),
                "cumulative_net_return_bps": candidate.get("cumulative_net_return_bps"),
                "validation_status": candidate.get("validation_status"),
                "ready_for_more_research": candidate.get("ready_for_more_research"),
            }
            if candidate
            else {"available": False}
        ),
        "latest_accepted_trades_csv_path": _latest_accepted_trades_csv(reports_root, candidate),
        "milestones": _milestone_progress(phase3, latest_walk_forward, candidate),
        "next_recommended_action": _next_recommended_action(phase3, candidate, validation),
        "source_paths": {
            "phase3_audit": str(phase3_path),
            "walk_forward_dir": str(_walk_forward_dir(reports_root)),
            "candidate_validation_dir": str(_candidate_validation_dir(reports_root)),
            "validation_reports_dir": str(data_root / "reports"),
            "historical_1s_dir": str(data_root / "historical_1s"),
            "historical_15s_dir": str(data_root / "historical_15s"),
        },
    }
    if phase3 is not None:
        summary["phase_backfill"]["audit_path"] = str(phase3_path)
    return summary


def write_dashboard_json(
    summary: dict[str, Any],
    output_dir: Path,
    json_name: str = DEFAULT_JSON_NAME,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / json_name
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    summary["json_report_path"] = str(output_path)
    return output_path


def write_dashboard_markdown(
    summary: dict[str, Any],
    output_dir: Path,
    markdown_name: str = DEFAULT_MARKDOWN_NAME,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / markdown_name

    lines = [
        "# Bid/Ask Dashboard Summary",
        "",
        f"- generated_at: {summary.get('generated_at')}",
        f"- paper_only: {summary.get('paper_only')}",
        f"- execution_enabled: {summary.get('execution_enabled')}",
        f"- real_execution_ready: {summary.get('real_execution_ready')}",
        f"- market_ready: {summary.get('market_ready')}",
        "",
        "## Safety",
        "",
        str(summary.get("safety_statement", "")),
        "",
        "## Phase Backfill",
        "",
    ]

    phase = summary.get("phase_backfill") or {}
    if phase.get("available"):
        lines.extend(
            [
                f"- imported_hours_mean: {phase.get('imported_hours_mean')}",
                f"- target_hours: {phase.get('target_hours')}",
                f"- progress_fraction: {phase.get('progress_fraction')}",
                f"- downloads_total_files: {phase.get('downloads_total_files')}",
                f"- scheduler_latest_action: {phase.get('scheduler_latest_action')}",
            ]
        )
    else:
        lines.append("- Phase 3 audit report not available.")

    lines.extend(["", "## Data Coverage", ""])
    for coin_row in (summary.get("data_coverage") or {}).get("coins") or []:
        lines.append(
            f"- {coin_row.get('coin')}: 1s rows={coin_row.get('rows_1s')} "
            f"15s rows={coin_row.get('rows_15s')} "
            f"latest={coin_row.get('last_timestamp') or 'n/a'}"
        )

    lines.extend(["", "## Validation Summary", ""])
    by_coin = (summary.get("validation") or {}).get("by_coin_interval") or {}
    for coin, intervals in by_coin.items():
        lines.append(f"### {coin}")
        for interval, details in intervals.items():
            if not details.get("available"):
                lines.append(f"- {interval}: not available")
                continue
            lines.append(
                f"- {interval}: passed={details.get('passed')} "
                f"rows={details.get('row_count')} "
                f"missing_periods={details.get('missing_periods_count')}"
            )
        lines.append("")

    lines.extend(["## Latest Walk-Forward", ""])
    latest_wf = summary.get("latest_walk_forward") or {}
    if latest_wf.get("report_path"):
        lines.extend(
            [
                f"- report: {latest_wf.get('report_path')}",
                f"- coin: {latest_wf.get('coin')}",
                f"- cumulative_net_bps: {latest_wf.get('cumulative_net_bps')}",
                f"- accepted_trades: {latest_wf.get('accepted_trades')}",
            ]
        )
    else:
        lines.append("- No walk-forward report found.")

    lines.extend(["", "## Candidate B", ""])
    for row in (summary.get("candidate_b") or {}).get("scenarios") or []:
        lines.append(
            f"- {row.get('config')}: trades={row.get('trades')} "
            f"net_bps={row.get('cumulative_net_bps')} source={row.get('source')}"
        )

    accepted_csv = summary.get("latest_accepted_trades_csv_path")
    lines.extend(["", "## Latest Accepted Trades CSV", ""])
    lines.append(f"- {accepted_csv or 'not available'}")

    lines.extend(["", "## Milestones", ""])
    for milestone in summary.get("milestones") or []:
        lines.append(
            f"- {milestone.get('name')}: {milestone.get('progress_pct')}% — {milestone.get('status')}"
        )

    lines.extend(["", "## Next Recommended Action", ""])
    lines.append(str(summary.get("next_recommended_action", "")))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["markdown_report_path"] = str(output_path)
    return output_path


def refresh_dashboard_artifacts(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    json_name: str = DEFAULT_JSON_NAME,
    markdown_name: str = DEFAULT_MARKDOWN_NAME,
) -> dict[str, Any]:
    summary = build_dashboard_summary(reports_root=reports_root, data_root=data_root)
    json_path = write_dashboard_json(summary, output_dir, json_name=json_name)
    md_path = write_dashboard_markdown(summary, output_dir, markdown_name=markdown_name)
    summary["output_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh read-only bid/ask dashboard summary artifacts from local "
            "JSON/CSV reports (paper-only, research-only)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    refresh = subparsers.add_parser("refresh", help="Rebuild dashboard summary JSON and Markdown")
    refresh.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    refresh.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    refresh.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    refresh.add_argument("--json-name", default=DEFAULT_JSON_NAME)
    refresh.add_argument("--markdown-name", default=DEFAULT_MARKDOWN_NAME)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "refresh":
        summary = refresh_dashboard_artifacts(
            reports_root=args.reports_root,
            data_root=args.data_root,
            output_dir=args.output_dir,
            json_name=args.json_name,
            markdown_name=args.markdown_name,
        )
        print(json.dumps(summary["output_paths"], indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
