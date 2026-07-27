"""Standalone read-only stale result detector for bid/ask research artifacts.

Paper-only and research-only. Reads local JSON/CSV reports and writes audit output
under reports/stale_result_detector/. No execution, orders, wallets, AWS, network,
or shell commands.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT, configure_csv_field_size_limit
from bid_ask_validator import bid_ask_csv_path

DEFAULT_REPORTS_ROOT = Path("reports")
DEFAULT_DATA_ROOT = Path("data/bid_ask_csv")
DEFAULT_OUTPUT_DIR = Path("reports/stale_result_detector")
DEFAULT_MAX_ARTIFACT_AGE_HOURS = 6.0
DEFAULT_CANDIDATE_REPEAT_THRESHOLD = 3
DEFAULT_JSON_NAME = "stale_result_latest.json"
DEFAULT_MARKDOWN_NAME = "stale_result_latest.md"

RESEARCH_COINS = ("BTC", "ETH")
VALIDATION_INTERVALS = ("1s", "15s")
HISTORICAL_SOURCE = "historical"
AUDIT_SNAPSHOT_LIMIT = 3

DASHBOARD_ARTIFACT_REL = Path("dashboard_artifacts/dashboard_summary_latest.json")
PHASE3_AUDIT_REL = Path("bid_ask_research_scheduler/phase3_audit_latest.json")
WALK_FORWARD_REL = Path("bid_ask_walk_forward")
CANDIDATE_VALIDATION_REL = Path("bid_ask_candidate_validation")
SCHEDULER_RUNS_REL = Path("bid_ask_research_scheduler/runs")

STATUS_HEALTHY = "healthy"
STATUS_WARNING = "warning"
STATUS_CRITICAL = "critical"

SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"


@dataclass
class StaleItem:
    severity: str
    code: str
    message: str
    coin: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.coin is not None:
            payload["coin"] = self.coin
        if self.details:
            payload["details"] = self.details
        return payload


def _utc_now(reference_time: datetime | None = None) -> datetime:
    if reference_time is not None:
        return reference_time.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _utc_now_iso(reference_time: datetime | None = None) -> str:
    return _utc_now(reference_time).isoformat().replace("+00:00", "Z")


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


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _artifact_age_hours(
    generated_at: str | None,
    *,
    reference_time: datetime | None = None,
) -> float | None:
    parsed = _parse_iso_timestamp(generated_at)
    if parsed is None:
        return None
    delta = _utc_now(reference_time) - parsed
    return max(delta.total_seconds(), 0.0) / 3600.0


def _parse_report_stamp(path: Path) -> str:
    stem = path.stem
    if stem.startswith("walk_forward_"):
        return stem.rsplit("_", 1)[-1]
    if "_validation_" in stem:
        return stem.rsplit("_validation_", 1)[-1]
    if stem.startswith("candidate_validation_"):
        return stem.rsplit("_", 1)[-1]
    if stem.startswith("bid_ask_scheduler_"):
        return stem.rsplit("_", 1)[-1]
    return stem


def _csv_timestamp_bounds(path: Path, *, interval: str) -> tuple[int | None, int | None]:
    if not path.is_file():
        return None, None
    column = "timestamp_ms" if interval == "1s" else "start_timestamp_ms"
    first_ts: int | None = None
    last_ts: int | None = None
    try:
        configure_csv_field_size_limit()
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


def _candidate_status_label(report: dict[str, Any]) -> str:
    status = report.get("validation_status") or {}
    if status.get("paper_candidate_pass"):
        return "paper_candidate_pass"
    return "not_market_ready"


def load_candidate_validation_reports(reports_dir: Path) -> list[dict[str, Any]]:
    if not reports_dir.is_dir():
        return []

    reports: list[dict[str, Any]] = []
    for path in sorted(reports_dir.glob("candidate_validation_*.json")):
        payload = _load_json(path)
        if payload is None:
            continue
        payload = dict(payload)
        payload.setdefault("json_report_path", str(path))
        reports.append(payload)

    reports.sort(key=lambda item: str(item.get("run_timestamp") or ""))
    return reports


def load_walk_forward_reports(reports_dir: Path, *, coin: str | None = None) -> list[dict[str, Any]]:
    if not reports_dir.is_dir():
        return []

    pattern = f"walk_forward_{coin}_*.json" if coin else "walk_forward_*.json"
    reports: list[dict[str, Any]] = []
    for path in sorted(reports_dir.glob(pattern), key=_parse_report_stamp):
        payload = _load_json(path)
        if payload is None:
            continue
        entry = dict(payload)
        entry["json_report_path"] = str(path)
        entry["_stamp"] = _parse_report_stamp(path)
        reports.append(entry)
    return reports


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
            entry = dict(payload)
            entry["_report_path"] = str(path)
            return entry
    return None


def _detect_repeated_unchanged_reports(
    reports: list[dict[str, Any]],
    *,
    signature_fn,
    limit: int,
) -> bool:
    if len(reports) < limit:
        return False
    recent = reports[-limit:]
    signatures = {signature_fn(item) for item in recent}
    return len(signatures) == 1


def _candidate_signature(report: dict[str, Any]) -> tuple[Any, ...]:
    return (
        report.get("filtered_paper_trades"),
        float(report.get("cumulative_net_return_bps", 0.0)),
        _candidate_status_label(report),
    )


def _walk_forward_signature(report: dict[str, Any]) -> tuple[Any, ...]:
    trades = report.get("filtered_paper_trades")
    if trades is None:
        trades = report.get("accepted_trades_count")
    return (
        trades,
        float(report.get("cumulative_net_return_bps", 0.0)),
        report.get("completed_windows"),
    )


def _walk_forward_trade_count(report: dict[str, Any]) -> int:
    trades = report.get("filtered_paper_trades")
    if trades is None:
        trades = report.get("accepted_trades_count")
    return int(trades or 0)


def _collect_historical_1s_audit_snapshots(
    reports_root: Path,
    phase3: dict[str, Any] | None,
    *,
    limit: int = AUDIT_SNAPSHOT_LIMIT,
) -> dict[str, list[dict[str, Any]]]:
    snapshots: dict[str, list[dict[str, Any]]] = {coin: [] for coin in RESEARCH_COINS}

    if phase3 is not None:
        imported = phase3.get("imported_historical_1s") or {}
        for coin in RESEARCH_COINS:
            entry = imported.get(coin) or {}
            last_ms = entry.get("last_timestamp_ms")
            if last_ms is not None:
                snapshots[coin].append(
                    {
                        "source": "phase3_audit_latest",
                        "generated_at": phase3.get("generated_at"),
                        "last_timestamp_ms": int(last_ms),
                    }
                )

    runs_dir = reports_root / SCHEDULER_RUNS_REL
    if runs_dir.is_dir():
        run_paths = sorted(runs_dir.glob("bid_ask_scheduler_*.json"), key=_parse_report_stamp, reverse=True)
        for path in run_paths[:limit]:
            payload = _load_json(path)
            if payload is None:
                continue
            coverage = payload.get("post_import_coverage") or payload.get("pre_import_coverage") or {}
            for coin in RESEARCH_COINS:
                entry = coverage.get(coin) or {}
                last_ms = entry.get("last_timestamp_ms")
                if last_ms is None:
                    continue
                snapshots[coin].append(
                    {
                        "source": path.name,
                        "generated_at": payload.get("generated_at"),
                        "last_timestamp_ms": int(last_ms),
                    }
                )

    return snapshots


def _check_stale_artifact(
    *,
    code: str,
    label: str,
    path: Path,
    generated_at: str | None,
    max_age_hours: float,
    reference_time: datetime | None,
) -> StaleItem | None:
    if generated_at is None:
        return StaleItem(
            severity=SEVERITY_WARNING,
            code=code,
            message=f"{label} artifact missing generated_at at {path}",
            details={"path": str(path)},
        )

    age_hours = _artifact_age_hours(generated_at, reference_time=reference_time)
    if age_hours is None:
        return StaleItem(
            severity=SEVERITY_WARNING,
            code=code,
            message=f"{label} artifact has an unreadable generated_at at {path}",
            details={"path": str(path), "generated_at": generated_at},
        )

    if age_hours > max_age_hours:
        return StaleItem(
            severity=SEVERITY_WARNING,
            code=code,
            message=(
                f"{label} artifact is {age_hours:.1f}h old "
                f"(threshold {max_age_hours}h)"
            ),
            details={
                "path": str(path),
                "generated_at": generated_at,
                "age_hours": round(age_hours, 3),
                "max_age_hours": max_age_hours,
            },
        )
    return None


def build_safety_flags() -> dict[str, Any]:
    return {
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "real_execution_ready": False,
        "read_only": True,
        "no_repairs_performed": True,
        "no_network_calls": True,
        "no_shell_commands": True,
        "research_only": True,
        "safety_statement": SAFETY_STATEMENT,
    }


def _build_recommended_commands(stale_items: list[StaleItem]) -> list[str]:
    codes = {item.code for item in stale_items}
    commands: list[str] = []

    if "stale_dashboard_artifact" in codes:
        commands.append("python3 bid_ask_dashboard_artifact_refresher.py refresh")

    if "stale_phase3_audit" in codes:
        commands.append("python3 bid_ask_research_scheduler.py phase-backfill-audit")

    if "candidate_result_unchanged" in codes:
        commands.append("python3 bid_ask_candidate_validator.py run")

    if "walk_forward_result_unchanged" in codes:
        commands.append(
            "python3 bid_ask_walk_forward_evaluator.py --coin BTC --interval 15s --source historical"
        )

    if "missing_accepted_trades_csv" in codes:
        commands.append(
            "python3 bid_ask_walk_forward_evaluator.py --coin BTC --interval 15s --source historical"
        )

    if "historical_1s_unchanged_across_audits" in codes:
        commands.append("python3 bid_ask_research_scheduler.py autopilot-tick")

    if "stale_15s_while_1s_advanced" in codes:
        commands.append(
            "python3 bid_ask_resampler.py --coins BTC ETH --interval 15s --source historical"
        )

    if "missing_validation_report" in codes:
        commands.append(
            "python3 bid_ask_research_orchestrator.py resample-validate "
            "--coins BTC ETH --source historical"
        )

    if "missing_candidate_report" in codes:
        commands.append("python3 bid_ask_candidate_validator.py run")

    if not commands:
        commands.append("python3 bid_ask_stale_result_detector.py check")

    deduped: list[str] = []
    for command in commands:
        if command not in deduped:
            deduped.append(command)
    return deduped


def _aggregate_status(stale_items: list[StaleItem]) -> str:
    if any(item.severity == SEVERITY_CRITICAL for item in stale_items):
        return STATUS_CRITICAL
    if stale_items:
        return STATUS_WARNING
    return STATUS_HEALTHY


def build_stale_result_audit(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    max_artifact_age_hours: float = DEFAULT_MAX_ARTIFACT_AGE_HOURS,
    candidate_repeat_threshold: int = DEFAULT_CANDIDATE_REPEAT_THRESHOLD,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    """Run read-only stale result checks against local research artifacts."""
    stale_items: list[StaleItem] = []

    dashboard_path = reports_root / DASHBOARD_ARTIFACT_REL
    dashboard = _load_json(dashboard_path)
    if dashboard is None:
        stale_items.append(
            StaleItem(
                severity=SEVERITY_WARNING,
                code="stale_dashboard_artifact",
                message=f"Dashboard summary artifact missing at {dashboard_path}",
                details={"path": str(dashboard_path)},
            )
        )
    else:
        dashboard_issue = _check_stale_artifact(
            code="stale_dashboard_artifact",
            label="Dashboard summary",
            path=dashboard_path,
            generated_at=dashboard.get("generated_at"),
            max_age_hours=max_artifact_age_hours,
            reference_time=reference_time,
        )
        if dashboard_issue is not None:
            stale_items.append(dashboard_issue)

    phase3_path = reports_root / PHASE3_AUDIT_REL
    phase3 = _load_json(phase3_path)
    if phase3 is None:
        stale_items.append(
            StaleItem(
                severity=SEVERITY_WARNING,
                code="stale_phase3_audit",
                message=f"Phase 3 audit artifact missing at {phase3_path}",
                details={"path": str(phase3_path)},
            )
        )
    else:
        phase3_issue = _check_stale_artifact(
            code="stale_phase3_audit",
            label="Phase 3 audit",
            path=phase3_path,
            generated_at=phase3.get("generated_at"),
            max_age_hours=max_artifact_age_hours,
            reference_time=reference_time,
        )
        if phase3_issue is not None:
            stale_items.append(phase3_issue)

    candidate_dir = reports_root / CANDIDATE_VALIDATION_REL
    candidate_reports = load_candidate_validation_reports(candidate_dir)
    if not candidate_reports:
        stale_items.append(
            StaleItem(
                severity=SEVERITY_CRITICAL,
                code="missing_candidate_report",
                message="Latest candidate validation report is missing",
                details={"reports_dir": str(candidate_dir)},
            )
        )
    elif _detect_repeated_unchanged_reports(
        candidate_reports,
        signature_fn=_candidate_signature,
        limit=candidate_repeat_threshold,
    ):
        stale_items.append(
            StaleItem(
                severity=SEVERITY_WARNING,
                code="candidate_result_unchanged",
                message=(
                    f"Candidate validation results unchanged across the last "
                    f"{candidate_repeat_threshold} reports"
                ),
                details={
                    "reports_checked": candidate_repeat_threshold,
                    "latest_report_path": candidate_reports[-1].get("json_report_path"),
                },
            )
        )

    walk_forward_dir = reports_root / WALK_FORWARD_REL
    walk_forward_reports = load_walk_forward_reports(walk_forward_dir)
    if walk_forward_reports and _detect_repeated_unchanged_reports(
        walk_forward_reports,
        signature_fn=_walk_forward_signature,
        limit=candidate_repeat_threshold,
    ):
        stale_items.append(
            StaleItem(
                severity=SEVERITY_WARNING,
                code="walk_forward_result_unchanged",
                message=(
                    f"Walk-forward results unchanged across the last "
                    f"{candidate_repeat_threshold} reports"
                ),
                details={
                    "reports_checked": candidate_repeat_threshold,
                    "latest_report_path": walk_forward_reports[-1].get("json_report_path"),
                },
            )
        )

    if walk_forward_reports:
        latest_walk_forward = walk_forward_reports[-1]
        trade_count = _walk_forward_trade_count(latest_walk_forward)
        csv_path_value = latest_walk_forward.get("accepted_trades_csv_path")
        csv_path = Path(str(csv_path_value)) if csv_path_value else None
        if trade_count > 0 and (csv_path is None or not csv_path.is_file()):
            stale_items.append(
                StaleItem(
                    severity=SEVERITY_CRITICAL,
                    code="missing_accepted_trades_csv",
                    message=(
                        "Latest walk-forward report has paper trades but no accepted "
                        "trades CSV path on disk"
                    ),
                    details={
                        "report_path": latest_walk_forward.get("json_report_path"),
                        "accepted_trades_csv_path": csv_path_value,
                        "filtered_paper_trades": trade_count,
                    },
                )
            )

    audit_snapshots = _collect_historical_1s_audit_snapshots(reports_root, phase3)
    for coin in RESEARCH_COINS:
        coin_snapshots = audit_snapshots.get(coin) or []
        if len(coin_snapshots) < 2:
            continue
        timestamps = {entry["last_timestamp_ms"] for entry in coin_snapshots}
        if len(timestamps) == 1:
            stale_items.append(
                StaleItem(
                    severity=SEVERITY_WARNING,
                    code="historical_1s_unchanged_across_audits",
                    message=(
                        f"{coin} historical_1s latest timestamp unchanged across "
                        f"{len(coin_snapshots)} recent audits"
                    ),
                    coin=coin,
                    details={
                        "last_timestamp_ms": next(iter(timestamps)),
                        "snapshots": coin_snapshots,
                    },
                )
            )

    imported_1s = (phase3 or {}).get("imported_historical_1s") or {}
    resampled = (phase3 or {}).get("resampled_coverage") or {}
    for coin in RESEARCH_COINS:
        one_s = imported_1s.get(coin) or {}
        freshness = one_s.get("freshness_status")
        one_s_end = one_s.get("last_timestamp_ms")
        if one_s_end is None:
            path_1s = bid_ask_csv_path(coin, "1s", data_root, source=HISTORICAL_SOURCE)
            _, one_s_end = _csv_timestamp_bounds(path_1s, interval="1s")

        fifteen = (resampled.get(coin) or {}).get("15s") or {}
        fifteen_end = fifteen.get("last_timestamp_ms")
        if fifteen_end is None:
            path_15s = bid_ask_csv_path(coin, "15s", data_root, source=HISTORICAL_SOURCE)
            _, fifteen_end = _csv_timestamp_bounds(path_15s, interval="15s")

        if freshness != "advancing":
            continue
        if one_s_end is None or fifteen_end is None:
            continue
        if int(fifteen_end) < int(one_s_end):
            stale_items.append(
                StaleItem(
                    severity=SEVERITY_WARNING,
                    code="stale_15s_while_1s_advanced",
                    message=(
                        f"{coin} 15s data ends at {fifteen_end} while 1s data ends at {one_s_end}"
                    ),
                    coin=coin,
                    details={
                        "one_s_last_timestamp_ms": int(one_s_end),
                        "fifteen_s_last_timestamp_ms": int(fifteen_end),
                    },
                )
            )

    missing_validation: list[str] = []
    for coin in RESEARCH_COINS:
        for interval in VALIDATION_INTERVALS:
            report = _latest_validation_report(coin, interval, data_root)
            if report is None:
                missing_validation.append(f"{coin} {interval}")

    if missing_validation:
        stale_items.append(
            StaleItem(
                severity=SEVERITY_CRITICAL,
                code="missing_validation_report",
                message="Latest validation report missing for: " + ", ".join(missing_validation),
                details={"missing": missing_validation},
            )
        )

    status = _aggregate_status(stale_items)
    recommended_commands = _build_recommended_commands(stale_items)

    return {
        "generated_at": _utc_now_iso(reference_time),
        "command": "check",
        "status": status,
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "real_execution_ready": False,
        "safety_statement": SAFETY_STATEMENT,
        "reports_root": str(reports_root),
        "data_root": str(data_root),
        "max_artifact_age_hours": max_artifact_age_hours,
        "candidate_repeat_threshold": candidate_repeat_threshold,
        "stale_items": [item.to_dict() for item in stale_items],
        "recommended_commands": recommended_commands,
        "safety_flags": build_safety_flags(),
        "artifact_paths": {
            "dashboard_summary": str(dashboard_path),
            "phase3_audit": str(phase3_path),
            "walk_forward_dir": str(walk_forward_dir),
            "candidate_validation_dir": str(candidate_dir),
            "validation_reports_dir": str(data_root / "reports"),
        },
        "counts": {
            "candidate_reports": len(candidate_reports),
            "walk_forward_reports": len(walk_forward_reports),
            "stale_item_count": len(stale_items),
        },
    }


def write_stale_result_json(
    report: dict[str, Any],
    output_dir: Path,
    json_name: str = DEFAULT_JSON_NAME,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / json_name
    output_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    report["json_report_path"] = str(output_path)
    return output_path


def write_stale_result_markdown(
    report: dict[str, Any],
    output_dir: Path,
    markdown_name: str = DEFAULT_MARKDOWN_NAME,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / markdown_name

    lines = [
        "# Bid/Ask Stale Result Detector",
        "",
        f"- generated_at: {report.get('generated_at')}",
        f"- status: {report.get('status')}",
        f"- paper_only: {report.get('paper_only')}",
        f"- execution_enabled: {report.get('execution_enabled')}",
        "",
        "## Safety",
        "",
        str(report.get("safety_statement", "")),
        "",
        "## Stale Items",
        "",
    ]

    stale_items = report.get("stale_items") or []
    if not stale_items:
        lines.append("- None")
    else:
        for item in stale_items:
            coin_suffix = f" ({item['coin']})" if item.get("coin") else ""
            lines.append(
                f"- [{item.get('severity')}] {item.get('code')}{coin_suffix}: {item.get('message')}"
            )

    lines.extend(["", "## Recommended Commands", ""])
    for command in report.get("recommended_commands") or []:
        lines.append(f"- `{command}`")

    lines.extend(["", "## Safety Flags", ""])
    for key, value in (report.get("safety_flags") or {}).items():
        lines.append(f"- {key}: {value}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["markdown_report_path"] = str(output_path)
    return output_path


def run_stale_result_check(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_artifact_age_hours: float = DEFAULT_MAX_ARTIFACT_AGE_HOURS,
    candidate_repeat_threshold: int = DEFAULT_CANDIDATE_REPEAT_THRESHOLD,
    json_name: str = DEFAULT_JSON_NAME,
    markdown_name: str = DEFAULT_MARKDOWN_NAME,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    report = build_stale_result_audit(
        reports_root=reports_root,
        data_root=data_root,
        max_artifact_age_hours=max_artifact_age_hours,
        candidate_repeat_threshold=candidate_repeat_threshold,
        reference_time=reference_time,
    )
    json_path = write_stale_result_json(report, output_dir, json_name=json_name)
    md_path = write_stale_result_markdown(report, output_dir, markdown_name=markdown_name)
    report["output_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect stale bid/ask research results and dashboard artifacts "
            "(paper-only, research-only, read-only)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Run stale result checks and write reports")
    check.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    check.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    check.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    check.add_argument(
        "--max-artifact-age-hours",
        type=float,
        default=DEFAULT_MAX_ARTIFACT_AGE_HOURS,
    )
    check.add_argument(
        "--candidate-repeat-threshold",
        type=int,
        default=DEFAULT_CANDIDATE_REPEAT_THRESHOLD,
    )
    check.add_argument("--json-name", default=DEFAULT_JSON_NAME)
    check.add_argument("--markdown-name", default=DEFAULT_MARKDOWN_NAME)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        report = run_stale_result_check(
            reports_root=args.reports_root,
            data_root=args.data_root,
            output_dir=args.output_dir,
            max_artifact_age_hours=args.max_artifact_age_hours,
            candidate_repeat_threshold=args.candidate_repeat_threshold,
            json_name=args.json_name,
            markdown_name=args.markdown_name,
        )
        print(json.dumps(report["output_paths"], indent=2))
        if report["status"] == STATUS_CRITICAL:
            return 1
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
