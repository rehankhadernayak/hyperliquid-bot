"""Standalone daily research progress digest for bid/ask pipeline work.

Paper-only and research-only. Reads local JSON reports and CSV metadata only.
Writes digest files under reports/daily_progress/. No orders, execution endpoints,
wallets, keys, AWS calls, downloads, shell commands, or network access.
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
DEFAULT_OUTPUT_DIR = Path("reports/daily_progress")
DEFAULT_LATEST_MD_NAME = "daily_progress_latest.md"
DEFAULT_LATEST_JSON_NAME = "daily_progress_latest.json"

RESEARCH_COINS = ("BTC", "ETH")
INTERVALS = ("1s", "15s", "1m")
HISTORICAL_SOURCE = "historical"
LOCAL_SOURCE = "local"

PHASE3_AUDIT_REL = Path("bid_ask_research_scheduler/phase3_audit_latest.json")
CANDIDATE_VALIDATION_REL = Path("bid_ask_candidate_validation")
SCHEDULER_RUNS_REL = Path("bid_ask_research_scheduler/runs")

RELAXED_NAME_MARKER = "RELAXED"

RESEARCH_ONLY_NOTE = (
    "Research-only summary. Metrics do not prove profitability. "
    "Real trading remains disabled."
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_date_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


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


def _parse_report_stamp(path: Path) -> str:
    stem = path.stem
    for marker in ("_validation_", "_resample_"):
        if marker in stem:
            return stem.rsplit(marker, 1)[-1]
    if stem.startswith("candidate_validation_"):
        return stem.rsplit("_", 1)[-1]
    if stem.startswith("bid_ask_scheduler_"):
        parts = stem.split("_")
        if parts:
            return parts[-1]
    return stem


def _stamp_matches_date(stamp: str, date_yyyymmdd: str) -> bool:
    return stamp.startswith(date_yyyymmdd)


def _generated_at_matches_date(generated_at: str | None, date_yyyymmdd: str) -> bool:
    if not generated_at:
        return False
    normalized = generated_at.replace("-", "").replace(":", "")
    return normalized.startswith(date_yyyymmdd)


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


def _candidate_validation_dir(reports_root: Path) -> Path:
    return reports_root / CANDIDATE_VALIDATION_REL


def _reports_changed_today(
    *,
    data_root: Path,
    reports_root: Path,
    date_yyyymmdd: str,
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []

    validation_dir = data_root / "reports"
    if validation_dir.is_dir():
        for path in sorted(validation_dir.glob("*_validation_*.json")):
            stamp = _parse_report_stamp(path)
            if not _stamp_matches_date(stamp, date_yyyymmdd):
                continue
            payload = _load_json(path) or {}
            changes.append(
                {
                    "kind": "validation",
                    "path": str(path),
                    "stamp": stamp,
                    "coin": payload.get("coin"),
                    "interval": payload.get("interval"),
                    "passed": payload.get("passed"),
                    "row_count": payload.get("row_count"),
                }
            )

        for path in sorted(validation_dir.glob("*_resample_*.json")):
            stamp = _parse_report_stamp(path)
            if not _stamp_matches_date(stamp, date_yyyymmdd):
                continue
            payload = _load_json(path) or {}
            changes.append(
                {
                    "kind": "resample",
                    "path": str(path),
                    "stamp": stamp,
                    "coin": payload.get("coin"),
                    "interval": payload.get("interval"),
                    "input_rows": payload.get("input_rows"),
                    "output_rows": payload.get("output_rows"),
                }
            )

    scheduler_runs = reports_root / SCHEDULER_RUNS_REL
    if scheduler_runs.is_dir():
        for path in sorted(scheduler_runs.glob("bid_ask_scheduler_*.json")):
            stamp = _parse_report_stamp(path)
            if not _stamp_matches_date(stamp, date_yyyymmdd):
                payload = _load_json(path) or {}
                if not _generated_at_matches_date(payload.get("generated_at"), date_yyyymmdd):
                    continue
            payload = _load_json(path) or {}
            changes.append(
                {
                    "kind": "scheduler_run",
                    "path": str(path),
                    "stamp": stamp,
                    "workflow": payload.get("workflow") or payload.get("command"),
                    "status": payload.get("status") or payload.get("result"),
                }
            )

    candidate_dir = _candidate_validation_dir(reports_root)
    if candidate_dir.is_dir():
        for path in sorted(candidate_dir.glob("candidate_validation_*.json")):
            stamp = _parse_report_stamp(path)
            if not _stamp_matches_date(stamp, date_yyyymmdd):
                payload = _load_json(path) or {}
                if not _generated_at_matches_date(payload.get("run_timestamp"), date_yyyymmdd):
                    continue
            payload = _load_json(path) or {}
            changes.append(
                {
                    "kind": "candidate_validation",
                    "path": str(path),
                    "stamp": stamp,
                    "candidate_name": payload.get("candidate_name"),
                    "filtered_paper_trades": payload.get("filtered_paper_trades"),
                    "cumulative_net_return_bps": payload.get("cumulative_net_return_bps"),
                }
            )

    return changes


def _describe_changes(changes: list[dict[str, Any]]) -> list[str]:
    if not changes:
        return ["No new validation, resample, scheduler, or candidate reports recorded for this date."]

    lines: list[str] = []
    for change in changes:
        kind = change.get("kind")
        if kind == "validation":
            passed = change.get("passed")
            status = "passed" if passed else "did not pass"
            lines.append(
                f"Validated {change.get('coin')} {change.get('interval')} data "
                f"({change.get('row_count')} rows, {status})."
            )
        elif kind == "resample":
            lines.append(
                f"Resampled {change.get('coin')} to {change.get('interval')} "
                f"({change.get('input_rows')} input rows -> {change.get('output_rows')} output rows)."
            )
        elif kind == "scheduler_run":
            workflow = change.get("workflow") or "scheduler"
            status = change.get("status") or "completed"
            lines.append(f"Scheduler run ({workflow}) reported status: {status}.")
        elif kind == "candidate_validation":
            trades = change.get("filtered_paper_trades")
            bps = change.get("cumulative_net_return_bps")
            name = change.get("candidate_name") or "candidate"
            lines.append(
                f"Candidate validation updated for {name}: "
                f"{trades} paper trades, {bps} cumulative net bps (research-only)."
            )
    return lines


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
            entry["_stamp"] = _parse_report_stamp(path)
            return entry
    return None


def _collect_validation_status(data_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    by_coin: dict[str, dict[str, dict[str, Any]]] = {}
    for coin in RESEARCH_COINS:
        by_interval: dict[str, dict[str, Any]] = {}
        for interval in INTERVALS:
            report = _latest_validation_report(coin, interval, data_root)
            if report is None:
                by_interval[interval] = {
                    "available": False,
                    "passed": None,
                    "row_count": None,
                    "warnings": [],
                    "report_path": None,
                }
                continue
            by_interval[interval] = {
                "available": True,
                "passed": report.get("passed"),
                "row_count": report.get("row_count"),
                "missing_periods_count": report.get("missing_periods_count"),
                "warnings": list(report.get("warnings") or []),
                "report_path": report.get("_report_path"),
                "stamp": report.get("_stamp"),
            }
        by_coin[coin] = by_interval
    return by_coin


def _candidate_summary_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    validation_status = payload.get("validation_status") or {}
    return {
        "available": True,
        "candidate_name": payload.get("candidate_name"),
        "report_path": payload.get("_report_path"),
        "filtered_paper_trades": payload.get("filtered_paper_trades"),
        "cumulative_net_return_bps": payload.get("cumulative_net_return_bps"),
        "average_net_return_bps": payload.get("average_net_return_bps"),
        "win_rate_after_costs": payload.get("win_rate_after_costs"),
        "positive_after_costs": payload.get("positive_after_costs"),
        "paper_candidate_pass": validation_status.get("paper_candidate_pass"),
        "insufficient_trades": validation_status.get("insufficient_trades"),
        "not_market_ready": validation_status.get("not_market_ready"),
        "stability_warnings": list(payload.get("stability_warnings") or []),
        "run_timestamp": payload.get("run_timestamp"),
    }


def _latest_candidate_validation(
    reports_root: Path,
    *,
    relaxed: bool,
) -> dict[str, Any]:
    candidate_dir = _candidate_validation_dir(reports_root)
    if not candidate_dir.is_dir():
        return {"available": False}

    candidates = sorted(
        candidate_dir.glob("candidate_validation_*.json"),
        key=_parse_report_stamp,
        reverse=True,
    )
    for path in candidates:
        payload = _load_json(path)
        if payload is None:
            continue
        name = str(payload.get("candidate_name") or path.name).upper()
        is_relaxed = RELAXED_NAME_MARKER in name
        if relaxed != is_relaxed:
            continue
        entry = dict(payload)
        entry["_report_path"] = str(path)
        return _candidate_summary_from_payload(entry)

    return {"available": False}


def _rows_by_coin_interval(
    data_root: Path,
    phase3: dict[str, Any] | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    imported = (phase3 or {}).get("imported_historical_1s") or {}
    resampled = (phase3 or {}).get("resampled_coverage") or {}
    rows: dict[str, dict[str, dict[str, Any]]] = {}

    for coin in RESEARCH_COINS:
        coin_rows: dict[str, dict[str, Any]] = {}
        coin_import = imported.get(coin) or {}
        coin_resampled = resampled.get(coin) or {}

        for interval in INTERVALS:
            historical_path = bid_ask_csv_path(coin, interval, data_root, source=HISTORICAL_SOURCE)
            local_path = bid_ask_csv_path(coin, interval, data_root, source=LOCAL_SOURCE)

            if interval == "1s":
                row_count = coin_import.get("row_count")
                if row_count is None:
                    row_count = _count_csv_rows(historical_path) or _count_csv_rows(local_path)
                last_ts = coin_import.get("last_timestamp_ms")
                if last_ts is None:
                    _, csv_last = _csv_timestamp_bounds(historical_path, interval=interval)
                    if csv_last is None:
                        _, csv_last = _csv_timestamp_bounds(local_path, interval=interval)
                    last_ts = csv_last
            else:
                interval_data = coin_resampled.get(interval) or {}
                row_count = interval_data.get("row_count")
                if row_count is None:
                    row_count = _count_csv_rows(historical_path) or _count_csv_rows(local_path)
                last_ts = interval_data.get("last_timestamp_ms")
                if last_ts is None:
                    _, csv_last = _csv_timestamp_bounds(historical_path, interval=interval)
                    if csv_last is None:
                        _, csv_last = _csv_timestamp_bounds(local_path, interval=interval)
                    last_ts = csv_last

            coin_rows[interval] = {
                "row_count": row_count,
                "last_timestamp_ms": last_ts,
                "last_timestamp": _fmt_ts_ms(last_ts),
                "historical_path": str(historical_path),
                "local_path": str(local_path),
            }
        rows[coin] = coin_rows
    return rows


def _backfill_progress(phase3: dict[str, Any] | None) -> dict[str, Any]:
    if phase3 is None:
        return {"available": False}

    phase_config = phase3.get("phase_config") or {}
    scheduler = phase3.get("scheduler_progress") or {}
    downloads = phase3.get("downloads") or {}
    imported = phase3.get("imported_historical_1s") or {}

    coin_progress: dict[str, Any] = {}
    for coin in RESEARCH_COINS:
        coin_data = imported.get(coin) or {}
        coin_progress[coin] = {
            "row_count_1s": coin_data.get("row_count"),
            "coverage_hours": coin_data.get("coverage_hours"),
            "freshness_status": coin_data.get("freshness_status"),
            "last_timestamp": coin_data.get("last_timestamp"),
        }

    return {
        "available": True,
        "phase_name": phase_config.get("phase_backfill_name"),
        "target_days": phase_config.get("target_days"),
        "estimated_real_days_to_complete": phase_config.get("estimated_real_days_to_complete"),
        "latest_action": scheduler.get("latest_action"),
        "chunks_completed_today": scheduler.get("chunks_completed_today"),
        "failed_runs_in_row": scheduler.get("failed_runs_in_row"),
        "paused": scheduler.get("paused"),
        "downloads_total_files": downloads.get("total_files"),
        "downloads_by_coin": downloads.get("by_coin"),
        "next_recommended_action": phase3.get("next_recommended_action"),
        "coin_progress": coin_progress,
        "report_path": str(_phase3_audit_path(Path(phase3.get("_reports_root") or "."))),
    }


def _files_downloaded_today(phase3: dict[str, Any] | None) -> dict[str, Any]:
    if phase3 is None:
        return {"available": False, "count": None, "details": None}

    scheduler = phase3.get("scheduler_progress") or {}
    count = scheduler.get("files_downloaded_today")
    new_count = scheduler.get("files_downloaded_new")
    return {
        "available": count is not None,
        "count": count,
        "new_count": new_count,
        "chunks_completed_today": scheduler.get("chunks_completed_today"),
    }


def _collect_warnings_and_failures(
    *,
    validation_status: dict[str, dict[str, dict[str, Any]]],
    phase3: dict[str, Any] | None,
    latest_candidate: dict[str, Any],
    latest_relaxed_candidate: dict[str, Any],
) -> tuple[list[str], list[str]]:
    warnings: list[str] = [RESEARCH_ONLY_NOTE]
    failures: list[str] = []

    for coin, intervals in validation_status.items():
        for interval, status in intervals.items():
            if not status.get("available"):
                warnings.append(f"No validation report for {coin} {interval}.")
                continue
            if status.get("passed") is False:
                failures.append(f"{coin} {interval} validation did not pass.")
            for note in status.get("warnings") or []:
                warnings.append(f"{coin} {interval}: {note}")

    if phase3 is not None:
        for note in phase3.get("warnings") or []:
            warnings.append(str(note))
        scheduler = phase3.get("scheduler_progress") or {}
        for note in scheduler.get("warnings") or []:
            warnings.append(str(note))
        for note in scheduler.get("errors") or []:
            failures.append(str(note))
        if scheduler.get("paused"):
            failures.append("Scheduler autopilot is paused.")
        failed_runs = scheduler.get("failed_runs_in_row")
        if isinstance(failed_runs, int) and failed_runs > 0:
            warnings.append(f"Scheduler has {failed_runs} failed run(s) in a row.")

        candidate_freshness = phase3.get("candidate_freshness") or {}
        if candidate_freshness.get("repeated_unchanged_warning"):
            warnings.append("Candidate validation results unchanged across recent reports.")

    if latest_candidate.get("available"):
        if latest_candidate.get("insufficient_trades"):
            warnings.append("Latest candidate has insufficient paper trades for confidence.")
        if latest_candidate.get("not_market_ready"):
            warnings.append("Latest candidate is not market ready (paper-only research).")
        for note in latest_candidate.get("stability_warnings") or []:
            warnings.append(f"Candidate stability: {note}")

    if latest_relaxed_candidate.get("available"):
        bps = latest_relaxed_candidate.get("cumulative_net_return_bps")
        if bps is not None and float(bps) < 0:
            warnings.append(
                "Latest relaxed candidate shows negative cumulative net return after costs "
                "(research-only; does not prove future results)."
            )

    deduped_warnings: list[str] = []
    seen: set[str] = set()
    for item in warnings:
        if item not in seen:
            seen.add(item)
            deduped_warnings.append(item)

    deduped_failures: list[str] = []
    seen_failures: set[str] = set()
    for item in failures:
        if item not in seen_failures:
            seen_failures.add(item)
            deduped_failures.append(item)

    return deduped_warnings, deduped_failures


def _derive_blockers(
    *,
    backfill: dict[str, Any],
    failures: list[str],
    validation_status: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    blockers: list[str] = []

    if backfill.get("available"):
        if backfill.get("paused"):
            blockers.append("Backfill scheduler is paused.")
        latest_action = backfill.get("latest_action")
        if latest_action in {"no_import_progress", "import_stalled"}:
            blockers.append(
                "Historical import is not advancing — check date/hour planning or importer path."
            )
        for coin, progress in (backfill.get("coin_progress") or {}).items():
            if progress.get("freshness_status") == "stale":
                blockers.append(f"{coin} historical 1s data looks stale.")

    for coin, intervals in validation_status.items():
        for interval, status in intervals.items():
            if status.get("available") and status.get("passed") is False:
                blockers.append(f"Fix {coin} {interval} validation before relying on that data.")

    for failure in failures:
        if failure not in blockers:
            blockers.append(failure)

    if not blockers:
        blockers.append("No hard blockers detected — continue planned research steps.")

    return blockers


def _tomorrow_tasks(
    *,
    backfill: dict[str, Any],
    blockers: list[str],
    validation_status: dict[str, dict[str, dict[str, Any]]],
    latest_candidate: dict[str, Any],
    latest_relaxed_candidate: dict[str, Any],
    changes: list[dict[str, Any]],
) -> list[str]:
    tasks: list[str] = []

    if backfill.get("available"):
        action = backfill.get("next_recommended_action")
        if action:
            tasks.append(f"Follow backfill recommendation: {action}.")
        else:
            tasks.append("Continue Phase 3 historical backfill for BTC and ETH.")
    else:
        tasks.append("Refresh the Phase 3 backfill audit so tomorrow's digest can track import progress.")

    if not changes:
        tasks.append("Run validation or resample steps if new CSV rows were collected today.")

    for coin, intervals in validation_status.items():
        for interval, status in intervals.items():
            if not status.get("available"):
                tasks.append(f"Run validation for {coin} {interval} when CSV data is available.")

    if latest_candidate.get("available") and latest_candidate.get("insufficient_trades"):
        tasks.append(
            "Review strict Candidate B filters — paper trade count is still low for research confidence."
        )

    if latest_relaxed_candidate.get("available"):
        tasks.append(
            "Compare relaxed Candidate B results against strict filters before changing research focus."
        )

    if any("not advancing" in blocker.lower() for blocker in blockers):
        tasks.append("Investigate importer path and download date/hour coverage before scaling tests.")

    tasks.append("Re-run this digest after the next research session: python3 bid_ask_daily_progress_digest.py build")

    deduped: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        if task not in seen:
            seen.add(task)
            deduped.append(task)
    return deduped


def build_daily_progress_digest(
    *,
    date_yyyymmdd: str,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    """Build the daily progress digest from local reports and CSV metadata."""
    phase3_path = _phase3_audit_path(reports_root)
    phase3 = _load_json(phase3_path)
    if phase3 is not None:
        phase3 = dict(phase3)
        phase3["_reports_root"] = str(reports_root)

    changes = _reports_changed_today(
        data_root=data_root,
        reports_root=reports_root,
        date_yyyymmdd=date_yyyymmdd,
    )
    validation_status = _collect_validation_status(data_root)
    rows = _rows_by_coin_interval(data_root, phase3)
    backfill = _backfill_progress(phase3)
    files_downloaded = _files_downloaded_today(phase3)
    latest_candidate = _latest_candidate_validation(reports_root, relaxed=False)
    latest_relaxed_candidate = _latest_candidate_validation(reports_root, relaxed=True)
    warnings, failures = _collect_warnings_and_failures(
        validation_status=validation_status,
        phase3=phase3,
        latest_candidate=latest_candidate,
        latest_relaxed_candidate=latest_relaxed_candidate,
    )
    blockers = _derive_blockers(
        backfill=backfill,
        failures=failures,
        validation_status=validation_status,
    )
    tomorrow_tasks = _tomorrow_tasks(
        backfill=backfill,
        blockers=blockers,
        validation_status=validation_status,
        latest_candidate=latest_candidate,
        latest_relaxed_candidate=latest_relaxed_candidate,
        changes=changes,
    )

    latest_timestamps = {
        coin: {
            "last_timestamp": (rows.get(coin) or {}).get("1s", {}).get("last_timestamp"),
            "last_timestamp_ms": (rows.get(coin) or {}).get("1s", {}).get("last_timestamp_ms"),
        }
        for coin in RESEARCH_COINS
    }

    return {
        "date": date_yyyymmdd,
        "generated_at": _utc_now_iso(),
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "real_execution_ready": False,
        "safety_statement": SAFETY_STATEMENT,
        "what_changed_today": {
            "report_count": len(changes),
            "items": changes,
            "summary_lines": _describe_changes(changes),
        },
        "files_downloaded_today": files_downloaded,
        "backfill_progress": backfill,
        "latest_timestamps": latest_timestamps,
        "rows_by_coin_interval": rows,
        "validation_status": validation_status,
        "latest_candidate_result": latest_candidate,
        "latest_relaxed_candidate_result": latest_relaxed_candidate,
        "warnings": warnings,
        "failures": failures,
        "blockers": blockers,
        "tomorrow_recommended_tasks": tomorrow_tasks,
        "inputs": {
            "reports_root": str(reports_root),
            "data_root": str(data_root),
            "phase3_audit_path": str(phase3_path),
            "phase3_available": phase3 is not None,
        },
    }


def _format_candidate_section(candidate: dict[str, Any], *, title: str) -> list[str]:
    lines = [f"## {title}", ""]
    if not candidate.get("available"):
        lines.append("- No report available yet.")
        lines.append("")
        return lines

    lines.extend(
        [
            f"- candidate: {candidate.get('candidate_name') or 'n/a'}",
            f"- paper trades: {candidate.get('filtered_paper_trades')}",
            f"- cumulative net bps: {candidate.get('cumulative_net_return_bps')}",
            f"- positive after costs: {candidate.get('positive_after_costs')}",
            f"- paper candidate pass: {candidate.get('paper_candidate_pass')}",
            f"- report: {candidate.get('report_path') or 'n/a'}",
            "",
        ]
    )
    return lines


def write_digest_markdown(
    digest: dict[str, Any],
    output_dir: Path,
    *,
    filename: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    lines = [
        "# Bid/Ask Daily Progress Digest",
        "",
        f"Date: {digest['date']}",
        f"Generated at: {digest['generated_at']}",
        "",
        "## Safety",
        "",
        f"- paper_only: {str(digest['paper_only']).lower()}",
        f"- execution_enabled: {str(digest['execution_enabled']).lower()}",
        f"- real_execution_ready: {str(digest['real_execution_ready']).lower()}",
        "",
        RESEARCH_ONLY_NOTE,
        "",
        "## What Changed Today",
        "",
    ]
    lines.extend(f"- {line}" for line in digest["what_changed_today"]["summary_lines"])
    lines.extend(["", "## Files Downloaded Today", ""])

    downloaded = digest["files_downloaded_today"]
    if downloaded.get("available"):
        lines.append(f"- count: {downloaded.get('count')}")
        lines.append(f"- new_count: {downloaded.get('new_count')}")
        lines.append(f"- chunks_completed_today: {downloaded.get('chunks_completed_today')}")
    else:
        lines.append("- Download counts not available (Phase 3 audit missing).")
    lines.extend(["", "## Backfill Progress", ""])

    backfill = digest["backfill_progress"]
    if backfill.get("available"):
        lines.extend(
            [
                f"- phase: {backfill.get('phase_name') or 'n/a'}",
                f"- target_days: {backfill.get('target_days')}",
                f"- latest_action: {backfill.get('latest_action')}",
                f"- downloads_total_files: {backfill.get('downloads_total_files')}",
                f"- next_recommended_action: {backfill.get('next_recommended_action') or 'n/a'}",
            ]
        )
        for coin, progress in (backfill.get("coin_progress") or {}).items():
            lines.append(
                f"- {coin}: {progress.get('row_count_1s')} rows, "
                f"{progress.get('coverage_hours')} hours, "
                f"freshness={progress.get('freshness_status')}, "
                f"last={progress.get('last_timestamp') or 'n/a'}"
            )
    else:
        lines.append("- Phase 3 backfill audit not available.")

    lines.extend(["", "## Latest Timestamps (BTC / ETH)", ""])
    for coin in RESEARCH_COINS:
        ts = digest["latest_timestamps"].get(coin) or {}
        lines.append(f"- {coin}: {ts.get('last_timestamp') or 'n/a'}")

    lines.extend(["", "## Rows by Coin / Interval", ""])
    for coin in RESEARCH_COINS:
        intervals = digest["rows_by_coin_interval"].get(coin) or {}
        parts = []
        for interval in INTERVALS:
            row = intervals.get(interval) or {}
            parts.append(f"{interval}={row.get('row_count')}")
        lines.append(f"- {coin}: " + ", ".join(parts))

    lines.extend(["", "## Validation Status", ""])
    for coin in RESEARCH_COINS:
        intervals = digest["validation_status"].get(coin) or {}
        for interval in INTERVALS:
            status = intervals.get(interval) or {}
            if not status.get("available"):
                lines.append(f"- {coin} {interval}: no report")
                continue
            passed = status.get("passed")
            lines.append(
                f"- {coin} {interval}: passed={passed}, rows={status.get('row_count')}, "
                f"missing={status.get('missing_periods_count')}"
            )

    lines.extend(_format_candidate_section(digest["latest_candidate_result"], title="Latest Candidate Result"))
    lines.extend(
        _format_candidate_section(
            digest["latest_relaxed_candidate_result"],
            title="Latest Relaxed Candidate Result",
        )
    )

    lines.extend(["## Warnings", ""])
    lines.extend(f"- {item}" for item in digest["warnings"])
    lines.extend(["", "## Failures", ""])
    if digest["failures"]:
        lines.extend(f"- {item}" for item in digest["failures"])
    else:
        lines.append("- None recorded.")

    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- {item}" for item in digest["blockers"])

    lines.extend(["", "## Tomorrow's Recommended Tasks", ""])
    lines.extend(f"- {item}" for item in digest["tomorrow_recommended_tasks"])
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_digest_json(
    digest: dict[str, Any],
    output_dir: Path,
    *,
    filename: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    output_path.write_text(json.dumps(digest, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def build_and_write_digest(
    *,
    date_yyyymmdd: str,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    latest_md_name: str = DEFAULT_LATEST_MD_NAME,
    latest_json_name: str = DEFAULT_LATEST_JSON_NAME,
) -> dict[str, Any]:
    digest = build_daily_progress_digest(
        date_yyyymmdd=date_yyyymmdd,
        reports_root=reports_root,
        data_root=data_root,
    )

    dated_json_name = f"daily_progress_{date_yyyymmdd}.json"
    dated_md_name = f"daily_progress_{date_yyyymmdd}.md"

    json_dated = write_digest_json(digest, output_dir, filename=dated_json_name)
    md_dated = write_digest_markdown(digest, output_dir, filename=dated_md_name)
    json_latest = write_digest_json(digest, output_dir, filename=latest_json_name)
    md_latest = write_digest_markdown(digest, output_dir, filename=latest_md_name)

    digest["output_paths"] = {
        "json_dated": str(json_dated),
        "markdown_dated": str(md_dated),
        "json_latest": str(json_latest),
        "markdown_latest": str(md_latest),
    }
    return digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a daily bid/ask research progress digest (paper-only, read-only inputs).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build",
        help="Read local reports and write daily progress digest files.",
    )
    build_parser.add_argument(
        "--date",
        default=None,
        help="Digest date as YYYYMMDD (default: today UTC).",
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
        help="Directory for digest output files.",
    )
    build_parser.add_argument(
        "--latest-name",
        default=DEFAULT_LATEST_MD_NAME,
        help="Filename for the latest Markdown digest copy.",
    )
    build_parser.add_argument(
        "--json-latest-name",
        default=DEFAULT_LATEST_JSON_NAME,
        help="Filename for the latest JSON digest copy.",
    )

    args = parser.parse_args(argv)

    if args.command != "build":
        parser.error(f"Unknown command: {args.command}")
        return 2

    date_yyyymmdd = args.date or _default_date_utc()
    if len(date_yyyymmdd) != 8 or not date_yyyymmdd.isdigit():
        parser.error("--date must be YYYYMMDD")

    digest = build_and_write_digest(
        date_yyyymmdd=date_yyyymmdd,
        reports_root=args.reports_root,
        data_root=args.data_root,
        output_dir=args.output_dir,
        latest_md_name=args.latest_name,
        latest_json_name=args.json_latest_name,
    )

    paths = digest["output_paths"]
    print(f"date={digest['date']}")
    print(f"json_dated_path={paths['json_dated']}")
    print(f"markdown_dated_path={paths['markdown_dated']}")
    print(f"json_latest_path={paths['json_latest']}")
    print(f"markdown_latest_path={paths['markdown_latest']}")
    print(f"paper_only={str(digest['paper_only']).lower()}")
    print(f"execution_enabled={str(digest['execution_enabled']).lower()}")
    print(f"real_execution_ready={str(digest['real_execution_ready']).lower()}")
    print(f"changes_today={digest['what_changed_today']['report_count']}")
    print(f"warnings={len(digest['warnings'])}")
    print(f"failures={len(digest['failures'])}")
    print(f"blockers={len(digest['blockers'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
