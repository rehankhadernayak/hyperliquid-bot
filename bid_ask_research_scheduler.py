"""Daily research scheduler for bounded historical backfill, live data, training, and reports.

Research-only, paper-only layer on top of bid_ask_research_orchestrator.py.
No wallets, signing, orders, or exchange execution endpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import bid_ask_daily_summary
import bid_ask_research_orchestrator as orchestrator
from bid_ask_schema import SAFETY_STATEMENT, TIMESTAMP_MS
from historical_l2_backfill_planner import validate_date_yyyymmdd
from historical_l2_importer import default_historical_output_path, summarize_historical_coverage

logger = logging.getLogger(__name__)

DEFAULT_STATE_DIR = Path("reports/bid_ask_research_scheduler")
DEFAULT_STATE_PATH = DEFAULT_STATE_DIR / "state.json"
DEFAULT_RUNS_DIR = DEFAULT_STATE_DIR / "runs"
DEFAULT_HOURS: list[int] = [0, 1, 2]
DEFAULT_MAX_FILES = orchestrator.DEFAULT_MAX_FILES
DEFAULT_LIVE_MAX_DURATION_SECONDS = orchestrator.DEFAULT_LIVE_MAX_DURATION_SECONDS
LARGE_RANGE_DAYS = 90
ALL_HOURS = list(range(24))
DEFAULT_CHUNK_SIZE_HOURS = 3
MAX_CHUNK_SIZE_HOURS = 8
MAX_TOTAL_HOURS_PER_RUN = 12
MAX_TOTAL_HOURS_ALLOW_MORE = 24

ARCHIVE_AVAILABILITY_WARNING = (
    "Historical archive availability is not guaranteed; data may be missing or uploaded late."
)

LAUNCHD_PLIST_NAME = "com.hyperliquid.research.daily.plist"
LAUNCHD_SETUP_MD = "launchd_setup.md"
DEFAULT_HISTORICAL_START_HOUR = 0
DEFAULT_HISTORICAL_END_HOUR = 23
AUTO_BACKFILL_COMPLETE_MESSAGE = "Historical backfill range complete; no chunks remaining."
REPAIR_INFERRED_WARNING = (
    "Repair inferred completion from existing data; re-validate before production use."
)
AUTOPILOT_FAILURE_PAUSE_THRESHOLD = 3
AUTOPILOT_DISABLED_MESSAGE = "Autopilot is disabled; no work performed."
AUTOPILOT_PAUSED_MESSAGE = "Autopilot is paused; no work performed."
LAUNCHD_PLIST_NAME_AUTOPILOT = "com.hyperliquid.research.autopilot.plist"


@dataclass
class AutopilotState:
    enabled: bool = False
    started_at: str | None = None
    last_run_date: str | None = None
    last_success_date: str | None = None
    runs_today: int = 0
    chunks_completed_today: int = 0
    files_downloaded_today: int = 0
    daily_chunk_limit: int = 1
    daily_file_limit: int = 9
    ramp_enabled: bool = True
    ramp_every_successful_days: int = 2
    ramp_chunk_increment: int = 1
    max_daily_chunk_limit: int = 4
    max_daily_file_limit: int = 36
    successful_days_in_row: int = 0
    failed_runs_in_row: int = 0
    paused: bool = False
    pause_reason: str | None = None
    last_autopilot_report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "started_at": self.started_at,
            "last_run_date": self.last_run_date,
            "last_success_date": self.last_success_date,
            "runs_today": self.runs_today,
            "chunks_completed_today": self.chunks_completed_today,
            "files_downloaded_today": self.files_downloaded_today,
            "daily_chunk_limit": self.daily_chunk_limit,
            "daily_file_limit": self.daily_file_limit,
            "ramp_enabled": self.ramp_enabled,
            "ramp_every_successful_days": self.ramp_every_successful_days,
            "ramp_chunk_increment": self.ramp_chunk_increment,
            "max_daily_chunk_limit": self.max_daily_chunk_limit,
            "max_daily_file_limit": self.max_daily_file_limit,
            "successful_days_in_row": self.successful_days_in_row,
            "failed_runs_in_row": self.failed_runs_in_row,
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "last_autopilot_report_path": self.last_autopilot_report_path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> AutopilotState:
        if not payload:
            return cls()
        return cls(
            enabled=bool(payload.get("enabled", False)),
            started_at=payload.get("started_at"),
            last_run_date=payload.get("last_run_date"),
            last_success_date=payload.get("last_success_date"),
            runs_today=int(payload.get("runs_today") or 0),
            chunks_completed_today=int(payload.get("chunks_completed_today") or 0),
            files_downloaded_today=int(payload.get("files_downloaded_today") or 0),
            daily_chunk_limit=int(payload.get("daily_chunk_limit") or 1),
            daily_file_limit=int(payload.get("daily_file_limit") or 9),
            ramp_enabled=bool(payload.get("ramp_enabled", True)),
            ramp_every_successful_days=int(payload.get("ramp_every_successful_days") or 2),
            ramp_chunk_increment=int(payload.get("ramp_chunk_increment") or 1),
            max_daily_chunk_limit=int(payload.get("max_daily_chunk_limit") or 4),
            max_daily_file_limit=int(payload.get("max_daily_file_limit") or 36),
            successful_days_in_row=int(payload.get("successful_days_in_row") or 0),
            failed_runs_in_row=int(payload.get("failed_runs_in_row") or 0),
            paused=bool(payload.get("paused", False)),
            pause_reason=payload.get("pause_reason"),
            last_autopilot_report_path=payload.get("last_autopilot_report_path"),
        )


@dataclass
class SchedulerState:
    historical_start_date: str | None = None
    historical_end_date_target: str | None = None
    historical_start_hour: int = DEFAULT_HISTORICAL_START_HOUR
    historical_end_hour: int = DEFAULT_HISTORICAL_END_HOUR
    next_historical_date: str | None = None
    completed_dates: list[str] = field(default_factory=list)
    failed_dates: list[str] = field(default_factory=list)
    completed_batches: list[dict[str, Any]] = field(default_factory=list)
    failed_batches: list[dict[str, Any]] = field(default_factory=list)
    last_live_session_at: str | None = None
    last_training_at: str | None = None
    total_historical_files_downloaded: int = 0
    total_historical_days_completed: int = 0
    last_report_path: str | None = None
    last_run_at: str | None = None
    next_recommended_action: str | None = None
    warnings: list[str] = field(default_factory=list)
    coins: list[str] = field(default_factory=list)
    autopilot: AutopilotState = field(default_factory=AutopilotState)

    def to_dict(self) -> dict[str, Any]:
        return {
            "historical_start_date": self.historical_start_date,
            "historical_end_date_target": self.historical_end_date_target,
            "historical_start_hour": self.historical_start_hour,
            "historical_end_hour": self.historical_end_hour,
            "next_historical_date": self.next_historical_date,
            "completed_dates": self.completed_dates,
            "failed_dates": self.failed_dates,
            "completed_batches": self.completed_batches,
            "failed_batches": self.failed_batches,
            "last_live_session_at": self.last_live_session_at,
            "last_training_at": self.last_training_at,
            "total_historical_files_downloaded": self.total_historical_files_downloaded,
            "total_historical_days_completed": self.total_historical_days_completed,
            "last_report_path": self.last_report_path,
            "last_run_at": self.last_run_at,
            "next_recommended_action": self.next_recommended_action,
            "warnings": self.warnings,
            "coins": self.coins,
            "autopilot": self.autopilot.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SchedulerState:
        return cls(
            historical_start_date=payload.get("historical_start_date"),
            historical_end_date_target=payload.get("historical_end_date_target"),
            historical_start_hour=int(
                payload.get("historical_start_hour", DEFAULT_HISTORICAL_START_HOUR)
            ),
            historical_end_hour=int(
                payload.get("historical_end_hour", DEFAULT_HISTORICAL_END_HOUR)
            ),
            next_historical_date=payload.get("next_historical_date"),
            completed_dates=list(payload.get("completed_dates") or []),
            failed_dates=list(payload.get("failed_dates") or []),
            completed_batches=list(payload.get("completed_batches") or []),
            failed_batches=list(payload.get("failed_batches") or []),
            last_live_session_at=payload.get("last_live_session_at"),
            last_training_at=payload.get("last_training_at"),
            total_historical_files_downloaded=int(
                payload.get("total_historical_files_downloaded") or 0
            ),
            total_historical_days_completed=int(
                payload.get("total_historical_days_completed") or 0
            ),
            last_report_path=payload.get("last_report_path"),
            last_run_at=payload.get("last_run_at"),
            next_recommended_action=payload.get("next_recommended_action"),
            warnings=list(payload.get("warnings") or []),
            coins=list(payload.get("coins") or []),
            autopilot=AutopilotState.from_dict(payload.get("autopilot")),
        )


@dataclass
class AutopilotReport:
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    paper_only: bool = True
    execution_enabled: bool = False
    command: str = "autopilot-tick"
    safety_statement: str = SAFETY_STATEMENT
    action: str = "noop"
    coins: list[str] = field(default_factory=list)
    chunk_size_hours: int = DEFAULT_CHUNK_SIZE_HOURS
    estimated_files: int = 0
    requested_files: int = 0
    files_downloaded: int = 0
    daily_chunk_limit: int | None = None
    daily_file_limit: int | None = None
    chunks_completed_today: int | None = None
    files_downloaded_today: int | None = None
    autopilot_snapshot: dict[str, Any] = field(default_factory=dict)
    workflows_run: list[str] = field(default_factory=list)
    historical_date: str | None = None
    historical_hours: list[str] = field(default_factory=list)
    train_interval: str | None = None
    horizon: int | None = None
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    next_recommended_action: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "paper_only": self.paper_only,
            "execution_enabled": self.execution_enabled,
            "command": self.command,
            "safety_statement": self.safety_statement,
            "action": self.action,
            "coins": self.coins,
            "chunk_size_hours": self.chunk_size_hours,
            "estimated_files": self.estimated_files,
            "requested_files": self.requested_files,
            "files_downloaded": self.files_downloaded,
            "daily_chunk_limit": self.daily_chunk_limit,
            "daily_file_limit": self.daily_file_limit,
            "chunks_completed_today": self.chunks_completed_today,
            "files_downloaded_today": self.files_downloaded_today,
            "autopilot_snapshot": self.autopilot_snapshot,
            "workflows_run": self.workflows_run,
            "historical_date": self.historical_date,
            "historical_hours": self.historical_hours,
            "train_interval": self.train_interval,
            "horizon": self.horizon,
            "warnings": self.warnings,
            "failures": self.failures,
            "next_recommended_action": self.next_recommended_action,
            "details": self.details,
        }


@dataclass
class SchedulerReport:
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    paper_only: bool = True
    execution_enabled: bool = False
    safety_statement: str = SAFETY_STATEMENT
    workflows_run: list[str] = field(default_factory=list)
    coins: list[str] = field(default_factory=list)
    historical_date: str | None = None
    historical_hours: list[str] = field(default_factory=list)
    live_session_duration: float | None = None
    source: str | None = None
    train_interval: str | None = None
    horizon: int | None = None
    row_counts: dict[str, Any] = field(default_factory=dict)
    training_reports: list[str] = field(default_factory=list)
    model_artifact_paths: list[str] = field(default_factory=list)
    evaluation_report_paths: list[str] = field(default_factory=list)
    filter_report_paths: list[str] = field(default_factory=list)
    model_paths: list[str] = field(default_factory=list)
    evaluation_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    next_recommended_action: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "paper_only": self.paper_only,
            "execution_enabled": self.execution_enabled,
            "safety_statement": self.safety_statement,
            "workflows_run": self.workflows_run,
            "coins": self.coins,
            "historical_date": self.historical_date,
            "historical_hours": self.historical_hours,
            "live_session_duration": self.live_session_duration,
            "source": self.source,
            "train_interval": self.train_interval,
            "horizon": self.horizon,
            "row_counts": self.row_counts,
            "training_reports": self.training_reports,
            "model_artifact_paths": self.model_artifact_paths,
            "evaluation_report_paths": self.evaluation_report_paths,
            "filter_report_paths": self.filter_report_paths,
            "model_paths": self.model_paths,
            "evaluation_paths": self.evaluation_paths,
            "warnings": self.warnings,
            "failures": self.failures,
            "next_recommended_action": self.next_recommended_action,
            "details": self.details,
        }


@dataclass
class BatchSeriesReport:
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    paper_only: bool = True
    execution_enabled: bool = False
    command: str = "run-batch-series"
    safety_statement: str = SAFETY_STATEMENT
    date: str | None = None
    coins: list[str] = field(default_factory=list)
    chunks: list[list[str]] = field(default_factory=list)
    chunks_completed: int = 0
    chunks_failed: int = 0
    chunks_skipped: int = 0
    per_chunk_report_paths: list[dict[str, Any]] = field(default_factory=list)
    train_interval: str | None = None
    horizon: int | None = None
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    next_recommended_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "paper_only": self.paper_only,
            "execution_enabled": self.execution_enabled,
            "command": self.command,
            "safety_statement": self.safety_statement,
            "date": self.date,
            "coins": self.coins,
            "chunks": self.chunks,
            "chunks_completed": self.chunks_completed,
            "chunks_failed": self.chunks_failed,
            "chunks_skipped": self.chunks_skipped,
            "per_chunk_report_paths": self.per_chunk_report_paths,
            "train_interval": self.train_interval,
            "horizon": self.horizon,
            "warnings": self.warnings,
            "failures": self.failures,
            "next_recommended_action": self.next_recommended_action,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _estimate_chunk_files(num_coins: int, chunk_size_hours: int) -> int:
    return num_coins * chunk_size_hours


def _reset_autopilot_daily_counters(autopilot: AutopilotState, today: str) -> None:
    autopilot.runs_today = 0
    autopilot.chunks_completed_today = 0
    autopilot.files_downloaded_today = 0
    autopilot.last_run_date = today


def _utc_date_from_iso_timestamp(iso_timestamp: str) -> str | None:
    """Extract UTC YYYYMMDD from an ISO-8601 timestamp."""
    if not iso_timestamp or len(iso_timestamp) < 10:
        return None
    date_part = iso_timestamp[:10]
    if len(date_part) != 10 or date_part[4] != "-" or date_part[7] != "-":
        return None
    return date_part.replace("-", "")


def _reconcile_autopilot_usage_from_batches(
    state: SchedulerState,
    today: str,
) -> int:
    """Count completed batches whose completed_at falls on today (UTC)."""
    chunks = 0
    for batch in state.completed_batches:
        completed_at = batch.get("completed_at")
        if not completed_at:
            continue
        if _utc_date_from_iso_timestamp(str(completed_at)) == today:
            chunks += 1
    return chunks


def _reconcile_autopilot_usage_from_reports(
    runs_dir: Path,
    today: str,
) -> tuple[int, int]:
    """Count chunk_completed autopilot reports for today (UTC). Conservative usage estimate."""
    chunks = 0
    files = 0
    if not runs_dir.is_dir():
        return chunks, files
    for path in sorted(runs_dir.glob("bid_ask_scheduler_autopilot_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("command") != "autopilot-tick":
            continue
        if _utc_date_from_iso_timestamp(str(payload.get("generated_at") or "")) != today:
            continue
        if payload.get("action") != "chunk_completed":
            continue
        chunks += 1
        files += int(
            payload.get("files_downloaded")
            or payload.get("requested_files")
            or payload.get("estimated_files")
            or 0
        )
    return chunks, files


def _reconcile_autopilot_daily_usage(
    autopilot: AutopilotState,
    state: SchedulerState,
    *,
    today: str,
    runs_dir: Path | None = None,
) -> bool:
    """Align daily usage counters with persisted state and today's evidence. Returns True if modified."""
    if autopilot.last_run_date != today:
        _reset_autopilot_daily_counters(autopilot, today)
        return True

    observed_chunks = autopilot.chunks_completed_today
    observed_files = autopilot.files_downloaded_today
    observed_chunks = max(observed_chunks, _reconcile_autopilot_usage_from_batches(state, today))
    if runs_dir is not None:
        report_chunks, report_files = _reconcile_autopilot_usage_from_reports(runs_dir, today)
        observed_chunks = max(observed_chunks, report_chunks)
        observed_files = max(observed_files, report_files)

    modified = False
    if observed_chunks > autopilot.chunks_completed_today:
        autopilot.chunks_completed_today = observed_chunks
        modified = True
    if observed_files > autopilot.files_downloaded_today:
        autopilot.files_downloaded_today = observed_files
        modified = True
    return modified


def _maybe_reset_autopilot_for_new_day(autopilot: AutopilotState) -> None:
    today = _utc_today_yyyymmdd()
    if autopilot.last_run_date != today:
        _reset_autopilot_daily_counters(autopilot, today)


def _populate_autopilot_report_limits(
    report: AutopilotReport,
    autopilot: AutopilotState,
    *,
    requested_files: int,
) -> None:
    report.requested_files = requested_files
    report.estimated_files = requested_files
    report.daily_chunk_limit = autopilot.daily_chunk_limit
    report.daily_file_limit = autopilot.daily_file_limit
    report.chunks_completed_today = autopilot.chunks_completed_today
    report.files_downloaded_today = autopilot.files_downloaded_today


def _autopilot_would_exceed_daily_limits(
    autopilot: AutopilotState,
    requested_files: int,
) -> str | None:
    """Return skip reason when the next chunk would exceed daily limits, else None."""
    if autopilot.chunks_completed_today + 1 > autopilot.daily_chunk_limit:
        return (
            f"Daily chunk limit would be exceeded "
            f"({autopilot.chunks_completed_today}+1 > {autopilot.daily_chunk_limit}); skipping chunk"
        )
    if autopilot.files_downloaded_today + requested_files > autopilot.daily_file_limit:
        return (
            f"Next chunk would exceed daily file limit "
            f"({autopilot.files_downloaded_today}+{requested_files} > "
            f"{autopilot.daily_file_limit}); skipping chunk"
        )
    return None


def _record_autopilot_success_day(autopilot: AutopilotState, today: str) -> None:
    """Increment successful_days_in_row once per calendar day on first chunk success."""
    if autopilot.last_success_date == today:
        return
    yesterday = _date_add_days(today, -1)
    if autopilot.last_success_date == yesterday:
        autopilot.successful_days_in_row += 1
    else:
        autopilot.successful_days_in_row = 1
    autopilot.last_success_date = today


def _maybe_ramp_autopilot_limits(
    autopilot: AutopilotState,
    *,
    chunk_size_hours: int,
    num_coins: int,
) -> bool:
    """Apply ramp when streak threshold met. Returns True if limits were increased."""
    if not autopilot.ramp_enabled:
        return False
    if autopilot.successful_days_in_row < autopilot.ramp_every_successful_days:
        return False
    autopilot.daily_chunk_limit = min(
        autopilot.max_daily_chunk_limit,
        autopilot.daily_chunk_limit + autopilot.ramp_chunk_increment,
    )
    autopilot.daily_file_limit = min(
        autopilot.max_daily_file_limit,
        autopilot.daily_chunk_limit * chunk_size_hours * num_coins,
    )
    autopilot.successful_days_in_row = 0
    return True


def _autopilot_next_recommended_action(state: SchedulerState, autopilot: AutopilotState) -> str:
    if autopilot.paused:
        return (
            "python3 bid_ask_research_scheduler.py configure-autopilot --resume"
        )
    if not autopilot.enabled:
        return "python3 bid_ask_research_scheduler.py configure-autopilot --enable"
    if autopilot.chunks_completed_today + 1 > autopilot.daily_chunk_limit:
        return "Wait until tomorrow or increase daily limits via configure-autopilot"
    if state.next_historical_date is None:
        return "python3 bid_ask_research_scheduler.py status"
    return "python3 bid_ask_research_scheduler.py autopilot-tick"


def write_autopilot_report(
    report: AutopilotReport,
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    label: str = "autopilot",
) -> tuple[Path, Path]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.replace(":", "").replace("-", "").replace(".", "")
    json_path = runs_dir / f"bid_ask_scheduler_{label}_{stamp}.json"
    md_path = runs_dir / f"bid_ask_scheduler_{label}_{stamp}.md"

    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    lines = [
        "# Bid/Ask Research Scheduler Autopilot Report",
        "",
        f"- generated_at: {report.generated_at}",
        f"- paper_only: {report.paper_only}",
        f"- execution_enabled: {report.execution_enabled}",
        f"- command: {report.command}",
        f"- action: {report.action}",
        f"- coins: {', '.join(report.coins) or 'none'}",
        f"- chunk_size_hours: {report.chunk_size_hours}",
        f"- requested_files: {report.requested_files}",
        f"- estimated_files: {report.estimated_files}",
        f"- files_downloaded: {report.files_downloaded}",
    ]
    if report.daily_chunk_limit is not None:
        lines.append(f"- daily_chunk_limit: {report.daily_chunk_limit}")
    if report.daily_file_limit is not None:
        lines.append(f"- daily_file_limit: {report.daily_file_limit}")
    if report.chunks_completed_today is not None:
        lines.append(f"- chunks_completed_today: {report.chunks_completed_today}")
    if report.files_downloaded_today is not None:
        lines.append(f"- files_downloaded_today: {report.files_downloaded_today}")
    if report.historical_date:
        lines.append(f"- historical_date: {report.historical_date}")
    if report.historical_hours:
        lines.append(f"- historical_hours: {', '.join(report.historical_hours)}")
    if report.train_interval:
        lines.append(f"- train_interval: {report.train_interval}")
    if report.horizon is not None:
        lines.append(f"- horizon: {report.horizon}")

    ap = report.autopilot_snapshot
    if ap:
        lines.extend(
            [
                "",
                "## Autopilot state",
                "",
                f"- enabled: {ap.get('enabled')}",
                f"- paused: {ap.get('paused')}",
                f"- daily_chunk_limit: {ap.get('daily_chunk_limit')}",
                f"- daily_file_limit: {ap.get('daily_file_limit')}",
                f"- chunks_completed_today: {ap.get('chunks_completed_today')}",
                f"- files_downloaded_today: {ap.get('files_downloaded_today')}",
                f"- successful_days_in_row: {ap.get('successful_days_in_row')}",
                f"- failed_runs_in_row: {ap.get('failed_runs_in_row')}",
            ]
        )

    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)

    if report.failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in report.failures)

    if report.next_recommended_action:
        lines.extend(["", "## Next recommended action", "", f"- {report.next_recommended_action}"])

    lines.extend(["", report.safety_statement, ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def _date_add_days(date: str, days: int) -> str:
    normalized = validate_date_yyyymmdd(date)
    parsed = datetime.strptime(normalized, "%Y%m%d")
    return (parsed + timedelta(days=days)).strftime("%Y%m%d")


def _days_in_range(start: str, end: str) -> int:
    start_norm = validate_date_yyyymmdd(start)
    end_norm = validate_date_yyyymmdd(end)
    start_dt = datetime.strptime(start_norm, "%Y%m%d")
    end_dt = datetime.strptime(end_norm, "%Y%m%d")
    if end_dt < start_dt:
        raise ValueError(f"historical end date {end_norm} is before start date {start_norm}")
    return (end_dt - start_dt).days + 1


def _batch_key(date: str, hours: list[str]) -> tuple[str, tuple[str, ...]]:
    return date, tuple(sorted(hours, key=int))


def _validate_hour_in_range(hour: int, *, label: str = "hour") -> int:
    if not isinstance(hour, int) or hour < 0 or hour > 23:
        raise ValueError(f"{label} must be an integer between 0 and 23, got {hour!r}")
    return hour


def validate_batch_series_hours(
    *,
    start_hour: int,
    end_hour: int,
    chunk_size_hours: int = DEFAULT_CHUNK_SIZE_HOURS,
    allow_more: bool = False,
) -> None:
    """Validate bounded hour-range parameters for run-batch-series."""
    start = _validate_hour_in_range(start_hour, label="start-hour")
    end = _validate_hour_in_range(end_hour, label="end-hour")
    if start > end:
        raise ValueError(
            f"start-hour {start} must be less than or equal to end-hour {end}"
        )

    if chunk_size_hours < 1:
        raise ValueError(f"chunk-size-hours must be at least 1, got {chunk_size_hours}")

    max_chunk = MAX_CHUNK_SIZE_HOURS
    if chunk_size_hours > max_chunk and not allow_more:
        raise ValueError(
            f"chunk-size-hours {chunk_size_hours} exceeds max {max_chunk}; "
            "pass --allow-more to proceed"
        )

    total_hours = end - start + 1
    max_total = MAX_TOTAL_HOURS_ALLOW_MORE if allow_more else MAX_TOTAL_HOURS_PER_RUN
    if total_hours > max_total:
        raise ValueError(
            f"hour range spans {total_hours} hours (max {max_total}"
            f"{'' if allow_more else '; pass --allow-more for up to 24 hours'})"
        )


def split_hour_range_into_chunks(
    start_hour: int,
    end_hour: int,
    chunk_size_hours: int,
) -> list[list[str]]:
    """Split [start_hour, end_hour] into bounded chunks of hour strings."""
    validate_batch_series_hours(
        start_hour=start_hour,
        end_hour=end_hour,
        chunk_size_hours=chunk_size_hours,
        allow_more=True,
    )
    chunks: list[list[str]] = []
    current = start_hour
    while current <= end_hour:
        chunk_end = min(current + chunk_size_hours - 1, end_hour)
        chunks.append([str(h) for h in range(current, chunk_end + 1)])
        current = chunk_end + 1
    return chunks


def _hours_completed_for_date(state: SchedulerState, date: str) -> set[str]:
    completed: set[str] = set()
    for batch in state.completed_batches:
        if batch.get("date") == date:
            completed.update(str(h) for h in batch.get("hours") or [])
    return completed


def _expected_hours_for_state(state: SchedulerState) -> set[str]:
    start = _validate_hour_in_range(state.historical_start_hour, label="historical_start_hour")
    end = _validate_hour_in_range(state.historical_end_hour, label="historical_end_hour")
    if start > end:
        raise ValueError(
            f"historical_start_hour {start} must be <= historical_end_hour {end} in state"
        )
    return {str(h) for h in range(start, end + 1)}


def _all_hours_completed_for_date(state: SchedulerState, date: str) -> bool:
    return _hours_completed_for_date(state, date) >= _expected_hours_for_state(state)


def find_next_incomplete_chunk(
    state: SchedulerState,
    chunk_size_hours: int = DEFAULT_CHUNK_SIZE_HOURS,
) -> tuple[str, list[str]] | None:
    """Return the earliest incomplete (date, hour-chunk) from scheduler state."""
    if not state.next_historical_date:
        return None

    target_date = validate_date_yyyymmdd(state.next_historical_date)
    start_hour = state.historical_start_hour
    end_hour = state.historical_end_hour
    hour_chunks = split_hour_range_into_chunks(start_hour, end_hour, chunk_size_hours)

    for chunk_hours in hour_chunks:
        if not _chunk_hours_completed(state, target_date, chunk_hours):
            return target_date, chunk_hours

    return None


def _hour_range_timestamp_bounds(date: str, hours: list[str]) -> tuple[int, int]:
    """Return inclusive UTC millisecond bounds for a date and hour list."""
    date_norm = validate_date_yyyymmdd(date)
    hour_ints = sorted(int(h) for h in hours)
    base = datetime.strptime(date_norm, "%Y%m%d").replace(tzinfo=timezone.utc)
    start_dt = base.replace(hour=hour_ints[0], minute=0, second=0, microsecond=0)
    end_dt = base.replace(hour=hour_ints[-1], minute=59, second=59, microsecond=999000)
    return int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)


def _count_rows_in_timestamp_range(csv_path: Path, start_ms: int, end_ms: int) -> int:
    count = 0
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                timestamp_ms = int(row[TIMESTAMP_MS])
            except (KeyError, TypeError, ValueError):
                continue
            if start_ms <= timestamp_ms <= end_ms:
                count += 1
    return count


def validate_chunk_historical_data(
    *,
    date: str,
    hours: list[str],
    coins: list[str],
    data_root: Path,
) -> tuple[bool, list[str]]:
    """Conservative offline check that historical 1s data appears present for a chunk."""
    issues: list[str] = []
    start_ms, end_ms = _hour_range_timestamp_bounds(date, hours)

    for coin in coins:
        csv_path = default_historical_output_path(coin, output_root=data_root)
        if not csv_path.is_file():
            issues.append(f"{coin}: historical_1s CSV missing at {csv_path}")
            continue

        try:
            coverage = summarize_historical_coverage(csv_path)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"{coin}: coverage summary failed: {exc}")
            continue

        if coverage.duplicate_timestamps != 0:
            issues.append(f"{coin}: duplicate_timestamps={coverage.duplicate_timestamps}")
        if coverage.invalid_bid_ask_rows != 0:
            issues.append(f"{coin}: invalid_bid_ask_rows={coverage.invalid_bid_ask_rows}")

        rows_in_range = _count_rows_in_timestamp_range(csv_path, start_ms, end_ms)
        if rows_in_range == 0:
            issues.append(
                f"{coin}: zero rows in requested range {date} hours {','.join(hours)}"
            )

    return len(issues) == 0, issues


def _cycle_succeeded(report: SchedulerReport) -> bool:
    if report.failures:
        return False
    validate_payload = (report.details or {}).get("resample_validate") or {}
    if isinstance(validate_payload, dict):
        if validate_payload.get("failures"):
            return False
        for warning in validate_payload.get("warnings") or []:
            if "validation failed" in str(warning).lower():
                return False
    return not any("validation failed" in warning.lower() for warning in report.warnings)


def load_state(state_path: Path = DEFAULT_STATE_PATH) -> SchedulerState:
    if not state_path.is_file():
        raise FileNotFoundError(
            f"scheduler state not found at {state_path}; run init-state first"
        )
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    return SchedulerState.from_dict(payload)


def save_state(state: SchedulerState, state_path: Path = DEFAULT_STATE_PATH) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")


def write_scheduler_report(
    report: SchedulerReport,
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    label: str = "scheduler",
) -> tuple[Path, Path]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.replace(":", "").replace("-", "").replace(".", "")
    json_path = runs_dir / f"bid_ask_scheduler_{label}_{stamp}.json"
    md_path = runs_dir / f"bid_ask_scheduler_{label}_{stamp}.md"

    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    lines = [
        "# Bid/Ask Research Scheduler Report",
        "",
        f"- generated_at: {report.generated_at}",
        f"- paper_only: {report.paper_only}",
        f"- execution_enabled: {report.execution_enabled}",
        f"- workflows_run: {', '.join(report.workflows_run) or 'none'}",
        f"- coins: {', '.join(report.coins) or 'none'}",
    ]
    if report.historical_date:
        lines.append(f"- historical_date: {report.historical_date}")
    if report.historical_hours:
        lines.append(f"- historical_hours: {', '.join(report.historical_hours)}")
    if report.live_session_duration is not None:
        lines.append(f"- live_session_duration: {report.live_session_duration}")
    if report.source:
        lines.append(f"- source: {report.source}")
    if report.train_interval:
        lines.append(f"- train_interval: {report.train_interval}")
    if report.horizon is not None:
        lines.append(f"- horizon: {report.horizon}")

    lines.extend(["", "## Row counts", ""])
    for key, value in report.row_counts.items():
        lines.append(f"- {key}: {value}")

    if report.model_paths:
        lines.extend(["", "## Model paths", ""])
        lines.extend(f"- {path}" for path in report.model_paths)

    if report.evaluation_paths:
        lines.extend(["", "## Evaluation paths", ""])
        lines.extend(f"- {path}" for path in report.evaluation_paths)

    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)

    if report.failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in report.failures)

    if report.next_recommended_action:
        lines.extend(["", "## Next recommended action", "", f"- {report.next_recommended_action}"])

    lines.extend(["", report.safety_statement, ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def write_batch_series_report(
    report: BatchSeriesReport,
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> tuple[Path, Path]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.replace(":", "").replace("-", "").replace(".", "")
    json_path = runs_dir / f"bid_ask_scheduler_batch_series_{stamp}.json"
    md_path = runs_dir / f"bid_ask_scheduler_batch_series_{stamp}.md"

    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    lines = [
        "# Bid/Ask Research Scheduler Batch Series Report",
        "",
        f"- generated_at: {report.generated_at}",
        f"- paper_only: {report.paper_only}",
        f"- execution_enabled: {report.execution_enabled}",
        f"- command: {report.command}",
        f"- date: {report.date}",
        f"- coins: {', '.join(report.coins) or 'none'}",
        f"- chunks_completed: {report.chunks_completed}",
        f"- chunks_failed: {report.chunks_failed}",
        f"- chunks_skipped: {report.chunks_skipped}",
    ]
    if report.train_interval:
        lines.append(f"- train_interval: {report.train_interval}")
    if report.horizon is not None:
        lines.append(f"- horizon: {report.horizon}")

    lines.extend(["", "## Chunks", ""])
    for chunk in report.chunks:
        lines.append(f"- [{', '.join(chunk)}]")

    if report.per_chunk_report_paths:
        lines.extend(["", "## Per-chunk reports", ""])
        for entry in report.per_chunk_report_paths:
            hours = entry.get("hours") or []
            json_report = entry.get("json")
            md_report = entry.get("markdown")
            lines.append(f"- hours {', '.join(str(h) for h in hours)}: {json_report}")
            if md_report:
                lines.append(f"  - markdown: {md_report}")

    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)

    if report.failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in report.failures)

    if report.next_recommended_action:
        lines.extend(["", "## Next recommended action", "", f"- {report.next_recommended_action}"])

    lines.extend(["", report.safety_statement, ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def _chunk_completed_in_state(
    state: SchedulerState,
    date: str,
    hours: list[str],
) -> bool:
    batch_key = _batch_key(date, hours)
    existing_keys = {
        _batch_key(str(b.get("date")), [str(h) for h in b.get("hours") or []])
        for b in state.completed_batches
    }
    return batch_key in existing_keys


def _chunk_hours_completed(state: SchedulerState, date: str, hours: list[str]) -> bool:
    completed = _hours_completed_for_date(state, date)
    return set(hours) <= completed


def _count_downloaded_files(batch_report: orchestrator.OrchestratorReport) -> int:
    download = batch_report.details.get("download") or {}
    file_results = download.get("file_results") or []
    return sum(1 for item in file_results if item.get("status") == "downloaded")


def _extract_evaluation_paths(train_report: orchestrator.OrchestratorReport) -> list[str]:
    paths: list[str] = []
    decision_eval = (train_report.details or {}).get("decision_evaluation") or {}
    json_path = decision_eval.get("json_report_path")
    if json_path:
        paths.append(str(json_path))
    filter_eval = (train_report.details or {}).get("decision_filter") or {}
    filter_path = filter_eval.get("json_report_path")
    if filter_path:
        paths.append(str(filter_path))
    paths.extend(str(p) for p in train_report.report_paths)
    return paths


def _is_training_report_path(path: str) -> bool:
    return "bid_ask_training" in path and path.endswith(".json")


def _extract_training_metadata(
    train_report: orchestrator.OrchestratorReport,
) -> dict[str, list[str]]:
    """Pull training artifact and report paths from an orchestrator train-evaluate report."""
    decision_eval = (train_report.details or {}).get("decision_evaluation") or {}
    filter_eval = (train_report.details or {}).get("decision_filter") or {}
    evaluation_path = decision_eval.get("json_report_path")
    filter_path = filter_eval.get("json_report_path")
    return {
        "training_reports": [
            str(path) for path in train_report.report_paths if _is_training_report_path(str(path))
        ],
        "model_artifact_paths": [str(path) for path in train_report.model_paths],
        "evaluation_report_paths": [str(evaluation_path)] if evaluation_path else [],
        "filter_report_paths": [str(filter_path)] if filter_path else [],
    }


def _apply_train_metadata(report: SchedulerReport, train_report: orchestrator.OrchestratorReport) -> None:
    metadata = _extract_training_metadata(train_report)
    report.training_reports.extend(metadata["training_reports"])
    report.model_artifact_paths.extend(metadata["model_artifact_paths"])
    report.evaluation_report_paths.extend(metadata["evaluation_report_paths"])
    report.filter_report_paths.extend(metadata["filter_report_paths"])
    report.model_paths.extend(train_report.model_paths)
    report.evaluation_paths.extend(_extract_evaluation_paths(train_report))


def cmd_init_state(
    *,
    historical_start_date: str,
    historical_end_date: str | None = None,
    historical_start_hour: int = DEFAULT_HISTORICAL_START_HOUR,
    historical_end_hour: int = DEFAULT_HISTORICAL_END_HOUR,
    coins: list[str],
    reset: bool = False,
    confirm_large_range: bool = False,
    state_path: Path = DEFAULT_STATE_PATH,
) -> SchedulerState:
    """Initialize scheduler state without downloading anything."""
    start = validate_date_yyyymmdd(historical_start_date)
    end = validate_date_yyyymmdd(historical_end_date) if historical_end_date else None
    normalized_coins = orchestrator._normalize_coins(coins)
    start_hour = _validate_hour_in_range(historical_start_hour, label="historical-start-hour")
    end_hour = _validate_hour_in_range(historical_end_hour, label="historical-end-hour")
    if start_hour > end_hour:
        raise ValueError(
            f"historical-start-hour {start_hour} must be <= historical-end-hour {end_hour}"
        )

    if end is not None:
        span = _days_in_range(start, end)
        if span > LARGE_RANGE_DAYS and not confirm_large_range:
            raise ValueError(
                f"historical range spans {span} days (>{LARGE_RANGE_DAYS}); "
                "pass --confirm-large-range to proceed"
            )

    if state_path.is_file() and not reset:
        raise FileExistsError(
            f"state already exists at {state_path}; pass --reset to overwrite"
        )

    state = SchedulerState(
        historical_start_date=start,
        historical_end_date_target=end,
        historical_start_hour=start_hour,
        historical_end_hour=end_hour,
        next_historical_date=start,
        coins=normalized_coins,
        warnings=[ARCHIVE_AVAILABILITY_WARNING],
    )
    save_state(state, state_path)
    return state


def cmd_status(*, state_path: Path = DEFAULT_STATE_PATH) -> dict[str, Any]:
    """Return printable scheduler status snapshot."""
    state = load_state(state_path)
    last_warning = state.warnings[-1] if state.warnings else None
    return {
        "next_historical_date": state.next_historical_date,
        "completed_days": len(state.completed_dates),
        "completed_dates": state.completed_dates,
        "failed_days": len(state.failed_dates),
        "failed_dates": state.failed_dates,
        "last_run_at": state.last_run_at,
        "last_warning": last_warning,
        "latest_report": state.last_report_path,
        "total_historical_files_downloaded": state.total_historical_files_downloaded,
        "total_historical_days_completed": state.total_historical_days_completed,
        "coins": state.coins,
        "historical_start_date": state.historical_start_date,
        "historical_end_date_target": state.historical_end_date_target,
        "historical_start_hour": state.historical_start_hour,
        "historical_end_hour": state.historical_end_hour,
        "next_recommended_action": state.next_recommended_action,
        "completed_batches": len(state.completed_batches),
        "failed_batches": len(state.failed_batches),
    }


def _advance_date_if_complete(state: SchedulerState) -> None:
    current = state.next_historical_date
    if not current:
        return
    if not _all_hours_completed_for_date(state, current):
        return

    if current not in state.completed_dates:
        state.completed_dates.append(current)
        state.total_historical_days_completed += 1

    next_date = _date_add_days(current, 1)
    if state.historical_end_date_target and next_date > state.historical_end_date_target:
        state.next_historical_date = None
        state.warnings.append(
            f"Historical backfill reached end target {state.historical_end_date_target}"
        )
    else:
        state.next_historical_date = next_date


def cmd_historical_backfill_next(
    *,
    coins: list[str] | None = None,
    hours: list[int] | None = None,
    date: str | None = None,
    max_files: int | None = None,
    allow_more: bool = False,
    all_time: bool = False,
    aws_profile: str | None = None,
    requester_pays: bool = False,
    data_root: Path = orchestrator.DEFAULT_DATA_ROOT,
    report_dir: Path = orchestrator.DEFAULT_REPORT_DIR,
    state_path: Path = DEFAULT_STATE_PATH,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    s3_client: Any | None = None,
    train_interval: str | None = None,
    horizon: int | None = None,
    source: str | None = None,
    defer_batch_completion: bool = False,
) -> SchedulerReport:
    """Run the next bounded historical batch from scheduler state."""
    orchestrator.reject_all_time(all_time=all_time)
    state = load_state(state_path)
    normalized_coins = orchestrator._normalize_coins(coins or state.coins)
    selected_hours = hours if hours is not None else DEFAULT_HOURS
    normalized_hours = orchestrator._normalize_hours(selected_hours)
    orchestrator.reject_all_time(hours=normalized_hours)

    resolved_max = orchestrator.resolve_orchestrator_max_files(max_files, allow_more=allow_more)

    report = SchedulerReport(
        workflows_run=["historical-backfill-next"],
        coins=normalized_coins,
        source=source or "historical",
        train_interval=train_interval,
        horizon=horizon,
    )

    if not state.next_historical_date and not date:
        report.warnings.append("No next historical date; backfill may be complete")
        report.next_recommended_action = "python3 bid_ask_research_scheduler.py status"
        json_path, _ = write_scheduler_report(report, runs_dir=runs_dir, label="backfill_skip")
        state.last_report_path = str(json_path)
        state.last_run_at = _utc_now_iso()
        save_state(state, state_path)
        return report

    target_date = validate_date_yyyymmdd(date or state.next_historical_date)
    batch_key = _batch_key(target_date, normalized_hours)

    existing_keys = {
        _batch_key(str(b.get("date")), [str(h) for h in b.get("hours") or []])
        for b in state.completed_batches
    }
    if batch_key in existing_keys:
        report.warnings.append(
            f"Batch already completed for {target_date} hours {normalized_hours}; skipping download"
        )
        report.historical_date = target_date
        report.historical_hours = normalized_hours
        report.next_recommended_action = (
            "python3 bid_ask_research_scheduler.py historical-backfill-next "
            f"--coins {' '.join(normalized_coins)}"
        )
        json_path, _ = write_scheduler_report(report, runs_dir=runs_dir, label="backfill_dedupe")
        state.last_report_path = str(json_path)
        state.last_run_at = _utc_now_iso()
        save_state(state, state_path)
        return report

    report.historical_date = target_date
    report.historical_hours = normalized_hours

    try:
        batch_report = orchestrator.cmd_historical_batch(
            date=target_date,
            hours=normalized_hours,
            coins=normalized_coins,
            max_files=resolved_max,
            allow_more=allow_more,
            all_time=all_time,
            aws_profile=aws_profile,
            requester_pays=requester_pays,
            data_root=data_root,
            report_dir=report_dir,
            s3_client=s3_client,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        failure_key = f"{target_date}:{','.join(normalized_hours)}"
        if failure_key not in state.failed_dates:
            state.failed_dates.append(failure_key)
        state.warnings.append(f"Historical batch failed for {target_date}: {exc}")
        report.failures.append(str(exc))
        report.next_recommended_action = (
            "Review failure, fix tooling/network, then rerun historical-backfill-next"
        )
        json_path, _ = write_scheduler_report(report, runs_dir=runs_dir, label="backfill_failed")
        state.last_report_path = str(json_path)
        state.last_run_at = _utc_now_iso()
        save_state(state, state_path)
        return report

    if batch_report.failures:
        failure_key = f"{target_date}:{','.join(normalized_hours)}"
        if failure_key not in state.failed_dates:
            state.failed_dates.append(failure_key)
        report.failures.extend(batch_report.failures)
        report.warnings.extend(batch_report.warnings)
        report.next_recommended_action = "Fix batch failures before advancing scheduler state"
        json_path, _ = write_scheduler_report(report, runs_dir=runs_dir, label="backfill_failed")
        state.last_report_path = str(json_path)
        state.last_run_at = _utc_now_iso()
        save_state(state, state_path)
        return report

    downloaded = _count_downloaded_files(batch_report)
    state.total_historical_files_downloaded += downloaded
    report.row_counts.update(batch_report.row_counts)
    report.warnings.extend(batch_report.warnings)
    report.details["orchestrator_batch"] = batch_report.to_dict()

    if not defer_batch_completion:
        state.completed_batches.append(
            {
                "date": target_date,
                "hours": normalized_hours,
                "completed_at": _utc_now_iso(),
            }
        )
        _advance_date_if_complete(state)
    report.next_recommended_action = (
        "python3 bid_ask_research_scheduler.py daily-cycle "
        f"--coins {' '.join(normalized_coins)}"
    )
    json_path, _ = write_scheduler_report(report, runs_dir=runs_dir, label="backfill")
    state.last_report_path = str(json_path)
    state.last_run_at = _utc_now_iso()
    save_state(state, state_path)
    return report


def cmd_daily_cycle(
    *,
    coins: list[str] | None = None,
    historical_hours: list[int] | None = None,
    date: str | None = None,
    max_files: int | None = None,
    allow_more: bool = False,
    aws_profile: str | None = None,
    requester_pays: bool = False,
    train_interval: str = "15s",
    horizon: int = 5,
    include_combined: bool = False,
    data_root: Path = orchestrator.DEFAULT_DATA_ROOT,
    orchestrator_report_dir: Path = orchestrator.DEFAULT_REPORT_DIR,
    state_path: Path = DEFAULT_STATE_PATH,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    daily_summary_dir: Path = bid_ask_daily_summary.DEFAULT_OUTPUT_DIR,
    s3_client: Any | None = None,
    defer_batch_completion: bool = False,
) -> SchedulerReport:
    """Run one bounded daily research cycle and update scheduler state."""
    state = load_state(state_path)
    normalized_coins = orchestrator._normalize_coins(coins or state.coins)

    report = SchedulerReport(
        workflows_run=["daily-cycle"],
        coins=normalized_coins,
        source="combined" if include_combined else "historical",
        train_interval=train_interval,
        horizon=horizon,
    )

    backfill_report = cmd_historical_backfill_next(
        coins=normalized_coins,
        hours=historical_hours,
        date=date,
        max_files=max_files,
        allow_more=allow_more,
        aws_profile=aws_profile,
        requester_pays=requester_pays,
        data_root=data_root,
        report_dir=orchestrator_report_dir,
        state_path=state_path,
        runs_dir=runs_dir,
        s3_client=s3_client,
        train_interval=train_interval,
        horizon=horizon,
        source="historical",
        defer_batch_completion=defer_batch_completion,
    )
    report.workflows_run.append("historical-backfill-next")
    report.historical_date = backfill_report.historical_date
    report.historical_hours = backfill_report.historical_hours
    report.row_counts.update(backfill_report.row_counts)
    report.warnings.extend(backfill_report.warnings)
    report.failures.extend(backfill_report.failures)
    report.details["historical_backfill"] = backfill_report.to_dict()

    if backfill_report.failures:
        report.next_recommended_action = backfill_report.next_recommended_action
        json_path, _ = write_scheduler_report(report, runs_dir=runs_dir, label="daily_cycle")
        state = load_state(state_path)
        state.last_report_path = str(json_path)
        state.last_run_at = _utc_now_iso()
        save_state(state, state_path)
        return report

    validate_report = orchestrator.cmd_resample_validate(
        coins=normalized_coins,
        source="historical",
        data_root=data_root,
        report_dir=orchestrator_report_dir,
    )
    report.workflows_run.append("resample-validate")
    report.row_counts.update(validate_report.row_counts)
    report.warnings.extend(validate_report.warnings)
    report.failures.extend(validate_report.failures)
    report.details["resample_validate"] = validate_report.to_dict()

    train_report = orchestrator.cmd_train_evaluate(
        coins=normalized_coins,
        source="historical",
        interval=train_interval,
        horizon=horizon,
        data_root=data_root,
        report_dir=orchestrator_report_dir,
    )
    report.workflows_run.append("train-evaluate-historical")
    report.row_counts.update(train_report.row_counts)
    report.warnings.extend(train_report.warnings)
    report.failures.extend(train_report.failures)
    _apply_train_metadata(report, train_report)
    report.details["train_evaluate_historical"] = train_report.to_dict()

    state = load_state(state_path)
    state.last_training_at = _utc_now_iso()

    if include_combined:
        combined_report = orchestrator.cmd_combined_dataset(
            coins=normalized_coins,
            intervals=["1s", "15s", "1m"],
            data_root=data_root,
            report_dir=orchestrator_report_dir,
        )
        report.workflows_run.append("combined-dataset")
        report.row_counts.update(combined_report.row_counts)
        report.warnings.extend(combined_report.warnings)
        report.details["combined_dataset"] = combined_report.to_dict()

        combined_train = orchestrator.cmd_train_evaluate(
            coins=normalized_coins,
            source="combined",
            interval=train_interval,
            horizon=horizon,
            data_root=data_root,
            report_dir=orchestrator_report_dir,
        )
        report.workflows_run.append("train-evaluate-combined")
        report.row_counts.update(combined_train.row_counts)
        report.warnings.extend(combined_train.warnings)
        _apply_train_metadata(report, combined_train)
        report.details["train_evaluate_combined"] = combined_train.to_dict()

    summary = bid_ask_daily_summary.build_daily_summary(
        normalized_coins,
        data_root=data_root,
    )
    json_summary_path = bid_ask_daily_summary.write_summary_json(summary, daily_summary_dir)
    md_summary_path = bid_ask_daily_summary.write_summary_markdown(summary, daily_summary_dir)
    report.workflows_run.append("daily-summary")
    report.details["daily_summary"] = {
        "json_path": str(json_summary_path),
        "markdown_path": str(md_summary_path),
    }
    report.evaluation_paths.extend([str(json_summary_path), str(md_summary_path)])

    report.next_recommended_action = (
        "python3 bid_ask_research_scheduler.py status"
    )
    json_path, md_path = write_scheduler_report(report, runs_dir=runs_dir, label="daily_cycle")
    state.last_report_path = str(json_path)
    state.last_run_at = _utc_now_iso()
    if ARCHIVE_AVAILABILITY_WARNING not in state.warnings:
        state.warnings.append(ARCHIVE_AVAILABILITY_WARNING)
    save_state(state, state_path)
    report.details["scheduler_report_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    return report


def cmd_run_batch_series(
    *,
    date: str,
    start_hour: int,
    end_hour: int,
    coins: list[str] | None = None,
    chunk_size_hours: int = DEFAULT_CHUNK_SIZE_HOURS,
    allow_more: bool = False,
    continue_on_error: bool = False,
    aws_profile: str | None = None,
    requester_pays: bool = False,
    train_interval: str = "15s",
    horizon: int = 5,
    include_combined: bool = False,
    max_files: int | None = None,
    data_root: Path = orchestrator.DEFAULT_DATA_ROOT,
    orchestrator_report_dir: Path = orchestrator.DEFAULT_REPORT_DIR,
    state_path: Path = DEFAULT_STATE_PATH,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    daily_summary_dir: Path = bid_ask_daily_summary.DEFAULT_OUTPUT_DIR,
    s3_client: Any | None = None,
) -> BatchSeriesReport:
    """Run multiple bounded daily-cycle chunks sequentially for one date."""
    orchestrator.reject_all_time(all_time=False)
    validate_batch_series_hours(
        start_hour=start_hour,
        end_hour=end_hour,
        chunk_size_hours=chunk_size_hours,
        allow_more=allow_more,
    )

    target_date = validate_date_yyyymmdd(date)
    state = load_state(state_path)
    normalized_coins = orchestrator._normalize_coins(coins or state.coins)
    hour_chunks = split_hour_range_into_chunks(start_hour, end_hour, chunk_size_hours)

    series_report = BatchSeriesReport(
        date=target_date,
        coins=normalized_coins,
        chunks=hour_chunks,
        train_interval=train_interval,
        horizon=horizon,
    )

    stopped_early = False
    for chunk_hours in hour_chunks:
        if stopped_early:
            break

        state = load_state(state_path)
        if _chunk_hours_completed(state, target_date, chunk_hours):
            series_report.chunks_skipped += 1
            series_report.warnings.append(
                f"Chunk hours {', '.join(chunk_hours)} already completed for "
                f"{target_date}; skipping"
            )
            continue

        chunk_report = cmd_daily_cycle(
            coins=normalized_coins,
            historical_hours=[int(h) for h in chunk_hours],
            date=target_date,
            max_files=max_files,
            allow_more=allow_more,
            aws_profile=aws_profile,
            requester_pays=requester_pays,
            train_interval=train_interval,
            horizon=horizon,
            include_combined=include_combined,
            data_root=data_root,
            orchestrator_report_dir=orchestrator_report_dir,
            state_path=state_path,
            runs_dir=runs_dir,
            daily_summary_dir=daily_summary_dir,
            s3_client=s3_client,
        )

        chunk_paths = (chunk_report.details or {}).get("scheduler_report_paths") or {}
        per_chunk_entry: dict[str, Any] = {
            "hours": chunk_hours,
            "json": chunk_paths.get("json"),
            "markdown": chunk_paths.get("markdown"),
        }
        series_report.per_chunk_report_paths.append(per_chunk_entry)

        if chunk_report.failures:
            series_report.chunks_failed += 1
            series_report.failures.extend(chunk_report.failures)
            series_report.warnings.extend(chunk_report.warnings)
            if not continue_on_error:
                stopped_early = True
                series_report.warnings.append(
                    f"Stopped batch series after failed chunk hours {', '.join(chunk_hours)}"
                )
        else:
            series_report.chunks_completed += 1
            series_report.warnings.extend(chunk_report.warnings)

    if series_report.chunks_failed:
        series_report.next_recommended_action = (
            "Review failures, fix tooling/network, then rerun run-batch-series "
            f"--date {target_date} --start-hour {start_hour} --end-hour {end_hour}"
        )
    elif series_report.chunks_completed == len(hour_chunks):
        series_report.next_recommended_action = "python3 bid_ask_research_scheduler.py status"
    elif series_report.chunks_skipped == len(hour_chunks):
        series_report.next_recommended_action = (
            f"All chunks already completed for {target_date}; advance to next date or hours"
        )
    else:
        series_report.next_recommended_action = (
            "python3 bid_ask_research_scheduler.py run-batch-series "
            f"--date {target_date} --coins {' '.join(normalized_coins)} "
            f"--start-hour {start_hour} --end-hour {end_hour}"
        )

    json_path, md_path = write_batch_series_report(series_report, runs_dir=runs_dir)
    state = load_state(state_path)
    state.last_report_path = str(json_path)
    state.last_run_at = _utc_now_iso()
    save_state(state, state_path)
    return series_report


def cmd_live_long_session(
    *,
    coins: list[str],
    duration_seconds: float,
    allow_long: bool = False,
    data_root: Path = orchestrator.DEFAULT_DATA_ROOT,
    report_dir: Path = orchestrator.DEFAULT_REPORT_DIR,
    state_path: Path = DEFAULT_STATE_PATH,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    run_recorder_fn: Any | None = None,
) -> SchedulerReport:
    """Run a finite live recorder session through the orchestrator."""
    duration = orchestrator.validate_live_duration(duration_seconds, allow_long=allow_long)
    normalized_coins = orchestrator._normalize_coins(coins)

    report = SchedulerReport(
        workflows_run=["live-long-session"],
        coins=normalized_coins,
        live_session_duration=duration,
    )

    live_report = orchestrator.cmd_live_session(
        coins=normalized_coins,
        duration_seconds=duration,
        allow_long=allow_long,
        data_root=data_root,
        report_dir=report_dir,
        run_recorder_fn=run_recorder_fn,
    )
    report.warnings.extend(live_report.warnings)
    report.failures.extend(live_report.failures)
    report.details["live_session"] = live_report.to_dict()

    state = load_state(state_path)
    state.last_live_session_at = _utc_now_iso()
    report.next_recommended_action = (
        "python3 bid_ask_research_orchestrator.py resample-validate "
        f"--source local --coins {' '.join(normalized_coins)}"
    )
    json_path, _ = write_scheduler_report(report, runs_dir=runs_dir, label="live_session")
    state.last_report_path = str(json_path)
    state.last_run_at = _utc_now_iso()
    save_state(state, state_path)
    return report


def cmd_auto_backfill_next(
    *,
    coins: list[str] | None = None,
    chunk_size_hours: int = DEFAULT_CHUNK_SIZE_HOURS,
    max_files: int | None = None,
    allow_more: bool = False,
    aws_profile: str | None = None,
    requester_pays: bool = False,
    train_interval: str = "15s",
    horizon: int = 5,
    include_combined: bool = False,
    data_root: Path = orchestrator.DEFAULT_DATA_ROOT,
    orchestrator_report_dir: Path = orchestrator.DEFAULT_REPORT_DIR,
    state_path: Path = DEFAULT_STATE_PATH,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    daily_summary_dir: Path = bid_ask_daily_summary.DEFAULT_OUTPUT_DIR,
    s3_client: Any | None = None,
) -> SchedulerReport:
    """Run one bounded historical chunk: backfill, validate, train, update state, exit."""
    orchestrator.reject_all_time(all_time=False)
    state = load_state(state_path)
    if state.next_historical_date and _all_hours_completed_for_date(
        state, state.next_historical_date
    ):
        _advance_date_if_complete(state)
        save_state(state, state_path)
        state = load_state(state_path)
    normalized_coins = orchestrator._normalize_coins(coins or state.coins)

    report = SchedulerReport(
        workflows_run=["auto-backfill-next"],
        coins=normalized_coins,
        source="historical",
        train_interval=train_interval,
        horizon=horizon,
        details={"chunk_size_hours": chunk_size_hours},
    )

    next_chunk = find_next_incomplete_chunk(state, chunk_size_hours)
    if next_chunk is None:
        report.warnings.append(AUTO_BACKFILL_COMPLETE_MESSAGE)
        report.next_recommended_action = "python3 bid_ask_research_scheduler.py status"
        json_path, md_path = write_scheduler_report(
            report, runs_dir=runs_dir, label="auto_backfill_complete"
        )
        state.next_recommended_action = report.next_recommended_action
        state.last_report_path = str(json_path)
        state.last_run_at = _utc_now_iso()
        save_state(state, state_path)
        report.details["scheduler_report_paths"] = {"json": str(json_path), "markdown": str(md_path)}
        return report

    target_date, chunk_hours = next_chunk
    report.historical_date = target_date
    report.historical_hours = chunk_hours

    cycle_report = cmd_daily_cycle(
        coins=normalized_coins,
        historical_hours=[int(h) for h in chunk_hours],
        date=target_date,
        max_files=max_files,
        allow_more=allow_more,
        aws_profile=aws_profile,
        requester_pays=requester_pays,
        train_interval=train_interval,
        horizon=horizon,
        include_combined=include_combined,
        data_root=data_root,
        orchestrator_report_dir=orchestrator_report_dir,
        state_path=state_path,
        runs_dir=runs_dir,
        daily_summary_dir=daily_summary_dir,
        s3_client=s3_client,
        defer_batch_completion=True,
    )

    report.workflows_run.extend(cycle_report.workflows_run)
    report.row_counts.update(cycle_report.row_counts)
    report.warnings.extend(cycle_report.warnings)
    report.failures.extend(cycle_report.failures)
    report.details.update(cycle_report.details)
    report.training_reports.extend(cycle_report.training_reports)
    report.model_artifact_paths.extend(cycle_report.model_artifact_paths)
    report.evaluation_report_paths.extend(cycle_report.evaluation_report_paths)
    report.filter_report_paths.extend(cycle_report.filter_report_paths)
    report.model_paths.extend(cycle_report.model_paths)
    report.evaluation_paths.extend(cycle_report.evaluation_paths)

    state = load_state(state_path)
    cycle_paths = (cycle_report.details or {}).get("scheduler_report_paths") or {}
    json_path_str = cycle_paths.get("json")
    md_path_str = cycle_paths.get("markdown")

    if _cycle_succeeded(cycle_report):
        batch_entry = {
            "date": target_date,
            "hours": chunk_hours,
            "completed_at": _utc_now_iso(),
            "report_path": json_path_str,
        }
        state.completed_batches.append(batch_entry)
        backfill_details = (cycle_report.details or {}).get("historical_backfill") or {}
        orch_batch = (backfill_details.get("details") or {}).get("orchestrator_batch") or {}
        downloaded = _count_downloaded_files(
            orchestrator.OrchestratorReport(details=orch_batch)
        )
        state.total_historical_files_downloaded += downloaded
        _advance_date_if_complete(state)
        report.next_recommended_action = (
            "python3 bid_ask_research_scheduler.py auto-backfill-next "
            f"--coins {' '.join(normalized_coins)} "
            f"--chunk-size-hours {chunk_size_hours} "
            f"--train-interval {train_interval} --horizon {horizon}"
        )
        if state.next_historical_date is None:
            report.next_recommended_action = "python3 bid_ask_research_scheduler.py status"
    else:
        reason_parts = list(cycle_report.failures)
        reason_parts.extend(
            w for w in cycle_report.warnings if "validation failed" in w.lower()
        )
        reason = "; ".join(reason_parts) or "daily cycle did not pass validation"
        failed_entry = {
            "date": target_date,
            "hours": chunk_hours,
            "failed_at": _utc_now_iso(),
            "reason": reason,
            "report_path": json_path_str,
        }
        state.failed_batches.append(failed_entry)
        report.next_recommended_action = (
            "Review failure report, fix data/tooling, then rerun auto-backfill-next"
        )

    state.next_recommended_action = report.next_recommended_action
    auto_json_path, auto_md_path = write_scheduler_report(
        report, runs_dir=runs_dir, label="auto_backfill"
    )
    state.last_report_path = str(auto_json_path)
    state.last_run_at = _utc_now_iso()
    save_state(state, state_path)
    report.details["scheduler_report_paths"] = {
        "json": str(auto_json_path),
        "markdown": str(auto_md_path),
        "daily_cycle_json": json_path_str,
        "daily_cycle_markdown": md_path_str,
    }
    return report


def cmd_repair_state_from_existing_data(
    *,
    date: str,
    start_hour: int,
    end_hour: int,
    coins: list[str] | None = None,
    chunk_size_hours: int = DEFAULT_CHUNK_SIZE_HOURS,
    data_root: Path = orchestrator.DEFAULT_DATA_ROOT,
    state_path: Path = DEFAULT_STATE_PATH,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> SchedulerReport:
    """Mark completed chunks inferred from existing historical CSV data (no download/train)."""
    target_date = validate_date_yyyymmdd(date)
    validate_batch_series_hours(
        start_hour=start_hour,
        end_hour=end_hour,
        chunk_size_hours=chunk_size_hours,
        allow_more=True,
    )
    state = load_state(state_path)
    normalized_coins = orchestrator._normalize_coins(coins or state.coins)
    hour_chunks = split_hour_range_into_chunks(start_hour, end_hour, chunk_size_hours)

    report = SchedulerReport(
        workflows_run=["repair-state-from-existing-data"],
        coins=normalized_coins,
        historical_date=target_date,
        source="historical",
        warnings=[REPAIR_INFERRED_WARNING],
        details={"chunk_size_hours": chunk_size_hours, "chunks_repaired": []},
    )

    repaired = 0
    skipped = 0
    failed = 0

    for chunk_hours in hour_chunks:
        if _chunk_hours_completed(state, target_date, chunk_hours):
            skipped += 1
            report.warnings.append(
                f"Chunk hours {', '.join(chunk_hours)} already completed; skipping"
            )
            continue

        valid, issues = validate_chunk_historical_data(
            date=target_date,
            hours=chunk_hours,
            coins=normalized_coins,
            data_root=data_root,
        )
        if not valid:
            failed += 1
            report.failures.extend(
                [f"Chunk {', '.join(chunk_hours)}: {issue}" for issue in issues]
            )
            continue

        batch_entry = {
            "date": target_date,
            "hours": chunk_hours,
            "completed_at": _utc_now_iso(),
            "report_path": None,
            "repaired_from_existing_data": True,
        }
        state.completed_batches.append(batch_entry)
        repaired += 1
        report.details["chunks_repaired"].append(chunk_hours)

    if target_date == state.next_historical_date:
        _advance_date_if_complete(state)

    report.row_counts["chunks_repaired"] = repaired
    report.row_counts["chunks_skipped"] = skipped
    report.row_counts["chunks_failed"] = failed

    if failed:
        report.next_recommended_action = (
            "Fix missing/invalid data for failed chunks, then rerun repair-state-from-existing-data"
        )
    elif repaired:
        report.next_recommended_action = (
            "python3 bid_ask_research_scheduler.py auto-backfill-next "
            f"--coins {' '.join(normalized_coins)}"
        )
    else:
        report.next_recommended_action = "python3 bid_ask_research_scheduler.py status"

    json_path, md_path = write_scheduler_report(report, runs_dir=runs_dir, label="repair_state")
    state.next_recommended_action = report.next_recommended_action
    state.last_report_path = str(json_path)
    state.last_run_at = _utc_now_iso()
    if REPAIR_INFERRED_WARNING not in state.warnings:
        state.warnings.append(REPAIR_INFERRED_WARNING)
    save_state(state, state_path)
    report.details["scheduler_report_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    return report


def cmd_configure_autopilot(
    *,
    enable: bool = False,
    disable: bool = False,
    pause_reason: str | None = None,
    resume: bool = False,
    reset_usage: bool = False,
    daily_chunk_limit: int | None = None,
    daily_file_limit: int | None = None,
    ramp_enabled: bool | None = None,
    ramp_every_successful_days: int | None = None,
    ramp_chunk_increment: int | None = None,
    max_daily_chunk_limit: int | None = None,
    max_daily_file_limit: int | None = None,
    state_path: Path = DEFAULT_STATE_PATH,
) -> AutopilotState:
    """Create or update autopilot config in state without downloading or training."""
    state = load_state(state_path)
    autopilot = state.autopilot

    if enable and disable:
        raise ValueError("pass --enable or --disable, not both")
    if pause_reason and resume:
        raise ValueError("pass --pause or --resume, not both")

    if reset_usage:
        _reset_autopilot_daily_counters(autopilot, _utc_today_yyyymmdd())

    if enable:
        autopilot.enabled = True
        autopilot.paused = False
        autopilot.pause_reason = None
        if autopilot.started_at is None:
            autopilot.started_at = _utc_now_iso()
    if disable:
        autopilot.enabled = False
    if pause_reason:
        autopilot.paused = True
        autopilot.pause_reason = pause_reason
    if resume:
        autopilot.paused = False
        autopilot.pause_reason = None
        autopilot.failed_runs_in_row = 0

    if daily_chunk_limit is not None:
        if daily_chunk_limit < 1:
            raise ValueError("daily-chunk-limit must be at least 1")
        autopilot.daily_chunk_limit = daily_chunk_limit
    if daily_file_limit is not None:
        if daily_file_limit < 1:
            raise ValueError("daily-file-limit must be at least 1")
        autopilot.daily_file_limit = daily_file_limit
    if ramp_enabled is not None:
        autopilot.ramp_enabled = ramp_enabled
    if ramp_every_successful_days is not None:
        if ramp_every_successful_days < 1:
            raise ValueError("ramp-every-successful-days must be at least 1")
        autopilot.ramp_every_successful_days = ramp_every_successful_days
    if ramp_chunk_increment is not None:
        if ramp_chunk_increment < 1:
            raise ValueError("ramp-chunk-increment must be at least 1")
        autopilot.ramp_chunk_increment = ramp_chunk_increment
    if max_daily_chunk_limit is not None:
        if max_daily_chunk_limit < 1:
            raise ValueError("max-daily-chunk-limit must be at least 1")
        autopilot.max_daily_chunk_limit = max_daily_chunk_limit
    if max_daily_file_limit is not None:
        if max_daily_file_limit < 1:
            raise ValueError("max-daily-file-limit must be at least 1")
        autopilot.max_daily_file_limit = max_daily_file_limit

    if autopilot.daily_chunk_limit > autopilot.max_daily_chunk_limit:
        raise ValueError(
            f"daily_chunk_limit {autopilot.daily_chunk_limit} exceeds "
            f"max_daily_chunk_limit {autopilot.max_daily_chunk_limit}"
        )
    if autopilot.daily_file_limit > autopilot.max_daily_file_limit:
        raise ValueError(
            f"daily_file_limit {autopilot.daily_file_limit} exceeds "
            f"max_daily_file_limit {autopilot.max_daily_file_limit}"
        )

    state.autopilot = autopilot
    save_state(state, state_path)
    return autopilot


def cmd_autopilot_status(
    *,
    state_path: Path = DEFAULT_STATE_PATH,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> dict[str, Any]:
    """Return printable autopilot status snapshot."""
    state = load_state(state_path)
    autopilot = state.autopilot
    today = _utc_today_yyyymmdd()
    if _reconcile_autopilot_daily_usage(
        autopilot,
        state,
        today=today,
        runs_dir=runs_dir,
    ):
        state.autopilot = autopilot
        save_state(state, state_path)
    return {
        "enabled": autopilot.enabled,
        "paused": autopilot.paused,
        "pause_reason": autopilot.pause_reason,
        "started_at": autopilot.started_at,
        "last_run_date": autopilot.last_run_date,
        "daily_chunk_limit": autopilot.daily_chunk_limit,
        "daily_file_limit": autopilot.daily_file_limit,
        "runs_today": autopilot.runs_today,
        "chunks_completed_today": autopilot.chunks_completed_today,
        "files_downloaded_today": autopilot.files_downloaded_today,
        "ramp_enabled": autopilot.ramp_enabled,
        "ramp_every_successful_days": autopilot.ramp_every_successful_days,
        "ramp_chunk_increment": autopilot.ramp_chunk_increment,
        "max_daily_chunk_limit": autopilot.max_daily_chunk_limit,
        "max_daily_file_limit": autopilot.max_daily_file_limit,
        "successful_days_in_row": autopilot.successful_days_in_row,
        "failed_runs_in_row": autopilot.failed_runs_in_row,
        "next_recommended_action": _autopilot_next_recommended_action(state, autopilot),
        "latest_report": autopilot.last_autopilot_report_path,
    }


def _finalize_autopilot_report(
    report: AutopilotReport,
    state: SchedulerState,
    *,
    state_path: Path,
    runs_dir: Path,
) -> AutopilotReport:
    report.autopilot_snapshot = state.autopilot.to_dict()
    report.next_recommended_action = _autopilot_next_recommended_action(state, state.autopilot)
    json_path, md_path = write_autopilot_report(report, runs_dir=runs_dir, label="autopilot")
    state.autopilot.last_autopilot_report_path = str(json_path)
    save_state(state, state_path)
    report.details["scheduler_report_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    return report


def cmd_autopilot_tick(
    *,
    coins: list[str] | None = None,
    chunk_size_hours: int = DEFAULT_CHUNK_SIZE_HOURS,
    max_files: int | None = None,
    allow_more: bool = False,
    aws_profile: str | None = None,
    requester_pays: bool = False,
    train_interval: str = "15s",
    horizon: int = 5,
    include_combined: bool = False,
    data_root: Path = orchestrator.DEFAULT_DATA_ROOT,
    orchestrator_report_dir: Path = orchestrator.DEFAULT_REPORT_DIR,
    state_path: Path = DEFAULT_STATE_PATH,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    daily_summary_dir: Path = bid_ask_daily_summary.DEFAULT_OUTPUT_DIR,
    s3_client: Any | None = None,
    today_override: str | None = None,
) -> AutopilotReport:
    """Run one bounded autopilot tick: check limits, auto-backfill-next, update state, exit."""
    orchestrator.reject_all_time(all_time=False)
    state = load_state(state_path)
    autopilot = state.autopilot
    normalized_coins = orchestrator._normalize_coins(coins or state.coins)
    today = today_override or _utc_today_yyyymmdd()
    requested_files = _estimate_chunk_files(len(normalized_coins), chunk_size_hours)

    report = AutopilotReport(
        coins=normalized_coins,
        chunk_size_hours=chunk_size_hours,
        estimated_files=requested_files,
        requested_files=requested_files,
        train_interval=train_interval,
        horizon=horizon,
    )

    if not autopilot.enabled:
        report.action = "disabled"
        report.warnings.append(AUTOPILOT_DISABLED_MESSAGE)
        return _finalize_autopilot_report(report, state, state_path=state_path, runs_dir=runs_dir)

    if autopilot.paused:
        report.action = "paused"
        report.warnings.append(
            f"{AUTOPILOT_PAUSED_MESSAGE} Reason: {autopilot.pause_reason or 'unspecified'}"
        )
        return _finalize_autopilot_report(report, state, state_path=state_path, runs_dir=runs_dir)

    if _reconcile_autopilot_daily_usage(
        autopilot,
        state,
        today=today,
        runs_dir=runs_dir,
    ):
        state.autopilot = autopilot
        save_state(state, state_path)

    _populate_autopilot_report_limits(report, autopilot, requested_files=requested_files)

    limit_reason = _autopilot_would_exceed_daily_limits(autopilot, requested_files)
    if limit_reason:
        report.action = "limit_skipped"
        report.warnings.append(limit_reason)
        state.autopilot = autopilot
        return _finalize_autopilot_report(report, state, state_path=state_path, runs_dir=runs_dir)

    state.autopilot = autopilot
    save_state(state, state_path)

    backfill_report = cmd_auto_backfill_next(
        coins=normalized_coins,
        chunk_size_hours=chunk_size_hours,
        max_files=max_files,
        allow_more=allow_more,
        aws_profile=aws_profile,
        requester_pays=requester_pays,
        train_interval=train_interval,
        horizon=horizon,
        include_combined=include_combined,
        data_root=data_root,
        orchestrator_report_dir=orchestrator_report_dir,
        state_path=state_path,
        runs_dir=runs_dir,
        daily_summary_dir=daily_summary_dir,
        s3_client=s3_client,
    )

    state = load_state(state_path)
    autopilot = state.autopilot
    report.workflows_run = list(backfill_report.workflows_run)
    report.historical_date = backfill_report.historical_date
    report.historical_hours = backfill_report.historical_hours
    report.warnings.extend(backfill_report.warnings)
    report.failures.extend(backfill_report.failures)
    report.details["auto_backfill"] = backfill_report.to_dict()

    _populate_autopilot_report_limits(report, autopilot, requested_files=requested_files)

    if any(AUTO_BACKFILL_COMPLETE_MESSAGE in w for w in backfill_report.warnings):
        report.action = "backfill_complete"
        report.next_recommended_action = backfill_report.next_recommended_action
        state.autopilot = autopilot
        return _finalize_autopilot_report(report, state, state_path=state_path, runs_dir=runs_dir)

    chunk_succeeded = _cycle_succeeded(backfill_report) and not backfill_report.failures

    if chunk_succeeded:
        autopilot.runs_today += 1
        autopilot.chunks_completed_today += 1
        backfill_details = (backfill_report.details or {}).get("historical_backfill") or {}
        if isinstance(backfill_details, dict):
            orch_batch = (backfill_details.get("details") or {}).get("orchestrator_batch") or {}
            actual_files = _count_downloaded_files(
                orchestrator.OrchestratorReport(details=orch_batch if isinstance(orch_batch, dict) else {})
            )
        else:
            actual_files = 0
        files_added = actual_files if actual_files > 0 else requested_files
        autopilot.files_downloaded_today += files_added
        report.files_downloaded = files_added
        autopilot.failed_runs_in_row = 0
        autopilot.last_run_date = today
        _populate_autopilot_report_limits(report, autopilot, requested_files=requested_files)
        _record_autopilot_success_day(autopilot, today)
        ramped = _maybe_ramp_autopilot_limits(
            autopilot,
            chunk_size_hours=chunk_size_hours,
            num_coins=len(normalized_coins),
        )
        report.action = "chunk_completed"
        if ramped:
            report.warnings.append(
                f"Ramp applied: daily_chunk_limit={autopilot.daily_chunk_limit}, "
                f"daily_file_limit={autopilot.daily_file_limit}"
            )
        report.next_recommended_action = backfill_report.next_recommended_action
    else:
        autopilot.failed_runs_in_row += 1
        autopilot.last_run_date = today
        report.action = "failed"
        report.failures.extend(backfill_report.failures or ["chunk did not pass validation"])
        if autopilot.failed_runs_in_row >= AUTOPILOT_FAILURE_PAUSE_THRESHOLD:
            autopilot.paused = True
            autopilot.pause_reason = (
                f"Paused after {autopilot.failed_runs_in_row} consecutive failed autopilot runs"
            )
            report.warnings.append(autopilot.pause_reason)
        report.next_recommended_action = (
            "Review failure report, fix data/tooling, then "
            "configure-autopilot --resume and rerun autopilot-tick"
        )

    state.autopilot = autopilot
    return _finalize_autopilot_report(report, state, state_path=state_path, runs_dir=runs_dir)


def cmd_write_launchd_template(
    *,
    run_time: str = "02:00",
    coins: list[str],
    aws_profile: str | None = None,
    requester_pays: bool = False,
    chunk_size_hours: int = DEFAULT_CHUNK_SIZE_HOURS,
    train_interval: str = "15s",
    horizon: int = 5,
    state_dir: Path = DEFAULT_STATE_DIR,
    project_root: Path | None = None,
    autopilot: bool = False,
    interval_hours: int | None = None,
) -> tuple[Path, Path]:
    """Write macOS launchd plist template and setup instructions (does not install)."""
    normalized_coins = orchestrator._normalize_coins(coins)
    root = project_root or Path.cwd()
    python_path = sys.executable
    scheduler_script = root / "bid_ask_research_scheduler.py"

    coin_args = " ".join(normalized_coins)
    subcommand = "autopilot-tick" if autopilot else "auto-backfill-next"
    cmd_parts = [
        python_path,
        str(scheduler_script),
        subcommand,
        "--coins",
        *normalized_coins,
        "--chunk-size-hours",
        str(chunk_size_hours),
        "--train-interval",
        train_interval,
        "--horizon",
        str(horizon),
    ]
    if aws_profile:
        cmd_parts.extend(["--aws-profile", aws_profile])
    if requester_pays:
        cmd_parts.append("--requester-pays")

    plist_name = LAUNCHD_PLIST_NAME_AUTOPILOT if autopilot else LAUNCHD_PLIST_NAME
    label = "com.hyperliquid.research.autopilot" if autopilot else "com.hyperliquid.research.daily"
    plist_path = state_dir / plist_name
    state_dir.mkdir(parents=True, exist_ok=True)

    if interval_hours is not None:
        if interval_hours < 1:
            raise ValueError("interval-hours must be at least 1")
        schedule_block = f"""    <key>StartInterval</key>
    <integer>{interval_hours * 3600}</integer>"""
        schedule_description = f"every **{interval_hours}** hours"
    else:
        hour, minute = run_time.split(":")
        schedule_block = f"""    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{int(hour)}</integer>
        <key>Minute</key>
        <integer>{int(minute)}</integer>
    </dict>"""
        schedule_description = f"**{run_time}** local time daily"

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
{''.join(f'        <string>{part}</string>\n' for part in cmd_parts)}    </array>
    <key>WorkingDirectory</key>
    <string>{root}</string>
{schedule_block}
    <key>StandardOutPath</key>
    <string>{state_dir / 'launchd_stdout.log'}</string>
    <key>StandardErrorPath</key>
    <string>{state_dir / 'launchd_stderr.log'}</string>
</dict>
</plist>
"""
    plist_path.write_text(plist_content, encoding="utf-8")

    mode_name = "autopilot-tick" if autopilot else "auto-backfill-next"
    setup_md = state_dir / (LAUNCHD_SETUP_MD if not autopilot else "launchd_autopilot_setup.md")
    setup_lines = [
        "# macOS launchd Setup for Daily Research Scheduler",
        "",
        f"This template runs `{mode_name}` once per schedule tick. "
        "Each invocation processes **one bounded chunk** and exits. "
        "It is **not installed automatically**.",
        "",
    ]
    if autopilot:
        setup_lines.extend(
            [
                "Autopilot mode enforces daily chunk/file limits and gradual ramp-up. "
                "Configure autopilot before loading the plist:",
                "",
                "```bash",
                "python3 bid_ask_research_scheduler.py configure-autopilot \\",
                "  --enable \\",
                "  --daily-chunk-limit 1 \\",
                "  --daily-file-limit 9 \\",
                "  --ramp-enabled \\",
                "  --ramp-every-successful-days 2 \\",
                "  --ramp-chunk-increment 1 \\",
                "  --max-daily-chunk-limit 4 \\",
                "  --max-daily-file-limit 36",
                "```",
                "",
                "Set up **AWS Budget alerts** before enabling scheduled downloads.",
                "",
            ]
        )

    setup_lines.extend(
        [
            "## Files",
            "",
            f"- Plist template: `{plist_path}`",
            f"- Scheduler state: `{state_dir / 'state.json'}`",
            f"- Run reports: `{state_dir / 'runs/'}`",
            "",
            "## Prerequisites",
            "",
            "1. Initialize scheduler state:",
            "",
            "```bash",
            "python3 bid_ask_research_scheduler.py init-state \\",
            "  --historical-start-date YYYYMMDD \\",
            "  --historical-end-date YYYYMMDD \\",
            f"  --coins {coin_args} \\",
            "  --historical-start-hour 0 \\",
            "  --historical-end-hour 23",
            "```",
            "",
            f"2. Verify a manual {mode_name} succeeds:",
            "",
            "```bash",
            f"python3 bid_ask_research_scheduler.py {mode_name} \\",
            f"  --coins {coin_args} \\",
            f"  --chunk-size-hours {chunk_size_hours} \\",
            f"  --train-interval {train_interval} \\",
            f"  --horizon {horizon}",
            "```",
            "",
            "## Install (manual)",
            "",
            "```bash",
            f"cp {plist_path} ~/Library/LaunchAgents/",
            f"launchctl load ~/Library/LaunchAgents/{plist_name}",
            "```",
            "",
            "## Uninstall",
            "",
            "```bash",
            f"launchctl unload ~/Library/LaunchAgents/{plist_name}",
            f"rm ~/Library/LaunchAgents/{plist_name}",
            "```",
            "",
            "## Safety",
            "",
            "- Research-only, paper-only; no orders or execution endpoints.",
            "- Each scheduled run processes one bounded chunk and exits (no infinite loop).",
            "- Archive availability is not guaranteed.",
            "- Use AWS Budget alerts; requester-pays downloads incur charges.",
            "",
            f"Schedule: {schedule_description}.",
            f"Chunk size: **{chunk_size_hours}** hours; train interval: **{train_interval}**; horizon: **{horizon}**.",
        ]
    )
    setup_md.write_text("\n".join(setup_lines) + "\n", encoding="utf-8")
    return plist_path, setup_md


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bid/ask research scheduler (read-only, paper-only, bounded backfill)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_state = subparsers.add_parser("init-state", help="Initialize scheduler state")
    init_state.add_argument("--historical-start-date", required=True, help="YYYYMMDD")
    init_state.add_argument("--historical-end-date", default=None, help="YYYYMMDD or omit")
    init_state.add_argument("--coins", nargs="+", required=True)
    init_state.add_argument("--reset", action="store_true", help="Overwrite existing state")
    init_state.add_argument(
        "--confirm-large-range",
        action="store_true",
        help=f"Allow ranges larger than {LARGE_RANGE_DAYS} days",
    )
    init_state.add_argument(
        "--historical-start-hour",
        type=int,
        default=DEFAULT_HISTORICAL_START_HOUR,
        help="First UTC hour to backfill each day (default 0)",
    )
    init_state.add_argument(
        "--historical-end-hour",
        type=int,
        default=DEFAULT_HISTORICAL_END_HOUR,
        help="Last UTC hour to backfill each day (default 23)",
    )
    init_state.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)

    status = subparsers.add_parser("status", help="Print scheduler state")
    status.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)

    backfill = subparsers.add_parser(
        "historical-backfill-next",
        help="Run next bounded historical batch",
    )
    backfill.add_argument("--coins", nargs="+", default=None)
    backfill.add_argument("--hours", nargs="+", type=int, default=None)
    backfill.add_argument("--max-files", type=int, default=None)
    backfill.add_argument("--allow-more", action="store_true")
    backfill.add_argument("--all-time", action="store_true", help="Rejected: unbounded downloads")
    backfill.add_argument("--aws-profile", default=None)
    backfill.add_argument("--requester-pays", action="store_true")
    backfill.add_argument("--data-root", type=Path, default=orchestrator.DEFAULT_DATA_ROOT)
    backfill.add_argument("--report-dir", type=Path, default=orchestrator.DEFAULT_REPORT_DIR)
    backfill.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    backfill.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)

    daily = subparsers.add_parser("daily-cycle", help="Run one bounded daily research cycle")
    daily.add_argument("--coins", nargs="+", default=None)
    daily.add_argument("--historical-hours", nargs="+", type=int, default=None)
    daily.add_argument("--max-files", type=int, default=None)
    daily.add_argument("--allow-more", action="store_true")
    daily.add_argument("--aws-profile", default=None)
    daily.add_argument("--requester-pays", action="store_true")
    daily.add_argument("--train-interval", default="15s", choices=["15s", "1m"])
    daily.add_argument("--horizon", type=int, default=5)
    daily.add_argument("--include-combined", action="store_true")
    daily.add_argument("--data-root", type=Path, default=orchestrator.DEFAULT_DATA_ROOT)
    daily.add_argument("--report-dir", type=Path, default=orchestrator.DEFAULT_REPORT_DIR)
    daily.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    daily.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    daily.add_argument(
        "--daily-summary-dir",
        type=Path,
        default=bid_ask_daily_summary.DEFAULT_OUTPUT_DIR,
    )

    batch_series = subparsers.add_parser(
        "run-batch-series",
        help="Run multiple bounded daily-cycle chunks for one date",
    )
    batch_series.add_argument("--date", required=True, help="YYYYMMDD")
    batch_series.add_argument("--start-hour", type=int, required=True)
    batch_series.add_argument("--end-hour", type=int, required=True)
    batch_series.add_argument("--coins", nargs="+", default=None)
    batch_series.add_argument(
        "--chunk-size-hours",
        type=int,
        default=DEFAULT_CHUNK_SIZE_HOURS,
    )
    batch_series.add_argument("--allow-more", action="store_true")
    batch_series.add_argument("--continue-on-error", action="store_true")
    batch_series.add_argument("--aws-profile", default=None)
    batch_series.add_argument("--requester-pays", action="store_true")
    batch_series.add_argument("--train-interval", default="15s", choices=["15s", "1m"])
    batch_series.add_argument("--horizon", type=int, default=5)
    batch_series.add_argument("--include-combined", action="store_true")
    batch_series.add_argument("--max-files", type=int, default=None)
    batch_series.add_argument("--data-root", type=Path, default=orchestrator.DEFAULT_DATA_ROOT)
    batch_series.add_argument("--report-dir", type=Path, default=orchestrator.DEFAULT_REPORT_DIR)
    batch_series.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    batch_series.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    batch_series.add_argument(
        "--daily-summary-dir",
        type=Path,
        default=bid_ask_daily_summary.DEFAULT_OUTPUT_DIR,
    )

    auto_backfill = subparsers.add_parser(
        "auto-backfill-next",
        help="Run one self-validating historical chunk and exit",
    )
    auto_backfill.add_argument("--coins", nargs="+", default=None)
    auto_backfill.add_argument(
        "--chunk-size-hours",
        type=int,
        default=DEFAULT_CHUNK_SIZE_HOURS,
    )
    auto_backfill.add_argument("--max-files", type=int, default=None)
    auto_backfill.add_argument("--allow-more", action="store_true")
    auto_backfill.add_argument("--aws-profile", default=None)
    auto_backfill.add_argument("--requester-pays", action="store_true")
    auto_backfill.add_argument("--train-interval", default="15s", choices=["15s", "1m"])
    auto_backfill.add_argument("--horizon", type=int, default=5)
    auto_backfill.add_argument("--include-combined", action="store_true")
    auto_backfill.add_argument("--data-root", type=Path, default=orchestrator.DEFAULT_DATA_ROOT)
    auto_backfill.add_argument("--report-dir", type=Path, default=orchestrator.DEFAULT_REPORT_DIR)
    auto_backfill.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    auto_backfill.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    auto_backfill.add_argument(
        "--daily-summary-dir",
        type=Path,
        default=bid_ask_daily_summary.DEFAULT_OUTPUT_DIR,
    )

    repair_state = subparsers.add_parser(
        "repair-state-from-existing-data",
        help="Mark chunks complete from existing historical CSV data",
    )
    repair_state.add_argument("--date", required=True, help="YYYYMMDD")
    repair_state.add_argument("--start-hour", type=int, required=True)
    repair_state.add_argument("--end-hour", type=int, required=True)
    repair_state.add_argument("--coins", nargs="+", default=None)
    repair_state.add_argument(
        "--chunk-size-hours",
        type=int,
        default=DEFAULT_CHUNK_SIZE_HOURS,
    )
    repair_state.add_argument("--data-root", type=Path, default=orchestrator.DEFAULT_DATA_ROOT)
    repair_state.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    repair_state.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)

    live = subparsers.add_parser("live-long-session", help="Finite live recorder session")
    live.add_argument("--coins", nargs="+", required=True)
    live.add_argument("--duration-seconds", type=float, required=True)
    live.add_argument("--allow-long", action="store_true")
    live.add_argument("--data-root", type=Path, default=orchestrator.DEFAULT_DATA_ROOT)
    live.add_argument("--report-dir", type=Path, default=orchestrator.DEFAULT_REPORT_DIR)
    live.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    live.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)

    launchd = subparsers.add_parser(
        "write-launchd-template",
        help="Write launchd plist template (does not install)",
    )
    launchd.add_argument("--time", default="02:00", help="Daily run time HH:MM")
    launchd.add_argument("--coins", nargs="+", required=True)
    launchd.add_argument("--aws-profile", default=None)
    launchd.add_argument("--requester-pays", action="store_true")
    launchd.add_argument(
        "--chunk-size-hours",
        type=int,
        default=DEFAULT_CHUNK_SIZE_HOURS,
    )
    launchd.add_argument("--train-interval", default="15s", choices=["15s", "1m"])
    launchd.add_argument("--horizon", type=int, default=5)
    launchd.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    launchd.add_argument("--project-root", type=Path, default=None)
    launchd.add_argument(
        "--autopilot",
        action="store_true",
        help="Schedule autopilot-tick instead of auto-backfill-next",
    )
    launchd.add_argument(
        "--interval-hours",
        type=int,
        default=None,
        help="Run every N hours (StartInterval) instead of daily calendar time",
    )

    configure_autopilot = subparsers.add_parser(
        "configure-autopilot",
        help="Configure autopilot limits and ramp (no download/train)",
    )
    configure_autopilot.add_argument("--enable", action="store_true")
    configure_autopilot.add_argument("--disable", action="store_true")
    configure_autopilot.add_argument("--pause", dest="pause_reason", default=None, metavar="REASON")
    configure_autopilot.add_argument("--resume", action="store_true")
    configure_autopilot.add_argument(
        "--reset-usage",
        action="store_true",
        help="Reset daily usage counters (runs/chunks/files today); default preserves them",
    )
    configure_autopilot.add_argument("--daily-chunk-limit", type=int, default=None)
    configure_autopilot.add_argument("--daily-file-limit", type=int, default=None)
    configure_autopilot.add_argument("--ramp-enabled", action="store_true", default=None)
    configure_autopilot.add_argument("--no-ramp-enabled", action="store_false", dest="ramp_enabled")
    configure_autopilot.add_argument("--ramp-every-successful-days", type=int, default=None)
    configure_autopilot.add_argument("--ramp-chunk-increment", type=int, default=None)
    configure_autopilot.add_argument("--max-daily-chunk-limit", type=int, default=None)
    configure_autopilot.add_argument("--max-daily-file-limit", type=int, default=None)
    configure_autopilot.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)

    autopilot_status = subparsers.add_parser(
        "autopilot-status",
        help="Print autopilot settings and usage",
    )
    autopilot_status.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)

    autopilot_tick = subparsers.add_parser(
        "autopilot-tick",
        help="Run one bounded autopilot chunk when within daily limits",
    )
    autopilot_tick.add_argument("--coins", nargs="+", default=None)
    autopilot_tick.add_argument(
        "--chunk-size-hours",
        type=int,
        default=DEFAULT_CHUNK_SIZE_HOURS,
    )
    autopilot_tick.add_argument("--max-files", type=int, default=None)
    autopilot_tick.add_argument("--allow-more", action="store_true")
    autopilot_tick.add_argument("--aws-profile", default=None)
    autopilot_tick.add_argument("--requester-pays", action="store_true")
    autopilot_tick.add_argument("--train-interval", default="15s", choices=["15s", "1m"])
    autopilot_tick.add_argument("--horizon", type=int, default=5)
    autopilot_tick.add_argument("--include-combined", action="store_true")
    autopilot_tick.add_argument("--data-root", type=Path, default=orchestrator.DEFAULT_DATA_ROOT)
    autopilot_tick.add_argument("--report-dir", type=Path, default=orchestrator.DEFAULT_REPORT_DIR)
    autopilot_tick.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    autopilot_tick.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    autopilot_tick.add_argument(
        "--daily-summary-dir",
        type=Path,
        default=bid_ask_daily_summary.DEFAULT_OUTPUT_DIR,
    )

    return parser


def _print_autopilot_status(status: dict[str, Any]) -> None:
    print("Bid/Ask Research Scheduler Autopilot Status")
    print(f"  enabled: {status['enabled']}")
    print(f"  paused: {status['paused']}")
    if status["pause_reason"]:
        print(f"  pause_reason: {status['pause_reason']}")
    print(f"  started_at: {status['started_at']}")
    print(f"  last_run_date: {status['last_run_date']}")
    print(f"  daily_chunk_limit: {status['daily_chunk_limit']}")
    print(f"  daily_file_limit: {status['daily_file_limit']}")
    print(f"  runs_today: {status['runs_today']}")
    print(f"  chunks_completed_today: {status['chunks_completed_today']}")
    print(f"  files_downloaded_today: {status['files_downloaded_today']}")
    print(f"  ramp_enabled: {status['ramp_enabled']}")
    print(f"  ramp_every_successful_days: {status['ramp_every_successful_days']}")
    print(f"  ramp_chunk_increment: {status['ramp_chunk_increment']}")
    print(f"  max_daily_chunk_limit: {status['max_daily_chunk_limit']}")
    print(f"  max_daily_file_limit: {status['max_daily_file_limit']}")
    print(f"  successful_days_in_row: {status['successful_days_in_row']}")
    print(f"  failed_runs_in_row: {status['failed_runs_in_row']}")
    print(f"  next_recommended_action: {status['next_recommended_action']}")
    print(f"  latest_report: {status['latest_report']}")


def _print_status(status: dict[str, Any]) -> None:
    print("Bid/Ask Research Scheduler Status")
    print(f"  next_historical_date: {status['next_historical_date']}")
    print(f"  completed_days: {status['completed_days']}")
    print(f"  failed_days: {status['failed_days']}")
    print(f"  last_run_at: {status['last_run_at']}")
    print(f"  last_warning: {status['last_warning']}")
    print(f"  latest_report: {status['latest_report']}")
    print(f"  total_historical_files_downloaded: {status['total_historical_files_downloaded']}")
    print(f"  coins: {', '.join(status['coins']) or 'none'}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "init-state":
            state = cmd_init_state(
                historical_start_date=args.historical_start_date,
                historical_end_date=args.historical_end_date,
                historical_start_hour=args.historical_start_hour,
                historical_end_hour=args.historical_end_hour,
                coins=args.coins,
                reset=args.reset,
                confirm_large_range=args.confirm_large_range,
                state_path=args.state_path,
            )
            print(f"Initialized scheduler state at {args.state_path}")
            print(f"  next_historical_date: {state.next_historical_date}")
            print(
                f"  hour window: {state.historical_start_hour}-{state.historical_end_hour} UTC"
            )
            print(f"  warning: {ARCHIVE_AVAILABILITY_WARNING}")
        elif args.command == "status":
            status = cmd_status(state_path=args.state_path)
            _print_status(status)
        elif args.command == "historical-backfill-next":
            report = cmd_historical_backfill_next(
                coins=args.coins,
                hours=args.hours,
                max_files=args.max_files,
                allow_more=args.allow_more,
                all_time=args.all_time,
                aws_profile=args.aws_profile,
                requester_pays=args.requester_pays,
                data_root=args.data_root,
                report_dir=args.report_dir,
                state_path=args.state_path,
                runs_dir=args.runs_dir,
            )
            print(f"Historical backfill report: workflows={report.workflows_run}")
            if report.next_recommended_action:
                print(f"Next: {report.next_recommended_action}")
        elif args.command == "daily-cycle":
            report = cmd_daily_cycle(
                coins=args.coins,
                historical_hours=args.historical_hours,
                max_files=args.max_files,
                allow_more=args.allow_more,
                aws_profile=args.aws_profile,
                requester_pays=args.requester_pays,
                train_interval=args.train_interval,
                horizon=args.horizon,
                include_combined=args.include_combined,
                data_root=args.data_root,
                orchestrator_report_dir=args.report_dir,
                state_path=args.state_path,
                runs_dir=args.runs_dir,
                daily_summary_dir=args.daily_summary_dir,
            )
            print(f"Daily cycle complete: workflows={report.workflows_run}")
            if report.next_recommended_action:
                print(f"Next: {report.next_recommended_action}")
        elif args.command == "run-batch-series":
            report = cmd_run_batch_series(
                date=args.date,
                start_hour=args.start_hour,
                end_hour=args.end_hour,
                coins=args.coins,
                chunk_size_hours=args.chunk_size_hours,
                allow_more=args.allow_more,
                continue_on_error=args.continue_on_error,
                aws_profile=args.aws_profile,
                requester_pays=args.requester_pays,
                train_interval=args.train_interval,
                horizon=args.horizon,
                include_combined=args.include_combined,
                max_files=args.max_files,
                data_root=args.data_root,
                orchestrator_report_dir=args.report_dir,
                state_path=args.state_path,
                runs_dir=args.runs_dir,
                daily_summary_dir=args.daily_summary_dir,
            )
            print(
                "Batch series complete: "
                f"completed={report.chunks_completed} "
                f"failed={report.chunks_failed} "
                f"skipped={report.chunks_skipped}"
            )
            if report.next_recommended_action:
                print(f"Next: {report.next_recommended_action}")
        elif args.command == "auto-backfill-next":
            report = cmd_auto_backfill_next(
                coins=args.coins,
                chunk_size_hours=args.chunk_size_hours,
                max_files=args.max_files,
                allow_more=args.allow_more,
                aws_profile=args.aws_profile,
                requester_pays=args.requester_pays,
                train_interval=args.train_interval,
                horizon=args.horizon,
                include_combined=args.include_combined,
                data_root=args.data_root,
                orchestrator_report_dir=args.report_dir,
                state_path=args.state_path,
                runs_dir=args.runs_dir,
                daily_summary_dir=args.daily_summary_dir,
            )
            if any(AUTO_BACKFILL_COMPLETE_MESSAGE in w for w in report.warnings):
                print(AUTO_BACKFILL_COMPLETE_MESSAGE)
            else:
                print(
                    f"Auto backfill complete: date={report.historical_date} "
                    f"hours={report.historical_hours} workflows={report.workflows_run}"
                )
            if report.next_recommended_action:
                print(f"Next: {report.next_recommended_action}")
        elif args.command == "repair-state-from-existing-data":
            report = cmd_repair_state_from_existing_data(
                date=args.date,
                start_hour=args.start_hour,
                end_hour=args.end_hour,
                coins=args.coins,
                chunk_size_hours=args.chunk_size_hours,
                data_root=args.data_root,
                state_path=args.state_path,
                runs_dir=args.runs_dir,
            )
            print(
                "Repair complete: "
                f"repaired={report.row_counts.get('chunks_repaired', 0)} "
                f"skipped={report.row_counts.get('chunks_skipped', 0)} "
                f"failed={report.row_counts.get('chunks_failed', 0)}"
            )
            print(f"Warning: {REPAIR_INFERRED_WARNING}")
            if report.next_recommended_action:
                print(f"Next: {report.next_recommended_action}")
        elif args.command == "live-long-session":
            report = cmd_live_long_session(
                coins=args.coins,
                duration_seconds=args.duration_seconds,
                allow_long=args.allow_long,
                data_root=args.data_root,
                report_dir=args.report_dir,
                state_path=args.state_path,
                runs_dir=args.runs_dir,
            )
            print(f"Live session report written; duration={report.live_session_duration}s")
            if report.next_recommended_action:
                print(f"Next: {report.next_recommended_action}")
        elif args.command == "write-launchd-template":
            plist_path, setup_path = cmd_write_launchd_template(
                run_time=args.time,
                coins=args.coins,
                aws_profile=args.aws_profile,
                requester_pays=args.requester_pays,
                chunk_size_hours=args.chunk_size_hours,
                train_interval=args.train_interval,
                horizon=args.horizon,
                state_dir=args.state_dir,
                project_root=args.project_root,
                autopilot=args.autopilot,
                interval_hours=args.interval_hours,
            )
            print(f"Wrote launchd template: {plist_path}")
            print(f"Wrote setup instructions: {setup_path}")
            print("Not installed automatically.")
        elif args.command == "configure-autopilot":
            autopilot = cmd_configure_autopilot(
                enable=args.enable,
                disable=args.disable,
                pause_reason=args.pause_reason,
                resume=args.resume,
                reset_usage=args.reset_usage,
                daily_chunk_limit=args.daily_chunk_limit,
                daily_file_limit=args.daily_file_limit,
                ramp_enabled=args.ramp_enabled,
                ramp_every_successful_days=args.ramp_every_successful_days,
                ramp_chunk_increment=args.ramp_chunk_increment,
                max_daily_chunk_limit=args.max_daily_chunk_limit,
                max_daily_file_limit=args.max_daily_file_limit,
                state_path=args.state_path,
            )
            _print_autopilot_status(cmd_autopilot_status(state_path=args.state_path))
            print(f"Autopilot configured: enabled={autopilot.enabled} paused={autopilot.paused}")
        elif args.command == "autopilot-status":
            status = cmd_autopilot_status(state_path=args.state_path)
            _print_autopilot_status(status)
        elif args.command == "autopilot-tick":
            report = cmd_autopilot_tick(
                coins=args.coins,
                chunk_size_hours=args.chunk_size_hours,
                max_files=args.max_files,
                allow_more=args.allow_more,
                aws_profile=args.aws_profile,
                requester_pays=args.requester_pays,
                train_interval=args.train_interval,
                horizon=args.horizon,
                include_combined=args.include_combined,
                data_root=args.data_root,
                orchestrator_report_dir=args.report_dir,
                state_path=args.state_path,
                runs_dir=args.runs_dir,
                daily_summary_dir=args.daily_summary_dir,
            )
            print(f"Autopilot tick: action={report.action}")
            if report.warnings:
                for warning in report.warnings:
                    print(f"  warning: {warning}")
            if report.failures:
                for failure in report.failures:
                    print(f"  failure: {failure}")
            if report.next_recommended_action:
                print(f"Next: {report.next_recommended_action}")
        else:
            parser.error(f"unknown command: {args.command}")
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
