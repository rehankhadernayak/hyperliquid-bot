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
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import bid_ask_candidate_validator as candidate_validator
import bid_ask_daily_summary
import bid_ask_research_orchestrator as orchestrator
from bid_ask_resampler import normalize_source, resampled_csv_path
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
NO_IMPORT_PROGRESS_MESSAGE = (
    "Chunk cycle completed but historical_1s CSVs did not advance; "
    "check date/hour planning or importer path."
)
LAUNCHD_PLIST_NAME_AUTOPILOT = "com.hyperliquid.research.autopilot.plist"
LAUNCHD_PLIST_NAME_AUTOPILOT_HOURLY = "com.hyperliquid.research.autopilot.hourly.plist"
LAUNCHD_SETUP_MD_HOURLY = "launchd_hourly_autopilot_setup.md"
DEFAULT_HOURLY_INTERVAL_SECONDS = 3600
DEFAULT_HOURLY_LABEL = "com.hyperliquid.research.autopilot.hourly"
DEFAULT_HOURLY_STDOUT_LOG = DEFAULT_STATE_DIR / "hourly_launchd.out.log"
DEFAULT_HOURLY_STDERR_LOG = DEFAULT_STATE_DIR / "hourly_launchd.err.log"
HOURLY_AUTOPILOT_RECOMMENDED_DAILY_CHUNK_LIMIT = 8
HOURLY_AUTOPILOT_RECOMMENDED_DAILY_FILE_LIMIT = 24

PHASE_BACKFILL_DEFAULT_NAME = "phase3_btc_eth_recent"
PHASE_BACKFILL_DEFAULT_COINS = ["BTC", "ETH"]
PHASE_BACKFILL_DEFAULT_TARGET_DAYS = 14
PHASE_BACKFILL_DEFAULT_DAILY_FILE_LIMIT = 144
PHASE_BACKFILL_DEFAULT_DAILY_CHUNK_LIMIT = 24
PHASE_BACKFILL_ABSOLUTE_MAX_DAILY_FILE_LIMIT = 144
PHASE_BACKFILL_ABSOLUTE_MAX_DAILY_CHUNK_LIMIT = 24
PHASE_BACKFILL_CONFIRMATION_TEXT = "I understand this may increase AWS Requester Pays costs"
DEFAULT_AWS_PROFILE = "hyperliquid-research"
DEFAULT_REQUESTER_PAYS = True
PHASE_REQUESTER_PAYS_WARNING = (
    "Phase backfill enabled: downloads may incur AWS Requester Pays costs."
)
REQUESTER_PAYS_DISABLED_WARNING = (
    "requester_pays disabled; downloads may fail for Requester Pays buckets"
)
REQUESTER_PAYS_AUTH_NEXT_ACTION = (
    "Check AWS profile and requester pays settings, then rerun autopilot-tick"
)
PHASE_CHUNK_LIMIT_BOTTLENECK_WARNING = (
    "daily_chunk_limit bottlenecks requested daily_file_limit"
)
PHASE_BACKFILL_BASELINE_DATE = "20230916"
DEFAULT_DOWNLOADS_ROOT = Path("downloads/hyperliquid")
DEFAULT_CANDIDATE_VALIDATION_REPORTS_DIR = Path("reports/bid_ask_candidate_validation")
PHASE_AUDIT_CANDIDATE_UNCHANGED_LIMIT = 3
AUTOPILOT_REPORT_GLOB = "bid_ask_scheduler_autopilot_*.json"
CONSERVATIVE_DAILY_CHUNK_LIMIT = 4
CONSERVATIVE_DAILY_FILE_LIMIT = 24
CONSERVATIVE_MAX_DAILY_CHUNK_LIMIT = 4
CONSERVATIVE_MAX_DAILY_FILE_LIMIT = 36

PHASE3_AUDIT_JSON_PATH = DEFAULT_STATE_DIR / "phase3_audit_latest.json"
PHASE3_AUDIT_MARKDOWN_PATH = DEFAULT_STATE_DIR / "phase3_audit_latest.md"
DEFAULT_RESEARCH_SESSIONS_DIR = DEFAULT_STATE_DIR / "sessions"
DEFAULT_RESEARCH_SESSION_LABEL_PREFIX = "com.hyperliquid.research.session"
LAUNCHD_RESEARCH_SESSION_MORNING_PLIST = "com.hyperliquid.research.session.morning.plist"
LAUNCHD_RESEARCH_SESSION_EVENING_PLIST = "com.hyperliquid.research.session.evening.plist"
LAUNCHD_RESEARCH_SESSION_SETUP_MD = "launchd_research_session_setup.md"
DEFAULT_RESEARCH_SESSION_DURATION_HOURS = 8
DEFAULT_RESEARCH_SESSION_TICK_INTERVAL_MINUTES = 60
DEFAULT_RESEARCH_SESSION_MORNING_HOUR = 8
DEFAULT_RESEARCH_SESSION_EVENING_HOUR = 16
DEFAULT_RESEARCH_SESSION_OUTPUT_DIR = Path("launchd")
RESEARCH_SESSION_RESAMPLE_COINS = ["BTC", "ETH"]
RELAXED_CANDIDATE_B_NAME = "BTC_DOWN_MICRO_H20_CANDIDATE_B_RELAXED"
RELAXED_CANDIDATE_B_HORIZON = 20
RELAXED_CANDIDATE_B_MIN_CONFIDENCE = 0.6
RELAXED_CANDIDATE_B_MIN_EDGE_BPS = 1.0
RELAXED_CANDIDATE_B_COST_MULTIPLIER = 1.25
RELAXED_CANDIDATE_B_MAX_SPREAD_BPS = 8.0
RELAXED_CANDIDATE_B_FEE_BPS = 2.0
RELAXED_CANDIDATE_B_SLIPPAGE_BPS = 1.0


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
    phase_backfill_enabled: bool = False
    phase_backfill_name: str | None = None
    focused_coins: list[str] = field(default_factory=list)
    target_days: int = 0
    aws_profile: str = DEFAULT_AWS_PROFILE
    requester_pays: bool = DEFAULT_REQUESTER_PAYS

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
            "phase_backfill_enabled": self.phase_backfill_enabled,
            "phase_backfill_name": self.phase_backfill_name,
            "focused_coins": self.focused_coins,
            "target_days": self.target_days,
            "aws_profile": self.aws_profile,
            "requester_pays": self.requester_pays,
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
            phase_backfill_enabled=bool(payload.get("phase_backfill_enabled", False)),
            phase_backfill_name=payload.get("phase_backfill_name"),
            focused_coins=list(payload.get("focused_coins") or []),
            target_days=int(payload.get("target_days") or 0),
            aws_profile=str(payload.get("aws_profile") or DEFAULT_AWS_PROFILE),
            requester_pays=bool(
                payload.get("requester_pays", DEFAULT_REQUESTER_PAYS)
            ),
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
    files_requested: int = 0
    files_downloaded: int = 0
    files_downloaded_new: int = 0
    files_skipped_existing: int = 0
    files_imported: int = 0
    pre_import_coverage: dict[str, dict[str, Any]] = field(default_factory=dict)
    post_import_coverage: dict[str, dict[str, Any]] = field(default_factory=dict)
    import_progress_by_coin: dict[str, bool] = field(default_factory=dict)
    rows_added_by_coin: dict[str, int] = field(default_factory=dict)
    timestamp_advanced_by_coin: dict[str, bool] = field(default_factory=dict)
    real_execution_ready: bool = False
    daily_chunk_limit: int | None = None
    daily_file_limit: int | None = None
    effective_daily_file_capacity: int | None = None
    chunks_completed_today: int | None = None
    files_downloaded_today: int | None = None
    phase_backfill_enabled: bool = False
    phase_backfill_name: str | None = None
    focused_coins: list[str] = field(default_factory=list)
    target_days: int | None = None
    ramp_skipped_for_phase_backfill: bool = False
    aws_profile: str | None = None
    requester_pays: bool | None = None
    download_error_type: str | None = None
    files_failed: int = 0
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
            "files_requested": self.files_requested,
            "files_downloaded": self.files_downloaded,
            "files_downloaded_new": self.files_downloaded_new,
            "files_skipped_existing": self.files_skipped_existing,
            "files_imported": self.files_imported,
            "pre_import_coverage": self.pre_import_coverage,
            "post_import_coverage": self.post_import_coverage,
            "import_progress_by_coin": self.import_progress_by_coin,
            "rows_added_by_coin": self.rows_added_by_coin,
            "timestamp_advanced_by_coin": self.timestamp_advanced_by_coin,
            "real_execution_ready": self.real_execution_ready,
            "daily_chunk_limit": self.daily_chunk_limit,
            "daily_file_limit": self.daily_file_limit,
            "effective_daily_file_capacity": self.effective_daily_file_capacity,
            "chunks_completed_today": self.chunks_completed_today,
            "files_downloaded_today": self.files_downloaded_today,
            "phase_backfill_enabled": self.phase_backfill_enabled,
            "phase_backfill_name": self.phase_backfill_name,
            "focused_coins": self.focused_coins,
            "target_days": self.target_days,
            "ramp_skipped_for_phase_backfill": self.ramp_skipped_for_phase_backfill,
            "aws_profile": self.aws_profile,
            "requester_pays": self.requester_pays,
            "download_error_type": self.download_error_type,
            "files_failed": self.files_failed,
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
    aws_profile: str | None = None
    requester_pays: bool | None = None
    download_error_type: str | None = None
    files_downloaded_new: int = 0
    files_skipped_existing: int = 0
    files_failed: int = 0

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
            "aws_profile": self.aws_profile,
            "requester_pays": self.requester_pays,
            "download_error_type": self.download_error_type,
            "files_downloaded_new": self.files_downloaded_new,
            "files_skipped_existing": self.files_skipped_existing,
            "files_failed": self.files_failed,
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


def compute_effective_daily_file_capacity(
    *,
    daily_file_limit: int,
    daily_chunk_limit: int,
    chunk_size_hours: int,
    num_coins: int,
) -> int:
    """Return achievable daily file throughput given chunk and file caps."""
    chunk_daily_capacity = daily_chunk_limit * chunk_size_hours * num_coins
    return min(daily_file_limit, chunk_daily_capacity)


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
            payload.get("files_downloaded_new")
            or payload.get("files_downloaded")
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
    report.phase_backfill_enabled = autopilot.phase_backfill_enabled
    report.phase_backfill_name = autopilot.phase_backfill_name
    report.focused_coins = list(autopilot.focused_coins)
    report.target_days = autopilot.target_days if autopilot.target_days > 0 else None
    report.ramp_skipped_for_phase_backfill = autopilot.phase_backfill_enabled
    if autopilot.phase_backfill_enabled and autopilot.focused_coins:
        num_coins = len(orchestrator._normalize_coins(autopilot.focused_coins))
        report.effective_daily_file_capacity = compute_effective_daily_file_capacity(
            daily_file_limit=autopilot.daily_file_limit,
            daily_chunk_limit=autopilot.daily_chunk_limit,
            chunk_size_hours=report.chunk_size_hours,
            num_coins=num_coins,
        )


def _resolve_aws_access_settings(
    autopilot: AutopilotState,
    *,
    aws_profile: str | None = None,
    requester_pays: bool | None = None,
) -> tuple[str, bool]:
    """Resolve AWS profile and requester_pays from CLI overrides or autopilot state."""
    resolved_profile = (
        aws_profile if aws_profile is not None else autopilot.aws_profile
    )
    resolved_requester_pays = (
        requester_pays if requester_pays is not None else autopilot.requester_pays
    )
    return resolved_profile, resolved_requester_pays


def _extract_download_failures(
    batch_payload: dict[str, Any],
) -> tuple[list[str], str | None, int]:
    """Return download failure messages, primary error type, and failed file count."""
    download = batch_payload.get("download") or {}
    file_results = download.get("file_results") or []
    if not isinstance(file_results, list):
        file_results = []

    failures: list[str] = []
    error_types: list[str] = []
    files_failed = 0
    for item in file_results:
        if not isinstance(item, dict):
            continue
        if item.get("download_status") != "failed":
            continue
        files_failed += 1
        error_type = str(item.get("download_error_type") or "unknown_error")
        error_types.append(error_type)
        coin = item.get("coin") or "unknown"
        s3_path = item.get("s3_path") or "unknown path"
        failures.append(f"{coin}: download failed ({error_type}): {s3_path}")

    primary_error_type = error_types[0] if error_types else None
    return failures, primary_error_type, files_failed


def _download_failure_next_action(
    *,
    aws_profile: str,
    requester_pays: bool,
    download_error_type: str | None,
) -> str:
    if download_error_type == "requester_pays_auth_required":
        return (
            f"{REQUESTER_PAYS_AUTH_NEXT_ACTION} "
            f"(aws_profile={aws_profile!r}, requester_pays={requester_pays})"
        )
    return (
        "Review download failure report, fix data/tooling/network, then rerun auto-backfill-next"
    )


def _populate_aws_access_report_fields(
    report: AutopilotReport | SchedulerReport,
    *,
    aws_profile: str,
    requester_pays: bool,
) -> None:
    report.aws_profile = aws_profile
    report.requester_pays = requester_pays
    if hasattr(report, "details") and isinstance(report.details, dict):
        report.details["aws_profile"] = aws_profile
        report.details["requester_pays"] = requester_pays


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
    if autopilot.phase_backfill_enabled:
        return False
    if not autopilot.ramp_enabled:
        return False
    if autopilot.successful_days_in_row < autopilot.ramp_every_successful_days:
        return False
    prior_chunk_limit = autopilot.daily_chunk_limit
    prior_file_limit = autopilot.daily_file_limit
    new_chunk_limit = min(
        autopilot.max_daily_chunk_limit,
        autopilot.daily_chunk_limit + autopilot.ramp_chunk_increment,
    )
    computed_file_limit = min(
        autopilot.max_daily_file_limit,
        new_chunk_limit * chunk_size_hours * num_coins,
    )
    autopilot.daily_chunk_limit = max(prior_chunk_limit, new_chunk_limit)
    autopilot.daily_file_limit = max(prior_file_limit, computed_file_limit)
    if (
        autopilot.daily_chunk_limit == prior_chunk_limit
        and autopilot.daily_file_limit == prior_file_limit
    ):
        return False
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


def _effective_scheduler_coins(
    state: SchedulerState,
    coins: list[str] | None = None,
) -> list[str]:
    """Resolve coins for backfill; phase mode prefers focused_coins when set."""
    if coins:
        return orchestrator._normalize_coins(coins)
    autopilot = state.autopilot
    if autopilot.phase_backfill_enabled and autopilot.focused_coins:
        return orchestrator._normalize_coins(autopilot.focused_coins)
    return orchestrator._normalize_coins(state.coins)


def compute_phase_backfill_estimate(
    *,
    target_days: int,
    coins: list[str],
    daily_file_limit: int,
    daily_chunk_limit: int = PHASE_BACKFILL_DEFAULT_DAILY_CHUNK_LIMIT,
    chunk_size_hours: int = DEFAULT_CHUNK_SIZE_HOURS,
) -> dict[str, Any]:
    """Compute phase backfill file counts and ETA without changing state."""
    normalized_coins = orchestrator._normalize_coins(coins)
    num_coins = len(normalized_coins)
    if num_coins < 1:
        raise ValueError("coins must contain at least one coin")
    if target_days < 1:
        raise ValueError("target-days must be at least 1")
    if daily_file_limit < 1:
        raise ValueError("daily-file-limit must be at least 1")
    if daily_chunk_limit < 1:
        raise ValueError("daily-chunk-limit must be at least 1")
    if chunk_size_hours < 1:
        raise ValueError("chunk-size-hours must be at least 1")

    files_per_archive_day = 24 * num_coins
    files_required = target_days * files_per_archive_day
    effective_daily_file_capacity = compute_effective_daily_file_capacity(
        daily_file_limit=daily_file_limit,
        daily_chunk_limit=daily_chunk_limit,
        chunk_size_hours=chunk_size_hours,
        num_coins=num_coins,
    )
    archive_days_per_real_day = effective_daily_file_capacity / files_per_archive_day
    estimated_real_days_to_complete = files_required / effective_daily_file_capacity

    warnings: list[str] = []
    if effective_daily_file_capacity < daily_file_limit:
        warnings.append(PHASE_CHUNK_LIMIT_BOTTLENECK_WARNING)

    return {
        "target_days": target_days,
        "coins": normalized_coins,
        "num_coins": num_coins,
        "daily_file_limit": daily_file_limit,
        "daily_chunk_limit": daily_chunk_limit,
        "chunk_size_hours": chunk_size_hours,
        "effective_daily_file_capacity": effective_daily_file_capacity,
        "files_required": files_required,
        "archive_days_per_real_day": archive_days_per_real_day,
        "estimated_real_days_to_complete": estimated_real_days_to_complete,
        "warnings": warnings,
    }


def _bound_phase_backfill_limits(
    *,
    daily_chunk_limit: int,
    daily_file_limit: int,
    max_daily_chunk_limit: int,
    max_daily_file_limit: int,
) -> tuple[int, int, int, int]:
    """Validate and clamp phase limits to absolute hard caps."""
    if daily_chunk_limit > PHASE_BACKFILL_ABSOLUTE_MAX_DAILY_CHUNK_LIMIT:
        raise ValueError(
            f"daily_chunk_limit {daily_chunk_limit} exceeds absolute max "
            f"{PHASE_BACKFILL_ABSOLUTE_MAX_DAILY_CHUNK_LIMIT}"
        )
    if max_daily_chunk_limit > PHASE_BACKFILL_ABSOLUTE_MAX_DAILY_CHUNK_LIMIT:
        raise ValueError(
            f"max_daily_chunk_limit {max_daily_chunk_limit} exceeds absolute max "
            f"{PHASE_BACKFILL_ABSOLUTE_MAX_DAILY_CHUNK_LIMIT}"
        )
    bounded_daily_chunk = daily_chunk_limit
    bounded_max_chunk = max_daily_chunk_limit
    bounded_daily_file = min(daily_file_limit, PHASE_BACKFILL_ABSOLUTE_MAX_DAILY_FILE_LIMIT)
    bounded_max_file = min(max_daily_file_limit, PHASE_BACKFILL_ABSOLUTE_MAX_DAILY_FILE_LIMIT)

    if bounded_daily_chunk < 1:
        raise ValueError("daily-chunk-limit must be at least 1")
    if bounded_daily_file < 1:
        raise ValueError("daily-file-limit must be at least 1")
    if bounded_max_chunk < 1:
        raise ValueError("max-daily-chunk-limit must be at least 1")
    if bounded_max_file < 1:
        raise ValueError("max-daily-file-limit must be at least 1")
    if bounded_daily_chunk > bounded_max_chunk:
        raise ValueError(
            f"daily_chunk_limit {bounded_daily_chunk} exceeds "
            f"max_daily_chunk_limit {bounded_max_chunk}"
        )
    if bounded_daily_file > bounded_max_file:
        raise ValueError(
            f"daily_file_limit {bounded_daily_file} exceeds "
            f"max_daily_file_limit {bounded_max_file}"
        )
    return bounded_daily_chunk, bounded_daily_file, bounded_max_chunk, bounded_max_file


def _phase_backfill_status_fields(
    autopilot: AutopilotState,
    *,
    chunk_size_hours: int = DEFAULT_CHUNK_SIZE_HOURS,
) -> dict[str, Any]:
    """Return phase backfill fields for status output."""
    fields: dict[str, Any] = {
        "phase_backfill_enabled": autopilot.phase_backfill_enabled,
        "phase_backfill_name": autopilot.phase_backfill_name,
        "focused_coins": list(autopilot.focused_coins),
        "target_days": autopilot.target_days,
        "aws_profile": autopilot.aws_profile,
        "requester_pays": autopilot.requester_pays,
        "estimated_files_required": None,
        "estimated_real_days_to_complete": None,
        "archive_days_per_real_day": None,
        "effective_daily_file_capacity": None,
        "requester_pays_warning": None,
        "warnings": [],
    }
    if autopilot.phase_backfill_enabled and autopilot.target_days > 0 and autopilot.focused_coins:
        estimate = compute_phase_backfill_estimate(
            target_days=autopilot.target_days,
            coins=autopilot.focused_coins,
            daily_file_limit=autopilot.daily_file_limit,
            daily_chunk_limit=autopilot.daily_chunk_limit,
            chunk_size_hours=chunk_size_hours,
        )
        fields["estimated_files_required"] = estimate["files_required"]
        fields["estimated_real_days_to_complete"] = estimate["estimated_real_days_to_complete"]
        fields["archive_days_per_real_day"] = estimate["archive_days_per_real_day"]
        fields["effective_daily_file_capacity"] = estimate["effective_daily_file_capacity"]
        fields["requester_pays_warning"] = PHASE_REQUESTER_PAYS_WARNING
        fields["warnings"] = list(estimate.get("warnings") or [])
    return fields


def cmd_phase_backfill_estimate(
    *,
    coins: list[str] | None = None,
    target_days: int = PHASE_BACKFILL_DEFAULT_TARGET_DAYS,
    daily_file_limit: int = PHASE_BACKFILL_DEFAULT_DAILY_FILE_LIMIT,
    daily_chunk_limit: int = PHASE_BACKFILL_DEFAULT_DAILY_CHUNK_LIMIT,
    chunk_size_hours: int = DEFAULT_CHUNK_SIZE_HOURS,
) -> dict[str, Any]:
    """Dry-run phase backfill ETA without changing state."""
    normalized_coins = orchestrator._normalize_coins(
        coins or PHASE_BACKFILL_DEFAULT_COINS
    )
    return compute_phase_backfill_estimate(
        target_days=target_days,
        coins=normalized_coins,
        daily_file_limit=daily_file_limit,
        daily_chunk_limit=daily_chunk_limit,
        chunk_size_hours=chunk_size_hours,
    )


def _load_state_optional(state_path: Path) -> SchedulerState | None:
    if not state_path.is_file():
        return None
    return load_state(state_path)


def _phase_baseline_end_ms() -> int:
    base = datetime.strptime(PHASE_BACKFILL_BASELINE_DATE, "%Y%m%d").replace(tzinfo=timezone.utc)
    end_dt = base.replace(hour=23, minute=59, second=59, microsecond=999000)
    return int(end_dt.timestamp() * 1000)


def _timestamp_ms_to_iso(timestamp_ms: int | None) -> str | None:
    if timestamp_ms is None:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _coverage_hours(start_ms: int | None, end_ms: int | None) -> float | None:
    if start_ms is None or end_ms is None or end_ms < start_ms:
        return None
    return round((end_ms - start_ms) / 3_600_000, 4)


def _historical_freshness_status(end_timestamp_ms: int | None) -> str:
    if end_timestamp_ms is None:
        return "missing"
    if end_timestamp_ms > _phase_baseline_end_ms():
        return "advancing"
    return "stale"


def _csv_coverage_entry(csv_path: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": str(csv_path),
        "exists": csv_path.is_file(),
        "row_count": 0,
        "first_timestamp_ms": None,
        "last_timestamp_ms": None,
        "first_timestamp": None,
        "last_timestamp": None,
        "coverage_hours": None,
        "freshness_status": "missing",
    }
    if not entry["exists"]:
        return entry

    try:
        coverage = summarize_historical_coverage(csv_path)
    except Exception as exc:  # noqa: BLE001
        entry["error"] = str(exc)
        return entry

    entry["row_count"] = coverage.row_count
    entry["first_timestamp_ms"] = coverage.start_timestamp_ms
    entry["last_timestamp_ms"] = coverage.end_timestamp_ms
    entry["first_timestamp"] = _timestamp_ms_to_iso(coverage.start_timestamp_ms)
    entry["last_timestamp"] = _timestamp_ms_to_iso(coverage.end_timestamp_ms)
    entry["coverage_hours"] = _coverage_hours(
        coverage.start_timestamp_ms,
        coverage.end_timestamp_ms,
    )
    entry["freshness_status"] = _historical_freshness_status(coverage.end_timestamp_ms)
    return entry


def _historical_1s_coverage_by_coin(
    coins: list[str],
    data_root: Path,
) -> dict[str, dict[str, Any]]:
    return {
        coin: _csv_coverage_entry(default_historical_output_path(coin, output_root=data_root))
        for coin in coins
    }


def _compute_import_progress_delta(
    pre_coverage: dict[str, dict[str, Any]],
    post_coverage: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows_added_by_coin: dict[str, int] = {}
    timestamp_advanced_by_coin: dict[str, bool] = {}
    import_progress_by_coin: dict[str, bool] = {}

    for coin in pre_coverage:
        pre = pre_coverage.get(coin) or {}
        post = post_coverage.get(coin) or {}
        pre_rows = int(pre.get("row_count") or 0)
        post_rows = int(post.get("row_count") or 0)
        pre_last = pre.get("last_timestamp_ms")
        post_last = post.get("last_timestamp_ms")

        rows_added = max(0, post_rows - pre_rows)
        ts_advanced = post_last is not None and (
            pre_last is None or int(post_last) > int(pre_last)
        )
        rows_added_by_coin[coin] = rows_added
        timestamp_advanced_by_coin[coin] = ts_advanced
        import_progress_by_coin[coin] = rows_added > 0 or ts_advanced

    return {
        "rows_added_by_coin": rows_added_by_coin,
        "timestamp_advanced_by_coin": timestamp_advanced_by_coin,
        "import_progress_by_coin": import_progress_by_coin,
        "any_import_progress": any(import_progress_by_coin.values()),
    }


def _orchestrator_batch_from_scheduler_report(report: SchedulerReport) -> dict[str, Any]:
    backfill = (report.details or {}).get("historical_backfill") or {}
    if not isinstance(backfill, dict):
        return {}
    details = backfill.get("details") or {}
    if not isinstance(details, dict):
        return {}
    batch = details.get("orchestrator_batch") or {}
    return batch if isinstance(batch, dict) else {}


def _summarize_batch_file_stats(batch_payload: dict[str, Any]) -> dict[str, int]:
    download = batch_payload.get("download") or {}
    file_results = download.get("file_results") or []
    if not isinstance(file_results, list):
        file_results = []

    files_downloaded_new = int(download.get("downloaded_count") or 0)
    files_skipped_existing = int(download.get("skipped_count") or 0)

    if file_results:
        files_downloaded_new = sum(
            1
            for item in file_results
            if item.get("download_status") == "downloaded" or item.get("status") == "downloaded"
        )
        files_skipped_existing = sum(
            1
            for item in file_results
            if item.get("download_status") == "skipped_existing"
            or item.get("decompress_status") == "skipped_existing"
            or item.get("status") in {"skipped", "skipped_existing"}
        )

    import_results = batch_payload.get("import") or []
    if not isinstance(import_results, list):
        import_results = []
    files_imported = sum(
        1
        for item in import_results
        if item.get("import_status") == "imported" and int(item.get("rows_written") or 0) > 0
    )

    files_failed = int(download.get("failed_count") or 0)
    if file_results:
        files_failed = sum(
            1
            for item in file_results
            if isinstance(item, dict) and item.get("download_status") == "failed"
        )

    command_plan = batch_payload.get("command_plan") or {}
    files_requested = int(command_plan.get("total_files") or command_plan.get("max_files") or 0)
    if files_requested == 0:
        files_requested = int(download.get("selected_files") or download.get("requested_files") or 0)

    return {
        "files_requested": files_requested,
        "files_downloaded_new": files_downloaded_new,
        "files_skipped_existing": files_skipped_existing,
        "files_imported": files_imported,
        "files_failed": files_failed,
    }


def _extract_download_error_type(batch_payload: dict[str, Any]) -> str | None:
    _, error_type, _ = _extract_download_failures(batch_payload)
    return error_type


def _extract_import_failures(batch_payload: dict[str, Any]) -> list[str]:
    import_results = batch_payload.get("import") or []
    if not isinstance(import_results, list):
        return []
    failures: list[str] = []
    for item in import_results:
        if item.get("import_status") == "failed":
            coin = item.get("coin") or "unknown"
            error = item.get("error") or "unknown import error"
            failures.append(f"{coin}: import failed: {error}")
    return failures


def _build_import_progress_report(
    *,
    coins: list[str],
    data_root: Path,
    pre_coverage: dict[str, dict[str, Any]],
    batch_payload: dict[str, Any],
) -> dict[str, Any]:
    post_coverage = _historical_1s_coverage_by_coin(coins, data_root)
    delta = _compute_import_progress_delta(pre_coverage, post_coverage)
    file_stats = _summarize_batch_file_stats(batch_payload)
    import_failures = _extract_import_failures(batch_payload)
    download_failures, download_error_type, _ = _extract_download_failures(batch_payload)
    return {
        "pre_import_coverage": pre_coverage,
        "post_import_coverage": post_coverage,
        "import_failures": import_failures,
        "download_failures": download_failures,
        "download_error_type": download_error_type,
        **delta,
        **file_stats,
    }


def _populate_autopilot_import_fields(
    report: AutopilotReport,
    progress: dict[str, Any],
    *,
    requested_files: int,
) -> None:
    report.files_requested = requested_files
    report.files_downloaded = int(progress.get("files_downloaded_new") or 0)
    report.files_downloaded_new = int(progress.get("files_downloaded_new") or 0)
    report.files_skipped_existing = int(progress.get("files_skipped_existing") or 0)
    report.files_imported = int(progress.get("files_imported") or 0)
    report.files_failed = int(progress.get("files_failed") or 0)
    report.download_error_type = progress.get("download_error_type")
    report.pre_import_coverage = dict(progress.get("pre_import_coverage") or {})
    report.post_import_coverage = dict(progress.get("post_import_coverage") or {})
    report.import_progress_by_coin = dict(progress.get("import_progress_by_coin") or {})
    report.rows_added_by_coin = dict(progress.get("rows_added_by_coin") or {})
    report.timestamp_advanced_by_coin = dict(progress.get("timestamp_advanced_by_coin") or {})
    report.real_execution_ready = False


def _resampled_coverage_entry(
    coin: str,
    interval: str,
    *,
    source: str,
    data_root: Path,
) -> dict[str, Any]:
    csv_path = resampled_csv_path(coin, interval, data_root, source=source)
    entry: dict[str, Any] = {
        "path": str(csv_path),
        "exists": csv_path.is_file(),
        "row_count": 0,
        "first_timestamp_ms": None,
        "last_timestamp_ms": None,
        "first_timestamp": None,
        "last_timestamp": None,
        "coverage_hours": None,
    }
    if not entry["exists"]:
        return entry

    start_values: list[int] = []
    end_values: list[int] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entry["row_count"] += 1
            try:
                start_values.append(int(row["start_timestamp_ms"]))
                end_values.append(int(row["end_timestamp_ms"]))
            except (KeyError, TypeError, ValueError):
                continue

    if start_values and end_values:
        first_ms = min(start_values)
        last_ms = max(end_values)
        entry["first_timestamp_ms"] = first_ms
        entry["last_timestamp_ms"] = last_ms
        entry["first_timestamp"] = _timestamp_ms_to_iso(first_ms)
        entry["last_timestamp"] = _timestamp_ms_to_iso(last_ms)
        entry["coverage_hours"] = _coverage_hours(first_ms, last_ms)
    return entry


def _find_latest_autopilot_report(
    runs_dir: Path,
    state: SchedulerState | None,
) -> tuple[str | None, dict[str, Any] | None]:
    candidate_path: Path | None = None
    if state and state.autopilot.last_autopilot_report_path:
        saved = Path(state.autopilot.last_autopilot_report_path)
        if saved.is_file():
            candidate_path = saved

    if candidate_path is None and runs_dir.is_dir():
        matches = sorted(
            runs_dir.glob(AUTOPILOT_REPORT_GLOB),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if matches:
            candidate_path = matches[0]

    if candidate_path is None:
        return None, None

    try:
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return str(candidate_path), None
    return str(candidate_path), payload


def _build_scheduler_progress(
    *,
    state: SchedulerState | None,
    runs_dir: Path,
) -> dict[str, Any]:
    report_path, report = _find_latest_autopilot_report(runs_dir, state)
    autopilot = state.autopilot if state else AutopilotState()
    snapshot = (report or {}).get("autopilot_snapshot") or {}

    def _pick_int(*values: Any) -> int:
        for value in values:
            if value is not None:
                return int(value)
        return 0

    warnings = list((report or {}).get("warnings") or [])
    failures = list((report or {}).get("failures") or [])
    coins = list((report or {}).get("coins") or [])
    if not coins and autopilot.focused_coins:
        coins = list(autopilot.focused_coins)
    if not coins and state:
        coins = list(state.coins)

    return {
        "latest_autopilot_report_path": report_path,
        "latest_action": (report or {}).get("action"),
        "files_downloaded_today": _pick_int(
            (report or {}).get("files_downloaded_today"),
            snapshot.get("files_downloaded_today"),
            autopilot.files_downloaded_today,
        ),
        "files_downloaded_new": _pick_int((report or {}).get("files_downloaded_new")),
        "files_imported": _pick_int((report or {}).get("files_imported")),
        "files_skipped_existing": _pick_int((report or {}).get("files_skipped_existing")),
        "chunks_completed_today": _pick_int(
            (report or {}).get("chunks_completed_today"),
            snapshot.get("chunks_completed_today"),
            autopilot.chunks_completed_today,
        ),
        "failed_runs_in_row": _pick_int(
            snapshot.get("failed_runs_in_row"),
            autopilot.failed_runs_in_row,
        ),
        "paused": bool(snapshot.get("paused", autopilot.paused)),
        "warnings": warnings,
        "errors": failures,
        "latest_coins": coins,
    }


def _summarize_downloads(downloads_root: Path, coins: list[str]) -> dict[str, Any]:
    coin_set = {coin.upper() for coin in coins}
    summary: dict[str, Any] = {
        "downloads_root": str(downloads_root),
        "exists": downloads_root.is_dir(),
        "total_files": 0,
        "by_coin": {coin: 0 for coin in coins},
        "by_date_hour": {},
        "newest_mtime_iso": None,
        "latest_files_per_coin": {},
    }
    if not downloads_root.is_dir():
        return summary

    newest_mtime = 0.0
    latest_per_coin: dict[str, dict[str, Any]] = {}

    for path in downloads_root.rglob("*"):
        if not path.is_file() or path.suffix not in {".lz4", ".jsonl"}:
            continue

        try:
            relative = path.relative_to(downloads_root)
        except ValueError:
            continue
        if len(relative.parts) < 3:
            continue

        date_str, hour_str = relative.parts[0], relative.parts[1]
        coin = path.stem.upper()
        if coin not in coin_set:
            continue

        summary["total_files"] += 1
        summary["by_coin"][coin] = int(summary["by_coin"].get(coin, 0)) + 1
        date_hour_key = f"{date_str}/{hour_str}"
        summary["by_date_hour"][date_hour_key] = (
            int(summary["by_date_hour"].get(date_hour_key, 0)) + 1
        )

        mtime = path.stat().st_mtime
        if mtime > newest_mtime:
            newest_mtime = mtime

        current = latest_per_coin.get(coin)
        sort_key = (date_str, hour_str, path.name)
        if current is None or sort_key > tuple(current["_sort_key"]):
            latest_per_coin[coin] = {
                "_sort_key": list(sort_key),
                "path": str(path),
                "date": date_str,
                "hour": hour_str,
                "suffix": path.suffix,
                "mtime_iso": datetime.fromtimestamp(mtime, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            }

    for coin, payload in latest_per_coin.items():
        payload.pop("_sort_key", None)
        summary["latest_files_per_coin"][coin] = payload

    if newest_mtime > 0:
        summary["newest_mtime_iso"] = (
            datetime.fromtimestamp(newest_mtime, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    return summary


def _candidate_status_label(report: dict[str, Any]) -> str:
    status = report.get("validation_status") or {}
    if status.get("paper_candidate_pass"):
        return "paper_candidate_pass"
    return "not_market_ready"


def _detect_repeated_unchanged_candidate_reports(
    reports: list[dict[str, Any]],
    *,
    limit: int = PHASE_AUDIT_CANDIDATE_UNCHANGED_LIMIT,
) -> bool:
    if len(reports) < limit:
        return False
    recent = reports[-limit:]
    signatures = {
        (
            item.get("filtered_paper_trades"),
            float(item.get("cumulative_net_return_bps", 0.0)),
            _candidate_status_label(item),
        )
        for item in recent
    }
    return len(signatures) == 1


def _build_candidate_freshness(
    candidate_reports_dir: Path,
) -> dict[str, Any]:
    reports = candidate_validator.load_candidate_validation_reports(
        reports_dir=candidate_reports_dir
    )
    if not reports:
        return {
            "latest_candidate_report_path": None,
            "latest_run_timestamp": None,
            "filtered_paper_trades": None,
            "cumulative_net_return_bps": None,
            "status": None,
            "repeated_unchanged_warning": False,
            "unchanged_reports_checked": 0,
        }

    latest = reports[-1]
    repeated_unchanged = _detect_repeated_unchanged_candidate_reports(reports)
    return {
        "latest_candidate_report_path": latest.get("json_report_path"),
        "latest_run_timestamp": latest.get("run_timestamp"),
        "filtered_paper_trades": latest.get("filtered_paper_trades"),
        "cumulative_net_return_bps": latest.get("cumulative_net_return_bps"),
        "status": _candidate_status_label(latest),
        "repeated_unchanged_warning": repeated_unchanged,
        "unchanged_reports_checked": min(len(reports), PHASE_AUDIT_CANDIDATE_UNCHANGED_LIMIT),
    }


def _is_resample_stale_for_coin(
    coin: str,
    *,
    imported_1s: dict[str, dict[str, Any]],
    resampled: dict[str, dict[str, dict[str, Any]]],
) -> bool:
    one_s = imported_1s.get(coin, {})
    if one_s.get("freshness_status") != "advancing":
        return False

    one_s_end = one_s.get("last_timestamp_ms")
    fifteen = resampled.get(coin, {}).get("15s", {})
    if not fifteen.get("exists"):
        return True

    fifteen_end = fifteen.get("last_timestamp_ms")
    if one_s_end and fifteen_end and fifteen_end < one_s_end:
        return True
    return False


def _recommend_phase_audit_action(
    *,
    downloads: dict[str, Any],
    imported_1s: dict[str, dict[str, Any]],
    resampled: dict[str, dict[str, dict[str, Any]]],
    candidate: dict[str, Any],
    coins: list[str],
    scheduler_progress: dict[str, Any] | None = None,
) -> str:
    progress = scheduler_progress or {}
    latest_action = progress.get("latest_action")
    if latest_action == "no_import_progress":
        return "investigate date/hour planning or importer path"
    if latest_action == "import_failed":
        return "investigate importer errors in latest autopilot report"

    has_downloads = int(downloads.get("total_files") or 0) > 0
    any_1s_stale = any(
        imported_1s.get(coin, {}).get("freshness_status") in {"stale", "missing"}
        for coin in coins
    )
    any_1s_advancing = any(
        imported_1s.get(coin, {}).get("freshness_status") == "advancing"
        for coin in coins
    )
    resample_stale = any(
        _is_resample_stale_for_coin(
            coin,
            imported_1s=imported_1s,
            resampled=resampled,
        )
        for coin in coins
    )

    if has_downloads and any_1s_stale:
        return "import step likely failing or not appending"
    if any_1s_advancing and resample_stale:
        return "run historical resample"
    if any_1s_advancing and not resample_stale and candidate.get("repeated_unchanged_warning"):
        return "rerun validator/robustness"
    if not has_downloads:
        return "check autopilot/download reports and AWS access"
    return "continue phase backfill autopilot ticks"


def _build_phase_config(
    autopilot: AutopilotState,
    *,
    chunk_size_hours: int = DEFAULT_CHUNK_SIZE_HOURS,
) -> dict[str, Any]:
    phase_fields = _phase_backfill_status_fields(autopilot, chunk_size_hours=chunk_size_hours)
    return {
        "phase_backfill_enabled": phase_fields["phase_backfill_enabled"],
        "phase_backfill_name": phase_fields["phase_backfill_name"],
        "focused_coins": list(phase_fields["focused_coins"]),
        "target_days": phase_fields["target_days"],
        "daily_file_limit": autopilot.daily_file_limit,
        "daily_chunk_limit": autopilot.daily_chunk_limit,
        "effective_daily_file_capacity": phase_fields["effective_daily_file_capacity"],
        "estimated_real_days_to_complete": phase_fields["estimated_real_days_to_complete"],
        "warnings": list(phase_fields.get("warnings") or []),
    }


def build_phase_backfill_audit(
    *,
    coins: list[str],
    source: str = "historical",
    reports_dir: Path = DEFAULT_RUNS_DIR,
    data_root: Path = orchestrator.DEFAULT_DATA_ROOT,
    downloads_root: Path = DEFAULT_DOWNLOADS_ROOT,
    candidate_reports_dir: Path = DEFAULT_CANDIDATE_VALIDATION_REPORTS_DIR,
    state: SchedulerState | None = None,
    chunk_size_hours: int = DEFAULT_CHUNK_SIZE_HOURS,
) -> dict[str, Any]:
    """Build a read-only Phase 3 progress and import audit report."""
    normalized_source = normalize_source(source)
    imported_1s: dict[str, dict[str, Any]] = {}
    resampled: dict[str, dict[str, dict[str, Any]]] = {}

    for coin in coins:
        csv_path = default_historical_output_path(coin, output_root=data_root)
        imported_1s[coin] = _csv_coverage_entry(csv_path)
        resampled[coin] = {
            "15s": _resampled_coverage_entry(
                coin,
                "15s",
                source=normalized_source,
                data_root=data_root,
            ),
            "1m": _resampled_coverage_entry(
                coin,
                "1m",
                source=normalized_source,
                data_root=data_root,
            ),
        }

    downloads = _summarize_downloads(downloads_root, coins)
    candidate = _build_candidate_freshness(candidate_reports_dir)
    autopilot = state.autopilot if state else AutopilotState()
    scheduler_progress = _build_scheduler_progress(state=state, runs_dir=reports_dir)

    audit: dict[str, Any] = {
        "generated_at": _utc_now_iso(),
        "command": "phase-backfill-audit",
        "paper_only": True,
        "execution_enabled": False,
        "real_execution_ready": False,
        "safety_statement": SAFETY_STATEMENT,
        "baseline_date": PHASE_BACKFILL_BASELINE_DATE,
        "source": normalized_source,
        "coins": coins,
        "phase_config": _build_phase_config(autopilot, chunk_size_hours=chunk_size_hours),
        "scheduler_progress": scheduler_progress,
        "downloads": downloads,
        "imported_historical_1s": imported_1s,
        "resampled_coverage": resampled,
        "candidate_freshness": candidate,
    }
    audit["next_recommended_action"] = _recommend_phase_audit_action(
        downloads=downloads,
        imported_1s=imported_1s,
        resampled=resampled,
        candidate=candidate,
        coins=coins,
        scheduler_progress=scheduler_progress,
    )
    if candidate.get("repeated_unchanged_warning"):
        audit.setdefault("warnings", []).append(
            "Candidate validation results unchanged across recent reports"
        )
    claimed_new = int(scheduler_progress.get("files_downloaded_new") or 0)
    claimed_today = int(scheduler_progress.get("files_downloaded_today") or 0)
    newest_mtime = downloads.get("newest_mtime_iso")
    if (
        (claimed_new > 0 or claimed_today > 0)
        and newest_mtime
        and str(newest_mtime) < "2026-07-04"
        and scheduler_progress.get("latest_action") == "chunk_completed"
    ):
        audit.setdefault("warnings", []).append(
            "Autopilot reports downloads but downloads_newest_mtime is old; "
            "skipped/existing files may have been counted as new downloads"
        )
    return audit


def format_phase_backfill_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Phase Backfill Audit",
        "",
        f"- generated_at: {audit['generated_at']}",
        f"- paper_only: {audit['paper_only']}",
        f"- execution_enabled: {audit['execution_enabled']}",
        f"- real_execution_ready: {audit['real_execution_ready']}",
        f"- baseline_date: {audit['baseline_date']}",
        f"- coins: {', '.join(audit.get('coins') or [])}",
        f"- next_recommended_action: {audit.get('next_recommended_action')}",
        "",
        "## Phase config",
        "",
    ]
    phase = audit.get("phase_config") or {}
    for key in (
        "phase_backfill_enabled",
        "phase_backfill_name",
        "focused_coins",
        "target_days",
        "daily_file_limit",
        "daily_chunk_limit",
        "effective_daily_file_capacity",
        "estimated_real_days_to_complete",
    ):
        value = phase.get(key)
        if value is not None:
            lines.append(f"- {key}: {value}")

    lines.extend(["", "## Scheduler progress", ""])
    scheduler = audit.get("scheduler_progress") or {}
    for key in (
        "latest_autopilot_report_path",
        "latest_action",
        "files_downloaded_today",
        "chunks_completed_today",
        "failed_runs_in_row",
        "paused",
        "latest_coins",
    ):
        lines.append(f"- {key}: {scheduler.get(key)}")

    lines.extend(["", "## Downloads", ""])
    downloads = audit.get("downloads") or {}
    lines.append(f"- total_files: {downloads.get('total_files')}")
    lines.append(f"- newest_mtime_iso: {downloads.get('newest_mtime_iso')}")

    lines.extend(["", "## Imported historical_1s", ""])
    for coin, coverage in (audit.get("imported_historical_1s") or {}).items():
        lines.append(
            f"- {coin}: exists={coverage.get('exists')} "
            f"rows={coverage.get('row_count')} "
            f"last={coverage.get('last_timestamp')} "
            f"freshness={coverage.get('freshness_status')}"
        )

    lines.extend(["", "## Resampled coverage", ""])
    for coin, intervals in (audit.get("resampled_coverage") or {}).items():
        for interval, coverage in intervals.items():
            lines.append(
                f"- {coin} {interval}: exists={coverage.get('exists')} "
                f"rows={coverage.get('row_count')} last={coverage.get('last_timestamp')}"
            )

    lines.extend(["", "## Candidate freshness", ""])
    candidate = audit.get("candidate_freshness") or {}
    for key in (
        "latest_candidate_report_path",
        "latest_run_timestamp",
        "filtered_paper_trades",
        "cumulative_net_return_bps",
        "status",
        "repeated_unchanged_warning",
    ):
        lines.append(f"- {key}: {candidate.get(key)}")

    for warning in audit.get("warnings") or []:
        lines.extend(["", "## Warnings", "", f"- {warning}"])

    lines.extend(["", audit.get("safety_statement", SAFETY_STATEMENT), ""])
    return "\n".join(lines)


def _print_phase_backfill_audit(audit: dict[str, Any]) -> None:
    print("Phase Backfill Audit (read-only)")
    print(f"  generated_at: {audit['generated_at']}")
    print(f"  paper_only: {audit['paper_only']}")
    print(f"  execution_enabled: {audit['execution_enabled']}")
    print(f"  real_execution_ready: {audit['real_execution_ready']}")
    print(f"  coins: {', '.join(audit.get('coins') or [])}")
    print(f"  next_recommended_action: {audit.get('next_recommended_action')}")

    phase = audit.get("phase_config") or {}
    print("  phase_backfill_enabled:", phase.get("phase_backfill_enabled"))
    if phase.get("phase_backfill_name"):
        print(f"  phase_backfill_name: {phase['phase_backfill_name']}")
    if phase.get("focused_coins"):
        print(f"  focused_coins: {', '.join(phase['focused_coins'])}")
    if phase.get("target_days"):
        print(f"  target_days: {phase['target_days']}")
    print(f"  daily_file_limit: {phase.get('daily_file_limit')}")
    print(f"  daily_chunk_limit: {phase.get('daily_chunk_limit')}")
    if phase.get("effective_daily_file_capacity") is not None:
        print(f"  effective_daily_file_capacity: {phase['effective_daily_file_capacity']}")
    if phase.get("estimated_real_days_to_complete") is not None:
        print(
            "  estimated_real_days_to_complete: "
            f"{phase['estimated_real_days_to_complete']:.2f}"
        )

    scheduler = audit.get("scheduler_progress") or {}
    print(f"  latest_autopilot_report: {scheduler.get('latest_autopilot_report_path')}")
    print(f"  latest_action: {scheduler.get('latest_action')}")
    print(f"  files_downloaded_today: {scheduler.get('files_downloaded_today')}")
    print(f"  chunks_completed_today: {scheduler.get('chunks_completed_today')}")

    downloads = audit.get("downloads") or {}
    print(f"  downloads_total_files: {downloads.get('total_files')}")
    print(f"  downloads_newest_mtime: {downloads.get('newest_mtime_iso')}")

    for coin, coverage in (audit.get("imported_historical_1s") or {}).items():
        print(
            f"  {coin} historical_1s: rows={coverage.get('row_count')} "
            f"last={coverage.get('last_timestamp')} "
            f"freshness={coverage.get('freshness_status')}"
        )

    candidate = audit.get("candidate_freshness") or {}
    print(f"  candidate_status: {candidate.get('status')}")
    if candidate.get("repeated_unchanged_warning"):
        print("  warning: candidate results unchanged across recent reports")

    for warning in audit.get("warnings") or []:
        print(f"  warning: {warning}")


def cmd_phase_backfill_audit(
    *,
    coins: list[str] | None = None,
    source: str = "historical",
    reports_dir: Path = DEFAULT_RUNS_DIR,
    data_root: Path = orchestrator.DEFAULT_DATA_ROOT,
    downloads_root: Path = DEFAULT_DOWNLOADS_ROOT,
    candidate_reports_dir: Path = DEFAULT_CANDIDATE_VALIDATION_REPORTS_DIR,
    state_path: Path = DEFAULT_STATE_PATH,
    json_path: Path | None = None,
    markdown_path: Path | None = None,
    chunk_size_hours: int = DEFAULT_CHUNK_SIZE_HOURS,
) -> dict[str, Any]:
    """Read-only audit of Phase 3 backfill progress across downloads, imports, and reports."""
    state = _load_state_optional(state_path)
    autopilot = state.autopilot if state else AutopilotState()
    if coins:
        normalized_coins = orchestrator._normalize_coins(coins)
    elif autopilot.phase_backfill_enabled and autopilot.focused_coins:
        normalized_coins = orchestrator._normalize_coins(autopilot.focused_coins)
    else:
        normalized_coins = orchestrator._normalize_coins(PHASE_BACKFILL_DEFAULT_COINS)

    audit = build_phase_backfill_audit(
        coins=normalized_coins,
        source=source,
        reports_dir=reports_dir,
        data_root=data_root,
        downloads_root=downloads_root,
        candidate_reports_dir=candidate_reports_dir,
        state=state,
        chunk_size_hours=chunk_size_hours,
    )

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(format_phase_backfill_audit_markdown(audit), encoding="utf-8")

    return audit


def cmd_configure_phase_backfill(
    *,
    enable: bool = False,
    disable: bool = False,
    phase_name: str = PHASE_BACKFILL_DEFAULT_NAME,
    coins: list[str] | None = None,
    target_days: int = PHASE_BACKFILL_DEFAULT_TARGET_DAYS,
    daily_file_limit: int = PHASE_BACKFILL_DEFAULT_DAILY_FILE_LIMIT,
    daily_chunk_limit: int = PHASE_BACKFILL_DEFAULT_DAILY_CHUNK_LIMIT,
    max_daily_file_limit: int = PHASE_BACKFILL_DEFAULT_DAILY_FILE_LIMIT,
    max_daily_chunk_limit: int = PHASE_BACKFILL_DEFAULT_DAILY_CHUNK_LIMIT,
    aws_profile: str = DEFAULT_AWS_PROFILE,
    requester_pays: bool = DEFAULT_REQUESTER_PAYS,
    reset_usage: bool = False,
    require_confirmation_text: str | None = None,
    state_path: Path = DEFAULT_STATE_PATH,
) -> dict[str, Any]:
    """Enable or disable targeted phase backfill mode with bounded daily caps."""
    if enable and disable:
        raise ValueError("pass --enable or --disable, not both")
    if not enable and not disable:
        raise ValueError("pass --enable or --disable")

    state = load_state(state_path)
    autopilot = state.autopilot

    if reset_usage:
        _reset_autopilot_daily_counters(autopilot, _utc_today_yyyymmdd())

    if enable:
        if require_confirmation_text != PHASE_BACKFILL_CONFIRMATION_TEXT:
            raise ValueError(
                f"pass --require-confirmation-text with exact text: "
                f"{PHASE_BACKFILL_CONFIRMATION_TEXT!r}"
            )
        normalized_coins = orchestrator._normalize_coins(
            coins or PHASE_BACKFILL_DEFAULT_COINS
        )
        (
            bounded_daily_chunk,
            bounded_daily_file,
            bounded_max_chunk,
            bounded_max_file,
        ) = _bound_phase_backfill_limits(
            daily_chunk_limit=daily_chunk_limit,
            daily_file_limit=daily_file_limit,
            max_daily_chunk_limit=max_daily_chunk_limit,
            max_daily_file_limit=max_daily_file_limit,
        )

        autopilot.enabled = True
        autopilot.paused = False
        autopilot.pause_reason = None
        if autopilot.started_at is None:
            autopilot.started_at = _utc_now_iso()
        autopilot.phase_backfill_enabled = True
        autopilot.phase_backfill_name = phase_name.strip() or PHASE_BACKFILL_DEFAULT_NAME
        autopilot.focused_coins = normalized_coins
        autopilot.target_days = target_days
        autopilot.daily_chunk_limit = bounded_daily_chunk
        autopilot.daily_file_limit = bounded_daily_file
        autopilot.max_daily_chunk_limit = bounded_max_chunk
        autopilot.max_daily_file_limit = bounded_max_file
        autopilot.aws_profile = aws_profile.strip() or DEFAULT_AWS_PROFILE
        autopilot.requester_pays = requester_pays
    else:
        autopilot.phase_backfill_enabled = False
        autopilot.daily_chunk_limit = CONSERVATIVE_DAILY_CHUNK_LIMIT
        autopilot.daily_file_limit = CONSERVATIVE_DAILY_FILE_LIMIT
        autopilot.max_daily_chunk_limit = CONSERVATIVE_MAX_DAILY_CHUNK_LIMIT
        autopilot.max_daily_file_limit = CONSERVATIVE_MAX_DAILY_FILE_LIMIT

    state.autopilot = autopilot
    save_state(state, state_path)

    result: dict[str, Any] = {
        "phase_backfill_enabled": autopilot.phase_backfill_enabled,
        "phase_backfill_name": autopilot.phase_backfill_name,
        "focused_coins": list(autopilot.focused_coins),
        "target_days": autopilot.target_days,
        "daily_chunk_limit": autopilot.daily_chunk_limit,
        "daily_file_limit": autopilot.daily_file_limit,
        "max_daily_chunk_limit": autopilot.max_daily_chunk_limit,
        "max_daily_file_limit": autopilot.max_daily_file_limit,
        "autopilot_enabled": autopilot.enabled,
        "aws_profile": autopilot.aws_profile,
        "requester_pays": autopilot.requester_pays,
    }
    result.update(_phase_backfill_status_fields(autopilot))
    if enable and not autopilot.requester_pays:
        result["warnings"] = list(result.get("warnings") or []) + [REQUESTER_PAYS_DISABLED_WARNING]
    return result


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
        f"- files_requested: {report.files_requested}",
        f"- estimated_files: {report.estimated_files}",
        f"- files_downloaded: {report.files_downloaded}",
        f"- files_downloaded_new: {report.files_downloaded_new}",
        f"- files_skipped_existing: {report.files_skipped_existing}",
        f"- files_imported: {report.files_imported}",
        f"- real_execution_ready: {report.real_execution_ready}",
    ]
    if report.daily_chunk_limit is not None:
        lines.append(f"- daily_chunk_limit: {report.daily_chunk_limit}")
    if report.daily_file_limit is not None:
        lines.append(f"- daily_file_limit: {report.daily_file_limit}")
    if report.effective_daily_file_capacity is not None:
        lines.append(
            f"- effective_daily_file_capacity: {report.effective_daily_file_capacity}"
        )
    if report.chunks_completed_today is not None:
        lines.append(f"- chunks_completed_today: {report.chunks_completed_today}")
    if report.files_downloaded_today is not None:
        lines.append(f"- files_downloaded_today: {report.files_downloaded_today}")
    if report.historical_date:
        lines.append(f"- historical_date: {report.historical_date}")
    if report.historical_hours:
        lines.append(f"- historical_hours: {', '.join(report.historical_hours)}")
    if report.import_progress_by_coin:
        lines.append(
            "- import_progress_by_coin: "
            + ", ".join(f"{coin}={value}" for coin, value in report.import_progress_by_coin.items())
        )
    if report.rows_added_by_coin:
        lines.append(
            "- rows_added_by_coin: "
            + ", ".join(f"{coin}={value}" for coin, value in report.rows_added_by_coin.items())
        )
    if report.train_interval:
        lines.append(f"- train_interval: {report.train_interval}")
    if report.horizon is not None:
        lines.append(f"- horizon: {report.horizon}")
    if report.aws_profile is not None:
        lines.append(f"- aws_profile: {report.aws_profile}")
    if report.requester_pays is not None:
        lines.append(f"- requester_pays: {report.requester_pays}")
    if report.download_error_type:
        lines.append(f"- download_error_type: {report.download_error_type}")
    if report.files_failed:
        lines.append(f"- files_failed: {report.files_failed}")

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
    if report.aws_profile is not None:
        lines.append(f"- aws_profile: {report.aws_profile}")
    if report.requester_pays is not None:
        lines.append(f"- requester_pays: {report.requester_pays}")
    if report.download_error_type:
        lines.append(f"- download_error_type: {report.download_error_type}")
    if report.files_downloaded_new:
        lines.append(f"- files_downloaded_new: {report.files_downloaded_new}")
    if report.files_skipped_existing:
        lines.append(f"- files_skipped_existing: {report.files_skipped_existing}")
    if report.files_failed:
        lines.append(f"- files_failed: {report.files_failed}")

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
    if download.get("downloaded_count") is not None:
        return int(download.get("downloaded_count") or 0)
    file_results = download.get("file_results") or []
    return sum(
        1
        for item in file_results
        if item.get("download_status") == "downloaded" or item.get("status") == "downloaded"
    )


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
    requester_pays: bool | None = None,
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
    resolved_aws_profile, resolved_requester_pays = _resolve_aws_access_settings(
        state.autopilot,
        aws_profile=aws_profile,
        requester_pays=requester_pays,
    )
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
    _populate_aws_access_report_fields(
        report,
        aws_profile=resolved_aws_profile,
        requester_pays=resolved_requester_pays,
    )
    if not resolved_requester_pays:
        report.warnings.append(REQUESTER_PAYS_DISABLED_WARNING)

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
            aws_profile=resolved_aws_profile,
            requester_pays=resolved_requester_pays,
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
    batch_payload = batch_report.details if isinstance(batch_report.details, dict) else {}
    file_stats = _summarize_batch_file_stats(batch_payload)
    report.files_downloaded_new = file_stats["files_downloaded_new"]
    report.files_skipped_existing = file_stats["files_skipped_existing"]
    report.files_failed = file_stats["files_failed"]
    report.download_error_type = _extract_download_error_type(batch_payload)
    download_failures, _, _ = _extract_download_failures(batch_payload)
    if download_failures:
        report.failures.extend(download_failures)

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
    requester_pays: bool | None = None,
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
    requester_pays: bool | None = None,
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
    resolved_aws_profile, resolved_requester_pays = _resolve_aws_access_settings(
        state.autopilot,
        aws_profile=aws_profile,
        requester_pays=requester_pays,
    )
    if state.next_historical_date and _all_hours_completed_for_date(
        state, state.next_historical_date
    ):
        _advance_date_if_complete(state)
        save_state(state, state_path)
        state = load_state(state_path)
    normalized_coins = _effective_scheduler_coins(state, coins)

    report = SchedulerReport(
        workflows_run=["auto-backfill-next"],
        coins=normalized_coins,
        source="historical",
        train_interval=train_interval,
        horizon=horizon,
        details={"chunk_size_hours": chunk_size_hours},
    )
    _populate_aws_access_report_fields(
        report,
        aws_profile=resolved_aws_profile,
        requester_pays=resolved_requester_pays,
    )
    if not resolved_requester_pays:
        report.warnings.append(REQUESTER_PAYS_DISABLED_WARNING)

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

    pre_coverage = _historical_1s_coverage_by_coin(normalized_coins, data_root)

    cycle_report = cmd_daily_cycle(
        coins=normalized_coins,
        historical_hours=[int(h) for h in chunk_hours],
        date=target_date,
        max_files=max_files,
        allow_more=allow_more,
        aws_profile=resolved_aws_profile,
        requester_pays=resolved_requester_pays,
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
    report.files_downloaded_new = cycle_report.files_downloaded_new
    report.files_skipped_existing = cycle_report.files_skipped_existing
    report.files_failed = cycle_report.files_failed
    report.download_error_type = cycle_report.download_error_type
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

    batch_payload = _orchestrator_batch_from_scheduler_report(cycle_report)
    import_progress = _build_import_progress_report(
        coins=normalized_coins,
        data_root=data_root,
        pre_coverage=pre_coverage,
        batch_payload=batch_payload,
    )
    report.details["import_progress"] = import_progress
    import_failures = list(import_progress.get("import_failures") or [])
    download_failures = list(import_progress.get("download_failures") or [])
    download_error_type = import_progress.get("download_error_type")
    files_downloaded_new = int(import_progress.get("files_downloaded_new") or 0)
    files_skipped_existing = int(import_progress.get("files_skipped_existing") or 0)
    files_failed = int(import_progress.get("files_failed") or 0)
    any_import_progress = bool(import_progress.get("any_import_progress"))
    report.files_failed = files_failed
    report.download_error_type = download_error_type

    if download_failures:
        report.failures.extend(download_failures)
        failed_entry = {
            "date": target_date,
            "hours": chunk_hours,
            "failed_at": _utc_now_iso(),
            "reason": "; ".join(download_failures),
            "report_path": json_path_str,
        }
        state.failed_batches.append(failed_entry)
        report.next_recommended_action = _download_failure_next_action(
            aws_profile=resolved_aws_profile,
            requester_pays=resolved_requester_pays,
            download_error_type=str(download_error_type) if download_error_type else None,
        )
    elif import_failures:
        report.failures.extend(import_failures)
        failed_entry = {
            "date": target_date,
            "hours": chunk_hours,
            "failed_at": _utc_now_iso(),
            "reason": "; ".join(import_failures),
            "report_path": json_path_str,
        }
        state.failed_batches.append(failed_entry)
        report.next_recommended_action = (
            "Review importer failures, fix data/tooling, then rerun auto-backfill-next"
        )
    elif _cycle_succeeded(cycle_report) and any_import_progress and not download_failures:
        batch_entry = {
            "date": target_date,
            "hours": chunk_hours,
            "completed_at": _utc_now_iso(),
            "report_path": json_path_str,
        }
        state.completed_batches.append(batch_entry)
        state.total_historical_files_downloaded += files_downloaded_new
        _advance_date_if_complete(state)
        report.next_recommended_action = (
            "python3 bid_ask_research_scheduler.py auto-backfill-next "
            f"--coins {' '.join(normalized_coins)} "
            f"--chunk-size-hours {chunk_size_hours} "
            f"--train-interval {train_interval} --horizon {horizon}"
        )
        if state.next_historical_date is None:
            report.next_recommended_action = "python3 bid_ask_research_scheduler.py status"
    elif _cycle_succeeded(cycle_report):
        report.warnings.append(NO_IMPORT_PROGRESS_MESSAGE)
        if files_downloaded_new == 0 and files_skipped_existing > 0:
            report.warnings.append(
                f"All {files_skipped_existing} file(s) were skipped as existing; "
                "historical_1s did not advance"
            )
        failed_entry = {
            "date": target_date,
            "hours": chunk_hours,
            "failed_at": _utc_now_iso(),
            "reason": "no import progress on historical_1s",
            "report_path": json_path_str,
        }
        state.failed_batches.append(failed_entry)
        report.next_recommended_action = (
            "Investigate date/hour planning or importer path, then rerun auto-backfill-next"
        )
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
        **_phase_backfill_status_fields(autopilot),
    }


def _finalize_autopilot_report(
    report: AutopilotReport,
    state: SchedulerState,
    *,
    state_path: Path,
    runs_dir: Path,
) -> AutopilotReport:
    report.autopilot_snapshot = state.autopilot.to_dict()
    if report.next_recommended_action is None:
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
    requester_pays: bool | None = None,
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
    resolved_aws_profile, resolved_requester_pays = _resolve_aws_access_settings(
        autopilot,
        aws_profile=aws_profile,
        requester_pays=requester_pays,
    )
    normalized_coins = _effective_scheduler_coins(state, coins)
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
    _populate_aws_access_report_fields(
        report,
        aws_profile=resolved_aws_profile,
        requester_pays=resolved_requester_pays,
    )
    if not resolved_requester_pays:
        report.warnings.append(REQUESTER_PAYS_DISABLED_WARNING)

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

    pre_coverage = _historical_1s_coverage_by_coin(normalized_coins, data_root)

    backfill_report = cmd_auto_backfill_next(
        coins=normalized_coins,
        chunk_size_hours=chunk_size_hours,
        max_files=max_files,
        allow_more=allow_more,
        aws_profile=resolved_aws_profile,
        requester_pays=resolved_requester_pays,
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
    report.aws_profile = backfill_report.aws_profile or resolved_aws_profile
    report.requester_pays = (
        backfill_report.requester_pays
        if backfill_report.requester_pays is not None
        else resolved_requester_pays
    )
    report.download_error_type = backfill_report.download_error_type
    report.files_failed = backfill_report.files_failed
    report.details["auto_backfill"] = backfill_report.to_dict()

    stored_progress = (backfill_report.details or {}).get("import_progress")
    batch_payload = _orchestrator_batch_from_scheduler_report(backfill_report)
    if isinstance(stored_progress, dict) and stored_progress:
        import_progress = stored_progress
    else:
        import_progress = _build_import_progress_report(
            coins=normalized_coins,
            data_root=data_root,
            pre_coverage=pre_coverage,
            batch_payload=batch_payload,
        )
    report.details["import_progress"] = import_progress
    _populate_autopilot_import_fields(report, import_progress, requested_files=requested_files)

    _populate_autopilot_report_limits(report, autopilot, requested_files=requested_files)

    if any(AUTO_BACKFILL_COMPLETE_MESSAGE in w for w in backfill_report.warnings):
        report.action = "backfill_complete"
        report.next_recommended_action = backfill_report.next_recommended_action
        state.autopilot = autopilot
        return _finalize_autopilot_report(report, state, state_path=state_path, runs_dir=runs_dir)

    import_failures = list(import_progress.get("import_failures") or [])
    download_failures = list(import_progress.get("download_failures") or [])
    download_error_type = import_progress.get("download_error_type")
    any_import_progress = bool(import_progress.get("any_import_progress"))
    files_downloaded_new = int(import_progress.get("files_downloaded_new") or 0)
    files_skipped_existing = int(import_progress.get("files_skipped_existing") or 0)
    chunk_succeeded = _cycle_succeeded(backfill_report) and not backfill_report.failures
    if download_error_type:
        report.download_error_type = str(download_error_type)

    if download_failures:
        autopilot.failed_runs_in_row += 1
        autopilot.last_run_date = today
        report.action = "download_failed"
        report.failures.extend(download_failures)
        if autopilot.failed_runs_in_row >= AUTOPILOT_FAILURE_PAUSE_THRESHOLD:
            autopilot.paused = True
            autopilot.pause_reason = (
                f"Paused after {autopilot.failed_runs_in_row} consecutive failed autopilot runs"
            )
            report.warnings.append(autopilot.pause_reason)
        report.next_recommended_action = _download_failure_next_action(
            aws_profile=resolved_aws_profile,
            requester_pays=resolved_requester_pays,
            download_error_type=str(download_error_type) if download_error_type else None,
        )
    elif import_failures:
        autopilot.failed_runs_in_row += 1
        autopilot.last_run_date = today
        report.action = "import_failed"
        report.failures.extend(import_failures)
        if autopilot.failed_runs_in_row >= AUTOPILOT_FAILURE_PAUSE_THRESHOLD:
            autopilot.paused = True
            autopilot.pause_reason = (
                f"Paused after {autopilot.failed_runs_in_row} consecutive failed autopilot runs"
            )
            report.warnings.append(autopilot.pause_reason)
        report.next_recommended_action = (
            "Review importer failures, fix data/tooling, then "
            "configure-autopilot --resume and rerun autopilot-tick"
        )
    elif chunk_succeeded and any_import_progress and not download_failures:
        autopilot.runs_today += 1
        autopilot.chunks_completed_today += 1
        autopilot.files_downloaded_today += files_downloaded_new
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
    elif chunk_succeeded:
        report.action = "no_import_progress"
        report.warnings.append(NO_IMPORT_PROGRESS_MESSAGE)
        if files_downloaded_new == 0 and files_skipped_existing > 0:
            report.warnings.append(
                f"All {files_skipped_existing} file(s) were skipped as existing; "
                "historical_1s did not advance"
            )
        report.next_recommended_action = (
            "Investigate date/hour planning or importer path, then rerun autopilot-tick"
        )
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


def _dashboard_phase_audit_paths(
    state_dir: Path = DEFAULT_STATE_DIR,
) -> tuple[Path, Path]:
    return (
        state_dir / PHASE3_AUDIT_JSON_PATH.name,
        state_dir / PHASE3_AUDIT_MARKDOWN_PATH.name,
    )


def _resolve_research_session_coins(
    state_path: Path,
    coins: list[str] | None,
) -> list[str]:
    state = _load_state_optional(state_path)
    autopilot = state.autopilot if state else AutopilotState()
    if coins:
        return orchestrator._normalize_coins(coins)
    if autopilot.phase_backfill_enabled and autopilot.focused_coins:
        return orchestrator._normalize_coins(autopilot.focused_coins)
    return orchestrator._normalize_coins(PHASE_BACKFILL_DEFAULT_COINS)


def _compute_research_session_ticks_planned(
    duration_hours: float,
    tick_interval_minutes: int,
    *,
    max_ticks: int | None = None,
) -> int:
    if duration_hours <= 0:
        raise ValueError("duration-hours must be positive")
    if tick_interval_minutes < 1:
        raise ValueError("tick-interval-minutes must be at least 1")
    planned = max(1, int(duration_hours * 60 // tick_interval_minutes))
    if max_ticks is not None:
        if max_ticks < 1:
            raise ValueError("max-ticks must be at least 1 when set")
        planned = min(planned, max_ticks)
    return planned


def _research_session_should_stop(state_path: Path) -> tuple[bool, str | None]:
    state = _load_state_optional(state_path)
    if state is None:
        return False, None
    autopilot = state.autopilot
    if autopilot.paused:
        return True, f"Autopilot paused: {autopilot.pause_reason or 'unspecified'}"
    if autopilot.failed_runs_in_row >= AUTOPILOT_FAILURE_PAUSE_THRESHOLD:
        return True, (
            f"Autopilot failed_runs_in_row={autopilot.failed_runs_in_row} "
            f">= {AUTOPILOT_FAILURE_PAUSE_THRESHOLD}"
        )
    return False, None


def _historical_1s_coverage_advanced(
    pre_audit: dict[str, Any],
    post_audit: dict[str, Any],
    *,
    coins: list[str] | None = None,
) -> bool:
    pre = pre_audit.get("imported_historical_1s") or {}
    post = post_audit.get("imported_historical_1s") or {}
    watch_coins = coins or RESEARCH_SESSION_RESAMPLE_COINS
    for coin in watch_coins:
        pre_last = (pre.get(coin) or {}).get("last_timestamp_ms")
        post_last = (post.get(coin) or {}).get("last_timestamp_ms")
        if post_last is not None and (pre_last is None or int(post_last) > int(pre_last)):
            return True
    return False


def _extract_coin_latest_timestamps(
    audit: dict[str, Any],
) -> tuple[str | None, str | None]:
    imported = audit.get("imported_historical_1s") or {}
    btc = (imported.get("BTC") or {}).get("last_timestamp")
    eth = (imported.get("ETH") or {}).get("last_timestamp")
    return btc, eth


def _build_research_session_tick_summary(
    *,
    tick_index: int,
    started_at: str,
    ended_at: str,
    pre_audit: dict[str, Any],
    post_audit: dict[str, Any],
    autopilot_report: AutopilotReport,
    candidate_report: dict[str, Any] | None = None,
    relaxed_report: dict[str, Any] | None = None,
    resample_ran: bool = False,
    warnings: list[str] | None = None,
    failures: list[str] | None = None,
) -> dict[str, Any]:
    post_btc_ts, post_eth_ts = _extract_coin_latest_timestamps(post_audit)
    summary: dict[str, Any] = {
        "tick_index": tick_index,
        "started_at": started_at,
        "ended_at": ended_at,
        "autopilot_action": autopilot_report.action,
        "files_downloaded_new": autopilot_report.files_downloaded_new,
        "files_skipped_existing": autopilot_report.files_skipped_existing,
        "files_failed": autopilot_report.files_failed,
        "rows_added_by_coin": dict(autopilot_report.rows_added_by_coin),
        "timestamp_advanced_by_coin": dict(autopilot_report.timestamp_advanced_by_coin),
        "btc_latest_timestamp": post_btc_ts,
        "eth_latest_timestamp": post_eth_ts,
        "historical_1s_advanced": _historical_1s_coverage_advanced(
            pre_audit,
            post_audit,
            coins=RESEARCH_SESSION_RESAMPLE_COINS,
        ),
        "candidate_trades": None,
        "candidate_cumulative_net_return_bps": None,
        "relaxed_trades": None,
        "relaxed_cumulative_net_return_bps": None,
        "resample_validate_ran": resample_ran,
        "warnings": list(warnings or []),
        "failures": list(failures or []),
    }
    if candidate_report is not None:
        summary["candidate_trades"] = int(candidate_report.get("filtered_paper_trades") or 0)
        summary["candidate_cumulative_net_return_bps"] = float(
            candidate_report.get("cumulative_net_return_bps") or 0.0
        )
    if relaxed_report is not None:
        summary["relaxed_trades"] = int(relaxed_report.get("filtered_paper_trades") or 0)
        summary["relaxed_cumulative_net_return_bps"] = float(
            relaxed_report.get("cumulative_net_return_bps") or 0.0
        )
    return summary


def format_research_session_markdown(session: dict[str, Any]) -> str:
    lines = [
        "# Research Session Report",
        "",
        f"- session_id: {session.get('session_id')}",
        f"- started_at: {session.get('started_at')}",
        f"- ended_at: {session.get('ended_at')}",
        f"- duration_hours: {session.get('duration_hours')}",
        f"- tick_interval_minutes: {session.get('tick_interval_minutes')}",
        f"- ticks_planned: {session.get('ticks_planned')}",
        f"- ticks_completed: {session.get('ticks_completed')}",
        f"- coins: {', '.join(session.get('coins') or [])}",
        f"- source: {session.get('source')}",
        f"- paper_only: {session.get('paper_only')}",
        f"- execution_enabled: {session.get('execution_enabled')}",
        f"- real_execution_ready: {session.get('real_execution_ready')}",
        f"- dry_run: {session.get('dry_run', False)}",
        f"- stop_reason: {session.get('stop_reason')}",
        f"- next_recommended_action: {session.get('next_recommended_action')}",
        "",
        "## Per-tick summaries",
        "",
    ]
    for tick in session.get("per_tick_summaries") or []:
        lines.extend(
            [
                f"### Tick {tick.get('tick_index')}",
                "",
                f"- started_at: {tick.get('started_at')}",
                f"- ended_at: {tick.get('ended_at')}",
                f"- autopilot_action: {tick.get('autopilot_action')}",
                f"- files_downloaded_new: {tick.get('files_downloaded_new')}",
                f"- files_skipped_existing: {tick.get('files_skipped_existing')}",
                f"- files_failed: {tick.get('files_failed')}",
                f"- historical_1s_advanced: {tick.get('historical_1s_advanced')}",
                f"- candidate_trades: {tick.get('candidate_trades')}",
                f"- candidate_cumulative_net_return_bps: {tick.get('candidate_cumulative_net_return_bps')}",
                f"- relaxed_trades: {tick.get('relaxed_trades')}",
                f"- relaxed_cumulative_net_return_bps: {tick.get('relaxed_cumulative_net_return_bps')}",
                "",
            ]
        )
        for warning in tick.get("warnings") or []:
            lines.append(f"- warning: {warning}")
        for failure in tick.get("failures") or []:
            lines.append(f"- failure: {failure}")
        lines.append("")

    final_summary = session.get("final_summary") or {}
    if final_summary:
        lines.extend(["## Final summary", ""])
        for key, value in final_summary.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    lines.extend(["", session.get("safety_statement", SAFETY_STATEMENT), ""])
    return "\n".join(lines)


def write_research_session_report(
    session: dict[str, Any],
    *,
    sessions_dir: Path = DEFAULT_RESEARCH_SESSIONS_DIR,
) -> tuple[Path, Path]:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_id = str(session["session_id"])
    json_path = sessions_dir / f"research_session_{session_id}.json"
    md_path = sessions_dir / f"research_session_{session_id}.md"
    json_path.write_text(json.dumps(session, indent=2), encoding="utf-8")
    md_path.write_text(format_research_session_markdown(session), encoding="utf-8")
    session["session_report_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    return json_path, md_path


def _run_relaxed_candidate_b_validation(
    *,
    source: str,
    data_root: Path,
    walk_forward_reports_dir: Path,
    validation_reports_dir: Path,
    walk_forward_runner: Any | None = None,
) -> dict[str, Any]:
    return candidate_validator.run_candidate_validation(
        coin="BTC",
        interval="15s",
        horizon=RELAXED_CANDIDATE_B_HORIZON,
        min_confidence=RELAXED_CANDIDATE_B_MIN_CONFIDENCE,
        min_edge_bps=RELAXED_CANDIDATE_B_MIN_EDGE_BPS,
        cost_multiplier=RELAXED_CANDIDATE_B_COST_MULTIPLIER,
        max_spread_bps=RELAXED_CANDIDATE_B_MAX_SPREAD_BPS,
        min_trades_required=candidate_validator.DEFAULT_MIN_TRADES_REQUIRED,
        require_positive=False,
        candidate_name=RELAXED_CANDIDATE_B_NAME,
        source=source,
        data_root=data_root,
        walk_forward_reports_dir=walk_forward_reports_dir,
        validation_reports_dir=validation_reports_dir,
        walk_forward_runner=walk_forward_runner,
    )


def _execute_research_session_tick(
    *,
    tick_index: int,
    coins: list[str],
    source: str,
    aws_profile: str,
    requester_pays: bool,
    run_candidate_tests: bool,
    run_relaxed_candidate_test: bool,
    update_dashboard_artifacts: bool,
    state_path: Path,
    runs_dir: Path,
    data_root: Path,
    downloads_root: Path,
    candidate_reports_dir: Path,
    orchestrator_report_dir: Path,
    chunk_size_hours: int = DEFAULT_CHUNK_SIZE_HOURS,
    audit_json_path: Path | None = None,
    audit_markdown_path: Path | None = None,
) -> dict[str, Any]:
    started_at = _utc_now_iso()
    tick_warnings: list[str] = []
    tick_failures: list[str] = []

    json_path = audit_json_path if update_dashboard_artifacts else None
    markdown_path = audit_markdown_path if update_dashboard_artifacts else None

    pre_audit = cmd_phase_backfill_audit(
        coins=coins,
        source=source,
        reports_dir=runs_dir,
        data_root=data_root,
        downloads_root=downloads_root,
        candidate_reports_dir=candidate_reports_dir,
        state_path=state_path,
        json_path=json_path,
        markdown_path=markdown_path,
        chunk_size_hours=chunk_size_hours,
    )

    autopilot_report = cmd_autopilot_tick(
        coins=coins,
        chunk_size_hours=chunk_size_hours,
        aws_profile=aws_profile,
        requester_pays=requester_pays,
        data_root=data_root,
        orchestrator_report_dir=orchestrator_report_dir,
        state_path=state_path,
        runs_dir=runs_dir,
    )

    post_audit = cmd_phase_backfill_audit(
        coins=coins,
        source=source,
        reports_dir=runs_dir,
        data_root=data_root,
        downloads_root=downloads_root,
        candidate_reports_dir=candidate_reports_dir,
        state_path=state_path,
        json_path=json_path,
        markdown_path=markdown_path,
        chunk_size_hours=chunk_size_hours,
    )

    resample_ran = False
    if _historical_1s_coverage_advanced(pre_audit, post_audit, coins=RESEARCH_SESSION_RESAMPLE_COINS):
        resample_report = orchestrator.cmd_resample_validate(
            coins=RESEARCH_SESSION_RESAMPLE_COINS,
            source=source,
            data_root=data_root,
            report_dir=orchestrator_report_dir,
        )
        resample_ran = True
        tick_warnings.extend(resample_report.warnings)
        tick_failures.extend(resample_report.failures)

    candidate_report: dict[str, Any] | None = None
    if run_candidate_tests:
        try:
            candidate_report = candidate_validator.run_candidate_validation(
                source=source,
                data_root=data_root,
                validation_reports_dir=candidate_reports_dir,
            )
            candidate_validator.summarize_candidate_validation_reports(
                reports_dir=candidate_reports_dir,
            )
        except Exception as exc:  # noqa: BLE001
            tick_failures.append(f"candidate validation failed: {exc}")

    relaxed_report: dict[str, Any] | None = None
    if run_relaxed_candidate_test:
        try:
            relaxed_report = _run_relaxed_candidate_b_validation(
                source=source,
                data_root=data_root,
                walk_forward_reports_dir=orchestrator_report_dir,
                validation_reports_dir=candidate_reports_dir,
            )
        except Exception as exc:  # noqa: BLE001
            tick_failures.append(f"relaxed candidate validation failed: {exc}")

    ended_at = _utc_now_iso()
    return _build_research_session_tick_summary(
        tick_index=tick_index,
        started_at=started_at,
        ended_at=ended_at,
        pre_audit=pre_audit,
        post_audit=post_audit,
        autopilot_report=autopilot_report,
        candidate_report=candidate_report,
        relaxed_report=relaxed_report,
        resample_ran=resample_ran,
        warnings=tick_warnings,
        failures=tick_failures,
    )


def cmd_research_session(
    *,
    duration_hours: float = DEFAULT_RESEARCH_SESSION_DURATION_HOURS,
    tick_interval_minutes: int = DEFAULT_RESEARCH_SESSION_TICK_INTERVAL_MINUTES,
    coins: list[str] | None = None,
    source: str = "historical",
    aws_profile: str | None = None,
    requester_pays: bool = DEFAULT_REQUESTER_PAYS,
    run_candidate_tests: bool = True,
    run_relaxed_candidate_test: bool = True,
    update_dashboard_artifacts: bool = True,
    max_ticks: int | None = None,
    dry_run: bool = False,
    state_path: Path = DEFAULT_STATE_PATH,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    sessions_dir: Path = DEFAULT_RESEARCH_SESSIONS_DIR,
    data_root: Path = orchestrator.DEFAULT_DATA_ROOT,
    downloads_root: Path = DEFAULT_DOWNLOADS_ROOT,
    candidate_reports_dir: Path = DEFAULT_CANDIDATE_VALIDATION_REPORTS_DIR,
    orchestrator_report_dir: Path = orchestrator.DEFAULT_REPORT_DIR,
    scheduler_state_dir: Path = DEFAULT_STATE_DIR,
    chunk_size_hours: int = DEFAULT_CHUNK_SIZE_HOURS,
    sleep_fn: Any | None = None,
    now_fn: Any | None = None,
) -> dict[str, Any]:
    """Run a bounded paper-only research session with periodic safe research ticks."""
    normalized_coins = _resolve_research_session_coins(state_path, coins)
    normalized_source = normalize_source(source)
    ticks_planned = _compute_research_session_ticks_planned(
        duration_hours,
        tick_interval_minutes,
        max_ticks=max_ticks,
    )
    started_at = _utc_now_iso()
    session_id = started_at.replace(":", "").replace("-", "").replace(".", "")[:15]

    state = _load_state_optional(state_path)
    autopilot = state.autopilot if state else AutopilotState()
    resolved_aws_profile, resolved_requester_pays = _resolve_aws_access_settings(
        autopilot,
        aws_profile=aws_profile,
        requester_pays=requester_pays,
    )
    audit_json_path, audit_markdown_path = _dashboard_phase_audit_paths(scheduler_state_dir)

    session: dict[str, Any] = {
        "session_id": session_id,
        "started_at": started_at,
        "ended_at": None,
        "duration_hours": duration_hours,
        "tick_interval_minutes": tick_interval_minutes,
        "ticks_planned": ticks_planned,
        "ticks_completed": 0,
        "coins": normalized_coins,
        "source": normalized_source,
        "paper_only": True,
        "execution_enabled": False,
        "real_execution_ready": False,
        "safety_statement": SAFETY_STATEMENT,
        "dry_run": dry_run,
        "update_dashboard_artifacts": update_dashboard_artifacts,
        "dashboard_audit_paths": {
            "json": str(audit_json_path),
            "markdown": str(audit_markdown_path),
        },
        "aws_profile": resolved_aws_profile,
        "requester_pays": resolved_requester_pays,
        "run_candidate_tests": run_candidate_tests,
        "run_relaxed_candidate_test": run_relaxed_candidate_test,
        "per_tick_summaries": [],
        "stop_reason": None,
        "final_summary": {},
        "next_recommended_action": "continue research-session on schedule",
    }

    if dry_run:
        session["ended_at"] = _utc_now_iso()
        session["planned_tick_actions"] = [
            "phase-backfill-audit -> dashboard artifacts",
            "autopilot-tick",
            "phase-backfill-audit -> dashboard artifacts",
            "conditional resample/validate BTC ETH historical 1s/15s",
            "conditional candidate validation run + summarize",
            "conditional relaxed Candidate B validation",
        ]
        session["final_summary"] = {
            "mode": "dry_run",
            "ticks_planned": ticks_planned,
            "actions_executed": 0,
        }
        session["next_recommended_action"] = (
            "Remove --dry-run to execute the planned research session ticks"
        )
        write_research_session_report(session, sessions_dir=sessions_dir)
        return session

    sleep = sleep_fn or time.sleep
    now = now_fn or (lambda: datetime.now(timezone.utc))
    end_at = now() + timedelta(hours=duration_hours)
    per_tick_summaries: list[dict[str, Any]] = []
    stop_reason: str | None = None

    for tick_index in range(ticks_planned):
        if now() >= end_at:
            stop_reason = "Session duration elapsed"
            break

        should_stop, stop_message = _research_session_should_stop(state_path)
        if should_stop:
            stop_reason = stop_message
            break

        tick_summary = _execute_research_session_tick(
            tick_index=tick_index,
            coins=normalized_coins,
            source=normalized_source,
            aws_profile=resolved_aws_profile,
            requester_pays=resolved_requester_pays,
            run_candidate_tests=run_candidate_tests,
            run_relaxed_candidate_test=run_relaxed_candidate_test,
            update_dashboard_artifacts=update_dashboard_artifacts,
            state_path=state_path,
            runs_dir=runs_dir,
            data_root=data_root,
            downloads_root=downloads_root,
            candidate_reports_dir=candidate_reports_dir,
            orchestrator_report_dir=orchestrator_report_dir,
            chunk_size_hours=chunk_size_hours,
            audit_json_path=audit_json_path,
            audit_markdown_path=audit_markdown_path,
        )
        per_tick_summaries.append(tick_summary)

        should_stop, stop_message = _research_session_should_stop(state_path)
        if should_stop:
            stop_reason = stop_message
            break

        if tick_index < ticks_planned - 1 and now() < end_at:
            remaining = (end_at - now()).total_seconds()
            sleep_seconds = min(tick_interval_minutes * 60, remaining)
            if sleep_seconds > 0:
                sleep(sleep_seconds)

    session["ticks_completed"] = len(per_tick_summaries)
    session["per_tick_summaries"] = per_tick_summaries
    session["ended_at"] = _utc_now_iso()
    session["stop_reason"] = stop_reason

    if stop_reason:
        session["next_recommended_action"] = (
            "Review stop_reason and autopilot status; "
            "configure-autopilot --resume if needed before next session"
        )
    elif session["ticks_completed"] < ticks_planned:
        session["next_recommended_action"] = "Session ended early; rerun research-session if needed"
    else:
        session["next_recommended_action"] = "Review session report and schedule next session"

    last_tick = per_tick_summaries[-1] if per_tick_summaries else {}
    session["final_summary"] = {
        "ticks_completed": session["ticks_completed"],
        "last_autopilot_action": last_tick.get("autopilot_action"),
        "last_btc_latest_timestamp": last_tick.get("btc_latest_timestamp"),
        "last_eth_latest_timestamp": last_tick.get("eth_latest_timestamp"),
        "total_warnings": sum(len(t.get("warnings") or []) for t in per_tick_summaries),
        "total_failures": sum(len(t.get("failures") or []) for t in per_tick_summaries),
    }

    write_research_session_report(session, sessions_dir=sessions_dir)
    return session


def _detect_python_path(python_path: str | None = None) -> str:
    if python_path:
        return python_path
    venv_candidate = Path(sys.prefix) / "bin" / "python3"
    if venv_candidate.is_file():
        return str(venv_candidate)
    return sys.executable


def cmd_write_research_session_launchd_template(
    *,
    label_prefix: str = DEFAULT_RESEARCH_SESSION_LABEL_PREFIX,
    morning_hour: int = DEFAULT_RESEARCH_SESSION_MORNING_HOUR,
    evening_hour: int = DEFAULT_RESEARCH_SESSION_EVENING_HOUR,
    duration_hours: float = DEFAULT_RESEARCH_SESSION_DURATION_HOURS,
    tick_interval_minutes: int = DEFAULT_RESEARCH_SESSION_TICK_INTERVAL_MINUTES,
    output_dir: Path = DEFAULT_RESEARCH_SESSION_OUTPUT_DIR,
    python_path: str | None = None,
    project_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Write morning/evening launchd plist templates for research-session (does not install)."""
    if not 0 <= morning_hour <= 23:
        raise ValueError("morning-hour must be between 0 and 23")
    if not 0 <= evening_hour <= 23:
        raise ValueError("evening-hour must be between 0 and 23")
    if duration_hours <= 0:
        raise ValueError("duration-hours must be positive")
    if tick_interval_minutes < 1:
        raise ValueError("tick-interval-minutes must be at least 1")

    root = Path(project_dir or Path.cwd()).resolve()
    resolved_python = _detect_python_path(python_path)
    scheduler_script = root / "bid_ask_research_scheduler.py"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd_parts = [
        resolved_python,
        str(scheduler_script),
        "research-session",
        "--duration-hours",
        str(duration_hours),
        "--tick-interval-minutes",
        str(tick_interval_minutes),
        "--requester-pays",
        "true",
        "--run-candidate-tests",
        "true",
        "--run-relaxed-candidate-test",
        "true",
        "--update-dashboard-artifacts",
        "true",
    ]

    stdout_log = DEFAULT_STATE_DIR / "research_session_launchd.out.log"
    stderr_log = DEFAULT_STATE_DIR / "research_session_launchd.err.log"

    def write_plist(label: str, hour: int, filename: str) -> Path:
        plist_path = output_dir / filename
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
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{hour}</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{stdout_log}</string>
    <key>StandardErrorPath</key>
    <string>{stderr_log}</string>
</dict>
</plist>
"""
        plist_path.write_text(plist_content, encoding="utf-8")
        return plist_path

    morning_label = f"{label_prefix}.morning"
    evening_label = f"{label_prefix}.evening"
    morning_plist = write_plist(morning_label, morning_hour, LAUNCHD_RESEARCH_SESSION_MORNING_PLIST)
    evening_plist = write_plist(evening_label, evening_hour, LAUNCHD_RESEARCH_SESSION_EVENING_PLIST)

    setup_md = output_dir / LAUNCHD_RESEARCH_SESSION_SETUP_MD
    setup_lines = [
        "# macOS launchd Setup for Paper-Only Research Sessions",
        "",
        "These templates run `research-session` once per schedule tick. "
        "Each invocation executes a **bounded paper-only research loop** and exits. "
        "They are **not installed automatically**.",
        "",
        "Research-only, paper-only; no orders, wallets, or execution endpoints.",
        "",
        "## Files",
        "",
        f"- Morning plist: `{morning_plist}`",
        f"- Evening plist: `{evening_plist}`",
        f"- Session reports: `{DEFAULT_RESEARCH_SESSIONS_DIR}/`",
        f"- Dashboard audit JSON: `{PHASE3_AUDIT_JSON_PATH}`",
        "",
        "## Install morning session",
        "",
        "```bash",
        f"cp {morning_plist} ~/Library/LaunchAgents/",
        f"launchctl unload ~/Library/LaunchAgents/{LAUNCHD_RESEARCH_SESSION_MORNING_PLIST} 2>/dev/null || true",
        f"launchctl load ~/Library/LaunchAgents/{LAUNCHD_RESEARCH_SESSION_MORNING_PLIST}",
        "```",
        "",
        "## Install evening session",
        "",
        "```bash",
        f"cp {evening_plist} ~/Library/LaunchAgents/",
        f"launchctl unload ~/Library/LaunchAgents/{LAUNCHD_RESEARCH_SESSION_EVENING_PLIST} 2>/dev/null || true",
        f"launchctl load ~/Library/LaunchAgents/{LAUNCHD_RESEARCH_SESSION_EVENING_PLIST}",
        "```",
        "",
        "## Check status",
        "",
        "```bash",
        "launchctl list | grep hyperliquid.research.session",
        "python3 bid_ask_research_scheduler.py autopilot-status",
        "```",
        "",
        "## Uninstall",
        "",
        "```bash",
        f"launchctl unload ~/Library/LaunchAgents/{LAUNCHD_RESEARCH_SESSION_MORNING_PLIST}",
        f"launchctl unload ~/Library/LaunchAgents/{LAUNCHD_RESEARCH_SESSION_EVENING_PLIST}",
        f"rm ~/Library/LaunchAgents/{LAUNCHD_RESEARCH_SESSION_MORNING_PLIST}",
        f"rm ~/Library/LaunchAgents/{LAUNCHD_RESEARCH_SESSION_EVENING_PLIST}",
        "```",
        "",
        f"Morning schedule: **{morning_hour:02d}:00** local time (`{morning_label}`).",
        f"Evening schedule: **{evening_hour:02d}:00** local time (`{evening_label}`).",
        f"Session duration: **{duration_hours}** hours; tick interval: **{tick_interval_minutes}** minutes.",
    ]
    setup_md.write_text("\n".join(setup_lines) + "\n", encoding="utf-8")
    return morning_plist, evening_plist, setup_md


def _resolve_project_root(project_root: Path | None = None) -> Path:
    """Resolve repository root from explicit path or this module's location."""
    if project_root is not None:
        return project_root.resolve()
    return Path(__file__).resolve().parent


def _parse_bool_arg(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def _parse_cli_requester_pays(args: argparse.Namespace) -> bool | None:
    """Return explicit CLI requester_pays override, or None to use autopilot state."""
    if getattr(args, "no_requester_pays", False):
        return False
    if getattr(args, "requester_pays", None):
        return True
    return None


def _parse_configure_phase_requester_pays(args: argparse.Namespace) -> bool:
    """Return requester_pays for configure-phase-backfill (default enabled)."""
    if getattr(args, "no_requester_pays", False):
        return False
    if getattr(args, "requester_pays", None):
        return True
    return DEFAULT_REQUESTER_PAYS


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
    root = _resolve_project_root(project_root)
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


def cmd_write_hourly_launchd_template(
    *,
    interval_seconds: int = DEFAULT_HOURLY_INTERVAL_SECONDS,
    label: str = DEFAULT_HOURLY_LABEL,
    run_at_load: bool = True,
    stdout_log: Path = DEFAULT_HOURLY_STDOUT_LOG,
    stderr_log: Path = DEFAULT_HOURLY_STDERR_LOG,
    state_dir: Path = DEFAULT_STATE_DIR,
    project_root: Path | None = None,
) -> tuple[Path, Path]:
    """Write hourly macOS launchd plist for autopilot-tick (does not install)."""
    if interval_seconds < 60:
        raise ValueError("interval-seconds must be at least 60")
    root = _resolve_project_root(project_root)
    python_path = sys.executable
    scheduler_script = root / "bid_ask_research_scheduler.py"
    cmd_parts = [python_path, str(scheduler_script), "autopilot-tick"]

    plist_name = LAUNCHD_PLIST_NAME_AUTOPILOT_HOURLY
    plist_path = state_dir / plist_name
    state_dir.mkdir(parents=True, exist_ok=True)

    run_at_load_block = ""
    if run_at_load:
        run_at_load_block = """    <key>RunAtLoad</key>
    <true/>
"""

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
    <key>StartInterval</key>
    <integer>{interval_seconds}</integer>
{run_at_load_block}    <key>StandardOutPath</key>
    <string>{stdout_log}</string>
    <key>StandardErrorPath</key>
    <string>{stderr_log}</string>
</dict>
</plist>
"""
    plist_path.write_text(plist_content, encoding="utf-8")

    setup_md = state_dir / LAUNCHD_SETUP_MD_HOURLY
    setup_lines = [
        "# macOS launchd Setup for Hourly Autopilot Extractor",
        "",
        "This template runs `autopilot-tick` on a fixed interval (default: every hour). "
        "Each invocation processes **at most one bounded chunk** and exits. "
        "It is **not installed automatically**.",
        "",
        "Research-only, paper-only; no orders, wallets, or execution endpoints.",
        "",
        "## 1. Configure safe daily caps",
        "",
        "Daily caps protect the hourly schedule from runaway downloads. "
        "Without them, hourly ticks could accumulate excessive AWS requester-pays charges.",
        "",
        "Recommended safe caps for hourly extraction:",
        "",
        f"- `daily_chunk_limit`: **{HOURLY_AUTOPILOT_RECOMMENDED_DAILY_CHUNK_LIMIT}**",
        f"- `daily_file_limit`: **{HOURLY_AUTOPILOT_RECOMMENDED_DAILY_FILE_LIMIT}**",
        "- Keep `max_daily_chunk_limit` and `max_daily_file_limit` capped (defaults are safe).",
        "",
        "```bash",
        "python3 bid_ask_research_scheduler.py configure-autopilot \\",
        "  --enable \\",
        f"  --daily-chunk-limit {HOURLY_AUTOPILOT_RECOMMENDED_DAILY_CHUNK_LIMIT} \\",
        f"  --daily-file-limit {HOURLY_AUTOPILOT_RECOMMENDED_DAILY_FILE_LIMIT}",
        "```",
        "",
        "Set up **AWS Budget alerts** before enabling scheduled downloads.",
        "",
        "## 2. Copy plist to LaunchAgents",
        "",
        "```bash",
        f"cp {plist_path} ~/Library/LaunchAgents/",
        "```",
        "",
        "## 3. Unload old daily plist (if present)",
        "",
        "```bash",
        f"launchctl unload ~/Library/LaunchAgents/{LAUNCHD_PLIST_NAME_AUTOPILOT} 2>/dev/null || true",
        f"launchctl unload ~/Library/LaunchAgents/{plist_name} 2>/dev/null || true",
        "```",
        "",
        "## 4. Load hourly plist",
        "",
        "```bash",
        f"launchctl load ~/Library/LaunchAgents/{plist_name}",
        "```",
        "",
        "## 5. Check launchctl status",
        "",
        "```bash",
        "launchctl list | grep hyperliquid",
        "```",
        "",
        "## 6. Check autopilot status",
        "",
        "```bash",
        "python3 bid_ask_research_scheduler.py autopilot-status",
        "```",
        "",
        "## 7. Inspect stdout/stderr logs",
        "",
        "```bash",
        f"tail -50 {stdout_log}",
        f"tail -50 {stderr_log}",
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
        "- Each hourly tick runs `autopilot-tick`, which enforces daily chunk/file limits.",
        "- When daily caps are reached, ticks exit cleanly with a limit-skipped report.",
        "- **Daily caps protect the hourly schedule** — do not disable or raise them without review.",
        "- Archive availability is not guaranteed.",
        "- Use AWS Budget alerts; requester-pays downloads incur charges.",
        "",
        f"Schedule: every **{interval_seconds}** seconds (`StartInterval`).",
        f"RunAtLoad: **{run_at_load}**.",
        f"Label: `{label}`.",
    ]
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
    backfill_requester_pays = backfill.add_mutually_exclusive_group()
    backfill_requester_pays.add_argument("--requester-pays", action="store_true", default=None)
    backfill_requester_pays.add_argument("--no-requester-pays", action="store_true", default=None)
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
    auto_backfill_requester_pays = auto_backfill.add_mutually_exclusive_group()
    auto_backfill_requester_pays.add_argument("--requester-pays", action="store_true", default=None)
    auto_backfill_requester_pays.add_argument("--no-requester-pays", action="store_true", default=None)
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

    hourly_launchd = subparsers.add_parser(
        "write-hourly-launchd-template",
        help="Write hourly launchd plist for autopilot-tick (does not install)",
    )
    hourly_launchd.add_argument(
        "--interval-seconds",
        type=int,
        default=DEFAULT_HOURLY_INTERVAL_SECONDS,
        help="StartInterval in seconds (default 3600)",
    )
    hourly_launchd.add_argument(
        "--label",
        default=DEFAULT_HOURLY_LABEL,
        help="launchd Label key",
    )
    hourly_launchd.add_argument(
        "--run-at-load",
        type=_parse_bool_arg,
        default=True,
        help="Set RunAtLoad true/false (default true)",
    )
    hourly_launchd.add_argument(
        "--stdout-log",
        type=Path,
        default=DEFAULT_HOURLY_STDOUT_LOG,
        help="StandardOutPath for launchd",
    )
    hourly_launchd.add_argument(
        "--stderr-log",
        type=Path,
        default=DEFAULT_HOURLY_STDERR_LOG,
        help="StandardErrorPath for launchd",
    )
    hourly_launchd.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    hourly_launchd.add_argument("--project-root", type=Path, default=None)

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

    configure_phase_backfill = subparsers.add_parser(
        "configure-phase-backfill",
        help="Enable/disable targeted phase backfill sprint (BTC+ETH, bounded caps)",
    )
    configure_phase_backfill.add_argument(
        "--phase-name",
        default=PHASE_BACKFILL_DEFAULT_NAME,
        help="Label for this phase backfill sprint",
    )
    configure_phase_backfill.add_argument(
        "--coins",
        nargs="+",
        default=None,
        help="Focused coins (default BTC ETH)",
    )
    configure_phase_backfill.add_argument(
        "--target-days",
        type=int,
        default=PHASE_BACKFILL_DEFAULT_TARGET_DAYS,
        help="Archive days to backfill in this phase",
    )
    configure_phase_backfill.add_argument(
        "--daily-file-limit",
        type=int,
        default=PHASE_BACKFILL_DEFAULT_DAILY_FILE_LIMIT,
    )
    configure_phase_backfill.add_argument(
        "--daily-chunk-limit",
        type=int,
        default=PHASE_BACKFILL_DEFAULT_DAILY_CHUNK_LIMIT,
    )
    configure_phase_backfill.add_argument(
        "--max-daily-file-limit",
        type=int,
        default=PHASE_BACKFILL_DEFAULT_DAILY_FILE_LIMIT,
    )
    configure_phase_backfill.add_argument(
        "--max-daily-chunk-limit",
        type=int,
        default=PHASE_BACKFILL_DEFAULT_DAILY_CHUNK_LIMIT,
    )
    configure_phase_backfill.add_argument("--enable", action="store_true")
    configure_phase_backfill.add_argument("--disable", action="store_true")
    configure_phase_backfill.add_argument(
        "--reset-usage",
        action="store_true",
        help="Reset daily usage counters (default preserves them)",
    )
    configure_phase_backfill.add_argument(
        "--require-confirmation-text",
        default=None,
        help=f"Required exact text when using --enable: {PHASE_BACKFILL_CONFIRMATION_TEXT!r}",
    )
    configure_phase_backfill.add_argument(
        "--aws-profile",
        default=DEFAULT_AWS_PROFILE,
        help=f"AWS profile for Requester Pays downloads (default: {DEFAULT_AWS_PROFILE})",
    )
    phase_requester_pays = configure_phase_backfill.add_mutually_exclusive_group()
    phase_requester_pays.add_argument(
        "--requester-pays",
        action="store_true",
        default=None,
        help="Use authenticated Requester Pays S3 access (default when enabling)",
    )
    phase_requester_pays.add_argument(
        "--no-requester-pays",
        action="store_true",
        default=None,
        help="Disable Requester Pays (may fail for Requester Pays buckets)",
    )
    configure_phase_backfill.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)

    phase_backfill_estimate = subparsers.add_parser(
        "phase-backfill-estimate",
        help="Print phase backfill ETA without changing state",
    )
    phase_backfill_estimate.add_argument(
        "--coins",
        nargs="+",
        default=None,
        help="Coins to estimate (default BTC ETH)",
    )
    phase_backfill_estimate.add_argument(
        "--target-days",
        type=int,
        default=PHASE_BACKFILL_DEFAULT_TARGET_DAYS,
    )
    phase_backfill_estimate.add_argument(
        "--daily-file-limit",
        type=int,
        default=PHASE_BACKFILL_DEFAULT_DAILY_FILE_LIMIT,
    )
    phase_backfill_estimate.add_argument(
        "--daily-chunk-limit",
        type=int,
        default=PHASE_BACKFILL_DEFAULT_DAILY_CHUNK_LIMIT,
    )
    phase_backfill_estimate.add_argument(
        "--chunk-size-hours",
        type=int,
        default=DEFAULT_CHUNK_SIZE_HOURS,
    )

    phase_backfill_audit = subparsers.add_parser(
        "phase-backfill-audit",
        help="Read-only Phase 3 progress and import audit",
    )
    phase_backfill_audit.add_argument(
        "--coins",
        nargs="+",
        default=None,
        help="Coins to audit (default phase focused_coins or BTC ETH)",
    )
    phase_backfill_audit.add_argument("--source", default="historical")
    phase_backfill_audit.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
    )
    phase_backfill_audit.add_argument(
        "--data-root",
        type=Path,
        default=orchestrator.DEFAULT_DATA_ROOT,
    )
    phase_backfill_audit.add_argument(
        "--downloads-root",
        type=Path,
        default=DEFAULT_DOWNLOADS_ROOT,
    )
    phase_backfill_audit.add_argument(
        "--candidate-reports-dir",
        type=Path,
        default=DEFAULT_CANDIDATE_VALIDATION_REPORTS_DIR,
    )
    phase_backfill_audit.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    phase_backfill_audit.add_argument("--json", type=Path, default=None, dest="json_path")
    phase_backfill_audit.add_argument(
        "--markdown",
        type=Path,
        default=None,
        dest="markdown_path",
    )
    phase_backfill_audit.add_argument(
        "--chunk-size-hours",
        type=int,
        default=DEFAULT_CHUNK_SIZE_HOURS,
    )

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
    autopilot_requester_pays = autopilot_tick.add_mutually_exclusive_group()
    autopilot_requester_pays.add_argument("--requester-pays", action="store_true", default=None)
    autopilot_requester_pays.add_argument("--no-requester-pays", action="store_true", default=None)
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

    research_session = subparsers.add_parser(
        "research-session",
        help="Run a bounded paper-only research session with periodic ticks",
    )
    research_session.add_argument(
        "--duration-hours",
        type=float,
        default=DEFAULT_RESEARCH_SESSION_DURATION_HOURS,
    )
    research_session.add_argument(
        "--tick-interval-minutes",
        type=int,
        default=DEFAULT_RESEARCH_SESSION_TICK_INTERVAL_MINUTES,
    )
    research_session.add_argument("--coins", nargs="+", default=None)
    research_session.add_argument("--source", default="historical")
    research_session.add_argument("--aws-profile", default=None)
    research_session_requester_pays = research_session.add_mutually_exclusive_group()
    research_session_requester_pays.add_argument(
        "--requester-pays",
        action="store_true",
        default=None,
    )
    research_session_requester_pays.add_argument(
        "--no-requester-pays",
        action="store_true",
        default=None,
    )
    research_session.add_argument(
        "--run-candidate-tests",
        type=_parse_bool_arg,
        default=True,
    )
    research_session.add_argument(
        "--run-relaxed-candidate-test",
        type=_parse_bool_arg,
        default=True,
    )
    research_session.add_argument(
        "--update-dashboard-artifacts",
        type=_parse_bool_arg,
        default=True,
    )
    research_session.add_argument("--max-ticks", type=int, default=None)
    research_session.add_argument("--dry-run", action="store_true")
    research_session.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    research_session.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    research_session.add_argument(
        "--sessions-dir",
        type=Path,
        default=DEFAULT_RESEARCH_SESSIONS_DIR,
    )
    research_session.add_argument("--data-root", type=Path, default=orchestrator.DEFAULT_DATA_ROOT)
    research_session.add_argument(
        "--downloads-root",
        type=Path,
        default=DEFAULT_DOWNLOADS_ROOT,
    )
    research_session.add_argument(
        "--candidate-reports-dir",
        type=Path,
        default=DEFAULT_CANDIDATE_VALIDATION_REPORTS_DIR,
    )
    research_session.add_argument(
        "--report-dir",
        type=Path,
        default=orchestrator.DEFAULT_REPORT_DIR,
    )
    research_session.add_argument(
        "--scheduler-state-dir",
        type=Path,
        default=DEFAULT_STATE_DIR,
    )
    research_session.add_argument(
        "--chunk-size-hours",
        type=int,
        default=DEFAULT_CHUNK_SIZE_HOURS,
    )

    write_research_session_launchd = subparsers.add_parser(
        "write-research-session-launchd-template",
        help="Write morning/evening launchd plist templates for research-session",
    )
    write_research_session_launchd.add_argument(
        "--label-prefix",
        default=DEFAULT_RESEARCH_SESSION_LABEL_PREFIX,
    )
    write_research_session_launchd.add_argument(
        "--morning-hour",
        type=int,
        default=DEFAULT_RESEARCH_SESSION_MORNING_HOUR,
    )
    write_research_session_launchd.add_argument(
        "--evening-hour",
        type=int,
        default=DEFAULT_RESEARCH_SESSION_EVENING_HOUR,
    )
    write_research_session_launchd.add_argument(
        "--duration-hours",
        type=float,
        default=DEFAULT_RESEARCH_SESSION_DURATION_HOURS,
    )
    write_research_session_launchd.add_argument(
        "--tick-interval-minutes",
        type=int,
        default=DEFAULT_RESEARCH_SESSION_TICK_INTERVAL_MINUTES,
    )
    write_research_session_launchd.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESEARCH_SESSION_OUTPUT_DIR,
    )
    write_research_session_launchd.add_argument("--python-path", default=None)
    write_research_session_launchd.add_argument(
        "--project-dir",
        type=Path,
        default=None,
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
    print(f"  phase_backfill_enabled: {status.get('phase_backfill_enabled', False)}")
    if status.get("phase_backfill_name"):
        print(f"  phase_backfill_name: {status['phase_backfill_name']}")
    if status.get("focused_coins"):
        print(f"  focused_coins: {', '.join(status['focused_coins'])}")
    if status.get("target_days"):
        print(f"  target_days: {status['target_days']}")
    if status.get("estimated_files_required") is not None:
        print(f"  estimated_files_required: {status['estimated_files_required']}")
    if status.get("estimated_real_days_to_complete") is not None:
        print(
            f"  estimated_real_days_to_complete: "
            f"{status['estimated_real_days_to_complete']:.2f}"
        )
    if status.get("archive_days_per_real_day") is not None:
        print(
            f"  archive_days_per_real_day: {status['archive_days_per_real_day']:.2f}"
        )
    if status.get("effective_daily_file_capacity") is not None:
        print(
            f"  effective_daily_file_capacity: {status['effective_daily_file_capacity']}"
        )
    for warning in status.get("warnings") or []:
        print(f"  warning: {warning}")
    if status.get("requester_pays_warning"):
        print(f"  requester_pays_warning: {status['requester_pays_warning']}")
    if status.get("aws_profile"):
        print(f"  aws_profile: {status['aws_profile']}")
    if status.get("requester_pays") is not None:
        print(f"  requester_pays: {status['requester_pays']}")


def _print_phase_backfill_estimate(estimate: dict[str, Any]) -> None:
    print("Phase Backfill Estimate (dry run, no state change)")
    print(f"  coins: {', '.join(estimate['coins'])}")
    print(f"  target_days: {estimate['target_days']}")
    print(f"  daily_file_limit: {estimate['daily_file_limit']}")
    print(f"  daily_chunk_limit: {estimate['daily_chunk_limit']}")
    print(f"  chunk_size_hours: {estimate['chunk_size_hours']}")
    print(f"  effective_daily_file_capacity: {estimate['effective_daily_file_capacity']}")
    print(f"  files_required: {estimate['files_required']}")
    print(f"  archive_days_per_real_day: {estimate['archive_days_per_real_day']:.2f}")
    print(
        f"  estimated_real_days_to_complete: "
        f"{estimate['estimated_real_days_to_complete']:.2f}"
    )
    for warning in estimate.get("warnings") or []:
        print(f"  warning: {warning}")


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
                requester_pays=_parse_cli_requester_pays(args),
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
                requester_pays=_parse_cli_requester_pays(args),
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
        elif args.command == "write-hourly-launchd-template":
            plist_path, setup_path = cmd_write_hourly_launchd_template(
                interval_seconds=args.interval_seconds,
                label=args.label,
                run_at_load=args.run_at_load,
                stdout_log=args.stdout_log,
                stderr_log=args.stderr_log,
                state_dir=args.state_dir,
                project_root=args.project_root,
            )
            print(f"Wrote hourly launchd template: {plist_path}")
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
        elif args.command == "configure-phase-backfill":
            result = cmd_configure_phase_backfill(
                enable=args.enable,
                disable=args.disable,
                phase_name=args.phase_name,
                coins=args.coins,
                target_days=args.target_days,
                daily_file_limit=args.daily_file_limit,
                daily_chunk_limit=args.daily_chunk_limit,
                max_daily_file_limit=args.max_daily_file_limit,
                max_daily_chunk_limit=args.max_daily_chunk_limit,
                aws_profile=args.aws_profile,
                requester_pays=_parse_configure_phase_requester_pays(args),
                reset_usage=args.reset_usage,
                require_confirmation_text=args.require_confirmation_text,
                state_path=args.state_path,
            )
            _print_autopilot_status(cmd_autopilot_status(state_path=args.state_path))
            print(
                "Phase backfill configured: "
                f"enabled={result['phase_backfill_enabled']} "
                f"name={result.get('phase_backfill_name')}"
            )
        elif args.command == "phase-backfill-estimate":
            estimate = cmd_phase_backfill_estimate(
                coins=args.coins,
                target_days=args.target_days,
                daily_file_limit=args.daily_file_limit,
                daily_chunk_limit=args.daily_chunk_limit,
                chunk_size_hours=args.chunk_size_hours,
            )
            _print_phase_backfill_estimate(estimate)
        elif args.command == "phase-backfill-audit":
            audit = cmd_phase_backfill_audit(
                coins=args.coins,
                source=args.source,
                reports_dir=args.reports_dir,
                data_root=args.data_root,
                downloads_root=args.downloads_root,
                candidate_reports_dir=args.candidate_reports_dir,
                state_path=args.state_path,
                json_path=args.json_path,
                markdown_path=args.markdown_path,
                chunk_size_hours=args.chunk_size_hours,
            )
            _print_phase_backfill_audit(audit)
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
                requester_pays=_parse_cli_requester_pays(args),
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
        elif args.command == "research-session":
            requester_pays = _parse_cli_requester_pays(args)
            if requester_pays is None:
                requester_pays = DEFAULT_REQUESTER_PAYS
            session = cmd_research_session(
                duration_hours=args.duration_hours,
                tick_interval_minutes=args.tick_interval_minutes,
                coins=args.coins,
                source=args.source,
                aws_profile=args.aws_profile,
                requester_pays=requester_pays,
                run_candidate_tests=args.run_candidate_tests,
                run_relaxed_candidate_test=args.run_relaxed_candidate_test,
                update_dashboard_artifacts=args.update_dashboard_artifacts,
                max_ticks=args.max_ticks,
                dry_run=args.dry_run,
                state_path=args.state_path,
                runs_dir=args.runs_dir,
                sessions_dir=args.sessions_dir,
                data_root=args.data_root,
                downloads_root=args.downloads_root,
                candidate_reports_dir=args.candidate_reports_dir,
                orchestrator_report_dir=args.report_dir,
                scheduler_state_dir=args.scheduler_state_dir,
                chunk_size_hours=args.chunk_size_hours,
            )
            print(
                "Research session "
                f"{'plan' if session.get('dry_run') else 'complete'}: "
                f"ticks_completed={session.get('ticks_completed')}/"
                f"{session.get('ticks_planned')}"
            )
            if session.get("stop_reason"):
                print(f"  stop_reason: {session['stop_reason']}")
            report_paths = session.get("session_report_paths") or {}
            if report_paths.get("json"):
                print(f"  session report: {report_paths['json']}")
            if session.get("next_recommended_action"):
                print(f"Next: {session['next_recommended_action']}")
        elif args.command == "write-research-session-launchd-template":
            morning_plist, evening_plist, setup_path = cmd_write_research_session_launchd_template(
                label_prefix=args.label_prefix,
                morning_hour=args.morning_hour,
                evening_hour=args.evening_hour,
                duration_hours=args.duration_hours,
                tick_interval_minutes=args.tick_interval_minutes,
                output_dir=args.output_dir,
                python_path=args.python_path,
                project_dir=args.project_dir,
            )
            print(f"Wrote morning launchd template: {morning_plist}")
            print(f"Wrote evening launchd template: {evening_plist}")
            print(f"Wrote setup instructions: {setup_path}")
            print("Not installed automatically.")
            print(
                f"Install morning: cp {morning_plist} ~/Library/LaunchAgents/ && "
                f"launchctl load ~/Library/LaunchAgents/{LAUNCHD_RESEARCH_SESSION_MORNING_PLIST}"
            )
            print(
                f"Install evening: cp {evening_plist} ~/Library/LaunchAgents/ && "
                f"launchctl load ~/Library/LaunchAgents/{LAUNCHD_RESEARCH_SESSION_EVENING_PLIST}"
            )
        else:
            parser.error(f"unknown command: {args.command}")
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
