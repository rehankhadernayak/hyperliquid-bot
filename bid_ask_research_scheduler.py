"""Daily research scheduler for bounded historical backfill, live data, training, and reports.

Research-only, paper-only layer on top of bid_ask_research_orchestrator.py.
No wallets, signing, orders, or exchange execution endpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import bid_ask_daily_summary
import bid_ask_research_orchestrator as orchestrator
from bid_ask_schema import SAFETY_STATEMENT
from historical_l2_backfill_planner import validate_date_yyyymmdd

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


@dataclass
class SchedulerState:
    historical_start_date: str | None = None
    historical_end_date_target: str | None = None
    next_historical_date: str | None = None
    completed_dates: list[str] = field(default_factory=list)
    failed_dates: list[str] = field(default_factory=list)
    completed_batches: list[dict[str, Any]] = field(default_factory=list)
    last_live_session_at: str | None = None
    last_training_at: str | None = None
    total_historical_files_downloaded: int = 0
    total_historical_days_completed: int = 0
    last_report_path: str | None = None
    last_run_at: str | None = None
    warnings: list[str] = field(default_factory=list)
    coins: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "historical_start_date": self.historical_start_date,
            "historical_end_date_target": self.historical_end_date_target,
            "next_historical_date": self.next_historical_date,
            "completed_dates": self.completed_dates,
            "failed_dates": self.failed_dates,
            "completed_batches": self.completed_batches,
            "last_live_session_at": self.last_live_session_at,
            "last_training_at": self.last_training_at,
            "total_historical_files_downloaded": self.total_historical_files_downloaded,
            "total_historical_days_completed": self.total_historical_days_completed,
            "last_report_path": self.last_report_path,
            "last_run_at": self.last_run_at,
            "warnings": self.warnings,
            "coins": self.coins,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SchedulerState:
        return cls(
            historical_start_date=payload.get("historical_start_date"),
            historical_end_date_target=payload.get("historical_end_date_target"),
            next_historical_date=payload.get("next_historical_date"),
            completed_dates=list(payload.get("completed_dates") or []),
            failed_dates=list(payload.get("failed_dates") or []),
            completed_batches=list(payload.get("completed_batches") or []),
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
            warnings=list(payload.get("warnings") or []),
            coins=list(payload.get("coins") or []),
        )


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


def _all_hours_completed_for_date(state: SchedulerState, date: str) -> bool:
    return _hours_completed_for_date(state, date) >= {str(h) for h in ALL_HOURS}


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
    coins: list[str],
    reset: bool = False,
    confirm_large_range: bool = False,
    state_path: Path = DEFAULT_STATE_PATH,
) -> SchedulerState:
    """Initialize scheduler state without downloading anything."""
    start = validate_date_yyyymmdd(historical_start_date)
    end = validate_date_yyyymmdd(historical_end_date) if historical_end_date else None
    normalized_coins = orchestrator._normalize_coins(coins)

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

    state.completed_batches.append(
        {"date": target_date, "hours": normalized_hours, "completed_at": _utc_now_iso()}
    )
    downloaded = _count_downloaded_files(batch_report)
    state.total_historical_files_downloaded += downloaded
    report.row_counts.update(batch_report.row_counts)
    report.warnings.extend(batch_report.warnings)
    report.details["orchestrator_batch"] = batch_report.to_dict()

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
        if _chunk_completed_in_state(state, target_date, chunk_hours):
            series_report.chunks_skipped += 1
            series_report.warnings.append(
                f"Chunk hours {', '.join(chunk_hours)} already completed for "
                f"{target_date}; skipping"
            )
            continue

        chunk_report = cmd_daily_cycle(
            coins=normalized_coins,
            historical_hours=chunk_int_hours,
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


def cmd_write_launchd_template(
    *,
    run_time: str = "02:00",
    coins: list[str],
    aws_profile: str | None = None,
    requester_pays: bool = False,
    state_dir: Path = DEFAULT_STATE_DIR,
    project_root: Path | None = None,
) -> tuple[Path, Path]:
    """Write macOS launchd plist template and setup instructions (does not install)."""
    normalized_coins = orchestrator._normalize_coins(coins)
    root = project_root or Path.cwd()
    python_path = sys.executable
    scheduler_script = root / "bid_ask_research_scheduler.py"

    coin_args = " ".join(normalized_coins)
    cmd_parts = [
        python_path,
        str(scheduler_script),
        "daily-cycle",
        "--coins",
        *normalized_coins,
    ]
    if aws_profile:
        cmd_parts.extend(["--aws-profile", aws_profile])
    if requester_pays:
        cmd_parts.append("--requester-pays")

    hour, minute = run_time.split(":")
    plist_path = state_dir / LAUNCHD_PLIST_NAME
    state_dir.mkdir(parents=True, exist_ok=True)

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hyperliquid.research.daily</string>
    <key>ProgramArguments</key>
    <array>
{''.join(f'        <string>{part}</string>\n' for part in cmd_parts)}    </array>
    <key>WorkingDirectory</key>
    <string>{root}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{int(hour)}</integer>
        <key>Minute</key>
        <integer>{int(minute)}</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{state_dir / 'launchd_stdout.log'}</string>
    <key>StandardErrorPath</key>
    <string>{state_dir / 'launchd_stderr.log'}</string>
</dict>
</plist>
"""
    plist_path.write_text(plist_content, encoding="utf-8")

    setup_md = state_dir / LAUNCHD_SETUP_MD
    setup_lines = [
        "# macOS launchd Setup for Daily Research Scheduler",
        "",
        "This template runs `daily-cycle` once per day. It is **not installed automatically**.",
        "",
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
        f"  --coins {coin_args}",
        "```",
        "",
        "2. Verify a manual daily cycle succeeds:",
        "",
        "```bash",
        "python3 bid_ask_research_scheduler.py daily-cycle \\",
        f"  --coins {coin_args}",
        "```",
        "",
        "## Install (manual)",
        "",
        "```bash",
        f"cp {plist_path} ~/Library/LaunchAgents/",
        "launchctl load ~/Library/LaunchAgents/com.hyperliquid.research.daily.plist",
        "```",
        "",
        "## Uninstall",
        "",
        "```bash",
        "launchctl unload ~/Library/LaunchAgents/com.hyperliquid.research.daily.plist",
        "rm ~/Library/LaunchAgents/com.hyperliquid.research.daily.plist",
        "```",
        "",
        "## Safety",
        "",
        "- Research-only, paper-only; no orders or execution endpoints.",
        "- Historical backfill progresses in bounded daily batches.",
        "- Archive availability is not guaranteed.",
        "",
        f"Scheduled time: **{run_time}** local time.",
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
    launchd.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    launchd.add_argument("--project-root", type=Path, default=None)

    return parser


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
                coins=args.coins,
                reset=args.reset,
                confirm_large_range=args.confirm_large_range,
                state_path=args.state_path,
            )
            print(f"Initialized scheduler state at {args.state_path}")
            print(f"  next_historical_date: {state.next_historical_date}")
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
                state_dir=args.state_dir,
                project_root=args.project_root,
            )
            print(f"Wrote launchd template: {plist_path}")
            print(f"Wrote setup instructions: {setup_path}")
            print("Not installed automatically.")
        else:
            parser.error(f"unknown command: {args.command}")
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
