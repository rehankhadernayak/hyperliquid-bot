"""Research-only orchestrator for bid/ask historical, live, and combined workflows.

Coordinates historical backfill, local WebSocket collection, validation, resampling,
training, and paper-only evaluation without mixing data sources or placing orders.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import bid_ask_data_pipeline
import bid_ask_decision_evaluator
import bid_ask_decision_filter
import bid_ask_decision_logger
import bid_ask_resampler
import bid_ask_train
import bid_ask_validator
from bid_ask_resampler import (
    DEFAULT_DATA_ROOT,
    normalize_source,
    resolve_bid_ask_data_dir,
)
from bid_ask_schema import BID_ASK_CSV_COLUMNS, SAFETY_STATEMENT
from bid_ask_train import MIN_ROWS_WARNING
from historical_l2_backfill_planner import (
    build_l2_s3_path,
    build_local_decompressed_path,
    build_local_download_path,
    validate_date_yyyymmdd,
)
from historical_l2_importer import (
    default_historical_output_path,
    import_decompressed_l2_file,
)
from historical_l2_python_downloader import (
    create_s3_client,
    run_sample_download,
)

logger = logging.getLogger(__name__)

DEFAULT_REPORT_DIR = Path("reports/bid_ask_research_orchestrator")
DEFAULT_MAX_FILES = 9
HARD_CAP_MAX_FILES = 9
HARD_CAP_MAX_FILES_ALLOW_MORE = 24
DEFAULT_LIVE_MAX_DURATION_SECONDS = 21600
ALL_TIME_ALIASES = frozenset({"all-time", "alltime", "all_time", "all time"})

ALLOWLISTED_COMMANDS = frozenset(
    {
        "python3",
        "historical_l2_importer.py",
        "bid_ask_extractor.py",
        "bid_ask_data_pipeline.py",
        "websocket_recorder.py",
    }
)

FORBIDDEN_SUBPROCESS_MARKERS = frozenset(
    {
        "aws_secret",
        "secret_access_key",
    }
)


@dataclass
class OrchestratorReport:
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    paper_only: bool = True
    execution_enabled: bool = False
    safety_statement: str = SAFETY_STATEMENT
    workflows_run: list[str] = field(default_factory=list)
    coins: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    intervals: list[str] = field(default_factory=list)
    row_counts: dict[str, Any] = field(default_factory=dict)
    model_paths: list[str] = field(default_factory=list)
    report_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    next_recommended_actions: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "paper_only": self.paper_only,
            "execution_enabled": self.execution_enabled,
            "safety_statement": self.safety_statement,
            "workflows_run": self.workflows_run,
            "coins": self.coins,
            "sources": self.sources,
            "intervals": self.intervals,
            "row_counts": self.row_counts,
            "model_paths": self.model_paths,
            "report_paths": self.report_paths,
            "warnings": self.warnings,
            "failures": self.failures,
            "next_recommended_actions": self.next_recommended_actions,
            "details": self.details,
        }


def _normalize_coins(coins: list[str]) -> list[str]:
    return [coin.strip().upper() for coin in coins if coin.strip()]


def _normalize_hours(hours: list[int | str]) -> list[str]:
    normalized: list[str] = []
    for hour in hours:
        value = int(hour)
        if value < 0 or value > 23:
            raise ValueError(f"hour must be 0-23, got {hour!r}")
        normalized.append(str(value))
    return normalized


def reject_all_time(*, all_time: bool = False, source: str | None = None, hours: list[str] | None = None) -> None:
    """Reject unbounded all-time download requests."""
    if all_time:
        raise ValueError(
            "all-time downloads are not supported; specify --date and --hours with --max-files"
        )
    if source is not None and source.strip().lower() in ALL_TIME_ALIASES:
        raise ValueError(f"source {source!r} is not supported; use local, historical, or combined")
    if hours is not None and len(hours) >= 24:
        raise ValueError(
            "requesting all 24 hours in one batch is not supported; split into smaller hour ranges"
        )


def resolve_orchestrator_max_files(
    max_files: int | None,
    *,
    allow_more: bool = False,
    planned_files: int | None = None,
) -> int:
    """Apply default and hard caps for historical batch downloads."""
    requested = DEFAULT_MAX_FILES if max_files is None else max_files
    if requested < 1:
        raise ValueError("max_files must be at least 1")

    cap = HARD_CAP_MAX_FILES_ALLOW_MORE if allow_more else HARD_CAP_MAX_FILES
    if requested > cap:
        message = (
            f"max_files {requested} exceeds hard cap of {HARD_CAP_MAX_FILES_ALLOW_MORE}"
            if allow_more
            else f"max_files {requested} exceeds hard cap of {HARD_CAP_MAX_FILES}; "
            "pass --allow-more to raise the cap (max 24)"
        )
        raise ValueError(message)

    if planned_files is not None and planned_files > cap:
        raise ValueError(
            f"planned file count {planned_files} exceeds hard cap of {cap}; "
            "reduce coins/hours or pass --allow-more"
        )

    return requested


def validate_live_duration(
    duration_seconds: float | None,
    *,
    allow_long: bool = False,
) -> float:
    """Require a finite live session duration within safety limits."""
    if duration_seconds is None:
        raise ValueError("--duration-seconds is required for live-session")
    if duration_seconds <= 0:
        raise ValueError("--duration-seconds must be positive")
    if duration_seconds > DEFAULT_LIVE_MAX_DURATION_SECONDS and not allow_long:
        raise ValueError(
            f"duration {duration_seconds}s exceeds {DEFAULT_LIVE_MAX_DURATION_SECONDS}s; "
            "pass --allow-long to permit longer sessions (still finite)"
        )
    return float(duration_seconds)


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run an allowlisted command without shell=True."""
    if not command:
        raise ValueError("command must not be empty")

    executable = Path(command[0]).name
    if executable not in ALLOWLISTED_COMMANDS:
        raise ValueError(f"command not allowlisted: {executable}")

    joined = " ".join(command)
    lowered = joined.lower()
    for marker in FORBIDDEN_SUBPROCESS_MARKERS:
        if marker.lower() in lowered:
            raise ValueError("command appears to contain sensitive material")
    if "private" in lowered and "key" in lowered:
        raise ValueError("command appears to contain sensitive material")
    if lowered.startswith("akia") or " akia" in lowered:
        raise ValueError("command appears to contain sensitive material")

    logger.info("run_command: %s", " ".join(command))
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=check,
        shell=False,
    )


def build_hourly_backfill_plan(
    date: str,
    hours: list[int | str],
    coins: list[str],
) -> dict[str, Any]:
    """Build a bounded command plan for specific date/hours/coins."""
    normalized_date = validate_date_yyyymmdd(date)
    normalized_hours = _normalize_hours(hours)
    normalized_coins = _normalize_coins(coins)
    if not normalized_hours:
        raise ValueError("at least one hour is required")
    if not normalized_coins:
        raise ValueError("at least one coin is required")

    reject_all_time(hours=normalized_hours)

    files: list[dict[str, str]] = []
    for hour in sorted(normalized_hours, key=int):
        for coin in normalized_coins:
            files.append(
                {
                    "date": normalized_date,
                    "hour": hour,
                    "coin": coin,
                    "s3_path": build_l2_s3_path(normalized_date, hour, coin),
                    "local_lz4_path": build_local_download_path(normalized_date, hour, coin),
                    "local_jsonl_path": build_local_decompressed_path(normalized_date, hour, coin),
                }
            )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "paper_only": True,
        "execution_enabled": False,
        "date": normalized_date,
        "hours": normalized_hours,
        "coins": normalized_coins,
        "files": files,
        "total_files": len(files),
    }


def _timestamp_key(interval: str) -> str:
    return "timestamp_ms" if interval == "1s" else "start_timestamp_ms"


def _csv_filename(coin: str, interval: str) -> str:
    return f"{coin}_1s.csv" if interval == "1s" else f"{coin}_{interval}.csv"


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_csv(df: pd.DataFrame, path: Path, *, columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = df.reindex(columns=list(columns))
    output.to_csv(path, index=False)


def write_orchestrator_report(
    report: OrchestratorReport,
    *,
    output_dir: Path = DEFAULT_REPORT_DIR,
    label: str = "orchestrator",
) -> tuple[Path, Path]:
    """Write JSON and Markdown orchestrator reports."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.replace(":", "").replace("-", "").replace(".", "")
    json_path = root / f"bid_ask_research_{label}_{stamp}.json"
    md_path = root / f"bid_ask_research_{label}_{stamp}.md"

    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    report.report_paths.extend([str(json_path), str(md_path)])

    lines = [
        "# Bid/Ask Research Orchestrator Report",
        "",
        f"- generated_at: {report.generated_at}",
        f"- paper_only: {report.paper_only}",
        f"- execution_enabled: {report.execution_enabled}",
        f"- workflows_run: {', '.join(report.workflows_run) or 'none'}",
        f"- coins: {', '.join(report.coins) or 'none'}",
        f"- sources: {', '.join(report.sources) or 'none'}",
        f"- intervals: {', '.join(report.intervals) or 'none'}",
        "",
        "## Row counts",
        "",
    ]
    for key, value in report.row_counts.items():
        lines.append(f"- {key}: {value}")

    if report.model_paths:
        lines.extend(["", "## Model paths", ""])
        lines.extend(f"- {path}" for path in report.model_paths)

    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)

    if report.failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in report.failures)

    if report.next_recommended_actions:
        lines.extend(["", "## Next recommended actions", ""])
        lines.extend(f"- {action}" for action in report.next_recommended_actions)

    lines.extend(["", report.safety_statement, ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def _import_decompressed_files(
    plan: dict[str, Any],
    *,
    data_root: Path,
    max_files: int,
) -> list[dict[str, Any]]:
    """Import decompressed JSONL files into historical 1s CSVs."""
    import_results: list[dict[str, Any]] = []
    for entry in plan["files"][:max_files]:
        jsonl_path = Path(entry["local_jsonl_path"])
        coin = entry["coin"]
        result: dict[str, Any] = {
            "coin": coin,
            "date": entry["date"],
            "hour": entry["hour"],
            "jsonl_path": str(jsonl_path),
            "import_status": "skipped_missing",
        }
        if not jsonl_path.is_file():
            import_results.append(result)
            continue

        output_path = default_historical_output_path(coin, output_root=data_root)
        try:
            _, stats = import_decompressed_l2_file(
                jsonl_path,
                coin,
                output_path,
                output_root=data_root,
                overwrite=False,
            )
            result["import_status"] = "imported"
            result["rows_written"] = stats.rows_written
            result["duplicate_rows_removed"] = stats.duplicate_rows_removed
            result["output_path"] = str(output_path)
        except Exception as exc:  # noqa: BLE001 - per-file boundary
            result["import_status"] = "failed"
            result["error"] = str(exc)
        import_results.append(result)
    return import_results


def cmd_historical_batch(
    *,
    date: str,
    hours: list[int | str],
    coins: list[str],
    max_files: int | None = None,
    allow_more: bool = False,
    all_time: bool = False,
    aws_profile: str | None = None,
    requester_pays: bool = False,
    overwrite: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
    report_dir: Path = DEFAULT_REPORT_DIR,
    s3_client: Any | None = None,
) -> OrchestratorReport:
    """Download, import, validate, and resample a bounded historical batch."""
    reject_all_time(all_time=all_time)
    normalized_coins = _normalize_coins(coins)
    plan = build_hourly_backfill_plan(date, hours, normalized_coins)
    resolved_max = resolve_orchestrator_max_files(
        max_files,
        allow_more=allow_more,
        planned_files=plan["total_files"],
    )

    report = OrchestratorReport(
        workflows_run=["historical-batch"],
        coins=normalized_coins,
        sources=["historical"],
        intervals=["1s", "15s", "1m"],
    )
    report.details["command_plan"] = {
        "date": plan["date"],
        "hours": plan["hours"],
        "total_files": plan["total_files"],
        "max_files": resolved_max,
    }

    plan["_source_plan_path"] = f"orchestrator:{plan['date']}:{'-'.join(plan['hours'])}"

    if s3_client is None:
        s3_client = create_s3_client(requester_pays=requester_pays, aws_profile=aws_profile)

    download_report = run_sample_download(
        plan,
        max_files=resolved_max,
        overwrite=overwrite,
        client=s3_client,
        requester_pays=requester_pays,
        aws_profile=aws_profile,
    )

    report.details["download"] = download_report
    import_results = _import_decompressed_files(plan, data_root=data_root, max_files=resolved_max)
    report.details["import"] = import_results

    validation_results = cmd_resample_validate(
        coins=normalized_coins,
        source="historical",
        data_root=data_root,
        report_dir=report_dir,
        write_top_level_report=False,
    )
    report.row_counts.update(validation_results.row_counts)
    report.warnings.extend(validation_results.warnings)
    report.failures.extend(validation_results.failures)
    report.details["validation"] = validation_results.details

    report.next_recommended_actions.append(
        "python3 bid_ask_research_orchestrator.py train-evaluate "
        f"--source historical --coins {' '.join(normalized_coins)} --interval 15s --horizon 5"
    )
    write_orchestrator_report(report, output_dir=report_dir, label="historical_batch")
    return report


def cmd_live_session(
    *,
    coins: list[str],
    duration_seconds: float | None,
    allow_long: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
    market_stream_dir: Path = bid_ask_data_pipeline.DEFAULT_MARKET_STREAM_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    run_recorder_fn: Any | None = None,
) -> OrchestratorReport:
    """Run a finite local WebSocket recorder session (research-only)."""
    duration = validate_live_duration(duration_seconds, allow_long=allow_long)
    normalized_coins = _normalize_coins(coins)

    report = OrchestratorReport(
        workflows_run=["live-session"],
        coins=normalized_coins,
        sources=["local"],
        intervals=["1s"],
        details={"duration_seconds": duration},
    )

    if run_recorder_fn is None:
        from market_data.recorder import run_recorder
        from market_data.recorder_config import build_config_from_cli

        def _default_recorder(**kwargs: Any) -> dict[str, Any]:
            config = build_config_from_cli(
                coins=tuple(normalized_coins),
                network="mainnet",
                output_dir=market_stream_dir,
                duration_seconds=kwargs["duration_seconds"],
                log_level="INFO",
                raw_only=True,
                disable_l2=False,
                candle_interval="1m",
                long_run=True,
            )
            return run_recorder(config)

        run_recorder_fn = _default_recorder

    try:
        recorder_summary = run_recorder_fn(duration_seconds=duration, coins=normalized_coins)
        report.details["recorder_summary"] = recorder_summary
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        report.failures.append(f"recorder failed: {exc}")
        write_orchestrator_report(report, output_dir=report_dir, label="live_session")
        return report

    extract_commands = [
        (
            f"python3 bid_ask_data_pipeline.py extract --coins {coin} "
            f"--source local --interval 1s"
        )
        for coin in normalized_coins
    ]
    report.next_recommended_actions.extend(extract_commands)
    report.next_recommended_actions.append(
        "python3 bid_ask_research_orchestrator.py resample-validate "
        f"--source local --coins {' '.join(normalized_coins)}"
    )
    write_orchestrator_report(report, output_dir=report_dir, label="live_session")
    return report


def cmd_resample_validate(
    *,
    coins: list[str],
    source: str = "historical",
    data_root: Path = DEFAULT_DATA_ROOT,
    report_dir: Path = DEFAULT_REPORT_DIR,
    write_top_level_report: bool = True,
) -> OrchestratorReport:
    """Validate 1s data, resample to 15s/1m, and validate resampled outputs."""
    normalized_source = normalize_source(source)
    normalized_coins = _normalize_coins(coins)
    reject_all_time(source=source)

    report = OrchestratorReport(
        workflows_run=["resample-validate"],
        coins=normalized_coins,
        sources=[normalized_source],
        intervals=["1s", "15s", "1m"],
    )

    for coin in normalized_coins:
        try:
            validation = bid_ask_validator.validate_coin_interval(
                coin,
                "1s",
                data_root=data_root,
                source=normalized_source,
            )
            report.row_counts[f"{coin}:1s"] = validation.row_count
            if not validation.passed:
                report.warnings.append(
                    f"{coin} 1s validation failed: {validation.warnings}"
                )
        except FileNotFoundError:
            report.warnings.append(f"{coin}: missing 1s CSV for source={normalized_source}")
            continue
        except Exception as exc:  # noqa: BLE001
            report.failures.append(f"{coin} 1s validation error: {exc}")
            continue

        try:
            resample_results = bid_ask_resampler.resample_coin(
                coin,
                intervals=("15s", "1m"),
                data_root=data_root,
                source=normalized_source,
            )
            report.details.setdefault("resample", {})[coin] = {
                interval: {
                    "output_path": str(paths["output_path"]),
                    "report_path": str(paths["report_path"]),
                }
                for interval, paths in resample_results.items()
            }
            for interval in ("15s", "1m"):
                validation = bid_ask_validator.validate_coin_interval(
                    coin,
                    interval,
                    data_root=data_root,
                    source=normalized_source,
                )
                report.row_counts[f"{coin}:{interval}"] = validation.row_count
                if not validation.passed:
                    report.warnings.append(
                        f"{coin} {interval} validation failed: {validation.warnings}"
                    )
        except FileNotFoundError as exc:
            report.warnings.append(f"{coin}: resample skipped ({exc})")
        except Exception as exc:  # noqa: BLE001
            report.failures.append(f"{coin}: resample error: {exc}")

    if write_top_level_report:
        write_orchestrator_report(report, output_dir=report_dir, label="resample_validate")
    return report


def cmd_train_evaluate(
    *,
    coins: list[str],
    source: str = "historical",
    interval: str = "15s",
    horizon: int = 5,
    data_root: Path = DEFAULT_DATA_ROOT,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> OrchestratorReport:
    """Train models and run paper decision logging, evaluation, and filtering."""
    normalized_source = normalize_source(source)
    normalized_coins = _normalize_coins(coins)
    reject_all_time(source=source)

    report = OrchestratorReport(
        workflows_run=["train-evaluate"],
        coins=normalized_coins,
        sources=[normalized_source],
        intervals=[interval],
        details={"horizon": horizon},
    )

    trained_coins: list[str] = []
    for coin in normalized_coins:
        try:
            raw_df = bid_ask_train.load_training_data(
                coin,
                interval,
                data_root,
                source=normalized_source,
            )
            if len(raw_df) < MIN_ROWS_WARNING:
                report.warnings.append(
                    f"{coin}: only {len(raw_df)} rows (< {MIN_ROWS_WARNING}); skipping training"
                )
                continue

            training_result = bid_ask_train.run_training_pipeline(
                coin=coin,
                interval=interval,
                horizon=horizon,
                data_root=data_root,
                source=normalized_source,
            )
            training_report = training_result["report"]
            report.row_counts[f"{coin}:train_rows"] = training_report.get("train_rows", 0)
            report.row_counts[f"{coin}:test_rows"] = training_report.get("test_rows", 0)
            if training_report.get("report_path"):
                report.report_paths.append(str(training_report["report_path"]))
            artifact_path = training_report.get("model_artifact_path")
            if artifact_path:
                report.model_paths.append(str(artifact_path))
            trained_coins.append(coin)
        except FileNotFoundError:
            report.warnings.append(f"{coin}: training CSV not found; skipped")
        except Exception as exc:  # noqa: BLE001
            report.failures.append(f"{coin}: training error: {exc}")

    if trained_coins:
        try:
            decision_summary = bid_ask_decision_logger.run_decision_logging(
                trained_coins,
                interval=interval,
                horizon=horizon,
                data_root=data_root,
            )
            report.details["decision_logging"] = decision_summary
            if decision_summary.get("output_path"):
                log_path = Path(decision_summary["output_path"])
                if log_path.is_file():
                    evaluation = bid_ask_decision_evaluator.run_evaluation(log_path)
                    report.details["decision_evaluation"] = {
                        "json_report_path": evaluation.get("json_report_path"),
                        "model_accuracy": evaluation.get("model_accuracy"),
                    }
                    filter_report = bid_ask_decision_filter.run_filter(log_path)
                    report.details["decision_filter"] = {
                        "json_report_path": filter_report.get("json_report_path"),
                        "accepted_count": filter_report.get("accepted_count"),
                        "blocked_count": filter_report.get("blocked_count"),
                    }
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"decision pipeline skipped: {exc}")

    write_orchestrator_report(report, output_dir=report_dir, label="train_evaluate")
    return report


def cmd_combined_dataset(
    *,
    coins: list[str],
    intervals: list[str] | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> OrchestratorReport:
    """Combine local and historical CSVs into combined_* folders without overwriting sources."""
    normalized_coins = _normalize_coins(coins)
    selected_intervals = intervals or ["1s", "15s", "1m"]

    report = OrchestratorReport(
        workflows_run=["combined-dataset"],
        coins=normalized_coins,
        sources=["local", "historical", "combined"],
        intervals=selected_intervals,
    )

    metadata_rows: list[dict[str, Any]] = []

    for coin in normalized_coins:
        for interval in selected_intervals:
            local_path = resolve_bid_ask_data_dir(interval, "local", data_root) / _csv_filename(
                coin, interval
            )
            historical_path = resolve_bid_ask_data_dir(interval, "historical", data_root) / _csv_filename(
                coin, interval
            )
            combined_path = resolve_bid_ask_data_dir(interval, "combined", data_root) / _csv_filename(
                coin, interval
            )

            local_df = _read_csv_if_exists(local_path)
            historical_df = _read_csv_if_exists(historical_path)
            local_count = len(local_df)
            historical_count = len(historical_df)

            if local_count == 0 and historical_count == 0:
                report.warnings.append(f"{coin} {interval}: no local or historical data to combine")
                continue

            key = _timestamp_key(interval)
            combined_frames = []
            if local_count:
                local_copy = local_df.copy()
                local_copy["_merge_origin"] = "local"
                combined_frames.append(local_copy)
            if historical_count:
                historical_copy = historical_df.copy()
                historical_copy["_merge_origin"] = "historical"
                combined_frames.append(historical_copy)

            merged = pd.concat(combined_frames, ignore_index=True)
            duplicates_removed = 0
            if not merged.empty and key in merged.columns:
                before = len(merged)
                merged = merged.sort_values(key).drop_duplicates(subset=[key], keep="first")
                duplicates_removed = before - len(merged)

            origin_counts = {
                "local": int((merged.get("_merge_origin") == "local").sum()) if "_merge_origin" in merged else local_count,
                "historical": int((merged.get("_merge_origin") == "historical").sum())
                if "_merge_origin" in merged
                else historical_count,
            }
            merged = merged.drop(columns=["_merge_origin"], errors="ignore")

            if interval == "1s":
                columns = tuple(BID_ASK_CSV_COLUMNS)
            else:
                columns = tuple(bid_ask_resampler.RESAMPLED_COLUMNS)

            _write_csv(merged, combined_path, columns=columns)
            report.row_counts[f"{coin}:{interval}:combined"] = len(merged)
            metadata_rows.append(
                {
                    "coin": coin,
                    "interval": interval,
                    "local_rows": local_count,
                    "historical_rows": historical_count,
                    "combined_rows": len(merged),
                    "duplicates_removed": duplicates_removed,
                    "local_path": str(local_path),
                    "historical_path": str(historical_path),
                    "combined_path": str(combined_path),
                    "origin_rows_in_combined": origin_counts,
                }
            )

            try:
                validation = bid_ask_validator.validate_coin_interval(
                    coin,
                    interval,
                    data_root=data_root,
                    source="combined",
                )
                if not validation.passed:
                    report.warnings.append(f"{coin} {interval} combined validation issues")
            except Exception as exc:  # noqa: BLE001
                report.warnings.append(f"{coin} {interval} combined validation skipped: {exc}")

    report.details["combined_metadata"] = metadata_rows
    metadata_path = report_dir / f"combined_metadata_{report.generated_at.replace(':', '').replace('-', '')}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata_rows, indent=2), encoding="utf-8")
    report.report_paths.append(str(metadata_path))

    report.next_recommended_actions.append(
        "python3 bid_ask_research_orchestrator.py train-evaluate "
        f"--source combined --coins {' '.join(normalized_coins)} --interval 15s --horizon 5"
    )
    write_orchestrator_report(report, output_dir=report_dir, label="combined_dataset")
    return report


def cmd_run_research_cycle(
    *,
    historical_date: str,
    historical_hours: list[int | str],
    coins: list[str],
    max_files: int | None = None,
    allow_more: bool = False,
    aws_profile: str | None = None,
    requester_pays: bool = False,
    train_interval: str = "15s",
    horizon: int = 5,
    include_combined: bool = False,
    include_live: bool = False,
    live_duration_seconds: float | None = None,
    allow_long: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
    report_dir: Path = DEFAULT_REPORT_DIR,
    s3_client: Any | None = None,
    run_recorder_fn: Any | None = None,
) -> OrchestratorReport:
    """Run historical batch, validation, training, and optional combined/live steps."""
    normalized_coins = _normalize_coins(coins)
    report = OrchestratorReport(
        workflows_run=["run-research-cycle"],
        coins=normalized_coins,
        sources=["historical"],
        intervals=[train_interval],
    )

    batch_report = cmd_historical_batch(
        date=historical_date,
        hours=historical_hours,
        coins=normalized_coins,
        max_files=max_files,
        allow_more=allow_more,
        aws_profile=aws_profile,
        requester_pays=requester_pays,
        data_root=data_root,
        report_dir=report_dir,
        s3_client=s3_client,
    )
    report.details["historical_batch"] = batch_report.to_dict()
    report.warnings.extend(batch_report.warnings)
    report.failures.extend(batch_report.failures)
    report.row_counts.update(batch_report.row_counts)

    train_report = cmd_train_evaluate(
        coins=normalized_coins,
        source="historical",
        interval=train_interval,
        horizon=horizon,
        data_root=data_root,
        report_dir=report_dir,
    )
    report.details["train_evaluate_historical"] = train_report.to_dict()
    report.warnings.extend(train_report.warnings)
    report.failures.extend(train_report.failures)
    report.model_paths.extend(train_report.model_paths)

    if include_combined:
        combined_report = cmd_combined_dataset(
            coins=normalized_coins,
            intervals=["1s", "15s", "1m"],
            data_root=data_root,
            report_dir=report_dir,
        )
        report.details["combined_dataset"] = combined_report.to_dict()
        report.warnings.extend(combined_report.warnings)
        report.row_counts.update(combined_report.row_counts)

        combined_train = cmd_train_evaluate(
            coins=normalized_coins,
            source="combined",
            interval=train_interval,
            horizon=horizon,
            data_root=data_root,
            report_dir=report_dir,
        )
        report.details["train_evaluate_combined"] = combined_train.to_dict()
        report.warnings.extend(combined_train.warnings)
        report.model_paths.extend(combined_train.model_paths)

    if include_live:
        if live_duration_seconds is None:
            report.failures.append("include-live requires --live-duration-seconds")
        else:
            live_report = cmd_live_session(
                coins=normalized_coins,
                duration_seconds=live_duration_seconds,
                allow_long=allow_long,
                data_root=data_root,
                report_dir=report_dir,
                run_recorder_fn=run_recorder_fn,
            )
            report.details["live_session"] = live_report.to_dict()
            report.warnings.extend(live_report.warnings)
            report.failures.extend(live_report.failures)
            report.next_recommended_actions.extend(live_report.next_recommended_actions)

    write_orchestrator_report(report, output_dir=report_dir, label="research_cycle")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bid/ask research orchestrator (read-only, paper-only, no execution)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    historical = subparsers.add_parser(
        "historical-batch",
        help="Download, import, validate, and resample a bounded historical batch",
    )
    historical.add_argument("--date", required=True, help="Archive date YYYYMMDD")
    historical.add_argument("--hours", nargs="+", type=int, required=True, help="UTC hours 0-23")
    historical.add_argument("--coins", nargs="+", required=True, help="Coins e.g. BTC ETH SOL")
    historical.add_argument("--max-files", type=int, default=None, help="Max archive files (default 9)")
    historical.add_argument("--allow-more", action="store_true", help="Raise max-files cap to 24")
    historical.add_argument("--all-time", action="store_true", help="Rejected: unbounded downloads")
    historical.add_argument("--aws-profile", default=None, help="Optional AWS profile name")
    historical.add_argument("--requester-pays", action="store_true", help="Use requester-pays S3 access")
    historical.add_argument("--overwrite", action="store_true", help="Overwrite existing downloads")
    historical.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    historical.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)

    live = subparsers.add_parser("live-session", help="Finite local WebSocket recorder session")
    live.add_argument("--coins", nargs="+", required=True)
    live.add_argument("--duration-seconds", type=float, required=True)
    live.add_argument("--allow-long", action="store_true")
    live.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    live.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)

    resample = subparsers.add_parser("resample-validate", help="Validate and resample bid/ask CSVs")
    resample.add_argument(
        "--source",
        default="historical",
        choices=["local", "historical", "combined"],
    )
    resample.add_argument("--coins", nargs="+", required=True)
    resample.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    resample.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)

    train = subparsers.add_parser("train-evaluate", help="Train and evaluate paper decisions")
    train.add_argument("--source", default="historical", choices=["local", "historical", "combined"])
    train.add_argument("--coins", nargs="+", required=True)
    train.add_argument("--interval", default="15s", choices=["15s", "1m"])
    train.add_argument("--horizon", type=int, default=5)
    train.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    train.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)

    combined = subparsers.add_parser("combined-dataset", help="Merge local and historical CSVs")
    combined.add_argument("--coins", nargs="+", required=True)
    combined.add_argument("--intervals", nargs="+", default=["1s", "15s", "1m"])
    combined.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    combined.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)

    cycle = subparsers.add_parser("run-research-cycle", help="End-to-end research workflow")
    cycle.add_argument("--historical-date", required=True)
    cycle.add_argument("--historical-hours", nargs="+", type=int, required=True)
    cycle.add_argument("--coins", nargs="+", required=True)
    cycle.add_argument("--max-files", type=int, default=None)
    cycle.add_argument("--allow-more", action="store_true")
    cycle.add_argument("--aws-profile", default=None)
    cycle.add_argument("--requester-pays", action="store_true")
    cycle.add_argument("--train-interval", default="15s", choices=["15s", "1m"])
    cycle.add_argument("--horizon", type=int, default=5)
    cycle.add_argument("--include-combined", action="store_true")
    cycle.add_argument("--include-live", action="store_true")
    cycle.add_argument("--live-duration-seconds", type=float, default=None)
    cycle.add_argument("--allow-long", action="store_true")
    cycle.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    cycle.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "historical-batch":
            cmd_historical_batch(
                date=args.date,
                hours=args.hours,
                coins=args.coins,
                max_files=args.max_files,
                allow_more=args.allow_more,
                all_time=args.all_time,
                aws_profile=args.aws_profile,
                requester_pays=args.requester_pays,
                overwrite=args.overwrite,
                data_root=args.data_root,
                report_dir=args.report_dir,
            )
        elif args.command == "live-session":
            cmd_live_session(
                coins=args.coins,
                duration_seconds=args.duration_seconds,
                allow_long=args.allow_long,
                data_root=args.data_root,
                report_dir=args.report_dir,
            )
        elif args.command == "resample-validate":
            cmd_resample_validate(
                coins=args.coins,
                source=args.source,
                data_root=args.data_root,
                report_dir=args.report_dir,
            )
        elif args.command == "train-evaluate":
            cmd_train_evaluate(
                coins=args.coins,
                source=args.source,
                interval=args.interval,
                horizon=args.horizon,
                data_root=args.data_root,
                report_dir=args.report_dir,
            )
        elif args.command == "combined-dataset":
            cmd_combined_dataset(
                coins=args.coins,
                intervals=args.intervals,
                data_root=args.data_root,
                report_dir=args.report_dir,
            )
        elif args.command == "run-research-cycle":
            cmd_run_research_cycle(
                historical_date=args.historical_date,
                historical_hours=args.historical_hours,
                coins=args.coins,
                max_files=args.max_files,
                allow_more=args.allow_more,
                aws_profile=args.aws_profile,
                requester_pays=args.requester_pays,
                train_interval=args.train_interval,
                horizon=args.horizon,
                include_combined=args.include_combined,
                include_live=args.include_live,
                live_duration_seconds=args.live_duration_seconds,
                allow_long=args.allow_long,
                data_root=args.data_root,
                report_dir=args.report_dir,
            )
        else:
            parser.error(f"unknown command: {args.command}")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(SAFETY_STATEMENT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
