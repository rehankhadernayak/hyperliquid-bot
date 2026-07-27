"""Daily research summary for local bid/ask CSV coverage, validation, and training."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bid_ask_schema import (
    EXECUTION_ENABLED,
    PAPER_ONLY,
    SAFETY_STATEMENT,
    configure_csv_field_size_limit,
)
from bid_ask_validator import SUPPORTED_INTERVALS, bid_ask_csv_path, reports_dir

DEFAULT_DATA_ROOT = Path("data/bid_ask_csv")
DEFAULT_TRAINING_REPORTS_DIR = Path("reports/bid_ask_training")
DEFAULT_OUTPUT_DIR = Path("reports/bid_ask_daily")

INTERVALS = ("1s", "15s", "1m")
RESEARCH_ONLY_WARNING = "Research-only. Does not prove profitability."
HIGH_MISSING_SECONDS_THRESHOLD = 3600

TIMESTAMP_COLUMNS = {
    "1s": "timestamp_ms",
    "15s": "start_timestamp_ms",
    "1m": "start_timestamp_ms",
}

NEXT_RECOMMENDED_COMMANDS = (
    "python3 bid_ask_data_pipeline.py validate --coin BTC --interval 1s",
    "python3 bid_ask_train.py --coin BTC --interval 15s",
    "python3 bid_ask_daily_summary.py --coins BTC ETH SOL",
    "python3 historical_backfill_tool_check.py",
)


def discover_latest_csvs(
    coins: list[str],
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, dict[str, Path | None]]:
    """Return standard bid/ask CSV paths for each coin and interval."""
    discovered: dict[str, dict[str, Path | None]] = {}
    for coin in coins:
        coin_paths: dict[str, Path | None] = {}
        for interval in INTERVALS:
            path = bid_ask_csv_path(coin, interval, data_root)
            coin_paths[interval] = path if path.is_file() else None
        discovered[coin] = coin_paths
    return discovered


def summarize_csv_file(path: Path) -> dict[str, Any]:
    """Summarize one bid/ask CSV without modifying it."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    file_size_bytes = path.stat().st_size
    row_count = 0
    columns: list[str] = []
    first_timestamp: int | None = None
    last_timestamp: int | None = None
    timestamp_column: str | None = None

    configure_csv_field_size_limit()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        for row in reader:
            row_count += 1
            if timestamp_column is None:
                if "timestamp_ms" in columns:
                    timestamp_column = "timestamp_ms"
                elif "start_timestamp_ms" in columns:
                    timestamp_column = "start_timestamp_ms"
            if timestamp_column and row.get(timestamp_column):
                value = int(float(row[timestamp_column]))
                if first_timestamp is None:
                    first_timestamp = value
                last_timestamp = value

    return {
        "path": str(path),
        "row_count": row_count,
        "columns": columns,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "file_size_bytes": file_size_bytes,
    }


def _validation_report_pattern(coin: str, interval: str) -> str:
    return f"{coin}_{interval}_validation_*.json"


def load_latest_validation_report(
    coin: str,
    interval: str,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any] | None:
    """Load the newest validation JSON report for a coin/interval."""
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval {interval!r}")

    report_dir = reports_dir(data_root)
    if not report_dir.is_dir():
        return None

    matches = sorted(
        report_dir.glob(_validation_report_pattern(coin, interval)),
        key=lambda path: path.name,
    )
    if not matches:
        return None

    latest = matches[-1]
    return json.loads(latest.read_text(encoding="utf-8"))


def load_latest_training_report(
    coin: str,
    training_reports_dir: Path = DEFAULT_TRAINING_REPORTS_DIR,
) -> dict[str, Any] | None:
    """Load the newest training JSON report for a coin."""
    if not training_reports_dir.is_dir():
        return None

    candidates: list[tuple[str, Path]] = []
    for path in training_reports_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("coin") != coin:
            continue
        generated_at = payload.get("generated_at") or ""
        candidates.append((generated_at, path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    latest_path = candidates[-1][1]
    return json.loads(latest_path.read_text(encoding="utf-8"))


def _csv_status(existing_intervals: list[str]) -> str:
    if len(existing_intervals) == len(INTERVALS):
        return "complete"
    if existing_intervals:
        return "partial"
    return "missing"


def _maybe_backfill_tool_warning() -> str | None:
    try:
        from historical_backfill_tool_check import check_backfill_tools
    except ImportError:
        return None

    result = check_backfill_tools()
    if result.get("ready_for_manual_download"):
        return None
    missing = result.get("missing_tools") or []
    if missing:
        return (
            "Historical backfill tools missing: "
            f"{', '.join(missing)}. Manual archive backfill may be unavailable."
        )
    return "Historical backfill tools are not fully available for manual archive download."


def _coin_summary(
    coin: str,
    *,
    data_root: Path,
    training_reports_dir: Path,
) -> dict[str, Any]:
    warnings: list[str] = [RESEARCH_ONLY_WARNING]
    csv_paths = discover_latest_csvs([coin], data_root=data_root)[coin]

    row_counts: dict[str, int | None] = {"1s": None, "15s": None, "1m": None}
    coverage_start: int | None = None
    coverage_end: int | None = None
    existing_intervals: list[str] = []

    for interval, csv_path in csv_paths.items():
        if csv_path is None:
            warnings.append(f"Missing {interval} CSV for {coin}")
            continue

        existing_intervals.append(interval)
        try:
            summary = summarize_csv_file(csv_path)
        except OSError as exc:
            warnings.append(f"Could not summarize {interval} CSV for {coin}: {exc}")
            continue

        row_counts[interval] = summary["row_count"]
        if interval == "1s":
            coverage_start = summary["first_timestamp"]
            coverage_end = summary["last_timestamp"]

    validation_by_interval: dict[str, dict[str, Any] | None] = {}
    validation_passed = True
    missing_seconds: int | None = None

    for interval in INTERVALS:
        report = load_latest_validation_report(coin, interval, data_root=data_root)
        validation_by_interval[interval] = report
        if report is None:
            if csv_paths.get(interval) is not None:
                warnings.append(f"No validation report found for {coin} {interval}")
            validation_passed = False
            continue
        if not report.get("passed", False):
            validation_passed = False
            for note in report.get("warnings") or []:
                warnings.append(f"{coin} {interval}: {note}")
        if interval == "1s":
            missing_seconds = report.get("missing_periods_count")
            if coverage_start is None:
                coverage_start = report.get("coverage_start")
            if coverage_end is None:
                coverage_end = report.get("coverage_end")

    if missing_seconds is not None and missing_seconds >= HIGH_MISSING_SECONDS_THRESHOLD:
        warnings.append(
            f"{coin}: high missing seconds detected ({missing_seconds} >= "
            f"{HIGH_MISSING_SECONDS_THRESHOLD})"
        )

    training_report = load_latest_training_report(coin, training_reports_dir)
    evaluation = (training_report or {}).get("evaluation") or {}
    training_accuracy = evaluation.get("accuracy")
    baseline_accuracy = evaluation.get("baseline_accuracy")
    train_rows = evaluation.get("train_rows")
    test_rows = evaluation.get("test_rows")

    if training_report is None:
        warnings.append(f"No training report found for {coin}")
    else:
        for note in evaluation.get("warnings") or []:
            if note not in warnings:
                warnings.append(note)

    return {
        "coin": coin,
        "csv_status": _csv_status(existing_intervals),
        "csv_paths": {interval: str(path) if path else None for interval, path in csv_paths.items()},
        "row_counts": row_counts,
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "validation_passed": validation_passed and bool(existing_intervals),
        "validation_by_interval": {
            interval: {
                "passed": (report or {}).get("passed"),
                "missing_periods_count": (report or {}).get("missing_periods_count"),
                "report_path": (report or {}).get("report_path"),
            }
            for interval, report in validation_by_interval.items()
        },
        "missing_seconds": missing_seconds,
        "training_accuracy": training_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "training_report_path": (training_report or {}).get("report_path"),
        "training_interval": (training_report or {}).get("interval"),
        "model_used": (training_report or {}).get("model_used"),
        "warnings": warnings,
    }


def build_daily_summary(
    coins: list[str],
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    training_reports_dir: Path = DEFAULT_TRAINING_REPORTS_DIR,
    include_backfill_tool_check: bool = True,
) -> dict[str, Any]:
    """Combine CSV coverage, validation, and training into one research summary."""
    coin_summaries = [
        _coin_summary(
            coin,
            data_root=data_root,
            training_reports_dir=training_reports_dir,
        )
        for coin in coins
    ]

    overall_warnings = [RESEARCH_ONLY_WARNING]
    data_quality_notes: list[str] = []
    limitations = [
        "Summary uses locally collected CSV files and existing JSON reports only.",
        "Training accuracy is research evaluation only and does not prove profitability.",
        "Missing seconds in 1s data can reduce resample and training quality.",
    ]

    ready_for_review = True
    for summary in coin_summaries:
        if summary["csv_status"] != "complete":
            ready_for_review = False
            data_quality_notes.append(f"{summary['coin']}: CSV coverage is {summary['csv_status']}")
        if not summary["validation_passed"]:
            ready_for_review = False
            data_quality_notes.append(f"{summary['coin']}: validation did not fully pass")
        for warning in summary["warnings"]:
            if warning not in overall_warnings:
                overall_warnings.append(warning)

    if include_backfill_tool_check:
        backfill_warning = _maybe_backfill_tool_warning()
        if backfill_warning:
            overall_warnings.append(backfill_warning)
            limitations.append(backfill_warning)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "locally_collected_data_only": True,
        "safety_statement": SAFETY_STATEMENT,
        "coins": coin_summaries,
        "overall_status": {
            "ready_for_review": ready_for_review,
            "data_quality_notes": data_quality_notes,
            "limitations": limitations,
            "warnings": overall_warnings,
        },
    }


def _summary_stamp(summary: dict[str, Any]) -> str:
    generated_at = summary.get("generated_at", "")
    if generated_at.endswith("Z"):
        parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        return parsed.strftime("%Y%m%dT%H%M%SZ")
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_summary_json(
    summary: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write the daily summary as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _summary_stamp(summary)
    output_path = output_dir / f"bid_ask_daily_summary_{stamp}.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["json_report_path"] = str(output_path)
    return output_path


def _format_timestamp_ms(value: int | None) -> str:
    if value is None:
        return "n/a"
    seconds = value / 1000.0
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _format_accuracy(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def write_summary_markdown(
    summary: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write the daily summary as Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _summary_stamp(summary)
    output_path = output_dir / f"bid_ask_daily_summary_{stamp}.md"

    overall = summary["overall_status"]
    lines = [
        "# Bid/Ask Daily Research Summary",
        "",
        f"Generated at: {summary['generated_at']}",
        "",
        "## Overall Status",
        "",
        f"- ready_for_review: {str(overall['ready_for_review']).lower()}",
        f"- paper_only: {str(summary['paper_only']).lower()}",
        f"- execution_enabled: {str(summary['execution_enabled']).lower()}",
        f"- locally_collected_data_only: {str(summary['locally_collected_data_only']).lower()}",
        "",
        "## Coin Table",
        "",
        "| Coin | CSV Status | 1s Rows | 15s Rows | 1m Rows | Validation Passed | Missing Seconds |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]

    for coin_summary in summary["coins"]:
        row_counts = coin_summary["row_counts"]
        lines.append(
            "| {coin} | {csv_status} | {rows_1s} | {rows_15s} | {rows_1m} | {validation} | {missing} |".format(
                coin=coin_summary["coin"],
                csv_status=coin_summary["csv_status"],
                rows_1s=row_counts.get("1s", "n/a"),
                rows_15s=row_counts.get("15s", "n/a"),
                rows_1m=row_counts.get("1m", "n/a"),
                validation=str(coin_summary["validation_passed"]).lower(),
                missing=coin_summary.get("missing_seconds", "n/a"),
            )
        )

    lines.extend(["", "## Validation Summary", ""])
    for coin_summary in summary["coins"]:
        lines.append(f"### {coin_summary['coin']}")
        lines.append(
            f"- coverage: {_format_timestamp_ms(coin_summary['coverage_start'])} "
            f"to {_format_timestamp_ms(coin_summary['coverage_end'])}"
        )
        for interval, details in coin_summary["validation_by_interval"].items():
            passed = details.get("passed")
            lines.append(
                f"- {interval}: passed={passed} "
                f"missing_periods={details.get('missing_periods_count', 'n/a')}"
            )
        lines.append("")

    lines.extend(["## Training Summary", ""])
    for coin_summary in summary["coins"]:
        lines.append(f"### {coin_summary['coin']}")
        lines.append(f"- interval: {coin_summary.get('training_interval', 'n/a')}")
        lines.append(f"- model_used: {coin_summary.get('model_used', 'n/a')}")
        lines.append(
            "- accuracy: "
            f"{_format_accuracy(coin_summary.get('training_accuracy'))} "
            f"(baseline {_format_accuracy(coin_summary.get('baseline_accuracy'))})"
        )
        lines.append(
            f"- rows: train={coin_summary.get('train_rows', 'n/a')} "
            f"test={coin_summary.get('test_rows', 'n/a')}"
        )
        lines.append(f"- note: {RESEARCH_ONLY_WARNING}")
        lines.append("")

    lines.extend(["## Limitations", ""])
    for note in overall["limitations"]:
        lines.append(f"- {note}")

    lines.extend(["", "## Warnings", ""])
    for warning in overall["warnings"]:
        lines.append(f"- {warning}")

    lines.extend(["", "## Next Recommended Commands", ""])
    for command in NEXT_RECOMMENDED_COMMANDS:
        lines.append(f"- `{command}`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["markdown_report_path"] = str(output_path)
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only daily bid/ask research summary from local CSVs "
            "and JSON reports (paper-only)."
        )
    )
    parser.add_argument(
        "--coins",
        nargs="+",
        default=["BTC", "ETH", "SOL"],
        help="Coins to include (default: BTC ETH SOL)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing bid/ask CSV files",
    )
    parser.add_argument(
        "--training-reports-dir",
        type=Path,
        default=DEFAULT_TRAINING_REPORTS_DIR,
        help="Directory containing bid/ask training JSON reports",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for daily summary JSON/Markdown output",
    )
    parser.add_argument(
        "--skip-backfill-tool-check",
        action="store_true",
        help="Do not inspect historical backfill tool availability",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    coins = [coin.upper() for coin in args.coins]

    summary = build_daily_summary(
        coins,
        data_root=args.data_root,
        training_reports_dir=args.training_reports_dir,
        include_backfill_tool_check=not args.skip_backfill_tool_check,
    )
    json_path = write_summary_json(summary, output_dir=args.output_dir)
    markdown_path = write_summary_markdown(summary, output_dir=args.output_dir)

    ready = summary["overall_status"]["ready_for_review"]
    print(f"json_report_path={json_path}")
    print(f"markdown_report_path={markdown_path}")
    print(f"ready_for_review={str(ready).lower()}")
    for warning in summary["overall_status"]["warnings"]:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
