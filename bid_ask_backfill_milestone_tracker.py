"""Read-only backfill milestone tracker for bid/ask historical CSV coverage.

Paper-only and research-only. Reads local CSV files and optional validator JSON
reports only. Writes milestone summaries under reports/backfill_milestones/.
No network, AWS, shell commands, orders, wallets, or execution.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bid_ask_schema import (
    ASK_SIZE,
    BID_SIZE,
    EXECUTION_ENABLED,
    PAPER_ONLY,
    SAFETY_STATEMENT,
    TIMESTAMP_ISO,
    TIMESTAMP_MS,
    configure_csv_field_size_limit,
    timestamp_ms_to_iso,
)
from bid_ask_validator import bid_ask_csv_path

DEFAULT_DATA_ROOT = Path("data/bid_ask_csv")
DEFAULT_OUTPUT_DIR = Path("reports/backfill_milestones")
DEFAULT_LATEST_JSON_NAME = "backfill_milestones_latest.json"
DEFAULT_LATEST_MD_NAME = "backfill_milestones_latest.md"

RESEARCH_COINS = ("BTC", "ETH")
INTERVALS = ("1s", "15s", "1m")
HISTORICAL_SOURCE = "historical"

MILESTONE_TARGETS_DAYS: tuple[tuple[str, int], ...] = (
    ("3_days", 3),
    ("7_days", 7),
    ("14_days", 14),
    ("30_days", 30),
)

RESEARCH_ONLY_NOTE = (
    "Research-only milestone tracker. Coverage metrics track historical data "
    "accumulation only. Real trading remains disabled."
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
    if "_validation_" in stem:
        return stem.rsplit("_validation_", 1)[-1]
    if stem.startswith("backfill_milestones_"):
        return stem.rsplit("_", 1)[-1]
    return stem


def _parse_iso_to_ms(iso: str) -> int | None:
    if not iso:
        return None
    try:
        normalized = iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError, OSError):
        return None


def _is_blank_timestamp_value(raw: str | None) -> bool:
    if raw is None:
        return True
    text = str(raw).strip()
    return not text or text.lower() in {"nan", "null", "none"}


def _parse_timestamp_ms(raw: str | None) -> int | None:
    if _is_blank_timestamp_value(raw):
        return None
    try:
        value = float(str(raw).strip())
        if value != value or value in {float("inf"), float("-inf")}:
            return None

        timestamp_ms = int(value)
        if timestamp_ms <= 0:
            return None

        return timestamp_ms
    except (TypeError, ValueError, OverflowError):
        return None


def _parse_timestamp_iso(raw: str | None) -> str | None:
    if _is_blank_timestamp_value(raw):
        return None
    text = str(raw).strip()
    if _parse_iso_to_ms(text) is None:
        return None
    return text


def _row_first_timestamp(row: dict[str, str], *, interval: str) -> tuple[int | None, str | None]:
    if interval == "1s":
        primary_raw = row.get(TIMESTAMP_MS)
    else:
        primary_raw = row.get("start_timestamp_ms")

    iso_raw = row.get(TIMESTAMP_ISO)
    iso = _parse_timestamp_iso(iso_raw)

    ts_ms = _parse_timestamp_ms(primary_raw)
    if ts_ms is not None:
        return ts_ms, iso or timestamp_ms_to_iso(ts_ms)

    if iso is not None:
        return _parse_iso_to_ms(iso), iso
    return None, None


def _row_latest_timestamp(row: dict[str, str], *, interval: str) -> tuple[int | None, str | None]:
    if interval == "1s":
        return _row_first_timestamp(row, interval=interval)

    end_raw = row.get("end_timestamp_ms")
    end_ms = _parse_timestamp_ms(end_raw)
    if end_ms is not None:
        iso = _parse_timestamp_iso(row.get(TIMESTAMP_ISO))
        return end_ms, iso or timestamp_ms_to_iso(end_ms)

    return _row_first_timestamp(row, interval=interval)


def _has_usable_depth_value(raw: str | None) -> bool:
    if raw is None:
        return False
    text = str(raw).strip()
    if not text:
        return False
    try:
        return float(text) == float(text)
    except (TypeError, ValueError):
        return False


def _analyze_csv(path: Path, *, interval: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "file_exists": path.is_file(),
        "path": str(path),
        "row_count": 0,
        "first_timestamp_ms": None,
        "latest_timestamp_ms": None,
        "first_timestamp_iso": None,
        "latest_timestamp_iso": None,
        "coverage_hours": 0.0,
        "bid_size_present": False,
        "ask_size_present": False,
        "microprice_features_expected": False,
    }

    if not path.is_file():
        return result

    first_ts: int | None = None
    latest_ts: int | None = None
    first_iso: str | None = None
    latest_iso: str | None = None
    has_bid_size = False
    has_ask_size = False
    header_has_bid = False
    header_has_ask = False

    try:
        configure_csv_field_size_limit()
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            header_has_bid = BID_SIZE in fieldnames
            header_has_ask = ASK_SIZE in fieldnames

            for row in reader:
                result["row_count"] += 1

                row_first_ms, row_first_iso = _row_first_timestamp(row, interval=interval)
                row_latest_ms, row_latest_iso = _row_latest_timestamp(row, interval=interval)

                if row_first_ms is not None:
                    if first_ts is None or row_first_ms < first_ts:
                        first_ts = row_first_ms
                        first_iso = row_first_iso
                if row_latest_ms is not None:
                    if latest_ts is None or row_latest_ms > latest_ts:
                        latest_ts = row_latest_ms
                        latest_iso = row_latest_iso

                if header_has_bid and _has_usable_depth_value(row.get(BID_SIZE)):
                    has_bid_size = True
                if header_has_ask and _has_usable_depth_value(row.get(ASK_SIZE)):
                    has_ask_size = True
    except OSError:
        return result

    result["first_timestamp_ms"] = first_ts
    result["latest_timestamp_ms"] = latest_ts
    result["first_timestamp_iso"] = first_iso or (timestamp_ms_to_iso(first_ts) if first_ts is not None else None)
    result["latest_timestamp_iso"] = latest_iso or (timestamp_ms_to_iso(latest_ts) if latest_ts is not None else None)

    if first_ts is not None and latest_ts is not None and latest_ts >= first_ts:
        result["coverage_hours"] = round((latest_ts - first_ts) / 3_600_000, 3)

    result["bid_size_present"] = header_has_bid and has_bid_size
    result["ask_size_present"] = header_has_ask and has_ask_size
    result["microprice_features_expected"] = result["bid_size_present"] and result["ask_size_present"]
    return result


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
            enriched = dict(payload)
            enriched["_report_path"] = str(path)
            return enriched
    return None


def _validator_warning_count(report: dict[str, Any] | None) -> int | None:
    if report is None:
        return None

    missing_count = report.get("missing_periods_count")
    if isinstance(missing_count, int):
        return missing_count

    warnings = report.get("warnings") or []
    if not isinstance(warnings, list):
        return None

    count = 0
    for warning in warnings:
        text = str(warning).lower()
        if "missing" in text or "incomplete" in text:
            count += 1
    return count if warnings else 0


def _milestone_progress(coverage_hours: float) -> dict[str, Any]:
    milestones: dict[str, Any] = {}
    next_milestone: str | None = None

    for label, days in MILESTONE_TARGETS_DAYS:
        target_hours = days * 24
        if target_hours <= 0:
            percent = 0.0
        else:
            percent = min(coverage_hours / target_hours * 100.0, 100.0)
        milestones[label] = {
            "target_days": days,
            "target_hours": target_hours,
            "percent_complete": round(percent, 2),
            "met": percent >= 100.0,
        }
        if next_milestone is None and percent < 100.0:
            next_milestone = label

    if next_milestone is None:
        next_milestone = "all_targets_met"

    return {
        "coverage_hours": coverage_hours,
        "milestones": milestones,
        "next_milestone": next_milestone,
    }


def _coin_interval_coverage(coin: str, interval: str, data_root: Path) -> dict[str, Any]:
    path = bid_ask_csv_path(coin, interval, data_root, source=HISTORICAL_SOURCE)
    csv_stats = _analyze_csv(path, interval=interval)
    validation = _latest_validation_report(coin, interval, data_root)
    warning_count = _validator_warning_count(validation)

    progress = _milestone_progress(float(csv_stats["coverage_hours"] or 0.0))

    return {
        "coin": coin,
        "interval": interval,
        "file_exists": csv_stats["file_exists"],
        "path": csv_stats["path"],
        "row_count": csv_stats["row_count"],
        "first_timestamp_ms": csv_stats["first_timestamp_ms"],
        "latest_timestamp_ms": csv_stats["latest_timestamp_ms"],
        "first_timestamp_iso": csv_stats["first_timestamp_iso"],
        "latest_timestamp_iso": csv_stats["latest_timestamp_iso"],
        "coverage_hours": csv_stats["coverage_hours"],
        "validator_report_available": validation is not None,
        "validator_report_path": (validation or {}).get("_report_path"),
        "validator_passed": (validation or {}).get("passed"),
        "missing_incomplete_warning_count": warning_count,
        "bid_size_present": csv_stats["bid_size_present"],
        "ask_size_present": csv_stats["ask_size_present"],
        "microprice_features_expected": csv_stats["microprice_features_expected"],
        "milestone_progress": progress,
        "next_milestone": progress["next_milestone"],
    }


def _collect_data_quality_warnings(coverage: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    warnings: list[str] = []
    for coin in RESEARCH_COINS:
        for interval in INTERVALS:
            entry = coverage.get(coin, {}).get(interval, {})
            if not entry.get("file_exists"):
                warnings.append(f"{coin} {interval}: CSV missing at {entry.get('path')}")
                continue
            if entry.get("row_count", 0) <= 0:
                warnings.append(f"{coin} {interval}: CSV exists but contains no rows")
                continue

            warning_count = entry.get("missing_incomplete_warning_count")
            if warning_count is None:
                warnings.append(f"{coin} {interval}: no validator report available")
            elif warning_count > 0:
                warnings.append(
                    f"{coin} {interval}: validator reports {warning_count} missing/incomplete period(s)"
                )

            if entry.get("validator_report_available") and entry.get("validator_passed") is False:
                warnings.append(f"{coin} {interval}: latest validator report failed")
    return warnings


def _collect_depth_feature_status(coverage: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    by_coin_interval: dict[str, dict[str, Any]] = {}
    gaps: list[str] = []

    for coin in RESEARCH_COINS:
        by_coin_interval[coin] = {}
        for interval in INTERVALS:
            entry = coverage.get(coin, {}).get(interval, {})
            status = {
                "bid_size_present": entry.get("bid_size_present", False),
                "ask_size_present": entry.get("ask_size_present", False),
                "microprice_features_expected": entry.get("microprice_features_expected", False),
            }
            by_coin_interval[coin][interval] = status

            if interval in {"15s", "1m"} and entry.get("file_exists") and entry.get("row_count", 0) > 0:
                if not status["microprice_features_expected"]:
                    gaps.append(
                        f"{coin} {interval}: bid_size/ask_size missing or empty; microprice family unavailable"
                    )

    return {
        "by_coin_interval": by_coin_interval,
        "gaps": gaps,
        "resampled_microprice_ready": not gaps,
    }


def _recommend_next_action(
    coverage: dict[str, dict[str, dict[str, Any]]],
    *,
    data_quality_warnings: list[str],
    depth_status: dict[str, Any],
) -> str:
    missing_csvs = [
        f"{coin} {interval}"
        for coin in RESEARCH_COINS
        for interval in INTERVALS
        if not coverage.get(coin, {}).get(interval, {}).get("file_exists")
    ]
    if missing_csvs:
        return (
            "Continue scheduled backfill and generate missing CSVs for "
            + ", ".join(missing_csvs[:4])
            + (" and others" if len(missing_csvs) > 4 else "")
        )

    lowest = None
    lowest_hours = None
    for coin in RESEARCH_COINS:
        entry = coverage.get(coin, {}).get("1s", {})
        hours = float(entry.get("coverage_hours") or 0.0)
        if lowest_hours is None or hours < lowest_hours:
            lowest_hours = hours
            lowest = coin

    if lowest and lowest_hours is not None:
        progress = coverage[lowest]["1s"]["milestone_progress"]
        next_milestone = progress.get("next_milestone")
        if next_milestone and next_milestone != "all_targets_met":
            target_days = progress["milestones"][next_milestone]["target_days"]
            return (
                f"Continue backfill for {lowest} 1s data toward the {target_days}-day milestone "
                f"({progress['milestones'][next_milestone]['percent_complete']:.1f}% complete)."
            )

    if depth_status.get("gaps"):
        return "Re-run resample for 15s/1m bars so bid_size/ask_size columns are populated."

    if data_quality_warnings:
        return "Run validator on latest CSV batches and review missing/incomplete warnings."

    return "All milestone targets met for tracked coins; continue validation and resample maintenance."


def build_backfill_milestone_report(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    coverage: dict[str, dict[str, dict[str, Any]]] = {}
    milestone_progress: dict[str, dict[str, Any]] = {}

    for coin in RESEARCH_COINS:
        coverage[coin] = {}
        milestone_progress[coin] = {}
        for interval in INTERVALS:
            entry = _coin_interval_coverage(coin, interval, data_root)
            coverage[coin][interval] = entry
            milestone_progress[coin][interval] = entry["milestone_progress"]

    data_quality_warnings = _collect_data_quality_warnings(coverage)
    depth_feature_status = _collect_depth_feature_status(coverage)

    return {
        "generated_at": _utc_now_iso(),
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "real_execution_ready": False,
        "safety_statement": SAFETY_STATEMENT,
        "research_only_note": RESEARCH_ONLY_NOTE,
        "milestone_targets_days": [days for _, days in MILESTONE_TARGETS_DAYS],
        "data_root": str(data_root),
        "coverage": coverage,
        "milestone_progress": milestone_progress,
        "data_quality_warnings": data_quality_warnings,
        "depth_feature_status": depth_feature_status,
        "next_recommended_action": _recommend_next_action(
            coverage,
            data_quality_warnings=data_quality_warnings,
            depth_status=depth_feature_status,
        ),
    }


def write_milestone_json(
    report: dict[str, Any],
    output_dir: Path,
    *,
    filename: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def write_milestone_markdown(
    report: dict[str, Any],
    output_dir: Path,
    *,
    filename: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    lines = [
        "# Bid/Ask Backfill Milestone Tracker",
        "",
        f"Generated at: {report['generated_at']}",
        "",
        "## 1. Safety Statement",
        "",
        f"- paper_only: {str(report['paper_only']).lower()}",
        f"- execution_enabled: {str(report['execution_enabled']).lower()}",
        f"- real_execution_ready: {str(report['real_execution_ready']).lower()}",
        "",
        report["safety_statement"],
        "",
        report["research_only_note"],
        "",
        "## 2. Coverage Summary",
        "",
        "| Coin | Interval | Rows | First | Latest | Coverage Hours |",
        "| --- | --- | ---: | --- | --- | ---: |",
    ]

    for coin in RESEARCH_COINS:
        for interval in INTERVALS:
            entry = report["coverage"][coin][interval]
            lines.append(
                f"| {coin} | {interval} | {entry.get('row_count', 0)} | "
                f"{entry.get('first_timestamp_iso') or 'n/a'} | "
                f"{entry.get('latest_timestamp_iso') or 'n/a'} | "
                f"{entry.get('coverage_hours', 0.0)} |"
            )

    lines.extend(["", "## 3. Milestone Progress Table", ""])
    header_labels = [label.replace("_", " ") for label, _ in MILESTONE_TARGETS_DAYS]
    lines.append(
        "| Coin | Interval | Coverage Hours | Next Milestone | "
        + " | ".join(header_labels)
        + " |"
    )
    lines.append(
        "| --- | --- | ---: | --- | "
        + " | ".join("---:" for _ in MILESTONE_TARGETS_DAYS)
        + " |"
    )

    for coin in RESEARCH_COINS:
        for interval in INTERVALS:
            entry = report["coverage"][coin][interval]
            progress = entry.get("milestone_progress") or {}
            milestones = progress.get("milestones") or {}
            percents = [
                f"{milestones.get(label, {}).get('percent_complete', 0.0):.1f}%"
                for label, _ in MILESTONE_TARGETS_DAYS
            ]
            lines.append(
                f"| {coin} | {interval} | {entry.get('coverage_hours', 0.0)} | "
                f"{entry.get('next_milestone', 'n/a')} | "
                + " | ".join(percents)
                + " |"
            )

    lines.extend(["", "## 4. Data Quality Warnings", ""])
    if report["data_quality_warnings"]:
        lines.extend(f"- {item}" for item in report["data_quality_warnings"])
    else:
        lines.append("- None recorded.")

    lines.extend(["", "## 5. Depth Feature Status", ""])
    depth = report["depth_feature_status"]
    for coin in RESEARCH_COINS:
        for interval in INTERVALS:
            status = depth["by_coin_interval"][coin][interval]
            lines.append(
                f"- {coin} {interval}: bid_size={status['bid_size_present']}, "
                f"ask_size={status['ask_size_present']}, "
                f"microprice_expected={status['microprice_features_expected']}"
            )
    if depth.get("gaps"):
        lines.append("")
        lines.append("Gaps:")
        lines.extend(f"- {item}" for item in depth["gaps"])

    lines.extend(
        [
            "",
            "## 6. Next Recommended Action",
            "",
            f"- {report['next_recommended_action']}",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def build_and_write_milestone_report(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    latest_json_name: str = DEFAULT_LATEST_JSON_NAME,
    latest_md_name: str = DEFAULT_LATEST_MD_NAME,
) -> dict[str, Any]:
    report = build_backfill_milestone_report(data_root=data_root)
    stamp = _utc_stamp()

    dated_json_name = f"backfill_milestones_{stamp}.json"
    dated_md_name = f"backfill_milestones_{stamp}.md"

    json_dated = write_milestone_json(report, output_dir, filename=dated_json_name)
    md_dated = write_milestone_markdown(report, output_dir, filename=dated_md_name)
    json_latest = write_milestone_json(report, output_dir, filename=latest_json_name)
    md_latest = write_milestone_markdown(report, output_dir, filename=latest_md_name)

    report["output_paths"] = {
        "json_dated": str(json_dated),
        "markdown_dated": str(md_dated),
        "json_latest": str(json_latest),
        "markdown_latest": str(md_latest),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build bid/ask backfill milestone tracker reports (paper-only, read-only inputs).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build",
        help="Read local CSV coverage and write milestone tracker reports.",
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
        help="Directory for milestone tracker output files.",
    )
    build_parser.add_argument(
        "--json-latest-name",
        default=DEFAULT_LATEST_JSON_NAME,
        help="Filename for the latest JSON report copy.",
    )
    build_parser.add_argument(
        "--latest-name",
        default=DEFAULT_LATEST_MD_NAME,
        help="Filename for the latest Markdown report copy.",
    )

    args = parser.parse_args(argv)

    if args.command == "build":
        report = build_and_write_milestone_report(
            data_root=args.data_root,
            output_dir=args.output_dir,
            latest_json_name=args.json_latest_name,
            latest_md_name=args.latest_name,
        )
        paths = report["output_paths"]
        print("backfill_milestone_tracker build complete")
        print(f"  json_latest_path={paths['json_latest']}")
        print(f"  markdown_latest_path={paths['markdown_latest']}")
        print(f"  json_dated_path={paths['json_dated']}")
        print(f"  markdown_dated_path={paths['markdown_dated']}")
        print(f"  paper_only={str(report['paper_only']).lower()}")
        print(f"  execution_enabled={str(report['execution_enabled']).lower()}")
        print(f"  real_execution_ready={str(report['real_execution_ready']).lower()}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
