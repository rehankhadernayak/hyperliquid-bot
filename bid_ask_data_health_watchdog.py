#!/usr/bin/env python3
"""Read-only data health watchdog for bid/ask CSV pipelines."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bid_ask_resampler import DEFAULT_DATA_ROOT, normalize_source
from bid_ask_schema import (
    BEST_ASK,
    BEST_BID,
    EXECUTION_ENABLED,
    PAPER_ONLY,
    SAFETY_STATEMENT,
    TIMESTAMP_ISO,
    TIMESTAMP_MS,
)
from bid_ask_validator import bid_ask_csv_path, reports_dir

DEFAULT_COINS = ("BTC", "ETH")
DEFAULT_SOURCE = "historical"
DEFAULT_REPORTS_ROOT = Path("reports")
DEFAULT_OUTPUT_DIR = Path("reports/data_health")
DEFAULT_JSON_NAME = "data_health_latest.json"
DEFAULT_MARKDOWN_NAME = "data_health_latest.md"
DEFAULT_MAX_STALE_HOURS = 12.0
CANDIDATE_UNCHANGED_LIMIT = 3
PHASE3_AUDIT_RELATIVE = Path("bid_ask_research_scheduler") / "phase3_audit_latest.json"
CANDIDATE_VALIDATION_DIR = Path("bid_ask_candidate_validation")
VALIDATION_INTERVALS = ("1s", "15s")

SEVERITY_CRITICAL = "critical"
SEVERITY_WARNING = "warning"
STATUS_HEALTHY = "healthy"
STATUS_WARNING = "warning"
STATUS_CRITICAL = "critical"


@dataclass
class HealthIssue:
    severity: str
    code: str
    message: str
    coin: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.coin is not None:
            payload["coin"] = self.coin
        return payload


@dataclass
class CoinCsvMetrics:
    coin: str
    interval: str
    path: str
    exists: bool = False
    row_count: int = 0
    latest_timestamp_ms: int | None = None
    duplicate_timestamp_count: int = 0
    non_monotonic_count: int = 0
    non_numeric_timestamp_count: int = 0
    missing_timestamp_iso_count: int = 0
    invalid_bid_ask_count: int = 0
    inverted_bid_ask_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "coin": self.coin,
            "interval": self.interval,
            "path": self.path,
            "exists": self.exists,
            "row_count": self.row_count,
            "latest_timestamp_ms": self.latest_timestamp_ms,
            "duplicate_timestamp_count": self.duplicate_timestamp_count,
            "non_monotonic_count": self.non_monotonic_count,
            "non_numeric_timestamp_count": self.non_numeric_timestamp_count,
            "missing_timestamp_iso_count": self.missing_timestamp_iso_count,
            "invalid_bid_ask_count": self.invalid_bid_ask_count,
            "inverted_bid_ask_count": self.inverted_bid_ask_count,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _duplicate_timestamp_count(series: pd.Series) -> int:
    if series.empty:
        return 0
    counts = series.value_counts()
    return int(sum(count - 1 for count in counts if count > 1))


def _non_monotonic_count(series: pd.Series) -> int:
    if len(series) < 2:
        return 0
    values = series.tolist()
    return sum(1 for index in range(1, len(values)) if values[index] <= values[index - 1])


def _count_missing_iso(series: pd.Series) -> int:
    if series.empty:
        return 0
    missing = series.isna() | (series.astype(str).str.strip() == "")
    return int(missing.sum())


def _count_invalid_prices(series: pd.Series) -> int:
    numeric = pd.to_numeric(series, errors="coerce")
    return int(numeric.isna().sum())


def _analyze_1s_csv(coin: str, path: Path) -> tuple[CoinCsvMetrics, list[HealthIssue]]:
    metrics = CoinCsvMetrics(coin=coin, interval="1s", path=str(path))
    issues: list[HealthIssue] = []

    if not path.is_file():
        issues.append(
            HealthIssue(
                severity=SEVERITY_CRITICAL,
                code="missing_csv",
                message=f"Missing historical_1s CSV: {path}",
                coin=coin,
            )
        )
        return metrics, issues

    metrics.exists = True
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        issues.append(
            HealthIssue(
                severity=SEVERITY_CRITICAL,
                code="csv_read_error",
                message=f"Unable to read {path}: {exc}",
                coin=coin,
            )
        )
        return metrics, issues

    metrics.row_count = len(df)
    if metrics.row_count == 0:
        issues.append(
            HealthIssue(
                severity=SEVERITY_CRITICAL,
                code="empty_csv",
                message=f"historical_1s CSV has zero rows: {path}",
                coin=coin,
            )
        )
        return metrics, issues

    if TIMESTAMP_MS not in df.columns:
        issues.append(
            HealthIssue(
                severity=SEVERITY_CRITICAL,
                code="missing_timestamp_column",
                message=f"Missing {TIMESTAMP_MS} column in {path}",
                coin=coin,
            )
        )
        return metrics, issues

    numeric_ts = pd.to_numeric(df[TIMESTAMP_MS], errors="coerce")
    metrics.non_numeric_timestamp_count = int(numeric_ts.isna().sum())
    if metrics.non_numeric_timestamp_count:
        issues.append(
            HealthIssue(
                severity=SEVERITY_CRITICAL,
                code="non_numeric_timestamp",
                message=(
                    f"{metrics.non_numeric_timestamp_count} non-numeric "
                    f"{TIMESTAMP_MS} value(s) in {path.name}"
                ),
                coin=coin,
            )
        )

    valid_ts = numeric_ts.dropna().astype("int64")
    if not valid_ts.empty:
        metrics.latest_timestamp_ms = int(valid_ts.max())
        metrics.duplicate_timestamp_count = _duplicate_timestamp_count(valid_ts)
        metrics.non_monotonic_count = _non_monotonic_count(valid_ts)

    if metrics.duplicate_timestamp_count:
        issues.append(
            HealthIssue(
                severity=SEVERITY_CRITICAL,
                code="duplicate_timestamps",
                message=(
                    f"{metrics.duplicate_timestamp_count} duplicate "
                    f"{TIMESTAMP_MS} value(s) in {path.name}"
                ),
                coin=coin,
            )
        )

    if metrics.non_monotonic_count:
        issues.append(
            HealthIssue(
                severity=SEVERITY_CRITICAL,
                code="non_monotonic_timestamps",
                message=(
                    f"{metrics.non_monotonic_count} non-monotonic "
                    f"{TIMESTAMP_MS} transition(s) in {path.name}"
                ),
                coin=coin,
            )
        )

    if TIMESTAMP_ISO in df.columns:
        metrics.missing_timestamp_iso_count = _count_missing_iso(df[TIMESTAMP_ISO])
        if metrics.missing_timestamp_iso_count:
            issues.append(
                HealthIssue(
                    severity=SEVERITY_WARNING,
                    code="missing_timestamp_iso",
                    message=(
                        f"{metrics.missing_timestamp_iso_count} missing "
                        f"{TIMESTAMP_ISO} value(s) in {path.name}"
                    ),
                    coin=coin,
                )
            )

    for price_column in (BEST_BID, BEST_ASK):
        if price_column not in df.columns:
            issues.append(
                HealthIssue(
                    severity=SEVERITY_CRITICAL,
                    code="missing_price_column",
                    message=f"Missing {price_column} column in {path.name}",
                    coin=coin,
                )
            )
            continue

        invalid_count = _count_invalid_prices(df[price_column])
        metrics.invalid_bid_ask_count += invalid_count
        if invalid_count:
            issues.append(
                HealthIssue(
                    severity=SEVERITY_CRITICAL,
                    code="invalid_bid_ask",
                    message=f"{invalid_count} invalid {price_column} value(s) in {path.name}",
                    coin=coin,
                )
            )

    if BEST_BID in df.columns and BEST_ASK in df.columns:
        bid = pd.to_numeric(df[BEST_BID], errors="coerce")
        ask = pd.to_numeric(df[BEST_ASK], errors="coerce")
        inverted_mask = ask.notna() & bid.notna() & (ask < bid)
        metrics.inverted_bid_ask_count = int(inverted_mask.sum())
        if metrics.inverted_bid_ask_count:
            issues.append(
                HealthIssue(
                    severity=SEVERITY_CRITICAL,
                    code="inverted_bid_ask",
                    message=(
                        f"{metrics.inverted_bid_ask_count} row(s) have "
                        f"{BEST_ASK} < {BEST_BID} in {path.name}"
                    ),
                    coin=coin,
                )
            )

    return metrics, issues


def _analyze_15s_csv(coin: str, path: Path) -> tuple[CoinCsvMetrics, list[HealthIssue]]:
    metrics = CoinCsvMetrics(coin=coin, interval="15s", path=str(path))
    issues: list[HealthIssue] = []

    if not path.is_file():
        issues.append(
            HealthIssue(
                severity=SEVERITY_CRITICAL,
                code="missing_csv",
                message=f"Missing historical_15s CSV: {path}",
                coin=coin,
            )
        )
        return metrics, issues

    metrics.exists = True
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        issues.append(
            HealthIssue(
                severity=SEVERITY_CRITICAL,
                code="csv_read_error",
                message=f"Unable to read {path}: {exc}",
                coin=coin,
            )
        )
        return metrics, issues

    metrics.row_count = len(df)
    if metrics.row_count == 0:
        issues.append(
            HealthIssue(
                severity=SEVERITY_CRITICAL,
                code="empty_csv",
                message=f"historical_15s CSV has zero rows: {path}",
                coin=coin,
            )
        )
        return metrics, issues

    timestamp_column = "end_timestamp_ms" if "end_timestamp_ms" in df.columns else "start_timestamp_ms"
    if timestamp_column not in df.columns:
        issues.append(
            HealthIssue(
                severity=SEVERITY_CRITICAL,
                code="missing_timestamp_column",
                message=f"Missing resampled timestamp column in {path.name}",
                coin=coin,
            )
        )
        return metrics, issues

    numeric_ts = pd.to_numeric(df[timestamp_column], errors="coerce")
    valid_ts = numeric_ts.dropna().astype("int64")
    if not valid_ts.empty:
        metrics.latest_timestamp_ms = int(valid_ts.max())

    return metrics, issues


def _validation_report_pattern(coin: str, interval: str) -> str:
    return f"{coin}_{interval}_validation_*.json"


def load_latest_validation_report(
    coin: str,
    interval: str,
    data_root: Path,
) -> dict[str, Any] | None:
    report_dir = reports_dir(data_root)
    if not report_dir.is_dir():
        return None

    matches = sorted(
        report_dir.glob(_validation_report_pattern(coin, interval)),
        key=lambda path: path.name,
    )
    if not matches:
        return None

    try:
        return json.loads(matches[-1].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _check_validation_reports(
    coins: list[str],
    data_root: Path,
) -> tuple[list[dict[str, Any]], list[HealthIssue]]:
    results: list[dict[str, Any]] = []
    issues: list[HealthIssue] = []

    for coin in coins:
        for interval in VALIDATION_INTERVALS:
            report = load_latest_validation_report(coin, interval, data_root)
            entry = {
                "coin": coin,
                "interval": interval,
                "report_path": report.get("report_path") if report else None,
                "passed": report.get("passed") if report else None,
                "generated_at": report.get("generated_at") if report else None,
            }
            results.append(entry)

            if report is None:
                issues.append(
                    HealthIssue(
                        severity=SEVERITY_WARNING,
                        code="missing_validation_report",
                        message=f"No validation report found for {coin} {interval}",
                        coin=coin,
                    )
                )
                continue

            if not report.get("passed", False):
                issues.append(
                    HealthIssue(
                        severity=SEVERITY_WARNING,
                        code="validation_failed",
                        message=f"Latest validation report failed for {coin} {interval}",
                        coin=coin,
                    )
                )

    return results, issues


def _check_stale_15s(
    coins: list[str],
    one_s_metrics: dict[str, CoinCsvMetrics],
    fifteen_s_metrics: dict[str, CoinCsvMetrics],
) -> list[HealthIssue]:
    issues: list[HealthIssue] = []
    for coin in coins:
        one_s = one_s_metrics.get(coin)
        fifteen_s = fifteen_s_metrics.get(coin)
        if not one_s or not fifteen_s:
            continue
        if one_s.latest_timestamp_ms is None or fifteen_s.latest_timestamp_ms is None:
            continue
        if fifteen_s.latest_timestamp_ms < one_s.latest_timestamp_ms:
            issues.append(
                HealthIssue(
                    severity=SEVERITY_WARNING,
                    code="stale_15s_while_1s_advanced",
                    message=(
                        f"{coin} 15s data ends at {fifteen_s.latest_timestamp_ms} "
                        f"while 1s data ends at {one_s.latest_timestamp_ms}"
                    ),
                    coin=coin,
                )
            )
    return issues


def _check_stale_data(
    coins: list[str],
    one_s_metrics: dict[str, CoinCsvMetrics],
    *,
    max_stale_hours: float,
) -> list[HealthIssue]:
    issues: list[HealthIssue] = []
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    threshold_ms = int(max_stale_hours * 3600 * 1000)

    for coin in coins:
        metrics = one_s_metrics.get(coin)
        if not metrics or metrics.latest_timestamp_ms is None:
            continue
        age_ms = now_ms - metrics.latest_timestamp_ms
        if age_ms > threshold_ms:
            age_hours = age_ms / 3_600_000
            issues.append(
                HealthIssue(
                    severity=SEVERITY_WARNING,
                    code="stale_1s_data",
                    message=(
                        f"{coin} latest 1s timestamp is {age_hours:.1f}h old "
                        f"(threshold {max_stale_hours}h)"
                    ),
                    coin=coin,
                )
            )
    return issues


def load_phase_audit(reports_root: Path) -> dict[str, Any] | None:
    audit_path = reports_root / PHASE3_AUDIT_RELATIVE
    if not audit_path.is_file():
        return None
    try:
        return json.loads(audit_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _check_phase_audit_downloads(audit: dict[str, Any] | None, coins: list[str]) -> list[HealthIssue]:
    if audit is None:
        return []

    issues: list[HealthIssue] = []
    downloads = audit.get("downloads") or {}
    total_files = int(downloads.get("total_files") or 0)
    if total_files <= 0:
        return []

    imported = audit.get("imported_historical_1s") or {}
    for coin in coins:
        entry = imported.get(coin) or {}
        freshness = entry.get("freshness_status")
        if freshness in {"stale", "missing"}:
            issues.append(
                HealthIssue(
                    severity=SEVERITY_WARNING,
                    code="downloads_exist_1s_not_advancing",
                    message=(
                        f"Phase audit shows downloads present but {coin} historical_1s "
                        f"freshness is {freshness}"
                    ),
                    coin=coin,
                )
            )
    return issues


def _candidate_status_label(report: dict[str, Any]) -> str:
    status = report.get("validation_status") or {}
    if status.get("paper_candidate_pass"):
        return "paper_candidate_pass"
    return "not_market_ready"


def load_candidate_validation_reports(reports_dir_path: Path) -> list[dict[str, Any]]:
    if not reports_dir_path.is_dir():
        return []

    reports: list[dict[str, Any]] = []
    for path in sorted(reports_dir_path.glob("candidate_validation_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        payload.setdefault("json_report_path", str(path))
        reports.append(payload)

    reports.sort(key=lambda item: str(item.get("run_timestamp") or ""))
    return reports


def _detect_repeated_unchanged_candidate_reports(
    reports: list[dict[str, Any]],
    *,
    limit: int = CANDIDATE_UNCHANGED_LIMIT,
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


def _check_candidate_unchanged(reports_root: Path) -> tuple[dict[str, Any], list[HealthIssue]]:
    candidate_dir = reports_root / CANDIDATE_VALIDATION_DIR
    reports = load_candidate_validation_reports(candidate_dir)
    summary: dict[str, Any] = {
        "report_count": len(reports),
        "repeated_unchanged_warning": False,
        "unchanged_reports_checked": min(len(reports), CANDIDATE_UNCHANGED_LIMIT),
        "latest_run_timestamp": reports[-1].get("run_timestamp") if reports else None,
    }
    issues: list[HealthIssue] = []

    if _detect_repeated_unchanged_candidate_reports(reports):
        summary["repeated_unchanged_warning"] = True
        issues.append(
            HealthIssue(
                severity=SEVERITY_WARNING,
                code="candidate_result_unchanged",
                message=(
                    f"Candidate validation results unchanged across the last "
                    f"{CANDIDATE_UNCHANGED_LIMIT} reports"
                ),
            )
        )

    return summary, issues


def _aggregate_status(issues: list[HealthIssue]) -> str:
    if any(issue.severity == SEVERITY_CRITICAL for issue in issues):
        return STATUS_CRITICAL
    if any(issue.severity == SEVERITY_WARNING for issue in issues):
        return STATUS_WARNING
    return STATUS_HEALTHY


def _build_recommended_commands(
    issues: list[HealthIssue],
    *,
    coins: list[str],
    source: str,
) -> list[str]:
    coins_arg = " ".join(coins)
    commands: list[str] = []
    codes = {issue.code for issue in issues}

    if "missing_csv" in codes or "empty_csv" in codes or "csv_read_error" in codes:
        commands.append(
            "python3 bid_ask_research_scheduler.py phase-backfill-audit"
        )

    if any(
        code in codes
        for code in {
            "duplicate_timestamps",
            "non_monotonic_timestamps",
            "invalid_bid_ask",
            "inverted_bid_ask",
            "non_numeric_timestamp",
        }
    ):
        commands.append(
            f"python3 bid_ask_validator.py --coins {coins_arg} --interval 1s --source {source}"
        )

    if "validation_failed" in codes or "missing_validation_report" in codes:
        commands.append(
            f"python3 bid_ask_research_orchestrator.py resample-validate "
            f"--coins {coins_arg} --source {source}"
        )

    if "stale_15s_while_1s_advanced" in codes:
        commands.append(
            f"python3 bid_ask_resampler.py --coins {coins_arg} --interval 15s --source {source}"
        )

    if "downloads_exist_1s_not_advancing" in codes:
        commands.append("python3 bid_ask_research_scheduler.py autopilot-tick")

    if "candidate_result_unchanged" in codes:
        commands.append("python3 bid_ask_candidate_validator.py validate-latest")

    if not commands:
        commands.append(
            f"python3 bid_ask_data_health_watchdog.py check --coins {coins_arg} --source {source}"
        )

    deduped: list[str] = []
    for command in commands:
        if command not in deduped:
            deduped.append(command)
    return deduped


def build_safety_flags() -> dict[str, Any]:
    return {
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "real_execution_ready": False,
        "read_only": True,
        "no_repairs_performed": True,
        "no_network_calls": True,
        "no_subprocess_calls": True,
        "research_only": True,
        "safety_statement": SAFETY_STATEMENT,
    }


def run_data_health_check(
    *,
    coins: list[str],
    source: str = DEFAULT_SOURCE,
    data_root: Path = DEFAULT_DATA_ROOT,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    max_stale_hours: float = DEFAULT_MAX_STALE_HOURS,
) -> dict[str, Any]:
    """Run all read-only data health checks and return a structured report."""
    normalized_source = normalize_source(source)
    issues: list[HealthIssue] = []
    one_s_metrics: dict[str, CoinCsvMetrics] = {}
    fifteen_s_metrics: dict[str, CoinCsvMetrics] = {}

    for coin in coins:
        one_s_path = bid_ask_csv_path(coin, "1s", data_root, source=normalized_source)
        metrics, coin_issues = _analyze_1s_csv(coin, one_s_path)
        one_s_metrics[coin] = metrics
        issues.extend(coin_issues)

        fifteen_s_path = bid_ask_csv_path(coin, "15s", data_root, source=normalized_source)
        metrics_15s, coin_issues_15s = _analyze_15s_csv(coin, fifteen_s_path)
        fifteen_s_metrics[coin] = metrics_15s
        issues.extend(coin_issues_15s)

    issues.extend(_check_stale_15s(coins, one_s_metrics, fifteen_s_metrics))
    issues.extend(_check_stale_data(coins, one_s_metrics, max_stale_hours=max_stale_hours))

    validation_reports, validation_issues = _check_validation_reports(coins, data_root)
    issues.extend(validation_issues)

    phase_audit = load_phase_audit(reports_root)
    issues.extend(_check_phase_audit_downloads(phase_audit, coins))

    candidate_summary, candidate_issues = _check_candidate_unchanged(reports_root)
    issues.extend(candidate_issues)

    status = _aggregate_status(issues)
    recommended_commands = _build_recommended_commands(
        issues,
        coins=coins,
        source=normalized_source,
    )

    return {
        "generated_at": _utc_now_iso(),
        "command": "check",
        "status": status,
        "coins": coins,
        "source": normalized_source,
        "data_root": str(data_root),
        "reports_root": str(reports_root),
        "max_stale_hours": max_stale_hours,
        "issues": [issue.to_dict() for issue in issues],
        "recommended_commands": recommended_commands,
        "safety_flags": build_safety_flags(),
        "csv_metrics": {
            "1s": [one_s_metrics[coin].to_dict() for coin in coins],
            "15s": [fifteen_s_metrics[coin].to_dict() for coin in coins],
        },
        "validation_reports": validation_reports,
        "phase_audit_available": phase_audit is not None,
        "phase_audit_generated_at": phase_audit.get("generated_at") if phase_audit else None,
        "candidate_freshness": candidate_summary,
    }


def write_health_json(report: dict[str, Any], output_dir: Path, json_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / json_name
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def write_health_markdown(report: dict[str, Any], output_dir: Path, markdown_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / markdown_name

    lines = [
        "# Bid/Ask Data Health Watchdog",
        "",
        f"Generated at: {report['generated_at']}",
        "",
        "## Status",
        "",
        f"- status: **{report['status']}**",
        f"- source: {report['source']}",
        f"- coins: {', '.join(report['coins'])}",
        "",
        "## Safety Flags",
        "",
    ]

    for key, value in report["safety_flags"].items():
        if key == "safety_statement":
            continue
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Issues", ""])
    if report["issues"]:
        for issue in report["issues"]:
            coin_suffix = f" ({issue['coin']})" if issue.get("coin") else ""
            lines.append(f"- [{issue['severity']}] {issue['code']}{coin_suffix}: {issue['message']}")
    else:
        lines.append("- none")

    lines.extend(["", "## Recommended Commands", ""])
    for command in report["recommended_commands"]:
        lines.append(f"- `{command}`")

    lines.extend(["", "## Latest Timestamps (1s)", ""])
    for metrics in report["csv_metrics"]["1s"]:
        latest = metrics.get("latest_timestamp_ms")
        lines.append(f"- {metrics['coin']}: {latest if latest is not None else 'n/a'}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _exit_code_for_status(status: str) -> int:
    if status == STATUS_HEALTHY:
        return 0
    if status == STATUS_WARNING:
        return 1
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only bid/ask data health watchdog (paper/research only).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Run data health checks")
    check_parser.add_argument(
        "--coins",
        nargs="+",
        default=list(DEFAULT_COINS),
        help="Coins to check (default: BTC ETH)",
    )
    check_parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Data source (default: historical)",
    )
    check_parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Bid/ask CSV root (default: data/bid_ask_csv)",
    )
    check_parser.add_argument(
        "--reports-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help="Reports root (default: reports)",
    )
    check_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for health reports (default: reports/data_health)",
    )
    check_parser.add_argument(
        "--max-stale-hours",
        type=float,
        default=DEFAULT_MAX_STALE_HOURS,
        help="Warn when latest 1s timestamp is older than this many hours (default: 12)",
    )
    check_parser.add_argument(
        "--json-name",
        default=DEFAULT_JSON_NAME,
        help="JSON report filename (default: data_health_latest.json)",
    )
    check_parser.add_argument(
        "--markdown-name",
        default=DEFAULT_MARKDOWN_NAME,
        help="Markdown report filename (default: data_health_latest.md)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "check":
        parser.error(f"Unsupported command: {args.command}")

    report = run_data_health_check(
        coins=[coin.upper() for coin in args.coins],
        source=args.source,
        data_root=args.data_root,
        reports_root=args.reports_root,
        max_stale_hours=args.max_stale_hours,
    )

    json_path = write_health_json(report, args.output_dir, args.json_name)
    markdown_path = write_health_markdown(report, args.output_dir, args.markdown_name)

    print(f"status={report['status']}")
    print(f"json_report_path={json_path}")
    print(f"markdown_report_path={markdown_path}")
    print(f"issue_count={len(report['issues'])}")
    for issue in report["issues"]:
        coin_suffix = f" coin={issue['coin']}" if issue.get("coin") else ""
        print(f"issue: [{issue['severity']}] {issue['code']}{coin_suffix} {issue['message']}")

    return _exit_code_for_status(report["status"])


if __name__ == "__main__":
    raise SystemExit(main())
