"""Standalone candidate regression ledger for paper-only research reports.

Aggregates candidate validation, walk-forward, and robustness JSON reports
into CSV/JSON/Markdown for progress tracking over time.

Research-only and paper-only. No orders, wallets, or network calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT

SAFETY_NOTE = (
    "Candidate ledger is research-only and paper-only. "
    "Aggregated metrics do not prove profitability. No real orders are placed."
)

DEFAULT_REPORTS_ROOT = Path("reports")
DEFAULT_OUTPUT_DIR = Path("reports/candidate_ledger")
DEFAULT_CSV_NAME = "candidate_ledger.csv"
DEFAULT_JSON_NAME = "candidate_ledger.json"
DEFAULT_MARKDOWN_NAME = "candidate_ledger.md"

STRICT_CANDIDATE_B_NAME = "BTC_DOWN_MICRO_H20_CANDIDATE_B"
RELAXED_CANDIDATE_B_NAME = "BTC_DOWN_MICRO_H20_CANDIDATE_B_RELAXED"

CANDIDATE_VALIDATION_DIR = "bid_ask_candidate_validation"
WALK_FORWARD_DIR = "bid_ask_walk_forward"

CANDIDATE_VALIDATION_GLOB = "candidate_validation_*.json"
WALK_FORWARD_GLOB = "walk_forward_*.json"
ROBUSTNESS_GLOB = "robustness_*.json"

REPORT_TYPE_CANDIDATE_VALIDATION = "candidate_validation"
REPORT_TYPE_WALK_FORWARD = "walk_forward"
REPORT_TYPE_ROBUSTNESS = "robustness"

LEDGER_COLUMNS = (
    "source_file",
    "report_type",
    "run_timestamp",
    "candidate_name",
    "coin",
    "interval",
    "horizon",
    "feature_set",
    "label_mode",
    "trade_classes",
    "min_confidence",
    "min_edge_bps",
    "cost_multiplier",
    "max_spread_bps",
    "filtered_paper_trades",
    "cumulative_net_return_bps",
    "average_net_return_bps",
    "win_rate_after_costs",
    "accuracy_edge_vs_baseline",
    "positive_after_costs",
    "beats_baseline",
    "paper_candidate_pass",
    "robustness_pass",
    "status",
    "warnings_count",
    "real_execution_ready",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _format_trade_classes(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _config_fields(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config") or {}
    trade_filter = payload.get("trade_filter_summary") or {}
    return {
        "candidate_name": _first_present(payload, "candidate_name") or config.get("candidate_name"),
        "coin": _first_present(payload, "coin") or config.get("coin"),
        "interval": _first_present(payload, "interval") or config.get("interval"),
        "horizon": _first_present(payload, "horizon") or config.get("horizon"),
        "feature_set": _first_present(payload, "feature_set") or config.get("feature_set"),
        "label_mode": _first_present(payload, "label_mode") or config.get("label_mode"),
        "trade_classes": _first_present(payload, "trade_classes")
        or config.get("trade_classes")
        or trade_filter.get("trade_classes"),
        "min_confidence": _first_present(payload, "min_confidence")
        or config.get("min_confidence")
        or trade_filter.get("min_confidence"),
        "min_edge_bps": _first_present(payload, "min_edge_bps")
        or config.get("min_edge_bps")
        or trade_filter.get("min_edge_bps"),
        "cost_multiplier": _first_present(payload, "cost_multiplier")
        or config.get("cost_multiplier")
        or trade_filter.get("cost_multiplier"),
        "max_spread_bps": _first_present(payload, "max_spread_bps")
        or config.get("max_spread_bps")
        or trade_filter.get("max_spread_bps"),
    }


def _resolve_coin(payload: dict[str, Any], config_fields: dict[str, Any]) -> str:
    coin = config_fields.get("coin")
    if coin:
        return str(coin).upper()
    coins = payload.get("coins")
    if isinstance(coins, list) and coins:
        return ",".join(str(item).upper() for item in coins)
    per_coin = payload.get("per_coin") or {}
    if isinstance(per_coin, dict) and per_coin:
        return ",".join(sorted(str(item).upper() for item in per_coin))
    return ""


def _extract_run_timestamp(payload: dict[str, Any]) -> str:
    return str(
        _first_present(payload, "run_timestamp", "generated_at", "timestamp") or ""
    )


def _count_warnings(payload: dict[str, Any]) -> int:
    warnings = payload.get("warnings")
    if isinstance(warnings, list):
        return len(warnings)
    stability_warnings = payload.get("stability_warnings")
    if isinstance(stability_warnings, list):
        return len(stability_warnings)
    return 0


def _derive_candidate_validation_status(payload: dict[str, Any]) -> str:
    validation_status = payload.get("validation_status") or {}
    if validation_status.get("paper_candidate_pass"):
        return "paper_candidate_pass"
    return "not_market_ready"


def _derive_walk_forward_status(payload: dict[str, Any]) -> str:
    if payload.get("ready_for_more_research"):
        return "ready_for_more_research"
    readiness = payload.get("readiness") or {}
    if readiness.get("ready_for_more_research"):
        return "ready_for_more_research"

    per_coin = payload.get("per_coin") or {}
    if isinstance(per_coin, dict) and per_coin:
        statuses = sorted(
            {
                str(summary.get("status"))
                for summary in per_coin.values()
                if isinstance(summary, dict) and summary.get("status")
            }
        )
        if statuses:
            return ",".join(statuses)

    beats_baseline = bool(
        payload.get(
            "beats_baseline_accuracy",
            readiness.get("beats_baseline_accuracy", False),
        )
    )
    positive_after_costs = bool(
        payload.get(
            "positive_after_costs",
            readiness.get("positive_after_costs", False),
        )
    )
    if beats_baseline and positive_after_costs:
        return "positive_beats_baseline"
    if beats_baseline:
        return "beats_baseline_only"
    return "not_ready"


def _derive_robustness_status(payload: dict[str, Any]) -> str:
    if payload.get("robustness_pass"):
        return "robustness_pass"
    return "robustness_fail"


def _base_row(
    *,
    source_file: str,
    report_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    config_fields = _config_fields(payload)
    readiness = payload.get("readiness") or {}
    validation_status = payload.get("validation_status") or {}

    beats_baseline = _first_present(payload, "beats_baseline", "beats_baseline_accuracy")
    if beats_baseline is None:
        beats_baseline = readiness.get("beats_baseline_accuracy")

    positive_after_costs = _first_present(payload, "positive_after_costs")
    if positive_after_costs is None:
        positive_after_costs = readiness.get("positive_after_costs")

    filtered_paper_trades = _first_present(
        payload,
        "filtered_paper_trades",
        "accepted_trades_count",
        "total_paper_trades",
    )

    return {
        "source_file": source_file,
        "report_type": report_type,
        "run_timestamp": _extract_run_timestamp(payload),
        "candidate_name": config_fields.get("candidate_name") or "",
        "coin": _resolve_coin(payload, config_fields),
        "interval": config_fields.get("interval") or "",
        "horizon": _optional_int(config_fields.get("horizon")),
        "feature_set": config_fields.get("feature_set") or "",
        "label_mode": config_fields.get("label_mode") or "",
        "trade_classes": _format_trade_classes(config_fields.get("trade_classes")),
        "min_confidence": _optional_float(config_fields.get("min_confidence")),
        "min_edge_bps": _optional_float(config_fields.get("min_edge_bps")),
        "cost_multiplier": _optional_float(config_fields.get("cost_multiplier")),
        "max_spread_bps": _optional_float(config_fields.get("max_spread_bps")),
        "filtered_paper_trades": _optional_int(filtered_paper_trades),
        "cumulative_net_return_bps": _optional_float(payload.get("cumulative_net_return_bps")),
        "average_net_return_bps": _optional_float(payload.get("average_net_return_bps")),
        "win_rate_after_costs": _optional_float(payload.get("win_rate_after_costs")),
        "accuracy_edge_vs_baseline": _optional_float(payload.get("accuracy_edge_vs_baseline")),
        "positive_after_costs": _optional_bool(positive_after_costs),
        "beats_baseline": _optional_bool(beats_baseline),
        "paper_candidate_pass": _optional_bool(validation_status.get("paper_candidate_pass")),
        "robustness_pass": None,
        "status": "",
        "warnings_count": _count_warnings(payload),
        "real_execution_ready": False,
    }


def parse_candidate_validation_row(source_file: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize one candidate validation JSON report into a ledger row."""
    row = _base_row(
        source_file=source_file,
        report_type=REPORT_TYPE_CANDIDATE_VALIDATION,
        payload=payload,
    )
    row["status"] = _derive_candidate_validation_status(payload)
    row["robustness_pass"] = None
    return row


def parse_walk_forward_row(source_file: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize one walk-forward JSON report into a ledger row."""
    row = _base_row(
        source_file=source_file,
        report_type=REPORT_TYPE_WALK_FORWARD,
        payload=payload,
    )
    row["paper_candidate_pass"] = None
    row["robustness_pass"] = None
    row["status"] = _derive_walk_forward_status(payload)
    return row


def parse_robustness_row(source_file: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize one robustness JSON report into a ledger row."""
    row = _base_row(
        source_file=source_file,
        report_type=REPORT_TYPE_ROBUSTNESS,
        payload=payload,
    )
    row["paper_candidate_pass"] = None
    row["robustness_pass"] = _optional_bool(payload.get("robustness_pass"))
    row["status"] = _derive_robustness_status(payload)
    return row


def _load_json_report(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def scan_report_paths(reports_root: Path = DEFAULT_REPORTS_ROOT) -> list[tuple[str, Path]]:
    """Return report paths grouped by parser type."""
    patterns = (
        (REPORT_TYPE_CANDIDATE_VALIDATION, reports_root / CANDIDATE_VALIDATION_DIR / CANDIDATE_VALIDATION_GLOB),
        (REPORT_TYPE_WALK_FORWARD, reports_root / WALK_FORWARD_DIR / WALK_FORWARD_GLOB),
        (REPORT_TYPE_ROBUSTNESS, reports_root / CANDIDATE_VALIDATION_DIR / ROBUSTNESS_GLOB),
    )
    discovered: list[tuple[str, Path]] = []
    for report_type, pattern in patterns:
        for path in sorted(pattern.parent.glob(pattern.name)):
            if path.is_file():
                discovered.append((report_type, path))
    return discovered


def build_ledger_rows(reports_root: Path = DEFAULT_REPORTS_ROOT) -> list[dict[str, Any]]:
    """Scan report directories and return normalized ledger rows sorted by timestamp."""
    parsers = {
        REPORT_TYPE_CANDIDATE_VALIDATION: parse_candidate_validation_row,
        REPORT_TYPE_WALK_FORWARD: parse_walk_forward_row,
        REPORT_TYPE_ROBUSTNESS: parse_robustness_row,
    }
    rows: list[dict[str, Any]] = []
    for report_type, path in scan_report_paths(reports_root):
        payload = _load_json_report(path)
        if payload is None:
            continue
        parser = parsers[report_type]
        rows.append(parser(str(path), payload))

    rows.sort(key=lambda item: (item.get("run_timestamp") or "", item.get("source_file") or ""))
    return rows


def _row_sort_timestamp(row: dict[str, Any]) -> str:
    return str(row.get("run_timestamp") or "")


def _latest_row_for_candidate(
    rows: list[dict[str, Any]],
    candidate_name: str,
) -> dict[str, Any] | None:
    matches = [
        row
        for row in rows
        if row.get("candidate_name") == candidate_name
        and row.get("report_type") == REPORT_TYPE_CANDIDATE_VALIDATION
    ]
    if not matches:
        return None
    return max(matches, key=_row_sort_timestamp)


def build_ledger_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build summary highlights for markdown/JSON output."""
    performance_rows = [
        row
        for row in rows
        if row.get("cumulative_net_return_bps") is not None
        or row.get("filtered_paper_trades") is not None
    ]

    best_by_cumulative: dict[str, Any] | None = None
    if performance_rows:
        best_by_cumulative = max(
            performance_rows,
            key=lambda item: (
                float(item.get("cumulative_net_return_bps") or 0.0),
                _row_sort_timestamp(item),
            ),
        )

    trade_rows = [row for row in rows if row.get("filtered_paper_trades") is not None]
    best_by_trade_count: dict[str, Any] | None = None
    if trade_rows:
        best_by_trade_count = max(
            trade_rows,
            key=lambda item: (
                int(item.get("filtered_paper_trades") or 0),
                _row_sort_timestamp(item),
            ),
        )

    status_distribution = dict(Counter(str(row.get("status") or "unknown") for row in rows))

    return {
        "row_count": len(rows),
        "best_by_cumulative_net_return_bps": best_by_cumulative,
        "best_by_trade_count": best_by_trade_count,
        "latest_strict_candidate_b": _latest_row_for_candidate(rows, STRICT_CANDIDATE_B_NAME),
        "latest_relaxed_candidate_b": _latest_row_for_candidate(rows, RELAXED_CANDIDATE_B_NAME),
        "status_distribution": status_distribution,
    }


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_ledger_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Write normalized ledger rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in LEDGER_COLUMNS})
    return output_path


def write_ledger_json(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
) -> Path:
    """Write ledger rows and summary metadata to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": _utc_now_iso(),
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "real_execution_ready": False,
        "safety_note": SAFETY_NOTE,
        "safety_statement": SAFETY_STATEMENT,
        "row_count": len(rows),
        "summary": summary,
        "rows": rows,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_path


def _format_summary_row(row: dict[str, Any] | None) -> list[str]:
    if row is None:
        return ["- none"]
    return [
        f"- source_file: {row.get('source_file')}",
        f"- report_type: {row.get('report_type')}",
        f"- run_timestamp: {row.get('run_timestamp')}",
        f"- candidate_name: {row.get('candidate_name')}",
        f"- coin: {row.get('coin')}",
        f"- filtered_paper_trades: {row.get('filtered_paper_trades')}",
        f"- cumulative_net_return_bps: {row.get('cumulative_net_return_bps')}",
        f"- status: {row.get('status')}",
    ]


def write_ledger_markdown(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
) -> Path:
    """Write a human-readable ledger summary to Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Candidate Regression Ledger",
        "",
        f"Generated at: {_utc_now_iso()}",
        "",
        "## Safety",
        "",
        f"- paper_only: {str(PAPER_ONLY).lower()}",
        f"- execution_enabled: {str(EXECUTION_ENABLED).lower()}",
        f"- real_execution_ready: false",
        "",
        f"Total rows: {summary.get('row_count', len(rows))}",
        "",
        "## Best by cumulative net return (bps)",
        "",
        *_format_summary_row(summary.get("best_by_cumulative_net_return_bps")),
        "",
        "## Best by trade count",
        "",
        *_format_summary_row(summary.get("best_by_trade_count")),
        "",
        "## Latest strict Candidate B",
        "",
        *_format_summary_row(summary.get("latest_strict_candidate_b")),
        "",
        "## Latest relaxed Candidate B",
        "",
        *_format_summary_row(summary.get("latest_relaxed_candidate_b")),
        "",
        "## Status distribution",
        "",
    ]

    status_distribution = summary.get("status_distribution") or {}
    if not status_distribution:
        lines.append("- none")
    else:
        for status, count in sorted(status_distribution.items()):
            lines.append(f"- {status}: {count}")

    lines.extend(
        [
            "",
            "## Recent rows",
            "",
            "| run_timestamp | report_type | candidate_name | coin | trades | cumulative_net_bps | status |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in rows[-10:]:
        lines.append(
            f"| {row.get('run_timestamp')} | {row.get('report_type')} | "
            f"{row.get('candidate_name')} | {row.get('coin')} | "
            f"{row.get('filtered_paper_trades', '')} | "
            f"{row.get('cumulative_net_return_bps', '')} | {row.get('status')} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _validate_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    if "candidate_ledger" not in resolved.parts:
        raise ValueError(
            f"output_dir must write under a candidate_ledger directory, got {output_dir}"
        )
    return resolved


def update_ledger(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    csv_name: str = DEFAULT_CSV_NAME,
    json_name: str = DEFAULT_JSON_NAME,
    markdown_name: str = DEFAULT_MARKDOWN_NAME,
) -> dict[str, Any]:
    """Scan reports and write CSV, JSON, and Markdown ledger outputs."""
    validated_output_dir = _validate_output_dir(output_dir)
    rows = build_ledger_rows(reports_root=reports_root)
    summary = build_ledger_summary(rows)

    csv_path = write_ledger_csv(rows, validated_output_dir / csv_name)
    json_path = write_ledger_json(rows, summary, validated_output_dir / json_name)
    markdown_path = write_ledger_markdown(rows, summary, validated_output_dir / markdown_name)

    return {
        "row_count": len(rows),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "summary": summary,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate candidate and walk-forward reports into a regression ledger.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    update_parser = subparsers.add_parser(
        "update",
        help="Scan report JSON files and refresh ledger CSV/JSON/Markdown outputs.",
    )
    update_parser.add_argument(
        "--reports-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help=f"Root reports directory (default: {DEFAULT_REPORTS_ROOT})",
    )
    update_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Ledger output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    update_parser.add_argument(
        "--csv-name",
        default=DEFAULT_CSV_NAME,
        help=f"CSV filename (default: {DEFAULT_CSV_NAME})",
    )
    update_parser.add_argument(
        "--json-name",
        default=DEFAULT_JSON_NAME,
        help=f"JSON filename (default: {DEFAULT_JSON_NAME})",
    )
    update_parser.add_argument(
        "--markdown-name",
        default=DEFAULT_MARKDOWN_NAME,
        help=f"Markdown filename (default: {DEFAULT_MARKDOWN_NAME})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "update":
        result = update_ledger(
            reports_root=args.reports_root,
            output_dir=args.output_dir,
            csv_name=args.csv_name,
            json_name=args.json_name,
            markdown_name=args.markdown_name,
        )
        print(
            f"Updated candidate ledger: rows={result['row_count']} "
            f"csv={result['csv_path']} json={result['json_path']} "
            f"markdown={result['markdown_path']}"
        )
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
