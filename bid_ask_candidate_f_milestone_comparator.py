"""Read-only milestone comparator for frozen Candidate F walk-forward results.

Compares the fixed Candidate F configuration across successive historical-data
milestones using local walk-forward and robustness report artifacts only.
Dataset bounds are derived read-only from CSV paths declared in walk-forward
reports. Research-only, paper-only, and execution-disabled.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bid_ask_backfill_milestone_tracker import _analyze_csv
from bid_ask_candidate_f_robustness_analyzer import (
    CANDIDATE_F_FEATURE_SET,
    CONFIG_MATCH_FIELDS,
    VERDICT_INSUFFICIENT_TRADES,
)
from bid_ask_resampler import resampled_csv_path
from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT

FROZEN_CONFIG_ID = "ETH_UP_H20_C0.65_E2_M1.25_S8"

FROZEN_CONFIG: dict[str, Any] = {
    "config_id": FROZEN_CONFIG_ID,
    "coin": "ETH",
    "trade_class": "UP",
    "feature_set": CANDIDATE_F_FEATURE_SET,
    "label_mode": "binary_tradeable_up",
    "horizon": 20,
    "interval": "15s",
    "min_confidence": 0.65,
    "min_edge_bps": 2.0,
    "cost_multiplier": 1.25,
    "max_spread_bps": 8.0,
}

DEFAULT_WALK_FORWARD_DIR = Path("reports/bid_ask_walk_forward")
DEFAULT_ROBUSTNESS_DIR = Path("reports/candidate_f_robustness")
DEFAULT_OUTPUT_DIR = Path("reports/candidate_f_milestone_comparison")

LATEST_JSON_NAME = "candidate_f_milestone_comparison_latest.json"
LATEST_MD_NAME = "candidate_f_milestone_comparison_latest.md"
STAMPED_JSON_PREFIX = "candidate_f_milestone_comparison_"
STAMPED_MD_PREFIX = "candidate_f_milestone_comparison_"

SAFETY_NOTE = (
    "Candidate F milestone comparison is read-only and paper-only. "
    "Reads local walk-forward and robustness report artifacts only. Dataset "
    "bounds are derived read-only from CSV paths declared in walk-forward "
    "reports. No orders, execution, wallets, AWS, network, or shell commands."
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
    return payload if isinstance(payload, dict) else None


def _parse_walk_forward_stamp(path: Path) -> str:
    stem = path.stem
    if stem.startswith("walk_forward_"):
        return stem.removeprefix("walk_forward_").split("_", 1)[-1]
    return stem


def _parse_robustness_stamp(path: Path) -> str:
    stem = path.stem
    if stem.startswith("candidate_f_robustness_"):
        return stem.removeprefix("candidate_f_robustness_")
    return stem


def _normalize_trade_classes(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value.upper()}
    try:
        return {str(item).upper() for item in value}
    except TypeError:
        return set()


def _float_equal(left: object, right: object) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


def config_matches_frozen(report: dict[str, Any]) -> bool:
    """Return True when a walk-forward report matches the frozen Candidate F config."""
    if report.get("feature_set") != FROZEN_CONFIG["feature_set"]:
        return False

    coin = str(FROZEN_CONFIG["coin"]).upper()
    coins = {str(item).upper() for item in report.get("coins") or []}
    if coins and coin not in coins:
        return False

    trade_class = str(FROZEN_CONFIG["trade_class"]).upper()
    trade_classes = _normalize_trade_classes(report.get("trade_classes"))
    if trade_classes and trade_class not in trade_classes:
        return False

    for field in CONFIG_MATCH_FIELDS:
        expected = FROZEN_CONFIG.get(field)
        actual = report.get(field)
        if expected is None:
            continue
        if field in {"min_confidence", "min_edge_bps", "cost_multiplier", "max_spread_bps"}:
            if not _float_equal(expected, actual):
                return False
        elif str(expected) != str(actual):
            return False
    return True


def _robustness_matches_frozen(report: dict[str, Any]) -> bool:
    analyses = report.get("config_analyses") or []
    for item in analyses:
        if not isinstance(item, dict):
            continue
        if item.get("config_id") != FROZEN_CONFIG_ID:
            continue
        grid_metrics = item.get("grid_metrics") or {}
        if grid_metrics.get("feature_set") not in {None, CANDIDATE_F_FEATURE_SET}:
            continue
        return True
    summary = report.get("candidate_f_summary") or {}
    if summary.get("feature_set") not in {None, CANDIDATE_F_FEATURE_SET}:
        return False
    return any(
        isinstance(item, dict) and item.get("config_id") == FROZEN_CONFIG_ID
        for item in report.get("positive_configs") or []
    )


def _is_valid_walk_forward_run(report: dict[str, Any]) -> bool:
    if not config_matches_frozen(report):
        return False
    if int(report.get("completed_windows") or 0) <= 0:
        return False
    if report.get("execution_enabled") is True:
        return False
    return True


def _resolve_dataset_csv_path(report: dict[str, Any]) -> Path | None:
    coin = str(FROZEN_CONFIG["coin"]).upper()
    interval = str(report.get("interval") or FROZEN_CONFIG["interval"])
    data_dir = report.get("data_dir")
    if isinstance(data_dir, str) and data_dir.strip():
        return Path(data_dir) / f"{coin}_{interval}.csv"

    data_root = report.get("data_root")
    source = report.get("source") or "historical"
    root = Path(data_root) if isinstance(data_root, str) and data_root.strip() else Path("data/bid_ask_csv")
    return resampled_csv_path(coin, interval, root, source=str(source))


def extract_dataset_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Extract dataset first/latest timestamps and coverage from report metadata."""
    embedded = report.get("dataset_summary")
    if isinstance(embedded, dict) and embedded.get("latest_timestamp_iso"):
        return {
            "first_timestamp_iso": embedded.get("first_timestamp_iso"),
            "latest_timestamp_iso": embedded.get("latest_timestamp_iso"),
            "coverage_hours": float(embedded.get("coverage_hours") or 0.0),
            "csv_path": embedded.get("csv_path"),
            "row_count": embedded.get("row_count"),
        }

    interval = str(report.get("interval") or FROZEN_CONFIG["interval"])
    csv_path = _resolve_dataset_csv_path(report)
    if csv_path is None:
        return {
            "first_timestamp_iso": None,
            "latest_timestamp_iso": None,
            "coverage_hours": 0.0,
            "csv_path": None,
            "row_count": None,
        }

    stats = _analyze_csv(csv_path, interval=interval)
    return {
        "first_timestamp_iso": stats.get("first_timestamp_iso"),
        "latest_timestamp_iso": stats.get("latest_timestamp_iso"),
        "coverage_hours": float(stats.get("coverage_hours") or 0.0),
        "csv_path": str(csv_path),
        "row_count": stats.get("row_count"),
    }


def dataset_group_key(dataset_summary: dict[str, Any]) -> tuple[Any, ...]:
    coverage = dataset_summary.get("coverage_hours")
    rounded_coverage = round(float(coverage or 0.0), 3)
    return (
        dataset_summary.get("first_timestamp_iso"),
        dataset_summary.get("latest_timestamp_iso"),
        rounded_coverage,
    )


def discover_walk_forward_reports(
    walk_forward_dir: Path = DEFAULT_WALK_FORWARD_DIR,
) -> list[tuple[str, Path, dict[str, Any]]]:
    """Discover walk-forward JSON reports matching the frozen Candidate F config."""
    if not walk_forward_dir.is_dir():
        return []

    coin = str(FROZEN_CONFIG["coin"]).upper()
    discovered: list[tuple[str, Path, dict[str, Any]]] = []
    for path in walk_forward_dir.glob(f"walk_forward_{coin}_*.json"):
        payload = _load_json(path)
        if payload is None or not config_matches_frozen(payload):
            continue
        discovered.append((_parse_walk_forward_stamp(path), path, payload))

    discovered.sort(key=lambda item: item[0])
    return discovered


def discover_robustness_reports(
    robustness_dir: Path = DEFAULT_ROBUSTNESS_DIR,
) -> list[tuple[str, Path, dict[str, Any]]]:
    """Discover robustness JSON reports for the frozen Candidate F config."""
    if not robustness_dir.is_dir():
        return []

    discovered: list[tuple[str, Path, dict[str, Any]]] = []
    for path in robustness_dir.glob("candidate_f_robustness_*.json"):
        if path.name == "candidate_f_robustness_latest.json":
            continue
        payload = _load_json(path)
        if payload is None or not _robustness_matches_frozen(payload):
            continue
        discovered.append((_parse_robustness_stamp(path), path, payload))

    discovered.sort(key=lambda item: item[0])
    return discovered


def _frozen_config_analysis(robustness_report: dict[str, Any]) -> dict[str, Any] | None:
    for item in robustness_report.get("config_analyses") or []:
        if not isinstance(item, dict):
            continue
        if item.get("config_id") != FROZEN_CONFIG_ID:
            continue
        grid_metrics = item.get("grid_metrics") or {}
        if grid_metrics.get("feature_set") not in {None, CANDIDATE_F_FEATURE_SET}:
            continue
        return item
    return None


def find_matching_robustness_report(
    walk_forward_report: dict[str, Any],
    robustness_reports: list[tuple[str, Path, dict[str, Any]]],
) -> tuple[str, Path, dict[str, Any], dict[str, Any]] | None:
    """Find the robustness report whose accepted-trades path matches walk-forward."""
    accepted_path = walk_forward_report.get("accepted_trades_csv_path")
    normalized_accepted = str(accepted_path).strip() if accepted_path else ""

    exact_matches: list[tuple[str, Path, dict[str, Any], dict[str, Any]]] = []
    for stamp, path, payload in robustness_reports:
        analysis = _frozen_config_analysis(payload)
        if analysis is None:
            continue
        analysis_csv = str(analysis.get("accepted_trades_csv_path") or "").strip()
        if normalized_accepted and analysis_csv and analysis_csv == normalized_accepted:
            exact_matches.append((stamp, path, payload, analysis))

    if exact_matches:
        exact_matches.sort(key=lambda item: item[0], reverse=True)
        return exact_matches[0]

    return None


def _dominant_day_metrics(trade_analysis: dict[str, Any] | None) -> dict[str, Any]:
    if not trade_analysis:
        return {
            "dominant_day": None,
            "dominant_day_trade_fraction": 0.0,
            "excluding_dominant_day": None,
        }

    by_day = trade_analysis.get("by_day") or {}
    trade_count = int(trade_analysis.get("trade_count") or 0)
    if not by_day or trade_count <= 0:
        return {
            "dominant_day": None,
            "dominant_day_trade_fraction": 0.0,
            "excluding_dominant_day": None,
        }

    dominant_day = max(by_day, key=lambda day: int((by_day[day] or {}).get("count") or 0))
    dominant_count = int((by_day[dominant_day] or {}).get("count") or 0)
    dominant_fraction = dominant_count / trade_count if trade_count else 0.0

    remaining_count = 0
    remaining_sum = 0.0
    for day, metrics in by_day.items():
        if day == dominant_day:
            continue
        count = int(metrics.get("count") or 0)
        total = float(metrics.get("sum_net_bps") or metrics.get("cumulative_net_bps") or 0.0)
        remaining_count += count
        remaining_sum += total

    if remaining_count <= 0:
        excluding = {
            "trade_count": 0,
            "average_net_bps": 0.0,
            "cumulative_net_bps": 0.0,
            "win_rate": None,
        }
    else:
        excluding = {
            "trade_count": remaining_count,
            "average_net_bps": remaining_sum / remaining_count,
            "cumulative_net_bps": remaining_sum,
            "win_rate": None,
        }

    return {
        "dominant_day": dominant_day,
        "dominant_day_trade_fraction": dominant_fraction,
        "excluding_dominant_day": excluding,
    }


def _regime_summary(trade_analysis: dict[str, Any] | None) -> dict[str, Any]:
    if not trade_analysis:
        return {
            "volatility_regime_summary": {},
            "spread_regime_summary": {},
            "dominant_volatility_regime": None,
            "dominant_volatility_regime_trade_fraction": 0.0,
            "dominant_spread_regime": None,
            "dominant_spread_regime_trade_fraction": 0.0,
        }

    by_volatility = trade_analysis.get("by_volatility_regime") or {}
    by_spread = trade_analysis.get("by_spread_regime") or {}
    return {
        "volatility_regime_summary": by_volatility,
        "spread_regime_summary": by_spread,
        "dominant_volatility_regime": trade_analysis.get("dominant_volatility_regime"),
        "dominant_volatility_regime_trade_fraction": float(
            trade_analysis.get("dominant_volatility_regime_trade_fraction") or 0.0
        ),
        "dominant_spread_regime": trade_analysis.get("dominant_spread_regime"),
        "dominant_spread_regime_trade_fraction": float(
            trade_analysis.get("dominant_spread_regime_trade_fraction") or 0.0
        ),
    }


def _compact_regime_summary(summary: dict[str, Any]) -> str:
    if not summary:
        return "n/a"
    parts: list[str] = []
    for regime, metrics in sorted(summary.items()):
        count = int(metrics.get("count") or 0)
        avg = float(metrics.get("avg_net_bps") or 0.0)
        parts.append(f"{regime}:{count}@{avg:.2f}bps")
    return "; ".join(parts) if parts else "n/a"


def select_newest_valid_runs(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate runs on the same dataset, keeping the newest valid run."""
    valid_runs = [run for run in runs if run.get("valid")]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for run in valid_runs:
        key = dataset_group_key(run["dataset_summary"])
        grouped.setdefault(key, []).append(run)

    selected: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for key, items in grouped.items():
        valid_items = list(items)
        pool = valid_items
        pool.sort(key=lambda item: item["report_stamp"], reverse=True)
        chosen = pool[0]
        chosen["dataset_key"] = key
        chosen["duplicate_count"] = max(len(items) - 1, 0)
        selected.append(chosen)
        for item in items:
            if item is chosen:
                continue
            duplicates.append(
                {
                    "report_stamp": item["report_stamp"],
                    "walk_forward_report_path": item["walk_forward_report_path"],
                    "dataset_key": key,
                    "superseded_by_stamp": chosen["report_stamp"],
                }
            )

    selected.sort(
        key=lambda item: (
            item["dataset_summary"].get("latest_timestamp_iso") or "",
            item["dataset_summary"].get("first_timestamp_iso") or "",
            float(item["dataset_summary"].get("coverage_hours") or 0.0),
            item["report_stamp"],
        )
    )
    return selected, duplicates


def build_run_record(
    *,
    report_stamp: str,
    walk_forward_path: Path,
    walk_forward_report: dict[str, Any],
    dataset_summary: dict[str, Any],
    robustness_match: tuple[str, Path, dict[str, Any], dict[str, Any]] | None,
) -> dict[str, Any]:
    trade_analysis = None
    robustness_verdict = VERDICT_INSUFFICIENT_TRADES
    robustness_report_path = None
    robustness_report_stamp = None
    if robustness_match is not None:
        robustness_report_stamp, robustness_path, _, analysis = robustness_match
        robustness_report_path = str(robustness_path)
        trade_analysis = analysis.get("trade_analysis")
        robustness_verdict = str(analysis.get("verdict") or VERDICT_INSUFFICIENT_TRADES)

    accepted_trades_count = int(walk_forward_report.get("accepted_trades_count") or 0)
    if trade_analysis is not None:
        accepted_trades_count = int(trade_analysis.get("trade_count") or accepted_trades_count)

    average_net_bps = float(walk_forward_report.get("average_net_return_bps") or 0.0)
    cumulative_net_bps = float(walk_forward_report.get("cumulative_net_return_bps") or 0.0)
    win_rate = None
    if trade_analysis is not None:
        average_net_bps = float(trade_analysis.get("average_net_bps") or average_net_bps)
        cumulative_net_bps = float(trade_analysis.get("cumulative_net_bps") or cumulative_net_bps)
        win_rate = float(trade_analysis.get("win_rate") or 0.0)
    else:
        up_diag = ((walk_forward_report.get("class_trade_diagnostics") or {}).get("UP") or {})
        if up_diag.get("win_rate_after_costs") is not None:
            win_rate = float(up_diag["win_rate_after_costs"])

    day_metrics = _dominant_day_metrics(trade_analysis)
    regime_metrics = _regime_summary(trade_analysis)

    return {
        "report_stamp": report_stamp,
        "walk_forward_report_path": str(walk_forward_path),
        "robustness_report_path": robustness_report_path,
        "robustness_report_stamp": robustness_report_stamp,
        "dataset_summary": dataset_summary,
        "valid": _is_valid_walk_forward_run(walk_forward_report),
        "accepted_trades_count": accepted_trades_count,
        "average_net_bps": average_net_bps,
        "cumulative_net_bps": cumulative_net_bps,
        "win_rate": win_rate,
        "completed_windows": int(walk_forward_report.get("completed_windows") or 0),
        "skipped_windows": int(walk_forward_report.get("skipped_windows") or 0),
        "dominant_utc_hour": (
            str(trade_analysis.get("dominant_utc_hour"))
            if trade_analysis and trade_analysis.get("dominant_utc_hour") is not None
            else None
        ),
        "dominant_utc_hour_trade_fraction": float(
            trade_analysis.get("dominant_utc_hour_trade_fraction") or 0.0
        )
        if trade_analysis
        else 0.0,
        "dominant_day": day_metrics["dominant_day"],
        "dominant_day_trade_fraction": day_metrics["dominant_day_trade_fraction"],
        "excluding_dominant_day": day_metrics["excluding_dominant_day"],
        "volatility_regime_summary": regime_metrics["volatility_regime_summary"],
        "spread_regime_summary": regime_metrics["spread_regime_summary"],
        "dominant_volatility_regime": regime_metrics["dominant_volatility_regime"],
        "dominant_volatility_regime_trade_fraction": regime_metrics[
            "dominant_volatility_regime_trade_fraction"
        ],
        "dominant_spread_regime": regime_metrics["dominant_spread_regime"],
        "dominant_spread_regime_trade_fraction": regime_metrics[
            "dominant_spread_regime_trade_fraction"
        ],
        "robustness_verdict": robustness_verdict,
    }


def build_milestone_comparison_table(milestones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a chronological comparison table from deduplicated milestone runs."""
    table: list[dict[str, Any]] = []
    for index, milestone in enumerate(milestones, start=1):
        dataset = milestone["dataset_summary"]
        excluding = milestone.get("excluding_dominant_day") or {}
        table.append(
            {
                "milestone_index": index,
                "dataset_first_timestamp_iso": dataset.get("first_timestamp_iso"),
                "dataset_latest_timestamp_iso": dataset.get("latest_timestamp_iso"),
                "dataset_coverage_hours": float(dataset.get("coverage_hours") or 0.0),
                "report_stamp": milestone["report_stamp"],
                "accepted_trades_count": milestone["accepted_trades_count"],
                "average_net_bps": milestone["average_net_bps"],
                "cumulative_net_bps": milestone["cumulative_net_bps"],
                "win_rate": milestone["win_rate"],
                "completed_windows": milestone["completed_windows"],
                "skipped_windows": milestone["skipped_windows"],
                "dominant_utc_hour": milestone["dominant_utc_hour"],
                "dominant_utc_hour_trade_fraction": milestone["dominant_utc_hour_trade_fraction"],
                "dominant_day": milestone["dominant_day"],
                "dominant_day_trade_fraction": milestone["dominant_day_trade_fraction"],
                "excluding_dominant_day_average_net_bps": excluding.get("average_net_bps"),
                "excluding_dominant_day_cumulative_net_bps": excluding.get("cumulative_net_bps"),
                "excluding_dominant_day_trade_count": excluding.get("trade_count"),
                "excluding_dominant_day_win_rate": excluding.get("win_rate"),
                "volatility_regime_summary": milestone["volatility_regime_summary"],
                "spread_regime_summary": milestone["spread_regime_summary"],
                "robustness_verdict": milestone["robustness_verdict"],
                "walk_forward_report_path": milestone["walk_forward_report_path"],
                "robustness_report_path": milestone["robustness_report_path"],
            }
        )
    return table


def build_milestone_comparison_report(
    *,
    milestones: list[dict[str, Any]],
    comparison_table: list[dict[str, Any]],
    duplicates: list[dict[str, Any]],
    discovered_walk_forward_count: int,
    discovered_robustness_count: int,
    generated_at: str | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at or _utc_now_iso(),
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "real_execution_ready": False,
        "frozen_config_id": FROZEN_CONFIG_ID,
        "frozen_config": dict(FROZEN_CONFIG),
        "feature_set": CANDIDATE_F_FEATURE_SET,
        "discovered_walk_forward_count": discovered_walk_forward_count,
        "discovered_robustness_count": discovered_robustness_count,
        "milestone_count": len(milestones),
        "duplicate_runs_superseded": duplicates,
        "milestones": milestones,
        "comparison_table": comparison_table,
        "safety_note": SAFETY_NOTE,
    }


def write_comparison_json(
    report: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    json_name: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = json_name or LATEST_JSON_NAME
    output_path = output_dir / filename
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["json_report_path"] = str(output_path)
    return output_path


def _format_bps(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def write_comparison_markdown(
    report: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    markdown_name: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = markdown_name or LATEST_MD_NAME
    output_path = output_dir / filename

    lines = [
        "# Candidate F Milestone Comparison",
        "",
        f"Generated at: {report.get('generated_at', 'n/a')}",
        f"Frozen config: {report.get('frozen_config_id', FROZEN_CONFIG_ID)}",
        f"Feature set: {report.get('feature_set', CANDIDATE_F_FEATURE_SET)}",
        "",
        "## Safety statement",
        "",
        f"- paper_only: {str(report.get('paper_only', True)).lower()}",
        f"- execution_enabled: {str(report.get('execution_enabled', False)).lower()}",
        f"- real_execution_ready: {str(report.get('real_execution_ready', False)).lower()}",
        "",
        SAFETY_NOTE,
        "",
        "## Discovery summary",
        "",
        f"- discovered_walk_forward_count: {report.get('discovered_walk_forward_count', 0)}",
        f"- discovered_robustness_count: {report.get('discovered_robustness_count', 0)}",
        f"- milestone_count: {report.get('milestone_count', 0)}",
        f"- duplicate_runs_superseded: {len(report.get('duplicate_runs_superseded') or [])}",
        "",
        "## Chronological comparison table",
        "",
        "| Milestone | Dataset first | Dataset latest | Coverage h | Trades | Avg bps | Cum bps | Win rate | Completed | Skipped | Dom hour (frac) | Dom day (frac) | Ex-dom avg bps | Vol regimes | Spread regimes | Verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | --- |",
    ]

    for row in report.get("comparison_table") or []:
        dom_hour = row.get("dominant_utc_hour")
        dom_hour_frac = _format_pct(float(row.get("dominant_utc_hour_trade_fraction") or 0.0))
        dom_day = row.get("dominant_day") or "n/a"
        dom_day_frac = _format_pct(float(row.get("dominant_day_trade_fraction") or 0.0))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("milestone_index", "")),
                    str(row.get("dataset_first_timestamp_iso") or "n/a"),
                    str(row.get("dataset_latest_timestamp_iso") or "n/a"),
                    str(row.get("dataset_coverage_hours") or 0.0),
                    str(row.get("accepted_trades_count") or 0),
                    _format_bps(float(row.get("average_net_bps") or 0.0)),
                    _format_bps(float(row.get("cumulative_net_bps") or 0.0)),
                    _format_pct(float(row.get("win_rate") or 0.0))
                    if row.get("win_rate") is not None
                    else "n/a",
                    str(row.get("completed_windows") or 0),
                    str(row.get("skipped_windows") or 0),
                    f"{dom_hour or 'n/a'} ({dom_hour_frac})",
                    f"{dom_day} ({dom_day_frac})",
                    _format_bps(
                        float(row.get("excluding_dominant_day_average_net_bps") or 0.0)
                    )
                    if row.get("excluding_dominant_day_average_net_bps") is not None
                    else "n/a",
                    _compact_regime_summary(row.get("volatility_regime_summary") or {}),
                    _compact_regime_summary(row.get("spread_regime_summary") or {}),
                    str(row.get("robustness_verdict") or "n/a"),
                ]
            )
            + " |"
        )

    duplicates = report.get("duplicate_runs_superseded") or []
    lines.extend(["", "## Superseded duplicate runs", ""])
    if not duplicates:
        lines.append("No duplicate runs were superseded.")
    else:
        for item in duplicates:
            lines.append(
                f"- {item.get('report_stamp')} superseded by {item.get('superseded_by_stamp')} "
                f"({item.get('walk_forward_report_path')})"
            )

    lines.extend(["", SAFETY_STATEMENT, ""])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["markdown_report_path"] = str(output_path)
    return output_path


def run_milestone_comparison(
    *,
    walk_forward_dir: Path = DEFAULT_WALK_FORWARD_DIR,
    robustness_dir: Path = DEFAULT_ROBUSTNESS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_stamped: bool = True,
) -> dict[str, Any]:
    """Discover, deduplicate, compare, and write Candidate F milestone reports."""
    walk_forward_reports = discover_walk_forward_reports(walk_forward_dir=walk_forward_dir)
    robustness_reports = discover_robustness_reports(robustness_dir=robustness_dir)

    run_records: list[dict[str, Any]] = []
    for stamp, path, payload in walk_forward_reports:
        dataset_summary = extract_dataset_summary(payload)
        robustness_match = find_matching_robustness_report(payload, robustness_reports)
        run_records.append(
            build_run_record(
                report_stamp=stamp,
                walk_forward_path=path,
                walk_forward_report=payload,
                dataset_summary=dataset_summary,
                robustness_match=robustness_match,
            )
        )

    milestones, duplicates = select_newest_valid_runs(run_records)
    comparison_table = build_milestone_comparison_table(milestones)
    report = build_milestone_comparison_report(
        milestones=milestones,
        comparison_table=comparison_table,
        duplicates=duplicates,
        discovered_walk_forward_count=len(walk_forward_reports),
        discovered_robustness_count=len(robustness_reports),
    )

    write_comparison_json(report, output_dir=output_dir, json_name=LATEST_JSON_NAME)
    write_comparison_markdown(report, output_dir=output_dir, markdown_name=LATEST_MD_NAME)
    latest_json_path = report["json_report_path"]
    latest_markdown_path = report["markdown_report_path"]

    if write_stamped:
        stamp = _utc_stamp()
        stamped_report = dict(report)
        write_comparison_json(
            stamped_report,
            output_dir=output_dir,
            json_name=f"{STAMPED_JSON_PREFIX}{stamp}.json",
        )
        write_comparison_markdown(
            stamped_report,
            output_dir=output_dir,
            markdown_name=f"{STAMPED_MD_PREFIX}{stamp}.md",
        )

    report["json_report_path"] = latest_json_path
    report["markdown_report_path"] = latest_markdown_path
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen Candidate F walk-forward milestones from local report "
            "artifacts (research-only, no execution)."
        )
    )
    parser.add_argument(
        "--walk-forward-dir",
        type=Path,
        default=DEFAULT_WALK_FORWARD_DIR,
        help=f"Walk-forward reports directory (default: {DEFAULT_WALK_FORWARD_DIR})",
    )
    parser.add_argument(
        "--robustness-dir",
        type=Path,
        default=DEFAULT_ROBUSTNESS_DIR,
        help=f"Candidate F robustness reports directory (default: {DEFAULT_ROBUSTNESS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-stamped",
        action="store_true",
        help="Skip writing timestamped JSON/Markdown copies",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    report = run_milestone_comparison(
        walk_forward_dir=args.walk_forward_dir,
        robustness_dir=args.robustness_dir,
        output_dir=args.output_dir,
        write_stamped=not args.no_stamped,
    )

    print(f"frozen_config_id={report.get('frozen_config_id')}")
    print(f"milestone_count={report.get('milestone_count')}")
    print(f"json_report_path={report.get('json_report_path')}")
    print(f"markdown_report_path={report.get('markdown_report_path')}")
    print(SAFETY_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
