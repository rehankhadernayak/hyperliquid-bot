"""Standalone research status and decision log builder for bid/ask ML research.

Paper-only and research-only. Reads local JSON/CSV/Markdown artifacts and writes
a consolidated status report under reports/research_status/. No execution, orders,
wallets, AWS, network, or shell commands.
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
DEFAULT_OUTPUT_DIR = Path("reports/research_status")

DEFAULT_LATEST_JSON_NAME = "research_status_latest.json"
DEFAULT_LATEST_MD_NAME = "research_status_latest.md"
STAMPED_JSON_PREFIX = "research_status_"
STAMPED_MD_PREFIX = "research_status_"

RESEARCH_COINS = ("BTC", "ETH")
INTERVALS = ("1s", "15s", "1m")
HISTORICAL_SOURCE = "historical"
LOCAL_SOURCE = "local"

MIN_ROBUST_TRADES = 10

CANDIDATE_LEDGER_REL = Path("candidate_ledger/candidate_ledger.json")
CANDIDATE_GRID_REL = Path("candidate_grid")
ACCEPTED_TRADES_ANALYSIS_REL = Path("accepted_trades_analysis")
PHASE3_AUDIT_REL = Path("bid_ask_research_scheduler/phase3_audit_latest.json")

RESEARCH_ONLY_NOTE = (
    "Research-only summary. Metrics do not prove profitability. "
    "Real trading remains disabled."
)

REPORT_SECTIONS: tuple[str, ...] = (
    "Safety Statement",
    "Data Status",
    "Candidate Timeline",
    "Latest Candidate Ledger Summary",
    "Latest Accepted-Trades Analysis Summary",
    "Current Blockers",
    "Next Actions",
)

CANDIDATE_TIMELINE: tuple[dict[str, str], ...] = (
    {
        "candidate": "B",
        "name": "BTC_DOWN_MICRO_H20_CANDIDATE_B",
        "status": "rejected",
        "summary": "Rejected after more data — positive early signal did not hold with expanded sample.",
    },
    {
        "candidate": "C",
        "name": "candidate_c_core",
        "status": "rejected/diagnostic",
        "summary": "Feature-engineered candidate_c_core runs remain negative after costs.",
    },
    {
        "candidate": "D",
        "name": "cost_aware_labels",
        "status": "diagnostic/useful but negative",
        "summary": "Cost-aware labels are useful for diagnostics but not predictive enough alone.",
    },
    {
        "candidate": "E",
        "name": "candidate_e_depth_aware",
        "status": "interesting but too sparse",
        "summary": (
            "Depth-aware features restored a sparse positive ETH UP H20 signal, "
            "but configs failed the minimum-trade robustness check."
        ),
    },
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _validate_output_dir(output_dir: Path) -> Path:
    if output_dir.name != "research_status":
        raise ValueError(
            "Output directory must be named research_status "
            f"(got {output_dir})"
        )
    return output_dir


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
    if stem.startswith("candidate_grid_"):
        return stem.removeprefix("candidate_grid_")
    if stem.startswith("accepted_trades_analysis_"):
        parts = stem.split("_")
        if parts:
            return parts[-1]
    return stem


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


def _csv_has_columns(path: Path, columns: tuple[str, ...]) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                return False
            header_set = {cell.strip() for cell in header}
            return all(column in header_set for column in columns)
    except OSError:
        return False


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
                    "missing_periods_count": None,
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


def _resolve_data_csv_path(
    data_root: Path,
    coin: str,
    interval: str,
) -> Path | None:
    historical_path = bid_ask_csv_path(coin, interval, data_root, source=HISTORICAL_SOURCE)
    if historical_path.is_file():
        return historical_path
    local_path = bid_ask_csv_path(coin, interval, data_root, source=LOCAL_SOURCE)
    if local_path.is_file():
        return local_path
    return None


def _build_data_status(
    *,
    data_root: Path,
    phase3: dict[str, Any] | None,
) -> dict[str, Any]:
    imported = (phase3 or {}).get("imported_historical_1s") or {}
    resampled = (phase3 or {}).get("resampled_coverage") or {}
    validation_status = _collect_validation_status(data_root)

    coins: dict[str, Any] = {}
    data_warnings: list[str] = []

    for coin in RESEARCH_COINS:
        coin_import = imported.get(coin) or {}
        coin_resampled = resampled.get(coin) or {}
        intervals: dict[str, Any] = {}

        for interval in INTERVALS:
            csv_path = _resolve_data_csv_path(data_root, coin, interval)
            exists = csv_path is not None

            if interval == "1s":
                row_count = coin_import.get("row_count")
                if row_count is None and csv_path is not None:
                    row_count = _count_csv_rows(csv_path)
                last_ts = coin_import.get("last_timestamp_ms")
                if last_ts is None and csv_path is not None:
                    _, last_ts = _csv_timestamp_bounds(csv_path, interval=interval)
            else:
                interval_data = coin_resampled.get(interval) or {}
                row_count = interval_data.get("row_count")
                if row_count is None and csv_path is not None:
                    row_count = _count_csv_rows(csv_path)
                last_ts = interval_data.get("last_timestamp_ms")
                if last_ts is None and csv_path is not None:
                    _, last_ts = _csv_timestamp_bounds(csv_path, interval=interval)

            has_bid_size = False
            has_ask_size = False
            if csv_path is not None:
                has_bid_size = _csv_has_columns(csv_path, ("bid_size",))
                has_ask_size = _csv_has_columns(csv_path, ("ask_size",))

            validation = validation_status.get(coin, {}).get(interval) or {}
            interval_warnings = list(validation.get("warnings") or [])
            for warning in interval_warnings:
                data_warnings.append(f"{coin} {interval}: {warning}")

            intervals[interval] = {
                "file_exists": exists,
                "csv_path": str(csv_path) if csv_path else None,
                "row_count": row_count,
                "last_timestamp_ms": last_ts,
                "last_timestamp": _fmt_ts_ms(last_ts),
                "has_bid_size": has_bid_size,
                "has_ask_size": has_ask_size,
                "validation": validation,
            }

        coins[coin] = intervals

    deduped_warnings: list[str] = []
    seen: set[str] = set()
    for warning in data_warnings:
        if warning not in seen:
            seen.add(warning)
            deduped_warnings.append(warning)

    return {
        "coins": coins,
        "validation_status": validation_status,
        "warnings": deduped_warnings,
    }


def _load_candidate_ledger(reports_root: Path) -> dict[str, Any]:
    path = reports_root / CANDIDATE_LEDGER_REL
    payload = _load_json(path)
    if payload is None:
        return {
            "available": False,
            "path": str(path),
            "row_count": 0,
            "rows": [],
        }

    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        rows = []

    return {
        "available": True,
        "path": str(path),
        "row_count": payload.get("row_count", len(rows)),
        "generated_at": payload.get("generated_at"),
        "rows": rows,
    }


def _latest_candidate_grid_path(reports_root: Path) -> Path | None:
    grid_dir = reports_root / CANDIDATE_GRID_REL
    if not grid_dir.is_dir():
        return None
    matches = sorted(grid_dir.glob("candidate_grid_*.json"), key=_parse_report_stamp, reverse=True)
    for path in matches:
        if _load_json(path) is not None:
            return path
    return None


def _trade_count_from_row(row: dict[str, Any]) -> int | None:
    for key in (
        "filtered_paper_trades",
        "trades_after_regime_filters",
        "accepted_edge_predictions",
        "accepted_trades_count",
    ):
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _is_sparse_row(row: dict[str, Any]) -> bool:
    trades = _trade_count_from_row(row)
    if trades is None:
        return bool(row.get("too_few_trades") or row.get("insufficient_trades"))
    return trades < MIN_ROBUST_TRADES


def _summarize_positive_after_costs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positives = [row for row in rows if row.get("positive_after_costs") is True]
    positives.sort(
        key=lambda row: (
            float(row.get("cumulative_net_return_bps") or 0.0),
            float(row.get("average_net_return_bps") or 0.0),
        ),
        reverse=True,
    )

    summary: list[dict[str, Any]] = []
    for row in positives[:10]:
        trades = _trade_count_from_row(row)
        sparse = _is_sparse_row(row)
        summary.append(
            {
                "candidate_name": row.get("candidate_name") or row.get("config_id"),
                "coin": row.get("coin"),
                "interval": row.get("interval"),
                "horizon": row.get("horizon"),
                "feature_set": row.get("feature_set"),
                "label_mode": row.get("label_mode"),
                "trade_count": trades,
                "cumulative_net_return_bps": row.get("cumulative_net_return_bps"),
                "average_net_return_bps": row.get("average_net_return_bps"),
                "win_rate_after_costs": row.get("win_rate_after_costs"),
                "sparse_warning": sparse,
                "source_file": row.get("source_file"),
            }
        )
    return summary


def _build_ledger_summary(
    *,
    reports_root: Path,
) -> dict[str, Any]:
    ledger = _load_candidate_ledger(reports_root)
    ledger_rows = ledger.get("rows") or []

    grid_path = _latest_candidate_grid_path(reports_root)
    grid_payload = _load_json(grid_path) if grid_path else None
    grid_results: list[dict[str, Any]] = []
    if grid_payload is not None:
        raw_results = grid_payload.get("results") or []
        if isinstance(raw_results, list):
            grid_results = [row for row in raw_results if isinstance(row, dict)]

    top_by_paper_score = sorted(
        grid_results,
        key=lambda row: float(row.get("paper_score") or 0.0),
        reverse=True,
    )[:5]

    top_paper_score_summary = [
        {
            "rank": row.get("rank"),
            "config_id": row.get("config_id"),
            "coin": row.get("coin"),
            "interval": row.get("interval"),
            "horizon": row.get("horizon"),
            "feature_set": row.get("feature_set"),
            "label_mode": row.get("label_mode"),
            "paper_score": row.get("paper_score"),
            "positive_after_costs": row.get("positive_after_costs"),
            "trade_count": _trade_count_from_row(row),
            "sparse_warning": _is_sparse_row(row),
            "status": row.get("status"),
            "rejection_reason": row.get("rejection_reason"),
        }
        for row in top_by_paper_score
    ]

    positive_from_ledger = _summarize_positive_after_costs(ledger_rows)
    positive_from_grid = _summarize_positive_after_costs(grid_results)

    return {
        "available": ledger.get("available") or grid_payload is not None,
        "ledger_path": ledger.get("path"),
        "ledger_row_count": ledger.get("row_count", 0),
        "ledger_generated_at": ledger.get("generated_at"),
        "latest_grid_path": str(grid_path) if grid_path else None,
        "latest_grid_result_count": len(grid_results),
        "top_by_paper_score": top_paper_score_summary,
        "positive_after_costs_ledger": positive_from_ledger,
        "positive_after_costs_grid": positive_from_grid,
    }


def _latest_accepted_trades_analysis(
    reports_root: Path,
    coin: str,
) -> dict[str, Any] | None:
    analysis_dir = reports_root / ACCEPTED_TRADES_ANALYSIS_REL
    if not analysis_dir.is_dir():
        return None

    prefix = f"accepted_trades_analysis_{coin.upper()}_"
    matches = sorted(
        (path for path in analysis_dir.glob(f"{prefix}*.json") if path.is_file()),
        key=_parse_report_stamp,
        reverse=True,
    )
    for path in matches:
        payload = _load_json(path)
        if payload is None:
            continue
        entry = dict(payload)
        entry["_report_path"] = str(path)
        return entry
    return None


def _build_accepted_trades_summary(reports_root: Path) -> dict[str, Any]:
    by_coin: dict[str, Any] = {}
    for coin in RESEARCH_COINS:
        payload = _latest_accepted_trades_analysis(reports_root, coin)
        if payload is None:
            by_coin[coin] = {"available": False}
            continue

        suspicious = payload.get("suspicious_regimes") or []
        by_coin[coin] = {
            "available": True,
            "report_path": payload.get("_report_path"),
            "source_csv_path": payload.get("source_csv_path"),
            "generated_at": payload.get("generated_at"),
            "trade_count": payload.get("trade_count"),
            "cumulative_net_return_bps": payload.get("cumulative_net_return_bps"),
            "average_net_return_bps": payload.get("average_net_return_bps"),
            "win_rate": payload.get("win_rate"),
            "suspicious_regimes": list(suspicious),
            "suspicious_regime_count": len(suspicious),
        }

    return {
        "available": any(entry.get("available") for entry in by_coin.values()),
        "by_coin": by_coin,
    }


def _derive_blockers(
    *,
    data_status: dict[str, Any],
    ledger_summary: dict[str, Any],
    accepted_trades: dict[str, Any],
    phase3: dict[str, Any] | None,
) -> list[str]:
    blockers: list[str] = []

    sparse_positive = False
    for section_key in ("positive_after_costs_ledger", "positive_after_costs_grid"):
        for row in ledger_summary.get(section_key) or []:
            if row.get("positive_after_costs") is True and row.get("sparse_warning"):
                sparse_positive = True
                break
        if sparse_positive:
            break
    if sparse_positive:
        blockers.append(
            "No robust positive candidates — positive configs have fewer than "
            f"{MIN_ROBUST_TRADES} accepted paper trades."
        )

    for warning in data_status.get("warnings") or []:
        lowered = warning.lower()
        if "missing" in lowered or "incomplete" in lowered:
            blockers.append(f"Data quality: {warning}")

    if phase3 is not None:
        imported = phase3.get("imported_historical_1s") or {}
        for coin in RESEARCH_COINS:
            coin_data = imported.get(coin) or {}
            coverage_hours = coin_data.get("coverage_hours")
            if coverage_hours is not None and float(coverage_hours) < 24 * 7:
                blockers.append(
                    f"{coin} historical archive still short ({coverage_hours} hours) — "
                    "need more out-of-sample data."
                )

    for coin in RESEARCH_COINS:
        analysis = (accepted_trades.get("by_coin") or {}).get(coin) or {}
        if analysis.get("available") and analysis.get("suspicious_regime_count", 0) > 0:
            blockers.append(
                f"{coin} accepted-trade analysis flagged suspicious regimes "
                f"({analysis.get('suspicious_regime_count')})."
            )

    if not blockers:
        blockers.append("No hard blockers detected — continue planned research steps.")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in blockers:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _derive_next_actions(
    *,
    blockers: list[str],
    data_status: dict[str, Any],
    ledger_summary: dict[str, Any],
) -> list[str]:
    actions = [
        "Continue historical backfill for BTC and ETH 1s data.",
        "Rerun depth-aware feature audit after new data lands.",
        "Rerun Candidate E and D experiments only after more out-of-sample data is available.",
        "Avoid further threshold tuning on the current sample — expand data first.",
    ]

    if any("missing" in blocker.lower() or "incomplete" in blocker.lower() for blocker in blockers):
        actions.insert(0, "Resolve missing/incomplete bar warnings before trusting resampled features.")

    if not ledger_summary.get("available"):
        actions.append("Refresh candidate ledger after the next validation or walk-forward batch.")

    if not data_status.get("warnings"):
        actions.append("Keep monitoring validator reports as backfill extends coverage.")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in actions:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def build_research_status(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    """Build the research status payload from local artifacts."""
    phase3_path = reports_root / PHASE3_AUDIT_REL
    phase3 = _load_json(phase3_path)

    data_status = _build_data_status(data_root=data_root, phase3=phase3)
    ledger_summary = _build_ledger_summary(reports_root=reports_root)
    accepted_trades = _build_accepted_trades_summary(reports_root)
    blockers = _derive_blockers(
        data_status=data_status,
        ledger_summary=ledger_summary,
        accepted_trades=accepted_trades,
        phase3=phase3,
    )
    next_actions = _derive_next_actions(
        blockers=blockers,
        data_status=data_status,
        ledger_summary=ledger_summary,
    )

    return {
        "generated_at": _utc_now_iso(),
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "real_execution_ready": False,
        "safety_statement": SAFETY_STATEMENT,
        "research_only_note": RESEARCH_ONLY_NOTE,
        "sections": list(REPORT_SECTIONS),
        "safety": {
            "paper_only": PAPER_ONLY,
            "execution_enabled": EXECUTION_ENABLED,
            "real_execution_ready": False,
            "safety_statement": SAFETY_STATEMENT,
        },
        "data_status": data_status,
        "candidate_timeline": list(CANDIDATE_TIMELINE),
        "ledger_summary": ledger_summary,
        "accepted_trades_analysis": accepted_trades,
        "blockers": blockers,
        "next_actions": next_actions,
        "inputs": {
            "reports_root": str(reports_root),
            "data_root": str(data_root),
            "phase3_audit_path": str(phase3_path),
            "phase3_available": phase3 is not None,
        },
    }


def write_research_status_markdown(
    status: dict[str, Any],
    output_dir: Path,
    *,
    filename: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    lines = [
        "# Bid/Ask Research Status and Decision Log",
        "",
        f"Generated at: {status['generated_at']}",
        "",
        "## Safety Statement",
        "",
        f"- paper_only: {str(status['paper_only']).lower()}",
        f"- execution_enabled: {str(status['execution_enabled']).lower()}",
        f"- real_execution_ready: {str(status['real_execution_ready']).lower()}",
        "",
        status.get("research_only_note", RESEARCH_ONLY_NOTE),
        "",
        "## Data Status",
        "",
    ]

    for coin in RESEARCH_COINS:
        coin_data = (status["data_status"].get("coins") or {}).get(coin) or {}
        lines.append(f"### {coin}")
        lines.append("")
        for interval in INTERVALS:
            row = coin_data.get(interval) or {}
            if not row.get("file_exists"):
                lines.append(f"- {interval}: file missing")
                continue
            lines.append(
                f"- {interval}: rows={row.get('row_count')}, "
                f"last={row.get('last_timestamp') or 'n/a'}, "
                f"bid_size={str(row.get('has_bid_size')).lower()}, "
                f"ask_size={str(row.get('has_ask_size')).lower()}"
            )
        lines.append("")

    data_warnings = status["data_status"].get("warnings") or []
    if data_warnings:
        lines.extend(["**Validator warnings**", ""])
        lines.extend(f"- {warning}" for warning in data_warnings)
        lines.append("")

    lines.extend(["## Candidate Timeline", ""])
    for entry in status.get("candidate_timeline") or []:
        lines.append(
            f"- Candidate {entry.get('candidate')} ({entry.get('name')}): "
            f"{entry.get('status')} — {entry.get('summary')}"
        )
    lines.append("")

    ledger = status.get("ledger_summary") or {}
    lines.extend(["## Latest Candidate Ledger Summary", ""])
    if not ledger.get("available"):
        lines.append("- Candidate ledger and grid artifacts not available.")
    else:
        lines.append(f"- ledger rows: {ledger.get('ledger_row_count', 0)}")
        lines.append(f"- ledger path: {ledger.get('ledger_path') or 'n/a'}")
        lines.append(f"- latest grid: {ledger.get('latest_grid_path') or 'n/a'}")
        lines.append("")

        lines.extend(["### Top 5 by paper_score (latest grid)", ""])
        top_scores = ledger.get("top_by_paper_score") or []
        if not top_scores:
            lines.append("- No grid results available.")
        else:
            for row in top_scores:
                sparse = " **SPARSE (<10 trades)**" if row.get("sparse_warning") else ""
                lines.append(
                    f"- {row.get('config_id') or row.get('coin')}: "
                    f"score={row.get('paper_score')}, trades={row.get('trade_count')}, "
                    f"positive_after_costs={row.get('positive_after_costs')}{sparse}"
                )
        lines.append("")

        lines.extend(["### Top positive_after_costs configs", ""])
        positives = (ledger.get("positive_after_costs_grid") or []) + (
            ledger.get("positive_after_costs_ledger") or []
        )
        if not positives:
            lines.append("- No positive-after-costs configs found in latest artifacts.")
        else:
            seen_keys: set[str] = set()
            for row in positives:
                key = "|".join(
                    str(row.get(field) or "")
                    for field in ("candidate_name", "config_id", "coin", "feature_set", "label_mode")
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                sparse = " **SPARSE (<10 trades)**" if row.get("sparse_warning") else ""
                lines.append(
                    f"- {row.get('candidate_name') or row.get('config_id') or row.get('coin')}: "
                    f"cumulative={row.get('cumulative_net_return_bps')} bps, "
                    f"trades={row.get('trade_count')}{sparse}"
                )
        lines.append("")

    lines.extend(["## Latest Accepted-Trades Analysis Summary", ""])
    accepted = status.get("accepted_trades_analysis") or {}
    if not accepted.get("available"):
        lines.append("- No accepted-trades analysis reports found.")
    else:
        for coin in RESEARCH_COINS:
            row = (accepted.get("by_coin") or {}).get(coin) or {}
            if not row.get("available"):
                lines.append(f"- {coin}: no analysis report")
                continue
            lines.append(
                f"- {coin}: trades={row.get('trade_count')}, "
                f"cumulative_net={row.get('cumulative_net_return_bps')} bps, "
                f"avg_net={row.get('average_net_return_bps')} bps, "
                f"win_rate={row.get('win_rate')}"
            )
            suspicious = row.get("suspicious_regimes") or []
            if suspicious:
                lines.append(f"  - suspicious regimes: {len(suspicious)} flagged")
                for regime in suspicious[:3]:
                    lines.append(f"    - {regime}")
    lines.append("")

    lines.extend(["## Current Blockers", ""])
    lines.extend(f"- {item}" for item in status.get("blockers") or [])
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in status.get("next_actions") or [])
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_research_status_json(
    status: dict[str, Any],
    output_dir: Path,
    *,
    filename: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    output_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def build_and_write_research_status(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    latest_json_name: str = DEFAULT_LATEST_JSON_NAME,
    latest_md_name: str = DEFAULT_LATEST_MD_NAME,
) -> dict[str, Any]:
    output_dir = _validate_output_dir(output_dir)
    status = build_research_status(reports_root=reports_root, data_root=data_root)
    stamp = _utc_stamp()

    json_stamped = write_research_status_json(
        status,
        output_dir,
        filename=f"{STAMPED_JSON_PREFIX}{stamp}.json",
    )
    md_stamped = write_research_status_markdown(
        status,
        output_dir,
        filename=f"{STAMPED_MD_PREFIX}{stamp}.md",
    )
    json_latest = write_research_status_json(status, output_dir, filename=latest_json_name)
    md_latest = write_research_status_markdown(status, output_dir, filename=latest_md_name)

    status["output_paths"] = {
        "json_latest": str(json_latest),
        "markdown_latest": str(md_latest),
        "json_stamped": str(json_stamped),
        "markdown_stamped": str(md_stamped),
        "stamp": stamp,
    }
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build bid/ask research status report (paper-only, read-only inputs).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build",
        help="Read local artifacts and write research status report files.",
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
        help="Directory for research status output files.",
    )

    args = parser.parse_args(argv)

    if args.command != "build":
        parser.error(f"Unknown command: {args.command}")
        return 2

    status = build_and_write_research_status(
        reports_root=args.reports_root,
        data_root=args.data_root,
        output_dir=args.output_dir,
    )
    paths = status["output_paths"]
    print(f"json_latest_path={paths['json_latest']}")
    print(f"markdown_latest_path={paths['markdown_latest']}")
    print(f"json_stamped_path={paths['json_stamped']}")
    print(f"markdown_stamped_path={paths['markdown_stamped']}")
    print(f"paper_only={str(status['paper_only']).lower()}")
    print(f"execution_enabled={str(status['execution_enabled']).lower()}")
    print(f"real_execution_ready={str(status['real_execution_ready']).lower()}")
    print(f"blockers={len(status['blockers'])}")
    print(f"next_actions={len(status['next_actions'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
