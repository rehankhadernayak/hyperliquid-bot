"""Read-only robustness analyzer for Candidate F grid and accepted-trades artifacts.

Research-only and paper-only. Reads local CSV/JSON/Markdown reports only.
No orders, execution, wallets, AWS, network, or shell commands.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bid_ask_features import REGIME_BREAKDOWN_COLUMNS
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, PAPER_ONLY, SAFETY_STATEMENT

CANDIDATE_F_FEATURE_SET = "candidate_f_regime_depth"

DEFAULT_CANDIDATE_GRID_DIR = Path("reports/candidate_grid")
DEFAULT_WALK_FORWARD_DIR = Path("reports/bid_ask_walk_forward")
DEFAULT_OUTPUT_DIR = Path("reports/candidate_f_robustness")

LATEST_JSON_NAME = "candidate_f_robustness_latest.json"
LATEST_MD_NAME = "candidate_f_robustness_latest.md"
STAMPED_JSON_PREFIX = "candidate_f_robustness_"
STAMPED_MD_PREFIX = "candidate_f_robustness_"

MIN_POSITIVE_TRADES = 10
MIN_ROBUST_TRADES = 30
TOP3_PROFIT_CONCENTRATION_THRESHOLD = 0.70
HOUR_TRADE_CONCENTRATION_THRESHOLD = 0.60

VERDICT_NO_POSITIVE = "no_positive_candidate_f_configs"
VERDICT_INSUFFICIENT_TRADES = "insufficient_matching_trades"
VERDICT_FRAGILE = "fragile_outlier_driven"
VERDICT_HOUR_CONCENTRATED = "regime_concentrated"
VERDICT_SPARSE = "diagnostic_positive_sparse"

SAFETY_NOTE = (
    "Candidate F robustness analysis is read-only and paper-only. "
    "Reads local report artifacts only. No orders, execution, wallets, "
    "AWS, network, or shell commands."
)

REQUIRED_TRADE_COLUMNS = ("timestamp_iso", "net_return_bps")

CONFIG_MATCH_FIELDS = (
    "feature_set",
    "label_mode",
    "horizon",
    "interval",
    "min_confidence",
    "min_edge_bps",
    "cost_multiplier",
    "max_spread_bps",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_report_stamp(path: Path) -> str:
    stem = path.stem
    if stem.startswith("candidate_grid_"):
        return stem.removeprefix("candidate_grid_")
    return stem


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _accepted_trades_count(row: dict[str, Any]) -> int:
    for key in (
        "accepted_trades_count",
        "filtered_paper_trades",
        "trades_after_regime_filters",
        "accepted_edge_predictions",
        "enforcement_trade_count",
    ):
        value = row.get(key)
        if value is None:
            continue
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count
    raw = row.get("accepted_trades_count")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    return 0


def _config_from_row(row: dict[str, Any]) -> dict[str, Any]:
    nested = row.get("config")
    if isinstance(nested, dict):
        return nested
    return row


def _row_feature_set(row: dict[str, Any]) -> str:
    config = _config_from_row(row)
    return str(row.get("feature_set") or config.get("feature_set") or "")


def _grid_has_candidate_f(rows: list[dict[str, Any]]) -> bool:
    return any(_row_feature_set(row) == CANDIDATE_F_FEATURE_SET for row in rows)


def _load_grid_rows_from_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _load_grid_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        payload = _load_json(path)
        if payload is None:
            return []
        for key in ("ranked", "results"):
            rows = payload.get(key)
            if isinstance(rows, list) and rows:
                return [row for row in rows if isinstance(row, dict)]
        return []
    if path.suffix == ".csv":
        return _load_grid_rows_from_csv(path)
    return []


def discover_latest_candidate_f_grid(
    grid_dir: Path = DEFAULT_CANDIDATE_GRID_DIR,
) -> Path | None:
    """Return the newest candidate grid report containing Candidate F rows."""
    if not grid_dir.is_dir():
        return None

    candidates: list[tuple[str, Path]] = []
    for pattern in ("candidate_grid_*.json", "candidate_grid_*.csv"):
        for path in grid_dir.glob(pattern):
            stamp = _parse_report_stamp(path)
            rows = _load_grid_rows(path)
            if rows and _grid_has_candidate_f(rows):
                candidates.append((stamp, path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def identify_positive_candidate_f_configs(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter grid rows that are positive Candidate F configs with enough trades."""
    positives: list[dict[str, Any]] = []
    for row in rows:
        if _row_feature_set(row) != CANDIDATE_F_FEATURE_SET:
            continue
        avg_net = float(row.get("average_net_return_bps") or 0.0)
        trade_count = _accepted_trades_count(row)
        if avg_net <= 0 or trade_count < MIN_POSITIVE_TRADES:
            continue
        config = _config_from_row(row)
        positives.append(
            {
                "config_id": row.get("config_id"),
                "coin": config.get("coin") or row.get("coin"),
                "interval": config.get("interval") or row.get("interval"),
                "horizon": config.get("horizon") or row.get("horizon"),
                "feature_set": CANDIDATE_F_FEATURE_SET,
                "label_mode": config.get("label_mode") or row.get("label_mode"),
                "trade_class": config.get("trade_class") or row.get("trade_class"),
                "min_confidence": config.get("min_confidence") or row.get("min_confidence"),
                "min_edge_bps": config.get("min_edge_bps") or row.get("min_edge_bps"),
                "cost_multiplier": config.get("cost_multiplier") or row.get("cost_multiplier"),
                "max_spread_bps": config.get("max_spread_bps") or row.get("max_spread_bps"),
                "average_net_return_bps": avg_net,
                "cumulative_net_return_bps": float(
                    row.get("cumulative_net_return_bps") or 0.0
                ),
                "accepted_trades_count": trade_count,
                "win_rate_after_costs": row.get("win_rate_after_costs"),
                "status": row.get("status"),
            }
        )
    positives.sort(
        key=lambda item: (
            float(item.get("average_net_return_bps") or 0.0),
            int(item.get("accepted_trades_count") or 0),
        ),
        reverse=True,
    )
    return positives


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


def _config_matches_walk_forward(
    config: dict[str, Any],
    report: dict[str, Any],
) -> bool:
    coin = str(config.get("coin", "")).upper()
    coins = {str(item).upper() for item in report.get("coins") or []}
    if coin and coins and coin not in coins:
        return False

    trade_class = str(config.get("trade_class", "")).upper()
    trade_classes = _normalize_trade_classes(report.get("trade_classes"))
    if trade_class and trade_classes and trade_class not in trade_classes:
        return False

    for field in CONFIG_MATCH_FIELDS:
        expected = config.get(field)
        actual = report.get(field)
        if expected is None:
            continue
        if field in {"min_confidence", "min_edge_bps", "cost_multiplier", "max_spread_bps"}:
            if not _float_equal(expected, actual):
                return False
        elif str(expected) != str(actual):
            return False
    return True


def find_matching_accepted_trades_csv(
    config: dict[str, Any],
    *,
    walk_forward_dir: Path = DEFAULT_WALK_FORWARD_DIR,
) -> Path | None:
    """Find an accepted-trades CSV whose walk-forward report matches the config."""
    if not walk_forward_dir.is_dir():
        return None

    matches: list[tuple[str, Path]] = []
    for report_path in walk_forward_dir.glob("walk_forward_*.json"):
        payload = _load_json(report_path)
        if payload is None:
            continue
        if not _config_matches_walk_forward(config, payload):
            continue

        csv_path_raw = payload.get("accepted_trades_csv_path")
        if isinstance(csv_path_raw, str) and csv_path_raw.strip():
            csv_path = Path(csv_path_raw)
            if not csv_path.is_file():
                csv_path = walk_forward_dir / Path(csv_path_raw).name
        else:
            stamp = report_path.stem.removeprefix("walk_forward_")
            coin = str(config.get("coin", "UNKNOWN")).upper()
            csv_path = walk_forward_dir / f"accepted_trades_{coin}_{stamp}.csv"

        if csv_path.is_file():
            matches.append((_parse_report_stamp(csv_path), csv_path))

    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def _parse_timestamp_iso(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _aggregate_bucket(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0, "sum_net_bps": 0.0, "avg_net_bps": 0.0}
    net_returns = [float(record["net_return_bps"]) for record in records]
    total = float(sum(net_returns))
    count = len(net_returns)
    return {
        "count": count,
        "sum_net_bps": total,
        "avg_net_bps": total / count,
    }


def _trade_summary(row: pd.Series) -> dict[str, Any]:
    return {
        "timestamp_iso": str(row.get("timestamp_iso", "")),
        "predicted_class": str(row.get("predicted_class", "")),
        "net_return_bps": float(row["net_return_bps"]),
    }


def analyze_accepted_trades_robustness(df: pd.DataFrame) -> dict[str, Any]:
    """Compute accepted-trades robustness metrics for Candidate F review."""
    missing = [column for column in REQUIRED_TRADE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    working = df.copy()
    working["net_return_bps"] = pd.to_numeric(working["net_return_bps"], errors="coerce")
    working = working[working["net_return_bps"].notna()]
    net_returns = working["net_return_bps"].astype(float).tolist()

    trade_count = len(net_returns)
    cumulative_net_bps = float(sum(net_returns))
    average_net_bps = cumulative_net_bps / trade_count if trade_count else 0.0
    wins = sum(1 for value in net_returns if value > 0)
    win_rate = wins / trade_count if trade_count else 0.0

    sorted_df = working.sort_values("net_return_bps")
    best_trade = _trade_summary(sorted_df.iloc[-1]) if trade_count else None
    worst_trade = _trade_summary(sorted_df.iloc[0]) if trade_count else None

    positive_returns = sorted(
        [value for value in net_returns if value > 0],
        reverse=True,
    )
    cumulative_profit = float(sum(positive_returns))
    top_1_sum = float(positive_returns[0]) if positive_returns else 0.0
    top_3_sum = float(sum(positive_returns[:3])) if positive_returns else 0.0
    top_1_trade_contribution_pct = (
        top_1_sum / cumulative_profit if cumulative_profit > 0 else 0.0
    )
    top_3_trades_contribution_pct = (
        top_3_sum / cumulative_profit if cumulative_profit > 0 else 0.0
    )

    losses = sorted(value for value in net_returns if value < 0)
    total_loss = float(sum(losses))
    worst_3_sum = float(sum(losses[:3])) if losses else 0.0
    worst_3_trades_loss_share_pct = (
        abs(worst_3_sum / total_loss) if total_loss < 0 else 0.0
    )

    by_utc_hour: dict[str, dict[str, Any]] = {}
    by_day: dict[str, dict[str, Any]] = {}
    hour_counts: dict[str, int] = {}

    for _, row in working.iterrows():
        parsed = _parse_timestamp_iso(row.get("timestamp_iso"))
        if parsed is None:
            continue
        hour_key = f"{parsed.hour:02d}"
        day_key = parsed.date().isoformat()
        hour_counts[hour_key] = hour_counts.get(hour_key, 0) + 1

    for hour, group in working.groupby(
        working["timestamp_iso"].map(
            lambda value: (
                f"{parsed.hour:02d}"
                if (parsed := _parse_timestamp_iso(value)) is not None
                else None
            )
        ),
        dropna=True,
    ):
        by_utc_hour[str(hour)] = _aggregate_bucket(group.to_dict(orient="records"))

    for day, group in working.groupby(
        working["timestamp_iso"].map(
            lambda value: (
                parsed.date().isoformat()
                if (parsed := _parse_timestamp_iso(value)) is not None
                else None
            )
        ),
        dropna=True,
    ):
        by_day[str(day)] = _aggregate_bucket(group.to_dict(orient="records"))

    max_hour_trade_fraction = (
        max(hour_counts.values()) / trade_count if trade_count and hour_counts else 0.0
    )
    dominant_hour = None
    if hour_counts:
        dominant_hour = max(hour_counts, key=hour_counts.get)

    by_regime: dict[str, dict[str, Any]] = {}
    for column in REGIME_BREAKDOWN_COLUMNS:
        if column not in working.columns:
            continue
        active = working[pd.to_numeric(working[column], errors="coerce") >= 0.5]
        by_regime[column] = _aggregate_bucket(active.to_dict(orient="records"))

    return {
        "trade_count": trade_count,
        "cumulative_net_bps": cumulative_net_bps,
        "average_net_bps": average_net_bps,
        "win_rate": win_rate,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "top_1_trade_contribution_pct": top_1_trade_contribution_pct,
        "top_3_trades_contribution_pct": top_3_trades_contribution_pct,
        "worst_3_trades_loss_share_pct": worst_3_trades_loss_share_pct,
        "by_utc_hour": dict(sorted(by_utc_hour.items())),
        "by_day": dict(sorted(by_day.items())),
        "by_regime": by_regime,
        "dominant_utc_hour": dominant_hour,
        "dominant_utc_hour_trade_fraction": max_hour_trade_fraction,
    }


def compute_robustness_verdict(
    *,
    trade_analysis: dict[str, Any] | None,
    has_matching_trades: bool,
) -> str:
    """Apply Candidate F robustness rules and return a verdict label."""
    if not has_matching_trades or trade_analysis is None:
        return VERDICT_INSUFFICIENT_TRADES

    trade_count = int(trade_analysis.get("trade_count") or 0)
    if trade_count < MIN_ROBUST_TRADES:
        return VERDICT_SPARSE

    top3_pct = float(trade_analysis.get("top_3_trades_contribution_pct") or 0.0)
    if top3_pct > TOP3_PROFIT_CONCENTRATION_THRESHOLD:
        return VERDICT_FRAGILE

    hour_fraction = float(
        trade_analysis.get("dominant_utc_hour_trade_fraction") or 0.0
    )
    if hour_fraction > HOUR_TRADE_CONCENTRATION_THRESHOLD:
        return VERDICT_HOUR_CONCENTRATED

    return VERDICT_SPARSE


def _next_recommended_action(verdict: str) -> str:
    actions = {
        VERDICT_NO_POSITIVE: (
            "Expand Candidate F grid search or revisit feature engineering before "
            "robustness review."
        ),
        VERDICT_INSUFFICIENT_TRADES: (
            "Re-run walk-forward evaluation with save_accepted_trades enabled for "
            "positive Candidate F configs, then re-run this analyzer."
        ),
        VERDICT_SPARSE: (
            "Continue paper-only data collection until at least 30 accepted trades "
            "exist for the leading Candidate F config."
        ),
        VERDICT_FRAGILE: (
            "Treat the positive grid result as outlier-driven; inspect top trades "
            "and tighten filters before any further promotion."
        ),
        VERDICT_HOUR_CONCENTRATED: (
            "Test hour-based exclusions or gather more out-of-sample hours before "
            "trusting Candidate F."
        ),
    }
    return actions.get(
        verdict,
        "Continue paper-only Candidate F research with local artifacts only.",
    )


def build_candidate_f_robustness_report(
    *,
    grid_path: Path | None,
    grid_rows: list[dict[str, Any]],
    positive_configs: list[dict[str, Any]],
    config_analyses: list[dict[str, Any]],
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Assemble the full Candidate F robustness report payload."""
    if not positive_configs:
        overall_verdict = VERDICT_NO_POSITIVE
    else:
        verdicts = [item.get("verdict") for item in config_analyses if item.get("verdict")]
        priority = (
            VERDICT_FRAGILE,
            VERDICT_HOUR_CONCENTRATED,
            VERDICT_SPARSE,
            VERDICT_INSUFFICIENT_TRADES,
        )
        overall_verdict = VERDICT_SPARSE
        for candidate in priority:
            if candidate in verdicts:
                overall_verdict = candidate
                break
        if not verdicts:
            overall_verdict = VERDICT_INSUFFICIENT_TRADES

    candidate_f_rows = [
        row for row in grid_rows if _row_feature_set(row) == CANDIDATE_F_FEATURE_SET
    ]
    best_positive = positive_configs[0] if positive_configs else None

    return {
        "generated_at": generated_at or _utc_now_iso(),
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "real_execution_ready": False,
        "robust_candidate": False,
        "candidate_grid_source": str(grid_path) if grid_path else None,
        "candidate_f_summary": {
            "feature_set": CANDIDATE_F_FEATURE_SET,
            "grid_row_count": len(candidate_f_rows),
            "positive_config_count": len(positive_configs),
            "best_config_id": best_positive.get("config_id") if best_positive else None,
            "best_average_net_return_bps": (
                best_positive.get("average_net_return_bps") if best_positive else None
            ),
            "best_accepted_trades_count": (
                best_positive.get("accepted_trades_count") if best_positive else None
            ),
        },
        "positive_configs": positive_configs,
        "config_analyses": config_analyses,
        "overall_verdict": overall_verdict,
        "next_recommended_action": _next_recommended_action(overall_verdict),
        "safety_note": SAFETY_NOTE,
    }


def analyze_config_robustness(
    config: dict[str, Any],
    *,
    walk_forward_dir: Path = DEFAULT_WALK_FORWARD_DIR,
) -> dict[str, Any]:
    """Analyze one positive Candidate F config against accepted-trades artifacts."""
    csv_path = find_matching_accepted_trades_csv(
        config,
        walk_forward_dir=walk_forward_dir,
    )
    analysis: dict[str, Any] = {
        "config_id": config.get("config_id"),
        "grid_metrics": config,
        "accepted_trades_csv_path": str(csv_path) if csv_path else None,
        "matching_status": "matched" if csv_path else "not_found",
        "trade_analysis": None,
        "verdict": VERDICT_INSUFFICIENT_TRADES,
        "robust_candidate": False,
    }

    if csv_path is None:
        return analysis

    df = pd.read_csv(csv_path)
    trade_analysis = analyze_accepted_trades_robustness(df)
    verdict = compute_robustness_verdict(
        trade_analysis=trade_analysis,
        has_matching_trades=True,
    )
    analysis["trade_analysis"] = trade_analysis
    analysis["verdict"] = verdict
    return analysis


def write_robustness_json(
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
    return f"{value:.2f} bps"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def write_robustness_markdown(
    report: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    markdown_name: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = markdown_name or LATEST_MD_NAME
    output_path = output_dir / filename

    summary = report.get("candidate_f_summary") or {}
    lines = [
        "# Candidate F Robustness Report",
        "",
        f"Generated at: {report.get('generated_at', 'n/a')}",
        f"Candidate grid source: {report.get('candidate_grid_source', 'n/a')}",
        "",
        "## Safety statement",
        "",
        f"- paper_only: {str(report.get('paper_only', True)).lower()}",
        f"- execution_enabled: {str(report.get('execution_enabled', False)).lower()}",
        f"- real_execution_ready: {str(report.get('real_execution_ready', False)).lower()}",
        "",
        SAFETY_NOTE,
        "",
        "## Candidate F summary",
        "",
        f"- feature_set: {summary.get('feature_set', CANDIDATE_F_FEATURE_SET)}",
        f"- grid_row_count: {summary.get('grid_row_count', 0)}",
        f"- positive_config_count: {summary.get('positive_config_count', 0)}",
        f"- best_config_id: {summary.get('best_config_id', 'n/a')}",
        f"- best_average_net_return_bps: {_format_bps(summary.get('best_average_net_return_bps'))}",
        f"- best_accepted_trades_count: {summary.get('best_accepted_trades_count', 'n/a')}",
        "",
        "## Positive configs",
        "",
    ]

    positive_configs = report.get("positive_configs") or []
    if not positive_configs:
        lines.append("No positive Candidate F configs met the review thresholds.")
    else:
        for index, config in enumerate(positive_configs, start=1):
            lines.append(
                f"{index}. **{config.get('config_id', 'n/a')}** — "
                f"avg={_format_bps(float(config.get('average_net_return_bps') or 0.0))}, "
                f"trades={config.get('accepted_trades_count', 'n/a')}"
            )

    lines.extend(["", "## Accepted-trades robustness", ""])
    config_analyses = report.get("config_analyses") or []
    if not config_analyses:
        lines.append("No config-level accepted-trades analysis available.")
    else:
        for item in config_analyses:
            lines.append(f"### {item.get('config_id', 'n/a')}")
            lines.append("")
            lines.append(f"- matching_status: {item.get('matching_status', 'n/a')}")
            lines.append(
                f"- accepted_trades_csv_path: {item.get('accepted_trades_csv_path', 'n/a')}"
            )
            trade = item.get("trade_analysis")
            if not trade:
                lines.append("- trade_analysis: unavailable")
            else:
                lines.extend(
                    [
                        f"- trade_count: {trade.get('trade_count', 0)}",
                        f"- cumulative_net_bps: {_format_bps(float(trade.get('cumulative_net_bps') or 0.0))}",
                        f"- average_net_bps: {_format_bps(float(trade.get('average_net_bps') or 0.0))}",
                        f"- win_rate: {_format_pct(float(trade.get('win_rate') or 0.0))}",
                        f"- top_1_trade_contribution_pct: {_format_pct(float(trade.get('top_1_trade_contribution_pct') or 0.0))}",
                        f"- top_3_trades_contribution_pct: {_format_pct(float(trade.get('top_3_trades_contribution_pct') or 0.0))}",
                        f"- worst_3_trades_loss_share_pct: {_format_pct(float(trade.get('worst_3_trades_loss_share_pct') or 0.0))}",
                    ]
                )
            lines.append(f"- verdict: {item.get('verdict', 'n/a')}")
            lines.append("")

    lines.extend(["## Regime/hour/day concentration", ""])
    if config_analyses and config_analyses[0].get("trade_analysis"):
        trade = config_analyses[0]["trade_analysis"]
        lines.append(
            f"- dominant UTC hour: {trade.get('dominant_utc_hour', 'n/a')} "
            f"({_format_pct(float(trade.get('dominant_utc_hour_trade_fraction') or 0.0))} of trades)"
        )
        by_regime = trade.get("by_regime") or {}
        if by_regime:
            for regime, metrics in sorted(by_regime.items()):
                lines.append(
                    f"- {regime}: count={metrics.get('count', 0)}, "
                    f"sum={_format_bps(float(metrics.get('sum_net_bps') or 0.0))}, "
                    f"avg={_format_bps(float(metrics.get('avg_net_bps') or 0.0))}"
                )
        else:
            lines.append("- regime columns not present in accepted-trades CSV")
        by_hour = trade.get("by_utc_hour") or {}
        if by_hour:
            lines.append("- by UTC hour:")
            for hour, metrics in by_hour.items():
                lines.append(
                    f"  - {hour}: count={metrics.get('count', 0)}, "
                    f"sum={_format_bps(float(metrics.get('sum_net_bps') or 0.0))}, "
                    f"avg={_format_bps(float(metrics.get('avg_net_bps') or 0.0))}"
                )
        by_day = trade.get("by_day") or {}
        if by_day:
            lines.append("- by day:")
            for day, metrics in by_day.items():
                lines.append(
                    f"  - {day}: count={metrics.get('count', 0)}, "
                    f"sum={_format_bps(float(metrics.get('sum_net_bps') or 0.0))}, "
                    f"avg={_format_bps(float(metrics.get('avg_net_bps') or 0.0))}"
                )
    else:
        lines.append("Concentration metrics unavailable without matching accepted trades.")

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- overall_verdict: {report.get('overall_verdict', 'n/a')}",
            f"- robust_candidate: {str(report.get('robust_candidate', False)).lower()}",
            "",
            "## Next recommended action",
            "",
            str(report.get("next_recommended_action", "")),
            "",
            SAFETY_STATEMENT,
            "",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["markdown_report_path"] = str(output_path)
    return output_path


def run_candidate_f_robustness_analysis(
    *,
    grid_dir: Path = DEFAULT_CANDIDATE_GRID_DIR,
    walk_forward_dir: Path = DEFAULT_WALK_FORWARD_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    grid_path: Path | None = None,
    write_stamped: bool = True,
) -> dict[str, Any]:
    """Load Candidate F grid artifacts, analyze robustness, and write reports."""
    resolved_grid = grid_path or discover_latest_candidate_f_grid(grid_dir=grid_dir)
    grid_rows = _load_grid_rows(resolved_grid) if resolved_grid else []
    positive_configs = identify_positive_candidate_f_configs(grid_rows)

    config_analyses = [
        analyze_config_robustness(config, walk_forward_dir=walk_forward_dir)
        for config in positive_configs
    ]

    report = build_candidate_f_robustness_report(
        grid_path=resolved_grid,
        grid_rows=grid_rows,
        positive_configs=positive_configs,
        config_analyses=config_analyses,
    )

    write_robustness_json(report, output_dir=output_dir, json_name=LATEST_JSON_NAME)
    write_robustness_markdown(report, output_dir=output_dir, markdown_name=LATEST_MD_NAME)
    latest_json_path = report["json_report_path"]
    latest_markdown_path = report["markdown_report_path"]

    if write_stamped:
        stamp = _utc_stamp()
        stamped_report = dict(report)
        write_robustness_json(
            stamped_report,
            output_dir=output_dir,
            json_name=f"{STAMPED_JSON_PREFIX}{stamp}.json",
        )
        write_robustness_markdown(
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
            "Analyze Candidate F grid robustness from local report artifacts "
            "(research-only, no execution)."
        )
    )
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=DEFAULT_CANDIDATE_GRID_DIR,
        help=f"Candidate grid reports directory (default: {DEFAULT_CANDIDATE_GRID_DIR})",
    )
    parser.add_argument(
        "--grid-path",
        type=Path,
        help="Explicit candidate grid JSON/CSV path",
    )
    parser.add_argument(
        "--walk-forward-dir",
        type=Path,
        default=DEFAULT_WALK_FORWARD_DIR,
        help=f"Walk-forward reports directory (default: {DEFAULT_WALK_FORWARD_DIR})",
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

    report = run_candidate_f_robustness_analysis(
        grid_dir=args.grid_dir,
        walk_forward_dir=args.walk_forward_dir,
        output_dir=args.output_dir,
        grid_path=args.grid_path,
        write_stamped=not args.no_stamped,
    )

    print(f"candidate_grid_source={report.get('candidate_grid_source')}")
    print(f"json_report_path={report.get('json_report_path')}")
    print(f"markdown_report_path={report.get('markdown_report_path')}")
    print(f"positive_config_count={report['candidate_f_summary']['positive_config_count']}")
    print(f"overall_verdict={report.get('overall_verdict')}")
    print(f"robust_candidate={report.get('robust_candidate')}")
    print(SAFETY_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
