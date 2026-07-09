"""Paper-only candidate configuration grid runner for walk-forward models.

Research-only and paper-only. No orders, wallets, keys, AWS, or network calls.
Does not modify bid_ask_candidate_validator.py.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import itertools
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT
from bid_ask_walk_forward_evaluator import (
    DEFAULT_DATA_ROOT,
    binary_tradeable_direction,
    is_binary_tradeable_label_mode,
    parse_label_mode,
    run_walk_forward_evaluation,
)

SAFETY_NOTE = (
    "Candidate grid search is research-only and paper-only. "
    "Simulated returns do not prove profitability. No real orders are placed."
)

DEFAULT_OUTPUT_DIR = Path("reports/candidate_grid")
DEFAULT_SOURCE = "historical"
SOURCE_CHOICES = ("historical", "combined", "local")

DEFAULT_COINS = ("BTC", "ETH")
DEFAULT_INTERVAL = "15s"
DEFAULT_HORIZONS = (10, 20, 30)
DEFAULT_FEATURE_SETS = ("microstructure",)
DEFAULT_LABEL_MODES = ("direction",)
DEFAULT_TRADE_CLASSES = ("DOWN", "UP")
DEFAULT_MIN_CONFIDENCES = (0.55, 0.60, 0.65)
DEFAULT_MIN_EDGE_BPS_VALUES = (0.0, 1.0, 2.0)
DEFAULT_COST_MULTIPLIERS = (1.0, 1.25, 1.5)
DEFAULT_MAX_SPREAD_BPS_VALUES = (8.0, 12.0)
DEFAULT_FEE_BPS = 2.0
DEFAULT_SLIPPAGE_BPS = 1.0
DEFAULT_TOP_N = 20
DEFAULT_MIN_TRADES_REQUIRED = 20
DEFAULT_EXCLUDE_HOURS_VALUES = ("none",)
DEFAULT_INCLUDE_HOURS_VALUES = ("none",)
DEFAULT_MAX_COST_BPS_VALUES = ("none",)
DEFAULT_MIN_TRADES_AFTER_REGIME_FILTER = 5

PAPER_SCORE_POSITIVE_AFTER_COSTS_BONUS = 1_000.0
PAPER_SCORE_BEATS_BASELINE_BONUS = 500.0
PAPER_SCORE_TRADE_COUNT_WEIGHT = 10.0
PAPER_SCORE_AVERAGE_NET_RETURN_WEIGHT = 100.0
PAPER_SCORE_STABILITY_WARNING_PENALTY = 150.0
PAPER_SCORE_INSUFFICIENT_TRADES_PENALTY = 500.0
PAPER_SCORE_INSUFFICIENT_REGIME_TRADES_PENALTY = 500.0

WalkForwardRunner = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class CandidateGridConfig:
    coin: str
    interval: str
    horizon: int
    feature_set: str
    label_mode: str
    trade_class: str
    min_confidence: float
    min_edge_bps: float
    cost_multiplier: float
    max_spread_bps: float
    fee_bps: float
    slippage_bps: float
    source: str
    exclude_hours: tuple[int, ...] | None = None
    include_hours: tuple[int, ...] | None = None
    max_cost_bps: float | None = None

    @property
    def config_id(self) -> str:
        trade = self.trade_class.upper()
        base = (
            f"{self.coin}_{trade}_H{self.horizon}_"
            f"C{self.min_confidence:.2f}_E{self.min_edge_bps:.0f}_"
            f"M{self.cost_multiplier:.2f}_S{self.max_spread_bps:.0f}"
        )
        suffix_parts: list[str] = []
        if self.exclude_hours:
            suffix_parts.append(f"EX{format_hours_label(self.exclude_hours)}")
        if self.include_hours:
            suffix_parts.append(f"IN{format_hours_label(self.include_hours)}")
        if self.max_cost_bps is not None:
            suffix_parts.append(f"MC{self.max_cost_bps:g}")
        if suffix_parts:
            return f"{base}_{'_'.join(suffix_parts)}"
        return base

    @property
    def regime_filter_active(self) -> bool:
        return (
            self.exclude_hours is not None
            or self.include_hours is not None
            or self.max_cost_bps is not None
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _report_stamp(run_timestamp: str) -> str:
    if run_timestamp:
        return run_timestamp.replace(":", "").replace("+", "").replace("-", "")[:15]
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_csv_values(raw: str, *, value_type: type = str) -> tuple[Any, ...]:
    """Parse comma-separated CLI values into a tuple."""
    parts = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not parts:
        raise ValueError("At least one comma-separated value is required")
    if value_type is str:
        return tuple(parts)
    if value_type is int:
        return tuple(int(item) for item in parts)
    if value_type is float:
        return tuple(float(item) for item in parts)
    raise TypeError(f"Unsupported value_type: {value_type}")


def parse_regime_filter_csv_values(raw: str) -> tuple[str, ...]:
    """Parse comma-separated regime filter dimension values."""
    parts = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not parts:
        return DEFAULT_EXCLUDE_HOURS_VALUES
    return tuple(parts)


def parse_hours_value(raw: str) -> tuple[int, ...] | None:
    """Parse hour filter token: 'none' -> None, '3+5' -> (3, 5)."""
    text = str(raw).strip().lower()
    if not text or text == "none":
        return None
    hours: list[int] = []
    for part in text.split("+"):
        token = part.strip()
        if not token:
            continue
        hour = int(token)
        if hour < 0 or hour > 23:
            raise ValueError(f"Invalid hour token: {token}")
        hours.append(hour)
    if not hours:
        return None
    return tuple(hours)


def parse_max_cost_bps_value(raw: str) -> float | None:
    """Parse max cost bps token: 'none' -> None, '5' -> 5.0."""
    text = str(raw).strip().lower()
    if not text or text == "none":
        return None
    return float(text)


def format_hours_label(hours: tuple[int, ...] | None) -> str:
    """Format hour tuple for config_id and output columns."""
    if not hours:
        return ""
    return "+".join(str(hour) for hour in hours)


def _filter_runner_kwargs(
    runner: WalkForwardRunner,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Pass regime filter kwargs only when the runner signature supports them."""
    regime_keys = ("exclude_hours", "include_hours", "max_cost_bps")
    try:
        params = inspect.signature(runner).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return kwargs
    filtered = dict(kwargs)
    for key in regime_keys:
        if key in filtered and key not in params:
            del filtered[key]
    return filtered


def build_grid_combinations(
    *,
    coins: tuple[str, ...] = DEFAULT_COINS,
    interval: str = DEFAULT_INTERVAL,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    feature_sets: tuple[str, ...] = DEFAULT_FEATURE_SETS,
    label_modes: tuple[str, ...] = DEFAULT_LABEL_MODES,
    trade_classes: tuple[str, ...] = DEFAULT_TRADE_CLASSES,
    min_confidences: tuple[float, ...] = DEFAULT_MIN_CONFIDENCES,
    min_edge_bps_values: tuple[float, ...] = DEFAULT_MIN_EDGE_BPS_VALUES,
    cost_multipliers: tuple[float, ...] = DEFAULT_COST_MULTIPLIERS,
    max_spread_bps_values: tuple[float, ...] = DEFAULT_MAX_SPREAD_BPS_VALUES,
    exclude_hours_values: tuple[str, ...] = DEFAULT_EXCLUDE_HOURS_VALUES,
    include_hours_values: tuple[str, ...] = DEFAULT_INCLUDE_HOURS_VALUES,
    max_cost_bps_values: tuple[str, ...] = DEFAULT_MAX_COST_BPS_VALUES,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    source: str = DEFAULT_SOURCE,
) -> list[CandidateGridConfig]:
    """Build the cartesian product of candidate grid configurations."""
    configs: list[CandidateGridConfig] = []
    for label_mode in label_modes:
        parse_label_mode(str(label_mode))
    for (
        coin,
        horizon,
        feature_set,
        label_mode,
        trade_class,
        min_confidence,
        min_edge_bps,
        cost_multiplier,
        max_spread_bps,
        exclude_hours_raw,
        include_hours_raw,
        max_cost_bps_raw,
    ) in itertools.product(
        coins,
        horizons,
        feature_sets,
        label_modes,
        trade_classes,
        min_confidences,
        min_edge_bps_values,
        cost_multipliers,
        max_spread_bps_values,
        exclude_hours_values,
        include_hours_values,
        max_cost_bps_values,
    ):
        normalized_label_mode = str(label_mode)
        normalized_trade_class = str(trade_class).strip().upper()
        if is_binary_tradeable_label_mode(normalized_label_mode):
            if normalized_trade_class != binary_tradeable_direction(normalized_label_mode):
                continue
        configs.append(
            CandidateGridConfig(
                coin=coin.strip().upper(),
                interval=interval,
                horizon=int(horizon),
                feature_set=str(feature_set),
                label_mode=normalized_label_mode,
                trade_class=normalized_trade_class,
                min_confidence=float(min_confidence),
                min_edge_bps=float(min_edge_bps),
                cost_multiplier=float(cost_multiplier),
                max_spread_bps=float(max_spread_bps),
                fee_bps=float(fee_bps),
                slippage_bps=float(slippage_bps),
                source=source,
                exclude_hours=parse_hours_value(exclude_hours_raw),
                include_hours=parse_hours_value(include_hours_raw),
                max_cost_bps=parse_max_cost_bps_value(max_cost_bps_raw),
            )
        )
    return configs


def compute_paper_score(
    metrics: dict[str, Any],
    *,
    min_trades_required: int = DEFAULT_MIN_TRADES_REQUIRED,
    min_trades_after_regime_filter: int = DEFAULT_MIN_TRADES_AFTER_REGIME_FILTER,
) -> dict[str, Any]:
    """Compute paper-only ranking score with explicit bonuses and penalties."""
    positive_after_costs = bool(metrics.get("positive_after_costs", False))
    beats_baseline = bool(metrics.get("beats_baseline_accuracy", False))
    filtered_paper_trades = int(metrics.get("filtered_paper_trades", 0))
    average_net_return_bps = float(metrics.get("average_net_return_bps", 0.0))
    stability_summary = metrics.get("stability_summary") or {}
    stability_warnings = list(metrics.get("stability_warnings") or [])
    regime_filters_enabled = bool(metrics.get("regime_filters_enabled", False))
    trades_after_regime_filters = metrics.get("trades_after_regime_filters")

    insufficient_trades = filtered_paper_trades < min_trades_required
    has_concentration_warning = bool(stability_summary.get("concentration_warning"))
    has_sparse_warning = bool(stability_summary.get("sparse_stability_warning"))
    warning_count = len(stability_warnings) + int(has_concentration_warning) + int(has_sparse_warning)

    insufficient_regime_trades = False
    insufficient_regime_trades_penalty = 0.0
    if regime_filters_enabled and trades_after_regime_filters is not None:
        insufficient_regime_trades = (
            int(trades_after_regime_filters) < min_trades_after_regime_filter
        )
        insufficient_regime_trades_penalty = (
            PAPER_SCORE_INSUFFICIENT_REGIME_TRADES_PENALTY
            if insufficient_regime_trades
            else 0.0
        )

    positive_bonus = PAPER_SCORE_POSITIVE_AFTER_COSTS_BONUS if positive_after_costs else 0.0
    baseline_bonus = PAPER_SCORE_BEATS_BASELINE_BONUS if beats_baseline else 0.0
    trade_bonus = filtered_paper_trades * PAPER_SCORE_TRADE_COUNT_WEIGHT
    average_return_bonus = max(average_net_return_bps, 0.0) * PAPER_SCORE_AVERAGE_NET_RETURN_WEIGHT
    stability_penalty = warning_count * PAPER_SCORE_STABILITY_WARNING_PENALTY
    insufficient_trades_penalty = (
        PAPER_SCORE_INSUFFICIENT_TRADES_PENALTY if insufficient_trades else 0.0
    )

    paper_score = (
        positive_bonus
        + baseline_bonus
        + trade_bonus
        + average_return_bonus
        - stability_penalty
        - insufficient_trades_penalty
        - insufficient_regime_trades_penalty
    )

    return {
        "paper_score": paper_score,
        "positive_after_costs_bonus": positive_bonus,
        "beats_baseline_bonus": baseline_bonus,
        "trade_count_bonus": trade_bonus,
        "average_net_return_bonus": average_return_bonus,
        "stability_warnings_penalty": stability_penalty,
        "insufficient_trades_penalty": insufficient_trades_penalty,
        "insufficient_regime_trades_penalty": insufficient_regime_trades_penalty,
        "insufficient_trades": insufficient_trades,
        "insufficient_regime_trades": insufficient_regime_trades,
        "stability_warning_count": warning_count,
        "real_execution_ready": False,
    }


def extract_metrics_from_walk_forward(walk_forward_report: dict[str, Any]) -> dict[str, Any]:
    """Extract ranking metrics from a walk-forward evaluation report."""
    readiness = walk_forward_report.get("readiness") or {}
    return {
        "beats_baseline_accuracy": bool(
            walk_forward_report.get(
                "beats_baseline_accuracy",
                readiness.get("beats_baseline_accuracy", False),
            )
        ),
        "accuracy_edge_vs_baseline": float(
            walk_forward_report.get("accuracy_edge_vs_baseline", 0.0)
        ),
        "filtered_paper_trades": int(walk_forward_report.get("filtered_paper_trades", 0)),
        "cumulative_net_return_bps": float(
            walk_forward_report.get("cumulative_net_return_bps", 0.0)
        ),
        "average_net_return_bps": float(walk_forward_report.get("average_net_return_bps", 0.0)),
        "win_rate_after_costs": float(walk_forward_report.get("win_rate_after_costs", 0.0)),
        "positive_after_costs": bool(
            walk_forward_report.get(
                "positive_after_costs",
                readiness.get("positive_after_costs", False),
            )
        ),
        "stability_summary": dict(walk_forward_report.get("stability_summary") or {}),
        "stability_warnings": list(walk_forward_report.get("stability_warnings") or []),
        "regime_filters_enabled": bool(walk_forward_report.get("regime_filters_enabled", False)),
        "trades_before_regime_filters": walk_forward_report.get("trades_before_regime_filters"),
        "trades_after_regime_filters": walk_forward_report.get("trades_after_regime_filters"),
        "trades_removed_by_regime_filters": walk_forward_report.get(
            "trades_removed_by_regime_filters"
        ),
        "cumulative_net_return_before_regime_filters": walk_forward_report.get(
            "cumulative_net_return_before_regime_filters"
        ),
        "cumulative_net_return_after_regime_filters": walk_forward_report.get(
            "cumulative_net_return_after_regime_filters"
        ),
    }


def evaluate_grid_config(
    config: CandidateGridConfig,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    min_trades_required: int = DEFAULT_MIN_TRADES_REQUIRED,
    min_trades_after_regime_filter: int = DEFAULT_MIN_TRADES_AFTER_REGIME_FILTER,
    walk_forward_runner: WalkForwardRunner | None = None,
) -> dict[str, Any]:
    """Evaluate one grid configuration via walk-forward evaluation."""
    runner = walk_forward_runner or run_walk_forward_evaluation
    runner_kwargs = _filter_runner_kwargs(
        runner,
        {
            "coins": [config.coin],
            "interval": config.interval,
            "source": config.source,
            "horizon": config.horizon,
            "feature_set": config.feature_set,
            "label_mode": config.label_mode,
            "trade_classes": frozenset({config.trade_class}),
            "min_confidence": config.min_confidence,
            "min_edge_bps": config.min_edge_bps,
            "cost_multiplier": config.cost_multiplier,
            "max_spread_bps": config.max_spread_bps,
            "fee_bps": config.fee_bps,
            "slippage_bps": config.slippage_bps,
            "data_root": data_root,
            "save_reports": False,
            "save_accepted_trades": False,
            "stability_report_enabled": True,
            "exclude_hours": list(config.exclude_hours) if config.exclude_hours else None,
            "include_hours": list(config.include_hours) if config.include_hours else None,
            "max_cost_bps": config.max_cost_bps,
        },
    )
    walk_forward_report = runner(**runner_kwargs)
    metrics = extract_metrics_from_walk_forward(walk_forward_report)
    if config.regime_filter_active and not metrics.get("regime_filters_enabled"):
        metrics["regime_filters_enabled"] = True
    score_details = compute_paper_score(
        metrics,
        min_trades_required=min_trades_required,
        min_trades_after_regime_filter=min_trades_after_regime_filter,
    )
    return {
        "config_id": config.config_id,
        "config": asdict(config),
        "exclude_hours": format_hours_label(config.exclude_hours),
        "include_hours": format_hours_label(config.include_hours),
        "max_cost_bps": config.max_cost_bps,
        "exploratory_overfit_warning": config.regime_filter_active,
        **metrics,
        **score_details,
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "real_execution_ready": False,
    }


def rank_grid_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank grid results by paper_score descending, then trade count, then avg return."""
    return sorted(
        results,
        key=lambda row: (
            float(row.get("paper_score", 0.0)),
            int(row.get("filtered_paper_trades", 0)),
            float(row.get("average_net_return_bps", 0.0)),
        ),
        reverse=True,
    )


def run_candidate_grid(
    configs: list[CandidateGridConfig],
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    min_trades_required: int = DEFAULT_MIN_TRADES_REQUIRED,
    min_trades_after_regime_filter: int = DEFAULT_MIN_TRADES_AFTER_REGIME_FILTER,
    max_runs: int | None = None,
    dry_run: bool = False,
    walk_forward_runner: WalkForwardRunner | None = None,
) -> dict[str, Any]:
    """Run or plan candidate grid evaluation."""
    planned = configs[:max_runs] if max_runs is not None else configs
    if dry_run:
        return {
            "dry_run": True,
            "planned_count": len(planned),
            "total_combinations": len(configs),
            "sample_configs": [asdict(item) for item in planned[:5]],
            "results": [],
            "failures": [],
            "ranked": [],
        }

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for config in planned:
        try:
            results.append(
                evaluate_grid_config(
                    config,
                    data_root=data_root,
                    min_trades_required=min_trades_required,
                    min_trades_after_regime_filter=min_trades_after_regime_filter,
                    walk_forward_runner=walk_forward_runner,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "config_id": config.config_id,
                    "config": asdict(config),
                    "error": str(exc),
                }
            )

    ranked = rank_grid_results(results)
    return {
        "dry_run": False,
        "planned_count": len(planned),
        "total_combinations": len(configs),
        "completed_count": len(results),
        "failed_count": len(failures),
        "results": results,
        "failures": failures,
        "ranked": ranked,
    }


def build_grid_report(
    *,
    grid_output: dict[str, Any],
    run_timestamp: str,
    grid_params: dict[str, Any],
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, Any]:
    """Build the top-level candidate grid report payload."""
    ranked = grid_output.get("ranked") or []
    return {
        "grid_id": uuid.uuid4().hex,
        "run_timestamp": run_timestamp,
        "dry_run": bool(grid_output.get("dry_run")),
        "grid_params": grid_params,
        "planned_count": grid_output.get("planned_count", 0),
        "total_combinations": grid_output.get("total_combinations", 0),
        "completed_count": grid_output.get("completed_count", 0),
        "failed_count": grid_output.get("failed_count", 0),
        "results": grid_output.get("results", []),
        "failures": grid_output.get("failures", []),
        "ranked": ranked,
        "top_n": ranked[:top_n],
        "best_result": ranked[0] if ranked else None,
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "real_execution_ready": False,
        "safety_note": SAFETY_NOTE,
        "safety_statement": SAFETY_STATEMENT,
    }


def write_grid_report_json(
    report: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write timestamped JSON grid report under reports/candidate_grid/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _report_stamp(str(report.get("run_timestamp", "")))
    output_path = output_dir / f"candidate_grid_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    report["json_report_path"] = str(output_path)
    return output_path


def write_grid_report_csv(
    report: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write ranked results CSV under reports/candidate_grid/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _report_stamp(str(report.get("run_timestamp", "")))
    output_path = output_dir / f"candidate_grid_{stamp}.csv"

    fieldnames = [
        "rank",
        "config_id",
        "coin",
        "interval",
        "horizon",
        "feature_set",
        "label_mode",
        "trade_class",
        "min_confidence",
        "min_edge_bps",
        "cost_multiplier",
        "max_spread_bps",
        "fee_bps",
        "slippage_bps",
        "source",
        "exclude_hours",
        "include_hours",
        "max_cost_bps",
        "paper_score",
        "positive_after_costs",
        "beats_baseline_accuracy",
        "filtered_paper_trades",
        "trades_before_regime_filters",
        "trades_after_regime_filters",
        "trades_removed_by_regime_filters",
        "regime_filters_enabled",
        "exploratory_overfit_warning",
        "average_net_return_bps",
        "cumulative_net_return_bps",
        "win_rate_after_costs",
        "stability_warning_count",
        "insufficient_trades",
        "real_execution_ready",
    ]

    ranked = report.get("ranked") or []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(ranked, start=1):
            config = row.get("config") or {}
            writer.writerow(
                {
                    "rank": index,
                    "config_id": row.get("config_id"),
                    "coin": config.get("coin"),
                    "interval": config.get("interval"),
                    "horizon": config.get("horizon"),
                    "feature_set": config.get("feature_set"),
                    "label_mode": config.get("label_mode"),
                    "trade_class": config.get("trade_class"),
                    "min_confidence": config.get("min_confidence"),
                    "min_edge_bps": config.get("min_edge_bps"),
                    "cost_multiplier": config.get("cost_multiplier"),
                    "max_spread_bps": config.get("max_spread_bps"),
                    "fee_bps": config.get("fee_bps"),
                    "slippage_bps": config.get("slippage_bps"),
                    "source": config.get("source"),
                    "exclude_hours": row.get("exclude_hours", ""),
                    "include_hours": row.get("include_hours", ""),
                    "max_cost_bps": row.get("max_cost_bps"),
                    "paper_score": row.get("paper_score"),
                    "positive_after_costs": row.get("positive_after_costs"),
                    "beats_baseline_accuracy": row.get("beats_baseline_accuracy"),
                    "filtered_paper_trades": row.get("filtered_paper_trades"),
                    "trades_before_regime_filters": row.get("trades_before_regime_filters"),
                    "trades_after_regime_filters": row.get("trades_after_regime_filters"),
                    "trades_removed_by_regime_filters": row.get("trades_removed_by_regime_filters"),
                    "regime_filters_enabled": row.get("regime_filters_enabled"),
                    "exploratory_overfit_warning": row.get("exploratory_overfit_warning"),
                    "average_net_return_bps": row.get("average_net_return_bps"),
                    "cumulative_net_return_bps": row.get("cumulative_net_return_bps"),
                    "win_rate_after_costs": row.get("win_rate_after_costs"),
                    "stability_warning_count": row.get("stability_warning_count"),
                    "insufficient_trades": row.get("insufficient_trades"),
                    "real_execution_ready": row.get("real_execution_ready"),
                }
            )

    report["csv_report_path"] = str(output_path)
    return output_path


def write_grid_report_markdown(
    report: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write timestamped Markdown grid report under reports/candidate_grid/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _report_stamp(str(report.get("run_timestamp", "")))
    output_path = output_dir / f"candidate_grid_{stamp}.md"

    ranked = report.get("ranked") or []
    top_n = report.get("top_n") or []
    failures = report.get("failures") or []
    best = report.get("best_result")

    lines = [
        "# Candidate Grid Report",
        "",
        f"Run timestamp: {report.get('run_timestamp', 'n/a')}",
        f"Dry run: {str(report.get('dry_run', False)).lower()}",
        f"Planned: {report.get('planned_count', 0)} "
        f"(total combinations={report.get('total_combinations', 0)}, "
        f"completed={report.get('completed_count', 0)}, "
        f"failed={report.get('failed_count', 0)})",
        "",
        "## Safety",
        "",
        f"- paper_only: {str(report.get('paper_only', True)).lower()}",
        f"- execution_enabled: {str(report.get('execution_enabled', False)).lower()}",
        f"- real_execution_ready: {str(report.get('real_execution_ready', False)).lower()}",
        "",
        report.get("safety_note", SAFETY_NOTE),
        "",
        "## Top Results",
        "",
        "| Rank | Config | Exploratory | Paper Score | Trades | Avg Net (bps) | Positive | Beats Baseline |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]

    for index, row in enumerate(top_n, start=1):
        exploratory = "yes" if row.get("exploratory_overfit_warning") else "no"
        lines.append(
            f"| {index} | {row.get('config_id', 'n/a')} | {exploratory} | "
            f"{float(row.get('paper_score', 0.0)):.2f} | "
            f"{row.get('filtered_paper_trades', 0)} | "
            f"{float(row.get('average_net_return_bps', 0.0)):.4f} | "
            f"{str(row.get('positive_after_costs', False)).lower()} | "
            f"{str(row.get('beats_baseline_accuracy', False)).lower()} |"
        )

    if not top_n:
        lines.append("| - | - | - | - | - | - | - | - |")

    lines.extend(["", "## Best Result", ""])
    if best is None:
        lines.append("No successful evaluations.")
    else:
        config = best.get("config") or {}
        lines.extend(
            [
                f"- config_id: {best.get('config_id')}",
                f"- coin: {config.get('coin')}",
                f"- horizon: {config.get('horizon')}",
                f"- trade_class: {config.get('trade_class')}",
                f"- min_confidence: {config.get('min_confidence')}",
                f"- min_edge_bps: {config.get('min_edge_bps')}",
                f"- exclude_hours: {best.get('exclude_hours') or 'none'}",
                f"- include_hours: {best.get('include_hours') or 'none'}",
                f"- max_cost_bps: {best.get('max_cost_bps') if best.get('max_cost_bps') is not None else 'none'}",
                f"- exploratory_overfit_warning: {str(best.get('exploratory_overfit_warning', False)).lower()}",
                f"- paper_score: {float(best.get('paper_score', 0.0)):.2f}",
                f"- filtered_paper_trades: {best.get('filtered_paper_trades', 0)}",
                f"- average_net_return_bps: {float(best.get('average_net_return_bps', 0.0)):.4f}",
                f"- real_execution_ready: false",
            ]
        )
        if best.get("exploratory_overfit_warning"):
            lines.extend(
                [
                    "",
                    "**EXPLORATORY REGIME FILTER WARNING:** This configuration uses "
                    "post-hoc regime filters (hour exclusions, cost caps, etc.). "
                    "Results are hypothesis-generating only and are overfit-prone. "
                    "Do not treat as validated walk-forward evidence.",
                ]
            )

    lines.extend(["", "## Failed Evaluations", ""])
    if not failures:
        lines.append("None.")
    else:
        for failure in failures:
            lines.append(f"- {failure.get('config_id')}: {failure.get('error')}")

    lines.extend(
        [
            "",
            "## Research Interpretation",
            "",
            "Grid results rank paper-only walk-forward configurations on local CSV data. "
            "Higher paper_score suggests stronger simulated performance after costs, "
            "but does not prove live profitability.",
            "",
            "Configurations marked **Exploratory** apply regime filters (hour exclusions, "
            "include-hour whitelists, max-cost caps). These are post-hoc hypotheses and "
            "carry high overfitting risk. Treat them as research leads, not validated edges.",
            "",
            report.get("safety_statement", SAFETY_STATEMENT),
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["markdown_report_path"] = str(output_path)
    return output_path


def format_top_n_summary(ranked: list[dict[str, Any]], top_n: int) -> str:
    """Format top-N ranked configs for console output."""
    lines = [f"Top {min(top_n, len(ranked))} configs by paper_score:"]
    for index, row in enumerate(ranked[:top_n], start=1):
        lines.append(
            f"{index}. {row.get('config_id')} "
            f"score={float(row.get('paper_score', 0.0)):.2f} "
            f"trades={row.get('filtered_paper_trades', 0)} "
            f"avg_net_bps={float(row.get('average_net_return_bps', 0.0)):.4f}"
        )
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-only candidate configuration grid runner "
            "(research-only, no real execution)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run or dry-run the candidate configuration grid",
    )
    run_parser.add_argument(
        "--source",
        choices=SOURCE_CHOICES,
        default=DEFAULT_SOURCE,
        help=f"Bid/ask CSV source (default: {DEFAULT_SOURCE})",
    )
    run_parser.add_argument(
        "--coins",
        default=",".join(DEFAULT_COINS),
        help=f"Comma-separated coins (default: {','.join(DEFAULT_COINS)})",
    )
    run_parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL,
        help=f"Resampled interval (default: {DEFAULT_INTERVAL})",
    )
    run_parser.add_argument(
        "--horizons",
        default=",".join(str(item) for item in DEFAULT_HORIZONS),
        help="Comma-separated label horizons",
    )
    run_parser.add_argument(
        "--feature-sets",
        default=",".join(DEFAULT_FEATURE_SETS),
        help="Comma-separated feature sets",
    )
    run_parser.add_argument(
        "--label-modes",
        default=",".join(DEFAULT_LABEL_MODES),
        help="Comma-separated label modes",
    )
    run_parser.add_argument(
        "--trade-classes",
        default=",".join(DEFAULT_TRADE_CLASSES),
        help="Comma-separated trade classes for grid",
    )
    run_parser.add_argument(
        "--min-confidences",
        default=",".join(str(item) for item in DEFAULT_MIN_CONFIDENCES),
        help="Comma-separated minimum confidence values",
    )
    run_parser.add_argument(
        "--min-edge-bps-values",
        default=",".join(str(item) for item in DEFAULT_MIN_EDGE_BPS_VALUES),
        help="Comma-separated minimum edge bps values",
    )
    run_parser.add_argument(
        "--cost-multipliers",
        default=",".join(str(item) for item in DEFAULT_COST_MULTIPLIERS),
        help="Comma-separated cost multipliers",
    )
    run_parser.add_argument(
        "--max-spread-bps-values",
        default=",".join(str(item) for item in DEFAULT_MAX_SPREAD_BPS_VALUES),
        help="Comma-separated max spread bps values",
    )
    run_parser.add_argument(
        "--exclude-hours-values",
        default=",".join(DEFAULT_EXCLUDE_HOURS_VALUES),
        help=(
            "Comma-separated exclude-hour filter values "
            "(none, single hour like 5, or combined like 3+5)"
        ),
    )
    run_parser.add_argument(
        "--include-hours-values",
        default=",".join(DEFAULT_INCLUDE_HOURS_VALUES),
        help=(
            "Comma-separated include-hour filter values "
            "(none, single hour like 5, or combined like 3+5)"
        ),
    )
    run_parser.add_argument(
        "--max-cost-bps-values",
        default=",".join(DEFAULT_MAX_COST_BPS_VALUES),
        help="Comma-separated max cost bps filter values (none or numeric)",
    )
    run_parser.add_argument(
        "--min-trades-after-regime-filter",
        type=int,
        default=DEFAULT_MIN_TRADES_AFTER_REGIME_FILTER,
        help=(
            "Minimum trades after regime filter before paper_score penalty "
            f"(default: {DEFAULT_MIN_TRADES_AFTER_REGIME_FILTER})"
        ),
    )
    run_parser.add_argument(
        "--fee-bps",
        type=float,
        default=DEFAULT_FEE_BPS,
        help=f"Fee bps for all configs (default: {DEFAULT_FEE_BPS})",
    )
    run_parser.add_argument(
        "--slippage-bps",
        type=float,
        default=DEFAULT_SLIPPAGE_BPS,
        help=f"Slippage bps for all configs (default: {DEFAULT_SLIPPAGE_BPS})",
    )
    run_parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of top configs to print (default: {DEFAULT_TOP_N})",
    )
    run_parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional safety cap on number of configs to evaluate",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for grid reports (default: {DEFAULT_OUTPUT_DIR})",
    )
    run_parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing bid/ask CSV files",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned count and sample configs without evaluation",
    )
    run_parser.add_argument(
        "--use-cli-fallback",
        action="store_true",
        help="Optional: invoke walk-forward via external CLI (disabled by default)",
    )

    return parser


def _resolve_grid_from_args(args: argparse.Namespace) -> tuple[list[CandidateGridConfig], dict[str, Any]]:
    grid_params = {
        "source": args.source,
        "coins": parse_csv_values(args.coins, value_type=str),
        "interval": args.interval,
        "horizons": parse_csv_values(args.horizons, value_type=int),
        "feature_sets": parse_csv_values(args.feature_sets, value_type=str),
        "label_modes": parse_csv_values(args.label_modes, value_type=str),
        "trade_classes": parse_csv_values(args.trade_classes, value_type=str),
        "min_confidences": parse_csv_values(args.min_confidences, value_type=float),
        "min_edge_bps_values": parse_csv_values(args.min_edge_bps_values, value_type=float),
        "cost_multipliers": parse_csv_values(args.cost_multipliers, value_type=float),
        "max_spread_bps_values": parse_csv_values(args.max_spread_bps_values, value_type=float),
        "exclude_hours_values": parse_regime_filter_csv_values(args.exclude_hours_values),
        "include_hours_values": parse_regime_filter_csv_values(args.include_hours_values),
        "max_cost_bps_values": parse_regime_filter_csv_values(args.max_cost_bps_values),
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "max_runs": args.max_runs,
        "top_n": args.top_n,
        "dry_run": args.dry_run,
        "min_trades_after_regime_filter": args.min_trades_after_regime_filter,
    }
    configs = build_grid_combinations(
        coins=tuple(str(item).upper() for item in grid_params["coins"]),
        interval=str(grid_params["interval"]),
        horizons=tuple(int(item) for item in grid_params["horizons"]),
        feature_sets=tuple(str(item) for item in grid_params["feature_sets"]),
        label_modes=tuple(str(item) for item in grid_params["label_modes"]),
        trade_classes=tuple(str(item).upper() for item in grid_params["trade_classes"]),
        min_confidences=tuple(float(item) for item in grid_params["min_confidences"]),
        min_edge_bps_values=tuple(float(item) for item in grid_params["min_edge_bps_values"]),
        cost_multipliers=tuple(float(item) for item in grid_params["cost_multipliers"]),
        max_spread_bps_values=tuple(float(item) for item in grid_params["max_spread_bps_values"]),
        exclude_hours_values=tuple(str(item) for item in grid_params["exclude_hours_values"]),
        include_hours_values=tuple(str(item) for item in grid_params["include_hours_values"]),
        max_cost_bps_values=tuple(str(item) for item in grid_params["max_cost_bps_values"]),
        fee_bps=float(grid_params["fee_bps"]),
        slippage_bps=float(grid_params["slippage_bps"]),
        source=str(grid_params["source"]),
    )
    return configs, grid_params


def _resolve_walk_forward_runner(use_cli_fallback: bool) -> WalkForwardRunner:
    if not use_cli_fallback:
        return run_walk_forward_evaluation
    raise NotImplementedError(
        "CLI fallback is optional and not enabled by default; "
        "use direct run_walk_forward_evaluation import instead."
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.command == "run":
        configs, grid_params = _resolve_grid_from_args(args)
        run_timestamp = _utc_now_iso()
        runner = _resolve_walk_forward_runner(args.use_cli_fallback)
        grid_output = run_candidate_grid(
            configs,
            data_root=args.data_root,
            max_runs=args.max_runs,
            dry_run=args.dry_run,
            min_trades_after_regime_filter=args.min_trades_after_regime_filter,
            walk_forward_runner=runner,
        )
        report = build_grid_report(
            grid_output=grid_output,
            run_timestamp=run_timestamp,
            grid_params=grid_params,
            top_n=args.top_n,
        )

        if args.dry_run:
            print(f"dry_run=true planned_count={grid_output['planned_count']}")
            print(f"total_combinations={grid_output['total_combinations']}")
            print("sample_configs:")
            for sample in grid_output.get("sample_configs") or []:
                print(f"  - {sample}")
            return 0

        write_grid_report_json(report, output_dir=args.output_dir)
        write_grid_report_csv(report, output_dir=args.output_dir)
        write_grid_report_markdown(report, output_dir=args.output_dir)

        print(
            f"planned={grid_output['planned_count']} "
            f"completed={grid_output['completed_count']} "
            f"failed={grid_output['failed_count']}"
        )
        print(format_top_n_summary(report.get("ranked") or [], args.top_n))
        if report.get("json_report_path"):
            print(f"json_report_path={report['json_report_path']}")
        if report.get("csv_report_path"):
            print(f"csv_report_path={report['csv_report_path']}")
        if report.get("markdown_report_path"):
            print(f"markdown_report_path={report['markdown_report_path']}")
        print("Paper-only. No trades placed.")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
