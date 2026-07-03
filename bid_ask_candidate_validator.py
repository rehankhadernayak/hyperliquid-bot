"""Daily paper-only candidate validation runner for walk-forward models.

Research-only and paper-only. No orders, wallets, or network calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT
from bid_ask_train import load_training_data
from bid_ask_walk_forward_evaluator import (
    DEFAULT_DATA_ROOT,
    DEFAULT_REPORTS_DIR,
    parse_trade_classes,
    run_walk_forward_evaluation,
)
import bid_ask_walk_forward_evaluator as walk_forward_module

SAFETY_NOTE = (
    "Candidate validation is research-only and paper-only. "
    "Simulated returns do not prove profitability. No real orders are placed."
)

DEFAULT_CANDIDATE_NAME = "BTC_DOWN_MICRO_H20_CANDIDATE_B"
DEFAULT_VALIDATION_REPORTS_DIR = Path("reports/bid_ask_candidate_validation")
DEFAULT_WALK_FORWARD_REPORTS_DIR = DEFAULT_REPORTS_DIR

DEFAULT_COIN = "BTC"
DEFAULT_INTERVAL = "15s"
DEFAULT_HORIZON = 20
DEFAULT_FEATURE_SET = "microstructure"
DEFAULT_LABEL_MODE = "direction"
DEFAULT_TRADE_CLASSES = "DOWN"
DEFAULT_MIN_CONFIDENCE = 0.65
DEFAULT_MIN_EDGE_BPS = 2.0
DEFAULT_COST_MULTIPLIER = 1.25
DEFAULT_MAX_SPREAD_BPS = 8.0
DEFAULT_FEE_BPS_CANDIDATE = 2.0
DEFAULT_SLIPPAGE_BPS_CANDIDATE = 1.0
DEFAULT_SOURCE = "historical"
SOURCE_CHOICES = ("historical", "combined", "local")
DEFAULT_MIN_TRADES_REQUIRED = 20
DEFAULT_REQUIRE_POSITIVE = True
TREND_RUN_LIMIT = 10
DEFAULT_SLICE_MODES = "recent_halves,recent_thirds,expanding"
SLICE_MODE_CHOICES = ("recent_halves", "recent_thirds", "expanding")
ROBUSTNESS_REPORT_PREFIX = "robustness_"

VALIDATION_REPORT_GLOB = "candidate_validation_*.json"

DEFAULT_LAUNCHD_LABEL = "com.hyperliquid.research.candidate-validator.daily"
DEFAULT_LAUNCHD_HOUR = 23
DEFAULT_LAUNCHD_MINUTE = 30
DEFAULT_LAUNCHD_STDOUT_LOG = DEFAULT_VALIDATION_REPORTS_DIR / "daily_validator.out.log"
DEFAULT_LAUNCHD_STDERR_LOG = DEFAULT_VALIDATION_REPORTS_DIR / "daily_validator.err.log"
LAUNCHD_SETUP_MD = "launchd_candidate_validator_setup.md"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _report_stamp(run_timestamp: str) -> str:
    if run_timestamp:
        return run_timestamp.replace(":", "").replace("+", "").replace("-", "")[:15]
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_bool(value: str) -> bool:
    """Parse a CLI true/false string."""
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}")


def build_candidate_config(
    *,
    coin: str = DEFAULT_COIN,
    interval: str = DEFAULT_INTERVAL,
    horizon: int = DEFAULT_HORIZON,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_edge_bps: float = DEFAULT_MIN_EDGE_BPS,
    cost_multiplier: float = DEFAULT_COST_MULTIPLIER,
    max_spread_bps: float = DEFAULT_MAX_SPREAD_BPS,
    min_trades_required: int = DEFAULT_MIN_TRADES_REQUIRED,
    require_positive: bool = DEFAULT_REQUIRE_POSITIVE,
    candidate_name: str = DEFAULT_CANDIDATE_NAME,
    source: str = DEFAULT_SOURCE,
) -> dict[str, Any]:
    """Return the fixed candidate configuration used for validation runs."""
    return {
        "candidate_name": candidate_name,
        "coin": coin.upper(),
        "interval": interval,
        "horizon": horizon,
        "feature_set": DEFAULT_FEATURE_SET,
        "label_mode": DEFAULT_LABEL_MODE,
        "trade_classes": list(parse_trade_classes(DEFAULT_TRADE_CLASSES)),
        "min_confidence": min_confidence,
        "min_edge_bps": min_edge_bps,
        "cost_multiplier": cost_multiplier,
        "max_spread_bps": max_spread_bps,
        "fee_bps": DEFAULT_FEE_BPS_CANDIDATE,
        "slippage_bps": DEFAULT_SLIPPAGE_BPS_CANDIDATE,
        "stability_report": True,
        "source": source,
        "min_trades_required": min_trades_required,
        "require_positive": require_positive,
    }


def compute_validation_status(
    *,
    beats_baseline_accuracy: bool,
    positive_after_costs: bool,
    filtered_paper_trades: int,
    stability_summary: dict[str, Any] | None,
    min_trades_required: int,
    require_positive: bool,
) -> dict[str, bool]:
    """Derive candidate validation status flags from walk-forward metrics."""
    stability = stability_summary or {}
    insufficient_trades = filtered_paper_trades < min_trades_required
    negative_after_costs = require_positive and not positive_after_costs
    unstable_concentrated = bool(stability.get("concentration_warning"))
    unstable_sparse = bool(stability.get("sparse_stability_warning"))
    not_market_ready = (
        insufficient_trades
        or negative_after_costs
        or unstable_concentrated
        or unstable_sparse
    )
    paper_candidate_pass = (
        beats_baseline_accuracy
        and positive_after_costs
        and not insufficient_trades
        and not unstable_concentrated
        and not unstable_sparse
    )
    return {
        "insufficient_trades": insufficient_trades,
        "negative_after_costs": negative_after_costs,
        "unstable_concentrated": unstable_concentrated,
        "unstable_sparse": unstable_sparse,
        "not_market_ready": not_market_ready,
        "paper_candidate_pass": paper_candidate_pass,
    }


def build_candidate_validation_report(
    walk_forward_report: dict[str, Any],
    *,
    config: dict[str, Any],
    run_timestamp: str,
) -> dict[str, Any]:
    """Build a candidate validation report from a walk-forward evaluation result."""
    readiness = walk_forward_report.get("readiness") or {}
    stability_summary = walk_forward_report.get("stability_summary") or {}
    filtered_paper_trades = int(walk_forward_report.get("filtered_paper_trades", 0))
    beats_baseline_accuracy = bool(
        walk_forward_report.get(
            "beats_baseline_accuracy",
            readiness.get("beats_baseline_accuracy", False),
        )
    )
    positive_after_costs = bool(
        walk_forward_report.get(
            "positive_after_costs",
            readiness.get("positive_after_costs", False),
        )
    )
    validation_status = compute_validation_status(
        beats_baseline_accuracy=beats_baseline_accuracy,
        positive_after_costs=positive_after_costs,
        filtered_paper_trades=filtered_paper_trades,
        stability_summary=stability_summary,
        min_trades_required=int(config.get("min_trades_required", DEFAULT_MIN_TRADES_REQUIRED)),
        require_positive=bool(config.get("require_positive", DEFAULT_REQUIRE_POSITIVE)),
    )

    return {
        "candidate_name": config.get("candidate_name", DEFAULT_CANDIDATE_NAME),
        "run_timestamp": run_timestamp,
        "config": config,
        "latest_walk_forward_report": walk_forward_report.get("json_report_path"),
        "accepted_trades_csv_path": walk_forward_report.get("accepted_trades_csv_path"),
        "beats_baseline_accuracy": beats_baseline_accuracy,
        "accuracy_edge_vs_baseline": float(
            walk_forward_report.get("accuracy_edge_vs_baseline", 0.0)
        ),
        "filtered_paper_trades": filtered_paper_trades,
        "cumulative_net_return_bps": float(
            walk_forward_report.get("cumulative_net_return_bps", 0.0)
        ),
        "average_net_return_bps": float(walk_forward_report.get("average_net_return_bps", 0.0)),
        "win_rate_after_costs": float(walk_forward_report.get("win_rate_after_costs", 0.0)),
        "positive_after_costs": positive_after_costs,
        "stability_summary": stability_summary,
        "stability_warnings": list(walk_forward_report.get("stability_warnings") or []),
        "ready_for_more_research": bool(
            walk_forward_report.get(
                "ready_for_more_research",
                readiness.get("ready_for_more_research", False),
            )
        ),
        "real_execution_ready": False,
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "validation_status": validation_status,
        "safety_note": SAFETY_NOTE,
        "safety_statement": SAFETY_STATEMENT,
    }


def write_candidate_validation_json(
    report: dict[str, Any],
    reports_dir: Path = DEFAULT_VALIDATION_REPORTS_DIR,
) -> Path:
    """Write timestamped JSON candidate validation report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    candidate_name = str(report.get("candidate_name", DEFAULT_CANDIDATE_NAME))
    stamp = _report_stamp(str(report.get("run_timestamp", "")))
    output_path = reports_dir / f"candidate_validation_{candidate_name}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    report["json_report_path"] = str(output_path)
    return output_path


def write_candidate_validation_markdown(
    report: dict[str, Any],
    reports_dir: Path = DEFAULT_VALIDATION_REPORTS_DIR,
) -> Path:
    """Write timestamped Markdown candidate validation report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    candidate_name = str(report.get("candidate_name", DEFAULT_CANDIDATE_NAME))
    stamp = _report_stamp(str(report.get("run_timestamp", "")))
    output_path = reports_dir / f"candidate_validation_{candidate_name}_{stamp}.md"

    config = report.get("config") or {}
    status = report.get("validation_status") or {}
    stability = report.get("stability_summary") or {}

    lines = [
        "# Candidate Validation Report",
        "",
        f"Candidate: {candidate_name}",
        f"Run timestamp: {report.get('run_timestamp', 'n/a')}",
        "",
        "## Safety",
        "",
        f"- paper_only: {str(report.get('paper_only', True)).lower()}",
        f"- execution_enabled: {str(report.get('execution_enabled', False)).lower()}",
        f"- real_execution_ready: {str(report.get('real_execution_ready', False)).lower()}",
        "",
        report.get("safety_note", SAFETY_NOTE),
        "",
        "## Configuration",
        "",
        f"- coin: {config.get('coin')}",
        f"- interval: {config.get('interval')}",
        f"- horizon: {config.get('horizon')}",
        f"- feature_set: {config.get('feature_set')}",
        f"- label_mode: {config.get('label_mode')}",
        f"- trade_classes: {', '.join(config.get('trade_classes') or [])}",
        f"- min_confidence: {config.get('min_confidence')}",
        f"- min_edge_bps: {config.get('min_edge_bps')}",
        f"- cost_multiplier: {config.get('cost_multiplier')}",
        f"- max_spread_bps: {config.get('max_spread_bps')}",
        f"- fee_bps: {config.get('fee_bps')}",
        f"- slippage_bps: {config.get('slippage_bps')}",
        f"- source: {config.get('source')}",
        f"- min_trades_required: {config.get('min_trades_required')}",
        f"- require_positive: {str(config.get('require_positive', True)).lower()}",
        "",
        "## Walk-Forward Link",
        "",
        f"- latest_walk_forward_report: {report.get('latest_walk_forward_report', 'n/a')}",
        f"- accepted_trades_csv_path: {report.get('accepted_trades_csv_path', 'n/a')}",
        "",
        "## Metrics",
        "",
        f"- beats_baseline_accuracy: {str(report.get('beats_baseline_accuracy', False)).lower()}",
        f"- accuracy_edge_vs_baseline: {report.get('accuracy_edge_vs_baseline', 0.0):.4f}",
        f"- filtered_paper_trades: {report.get('filtered_paper_trades', 0)}",
        f"- cumulative_net_return_bps: {report.get('cumulative_net_return_bps', 0.0):.4f}",
        f"- average_net_return_bps: {report.get('average_net_return_bps', 0.0):.4f}",
        f"- win_rate_after_costs: {report.get('win_rate_after_costs', 0.0):.4f}",
        f"- positive_after_costs: {str(report.get('positive_after_costs', False)).lower()}",
        f"- ready_for_more_research: {str(report.get('ready_for_more_research', False)).lower()}",
        "",
        "## Stability",
        "",
        f"- concentration_warning: {str(stability.get('concentration_warning', False)).lower()}",
        f"- sparse_stability_warning: {str(stability.get('sparse_stability_warning', False)).lower()}",
        f"- buckets_with_too_few_trades: {stability.get('buckets_with_too_few_trades', 0)}",
        "",
    ]

    warnings = report.get("stability_warnings") or []
    if warnings:
        lines.append("### Stability warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.extend(
        [
            "## Validation Status",
            "",
            f"- insufficient_trades: {str(status.get('insufficient_trades', False)).lower()}",
            f"- negative_after_costs: {str(status.get('negative_after_costs', False)).lower()}",
            f"- unstable_concentrated: {str(status.get('unstable_concentrated', False)).lower()}",
            f"- unstable_sparse: {str(status.get('unstable_sparse', False)).lower()}",
            f"- not_market_ready: {str(status.get('not_market_ready', False)).lower()}",
            f"- paper_candidate_pass: {str(status.get('paper_candidate_pass', False)).lower()}",
            "",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["markdown_report_path"] = str(output_path)
    return output_path


def run_candidate_validation(
    *,
    coin: str = DEFAULT_COIN,
    interval: str = DEFAULT_INTERVAL,
    horizon: int = DEFAULT_HORIZON,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_edge_bps: float = DEFAULT_MIN_EDGE_BPS,
    cost_multiplier: float = DEFAULT_COST_MULTIPLIER,
    max_spread_bps: float = DEFAULT_MAX_SPREAD_BPS,
    min_trades_required: int = DEFAULT_MIN_TRADES_REQUIRED,
    require_positive: bool = DEFAULT_REQUIRE_POSITIVE,
    candidate_name: str = DEFAULT_CANDIDATE_NAME,
    source: str = DEFAULT_SOURCE,
    data_root: Path = DEFAULT_DATA_ROOT,
    walk_forward_reports_dir: Path = DEFAULT_WALK_FORWARD_REPORTS_DIR,
    validation_reports_dir: Path = DEFAULT_VALIDATION_REPORTS_DIR,
    run_timestamp: str | None = None,
    walk_forward_runner=run_walk_forward_evaluation,
) -> dict[str, Any]:
    """Run walk-forward evaluation for the configured candidate and write validation reports."""
    timestamp = run_timestamp or _utc_now_iso()
    config = build_candidate_config(
        coin=coin,
        interval=interval,
        horizon=horizon,
        min_confidence=min_confidence,
        min_edge_bps=min_edge_bps,
        cost_multiplier=cost_multiplier,
        max_spread_bps=max_spread_bps,
        min_trades_required=min_trades_required,
        require_positive=require_positive,
        candidate_name=candidate_name,
        source=source,
    )

    walk_forward_report = walk_forward_runner(
        coins=[config["coin"]],
        interval=config["interval"],
        source=config["source"],
        horizon=config["horizon"],
        feature_set=config["feature_set"],
        label_mode=config["label_mode"],
        trade_classes=frozenset(config["trade_classes"]),
        min_confidence=config["min_confidence"],
        min_edge_bps=config["min_edge_bps"],
        cost_multiplier=config["cost_multiplier"],
        max_spread_bps=config["max_spread_bps"],
        fee_bps=config["fee_bps"],
        slippage_bps=config["slippage_bps"],
        data_root=data_root,
        reports_dir=walk_forward_reports_dir,
        save_reports=True,
        save_accepted_trades=True,
        stability_report_enabled=config["stability_report"],
    )

    report = build_candidate_validation_report(
        walk_forward_report,
        config=config,
        run_timestamp=timestamp,
    )
    write_candidate_validation_json(report, reports_dir=validation_reports_dir)
    write_candidate_validation_markdown(report, reports_dir=validation_reports_dir)
    return report


def parse_slice_modes(slices: str) -> list[str]:
    """Parse comma-separated rolling slice mode names."""
    modes = [item.strip() for item in str(slices).split(",") if item.strip()]
    if not modes:
        raise ValueError("At least one slice mode is required")
    invalid = [mode for mode in modes if mode not in SLICE_MODE_CHOICES]
    if invalid:
        supported = ", ".join(SLICE_MODE_CHOICES)
        raise ValueError(f"Unsupported slice mode(s) {invalid}; use one of: {supported}")
    return modes


def build_row_slices(total_rows: int, modes: list[str]) -> list[dict[str, Any]]:
    """Build chronological row-range slices for rolling robustness checks."""
    if total_rows <= 0:
        return []

    slices: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    def add_slice(slice_name: str, row_start: int, row_end: int) -> None:
        if row_end <= row_start:
            return
        key = (row_start, row_end)
        if key in seen:
            return
        seen.add(key)
        slices.append(
            {
                "slice_name": slice_name,
                "row_start": row_start,
                "row_end": row_end,
                "row_count": row_end - row_start,
            }
        )

    if "recent_halves" in modes:
        midpoint = total_rows // 2
        add_slice("half_first", 0, midpoint)
        add_slice("half_second", midpoint, total_rows)

    if "recent_thirds" in modes:
        third = total_rows // 3
        two_thirds = (total_rows * 2) // 3
        add_slice("third_first", 0, third)
        add_slice("third_middle", third, two_thirds)
        add_slice("third_last", two_thirds, total_rows)

    if "expanding" in modes:
        third = max(total_rows // 3, 1)
        two_thirds = max((total_rows * 2) // 3, third)
        add_slice("expand_33", 0, third)
        add_slice("expand_66", 0, two_thirds)
        add_slice("expand_100", 0, total_rows)

    return slices


def make_sliced_walk_forward_runner(
    *,
    row_start: int,
    row_end: int,
    base_runner=run_walk_forward_evaluation,
):
    """Return a walk-forward runner that evaluates only a row slice of the CSV."""

    def runner(**kwargs: Any) -> dict[str, Any]:
        coin = kwargs["coins"][0]
        interval = kwargs["interval"]
        source = kwargs["source"]
        data_root = kwargs["data_root"]
        original_loader = walk_forward_module.load_training_data

        def patched_loader(
            load_coin: str,
            load_interval: str,
            load_data_root: Path,
            *,
            source: str,
        ):
            frame = original_loader(
                load_coin,
                load_interval,
                load_data_root,
                source=source,
            )
            return frame.iloc[row_start:row_end].reset_index(drop=True)

        walk_forward_module.load_training_data = patched_loader
        try:
            return base_runner(**kwargs)
        finally:
            walk_forward_module.load_training_data = original_loader

    return runner


def build_slice_validation_result(
    walk_forward_report: dict[str, Any],
    *,
    config: dict[str, Any],
    slice_name: str,
    slice_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one rolling-slice validation result from walk-forward metrics."""
    stability_summary = walk_forward_report.get("stability_summary") or {}
    filtered_paper_trades = int(walk_forward_report.get("filtered_paper_trades", 0))
    beats_baseline_accuracy = bool(
        walk_forward_report.get(
            "beats_baseline_accuracy",
            (walk_forward_report.get("readiness") or {}).get("beats_baseline_accuracy", False),
        )
    )
    positive_after_costs = bool(
        walk_forward_report.get(
            "positive_after_costs",
            (walk_forward_report.get("readiness") or {}).get("positive_after_costs", False),
        )
    )
    validation_status = compute_validation_status(
        beats_baseline_accuracy=beats_baseline_accuracy,
        positive_after_costs=positive_after_costs,
        filtered_paper_trades=filtered_paper_trades,
        stability_summary=stability_summary,
        min_trades_required=int(config.get("min_trades_required", DEFAULT_MIN_TRADES_REQUIRED)),
        require_positive=bool(config.get("require_positive", DEFAULT_REQUIRE_POSITIVE)),
    )
    result = {
        "slice_name": slice_name,
        "config": config,
        "beats_baseline_accuracy": beats_baseline_accuracy,
        "accuracy_edge_vs_baseline": float(
            walk_forward_report.get("accuracy_edge_vs_baseline", 0.0)
        ),
        "filtered_paper_trades": filtered_paper_trades,
        "cumulative_net_return_bps": float(
            walk_forward_report.get("cumulative_net_return_bps", 0.0)
        ),
        "average_net_return_bps": float(walk_forward_report.get("average_net_return_bps", 0.0)),
        "win_rate_after_costs": float(walk_forward_report.get("win_rate_after_costs", 0.0)),
        "positive_after_costs": positive_after_costs,
        "stability_summary": stability_summary,
        "stability_warnings": list(walk_forward_report.get("stability_warnings") or []),
        "validation_status": validation_status,
    }
    if slice_meta:
        result["slice_meta"] = slice_meta
    return result


def build_slice_result_from_validation_report(
    report: dict[str, Any],
    *,
    slice_name: str,
) -> dict[str, Any]:
    """Map a saved candidate validation report into a rolling slice result."""
    config = dict(report.get("config") or {})
    stability_summary = report.get("stability_summary") or {}
    filtered_paper_trades = int(report.get("filtered_paper_trades", 0))
    beats_baseline_accuracy = bool(report.get("beats_baseline_accuracy", False))
    positive_after_costs = bool(report.get("positive_after_costs", False))
    validation_status = compute_validation_status(
        beats_baseline_accuracy=beats_baseline_accuracy,
        positive_after_costs=positive_after_costs,
        filtered_paper_trades=filtered_paper_trades,
        stability_summary=stability_summary,
        min_trades_required=int(config.get("min_trades_required", DEFAULT_MIN_TRADES_REQUIRED)),
        require_positive=bool(config.get("require_positive", DEFAULT_REQUIRE_POSITIVE)),
    )
    return {
        "slice_name": slice_name,
        "config": config,
        "beats_baseline_accuracy": beats_baseline_accuracy,
        "accuracy_edge_vs_baseline": float(report.get("accuracy_edge_vs_baseline", 0.0)),
        "filtered_paper_trades": filtered_paper_trades,
        "cumulative_net_return_bps": float(report.get("cumulative_net_return_bps", 0.0)),
        "average_net_return_bps": float(report.get("average_net_return_bps", 0.0)),
        "win_rate_after_costs": float(report.get("win_rate_after_costs", 0.0)),
        "positive_after_costs": positive_after_costs,
        "stability_summary": stability_summary,
        "stability_warnings": list(report.get("stability_warnings") or []),
        "validation_status": validation_status,
        "source_report_path": report.get("json_report_path"),
        "source_run_timestamp": report.get("run_timestamp"),
    }


def compute_robustness_summary(slice_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate rolling slice outcomes into top-level robustness flags."""
    slices_tested = len(slice_results)
    slices_positive_after_costs = sum(
        1 for item in slice_results if item.get("positive_after_costs")
    )
    slices_beating_baseline = sum(
        1 for item in slice_results if item.get("beats_baseline_accuracy")
    )
    slices_with_enough_trades = sum(
        1
        for item in slice_results
        if not (item.get("validation_status") or {}).get("insufficient_trades")
    )
    slices_without_stability_warnings = sum(
        1
        for item in slice_results
        if not (item.get("validation_status") or {}).get("unstable_concentrated")
        and not (item.get("validation_status") or {}).get("unstable_sparse")
    )
    robustness_pass = (
        slices_tested > 0
        and slices_positive_after_costs == slices_tested
        and slices_beating_baseline == slices_tested
        and slices_with_enough_trades == slices_tested
        and slices_without_stability_warnings == slices_tested
    )
    return {
        "slices_tested": slices_tested,
        "slices_positive_after_costs": slices_positive_after_costs,
        "slices_beating_baseline": slices_beating_baseline,
        "slices_with_enough_trades": slices_with_enough_trades,
        "slices_without_stability_warnings": slices_without_stability_warnings,
        "robustness_pass": robustness_pass,
        "not_market_ready": not robustness_pass,
        "real_execution_ready": False,
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
    }


def build_robustness_report(
    *,
    candidate_name: str,
    run_timestamp: str,
    config: dict[str, Any],
    slice_results: list[dict[str, Any]],
    slice_modes: list[str],
    evaluation_mode: str,
) -> dict[str, Any]:
    """Build the top-level rolling robustness validation report."""
    summary = compute_robustness_summary(slice_results)
    return {
        "candidate_name": candidate_name,
        "run_timestamp": run_timestamp,
        "evaluation_mode": evaluation_mode,
        "slice_modes": slice_modes,
        "config": config,
        "slices": slice_results,
        "robustness_summary": summary,
        "robustness_pass": summary["robustness_pass"],
        "not_market_ready": summary["not_market_ready"],
        "real_execution_ready": False,
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "safety_note": SAFETY_NOTE,
        "safety_statement": SAFETY_STATEMENT,
    }


def write_robustness_json(
    report: dict[str, Any],
    reports_dir: Path = DEFAULT_VALIDATION_REPORTS_DIR,
) -> Path:
    """Write timestamped JSON rolling robustness report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    candidate_name = str(report.get("candidate_name", DEFAULT_CANDIDATE_NAME))
    stamp = _report_stamp(str(report.get("run_timestamp", "")))
    output_path = reports_dir / f"{ROBUSTNESS_REPORT_PREFIX}{candidate_name}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    report["json_report_path"] = str(output_path)
    return output_path


def write_robustness_markdown(
    report: dict[str, Any],
    reports_dir: Path = DEFAULT_VALIDATION_REPORTS_DIR,
) -> Path:
    """Write timestamped Markdown rolling robustness report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    candidate_name = str(report.get("candidate_name", DEFAULT_CANDIDATE_NAME))
    stamp = _report_stamp(str(report.get("run_timestamp", "")))
    output_path = reports_dir / f"{ROBUSTNESS_REPORT_PREFIX}{candidate_name}_{stamp}.md"

    config = report.get("config") or {}
    summary = report.get("robustness_summary") or {}
    lines = [
        "# Rolling Robustness Validation Report",
        "",
        f"Candidate: {candidate_name}",
        f"Run timestamp: {report.get('run_timestamp', 'n/a')}",
        f"Evaluation mode: {report.get('evaluation_mode', 'n/a')}",
        f"Slice modes: {', '.join(report.get('slice_modes') or [])}",
        "",
        "## Safety",
        "",
        f"- paper_only: {str(report.get('paper_only', True)).lower()}",
        f"- execution_enabled: {str(report.get('execution_enabled', False)).lower()}",
        f"- real_execution_ready: {str(report.get('real_execution_ready', False)).lower()}",
        "",
        report.get("safety_note", SAFETY_NOTE),
        "",
        "## Configuration",
        "",
        f"- coin: {config.get('coin')}",
        f"- interval: {config.get('interval')}",
        f"- horizon: {config.get('horizon')}",
        f"- source: {config.get('source')}",
        f"- min_trades_required: {config.get('min_trades_required')}",
        f"- require_positive: {str(config.get('require_positive', True)).lower()}",
        "",
        "## Robustness Summary",
        "",
        f"- slices_tested: {summary.get('slices_tested', 0)}",
        f"- slices_positive_after_costs: {summary.get('slices_positive_after_costs', 0)}",
        f"- slices_beating_baseline: {summary.get('slices_beating_baseline', 0)}",
        f"- slices_with_enough_trades: {summary.get('slices_with_enough_trades', 0)}",
        f"- slices_without_stability_warnings: {summary.get('slices_without_stability_warnings', 0)}",
        f"- robustness_pass: {str(summary.get('robustness_pass', False)).lower()}",
        f"- not_market_ready: {str(summary.get('not_market_ready', True)).lower()}",
        "",
        "## Slice Results",
        "",
    ]

    for item in report.get("slices") or []:
        status = item.get("validation_status") or {}
        stability = item.get("stability_summary") or {}
        lines.extend(
            [
                f"### {item.get('slice_name', 'unknown')}",
                "",
                f"- beats_baseline_accuracy: {str(item.get('beats_baseline_accuracy', False)).lower()}",
                f"- accuracy_edge_vs_baseline: {float(item.get('accuracy_edge_vs_baseline', 0.0)):.4f}",
                f"- filtered_paper_trades: {item.get('filtered_paper_trades', 0)}",
                f"- cumulative_net_return_bps: {float(item.get('cumulative_net_return_bps', 0.0)):.4f}",
                f"- average_net_return_bps: {float(item.get('average_net_return_bps', 0.0)):.4f}",
                f"- win_rate_after_costs: {float(item.get('win_rate_after_costs', 0.0)):.4f}",
                f"- positive_after_costs: {str(item.get('positive_after_costs', False)).lower()}",
                f"- concentration_warning: {str(stability.get('concentration_warning', False)).lower()}",
                f"- sparse_stability_warning: {str(stability.get('sparse_stability_warning', False)).lower()}",
                f"- paper_candidate_pass: {str(status.get('paper_candidate_pass', False)).lower()}",
                "",
            ]
        )
        warnings = item.get("stability_warnings") or []
        if warnings:
            lines.append("Stability warnings:")
            for warning in warnings:
                lines.append(f"- {warning}")
            lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["markdown_report_path"] = str(output_path)
    return output_path


def build_robustness_from_prior_reports(
    *,
    candidate_name: str,
    reports_dir: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Fallback: treat prior candidate validation reports as rolling trend slices."""
    reports = load_candidate_validation_reports(reports_dir=reports_dir)
    matching = [item for item in reports if item.get("candidate_name") == candidate_name]
    slice_results: list[dict[str, Any]] = []
    for report in matching[-TREND_RUN_LIMIT:]:
        stamp = str(report.get("run_timestamp") or "unknown")
        slice_results.append(
            build_slice_result_from_validation_report(
                report,
                slice_name=f"trend_{stamp}",
            )
        )
    return slice_results


def _resolve_total_rows(
    *,
    coin: str,
    interval: str,
    source: str,
    data_root: Path,
    total_rows: int | None,
    data_loader=load_training_data,
) -> int | None:
    """Return available CSV row count, or None when data cannot be loaded."""
    if total_rows is not None:
        return total_rows
    try:
        frame = data_loader(coin, interval, data_root, source=source)
    except (FileNotFoundError, ValueError, OSError):
        return None
    return len(frame)


def run_robustness_validation(
    *,
    coin: str = DEFAULT_COIN,
    interval: str = DEFAULT_INTERVAL,
    horizon: int = DEFAULT_HORIZON,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_edge_bps: float = DEFAULT_MIN_EDGE_BPS,
    cost_multiplier: float = DEFAULT_COST_MULTIPLIER,
    max_spread_bps: float = DEFAULT_MAX_SPREAD_BPS,
    min_trades_required: int = DEFAULT_MIN_TRADES_REQUIRED,
    require_positive: bool = DEFAULT_REQUIRE_POSITIVE,
    candidate_name: str = DEFAULT_CANDIDATE_NAME,
    source: str = DEFAULT_SOURCE,
    slice_modes: str = DEFAULT_SLICE_MODES,
    data_root: Path = DEFAULT_DATA_ROOT,
    walk_forward_reports_dir: Path = DEFAULT_WALK_FORWARD_REPORTS_DIR,
    validation_reports_dir: Path = DEFAULT_VALIDATION_REPORTS_DIR,
    run_timestamp: str | None = None,
    total_rows: int | None = None,
    walk_forward_runner=run_walk_forward_evaluation,
    data_loader=load_training_data,
) -> dict[str, Any]:
    """Run rolling robustness validation across chronological data slices."""
    timestamp = run_timestamp or _utc_now_iso()
    modes = parse_slice_modes(slice_modes)
    config = build_candidate_config(
        coin=coin,
        interval=interval,
        horizon=horizon,
        min_confidence=min_confidence,
        min_edge_bps=min_edge_bps,
        cost_multiplier=cost_multiplier,
        max_spread_bps=max_spread_bps,
        min_trades_required=min_trades_required,
        require_positive=require_positive,
        candidate_name=candidate_name,
        source=source,
    )

    resolved_rows = _resolve_total_rows(
        coin=config["coin"],
        interval=config["interval"],
        source=config["source"],
        data_root=data_root,
        total_rows=total_rows,
        data_loader=data_loader,
    )

    slice_results: list[dict[str, Any]] = []
    evaluation_mode = "data_slicing"

    if resolved_rows is not None and resolved_rows > 0:
        row_slices = build_row_slices(resolved_rows, modes)
        for row_slice in row_slices:
            sliced_runner = make_sliced_walk_forward_runner(
                row_start=int(row_slice["row_start"]),
                row_end=int(row_slice["row_end"]),
                base_runner=walk_forward_runner,
            )
            walk_forward_report = sliced_runner(
                coins=[config["coin"]],
                interval=config["interval"],
                source=config["source"],
                horizon=config["horizon"],
                feature_set=config["feature_set"],
                label_mode=config["label_mode"],
                trade_classes=frozenset(config["trade_classes"]),
                min_confidence=config["min_confidence"],
                min_edge_bps=config["min_edge_bps"],
                cost_multiplier=config["cost_multiplier"],
                max_spread_bps=config["max_spread_bps"],
                fee_bps=config["fee_bps"],
                slippage_bps=config["slippage_bps"],
                data_root=data_root,
                reports_dir=walk_forward_reports_dir,
                save_reports=False,
                save_accepted_trades=False,
                stability_report_enabled=config["stability_report"],
            )
            slice_results.append(
                build_slice_validation_result(
                    walk_forward_report,
                    config=config,
                    slice_name=str(row_slice["slice_name"]),
                    slice_meta=row_slice,
                )
            )
    else:
        evaluation_mode = "trend_fallback"
        slice_results = build_robustness_from_prior_reports(
            candidate_name=candidate_name,
            reports_dir=validation_reports_dir,
            config=config,
        )

    report = build_robustness_report(
        candidate_name=candidate_name,
        run_timestamp=timestamp,
        config=config,
        slice_results=slice_results,
        slice_modes=modes,
        evaluation_mode=evaluation_mode,
    )
    write_robustness_json(report, reports_dir=validation_reports_dir)
    write_robustness_markdown(report, reports_dir=validation_reports_dir)
    return report


def _validation_report_sort_key(report: dict[str, Any]) -> str:
    return str(report.get("run_timestamp") or report.get("json_report_path") or "")


def load_candidate_validation_reports(
    reports_dir: Path = DEFAULT_VALIDATION_REPORTS_DIR,
) -> list[dict[str, Any]]:
    """Load all candidate validation JSON reports sorted by run timestamp."""
    if not reports_dir.is_dir():
        return []

    reports: list[dict[str, Any]] = []
    for path in sorted(reports_dir.glob(VALIDATION_REPORT_GLOB)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        payload.setdefault("json_report_path", str(path))
        reports.append(payload)

    reports.sort(key=_validation_report_sort_key)
    return reports


def _trend_status_label(report: dict[str, Any]) -> str:
    status = report.get("validation_status") or {}
    if status.get("paper_candidate_pass"):
        return "paper_candidate_pass"
    return "not_market_ready"


def summarize_candidate_validation_reports(
    reports_dir: Path = DEFAULT_VALIDATION_REPORTS_DIR,
) -> str:
    """Build a human-readable summary of candidate validation reports."""
    reports = load_candidate_validation_reports(reports_dir=reports_dir)
    if not reports:
        return "No candidate validation reports found."

    latest = reports[-1]
    best = max(reports, key=lambda item: float(item.get("cumulative_net_return_bps", 0.0)))
    passes = sum(
        1 for item in reports if (item.get("validation_status") or {}).get("paper_candidate_pass")
    )
    fails = len(reports) - passes
    trend_reports = reports[-TREND_RUN_LIMIT:]

    lines = [
        "Candidate Validation Summary",
        "",
        "Latest report:",
        f"- candidate_name: {latest.get('candidate_name')}",
        f"- run_timestamp: {latest.get('run_timestamp')}",
        f"- source: {(latest.get('config') or {}).get('source', 'n/a')}",
        f"- filtered_paper_trades: {latest.get('filtered_paper_trades')}",
        f"- cumulative_net_return_bps: {latest.get('cumulative_net_return_bps', 0.0):.4f}",
        f"- average_net_return_bps: {latest.get('average_net_return_bps', 0.0):.4f}",
        f"- win_rate_after_costs: {latest.get('win_rate_after_costs', 0.0):.4f}",
        f"- accuracy_edge_vs_baseline: {latest.get('accuracy_edge_vs_baseline', 0.0):.4f}",
        f"- status: {_trend_status_label(latest)}",
        "",
        "Best cumulative_net_return_bps:",
        f"- {best.get('cumulative_net_return_bps', 0.0):.4f} "
        f"({best.get('run_timestamp')} / {best.get('candidate_name')})",
        "",
        f"Passes: {passes} | Fails: {fails}",
        "",
        f"Trend (last {len(trend_reports)} runs):",
        "timestamp | source | trades | cumulative_net | avg_net | win_rate | accuracy_edge | status",
    ]

    for item in trend_reports:
        item_source = (item.get("config") or {}).get("source", "n/a")
        lines.append(
            f"{item.get('run_timestamp')} | "
            f"{item_source} | "
            f"{item.get('filtered_paper_trades', 0)} | "
            f"{float(item.get('cumulative_net_return_bps', 0.0)):.4f} | "
            f"{float(item.get('average_net_return_bps', 0.0)):.4f} | "
            f"{float(item.get('win_rate_after_costs', 0.0)):.4f} | "
            f"{float(item.get('accuracy_edge_vs_baseline', 0.0)):.4f} | "
            f"{_trend_status_label(item)}"
        )

    return "\n".join(lines)


def _resolve_project_root(project_root: Path | None = None) -> Path:
    """Resolve repository root from explicit path or this module's location."""
    if project_root is not None:
        return project_root.resolve()
    return Path(__file__).resolve().parent


def cmd_write_launchd_template(
    *,
    hour: int = DEFAULT_LAUNCHD_HOUR,
    minute: int = DEFAULT_LAUNCHD_MINUTE,
    label: str = DEFAULT_LAUNCHD_LABEL,
    source: str = DEFAULT_SOURCE,
    run_at_load: bool = False,
    stdout_log: Path = DEFAULT_LAUNCHD_STDOUT_LOG,
    stderr_log: Path = DEFAULT_LAUNCHD_STDERR_LOG,
    reports_dir: Path = DEFAULT_VALIDATION_REPORTS_DIR,
    project_root: Path | None = None,
) -> tuple[Path, Path]:
    """Write macOS launchd plist template and setup instructions (does not install)."""
    if not 0 <= hour <= 23:
        raise ValueError("hour must be between 0 and 23")
    if not 0 <= minute <= 59:
        raise ValueError("minute must be between 0 and 59")
    if source not in SOURCE_CHOICES:
        raise ValueError(f"source must be one of {SOURCE_CHOICES}")

    root = _resolve_project_root(project_root)
    python_path = sys.executable
    validator_script = root / "bid_ask_candidate_validator.py"
    cmd_parts = [
        python_path,
        str(validator_script),
        "run",
        "--source",
        source,
    ]

    plist_name = f"{label}.plist"
    plist_path = reports_dir / plist_name
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_at_load_block = ""
    if run_at_load:
        run_at_load_block = """    <key>RunAtLoad</key>
    <true/>
"""

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
{''.join(f'        <string>{part}</string>\n' for part in cmd_parts)}    </array>
    <key>WorkingDirectory</key>
    <string>{root}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{hour}</integer>
        <key>Minute</key>
        <integer>{minute}</integer>
    </dict>
{run_at_load_block}    <key>StandardOutPath</key>
    <string>{stdout_log}</string>
    <key>StandardErrorPath</key>
    <string>{stderr_log}</string>
</dict>
</plist>
"""
    plist_path.write_text(plist_content, encoding="utf-8")

    setup_md = reports_dir / LAUNCHD_SETUP_MD
    setup_lines = [
        "# macOS launchd Setup for Daily Candidate Validator",
        "",
        "This template runs `bid_ask_candidate_validator.py run` once per day "
        f"at **{hour:02d}:{minute:02d}** local time with `--source {source}`. "
        "It is **not installed automatically**.",
        "",
        "Research-only and paper-only; no orders, wallets, or execution endpoints.",
        "",
        "## 1. Copy plist to LaunchAgents",
        "",
        "```bash",
        f"cp {plist_path} ~/Library/LaunchAgents/",
        "```",
        "",
        "## 2. Unload old plist (if present)",
        "",
        "```bash",
        f"launchctl unload ~/Library/LaunchAgents/{plist_name} 2>/dev/null || true",
        "```",
        "",
        "## 3. Load plist",
        "",
        "```bash",
        f"launchctl load ~/Library/LaunchAgents/{plist_name}",
        "```",
        "",
        "## 4. Check launchctl status",
        "",
        "```bash",
        "launchctl list | grep hyperliquid",
        "```",
        "",
        "## 5. Inspect stdout/stderr logs",
        "",
        "```bash",
        f"tail -50 {stdout_log}",
        f"tail -50 {stderr_log}",
        "```",
        "",
        "## 6. Summarize validation reports",
        "",
        "```bash",
        "python3 bid_ask_candidate_validator.py summarize",
        "```",
        "",
        "## Uninstall",
        "",
        "```bash",
        f"launchctl unload ~/Library/LaunchAgents/{plist_name}",
        f"rm ~/Library/LaunchAgents/{plist_name}",
        "```",
    ]
    setup_md.write_text("\n".join(setup_lines) + "\n", encoding="utf-8")

    return plist_path, setup_md


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coin", default=DEFAULT_COIN, help="Coin symbol (default: BTC)")
    parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL,
        help="Resampled interval (default: 15s)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"Label horizon in bars (default: {DEFAULT_HORIZON})",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=f"Minimum model confidence (default: {DEFAULT_MIN_CONFIDENCE})",
    )
    parser.add_argument(
        "--min-edge-bps",
        type=float,
        default=DEFAULT_MIN_EDGE_BPS,
        help=f"Minimum edge above costs in bps (default: {DEFAULT_MIN_EDGE_BPS})",
    )
    parser.add_argument(
        "--cost-multiplier",
        type=float,
        default=DEFAULT_COST_MULTIPLIER,
        help=f"Cost multiplier for required edge (default: {DEFAULT_COST_MULTIPLIER})",
    )
    parser.add_argument(
        "--max-spread-bps",
        type=float,
        default=DEFAULT_MAX_SPREAD_BPS,
        help=f"Maximum spread in bps (default: {DEFAULT_MAX_SPREAD_BPS})",
    )
    parser.add_argument(
        "--min-trades-required",
        type=int,
        default=DEFAULT_MIN_TRADES_REQUIRED,
        help=f"Minimum filtered paper trades to pass (default: {DEFAULT_MIN_TRADES_REQUIRED})",
    )
    parser.add_argument(
        "--require-positive",
        type=parse_bool,
        default=DEFAULT_REQUIRE_POSITIVE,
        help="Require positive cumulative net return (default: true)",
    )
    parser.add_argument(
        "--candidate-name",
        default=DEFAULT_CANDIDATE_NAME,
        help=f"Candidate label (default: {DEFAULT_CANDIDATE_NAME})",
    )
    parser.add_argument(
        "--source",
        choices=SOURCE_CHOICES,
        default=DEFAULT_SOURCE,
        help=f"Bid/ask CSV source (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing bid/ask CSV files",
    )
    parser.add_argument(
        "--walk-forward-reports-dir",
        type=Path,
        default=DEFAULT_WALK_FORWARD_REPORTS_DIR,
        help="Directory for underlying walk-forward reports",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_VALIDATION_REPORTS_DIR,
        help="Directory for candidate validation JSON/Markdown reports",
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Daily paper-only candidate validation runner for walk-forward models "
            "(research-only, paper-only)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run walk-forward validation for the configured candidate",
    )
    _add_run_arguments(run_parser)

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize saved candidate validation reports",
    )
    summarize_parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_VALIDATION_REPORTS_DIR,
        help="Directory containing candidate validation JSON reports",
    )

    robustness_parser = subparsers.add_parser(
        "robustness",
        help="Run rolling robustness validation across chronological data slices",
    )
    _add_run_arguments(robustness_parser)
    robustness_parser.add_argument(
        "--slices",
        default=DEFAULT_SLICE_MODES,
        help=(
            "Comma-separated slice modes "
            f"(default: {DEFAULT_SLICE_MODES}; choices: {', '.join(SLICE_MODE_CHOICES)})"
        ),
    )
    robustness_parser.set_defaults(walk_forward_reports_dir=DEFAULT_WALK_FORWARD_REPORTS_DIR)

    launchd_parser = subparsers.add_parser(
        "write-launchd-template",
        help="Write daily launchd plist template for candidate validation (does not install)",
    )
    launchd_parser.add_argument(
        "--hour",
        type=int,
        default=DEFAULT_LAUNCHD_HOUR,
        help=f"Daily run hour 0-23 (default: {DEFAULT_LAUNCHD_HOUR})",
    )
    launchd_parser.add_argument(
        "--minute",
        type=int,
        default=DEFAULT_LAUNCHD_MINUTE,
        help=f"Daily run minute 0-59 (default: {DEFAULT_LAUNCHD_MINUTE})",
    )
    launchd_parser.add_argument(
        "--label",
        default=DEFAULT_LAUNCHD_LABEL,
        help="launchd Label key",
    )
    launchd_parser.add_argument(
        "--source",
        choices=SOURCE_CHOICES,
        default=DEFAULT_SOURCE,
        help=f"Bid/ask CSV source passed to run (default: {DEFAULT_SOURCE})",
    )
    launchd_parser.add_argument(
        "--stdout-log",
        type=Path,
        default=DEFAULT_LAUNCHD_STDOUT_LOG,
        help="StandardOutPath for launchd",
    )
    launchd_parser.add_argument(
        "--stderr-log",
        type=Path,
        default=DEFAULT_LAUNCHD_STDERR_LOG,
        help="StandardErrorPath for launchd",
    )
    launchd_parser.add_argument(
        "--run-at-load",
        type=parse_bool,
        default=False,
        help="Run immediately on load (default: false)",
    )
    launchd_parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_VALIDATION_REPORTS_DIR,
        help="Directory for plist and setup doc",
    )
    launchd_parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repository root (default: parent of this script)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.command == "run":
        report = run_candidate_validation(
            coin=args.coin,
            interval=args.interval,
            horizon=args.horizon,
            min_confidence=args.min_confidence,
            min_edge_bps=args.min_edge_bps,
            cost_multiplier=args.cost_multiplier,
            max_spread_bps=args.max_spread_bps,
            min_trades_required=args.min_trades_required,
            require_positive=args.require_positive,
            candidate_name=args.candidate_name,
            source=args.source,
            data_root=args.data_root,
            walk_forward_reports_dir=args.walk_forward_reports_dir,
            validation_reports_dir=args.reports_dir,
        )
        status = report.get("validation_status") or {}
        print(
            f"candidate={report.get('candidate_name')} "
            f"trades={report.get('filtered_paper_trades')} "
            f"cumulative_net_return_bps={report.get('cumulative_net_return_bps', 0.0):.4f} "
            f"paper_candidate_pass={status.get('paper_candidate_pass')}"
        )
        if report.get("json_report_path"):
            print(f"json_report_path={report['json_report_path']}")
        if report.get("markdown_report_path"):
            print(f"markdown_report_path={report['markdown_report_path']}")
        if report.get("latest_walk_forward_report"):
            print(f"latest_walk_forward_report={report['latest_walk_forward_report']}")
        if report.get("accepted_trades_csv_path"):
            print(f"accepted_trades_csv_path={report['accepted_trades_csv_path']}")
        return 0

    if args.command == "summarize":
        print(summarize_candidate_validation_reports(reports_dir=args.reports_dir))
        return 0

    if args.command == "robustness":
        report = run_robustness_validation(
            coin=args.coin,
            interval=args.interval,
            horizon=args.horizon,
            min_confidence=args.min_confidence,
            min_edge_bps=args.min_edge_bps,
            cost_multiplier=args.cost_multiplier,
            max_spread_bps=args.max_spread_bps,
            min_trades_required=args.min_trades_required,
            require_positive=args.require_positive,
            candidate_name=args.candidate_name,
            source=args.source,
            slice_modes=args.slices,
            data_root=args.data_root,
            walk_forward_reports_dir=args.walk_forward_reports_dir,
            validation_reports_dir=args.reports_dir,
        )
        summary = report.get("robustness_summary") or {}
        print(
            f"candidate={report.get('candidate_name')} "
            f"slices_tested={summary.get('slices_tested', 0)} "
            f"robustness_pass={summary.get('robustness_pass')} "
            f"evaluation_mode={report.get('evaluation_mode')}"
        )
        if report.get("json_report_path"):
            print(f"json_report_path={report['json_report_path']}")
        if report.get("markdown_report_path"):
            print(f"markdown_report_path={report['markdown_report_path']}")
        return 0

    if args.command == "write-launchd-template":
        plist_path, setup_path = cmd_write_launchd_template(
            hour=args.hour,
            minute=args.minute,
            label=args.label,
            source=args.source,
            run_at_load=args.run_at_load,
            stdout_log=args.stdout_log,
            stderr_log=args.stderr_log,
            reports_dir=args.reports_dir,
            project_root=args.project_root,
        )
        print(f"Wrote launchd template: {plist_path}")
        print(f"Wrote setup doc: {setup_path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
