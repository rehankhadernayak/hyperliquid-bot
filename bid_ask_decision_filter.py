"""Offline paper-only filter for bid/ask model decision logs (research-only)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bid_ask_decision_logger import DECISION_COLUMNS, SAFETY_NOTE
from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT

DEFAULT_DECISIONS_DIR = Path("reports/bid_ask_decisions")
DEFAULT_EVALUATION_DIR = Path("reports/bid_ask_decision_evaluation")
DEFAULT_OUTPUT_DIR = Path("reports/bid_ask_decision_filters")

FILTER_STATUS_ACCEPTED = "ACCEPTED"
FILTER_STATUS_BLOCKED = "BLOCKED"

BLOCK_REASONS = (
    "spread_too_wide",
    "coin_not_allowed",
    "signal_blocked",
    "confidence_too_low",
    "missing_features",
    "model_does_not_beat_baseline",
    "sample_too_small",
)

PAPER_ONLY_DISCLAIMER = (
    "Paper-only decision filter for research review. "
    "Blocking rules do not place orders or affect live trading. "
    "No trades are placed."
)


def discover_latest_decision_log(
    decisions_dir: Path = DEFAULT_DECISIONS_DIR,
) -> Path | None:
    """Find the newest decision log CSV, or None if none exist."""
    logs = sorted(
        decisions_dir.glob("bid_ask_decisions_*.csv"),
        key=lambda path: path.name,
    )
    if not logs:
        return None
    return logs[-1]


def load_decisions(path: Path) -> pd.DataFrame:
    """Load a decision log CSV safely."""
    if not path.is_file():
        raise FileNotFoundError(f"Decision log not found: {path}")

    df = pd.read_csv(path)
    missing = [column for column in DECISION_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    if df["paper_only"].astype(str).str.lower().eq("false").any():
        raise ValueError(f"Decision log must be paper-only: {path}")
    if df["execution_enabled"].astype(str).str.lower().eq("true").any():
        raise ValueError(f"Decision log must not enable execution: {path}")

    return df


def default_filter_config() -> dict[str, Any]:
    """Return default paper-only review gate configuration."""
    return {
        "max_spread_bps": 10.0,
        "allowed_coins": ("BTC", "ETH", "SOL"),
        "blocked_signals": ("baseline_fallback",),
        "min_confidence": None,
        "require_model_beats_baseline": True,
        "min_sample_size": 100,
        "block_if_missing_features": True,
    }


def _features_missing(features_json: Any) -> bool:
    if features_json is None or (isinstance(features_json, float) and pd.isna(features_json)):
        return True
    raw = str(features_json).strip()
    if not raw or raw.lower() in {"nan", "null", "none"}:
        return True
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return True
    if not isinstance(parsed, dict) or not parsed:
        return True
    return False


def _accuracy(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.astype(bool).sum()) / len(series)


def _group_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if df.empty or column not in df.columns:
        return {}
    counts: dict[str, int] = {}
    for key, count in df[column].value_counts().items():
        counts[str(key)] = int(count)
    return counts


def apply_decision_filter(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply review gates and add filter_status and block_reason columns."""
    cfg = default_filter_config()
    if config:
        cfg.update(config)

    allowed_coins = {str(coin).upper() for coin in cfg["allowed_coins"]}
    blocked_signals = {str(signal) for signal in cfg["blocked_signals"]}
    max_spread_bps = float(cfg["max_spread_bps"])
    min_confidence = cfg["min_confidence"]
    min_sample_size = int(cfg["min_sample_size"])

    result = df.copy()
    result["filter_status"] = FILTER_STATUS_ACCEPTED
    result["block_reason"] = ""

    total = len(result)
    if total < min_sample_size:
        result["filter_status"] = FILTER_STATUS_BLOCKED
        result["block_reason"] = "sample_too_small"
        return result

    if cfg["require_model_beats_baseline"]:
        model_accuracy = _accuracy(result["signal_correct"])
        baseline_accuracy = _accuracy(result["baseline_correct"])
        if model_accuracy <= baseline_accuracy:
            result["filter_status"] = FILTER_STATUS_BLOCKED
            result["block_reason"] = "model_does_not_beat_baseline"
            return result

    for index, row in result.iterrows():
        reasons: list[str] = []

        spread_bps = float(row["spread_bps_or_avg_spread_bps"])
        if spread_bps > max_spread_bps:
            reasons.append("spread_too_wide")

        coin = str(row["coin"]).upper()
        if coin not in allowed_coins:
            reasons.append("coin_not_allowed")

        model_signal = str(row["model_signal"])
        if model_signal in blocked_signals:
            reasons.append("signal_blocked")

        if min_confidence is not None and float(row["confidence"]) < float(min_confidence):
            reasons.append("confidence_too_low")

        if cfg["block_if_missing_features"] and _features_missing(row["features_json"]):
            reasons.append("missing_features")

        if reasons:
            result.at[index, "filter_status"] = FILTER_STATUS_BLOCKED
            result.at[index, "block_reason"] = reasons[0]

    return result


def evaluate_filtered_decisions(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize filter outcomes and accuracy on accepted vs blocked rows."""
    if "filter_status" not in df.columns:
        raise ValueError("DataFrame must include filter_status from apply_decision_filter")

    total = len(df)
    accepted = df[df["filter_status"] == FILTER_STATUS_ACCEPTED]
    blocked = df[df["filter_status"] == FILTER_STATUS_BLOCKED]

    accepted_count = len(accepted)
    blocked_count = len(blocked)

    accepted_accuracy = _accuracy(accepted["signal_correct"]) if accepted_count else 0.0
    blocked_accuracy = _accuracy(blocked["signal_correct"]) if blocked_count else 0.0
    baseline_accuracy_on_accepted = (
        _accuracy(accepted["baseline_correct"]) if accepted_count else 0.0
    )
    accepted_edge_vs_baseline = accepted_accuracy - baseline_accuracy_on_accepted

    unfiltered_accuracy = _accuracy(df["signal_correct"]) if total else 0.0
    unfiltered_baseline = _accuracy(df["baseline_correct"]) if total else 0.0

    block_reason_counts: dict[str, int] = {}
    if blocked_count:
        for reason, count in blocked["block_reason"].value_counts().items():
            reason_str = str(reason)
            if reason_str:
                block_reason_counts[reason_str] = int(count)

    warnings: list[str] = [SAFETY_NOTE]

    if accepted_count == 0:
        warnings.append("No decisions passed the filter; review gate settings")
    elif accepted_count < 30:
        warnings.append(
            f"Only {accepted_count} accepted decisions; metrics may be unstable"
        )

    if accepted_count > 0 and accepted_accuracy <= baseline_accuracy_on_accepted:
        warnings.append(
            "Accepted decisions do not beat baseline accuracy; "
            "tighten filter rules or improve model"
        )

    if total > 0 and unfiltered_accuracy <= unfiltered_baseline:
        warnings.append(
            "Unfiltered model accuracy does not beat baseline on this log"
        )

    if block_reason_counts.get("signal_blocked", 0) > 0:
        warnings.append(
            "Some decisions blocked because model_signal is in blocked_signals"
        )

    return {
        "total_decisions": total,
        "accepted_count": accepted_count,
        "blocked_count": blocked_count,
        "accepted_accuracy": accepted_accuracy,
        "blocked_accuracy": blocked_accuracy,
        "baseline_accuracy_on_accepted": baseline_accuracy_on_accepted,
        "accepted_edge_vs_baseline": accepted_edge_vs_baseline,
        "unfiltered_accuracy": unfiltered_accuracy,
        "unfiltered_baseline_accuracy": unfiltered_baseline,
        "accepted_by_coin": _group_counts(accepted, "coin"),
        "accepted_by_signal": _group_counts(accepted, "model_signal"),
        "block_reason_counts": block_reason_counts,
        "warnings": warnings,
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
    }


def load_latest_evaluation(
    evaluation_dir: Path = DEFAULT_EVALUATION_DIR,
) -> dict[str, Any] | None:
    """Optionally load the newest decision evaluation JSON report."""
    if not evaluation_dir.is_dir():
        return None
    candidates = sorted(evaluation_dir.glob("bid_ask_decision_evaluation_*.json"))
    if not candidates:
        return None
    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _report_stamp(report: dict[str, Any]) -> str:
    if report.get("generated_at"):
        raw = str(report["generated_at"])
        return raw.replace(":", "").replace("+", "").replace("-", "")[:15]
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_filter_report_json(
    report: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write filter report JSON to reports/bid_ask_decision_filters/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _report_stamp(report)
    output_path = output_dir / f"bid_ask_decision_filter_{stamp}.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report["json_report_path"] = str(output_path)
    return output_path


def _format_accuracy(value: float) -> str:
    return f"{value:.4f}"


def write_filter_report_markdown(
    report: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write filter report Markdown to reports/bid_ask_decision_filters/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _report_stamp(report)
    output_path = output_dir / f"bid_ask_decision_filter_{stamp}.md"

    lines = [
        "# Paper Decision Filter Report",
        "",
        f"Generated at: {report.get('generated_at', 'n/a')}",
        f"Source log: {report.get('source_log_path', 'n/a')}",
        "",
        "## Overall filter result",
        "",
        f"- total_decisions: {report.get('total_decisions', 0)}",
        f"- accepted_count: {report.get('accepted_count', 0)}",
        f"- blocked_count: {report.get('blocked_count', 0)}",
        f"- unfiltered_accuracy: "
        f"{_format_accuracy(float(report.get('unfiltered_accuracy', 0.0)))}",
        f"- paper_only: {str(report.get('paper_only', True)).lower()}",
        f"- execution_enabled: {str(report.get('execution_enabled', False)).lower()}",
        "",
        "## Accepted vs blocked counts",
        "",
        f"Accepted: {report.get('accepted_count', 0)}",
        f"Blocked: {report.get('blocked_count', 0)}",
        "",
        "## Accuracy after filtering",
        "",
        f"- accepted_accuracy: "
        f"{_format_accuracy(float(report.get('accepted_accuracy', 0.0)))}",
        f"- blocked_accuracy: "
        f"{_format_accuracy(float(report.get('blocked_accuracy', 0.0)))}",
        "",
        "## Baseline comparison",
        "",
        f"- baseline_accuracy_on_accepted: "
        f"{_format_accuracy(float(report.get('baseline_accuracy_on_accepted', 0.0)))}",
        f"- accepted_edge_vs_baseline: "
        f"{_format_accuracy(float(report.get('accepted_edge_vs_baseline', 0.0)))}",
        f"- unfiltered_baseline_accuracy: "
        f"{_format_accuracy(float(report.get('unfiltered_baseline_accuracy', 0.0)))}",
        "",
        "## Block reasons",
        "",
    ]

    block_reason_counts = report.get("block_reason_counts", {})
    if not block_reason_counts:
        lines.append("No rows blocked.")
    else:
        lines.extend(["| Reason | Count |", "| --- | ---: |"])
        for reason, count in sorted(block_reason_counts.items()):
            lines.append(f"| {reason} | {count} |")

    lines.extend(
        [
            "",
            "## Coin breakdown (accepted)",
            "",
        ]
    )
    accepted_by_coin = report.get("accepted_by_coin", {})
    if not accepted_by_coin:
        lines.append("No accepted decisions by coin.")
    else:
        lines.extend(["| Coin | Accepted |", "| --- | ---: |"])
        for coin, count in sorted(accepted_by_coin.items()):
            lines.append(f"| {coin} | {count} |")

    lines.extend(
        [
            "",
            "## Signal breakdown (accepted)",
            "",
        ]
    )
    accepted_by_signal = report.get("accepted_by_signal", {})
    if not accepted_by_signal:
        lines.append("No accepted decisions by signal.")
    else:
        lines.extend(["| Signal | Accepted |", "| --- | ---: |"])
        for signal, count in sorted(accepted_by_signal.items()):
            lines.append(f"| {signal} | {count} |")

    filter_config = report.get("filter_config", {})
    if filter_config:
        lines.extend(
            [
                "",
                "## Filter configuration",
                "",
                f"- max_spread_bps: {filter_config.get('max_spread_bps')}",
                f"- allowed_coins: {', '.join(filter_config.get('allowed_coins', []))}",
                f"- blocked_signals: {', '.join(filter_config.get('blocked_signals', []))}",
                f"- min_confidence: {filter_config.get('min_confidence')}",
                f"- require_model_beats_baseline: "
                f"{filter_config.get('require_model_beats_baseline')}",
                f"- min_sample_size: {filter_config.get('min_sample_size')}",
                f"- block_if_missing_features: "
                f"{filter_config.get('block_if_missing_features')}",
            ]
        )

    warnings = report.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Filter metrics reflect historical paper labels, not live fills or PnL.",
            "- Global gates (sample size, model vs baseline) block entire logs when triggered.",
            "- Per-row gates use first matching block reason only.",
            "- Blocking rules are review suggestions only; they do not place orders.",
            "",
            "## Paper-only disclaimer",
            "",
            PAPER_ONLY_DISCLAIMER,
            "",
            SAFETY_STATEMENT,
            "",
            SAFETY_NOTE,
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["markdown_report_path"] = str(output_path)
    return output_path


def run_filter(
    log_path: Path,
    *,
    config: dict[str, Any] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    evaluation_dir: Path = DEFAULT_EVALUATION_DIR,
) -> dict[str, Any]:
    """Load, filter, evaluate, and write reports for one decision log."""
    df = load_decisions(log_path)
    filter_config = default_filter_config()
    if config:
        filter_config.update(config)

    filtered_df = apply_decision_filter(df, filter_config)
    report = evaluate_filtered_decisions(filtered_df)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["source_log_path"] = str(log_path)
    report["filter_config"] = {
        **filter_config,
        "allowed_coins": list(filter_config["allowed_coins"]),
        "blocked_signals": list(filter_config["blocked_signals"]),
    }

    evaluation = load_latest_evaluation(evaluation_dir)
    if evaluation:
        report["evaluation_reference"] = {
            "generated_at": evaluation.get("generated_at"),
            "model_accuracy": evaluation.get("model_accuracy"),
            "baseline_accuracy": evaluation.get("baseline_accuracy"),
            "source_log_path": evaluation.get("source_log_path"),
        }

    write_filter_report_json(report, output_dir=output_dir)
    write_filter_report_markdown(report, output_dir=output_dir)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply paper-only review gates to bid/ask model decision logs "
            "(research-only, no execution)."
        )
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Filter the newest decision log in reports/bid_ask_decisions/",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        help="Explicit path to a decision log CSV",
    )
    parser.add_argument(
        "--decisions-dir",
        type=Path,
        default=DEFAULT_DECISIONS_DIR,
        help="Directory containing decision log CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for filter JSON and Markdown reports",
    )
    parser.add_argument(
        "--max-spread-bps",
        type=float,
        default=10.0,
        help="Block decisions when spread_bps exceeds this threshold",
    )
    parser.add_argument(
        "--allow-coins",
        nargs="+",
        default=["BTC", "ETH", "SOL"],
        help="Coins allowed through the filter",
    )
    parser.add_argument(
        "--block-signal",
        action="append",
        default=[],
        dest="block_signals",
        help="Model signals to block (repeatable; default includes baseline_fallback)",
    )
    parser.add_argument(
        "--min-sample-size",
        type=int,
        default=100,
        help="Minimum log size before per-row filtering",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Optional minimum confidence threshold",
    )
    parser.add_argument(
        "--no-require-model-beats-baseline",
        action="store_true",
        help="Disable global gate requiring model accuracy > baseline",
    )
    parser.add_argument(
        "--allow-missing-features",
        action="store_true",
        help="Do not block rows with empty or missing features_json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.log_path:
        log_path = args.log_path
    elif args.latest:
        log_path = discover_latest_decision_log(args.decisions_dir)
        if log_path is None:
            print(f"error: no decision logs found in {args.decisions_dir}")
            return 1
    else:
        print("error: specify --latest or --log-path")
        return 1

    blocked_signals = args.block_signals or ["baseline_fallback"]
    config = {
        "max_spread_bps": args.max_spread_bps,
        "allowed_coins": tuple(coin.upper() for coin in args.allow_coins),
        "blocked_signals": tuple(blocked_signals),
        "min_confidence": args.min_confidence,
        "require_model_beats_baseline": not args.no_require_model_beats_baseline,
        "min_sample_size": args.min_sample_size,
        "block_if_missing_features": not args.allow_missing_features,
    }

    report = run_filter(log_path, config=config, output_dir=args.output_dir)

    print(f"json_report_path={report['json_report_path']}")
    print(f"markdown_report_path={report['markdown_report_path']}")
    print(f"accepted_count={report['accepted_count']}")
    print(f"blocked_count={report['blocked_count']}")
    print(f"accepted_accuracy={report['accepted_accuracy']:.4f}")
    print(f"accepted_edge_vs_baseline={report['accepted_edge_vs_baseline']:.4f}")
    for warning in report["warnings"]:
        if warning != SAFETY_NOTE:
            print(f"warning: {warning}")
    print(SAFETY_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
