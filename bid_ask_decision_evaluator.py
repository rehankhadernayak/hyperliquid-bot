"""Offline evaluation of paper-only bid/ask model decision logs (research-only)."""

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
DEFAULT_OUTPUT_DIR = Path("reports/bid_ask_decision_evaluation")

LABELS = ("UP", "DOWN", "FLAT")
SPREAD_BUCKETS = ("tight", "normal", "wide", "too_wide")

MIN_SAMPLE_SIZE = 30
CLASS_IMBALANCE_RATIO = 0.70
TOO_WIDE_SPREAD_FRACTION = 0.10
MAX_INCORRECT_EXAMPLES = 10

RESEARCH_ONLY_DISCLAIMER = (
    "Research-only evaluation on historical paper decisions. "
    "Does not prove predictive power or profitability. "
    "No trades are placed."
)


def discover_decision_logs(
    decisions_dir: Path = DEFAULT_DECISIONS_DIR,
) -> list[Path]:
    """Find decision log CSV paths, newest first."""
    if not decisions_dir.is_dir():
        return []
    logs = sorted(
        decisions_dir.glob("bid_ask_decisions_*.csv"),
        key=lambda path: path.name,
    )
    return list(reversed(logs))


def load_decision_log(path: Path) -> pd.DataFrame:
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


def spread_bucket(spread_bps: float) -> str:
    """Classify spread into research review buckets."""
    if spread_bps <= 2:
        return "tight"
    if spread_bps <= 10:
        return "normal"
    if spread_bps <= 25:
        return "wide"
    return "too_wide"


def _accuracy_stats(correct: pd.Series) -> dict[str, Any]:
    total = int(len(correct))
    if total == 0:
        return {"count": 0, "correct": 0, "accuracy": 0.0}
    correct_count = int(correct.astype(bool).sum())
    return {
        "count": total,
        "correct": correct_count,
        "accuracy": correct_count / total,
    }


def _group_accuracy(
    df: pd.DataFrame,
    group_column: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for key, group in df.groupby(group_column, dropna=False):
        label = str(key)
        grouped[label] = _accuracy_stats(group["signal_correct"])
    return grouped


def _label_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = {label: 0 for label in LABELS}
    for label, count in df["actual_label"].value_counts().items():
        label_str = str(label)
        if label_str in counts:
            counts[label_str] = int(count)
    return counts


def _incorrect_examples(df: pd.DataFrame) -> list[dict[str, Any]]:
    incorrect = df[~df["signal_correct"].astype(bool)].copy()
    if incorrect.empty:
        return []

    examples: list[dict[str, Any]] = []
    for _, row in incorrect.head(MAX_INCORRECT_EXAMPLES).iterrows():
        spread_bps = float(row["spread_bps_or_avg_spread_bps"])
        examples.append(
            {
                "decision_id": str(row["decision_id"]),
                "coin": str(row["coin"]),
                "model_signal": str(row["model_signal"]),
                "baseline_signal": str(row["baseline_signal"]),
                "actual_label": str(row["actual_label"]),
                "spread_bps": spread_bps,
                "spread_bucket": spread_bucket(spread_bps),
                "confidence": float(row["confidence"]),
                "future_return_bps": float(row["future_return_bps"]),
            }
        )
    return examples


def _build_warnings(
    df: pd.DataFrame,
    *,
    total: int,
    model_accuracy: float,
    baseline_accuracy: float,
    label_counts: dict[str, int],
) -> list[str]:
    warnings: list[str] = [SAFETY_NOTE]

    if total < MIN_SAMPLE_SIZE:
        warnings.append(
            f"Small sample size ({total} decisions); "
            f"need at least {MIN_SAMPLE_SIZE} for stable metrics"
        )

    if model_accuracy <= baseline_accuracy:
        warnings.append(
            "Model accuracy does not beat baseline; "
            "review model or restrict paper decisions"
        )

    if total > 0:
        dominant = max(label_counts.values())
        if dominant / total >= CLASS_IMBALANCE_RATIO:
            dominant_label = max(label_counts, key=label_counts.get)
            warnings.append(
                f"High class imbalance: {dominant_label} is "
                f"{dominant / total:.1%} of actual labels"
            )

        spread_values = df["spread_bps_or_avg_spread_bps"].astype(float)
        too_wide_fraction = (spread_values > 25).sum() / total
        if too_wide_fraction >= TOO_WIDE_SPREAD_FRACTION:
            warnings.append(
                f"Spread too wide on {too_wide_fraction:.1%} of decisions "
                f"(>{25} bps); review spread filter for paper decisions"
            )

    if (df["model_signal"] == "baseline_fallback").any():
        warnings.append("Some decisions used baseline_fallback for model_signal")

    return warnings


def evaluate_decisions(df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate paper decision log quality and return research metrics."""
    total = len(df)
    if total == 0:
        return {
            "total_decisions": 0,
            "model_accuracy": 0.0,
            "baseline_accuracy": 0.0,
            "model_edge_vs_baseline": 0.0,
            "accuracy_by_coin": {},
            "accuracy_by_signal": {},
            "accuracy_by_spread_bucket": {},
            "label_counts": {label: 0 for label in LABELS},
            "incorrect_examples": [],
            "warnings": [SAFETY_NOTE, "Decision log is empty"],
            "paper_only": True,
            "execution_enabled": False,
        }

    model_stats = _accuracy_stats(df["signal_correct"])
    baseline_stats = _accuracy_stats(df["baseline_correct"])
    model_accuracy = model_stats["accuracy"]
    baseline_accuracy = baseline_stats["accuracy"]

    spread_series = df["spread_bps_or_avg_spread_bps"].astype(float)
    bucketed = spread_series.apply(spread_bucket)
    bucket_df = df.copy()
    bucket_df["_spread_bucket"] = bucketed

    label_counts = _label_counts(df)
    warnings = _build_warnings(
        df,
        total=total,
        model_accuracy=model_accuracy,
        baseline_accuracy=baseline_accuracy,
        label_counts=label_counts,
    )

    return {
        "total_decisions": total,
        "model_accuracy": model_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "model_edge_vs_baseline": model_accuracy - baseline_accuracy,
        "model_correct_count": model_stats["correct"],
        "baseline_correct_count": baseline_stats["correct"],
        "accuracy_by_coin": _group_accuracy(df, "coin"),
        "accuracy_by_signal": _group_accuracy(df, "model_signal"),
        "accuracy_by_spread_bucket": _group_accuracy(bucket_df, "_spread_bucket"),
        "label_counts": label_counts,
        "incorrect_examples": _incorrect_examples(df),
        "warnings": warnings,
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
    }


def suggest_blocking_rules(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    """Suggest paper-only review rules; does not execute trades."""
    rules: list[dict[str, Any]] = []
    baseline_accuracy = float(evaluation.get("baseline_accuracy", 0.0))
    model_accuracy = float(evaluation.get("model_accuracy", 0.0))
    total = int(evaluation.get("total_decisions", 0))

    if total > 0 and model_accuracy <= baseline_accuracy:
        rules.append(
            {
                "rule_type": "block_model",
                "action": "Skip paper model decisions until accuracy beats baseline",
                "reason": (
                    f"Model accuracy {model_accuracy:.4f} "
                    f"<= baseline {baseline_accuracy:.4f}"
                ),
                "paper_only": True,
            }
        )

    for coin, stats in evaluation.get("accuracy_by_coin", {}).items():
        count = int(stats.get("count", 0))
        if count < MIN_SAMPLE_SIZE:
            rules.append(
                {
                    "rule_type": "block_coin",
                    "coin": coin,
                    "action": f"Require more paper samples before trusting {coin} metrics",
                    "reason": f"Only {count} decisions (minimum {MIN_SAMPLE_SIZE})",
                    "paper_only": True,
                }
            )

    for signal, stats in evaluation.get("accuracy_by_signal", {}).items():
        accuracy = float(stats.get("accuracy", 0.0))
        count = int(stats.get("count", 0))
        if count >= 5 and accuracy < baseline_accuracy:
            rules.append(
                {
                    "rule_type": "block_signal",
                    "signal": signal,
                    "action": f"Review or skip paper decisions when model_signal is {signal!r}",
                    "reason": (
                        f"Signal accuracy {accuracy:.4f} "
                        f"below baseline {baseline_accuracy:.4f}"
                    ),
                    "paper_only": True,
                }
            )

    spread_stats = evaluation.get("accuracy_by_spread_bucket", {})
    for bucket_name in ("wide", "too_wide"):
        stats = spread_stats.get(bucket_name)
        if not stats:
            continue
        count = int(stats.get("count", 0))
        accuracy = float(stats.get("accuracy", 0.0))
        if count >= 5 and accuracy < baseline_accuracy:
            threshold = 10 if bucket_name == "wide" else 25
            rules.append(
                {
                    "rule_type": "block_spread_above",
                    "spread_bps_threshold": threshold,
                    "action": (
                        f"Block paper decisions when spread_bps > {threshold} "
                        f"(review {bucket_name} bucket)"
                    ),
                    "reason": (
                        f"{bucket_name} bucket accuracy {accuracy:.4f} "
                        f"below baseline {baseline_accuracy:.4f}"
                    ),
                    "paper_only": True,
                }
            )

    for warning in evaluation.get("warnings", []):
        if "Spread too wide" in warning:
            rules.append(
                {
                    "rule_type": "block_spread_above",
                    "spread_bps_threshold": 25,
                    "action": "Block paper decisions when spread_bps > 25",
                    "reason": warning,
                    "paper_only": True,
                }
            )
            break

    return rules


def _evaluation_stamp(evaluation: dict[str, Any]) -> str:
    if evaluation.get("generated_at"):
        raw = str(evaluation["generated_at"])
        return raw.replace(":", "").replace("+", "").replace("-", "")[:15]
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_evaluation_json(
    evaluation: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write evaluation JSON to reports/bid_ask_decision_evaluation/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _evaluation_stamp(evaluation)
    output_path = output_dir / f"bid_ask_decision_evaluation_{stamp}.json"
    output_path.write_text(
        json.dumps(evaluation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    evaluation["json_report_path"] = str(output_path)
    return output_path


def _format_accuracy(value: float) -> str:
    return f"{value:.4f}"


def write_evaluation_markdown(
    evaluation: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write evaluation Markdown to reports/bid_ask_decision_evaluation/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _evaluation_stamp(evaluation)
    output_path = output_dir / f"bid_ask_decision_evaluation_{stamp}.md"

    lines = [
        "# Paper Decision Evaluation Board",
        "",
        f"Generated at: {evaluation.get('generated_at', 'n/a')}",
        f"Source log: {evaluation.get('source_log_path', 'n/a')}",
        "",
        "## Overall result",
        "",
        f"- total_decisions: {evaluation.get('total_decisions', 0)}",
        f"- model_accuracy: {_format_accuracy(float(evaluation.get('model_accuracy', 0.0)))}",
        f"- baseline_accuracy: {_format_accuracy(float(evaluation.get('baseline_accuracy', 0.0)))}",
        f"- model_edge_vs_baseline: {_format_accuracy(float(evaluation.get('model_edge_vs_baseline', 0.0)))}",
        f"- paper_only: {str(evaluation.get('paper_only', True)).lower()}",
        f"- execution_enabled: {str(evaluation.get('execution_enabled', False)).lower()}",
        "",
        "## Accuracy vs baseline",
        "",
        f"Model correct: {evaluation.get('model_correct_count', 0)} / "
        f"{evaluation.get('total_decisions', 0)}",
        f"Baseline correct: {evaluation.get('baseline_correct_count', 0)} / "
        f"{evaluation.get('total_decisions', 0)}",
        f"Edge: {_format_accuracy(float(evaluation.get('model_edge_vs_baseline', 0.0)))}",
        "",
        "## Coin breakdown",
        "",
        "| Coin | Count | Correct | Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]

    for coin, stats in sorted(evaluation.get("accuracy_by_coin", {}).items()):
        lines.append(
            f"| {coin} | {stats['count']} | {stats['correct']} | "
            f"{_format_accuracy(float(stats['accuracy']))} |"
        )

    lines.extend(
        [
            "",
            "## Signal breakdown",
            "",
            "| Signal | Count | Correct | Accuracy |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for signal, stats in sorted(evaluation.get("accuracy_by_signal", {}).items()):
        lines.append(
            f"| {signal} | {stats['count']} | {stats['correct']} | "
            f"{_format_accuracy(float(stats['accuracy']))} |"
        )

    lines.extend(
        [
            "",
            "## Spread bucket breakdown",
            "",
            "| Bucket | Count | Correct | Accuracy |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for bucket in SPREAD_BUCKETS:
        stats = evaluation.get("accuracy_by_spread_bucket", {}).get(bucket)
        if not stats:
            lines.append(f"| {bucket} | 0 | 0 | n/a |")
            continue
        lines.append(
            f"| {bucket} | {stats['count']} | {stats['correct']} | "
            f"{_format_accuracy(float(stats['accuracy']))} |"
        )

    label_counts = evaluation.get("label_counts", {})
    lines.extend(
        [
            "",
            "### Actual label counts (UP / DOWN / FLAT)",
            "",
            f"- UP: {label_counts.get('UP', 0)}",
            f"- DOWN: {label_counts.get('DOWN', 0)}",
            f"- FLAT: {label_counts.get('FLAT', 0)}",
            "",
            "## Incorrect examples",
            "",
        ]
    )

    examples = evaluation.get("incorrect_examples", [])
    if not examples:
        lines.append("No incorrect model decisions in this log.")
    else:
        for index, example in enumerate(examples, start=1):
            lines.append(
                f"{index}. `{example['decision_id']}` "
                f"{example['coin']} model={example['model_signal']} "
                f"actual={example['actual_label']} "
                f"spread={example['spread_bps']:.2f}bps "
                f"({example['spread_bucket']})"
            )

    lines.extend(
        [
            "",
            "## Suggested paper-only blocking rules",
            "",
        ]
    )
    rules = evaluation.get("suggested_blocking_rules", [])
    if not rules:
        lines.append("No blocking rules suggested.")
    else:
        for index, rule in enumerate(rules, start=1):
            lines.append(f"{index}. **{rule['rule_type']}**: {rule['action']}")
            lines.append(f"   - Reason: {rule['reason']}")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Metrics reflect historical paper labels, not live fills or PnL.",
            "- Small samples and class imbalance can distort accuracy.",
            "- Spread buckets are descriptive; thresholds are research heuristics.",
            "- Blocking rules are review suggestions only; they do not place orders.",
            "",
            "## Research-only disclaimer",
            "",
            RESEARCH_ONLY_DISCLAIMER,
            "",
            SAFETY_STATEMENT,
            "",
            SAFETY_NOTE,
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    evaluation["markdown_report_path"] = str(output_path)
    return output_path


def run_evaluation(
    log_path: Path,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Load, evaluate, and write reports for one decision log."""
    df = load_decision_log(log_path)
    evaluation = evaluate_decisions(df)
    evaluation["generated_at"] = datetime.now(timezone.utc).isoformat()
    evaluation["source_log_path"] = str(log_path)
    evaluation["suggested_blocking_rules"] = suggest_blocking_rules(evaluation)
    write_evaluation_json(evaluation, output_dir=output_dir)
    write_evaluation_markdown(evaluation, output_dir=output_dir)
    return evaluation


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate paper-only bid/ask model decision logs "
            "(research-only, no execution)."
        )
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Evaluate the newest decision log in reports/bid_ask_decisions/",
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
        help="Directory for evaluation JSON and Markdown reports",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.log_path:
        log_path = args.log_path
    elif args.latest:
        logs = discover_decision_logs(args.decisions_dir)
        if not logs:
            print(f"error: no decision logs found in {args.decisions_dir}")
            return 1
        log_path = logs[0]
    else:
        print("error: specify --latest or --log-path")
        return 1

    evaluation = run_evaluation(log_path, output_dir=args.output_dir)

    print(f"json_report_path={evaluation['json_report_path']}")
    print(f"markdown_report_path={evaluation['markdown_report_path']}")
    print(f"model_accuracy={evaluation['model_accuracy']:.4f}")
    print(f"baseline_accuracy={evaluation['baseline_accuracy']:.4f}")
    for warning in evaluation["warnings"]:
        if warning != SAFETY_NOTE:
            print(f"warning: {warning}")
    print(SAFETY_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
