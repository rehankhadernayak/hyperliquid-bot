"""Paper-only bid/ask model decision logging on local CSV data (research-only)."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bid_ask_model_io import (
    DEFAULT_MODELS_DIR,
    feature_names_compatible,
    find_latest_model_artifact,
    predict_with_artifact,
)
from bid_ask_resampler import RESAMPLED_COLUMNS, resampled_csv_path
from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT
from bid_ask_train import (
    FEATURE_COLUMNS,
    MID_PRICE_COLUMN,
    MajorityClassBaseline,
    create_features,
    create_labels,
)

SAFETY_NOTE = "Paper-only. No trades placed."

DEFAULT_DATA_ROOT = Path("data/bid_ask_csv")
DEFAULT_TRAINING_REPORTS_DIR = Path("reports/bid_ask_training")
DEFAULT_DECISIONS_DIR = Path("reports/bid_ask_decisions")

SUPPORTED_INTERVALS = frozenset({"15s", "1m"})
LABELS = ("UP", "DOWN", "FLAT")

DECISION_COLUMNS = (
    "decision_id",
    "generated_at",
    "coin",
    "interval",
    "source_csv",
    "source_row_timestamp",
    "source_row_iso",
    "best_bid_or_avg_bid",
    "best_ask_or_avg_ask",
    "mid_or_close_mid",
    "spread_or_avg_spread",
    "spread_bps_or_avg_spread_bps",
    "model_signal",
    "baseline_signal",
    "confidence",
    "horizon",
    "future_return_bps",
    "actual_label",
    "signal_correct",
    "baseline_correct",
    "features_json",
    "paper_only",
    "execution_enabled",
    "notes",
)


def load_latest_training_report(
    coin: str,
    training_reports_dir: Path = DEFAULT_TRAINING_REPORTS_DIR,
    *,
    interval: str | None = None,
) -> dict[str, Any] | None:
    """Load the newest training JSON report for a coin (optionally matching interval)."""
    if not training_reports_dir.is_dir():
        return None

    candidates: list[tuple[str, Path]] = []
    for path in training_reports_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("coin") != coin.upper():
            continue
        if interval is not None and payload.get("interval") != interval:
            continue
        generated_at = payload.get("generated_at") or ""
        candidates.append((generated_at, path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    latest_path = candidates[-1][1]
    return json.loads(latest_path.read_text(encoding="utf-8"))


def load_signal_source_csv(
    coin: str,
    interval: str,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> pd.DataFrame:
    """Load resampled bid/ask CSV for decision logging."""
    if interval not in SUPPORTED_INTERVALS:
        supported = ", ".join(sorted(SUPPORTED_INTERVALS))
        raise ValueError(f"Unsupported interval {interval!r}; use one of: {supported}")

    path = resampled_csv_path(coin, interval, data_root)
    if not path.is_file():
        raise FileNotFoundError(f"Source CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [column for column in RESAMPLED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    return df.sort_values("start_timestamp_ms").reset_index(drop=True)


def compute_features_for_latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Compute model features using the same definitions as bid_ask_train."""
    return create_features(df)


def _feature_matrix_row(row: pd.Series) -> pd.DataFrame:
    matrix = pd.DataFrame([row[list(FEATURE_COLUMNS)]])
    matrix["avg_bbo_imbalance"] = matrix["avg_bbo_imbalance"].fillna(0.0)
    matrix["bbo_imbalance_change_1"] = matrix["bbo_imbalance_change_1"].fillna(0.0)
    return matrix.astype(float)


def _rows_ready_for_decisions(df: pd.DataFrame) -> pd.DataFrame:
    required = list(FEATURE_COLUMNS) + ["label", "future_return_bps", MID_PRICE_COLUMN]
    return df.dropna(subset=required).reset_index(drop=True)


def _row_has_artifact_features(row: pd.Series, artifact: dict[str, Any]) -> bool:
    feature_names = artifact.get("feature_names") or []
    for name in feature_names:
        if name not in row.index:
            return False
        if pd.isna(row[name]):
            return False
    return True


def generate_paper_decisions(
    coin: str,
    interval: str = "15s",
    horizon: int = 1,
    max_rows: int = 100,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    training_reports_dir: Path = DEFAULT_TRAINING_REPORTS_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    threshold_bps: float | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Generate paper-only model decisions for eligible historical rows."""
    coin = coin.upper()
    warnings: list[str] = [
        "Research-only decision log; does not prove predictive power or profitability"
    ]

    report = load_latest_training_report(
        coin, training_reports_dir=training_reports_dir, interval=interval
    )
    report_horizon = horizon
    report_threshold = threshold_bps if threshold_bps is not None else 1.0
    baseline_majority = "FLAT"

    if report is None:
        warnings.append(f"No training report found for {coin}; using default baseline only")
    else:
        report_horizon = int(report.get("horizon", horizon))
        report_threshold = float(
            threshold_bps if threshold_bps is not None else report.get("threshold_bps", 1.0)
        )
        baseline_majority = str(report.get("baseline_majority_class") or "FLAT")

    if report_horizon != horizon:
        warnings.append(
            f"Using horizon={report_horizon} from training report "
            f"(CLI requested horizon={horizon})"
        )

    artifact = find_latest_model_artifact(coin, interval, models_dir=models_dir)
    artifact_compatible = (
        artifact is not None and feature_names_compatible(artifact, FEATURE_COLUMNS)
    )
    if artifact is not None and not artifact_compatible:
        warnings.append(
            "Model artifact feature_names mismatch; falling back to baseline"
        )

    source_path = resampled_csv_path(coin, interval, data_root)
    raw_df = load_signal_source_csv(coin, interval, data_root)
    featured_df = compute_features_for_latest_rows(raw_df)
    labeled_df = create_labels(
        featured_df,
        horizon=report_horizon,
        threshold_bps=report_threshold,
    )
    ready_df = _rows_ready_for_decisions(labeled_df)

    if ready_df.empty:
        warnings.append("No eligible rows with complete features and future labels")
        return [], warnings

    selected_df = ready_df.iloc[-max_rows:].copy()
    majority_baseline = MajorityClassBaseline(baseline_majority)

    decisions: list[dict[str, Any]] = []
    generated_at = datetime.now(timezone.utc).isoformat()

    for _, row in selected_df.iterrows():
        baseline_predictions = majority_baseline.predict(_feature_matrix_row(row))
        baseline_signal = baseline_predictions[0]

        notes: str
        if artifact_compatible and _row_has_artifact_features(row, artifact):
            model_signal, confidence = predict_with_artifact(artifact, row)
            notes = "model_artifact_used"
        elif artifact is not None:
            model_signal = "baseline_fallback"
            confidence = 1.0 / len(LABELS)
            notes = "model_feature_mismatch_baseline_fallback"
        else:
            model_signal = "baseline_fallback"
            confidence = 1.0 / len(LABELS)
            notes = "baseline_fallback_no_model_artifact"

        actual_label = str(row["label"])
        future_return = float(row["future_return_bps"])
        feature_payload = {
            column: float(row[column]) if pd.notna(row[column]) else None
            for column in FEATURE_COLUMNS
        }

        using_model = notes == "model_artifact_used"
        decisions.append(
            {
                "decision_id": uuid.uuid4().hex,
                "generated_at": generated_at,
                "coin": coin,
                "interval": interval,
                "source_csv": str(source_path),
                "source_row_timestamp": int(row["start_timestamp_ms"]),
                "source_row_iso": str(row.get("timestamp_iso", "")),
                "best_bid_or_avg_bid": float(row["avg_bid"]),
                "best_ask_or_avg_ask": float(row["avg_ask"]),
                "mid_or_close_mid": float(row["close_mid"]),
                "spread_or_avg_spread": float(row["avg_spread"]),
                "spread_bps_or_avg_spread_bps": float(row["avg_spread_bps"]),
                "model_signal": model_signal,
                "baseline_signal": baseline_signal,
                "confidence": confidence,
                "horizon": report_horizon,
                "future_return_bps": future_return,
                "actual_label": actual_label,
                "signal_correct": model_signal == actual_label
                if using_model
                else baseline_signal == actual_label,
                "baseline_correct": baseline_signal == actual_label,
                "features_json": json.dumps(feature_payload, sort_keys=True),
                "paper_only": PAPER_ONLY,
                "execution_enabled": EXECUTION_ENABLED,
                "notes": notes,
            }
        )

    return decisions, warnings


def write_decision_log(
    decisions: list[dict[str, Any]],
    output_dir: Path = DEFAULT_DECISIONS_DIR,
) -> Path:
    """Write decision rows to CSV under reports/bid_ask_decisions/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"bid_ask_decisions_{stamp}.csv"

    frame = pd.DataFrame(decisions, columns=list(DECISION_COLUMNS))
    frame.to_csv(output_path, index=False)
    return output_path


def summarize_decision_log(path: Path) -> dict[str, Any]:
    """Summarize a decision log CSV for research review."""
    if not path.is_file():
        raise FileNotFoundError(f"Decision log not found: {path}")

    df = pd.read_csv(path)
    warnings: list[str] = [SAFETY_NOTE]

    total = len(df)
    if total == 0:
        warnings.append("Decision log is empty")
        return {
            "total_decisions": 0,
            "model_correct_count": 0,
            "model_accuracy": 0.0,
            "baseline_accuracy": 0.0,
            "label_counts": {label: 0 for label in LABELS},
            "warnings": warnings,
            "paper_only": True,
            "execution_enabled": False,
            "path": str(path),
        }

    model_correct = int(df["signal_correct"].astype(bool).sum())
    baseline_correct = int(df["baseline_correct"].astype(bool).sum())
    label_counts = {
        label: int((df["actual_label"] == label).sum()) for label in LABELS
    }

    if df["paper_only"].astype(str).str.lower().eq("false").any():
        warnings.append("Unexpected paper_only=false in decision log")
    if df["execution_enabled"].astype(str).str.lower().eq("true").any():
        warnings.append("Unexpected execution_enabled=true in decision log")
    if (df["notes"] == "baseline_fallback_no_model_artifact").any():
        warnings.append("Some decisions used baseline_fallback_no_model_artifact")
    if (df["model_signal"] == "baseline_fallback").any():
        warnings.append("Some decisions used baseline_fallback for model_signal")

    return {
        "total_decisions": total,
        "model_correct_count": model_correct,
        "model_accuracy": model_correct / total,
        "baseline_accuracy": baseline_correct / total,
        "label_counts": label_counts,
        "warnings": warnings,
        "paper_only": True,
        "execution_enabled": False,
        "path": str(path),
    }


def run_decision_logging(
    coins: list[str],
    interval: str = "15s",
    horizon: int = 1,
    max_rows: int = 100,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    training_reports_dir: Path = DEFAULT_TRAINING_REPORTS_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    output_dir: Path = DEFAULT_DECISIONS_DIR,
) -> dict[str, Any]:
    """Generate and write paper decisions for multiple coins."""
    all_decisions: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    per_coin_counts: dict[str, int] = {}

    for coin in coins:
        decisions, warnings = generate_paper_decisions(
            coin,
            interval=interval,
            horizon=horizon,
            max_rows=max_rows,
            data_root=data_root,
            training_reports_dir=training_reports_dir,
            models_dir=models_dir,
        )
        per_coin_counts[coin.upper()] = len(decisions)
        all_decisions.extend(decisions)
        for warning in warnings:
            tagged = f"{coin.upper()}: {warning}"
            if tagged not in all_warnings:
                all_warnings.append(tagged)

    output_path = write_decision_log(all_decisions, output_dir=output_dir)
    summary = summarize_decision_log(output_path)
    summary["per_coin_counts"] = per_coin_counts
    summary["output_path"] = str(output_path)
    summary["warnings"] = list(dict.fromkeys(summary["warnings"] + all_warnings))
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Log paper-only bid/ask model decisions on local CSV data "
            "(research-only, no execution)."
        )
    )
    parser.add_argument(
        "--coins",
        nargs="+",
        required=True,
        help="Coin symbols, e.g. BTC ETH SOL",
    )
    parser.add_argument(
        "--interval",
        default="15s",
        choices=sorted(SUPPORTED_INTERVALS),
        help="Resampled interval (default: 15s)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Label horizon in bars (default: 1)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100,
        help="Maximum eligible rows per coin (default: 100)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing bid/ask CSV files",
    )
    parser.add_argument(
        "--training-reports-dir",
        type=Path,
        default=DEFAULT_TRAINING_REPORTS_DIR,
        help="Directory containing JSON training reports",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory containing saved model artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DECISIONS_DIR,
        help="Directory for decision log CSV output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    summary = run_decision_logging(
        coins=[coin.upper() for coin in args.coins],
        interval=args.interval,
        horizon=args.horizon,
        max_rows=args.max_rows,
        data_root=args.data_root,
        training_reports_dir=args.training_reports_dir,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
    )

    print(f"output_path={summary['output_path']}")
    for coin, count in summary.get("per_coin_counts", {}).items():
        print(f"decisions_{coin}={count}")
    print(f"model_accuracy={summary['model_accuracy']:.4f}")
    print(f"baseline_accuracy={summary['baseline_accuracy']:.4f}")
    for warning in summary["warnings"]:
        print(f"warning: {warning}")
    print(SAFETY_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
