"""Read-only bid/ask movement training on local CSV data (research-only, paper-only)."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bid_ask_resampler import RESAMPLED_COLUMNS, resampled_csv_path
from bid_ask_schema import (
    EXECUTION_ENABLED,
    FORBIDDEN_MODULE_NAMES,
    FORBIDDEN_SOURCE_PATTERNS,
    PAPER_ONLY,
)

SAFETY_STATEMENT = (
    "Bid/ask movement training is read-only, paper-trading only, and research-only. "
    "No wallets, private keys, signing, leverage, margin, exchange actions, "
    "real orders, deposits, withdrawals, or execution endpoints are used. "
    "Training output is not connected to live trading and does not prove profitability."
)

DEFAULT_DATA_ROOT = Path("data/bid_ask_csv")
DEFAULT_REPORTS_DIR = Path("reports/bid_ask_training")
DEFAULT_MODELS_DIR = Path("models/bid_ask")

SUPPORTED_INTERVALS = frozenset({"15s", "1m"})
LABELS = ("UP", "DOWN", "FLAT")

FEATURE_COLUMNS = (
    "avg_spread_bps",
    "avg_bbo_imbalance",
    "mid_return_1",
    "rolling_return_3",
    "rolling_return_5",
    "rolling_volatility_5",
    "spread_change_1",
    "bbo_imbalance_change_1",
)

MID_PRICE_COLUMN = "close_mid"

MIN_ROWS_WARNING = 50
MIN_CLASS_FRACTION = 0.10

_SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    _SKLEARN_AVAILABLE = True
except ImportError:
    RandomForestClassifier = None  # type: ignore[assignment,misc]
    LogisticRegression = None  # type: ignore[assignment,misc]
    StandardScaler = None  # type: ignore[assignment,misc]


def load_training_data(
    coin: str,
    interval: str,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> pd.DataFrame:
    """Load resampled bid/ask CSV for the given coin and interval."""
    if interval not in SUPPORTED_INTERVALS:
        supported = ", ".join(sorted(SUPPORTED_INTERVALS))
        raise ValueError(f"Unsupported interval {interval!r}; use one of: {supported}")

    path = resampled_csv_path(coin, interval, data_root)
    if not path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [column for column in RESAMPLED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    return df.sort_values("start_timestamp_ms").reset_index(drop=True)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model features from resampled bid/ask bars."""
    if df.empty:
        return df.copy()

    out = df.copy()
    mid = out[MID_PRICE_COLUMN].astype(float)
    prior_mid = mid.shift(1)
    safe_prior = prior_mid.replace(0, np.nan)

    out["mid_return_1"] = (mid - prior_mid) / safe_prior * 10_000.0
    out["rolling_return_3"] = (mid - mid.shift(3)) / mid.shift(3).replace(0, np.nan) * 10_000.0
    out["rolling_return_5"] = (mid - mid.shift(5)) / mid.shift(5).replace(0, np.nan) * 10_000.0
    out["rolling_volatility_5"] = out["mid_return_1"].rolling(window=5, min_periods=5).std()
    out["spread_change_1"] = out["avg_spread_bps"].astype(float).diff(1)
    out["bbo_imbalance_change_1"] = out["avg_bbo_imbalance"].astype(float).diff(1)

    return out


def create_labels(
    df: pd.DataFrame,
    horizon: int = 1,
    threshold_bps: float = 1.0,
) -> pd.DataFrame:
    """Add future_return_bps and UP/DOWN/FLAT labels."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    out = df.copy()
    mid = out[MID_PRICE_COLUMN].astype(float)
    future_mid = mid.shift(-horizon)
    out["future_return_bps"] = (future_mid - mid) / mid.replace(0, np.nan) * 10_000.0

    labels = np.full(len(out), "FLAT", dtype=object)
    labels[out["future_return_bps"] > threshold_bps] = "UP"
    labels[out["future_return_bps"] < -threshold_bps] = "DOWN"
    out["label"] = labels
    return out


def chronological_train_test_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows in time order without shuffling."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    if df.empty:
        return df.copy(), df.copy()

    split_index = max(1, int(len(df) * train_fraction))
    if split_index >= len(df):
        split_index = len(df) - 1

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix = df[list(FEATURE_COLUMNS)].copy()
    matrix["avg_bbo_imbalance"] = matrix["avg_bbo_imbalance"].fillna(0.0)
    matrix["bbo_imbalance_change_1"] = matrix["bbo_imbalance_change_1"].fillna(0.0)
    return matrix.astype(float)


def _rows_ready_for_model(df: pd.DataFrame) -> pd.DataFrame:
    required = list(FEATURE_COLUMNS) + ["label", "future_return_bps", MID_PRICE_COLUMN]
    return df.dropna(subset=required).reset_index(drop=True)


class MajorityClassBaseline:
    """Predict the most common training label."""

    model_type = "majority_class"

    def __init__(self, majority_class: str):
        self.majority_class = majority_class

    def predict(self, X: pd.DataFrame | np.ndarray) -> list[str]:
        count = len(X) if hasattr(X, "__len__") else 1
        return [self.majority_class] * count


class RuleBaseline:
    """Simple rule: UP/DOWN/FLAT from the latest mid_return_1 sign."""

    model_type = "rule_mid_return"

    def predict(self, X: pd.DataFrame) -> list[str]:
        predictions: list[str] = []
        returns = X["mid_return_1"] if isinstance(X, pd.DataFrame) else pd.Series(X[:, 3])
        for value in returns:
            if value > 0:
                predictions.append("UP")
            elif value < 0:
                predictions.append("DOWN")
            else:
                predictions.append("FLAT")
        return predictions


def train_baseline_model(
    train_df: pd.DataFrame,
) -> dict[str, Any]:
    """Train majority-class and simple rule baselines on training labels."""
    labels = train_df["label"].tolist()
    if not labels:
        majority = "FLAT"
    else:
        counts = pd.Series(labels).value_counts()
        majority = str(counts.idxmax())

    majority_model = MajorityClassBaseline(majority)
    rule_model = RuleBaseline()
    return {
        "majority_model": majority_model,
        "rule_model": rule_model,
        "majority_class": majority,
    }


def train_optional_sklearn_model(
    train_df: pd.DataFrame,
) -> dict[str, Any]:
    """Train sklearn classifier when available; otherwise return skip metadata."""
    if not _SKLEARN_AVAILABLE:
        return {
            "available": False,
            "model": None,
            "message": "sklearn not installed; using baseline models only",
        }

    X_train = _feature_matrix(train_df)
    y_train = train_df["label"].tolist()
    if len(X_train) == 0:
        return {
            "available": True,
            "model": None,
            "message": "no training rows available for sklearn model",
        }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    if len(set(y_train)) < 2:
        model: Any = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            class_weight="balanced_subsample",
        )

    model.fit(X_scaled, y_train)
    return {
        "available": True,
        "model": model,
        "scaler": scaler,
        "model_type": type(model).__name__,
        "message": None,
    }


def _predict_sklearn(
    sklearn_result: dict[str, Any],
    test_df: pd.DataFrame,
) -> list[str] | None:
    model = sklearn_result.get("model")
    scaler = sklearn_result.get("scaler")
    if model is None or scaler is None:
        return None
    X_test = _feature_matrix(test_df)
    if len(X_test) == 0:
        return []
    predictions = model.predict(scaler.transform(X_test))
    return [str(label) for label in predictions]


def _confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, dict[str, int]]:
    matrix = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}
    for actual, predicted in zip(y_true, y_pred, strict=False):
        if actual in matrix and predicted in matrix[actual]:
            matrix[actual][predicted] += 1
    return matrix


def _accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for truth, pred in zip(y_true, y_pred, strict=False) if truth == pred)
    return correct / len(y_true)


def _label_counts(labels: list[str]) -> dict[str, int]:
    counts = {label: 0 for label in LABELS}
    for label in labels:
        if label in counts:
            counts[label] += 1
    return counts


def evaluate_predictions(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions: list[str],
    baseline_predictions: list[str],
    features_used: tuple[str, ...] = FEATURE_COLUMNS,
) -> dict[str, Any]:
    """Summarize model and baseline performance with research warnings."""
    y_true = test_df["label"].tolist()
    warnings: list[str] = []

    total_rows = len(train_df) + len(test_df)
    if total_rows < MIN_ROWS_WARNING:
        warnings.append(
            f"Low row count ({total_rows} total rows); results may be unreliable"
        )

    train_counts = _label_counts(train_df["label"].tolist())
    test_counts = _label_counts(y_true)
    for split_name, counts in (("train", train_counts), ("test", test_counts)):
        split_total = sum(counts.values())
        if split_total == 0:
            continue
        for label, count in counts.items():
            fraction = count / split_total
            if fraction < MIN_CLASS_FRACTION:
                warnings.append(
                    f"Class imbalance in {split_name}: {label} is {fraction:.1%} of rows"
                )

    warnings.append("Research-only evaluation; does not prove profitability")

    return {
        "accuracy": _accuracy(y_true, predictions),
        "baseline_accuracy": _accuracy(y_true, baseline_predictions),
        "confusion_matrix": _confusion_matrix(y_true, predictions),
        "label_counts": {
            "train": train_counts,
            "test": test_counts,
        },
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "features_used": list(features_used),
        "warnings": warnings,
    }


def save_training_report(
    report: dict[str, Any],
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    *,
    sklearn_result: dict[str, Any] | None = None,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> Path:
    """Persist JSON training report; optionally save sklearn model artifact."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_id = report.get("report_id") or uuid.uuid4().hex
    report["report_id"] = report_id

    model_path: str | None = None
    if sklearn_result and sklearn_result.get("model") is not None:
        try:
            import joblib
        except ImportError:
            report.setdefault("warnings", []).append(
                "sklearn model trained but joblib unavailable; model not saved"
            )
        else:
            models_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = models_dir / f"{report_id}.joblib"
            joblib.dump(
                {
                    "model": sklearn_result["model"],
                    "scaler": sklearn_result.get("scaler"),
                    "model_type": sklearn_result.get("model_type"),
                    "features": list(FEATURE_COLUMNS),
                },
                artifact_path,
            )
            model_path = str(artifact_path)
            report["model_path"] = model_path

    output_path = reports_dir / f"{report_id}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return output_path


def build_training_report(
    *,
    coin: str,
    interval: str,
    horizon: int,
    threshold_bps: float,
    train_fraction: float,
    source_path: str,
    evaluation: dict[str, Any],
    baseline: dict[str, Any],
    sklearn_result: dict[str, Any],
    model_used: str,
) -> dict[str, Any]:
    return {
        "coin": coin,
        "interval": interval,
        "horizon": horizon,
        "threshold_bps": threshold_bps,
        "train_fraction": train_fraction,
        "source_path": source_path,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_used": model_used,
        "baseline_majority_class": baseline.get("majority_class"),
        "sklearn_available": sklearn_result.get("available", False),
        "sklearn_model_type": sklearn_result.get("model_type"),
        "sklearn_message": sklearn_result.get("message"),
        "evaluation": evaluation,
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "safety_statement": SAFETY_STATEMENT,
    }


def run_training_pipeline(
    *,
    coin: str,
    interval: str,
    horizon: int = 1,
    threshold_bps: float = 1.0,
    train_fraction: float = 0.70,
    data_root: Path = DEFAULT_DATA_ROOT,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    save_report: bool = True,
) -> dict[str, Any]:
    """End-to-end training pipeline for CLI and tests."""
    raw_df = load_training_data(coin, interval, data_root)
    featured_df = create_features(raw_df)
    labeled_df = create_labels(featured_df, horizon=horizon, threshold_bps=threshold_bps)
    model_df = _rows_ready_for_model(labeled_df)

    train_df, test_df = chronological_train_test_split(model_df, train_fraction)
    baseline = train_baseline_model(train_df)
    sklearn_result = train_optional_sklearn_model(train_df)

    majority_model: MajorityClassBaseline = baseline["majority_model"]
    baseline_predictions = majority_model.predict(test_df)
    sklearn_predictions = _predict_sklearn(sklearn_result, test_df)

    if sklearn_predictions is not None:
        predictions = sklearn_predictions
        model_used = sklearn_result.get("model_type") or "sklearn"
    else:
        predictions = baseline_predictions
        model_used = majority_model.model_type

    evaluation = evaluate_predictions(
        train_df=train_df,
        test_df=test_df,
        predictions=predictions,
        baseline_predictions=baseline_predictions,
    )

    source_path = str(resampled_csv_path(coin, interval, data_root))
    report = build_training_report(
        coin=coin,
        interval=interval,
        horizon=horizon,
        threshold_bps=threshold_bps,
        train_fraction=train_fraction,
        source_path=source_path,
        evaluation=evaluation,
        baseline=baseline,
        sklearn_result=sklearn_result,
        model_used=model_used,
    )

    report_path: Path | None = None
    if save_report:
        report_path = save_training_report(
            report,
            reports_dir=reports_dir,
            sklearn_result=sklearn_result,
            models_dir=models_dir,
        )
        report["report_path"] = str(report_path)

    return {
        "report": report,
        "train_df": train_df,
        "test_df": test_df,
        "baseline": baseline,
        "sklearn_result": sklearn_result,
        "predictions": predictions,
        "baseline_predictions": baseline_predictions,
        "evaluation": evaluation,
        "report_path": report_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a read-only bid/ask movement model on local CSV data "
            "(research-only, paper-only)."
        )
    )
    parser.add_argument("--coin", required=True, help="Coin symbol, e.g. BTC")
    parser.add_argument(
        "--interval",
        required=True,
        choices=sorted(SUPPORTED_INTERVALS),
        help="Resampled interval (15s or 1m)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Number of bars ahead for label creation (default: 1)",
    )
    parser.add_argument(
        "--threshold-bps",
        type=float,
        default=1.0,
        help="UP/DOWN threshold in basis points (default: 1.0)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="Chronological train split fraction (default: 0.70)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing bid/ask CSV files",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory for JSON training reports",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = run_training_pipeline(
        coin=args.coin.upper(),
        interval=args.interval,
        horizon=args.horizon,
        threshold_bps=args.threshold_bps,
        train_fraction=args.train_fraction,
        data_root=args.data_root,
        reports_dir=args.reports_dir,
    )
    evaluation = result["evaluation"]
    print(f"coin={args.coin.upper()} interval={args.interval} horizon={args.horizon}")
    print(f"train_rows={evaluation['train_rows']} test_rows={evaluation['test_rows']}")
    print(f"accuracy={evaluation['accuracy']:.4f} baseline_accuracy={evaluation['baseline_accuracy']:.4f}")
    print(f"model_used={result['report']['model_used']}")
    if result["sklearn_result"].get("message"):
        print(result["sklearn_result"]["message"])
    if result["report_path"] is not None:
        print(f"report_path={result['report_path']}")
    for warning in evaluation["warnings"]:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
