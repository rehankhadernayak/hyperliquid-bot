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

from bid_ask_features import (
    CANDIDATE_C_CORE_COLUMNS,
    build_candidate_c_core_metadata,
    create_candidate_c_core_features,
)
from bid_ask_model_io import (
    DEFAULT_MODELS_DIR as MODEL_IO_DEFAULT_MODELS_DIR,
    _dump_artifact,
    build_artifact_payload,
)
from bid_ask_resampler import (
    RESAMPLED_COLUMNS,
    normalize_source,
    resolve_bid_ask_data_dir,
    resampled_csv_path,
)
from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY

SAFETY_STATEMENT = (
    "Bid/ask movement training is read-only, paper-trading only, and research-only. "
    "No wallets, private keys, signing, leverage, margin, exchange actions, "
    "real orders, deposits, withdrawals, or execution endpoints are used. "
    "Training output is not connected to live trading and does not prove profitability."
)

DEFAULT_DATA_ROOT = Path("data/bid_ask_csv")
DEFAULT_REPORTS_DIR = Path("reports/bid_ask_training")
DEFAULT_MODELS_DIR = MODEL_IO_DEFAULT_MODELS_DIR

SUPPORTED_INTERVALS = frozenset({"15s", "1m"})
DEFAULT_FEATURE_SET = "basic"
TRAINING_SUPPORTED_FEATURE_SETS = frozenset({"basic", "candidate_c_core"})
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
MAX_CLASS_FRACTION = 0.70

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
    *,
    source: str = "local",
) -> pd.DataFrame:
    """Load resampled bid/ask CSV for the given coin and interval."""
    if interval not in SUPPORTED_INTERVALS:
        supported = ", ".join(sorted(SUPPORTED_INTERVALS))
        raise ValueError(f"Unsupported interval {interval!r}; use one of: {supported}")

    path = resampled_csv_path(coin, interval, data_root, source=source)
    if not path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [column for column in RESAMPLED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    return df.sort_values("start_timestamp_ms").reset_index(drop=True)


def get_training_feature_columns(feature_set: str) -> tuple[str, ...]:
    """Return ordered feature columns for the requested training feature set."""
    mode = str(feature_set).strip().lower()
    if mode not in TRAINING_SUPPORTED_FEATURE_SETS:
        supported = ", ".join(sorted(TRAINING_SUPPORTED_FEATURE_SETS))
        raise ValueError(f"Unsupported training feature_set {feature_set!r}; use one of: {supported}")
    if mode == "candidate_c_core":
        return CANDIDATE_C_CORE_COLUMNS
    return FEATURE_COLUMNS


def create_training_features(
    df: pd.DataFrame,
    *,
    feature_set: str = DEFAULT_FEATURE_SET,
    interval: str = "15s",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create model features for the requested training feature set."""
    mode = str(feature_set).strip().lower()
    if mode not in TRAINING_SUPPORTED_FEATURE_SETS:
        supported = ", ".join(sorted(TRAINING_SUPPORTED_FEATURE_SETS))
        raise ValueError(f"Unsupported training feature_set {feature_set!r}; use one of: {supported}")
    if mode == "candidate_c_core":
        featured_df, skipped = create_candidate_c_core_features(df, interval=interval)
        feature_columns = tuple(
            column
            for column in CANDIDATE_C_CORE_COLUMNS
            if column in featured_df.columns and column not in skipped
        )
        metadata = build_candidate_c_core_metadata(
            feature_columns=feature_columns,
            skipped_candidate_c_core_features=skipped,
        )
        return featured_df, metadata

    featured_df = create_features(df)
    metadata = {
        "feature_set": mode,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
        "skipped_microstructure_features": [],
        "skipped_candidate_c_core_features": [],
        "feature_warning": None,
    }
    return featured_df, metadata


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


def _feature_matrix(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    matrix = df[list(feature_columns)].copy()
    for column in (
        "avg_bbo_imbalance",
        "bbo_imbalance",
        "bbo_imbalance_change_1",
        "imbalance_change_1",
    ):
        if column in matrix.columns:
            matrix[column] = matrix[column].fillna(0.0)
    return matrix.astype(float)


def _rows_ready_for_model(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    required = list(feature_columns) + ["label", "future_return_bps", MID_PRICE_COLUMN]
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
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> dict[str, Any]:
    """Train sklearn classifier when available; otherwise return skip metadata."""
    if not _SKLEARN_AVAILABLE:
        return {
            "available": False,
            "model": None,
            "message": "sklearn not installed; using baseline models only",
        }

    X_train = _feature_matrix(train_df, feature_columns)
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
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> list[str] | None:
    model = sklearn_result.get("model")
    scaler = sklearn_result.get("scaler")
    if model is None or scaler is None:
        return None
    X_test = _feature_matrix(test_df, feature_columns)
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


def _normalize_label_counts(counts: dict[str, int] | None) -> dict[str, int]:
    if counts is None:
        return {label: 0 for label in LABELS}
    return {label: int(counts.get(label, 0)) for label in LABELS}


def _label_percentages(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {label: 0.0 for label in LABELS}
    return {label: counts[label] / total for label in LABELS}


def _combined_label_counts(
    train_counts: dict[str, int],
    test_counts: dict[str, int],
) -> dict[str, int]:
    return {
        label: train_counts[label] + test_counts[label]
        for label in LABELS
    }


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
                    f"Class imbalance in {split_name}: {label} is {fraction:.1%} of rows "
                    f"(under {MIN_CLASS_FRACTION:.0%})"
                )
            if fraction > MAX_CLASS_FRACTION:
                warnings.append(
                    f"Class dominance in {split_name}: {label} is {fraction:.1%} of rows "
                    f"(over {MAX_CLASS_FRACTION:.0%})"
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


def _model_artifact_path(
    *,
    coin: str,
    interval: str,
    artifact_id: str,
    models_dir: Path,
    source: str = "local",
) -> Path:
    normalized_source = normalize_source(source)
    if normalized_source == "historical":
        filename = f"{coin.upper()}_historical_{interval}_{artifact_id}.joblib"
    else:
        filename = f"{coin.upper()}_{interval}_{artifact_id}.joblib"
    return models_dir / filename


def save_training_model_artifact(
    payload: dict[str, Any],
    models_dir: Path = DEFAULT_MODELS_DIR,
    *,
    artifact_id: str | None = None,
    source: str = "local",
) -> Path:
    """Persist a training artifact without overwriting local model names."""
    models_dir.mkdir(parents=True, exist_ok=True)
    coin = str(payload.get("coin", "UNKNOWN")).upper()
    interval = str(payload.get("interval", "unknown"))
    artifact_id = artifact_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = _model_artifact_path(
        coin=coin,
        interval=interval,
        artifact_id=artifact_id,
        models_dir=models_dir,
        source=source,
    )
    _dump_artifact(payload, path)
    return path


def save_training_report(
    report: dict[str, Any],
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    *,
    sklearn_result: dict[str, Any] | None = None,
    models_dir: Path = DEFAULT_MODELS_DIR,
    artifact_payload: dict[str, Any] | None = None,
    source: str = "local",
) -> Path:
    """Persist JSON training report; optionally save sklearn model artifact."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_id = report.get("report_id") or uuid.uuid4().hex
    report["report_id"] = report_id

    if artifact_payload is not None and sklearn_result and sklearn_result.get("model") is not None:
        try:
            artifact_path = save_training_model_artifact(
                artifact_payload,
                models_dir=models_dir,
                artifact_id=report_id,
                source=source,
            )
        except Exception as exc:
            report.setdefault("warnings", []).append(
                f"Failed to save model artifact: {exc}"
            )
        else:
            artifact_str = str(artifact_path)
            report["model_artifact_path"] = artifact_str
            report["model_path"] = artifact_str

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
    source: str = "local",
    data_dir: str | None = None,
    feature_set: str = DEFAULT_FEATURE_SET,
    feature_columns: tuple[str, ...] | None = None,
    skipped_candidate_c_core_features: list[str] | None = None,
) -> dict[str, Any]:
    model_accuracy = float(evaluation.get("accuracy", 0.0))
    baseline_accuracy = float(evaluation.get("baseline_accuracy", 0.0))
    model_edge_vs_baseline = model_accuracy - baseline_accuracy

    warnings: list[str] = list(evaluation.get("warnings", []))
    if sklearn_result.get("model") is not None and model_accuracy <= baseline_accuracy:
        warnings.append(
            f"Model accuracy ({model_accuracy:.4f}) does not exceed "
            f"baseline ({baseline_accuracy:.4f})"
        )

    label_counts_data = evaluation.get("label_counts", {})
    train_label_counts = _normalize_label_counts(label_counts_data.get("train"))
    test_label_counts = _normalize_label_counts(label_counts_data.get("test"))
    overall_label_counts = _combined_label_counts(train_label_counts, test_label_counts)

    resolved_feature_columns = tuple(
        feature_columns or get_training_feature_columns(feature_set)
    )

    return {
        "coin": coin,
        "interval": interval,
        "source": normalize_source(source),
        "data_dir": data_dir,
        "feature_set": str(feature_set).strip().lower(),
        "feature_columns": list(resolved_feature_columns),
        "feature_count": len(resolved_feature_columns),
        "skipped_candidate_c_core_features": list(skipped_candidate_c_core_features or []),
        "horizon": horizon,
        "threshold_bps": threshold_bps,
        "train_fraction": train_fraction,
        "source_path": source_path,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_used": model_used,
        "model_accuracy": model_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "model_edge_vs_baseline": model_edge_vs_baseline,
        "label_counts": overall_label_counts,
        "label_percentages": _label_percentages(overall_label_counts),
        "train_label_counts": train_label_counts,
        "test_label_counts": test_label_counts,
        "train_label_percentages": _label_percentages(train_label_counts),
        "test_label_percentages": _label_percentages(test_label_counts),
        "model_artifact_path": None,
        "baseline_majority_class": baseline.get("majority_class"),
        "sklearn_available": sklearn_result.get("available", False),
        "sklearn_model_type": sklearn_result.get("model_type"),
        "sklearn_message": sklearn_result.get("message"),
        "evaluation": evaluation,
        "warnings": warnings,
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
    max_rows: int | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    save_report: bool = True,
    source: str = "local",
    feature_set: str = DEFAULT_FEATURE_SET,
) -> dict[str, Any]:
    """End-to-end training pipeline for CLI and tests."""
    normalized_source = normalize_source(source)
    data_dir = resolve_bid_ask_data_dir(interval, normalized_source, data_root)
    raw_df = load_training_data(coin, interval, data_root, source=normalized_source)
    if max_rows is not None and max_rows > 0:
        raw_df = raw_df.iloc[:max_rows].copy()
    featured_df, feature_meta = create_training_features(
        raw_df,
        feature_set=feature_set,
        interval=interval,
    )
    feature_columns = tuple(feature_meta["feature_columns"])
    labeled_df = create_labels(featured_df, horizon=horizon, threshold_bps=threshold_bps)
    model_df = _rows_ready_for_model(labeled_df, feature_columns)

    train_df, test_df = chronological_train_test_split(model_df, train_fraction)
    baseline = train_baseline_model(train_df)
    sklearn_result = train_optional_sklearn_model(train_df, feature_columns)

    majority_model: MajorityClassBaseline = baseline["majority_model"]
    baseline_predictions = majority_model.predict(test_df)
    sklearn_predictions = _predict_sklearn(sklearn_result, test_df, feature_columns)

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
        features_used=feature_columns,
    )

    source_path = str(resampled_csv_path(coin, interval, data_root, source=normalized_source))
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
        source=normalized_source,
        data_dir=str(data_dir),
        feature_set=feature_set,
        feature_columns=feature_columns,
        skipped_candidate_c_core_features=feature_meta.get(
            "skipped_candidate_c_core_features", []
        ),
    )

    artifact_payload: dict[str, Any] | None = None
    if sklearn_result.get("model") is not None:
        artifact_payload = build_artifact_payload(
            model=sklearn_result["model"],
            scaler=sklearn_result.get("scaler"),
            feature_names=feature_columns,
            coin=coin,
            interval=interval,
            horizon=horizon,
            threshold_bps=threshold_bps,
            train_rows=evaluation["train_rows"],
            test_rows=evaluation["test_rows"],
            baseline_accuracy=evaluation["baseline_accuracy"],
            model_accuracy=evaluation["accuracy"],
            model_type=str(sklearn_result.get("model_type") or model_used),
        )
        artifact_payload["source"] = normalized_source
        artifact_payload["data_dir"] = str(data_dir)

    report_path: Path | None = None
    if save_report:
        report_path = save_training_report(
            report,
            reports_dir=reports_dir,
            sklearn_result=sklearn_result,
            models_dir=models_dir,
            artifact_payload=artifact_payload,
            source=normalized_source,
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
        "--source",
        default="local",
        choices=("local", "historical"),
        help="Data source: local recorder CSVs or historical archive CSVs (default: local)",
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
    parser.add_argument(
        "--feature-set",
        default=DEFAULT_FEATURE_SET,
        choices=sorted(TRAINING_SUPPORTED_FEATURE_SETS),
        help="Feature set: basic (default) or candidate_c_core",
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
        source=args.source,
        feature_set=args.feature_set,
    )
    evaluation = result["evaluation"]
    report = result["report"]
    print(f"coin={args.coin.upper()} interval={args.interval} horizon={args.horizon}")
    print(f"feature_set={report.get('feature_set', DEFAULT_FEATURE_SET)}")
    print(f"feature_count={report.get('feature_count', len(FEATURE_COLUMNS))}")
    print(f"source={report['source']} data_dir={report['data_dir']}")
    print(f"train_rows={evaluation['train_rows']} test_rows={evaluation['test_rows']}")
    print(f"accuracy={evaluation['accuracy']:.4f} baseline_accuracy={evaluation['baseline_accuracy']:.4f}")
    print(f"model_edge_vs_baseline={result['report']['model_edge_vs_baseline']:.4f}")
    print(f"model_used={result['report']['model_used']}")
    if result["report"].get("model_artifact_path"):
        print(f"model_artifact_path={result['report']['model_artifact_path']}")
    if result["sklearn_result"].get("message"):
        print(result["sklearn_result"]["message"])
    if result["report_path"] is not None:
        print(f"report_path={result['report_path']}")
    for warning in evaluation["warnings"]:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
