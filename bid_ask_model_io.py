"""Read-only bid/ask model artifact I/O (research-only, paper-only)."""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY

DEFAULT_MODELS_DIR = Path("models/bid_ask")
LABEL_ORDER = ("UP", "DOWN", "FLAT")


def build_artifact_payload(
    *,
    model: Any,
    scaler: Any | None,
    feature_names: tuple[str, ...] | list[str],
    coin: str,
    interval: str,
    horizon: int,
    threshold_bps: float,
    train_rows: int,
    test_rows: int,
    baseline_accuracy: float,
    model_accuracy: float,
    model_type: str,
    trained_at: str | None = None,
) -> dict[str, Any]:
    """Build a serializable model artifact payload."""
    return {
        "model": model,
        "scaler": scaler,
        "model_type": model_type,
        "feature_names": list(feature_names),
        "label_order": list(LABEL_ORDER),
        "coin": coin.upper(),
        "interval": interval,
        "horizon": horizon,
        "threshold_bps": threshold_bps,
        "trained_at": trained_at or datetime.now(timezone.utc).isoformat(),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "baseline_accuracy": baseline_accuracy,
        "model_accuracy": model_accuracy,
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
    }


def _dump_artifact(payload: dict[str, Any], path: Path) -> None:
    try:
        import joblib

        joblib.dump(payload, path)
    except ImportError:
        with path.open("wb") as handle:
            pickle.dump(payload, handle)


def _load_artifact_file(path: Path) -> Any:
    try:
        import joblib

        return joblib.load(path)
    except ImportError:
        with path.open("rb") as handle:
            return pickle.load(handle)


def save_model_artifact(
    payload: dict[str, Any],
    models_dir: Path = DEFAULT_MODELS_DIR,
    *,
    artifact_id: str | None = None,
) -> Path:
    """Persist model artifact to models/bid_ask/."""
    models_dir.mkdir(parents=True, exist_ok=True)
    coin = str(payload.get("coin", "UNKNOWN")).upper()
    interval = str(payload.get("interval", "unknown"))
    artifact_id = artifact_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = models_dir / f"{coin}_{interval}_{artifact_id}.joblib"
    _dump_artifact(payload, path)
    return path


def load_model_artifact(path: Path | str) -> dict[str, Any] | None:
    """Load a model artifact from disk; return None on failure."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        return None
    try:
        artifact = _load_artifact_file(artifact_path)
    except Exception:
        return None
    if not isinstance(artifact, dict) or artifact.get("model") is None:
        return None
    return artifact


def find_latest_model_artifact(
    coin: str,
    interval: str,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> dict[str, Any] | None:
    """Return the newest artifact for coin/interval, or None if missing."""
    if not models_dir.is_dir():
        return None

    coin = coin.upper()
    prefix = f"{coin}_{interval}_"
    candidates: list[tuple[str, Path]] = []
    for path in models_dir.glob("*.joblib"):
        if not path.name.startswith(prefix):
            continue
        artifact = load_model_artifact(path)
        if artifact is None:
            continue
        if artifact.get("coin", "").upper() != coin:
            continue
        if artifact.get("interval") != interval:
            continue
        trained_at = str(artifact.get("trained_at") or path.stat().st_mtime)
        candidates.append((trained_at, path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    latest_path = candidates[-1][1]
    loaded = load_model_artifact(latest_path)
    if loaded is not None:
        loaded["artifact_path"] = str(latest_path)
    return loaded


def feature_names_compatible(
    artifact: dict[str, Any],
    expected_features: tuple[str, ...] | list[str],
) -> bool:
    """Return True when artifact feature_names match expected columns."""
    artifact_features = artifact.get("feature_names")
    if not artifact_features:
        return False
    return list(artifact_features) == list(expected_features)


def _feature_matrix_from_row(
    row: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    matrix = pd.DataFrame([{name: row[name] for name in feature_names}])
    if "avg_bbo_imbalance" in matrix.columns:
        matrix["avg_bbo_imbalance"] = matrix["avg_bbo_imbalance"].fillna(0.0)
    if "bbo_imbalance_change_1" in matrix.columns:
        matrix["bbo_imbalance_change_1"] = matrix["bbo_imbalance_change_1"].fillna(0.0)
    return matrix.astype(float)


def predict_with_artifact(
    artifact: dict[str, Any],
    row: pd.Series,
) -> tuple[str, float]:
    """Run model.predict (and predict_proba when available) on one feature row."""
    feature_names = list(artifact.get("feature_names") or [])
    features = _feature_matrix_from_row(row, feature_names)

    model = artifact["model"]
    scaler = artifact.get("scaler")
    X = scaler.transform(features) if scaler is not None else features.to_numpy()

    prediction = model.predict(X)[0]
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(max(proba))
    return str(prediction), confidence


def predict_batch_with_artifact(
    artifact: dict[str, Any],
    df: pd.DataFrame,
) -> tuple[list[str], list[float]]:
    """Run predictions for multiple rows."""
    feature_names = list(artifact.get("feature_names") or [])
    matrix = df[list(feature_names)].copy()
    if "avg_bbo_imbalance" in matrix.columns:
        matrix["avg_bbo_imbalance"] = matrix["avg_bbo_imbalance"].fillna(0.0)
    if "bbo_imbalance_change_1" in matrix.columns:
        matrix["bbo_imbalance_change_1"] = matrix["bbo_imbalance_change_1"].fillna(0.0)
    matrix = matrix.astype(float)

    model = artifact["model"]
    scaler = artifact.get("scaler")
    X = scaler.transform(matrix) if scaler is not None else matrix.to_numpy()

    predictions = [str(label) for label in model.predict(X)]
    confidences: list[float] = []
    if hasattr(model, "predict_proba"):
        for proba in model.predict_proba(X):
            confidences.append(float(max(proba)))
    else:
        confidences = [0.0] * len(predictions)
    return predictions, confidences
