"""Focused offline tests for bid/ask model artifact I/O."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from bid_ask_model_io import (
    DEFAULT_MODELS_DIR,
    build_artifact_payload,
    feature_names_compatible,
    find_latest_model_artifact,
    load_model_artifact,
    predict_with_artifact,
    save_model_artifact,
)
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY
from bid_ask_train import FEATURE_COLUMNS

MODEL_IO_FILE = Path(__file__).resolve().parent.parent / "bid_ask_model_io.py"


class FakeModel:
    def __init__(self, prediction: str = "UP"):
        self.prediction = prediction

    def predict(self, X):
        return [self.prediction] * len(X)

    def predict_proba(self, X):
        return [[0.1, 0.2, 0.7]] * len(X)


class FakeScaler:
    def transform(self, X):
        return X


def _sample_row() -> pd.Series:
    return pd.Series(
        {
            "avg_spread_bps": 2.0,
            "avg_bbo_imbalance": 0.1,
            "mid_return_1": 0.5,
            "rolling_return_3": 1.0,
            "rolling_return_5": 1.5,
            "rolling_volatility_5": 0.3,
            "spread_change_1": 0.01,
            "bbo_imbalance_change_1": 0.02,
        }
    )


class TestBuildArtifactPayload:
    def test_payload_includes_required_fields(self) -> None:
        payload = build_artifact_payload(
            model=FakeModel(),
            scaler=FakeScaler(),
            feature_names=FEATURE_COLUMNS,
            coin="BTC",
            interval="15s",
            horizon=1,
            threshold_bps=1.0,
            train_rows=28,
            test_rows=12,
            baseline_accuracy=0.4,
            model_accuracy=0.6,
            model_type="FakeModel",
        )
        assert payload["coin"] == "BTC"
        assert payload["interval"] == "15s"
        assert payload["feature_names"] == list(FEATURE_COLUMNS)
        assert payload["label_order"] == ["UP", "DOWN", "FLAT"]
        assert payload["paper_only"] is True
        assert payload["execution_enabled"] is False


class TestSaveLoadArtifact:
    def test_save_and_load_model_artifact(self, tmp_path: Path) -> None:
        payload = build_artifact_payload(
            model=FakeModel("DOWN"),
            scaler=FakeScaler(),
            feature_names=FEATURE_COLUMNS,
            coin="ETH",
            interval="15s",
            horizon=1,
            threshold_bps=1.0,
            train_rows=10,
            test_rows=5,
            baseline_accuracy=0.5,
            model_accuracy=0.5,
            model_type="FakeModel",
            trained_at="2024-06-01T00:00:00+00:00",
        )
        path = save_model_artifact(payload, models_dir=tmp_path, artifact_id="test123")
        assert path.is_file()
        loaded = load_model_artifact(path)
        assert loaded is not None
        assert loaded["coin"] == "ETH"
        assert loaded["model_type"] == "FakeModel"

    def test_find_latest_model_artifact(self, tmp_path: Path) -> None:
        for trained_at, artifact_id in (
            ("2024-01-01T00:00:00+00:00", "older"),
            ("2024-06-01T00:00:00+00:00", "newer"),
        ):
            payload = build_artifact_payload(
                model=FakeModel(),
                scaler=FakeScaler(),
                feature_names=FEATURE_COLUMNS,
                coin="BTC",
                interval="15s",
                horizon=1,
                threshold_bps=1.0,
                train_rows=10,
                test_rows=5,
                baseline_accuracy=0.5,
                model_accuracy=0.5,
                model_type="FakeModel",
                trained_at=trained_at,
            )
            save_model_artifact(payload, models_dir=tmp_path, artifact_id=artifact_id)

        latest = find_latest_model_artifact("BTC", "15s", models_dir=tmp_path)
        assert latest is not None
        assert latest["trained_at"] == "2024-06-01T00:00:00+00:00"


class TestPredictWithArtifact:
    def test_predict_with_proba_confidence(self) -> None:
        artifact = build_artifact_payload(
            model=FakeModel("UP"),
            scaler=FakeScaler(),
            feature_names=FEATURE_COLUMNS,
            coin="BTC",
            interval="15s",
            horizon=1,
            threshold_bps=1.0,
            train_rows=10,
            test_rows=5,
            baseline_accuracy=0.5,
            model_accuracy=0.5,
            model_type="FakeModel",
        )
        signal, confidence = predict_with_artifact(artifact, _sample_row())
        assert signal == "UP"
        assert confidence == pytest.approx(0.7)

    def test_feature_names_compatible(self) -> None:
        artifact = {"feature_names": list(FEATURE_COLUMNS)}
        assert feature_names_compatible(artifact, FEATURE_COLUMNS) is True
        assert feature_names_compatible(artifact, ["avg_spread_bps"]) is False


class TestSafety:
    def test_no_forbidden_imports(self) -> None:
        source = MODEL_IO_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        lowered = source.lower()
        assert "subprocess" not in lowered
        for pattern in FORBIDDEN_SOURCE_PATTERNS:
            assert pattern not in lowered
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES

    def test_paper_only_constants(self) -> None:
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False
        assert DEFAULT_MODELS_DIR == Path("models/bid_ask")
