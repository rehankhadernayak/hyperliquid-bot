"""Focused offline tests for paper-only bid/ask decision logging."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from bid_ask_resampler import RESAMPLED_COLUMNS, resampled_csv_path
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY
from bid_ask_decision_logger import (
    DECISION_COLUMNS,
    compute_features_for_latest_rows,
    generate_paper_decisions,
    load_latest_training_report,
    load_signal_source_csv,
    main,
    run_decision_logging,
    summarize_decision_log,
    write_decision_log,
)
from bid_ask_train import FEATURE_COLUMNS, create_labels, create_features

LOGGER_FILE = Path(__file__).resolve().parent.parent / "bid_ask_decision_logger.py"

BASE_MS = 1_718_000_040_000


def _make_resampled_rows(
    *,
    coin: str = "BTC",
    count: int,
    start_ms: int = BASE_MS,
    step_ms: int = 15_000,
    mid_start: float = 100.0,
    mid_step: float = 0.5,
    spread_bps: float = 2.0,
    imbalance: float = 0.2,
) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        start = start_ms + index * step_ms
        end = start + step_ms - 1
        mid = mid_start + index * mid_step
        rows.append(
            {
                "start_timestamp_ms": start,
                "end_timestamp_ms": end,
                "timestamp_iso": f"2024-06-10T00:00:{index:02d}Z",
                "coin": coin,
                "open_mid": mid,
                "high_mid": mid + 0.1,
                "low_mid": mid - 0.1,
                "close_mid": mid,
                "avg_bid": mid - 0.5,
                "avg_ask": mid + 0.5,
                "avg_mid": mid,
                "avg_spread": 1.0,
                "avg_spread_bps": spread_bps + (index % 3) * 0.1,
                "avg_bbo_imbalance": imbalance + index * 0.01,
                "row_count": 15,
                "missing_seconds": 0,
            }
        )
    return rows


def _write_resampled_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS)).to_csv(path, index=False)


def _write_training_report(
    reports_dir: Path,
    *,
    coin: str = "BTC",
    interval: str = "15s",
    generated_at: str,
    baseline_majority_class: str = "UP",
    model_path: str | None = None,
    report_id: str | None = None,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_id = report_id or f"report-{generated_at.replace(':', '').replace('+', '')}"
    payload = {
        "report_id": report_id,
        "coin": coin,
        "interval": interval,
        "generated_at": generated_at,
        "horizon": 1,
        "threshold_bps": 1.0,
        "baseline_majority_class": baseline_majority_class,
        "model_used": "majority_class",
        "model_path": model_path,
        "paper_only": True,
        "execution_enabled": False,
    }
    path = reports_dir / f"{report_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "bid_ask_csv"


@pytest.fixture
def training_reports_dir(tmp_path: Path) -> Path:
    return tmp_path / "reports" / "bid_ask_training"


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "reports" / "bid_ask_decisions"


@pytest.fixture
def training_csv(data_root: Path) -> Path:
    rows = _make_resampled_rows(count=40)
    path = resampled_csv_path("BTC", "15s", data_root)
    _write_resampled_csv(path, rows)
    return path


class TestLoadLatestTrainingReport:
    def test_load_latest_training_report(
        self, training_reports_dir: Path
    ) -> None:
        _write_training_report(
            training_reports_dir,
            generated_at="2024-01-01T00:00:00+00:00",
            baseline_majority_class="FLAT",
            report_id="older",
        )
        _write_training_report(
            training_reports_dir,
            generated_at="2024-06-01T00:00:00+00:00",
            baseline_majority_class="UP",
            report_id="newer",
        )
        report = load_latest_training_report("BTC", training_reports_dir)
        assert report is not None
        assert report["baseline_majority_class"] == "UP"
        assert report["report_id"] == "newer"

    def test_missing_report_returns_none(self, training_reports_dir: Path) -> None:
        assert load_latest_training_report("BTC", training_reports_dir) is None


class TestLoadSignalSourceCsv:
    def test_load_signal_source_csv(self, data_root: Path, training_csv: Path) -> None:
        df = load_signal_source_csv("BTC", "15s", data_root)
        assert len(df) == 40
        assert list(df.columns) == list(RESAMPLED_COLUMNS)
        assert df["start_timestamp_ms"].is_monotonic_increasing


class TestComputeFeatures:
    def test_compute_features_for_latest_rows(
        self, data_root: Path, training_csv: Path
    ) -> None:
        df = load_signal_source_csv("BTC", "15s", data_root)
        featured = compute_features_for_latest_rows(df)
        for column in FEATURE_COLUMNS:
            assert column in featured.columns

        row = featured.iloc[10]
        expected_return = (
            (row["close_mid"] - featured.iloc[9]["close_mid"])
            / featured.iloc[9]["close_mid"]
            * 10_000.0
        )
        assert row["mid_return_1"] == pytest.approx(expected_return)


class TestFutureLabelCalculation:
    def test_future_label_calculation(
        self, data_root: Path, training_csv: Path
    ) -> None:
        df = create_features(load_signal_source_csv("BTC", "15s", data_root))
        labeled = create_labels(df, horizon=1, threshold_bps=1.0)

        for index in range(len(labeled) - 1):
            current_mid = labeled.iloc[index]["close_mid"]
            future_mid = labeled.iloc[index + 1]["close_mid"]
            expected = (future_mid - current_mid) / current_mid * 10_000.0
            assert labeled.iloc[index]["future_return_bps"] == pytest.approx(expected)

            if expected > 1.0:
                assert labeled.iloc[index]["label"] == "UP"
            elif expected < -1.0:
                assert labeled.iloc[index]["label"] == "DOWN"
            else:
                assert labeled.iloc[index]["label"] == "FLAT"


class TestGeneratePaperDecisions:
    def test_generate_paper_decisions(
        self,
        data_root: Path,
        training_csv: Path,
        training_reports_dir: Path,
    ) -> None:
        _write_training_report(
            training_reports_dir,
            generated_at="2024-06-01T00:00:00+00:00",
            baseline_majority_class="UP",
        )
        decisions, warnings = generate_paper_decisions(
            "BTC",
            interval="15s",
            horizon=1,
            max_rows=10,
            data_root=data_root,
            training_reports_dir=training_reports_dir,
        )
        assert len(decisions) == 10
        assert any("Research-only" in warning for warning in warnings)
        first = decisions[0]
        for column in DECISION_COLUMNS:
            assert column in first
        assert first["paper_only"] is True
        assert first["execution_enabled"] is False
        assert first["baseline_signal"] == "UP"
        assert first["actual_label"] in {"UP", "DOWN", "FLAT"}

    def test_fallback_baseline_behavior(
        self,
        data_root: Path,
        training_csv: Path,
        training_reports_dir: Path,
    ) -> None:
        _write_training_report(
            training_reports_dir,
            generated_at="2024-06-01T00:00:00+00:00",
            baseline_majority_class="DOWN",
            model_path="models/missing.joblib",
        )
        decisions, warnings = generate_paper_decisions(
            "BTC",
            interval="15s",
            max_rows=5,
            data_root=data_root,
            training_reports_dir=training_reports_dir,
        )
        assert len(decisions) == 5
        assert all(decision["model_signal"] == "baseline_fallback" for decision in decisions)
        assert any("artifact missing" in warning for warning in warnings)

    def test_missing_model_artifact_handled_safely(
        self,
        data_root: Path,
        training_csv: Path,
        training_reports_dir: Path,
    ) -> None:
        decisions, warnings = generate_paper_decisions(
            "BTC",
            interval="15s",
            max_rows=3,
            data_root=data_root,
            training_reports_dir=training_reports_dir,
        )
        assert len(decisions) == 3
        assert all(decision["model_signal"] == "baseline_fallback" for decision in decisions)
        assert any("No training report" in warning for warning in warnings)


class TestWriteDecisionLog:
    def test_write_decision_log(
        self,
        data_root: Path,
        training_csv: Path,
        training_reports_dir: Path,
        output_dir: Path,
    ) -> None:
        decisions, _ = generate_paper_decisions(
            "BTC",
            interval="15s",
            max_rows=5,
            data_root=data_root,
            training_reports_dir=training_reports_dir,
        )
        path = write_decision_log(decisions, output_dir=output_dir)
        assert path.is_file()
        loaded = pd.read_csv(path)
        assert list(loaded.columns) == list(DECISION_COLUMNS)
        assert len(loaded) == 5


class TestSummarizeDecisionLog:
    def test_summarize_decision_log(
        self,
        data_root: Path,
        training_csv: Path,
        training_reports_dir: Path,
        output_dir: Path,
    ) -> None:
        _write_training_report(
            training_reports_dir,
            generated_at="2024-06-01T00:00:00+00:00",
            baseline_majority_class="UP",
        )
        decisions, _ = generate_paper_decisions(
            "BTC",
            interval="15s",
            max_rows=10,
            data_root=data_root,
            training_reports_dir=training_reports_dir,
        )
        path = write_decision_log(decisions, output_dir=output_dir)
        summary = summarize_decision_log(path)

        assert summary["total_decisions"] == 10
        assert 0.0 <= summary["model_accuracy"] <= 1.0
        assert 0.0 <= summary["baseline_accuracy"] <= 1.0
        assert set(summary["label_counts"]) == {"UP", "DOWN", "FLAT"}
        assert summary["paper_only"] is True
        assert summary["execution_enabled"] is False
        assert any("Paper-only" in warning for warning in summary["warnings"])


class TestRunDecisionLogging:
    def test_run_decision_logging(
        self,
        data_root: Path,
        training_reports_dir: Path,
        output_dir: Path,
    ) -> None:
        for coin in ("BTC", "ETH"):
            path = resampled_csv_path(coin, "15s", data_root)
            _write_resampled_csv(path, _make_resampled_rows(coin=coin, count=30))
            _write_training_report(
                training_reports_dir,
                coin=coin,
                generated_at="2024-06-01T00:00:00+00:00",
            )

        summary = run_decision_logging(
            coins=["BTC", "ETH"],
            interval="15s",
            max_rows=5,
            data_root=data_root,
            training_reports_dir=training_reports_dir,
            output_dir=output_dir,
        )
        assert summary["total_decisions"] == 10
        assert summary["per_coin_counts"]["BTC"] == 5
        assert summary["per_coin_counts"]["ETH"] == 5
        assert Path(summary["output_path"]).is_file()


class TestCli:
    def test_main_prints_summary(
        self,
        data_root: Path,
        training_reports_dir: Path,
        output_dir: Path,
        capsys,
    ) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        _write_resampled_csv(path, _make_resampled_rows(count=30))
        _write_training_report(
            training_reports_dir,
            generated_at="2024-06-01T00:00:00+00:00",
        )
        exit_code = main(
            [
                "--coins",
                "BTC",
                "--interval",
                "15s",
                "--horizon",
                "1",
                "--max-rows",
                "5",
                "--data-root",
                str(data_root),
                "--training-reports-dir",
                str(training_reports_dir),
                "--output-dir",
                str(output_dir),
            ]
        )
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "output_path=" in captured.out
        assert "model_accuracy=" in captured.out
        assert "baseline_accuracy=" in captured.out
        assert "Paper-only. No trades placed." in captured.out


class TestSafety:
    def test_no_wallet_signing_order_execution_code(self) -> None:
        source = LOGGER_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        lowered = source.lower()
        assert "requests" not in lowered
        assert "urllib" not in lowered
        assert "socket" not in lowered
        assert "httpx" not in lowered
        assert "subprocess" not in lowered
        for pattern in FORBIDDEN_SOURCE_PATTERNS:
            assert pattern not in lowered
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES

    def test_no_network_calls(self) -> None:
        source = LOGGER_FILE.read_text(encoding="utf-8").lower()
        forbidden = (
            "requests.get",
            "requests.post",
            "urllib.request",
            "http.client",
            "aiohttp",
            "websocket",
        )
        for pattern in forbidden:
            assert pattern not in source

    def test_no_subprocess_calls(self) -> None:
        source = LOGGER_FILE.read_text(encoding="utf-8").lower()
        assert "subprocess" not in source
        assert "os.system" not in source
        assert "popen" not in source

    def test_paper_only_constants(self) -> None:
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False

    def test_decisions_always_paper_only(
        self,
        data_root: Path,
        training_csv: Path,
        training_reports_dir: Path,
    ) -> None:
        decisions, _ = generate_paper_decisions(
            "BTC",
            interval="15s",
            max_rows=3,
            data_root=data_root,
            training_reports_dir=training_reports_dir,
        )
        for decision in decisions:
            assert decision["paper_only"] is True
            assert decision["execution_enabled"] is False
