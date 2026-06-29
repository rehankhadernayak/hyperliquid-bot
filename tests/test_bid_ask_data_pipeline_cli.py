"""CLI tests for bid_ask_data_pipeline (offline, monkeypatched)."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

import bid_ask_data_pipeline as pipeline
import bid_ask_extractor
import bid_ask_resampler
import bid_ask_train
import bid_ask_validator
from bid_ask_extractor import ExtractionReport
from bid_ask_schema import FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS
from bid_ask_validator import ValidationReport

PIPELINE_FILE = Path(__file__).resolve().parent.parent / "bid_ask_data_pipeline.py"


def _extraction_report(coin: str, *, row_count: int = 10) -> ExtractionReport:
    return ExtractionReport(
        coin=coin,
        source_files=["/tmp/source.parquet"],
        output_path=f"data/bid_ask_csv/1s/{coin}_1s.csv",
        row_count=row_count,
        coverage_start_ms=1_000,
        coverage_end_ms=2_000,
        missing_seconds=0,
        duplicate_timestamps=0,
        invalid_rows_skipped=0,
    )


def _validation_report(
    coin: str,
    interval: str,
    *,
    passed: bool = True,
) -> ValidationReport:
    return ValidationReport(
        coin=coin,
        interval=interval,
        row_count=5,
        coverage_start=1_000,
        coverage_end=2_000,
        duplicate_timestamp_count=0,
        missing_periods_count=0,
        invalid_bid_ask_count=0,
        negative_spread_count=0,
        extreme_spread_count=0,
        warnings=[],
        passed=passed,
        input_path=f"data/bid_ask_csv/{interval}/{coin}_{interval}.csv",
        report_path=f"data/bid_ask_csv/reports/{coin}_{interval}_validation.json",
    )


class TestExtractCommand:
    def test_extract_calls_extractor_for_each_coin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []

        def fake_extract(coin: str, **kwargs: object) -> ExtractionReport:
            calls.append(coin)
            return _extraction_report(coin)

        monkeypatch.setattr(bid_ask_extractor, "extract_coin_bid_ask_1s", fake_extract)
        pipeline.main(["extract", "--coins", "BTC", "ETH", "SOL", "--source", "local", "--interval", "1s"])
        assert calls == ["BTC", "ETH", "SOL"]

    def test_historical_source_prints_skip_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        pipeline.main(
            [
                "extract",
                "--coins",
                "BTC",
                "--source",
                "historical",
                "--interval",
                "1s",
                "--dry-run",
            ]
        )
        captured = capsys.readouterr().out
        assert pipeline.HISTORICAL_EXTRACT_SKIP_MESSAGE in captured
        assert "skipped=true" in captured

    def test_all_time_flag_prints_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        pipeline.main(["extract", "--coins", "BTC", "--source", "local", "--interval", "1s", "--all-time", "--dry-run"])
        captured = capsys.readouterr().out
        assert "prefer --source historical" in captured


class TestHistoricalSource:
    def test_validate_uses_historical_folders(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, str, str]] = []

        def fake_validate(coin: str, interval: str, **kwargs: object) -> ValidationReport:
            calls.append((coin, interval, str(kwargs.get("source", "local"))))
            return _validation_report(coin, interval)

        monkeypatch.setattr(bid_ask_validator, "validate_coin_interval", fake_validate)
        pipeline.main(
            [
                "validate",
                "--coins",
                "SOL",
                "--interval",
                "1m",
                "--source",
                "historical",
            ]
        )
        assert calls == [("SOL", "1m", "historical")]

    def test_resample_uses_historical_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, str]] = []

        def fake_resample(coin: str, intervals: tuple[str, ...] = ("15s", "1m"), **kwargs: object):
            calls.append((coin, str(kwargs.get("source", "local"))))
            return {
                interval: {
                    "output_path": Path(f"/tmp/{coin}_{interval}.csv"),
                    "report_path": Path(f"/tmp/{coin}_{interval}_report.json"),
                }
                for interval in intervals
            }

        monkeypatch.setattr(bid_ask_resampler, "resample_coin", fake_resample)
        pipeline.main(
            [
                "resample",
                "--coins",
                "BTC",
                "--from-interval",
                "1s",
                "--to-interval",
                "all",
                "--source",
                "historical",
            ]
        )
        assert calls == [("BTC", "historical")]

    def test_train_uses_historical_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []

        def fake_train(**kwargs: object) -> dict[str, object]:
            calls.append(str(kwargs.get("source", "local")))
            return {
                "evaluation": {"train_rows": 7, "test_rows": 3, "warnings": []},
                "report": {"model_used": "majority_class", "source": "historical"},
                "report_path": Path("/tmp/report.json"),
            }

        monkeypatch.setattr(bid_ask_train, "run_training_pipeline", fake_train)
        pipeline.main(["train", "--coin", "SOL", "--interval", "1m", "--horizon", "5", "--source", "historical"])
        assert calls == ["historical"]


class TestValidateCommand:
    def test_validate_calls_validator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, str]] = []

        def fake_validate(coin: str, interval: str, **kwargs: object) -> ValidationReport:
            calls.append((coin, interval))
            return _validation_report(coin, interval)

        monkeypatch.setattr(bid_ask_validator, "validate_coin_interval", fake_validate)
        pipeline.main(["validate", "--coins", "BTC", "ETH", "--interval", "1s"])
        assert calls == [("BTC", "1s"), ("ETH", "1s")]


class TestResampleCommand:
    def test_resample_all_calls_both_intervals(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, tuple[str, ...]]] = []

        def fake_resample(
            coin: str,
            intervals: tuple[str, ...] = ("15s", "1m"),
            **kwargs: object,
        ) -> dict[str, dict[str, Path]]:
            calls.append((coin, intervals))
            return {
                interval: {
                    "output_path": Path(f"/tmp/{coin}_{interval}.csv"),
                    "report_path": Path(f"/tmp/{coin}_{interval}_report.json"),
                }
                for interval in intervals
            }

        monkeypatch.setattr(bid_ask_resampler, "resample_coin", fake_resample)
        pipeline.main(
            [
                "resample",
                "--coins",
                "BTC",
                "ETH",
                "--from-interval",
                "1s",
                "--to-interval",
                "all",
            ]
        )
        assert calls == [("BTC", ("15s", "1m")), ("ETH", ("15s", "1m"))]


class TestTrainCommand:
    def test_train_calls_trainer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, str, int]] = []

        def fake_train(**kwargs: object) -> dict[str, object]:
            calls.append((str(kwargs["coin"]), str(kwargs["interval"]), int(kwargs["horizon"])))
            return {
                "evaluation": {"train_rows": 7, "test_rows": 3, "warnings": []},
                "report": {"model_used": "majority_class"},
                "report_path": Path("/tmp/report.json"),
            }

        monkeypatch.setattr(bid_ask_train, "run_training_pipeline", fake_train)
        pipeline.main(["train", "--coin", "BTC", "--interval", "15s", "--horizon", "1"])
        assert calls == [("BTC", "15s", 1)]


class TestRunOneWorkflow:
    def test_run_one_step_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        order: list[str] = []

        monkeypatch.setattr(
            bid_ask_extractor,
            "extract_coin_bid_ask_1s",
            lambda coin, **kwargs: (order.append(f"extract:{coin}") or _extraction_report(coin)),
        )

        def fake_validate(coin: str, interval: str, **kwargs: object) -> ValidationReport:
            order.append(f"validate:{coin}:{interval}")
            return _validation_report(coin, interval)

        monkeypatch.setattr(bid_ask_validator, "validate_coin_interval", fake_validate)

        def fake_resample(coin: str, intervals: tuple[str, ...] = ("15s", "1m"), **kwargs: object):
            order.append(f"resample:{coin}:{','.join(intervals)}")
            return {
                interval: {
                    "output_path": Path(f"/tmp/{coin}_{interval}.csv"),
                    "report_path": Path(f"/tmp/{coin}_{interval}_report.json"),
                }
                for interval in intervals
            }

        monkeypatch.setattr(bid_ask_resampler, "resample_coin", fake_resample)
        monkeypatch.setattr(
            bid_ask_train,
            "run_training_pipeline",
            lambda **kwargs: (
                order.append(f"train:{kwargs['coin']}:{kwargs['interval']}") or {
                    "evaluation": {"train_rows": 1, "test_rows": 1, "warnings": []},
                    "report": {},
                    "report_path": Path("/tmp/train_report.json"),
                }
            ),
        )

        pipeline.main(["run-one", "--coin", "BTC", "--train-interval", "15s", "--horizon", "1"])

        assert order == [
            "extract:BTC",
            "validate:BTC:1s",
            "resample:BTC:15s,1m",
            "validate:BTC:15s",
            "validate:BTC:1m",
            "train:BTC:15s",
        ]

    def test_run_one_historical_skips_extract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        order: list[str] = []

        monkeypatch.setattr(
            bid_ask_extractor,
            "extract_coin_bid_ask_1s",
            lambda coin, **kwargs: (_ for _ in ()).throw(
                AssertionError("extract should not run for historical source")
            ),
        )

        def fake_validate(coin: str, interval: str, **kwargs: object) -> ValidationReport:
            order.append(f"validate:{coin}:{interval}")
            return _validation_report(coin, interval)

        monkeypatch.setattr(bid_ask_validator, "validate_coin_interval", fake_validate)

        def fake_resample(coin: str, intervals: tuple[str, ...] = ("15s", "1m"), **kwargs: object):
            order.append(f"resample:{coin}:{','.join(intervals)}")
            return {
                interval: {
                    "output_path": Path(f"/tmp/{coin}_{interval}.csv"),
                    "report_path": Path(f"/tmp/{coin}_{interval}_report.json"),
                }
                for interval in intervals
            }

        monkeypatch.setattr(bid_ask_resampler, "resample_coin", fake_resample)
        monkeypatch.setattr(
            bid_ask_train,
            "run_training_pipeline",
            lambda **kwargs: (
                order.append(f"train:{kwargs['coin']}:{kwargs['interval']}") or {
                    "evaluation": {"train_rows": 1, "test_rows": 1, "warnings": []},
                    "report": {"source": "historical"},
                    "report_path": Path("/tmp/train_report.json"),
                }
            ),
        )

        pipeline.main(
            [
                "run-one",
                "--coin",
                "BTC",
                "--train-interval",
                "15s",
                "--horizon",
                "1",
                "--source",
                "historical",
            ]
        )

        assert order == [
            "validate:BTC:1s",
            "resample:BTC:15s,1m",
            "validate:BTC:15s",
            "validate:BTC:1m",
            "train:BTC:15s",
        ]


class TestRunAllWorkflow:
    def test_run_all_continues_when_one_coin_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        processed: list[str] = []

        def fake_extract(coin: str, **kwargs: object) -> ExtractionReport:
            processed.append(coin)
            if coin == "ETH":
                raise RuntimeError("simulated extract failure")
            return _extraction_report(coin)

        monkeypatch.setattr(bid_ask_extractor, "extract_coin_bid_ask_1s", fake_extract)
        monkeypatch.setattr(
            bid_ask_validator,
            "validate_coin_interval",
            lambda coin, interval, **kwargs: _validation_report(coin, interval),
        )
        monkeypatch.setattr(
            bid_ask_resampler,
            "resample_coin",
            lambda coin, intervals=("15s", "1m"), **kwargs: {
                interval: {
                    "output_path": Path(f"/tmp/{coin}_{interval}.csv"),
                    "report_path": Path(f"/tmp/{coin}_{interval}_report.json"),
                }
                for interval in intervals
            },
        )
        monkeypatch.setattr(
            bid_ask_train,
            "run_training_pipeline",
            lambda **kwargs: {
                "evaluation": {"train_rows": 1, "test_rows": 1, "warnings": []},
                "report": {},
                "report_path": Path("/tmp/train_report.json"),
            },
        )

        exit_code = pipeline.main(
            ["run-all", "--coins", "BTC", "ETH", "SOL", "--train-interval", "15s", "--horizon", "1"]
        )
        assert processed == ["BTC", "ETH", "SOL"]
        assert exit_code == 1


class TestDryRun:
    def test_dry_run_writes_no_files(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        data_root = tmp_path / "bid_ask_csv"
        data_root.mkdir()

        def fail_if_called(*args: object, **kwargs: object) -> None:
            raise AssertionError("module function should not be called in dry-run")

        monkeypatch.setattr(bid_ask_extractor, "extract_coin_bid_ask_1s", fail_if_called)
        monkeypatch.setattr(bid_ask_validator, "validate_coin_interval", fail_if_called)
        monkeypatch.setattr(bid_ask_resampler, "resample_coin", fail_if_called)
        monkeypatch.setattr(bid_ask_train, "run_training_pipeline", fail_if_called)

        pipeline.main(
            [
                "run-one",
                "--coin",
                "BTC",
                "--train-interval",
                "15s",
                "--horizon",
                "1",
                "--dry-run",
                "--data-root",
                str(data_root),
            ]
        )

        assert list(data_root.rglob("*")) == []


class TestStrictMode:
    def test_strict_stops_after_validation_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        order: list[str] = []

        monkeypatch.setattr(
            bid_ask_extractor,
            "extract_coin_bid_ask_1s",
            lambda coin, **kwargs: (order.append("extract") or _extraction_report(coin)),
        )

        def fake_validate(coin: str, interval: str, **kwargs: object) -> ValidationReport:
            order.append(f"validate:{interval}")
            passed = interval != "1s"
            return _validation_report(coin, interval, passed=passed)

        monkeypatch.setattr(bid_ask_validator, "validate_coin_interval", fake_validate)
        monkeypatch.setattr(
            bid_ask_resampler,
            "resample_coin",
            lambda *args, **kwargs: (order.append("resample") or {}),
        )
        monkeypatch.setattr(
            bid_ask_train,
            "run_training_pipeline",
            lambda **kwargs: (order.append("train") or {}),
        )

        pipeline.main(
            [
                "run-one",
                "--coin",
                "BTC",
                "--train-interval",
                "15s",
                "--horizon",
                "1",
                "--strict",
            ]
        )

        assert order == ["extract", "validate:1s"]


class TestSafety:
    def test_no_network_calls(self) -> None:
        source = PIPELINE_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {
            "requests",
            "urllib",
            "http",
            "socket",
            "websockets",
            "httpx",
            "aiohttp",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in forbidden
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in forbidden

    def test_no_wallet_signing_order_execution_code(self) -> None:
        source = PIPELINE_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        lowered = source.lower()
        for pattern in FORBIDDEN_SOURCE_PATTERNS:
            assert pattern not in lowered
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES

    def test_allowed_imports_only(self) -> None:
        source = PIPELINE_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        allowed_modules = {
            "bid_ask_extractor",
            "bid_ask_resampler",
            "bid_ask_validator",
            "bid_ask_train",
            "argparse",
            "json",
            "sys",
            "dataclasses",
            "pathlib",
            "__future__",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    assert root in allowed_modules, f"unexpected import: {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".", 1)[0]
                assert root in allowed_modules, f"unexpected import from: {node.module}"

    def test_raw_files_not_modified(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        raw_file = tmp_path / "market_stream" / "raw" / "session.jsonl"
        raw_file.parent.mkdir(parents=True)
        raw_file.write_text('{"channel":"bbo"}\n', encoding="utf-8")
        before_hash = hashlib.sha256(raw_file.read_bytes()).hexdigest()

        monkeypatch.setattr(
            bid_ask_extractor,
            "extract_coin_bid_ask_1s",
            lambda coin, **kwargs: _extraction_report(coin),
        )

        data_root = tmp_path / "bid_ask_csv"
        pipeline.main(
            [
                "extract",
                "--coins",
                "BTC",
                "--source",
                "local",
                "--interval",
                "1s",
                "--data-root",
                str(data_root),
                "--market-stream-dir",
                str(tmp_path / "market_stream"),
            ]
        )

        after_hash = hashlib.sha256(raw_file.read_bytes()).hexdigest()
        assert after_hash == before_hash

    def test_data_coverage_statement_in_extract_and_train(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            bid_ask_extractor,
            "extract_coin_bid_ask_1s",
            lambda coin, **kwargs: _extraction_report(coin),
        )
        pipeline.main(["extract", "--coins", "BTC", "--source", "local", "--interval", "1s"])
        assert pipeline.DATA_COVERAGE_STATEMENT in capsys.readouterr().out

        capsys.readouterr()
        monkeypatch.setattr(
            bid_ask_train,
            "run_training_pipeline",
            lambda **kwargs: {
                "evaluation": {"train_rows": 1, "test_rows": 1, "warnings": []},
                "report": {},
                "report_path": Path("/tmp/report.json"),
            },
        )
        pipeline.main(["train", "--coin", "BTC", "--interval", "15s", "--horizon", "1"])
        assert pipeline.DATA_COVERAGE_STATEMENT in capsys.readouterr().out
