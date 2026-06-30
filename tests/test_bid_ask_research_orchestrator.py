"""Tests for bid_ask_research_orchestrator (offline, mocked, no network)."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import bid_ask_research_orchestrator as orchestrator
from bid_ask_schema import FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS
from bid_ask_train import MIN_ROWS_WARNING

ORCHESTRATOR_FILE = Path(__file__).resolve().parent.parent / "bid_ask_research_orchestrator.py"


def _write_1s_csv(path: Path, coin: str, timestamps: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ts in timestamps:
        rows.append(
            {
                "timestamp_ms": ts,
                "timestamp_iso": "2023-09-16T09:00:00Z",
                "coin": coin,
                "best_bid": 100.0,
                "best_ask": 100.1,
                "mid_price": 100.05,
                "spread": 0.1,
                "spread_bps": 9.99,
                "bid_size": 1.0,
                "ask_size": 1.0,
                "bbo_imbalance": 0.0,
                "source_session_id": "test",
                "source_channel": "test",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_resampled_csv(path: Path, coin: str, starts: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for start in starts:
        rows.append(
            {
                "start_timestamp_ms": start,
                "end_timestamp_ms": start + 14_999,
                "timestamp_iso": "2023-09-16T09:00:00Z",
                "coin": coin,
                "open_mid": 100.0,
                "high_mid": 100.1,
                "low_mid": 99.9,
                "close_mid": 100.05,
                "avg_bid": 100.0,
                "avg_ask": 100.1,
                "avg_mid": 100.05,
                "avg_spread": 0.1,
                "avg_spread_bps": 9.99,
                "avg_bbo_imbalance": 0.0,
                "row_count": 15,
                "missing_seconds": 0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


class TestHistoricalBatchCaps:
    def test_default_max_files_is_nine(self) -> None:
        assert orchestrator.resolve_orchestrator_max_files(None) == 9

    def test_rejects_above_nine_without_allow_more(self) -> None:
        with pytest.raises(ValueError, match="hard cap of 9"):
            orchestrator.resolve_orchestrator_max_files(10)

    def test_allow_more_raises_cap_to_twenty_four(self) -> None:
        assert orchestrator.resolve_orchestrator_max_files(20, allow_more=True) == 20
        with pytest.raises(ValueError, match="hard cap of 24"):
            orchestrator.resolve_orchestrator_max_files(25, allow_more=True)

    def test_rejects_all_time_flag(self) -> None:
        with pytest.raises(ValueError, match="all-time"):
            orchestrator.reject_all_time(all_time=True)

    def test_rejects_all_twenty_four_hours(self) -> None:
        with pytest.raises(ValueError, match="24 hours"):
            orchestrator.reject_all_time(hours=[str(h) for h in range(24)])

    def test_rejects_planned_files_above_cap(self) -> None:
        with pytest.raises(ValueError, match="planned file count"):
            orchestrator.resolve_orchestrator_max_files(9, planned_files=10)

    def test_historical_batch_uses_plan_hours(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_run_sample_download(plan: dict, **kwargs: object) -> dict:
            captured["hours"] = plan["hours"]
            captured["max_files"] = kwargs["max_files"]
            return {"file_results": [], "warnings": []}

        monkeypatch.setattr(orchestrator, "run_sample_download", fake_run_sample_download)
        monkeypatch.setattr(orchestrator, "_import_decompressed_files", lambda *a, **k: [])
        monkeypatch.setattr(
            orchestrator,
            "cmd_resample_validate",
            lambda **kwargs: orchestrator.OrchestratorReport(workflows_run=["resample-validate"]),
        )
        monkeypatch.setattr(
            orchestrator,
            "write_orchestrator_report",
            lambda *a, **k: (tmp_path / "r.json", tmp_path / "r.md"),
        )

        orchestrator.cmd_historical_batch(
            date="20230916",
            hours=[9, 10, 11],
            coins=["BTC", "ETH", "SOL"],
            max_files=9,
            s3_client=object(),
            report_dir=tmp_path,
        )
        assert captured["hours"] == ["9", "10", "11"]
        assert captured["max_files"] == 9


class TestLiveSessionDuration:
    def test_requires_duration(self) -> None:
        with pytest.raises(ValueError, match="duration-seconds"):
            orchestrator.validate_live_duration(None)

    def test_rejects_too_long_without_allow_long(self) -> None:
        with pytest.raises(ValueError, match="allow-long"):
            orchestrator.validate_live_duration(30_000)

    def test_allows_long_with_flag(self) -> None:
        assert orchestrator.validate_live_duration(30_000, allow_long=True) == 30_000

    def test_live_session_calls_recorder_with_duration(
        self,
        tmp_path: Path,
    ) -> None:
        calls: list[float] = []

        def fake_recorder(**kwargs: object) -> dict:
            calls.append(float(kwargs["duration_seconds"]))  # type: ignore[arg-type]
            return {"session_id": "test-session"}

        report = orchestrator.cmd_live_session(
            coins=["BTC"],
            duration_seconds=3600,
            report_dir=tmp_path,
            run_recorder_fn=fake_recorder,
        )
        assert calls == [3600.0]
        assert "live-session" in report.workflows_run
        assert any("extract" in action for action in report.next_recommended_actions)


class TestCombinedDataset:
    def test_keeps_source_files_separate(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        local_path = data_root / "1s" / "BTC_1s.csv"
        historical_path = data_root / "historical_1s" / "BTC_1s.csv"
        _write_1s_csv(local_path, "BTC", [1_000, 2_000])
        _write_1s_csv(historical_path, "BTC", [3_000, 4_000])

        local_before = local_path.read_text(encoding="utf-8")
        historical_before = historical_path.read_text(encoding="utf-8")

        report = orchestrator.cmd_combined_dataset(
            coins=["BTC"],
            intervals=["1s"],
            data_root=data_root,
            report_dir=tmp_path / "reports",
        )

        assert local_path.read_text(encoding="utf-8") == local_before
        assert historical_path.read_text(encoding="utf-8") == historical_before
        combined_path = data_root / "combined_1s" / "BTC_1s.csv"
        assert combined_path.is_file()
        assert report.row_counts["BTC:1s:combined"] == 4

    def test_dedupes_and_sorts(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        local_path = data_root / "1s" / "ETH_1s.csv"
        historical_path = data_root / "historical_1s" / "ETH_1s.csv"
        _write_1s_csv(local_path, "ETH", [3_000, 1_000])
        _write_1s_csv(historical_path, "ETH", [2_000, 1_000])

        orchestrator.cmd_combined_dataset(
            coins=["ETH"],
            intervals=["1s"],
            data_root=data_root,
            report_dir=tmp_path / "reports",
        )

        combined = pd.read_csv(data_root / "combined_1s" / "ETH_1s.csv")
        assert combined["timestamp_ms"].tolist() == [1_000, 2_000, 3_000]
        metadata_path = next((tmp_path / "reports").glob("combined_metadata_*.json"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata[0]["duplicates_removed"] == 1
        assert metadata[0]["local_rows"] == 2
        assert metadata[0]["historical_rows"] == 2


class TestTrainEvaluateSkips:
    def test_skips_too_few_rows(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        data_root = tmp_path / "data"
        resampled = data_root / "historical_15s" / "SOL_15s.csv"
        _write_resampled_csv(resampled, "SOL", list(range(1, MIN_ROWS_WARNING)))

        train_calls: list[str] = []

        def fake_run(**kwargs: object) -> dict:
            train_calls.append(str(kwargs["coin"]))
            return {"report": {"train_rows": 10, "test_rows": 5}}

        monkeypatch.setattr(orchestrator.bid_ask_train, "run_training_pipeline", fake_run)

        report = orchestrator.cmd_train_evaluate(
            coins=["SOL"],
            source="historical",
            interval="15s",
            horizon=5,
            data_root=data_root,
            report_dir=tmp_path / "reports",
        )
        assert train_calls == []
        assert any("skipping training" in warning for warning in report.warnings)

        _write_resampled_csv(
            resampled,
            "SOL",
            list(range(1, MIN_ROWS_WARNING + 10)),
        )
        report_ok = orchestrator.cmd_train_evaluate(
            coins=["SOL"],
            source="historical",
            interval="15s",
            horizon=5,
            data_root=data_root,
            report_dir=tmp_path / "reports",
        )
        assert train_calls == ["SOL"]
        assert not any("skipping training" in warning for warning in report_ok.warnings)


class TestReportsSafetyFields:
    def test_report_dict_includes_safety_flags(self, tmp_path: Path) -> None:
        report = orchestrator.OrchestratorReport(workflows_run=["test"])
        payload = report.to_dict()
        assert payload["paper_only"] is True
        assert payload["execution_enabled"] is False

        json_path, _ = orchestrator.write_orchestrator_report(
            report,
            output_dir=tmp_path,
            label="safety",
        )
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        assert saved["paper_only"] is True
        assert saved["execution_enabled"] is False


class TestSafetyConstraints:
    def test_no_wallet_signing_order_execution_imports(self) -> None:
        source = ORCHESTRATOR_FILE.read_text(encoding="utf-8")
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

    def test_no_shell_true_in_run_command(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            captured.update(kwargs)
            return subprocess.CompletedProcess(args=["python3"], returncode=0, stdout="", stderr="")

        monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
        orchestrator.run_command(["python3", "bid_ask_data_pipeline.py", "validate", "--coins", "BTC"])
        assert captured.get("shell") is False

    def test_rejects_unallowlisted_command(self) -> None:
        with pytest.raises(ValueError, match="not allowlisted"):
            orchestrator.run_command(["curl", "https://example.com"])

    def test_rejects_credential_like_command(self) -> None:
        with pytest.raises(ValueError, match="sensitive"):
            orchestrator.run_command(["python3", "example", "private", "key", "value"])

    def test_no_credentials_in_report_output(self, tmp_path: Path) -> None:
        report = orchestrator.OrchestratorReport(
            workflows_run=["test"],
            warnings=["requester pays may incur AWS charges"],
        )
        json_path, _ = orchestrator.write_orchestrator_report(
            report,
            output_dir=tmp_path,
            label="cred_check",
        )
        combined = json_path.read_text(encoding="utf-8")
        lowered = combined.lower()
        assert "secret_access_key" not in lowered
        assert "akia" not in lowered


class TestBuildHourlyPlan:
    def test_plan_includes_requested_hours_and_coins(self) -> None:
        plan = orchestrator.build_hourly_backfill_plan(
            date="20230916",
            hours=[9, 10],
            coins=["BTC", "SOL"],
        )
        assert plan["hours"] == ["9", "10"]
        assert plan["coins"] == ["BTC", "SOL"]
        assert plan["total_files"] == 4
        assert plan["paper_only"] is True
        assert plan["execution_enabled"] is False
