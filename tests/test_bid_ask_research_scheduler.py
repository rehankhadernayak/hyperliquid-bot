"""Tests for bid_ask_research_scheduler (offline, mocked, no network)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import bid_ask_research_scheduler as scheduler
import bid_ask_research_orchestrator as orchestrator
from bid_ask_schema import FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS

SCHEDULER_FILE = Path(__file__).resolve().parent.parent / "bid_ask_research_scheduler.py"


def _init_state(tmp_path: Path, *, start: str = "20230916", end: str = "20230930") -> Path:
    state_path = tmp_path / "state.json"
    scheduler.cmd_init_state(
        historical_start_date=start,
        historical_end_date=end,
        coins=["BTC", "ETH", "SOL"],
        state_path=state_path,
    )
    return state_path


def _fake_batch_report(*, failures: list[str] | None = None) -> orchestrator.OrchestratorReport:
    report = orchestrator.OrchestratorReport(
        workflows_run=["historical-batch"],
        coins=["BTC", "ETH", "SOL"],
        row_counts={"BTC:1s": 100},
    )
    report.details["download"] = {
        "file_results": [
            {"status": "downloaded"},
            {"status": "downloaded"},
            {"status": "skipped"},
        ]
    }
    if failures:
        report.failures.extend(failures)
    return report


class TestInitState:
    def test_creates_state(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        assert state_path.is_file()
        state = scheduler.load_state(state_path)
        assert state.historical_start_date == "20230916"
        assert state.next_historical_date == "20230916"
        assert state.coins == ["BTC", "ETH", "SOL"]
        assert scheduler.ARCHIVE_AVAILABILITY_WARNING in state.warnings

    def test_does_not_overwrite_without_reset(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        with pytest.raises(FileExistsError, match="already exists"):
            scheduler.cmd_init_state(
                historical_start_date="20230916",
                historical_end_date="20230917",
                coins=["BTC"],
                state_path=state_path,
            )

    def test_reset_overwrites(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        scheduler.cmd_init_state(
            historical_start_date="20231001",
            historical_end_date="20231002",
            coins=["BTC"],
            reset=True,
            state_path=state_path,
        )
        state = scheduler.load_state(state_path)
        assert state.historical_start_date == "20231001"
        assert state.coins == ["BTC"]

    def test_rejects_large_range_without_confirm(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        with pytest.raises(ValueError, match="confirm-large-range"):
            scheduler.cmd_init_state(
                historical_start_date="20230101",
                historical_end_date="20231231",
                coins=["BTC"],
                state_path=state_path,
            )

    def test_allows_large_range_with_confirm(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        scheduler.cmd_init_state(
            historical_start_date="20230101",
            historical_end_date="20231231",
            coins=["BTC"],
            confirm_large_range=True,
            state_path=state_path,
        )
        assert state_path.is_file()


class TestStatus:
    def test_reads_state(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        status = scheduler.cmd_status(state_path=state_path)
        assert status["next_historical_date"] == "20230916"
        assert status["completed_days"] == 0
        assert status["failed_days"] == 0
        assert status["coins"] == ["BTC", "ETH", "SOL"]


class TestHistoricalBackfillNext:
    def test_uses_next_date_from_state(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)
        captured: dict[str, str] = {}

        def fake_batch(**kwargs: object) -> orchestrator.OrchestratorReport:
            captured["date"] = str(kwargs["date"])
            return _fake_batch_report()

        monkeypatch.setattr(orchestrator, "cmd_historical_batch", fake_batch)

        report = scheduler.cmd_historical_backfill_next(
            state_path=state_path,
            runs_dir=tmp_path / "runs",
        )
        assert captured["date"] == "20230916"
        assert report.historical_date == "20230916"
        assert report.historical_hours == ["0", "1", "2"]

    def test_caps_max_files(
        self,
        tmp_path: Path,
    ) -> None:
        state_path = _init_state(tmp_path)
        with pytest.raises(ValueError, match="hard cap"):
            scheduler.cmd_historical_backfill_next(
                max_files=25,
                allow_more=True,
                state_path=state_path,
                runs_dir=tmp_path / "runs",
            )

    def test_rejects_all_time(
        self,
        tmp_path: Path,
    ) -> None:
        state_path = _init_state(tmp_path)
        with pytest.raises(ValueError, match="all-time"):
            scheduler.cmd_historical_backfill_next(
                all_time=True,
                state_path=state_path,
                runs_dir=tmp_path / "runs",
            )

    def test_rejects_all_twenty_four_hours(
        self,
        tmp_path: Path,
    ) -> None:
        state_path = _init_state(tmp_path)
        with pytest.raises(ValueError, match="24 hours"):
            scheduler.cmd_historical_backfill_next(
                hours=list(range(24)),
                state_path=state_path,
                runs_dir=tmp_path / "runs",
            )

    def test_dedupes_completed_batch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)
        calls: list[str] = []

        def fake_batch(**kwargs: object) -> orchestrator.OrchestratorReport:
            calls.append(str(kwargs["date"]))
            return _fake_batch_report()

        monkeypatch.setattr(orchestrator, "cmd_historical_batch", fake_batch)

        scheduler.cmd_historical_backfill_next(
            state_path=state_path,
            runs_dir=tmp_path / "runs",
        )
        scheduler.cmd_historical_backfill_next(
            state_path=state_path,
            runs_dir=tmp_path / "runs",
        )
        assert len(calls) == 1

    def test_updates_state_on_success(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)
        monkeypatch.setattr(orchestrator, "cmd_historical_batch", lambda **k: _fake_batch_report())

        scheduler.cmd_historical_backfill_next(
            state_path=state_path,
            runs_dir=tmp_path / "runs",
        )
        state = scheduler.load_state(state_path)
        assert len(state.completed_batches) == 1
        assert state.total_historical_files_downloaded == 2
        assert state.last_report_path is not None


class TestDailyCycle:
    def test_calls_workflows_in_order(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)
        order: list[str] = []

        def fake_backfill(**kwargs: object) -> scheduler.SchedulerReport:
            order.append("historical-backfill-next")
            return scheduler.SchedulerReport(
                workflows_run=["historical-backfill-next"],
                historical_date="20230916",
                historical_hours=["0", "1", "2"],
            )

        def fake_validate(**kwargs: object) -> orchestrator.OrchestratorReport:
            order.append("resample-validate")
            return orchestrator.OrchestratorReport(workflows_run=["resample-validate"])

        def fake_train(**kwargs: object) -> orchestrator.OrchestratorReport:
            order.append(f"train-{kwargs.get('source')}")
            return orchestrator.OrchestratorReport(workflows_run=["train-evaluate"])

        def fake_summary(*args: object, **kwargs: object) -> dict:
            order.append("daily-summary")
            return {"generated_at": "2026-01-01T00:00:00Z", "paper_only": True}

        monkeypatch.setattr(scheduler, "cmd_historical_backfill_next", fake_backfill)
        monkeypatch.setattr(orchestrator, "cmd_resample_validate", fake_validate)
        monkeypatch.setattr(orchestrator, "cmd_train_evaluate", fake_train)
        monkeypatch.setattr(scheduler.bid_ask_daily_summary, "build_daily_summary", fake_summary)
        monkeypatch.setattr(
            scheduler.bid_ask_daily_summary,
            "write_summary_json",
            lambda s, d: d / "summary.json",
        )
        monkeypatch.setattr(
            scheduler.bid_ask_daily_summary,
            "write_summary_markdown",
            lambda s, d: d / "summary.md",
        )

        report = scheduler.cmd_daily_cycle(
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            daily_summary_dir=tmp_path / "daily",
        )
        assert order == [
            "historical-backfill-next",
            "resample-validate",
            "train-historical",
            "daily-summary",
        ]
        assert "daily-cycle" in report.workflows_run
        assert "resample-validate" in report.workflows_run
        assert "train-evaluate-historical" in report.workflows_run
        assert "daily-summary" in report.workflows_run

    def test_updates_state_and_writes_report(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)
        runs_dir = tmp_path / "runs"

        monkeypatch.setattr(
            scheduler,
            "cmd_historical_backfill_next",
            lambda **k: scheduler.SchedulerReport(
                workflows_run=["historical-backfill-next"],
                historical_date="20230916",
            ),
        )
        monkeypatch.setattr(
            orchestrator,
            "cmd_resample_validate",
            lambda **k: orchestrator.OrchestratorReport(workflows_run=["resample-validate"]),
        )
        monkeypatch.setattr(
            orchestrator,
            "cmd_train_evaluate",
            lambda **k: orchestrator.OrchestratorReport(workflows_run=["train-evaluate"]),
        )
        monkeypatch.setattr(
            scheduler.bid_ask_daily_summary,
            "build_daily_summary",
            lambda *a, **k: {"generated_at": "2026-01-01T00:00:00Z"},
        )
        monkeypatch.setattr(
            scheduler.bid_ask_daily_summary,
            "write_summary_json",
            lambda s, d: d / "summary.json",
        )
        monkeypatch.setattr(
            scheduler.bid_ask_daily_summary,
            "write_summary_markdown",
            lambda s, d: d / "summary.md",
        )

        report = scheduler.cmd_daily_cycle(
            state_path=state_path,
            runs_dir=runs_dir,
            daily_summary_dir=tmp_path / "daily",
        )
        state = scheduler.load_state(state_path)
        assert state.last_training_at is not None
        assert state.last_report_path is not None
        assert runs_dir.is_dir()
        assert report.paper_only is True
        assert report.execution_enabled is False

    def test_daily_cycle_report_stores_train_interval_and_horizon(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)
        runs_dir = tmp_path / "runs"

        monkeypatch.setattr(
            scheduler,
            "cmd_historical_backfill_next",
            lambda **k: scheduler.SchedulerReport(
                workflows_run=["historical-backfill-next"],
                coins=["BTC", "ETH", "SOL"],
                historical_date="20230916",
                historical_hours=["0", "1", "2"],
                source="historical",
                train_interval=k.get("train_interval"),
                horizon=k.get("horizon"),
            ),
        )
        monkeypatch.setattr(
            orchestrator,
            "cmd_resample_validate",
            lambda **k: orchestrator.OrchestratorReport(workflows_run=["resample-validate"]),
        )
        monkeypatch.setattr(
            orchestrator,
            "cmd_train_evaluate",
            lambda **k: orchestrator.OrchestratorReport(workflows_run=["train-evaluate"]),
        )
        monkeypatch.setattr(
            scheduler.bid_ask_daily_summary,
            "build_daily_summary",
            lambda *a, **k: {"generated_at": "2026-01-01T00:00:00Z"},
        )
        monkeypatch.setattr(
            scheduler.bid_ask_daily_summary,
            "write_summary_json",
            lambda s, d: d / "summary.json",
        )
        monkeypatch.setattr(
            scheduler.bid_ask_daily_summary,
            "write_summary_markdown",
            lambda s, d: d / "summary.md",
        )

        report = scheduler.cmd_daily_cycle(
            state_path=state_path,
            runs_dir=runs_dir,
            daily_summary_dir=tmp_path / "daily",
            train_interval="15s",
            horizon=5,
        )
        assert report.train_interval == "15s"
        assert report.horizon == 5
        assert report.source == "historical"
        assert report.coins == ["BTC", "ETH", "SOL"]
        assert report.historical_date == "20230916"
        assert report.historical_hours == ["0", "1", "2"]

        state = scheduler.load_state(state_path)
        assert state.last_report_path is not None
        saved = json.loads(Path(state.last_report_path).read_text(encoding="utf-8"))
        assert saved["train_interval"] == "15s"
        assert saved["horizon"] == 5
        assert saved["source"] == "historical"
        assert saved["historical_date"] == "20230916"
        assert saved["historical_hours"] == ["0", "1", "2"]
        assert saved["coins"] == ["BTC", "ETH", "SOL"]

    def test_daily_cycle_report_includes_training_metadata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)
        runs_dir = tmp_path / "runs"

        def fake_train(**kwargs: object) -> orchestrator.OrchestratorReport:
            train_report = orchestrator.OrchestratorReport(
                workflows_run=["train-evaluate"],
                model_paths=["models/bid_ask/BTC_historical_15s_abc.joblib"],
                report_paths=[
                    "reports/bid_ask_training/abc123.json",
                    "reports/bid_ask_research_orchestrator/train.json",
                ],
            )
            train_report.details["decision_evaluation"] = {
                "json_report_path": "reports/bid_ask_decision_evaluation/eval.json",
            }
            train_report.details["decision_filter"] = {
                "json_report_path": "reports/bid_ask_decision_filters/filter.json",
            }
            return train_report

        monkeypatch.setattr(
            scheduler,
            "cmd_historical_backfill_next",
            lambda **k: scheduler.SchedulerReport(
                workflows_run=["historical-backfill-next"],
                historical_date="20230916",
                historical_hours=["0", "1", "2"],
            ),
        )
        monkeypatch.setattr(
            orchestrator,
            "cmd_resample_validate",
            lambda **k: orchestrator.OrchestratorReport(workflows_run=["resample-validate"]),
        )
        monkeypatch.setattr(orchestrator, "cmd_train_evaluate", fake_train)
        monkeypatch.setattr(
            scheduler.bid_ask_daily_summary,
            "build_daily_summary",
            lambda *a, **k: {"generated_at": "2026-01-01T00:00:00Z"},
        )
        monkeypatch.setattr(
            scheduler.bid_ask_daily_summary,
            "write_summary_json",
            lambda s, d: d / "summary.json",
        )
        monkeypatch.setattr(
            scheduler.bid_ask_daily_summary,
            "write_summary_markdown",
            lambda s, d: d / "summary.md",
        )

        report = scheduler.cmd_daily_cycle(
            state_path=state_path,
            runs_dir=runs_dir,
            daily_summary_dir=tmp_path / "daily",
            train_interval="15s",
            horizon=5,
        )
        assert report.training_reports == ["reports/bid_ask_training/abc123.json"]
        assert report.model_artifact_paths == ["models/bid_ask/BTC_historical_15s_abc.joblib"]
        assert report.evaluation_report_paths == [
            "reports/bid_ask_decision_evaluation/eval.json"
        ]
        assert report.filter_report_paths == ["reports/bid_ask_decision_filters/filter.json"]

        state = scheduler.load_state(state_path)
        saved = json.loads(Path(state.last_report_path).read_text(encoding="utf-8"))
        assert saved["training_reports"] == ["reports/bid_ask_training/abc123.json"]
        assert saved["model_artifact_paths"] == ["models/bid_ask/BTC_historical_15s_abc.joblib"]
        assert saved["evaluation_report_paths"] == [
            "reports/bid_ask_decision_evaluation/eval.json"
        ]
        assert saved["filter_report_paths"] == ["reports/bid_ask_decision_filters/filter.json"]



class TestLiveLongSession:
    def test_requires_finite_duration(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        with pytest.raises(ValueError, match="positive"):
            scheduler.cmd_live_long_session(
                coins=["BTC"],
                duration_seconds=0,
                state_path=state_path,
                runs_dir=tmp_path / "runs",
            )

    def test_rejects_too_long_without_allow_long(
        self,
        tmp_path: Path,
    ) -> None:
        state_path = _init_state(tmp_path)
        with pytest.raises(ValueError, match="allow-long"):
            scheduler.cmd_live_long_session(
                coins=["BTC"],
                duration_seconds=30_000,
                state_path=state_path,
                runs_dir=tmp_path / "runs",
            )

    def test_runs_with_mock_recorder(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)

        def fake_live(**kwargs: object) -> orchestrator.OrchestratorReport:
            return orchestrator.OrchestratorReport(
                workflows_run=["live-session"],
                details={"duration_seconds": kwargs.get("duration_seconds")},
            )

        monkeypatch.setattr(orchestrator, "cmd_live_session", fake_live)

        report = scheduler.cmd_live_long_session(
            coins=["BTC"],
            duration_seconds=3600,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            run_recorder_fn=lambda **k: {"status": "ok"},
        )
        assert report.live_session_duration == 3600
        state = scheduler.load_state(state_path)
        assert state.last_live_session_at is not None


class TestLaunchdTemplate:
    def test_writes_files_without_installing(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "scheduler"
        plist_path, setup_path = scheduler.cmd_write_launchd_template(
            run_time="02:00",
            coins=["BTC", "ETH", "SOL"],
            aws_profile="hyperliquid-research",
            requester_pays=True,
            state_dir=state_dir,
            project_root=tmp_path,
        )
        assert plist_path.is_file()
        assert setup_path.is_file()
        plist_text = plist_path.read_text(encoding="utf-8")
        assert "com.hyperliquid.research.daily" in plist_text
        assert "daily-cycle" in plist_text
        assert "hyperliquid-research" in plist_text
        setup_text = setup_path.read_text(encoding="utf-8")
        assert "not installed automatically" in setup_text.lower()


class TestReportsSafetyFields:
    def test_scheduler_report_safety_flags(self) -> None:
        report = scheduler.SchedulerReport(workflows_run=["test"])
        payload = report.to_dict()
        assert payload["paper_only"] is True
        assert payload["execution_enabled"] is False

    def test_written_report_includes_required_fields(self, tmp_path: Path) -> None:
        report = scheduler.SchedulerReport(
            workflows_run=["daily-cycle"],
            coins=["BTC"],
            historical_date="20230916",
            historical_hours=["0", "1", "2"],
            source="historical",
            train_interval="15s",
            horizon=5,
            next_recommended_action="status",
        )
        json_path, md_path = scheduler.write_scheduler_report(
            report,
            runs_dir=tmp_path,
            label="test",
        )
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        assert saved["generated_at"]
        assert saved["paper_only"] is True
        assert saved["execution_enabled"] is False
        assert saved["workflows_run"] == ["daily-cycle"]
        assert saved["historical_date"] == "20230916"
        assert saved["source"] == "historical"
        assert saved["train_interval"] == "15s"
        assert saved["horizon"] == 5
        assert saved["next_recommended_action"] == "status"
        assert md_path.is_file()


class TestSafetyConstraints:
    def test_no_wallet_signing_order_execution_imports(self) -> None:
        source = SCHEDULER_FILE.read_text(encoding="utf-8")
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

    def test_no_credentials_in_report_output(self, tmp_path: Path) -> None:
        report = scheduler.SchedulerReport(
            workflows_run=["test"],
            warnings=["requester pays may incur AWS charges"],
        )
        json_path, _ = scheduler.write_scheduler_report(report, runs_dir=tmp_path, label="cred")
        combined = json_path.read_text(encoding="utf-8").lower()
        assert "secret_access_key" not in combined
        assert "akia" not in combined
