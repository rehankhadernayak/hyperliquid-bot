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


def _init_state(
    tmp_path: Path,
    *,
    start: str = "20230916",
    end: str = "20230930",
    start_hour: int = 0,
    end_hour: int = 23,
) -> Path:
    state_path = tmp_path / "state.json"
    scheduler.cmd_init_state(
        historical_start_date=start,
        historical_end_date=end,
        historical_start_hour=start_hour,
        historical_end_hour=end_hour,
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

    def test_stores_start_and_end_hours(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        scheduler.cmd_init_state(
            historical_start_date="20230916",
            historical_end_date="20230917",
            historical_start_hour=9,
            historical_end_hour=14,
            coins=["BTC"],
            state_path=state_path,
        )
        state = scheduler.load_state(state_path)
        assert state.historical_start_hour == 9
        assert state.historical_end_hour == 14


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


class TestSplitHourRange:
    def test_splits_15_to_23_into_three_chunks(self) -> None:
        chunks = scheduler.split_hour_range_into_chunks(15, 23, 3)
        assert chunks == [["15", "16", "17"], ["18", "19", "20"], ["21", "22", "23"]]


class TestBatchSeriesValidation:
    def test_rejects_invalid_hour_range(self) -> None:
        with pytest.raises(ValueError, match="start-hour"):
            scheduler.validate_batch_series_hours(start_hour=-1, end_hour=5, chunk_size_hours=3)
        with pytest.raises(ValueError, match="end-hour"):
            scheduler.validate_batch_series_hours(start_hour=0, end_hour=24, chunk_size_hours=3)
        with pytest.raises(ValueError, match="less than or equal"):
            scheduler.validate_batch_series_hours(start_hour=10, end_hour=5, chunk_size_hours=3)

    def test_rejects_too_many_total_hours_without_allow_more(self) -> None:
        with pytest.raises(ValueError, match="hour range spans"):
            scheduler.validate_batch_series_hours(
                start_hour=0,
                end_hour=23,
                chunk_size_hours=3,
                allow_more=False,
            )

    def test_allows_up_to_24_hours_with_allow_more(self) -> None:
        scheduler.validate_batch_series_hours(
            start_hour=0,
            end_hour=23,
            chunk_size_hours=3,
            allow_more=True,
        )

    def test_rejects_too_large_chunk_size_without_allow_more(self) -> None:
        with pytest.raises(ValueError, match="chunk-size-hours"):
            scheduler.validate_batch_series_hours(
                start_hour=0,
                end_hour=8,
                chunk_size_hours=9,
                allow_more=False,
            )


def _mock_daily_cycle_success(
    monkeypatch: pytest.MonkeyPatch,
    *,
    runs_dir: Path,
) -> list[list[int]]:
    captured_hours: list[list[int]] = []

    def fake_daily_cycle(**kwargs: object) -> scheduler.SchedulerReport:
        hours = kwargs.get("historical_hours") or []
        captured_hours.append(list(hours))
        runs_dir.mkdir(parents=True, exist_ok=True)
        json_path = runs_dir / f"daily_{'-'.join(str(h) for h in hours)}.json"
        md_path = runs_dir / f"daily_{'-'.join(str(h) for h in hours)}.md"
        json_path.write_text("{}", encoding="utf-8")
        md_path.write_text("# daily", encoding="utf-8")
        return scheduler.SchedulerReport(
            workflows_run=["daily-cycle"],
            historical_date=str(kwargs.get("date")),
            historical_hours=[str(h) for h in hours],
            details={
                "scheduler_report_paths": {
                    "json": str(json_path),
                    "markdown": str(md_path),
                }
            },
        )

    monkeypatch.setattr(scheduler, "cmd_daily_cycle", fake_daily_cycle)
    return captured_hours


class TestRunBatchSeries:
    def test_runs_three_chunks_for_15_to_23(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)
        runs_dir = tmp_path / "runs"
        captured = _mock_daily_cycle_success(monkeypatch, runs_dir=runs_dir)

        report = scheduler.cmd_run_batch_series(
            date="20230916",
            start_hour=15,
            end_hour=23,
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=runs_dir,
            daily_summary_dir=tmp_path / "daily",
        )

        assert captured == [[15, 16, 17], [18, 19, 20], [21, 22, 23]]
        assert report.chunks == [["15", "16", "17"], ["18", "19", "20"], ["21", "22", "23"]]
        assert report.chunks_completed == 3
        assert report.chunks_failed == 0
        assert len(report.per_chunk_report_paths) == 3

    def test_stops_on_first_failed_chunk_by_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)
        runs_dir = tmp_path / "runs"
        calls: list[list[int]] = []

        def fake_daily_cycle(**kwargs: object) -> scheduler.SchedulerReport:
            hours = list(kwargs.get("historical_hours") or [])
            calls.append(hours)
            report = scheduler.SchedulerReport(
                workflows_run=["daily-cycle"],
                historical_hours=[str(h) for h in hours],
            )
            if hours == [15, 16, 17]:
                report.failures.append("simulated chunk failure")
            else:
                report.details["scheduler_report_paths"] = {
                    "json": str(runs_dir / "ok.json"),
                    "markdown": str(runs_dir / "ok.md"),
                }
            return report

        monkeypatch.setattr(scheduler, "cmd_daily_cycle", fake_daily_cycle)

        report = scheduler.cmd_run_batch_series(
            date="20230916",
            start_hour=15,
            end_hour=23,
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=runs_dir,
            daily_summary_dir=tmp_path / "daily",
        )

        assert calls == [[15, 16, 17]]
        assert report.chunks_failed == 1
        assert report.chunks_completed == 0
        assert any("Stopped batch series" in w for w in report.warnings)

    def test_continues_with_continue_on_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)
        runs_dir = tmp_path / "runs"
        calls: list[list[int]] = []

        def fake_daily_cycle(**kwargs: object) -> scheduler.SchedulerReport:
            hours = list(kwargs.get("historical_hours") or [])
            calls.append(hours)
            report = scheduler.SchedulerReport(
                workflows_run=["daily-cycle"],
                historical_hours=[str(h) for h in hours],
                details={
                    "scheduler_report_paths": {
                        "json": str(runs_dir / f"{hours[0]}.json"),
                        "markdown": str(runs_dir / f"{hours[0]}.md"),
                    }
                },
            )
            if hours == [15, 16, 17]:
                report.failures.append("simulated chunk failure")
            return report

        monkeypatch.setattr(scheduler, "cmd_daily_cycle", fake_daily_cycle)

        report = scheduler.cmd_run_batch_series(
            date="20230916",
            start_hour=15,
            end_hour=23,
            chunk_size_hours=3,
            continue_on_error=True,
            state_path=state_path,
            runs_dir=runs_dir,
            daily_summary_dir=tmp_path / "daily",
        )

        assert calls == [[15, 16, 17], [18, 19, 20], [21, 22, 23]]
        assert report.chunks_failed == 1
        assert report.chunks_completed == 2

    def test_skips_already_completed_chunk(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)
        state = scheduler.load_state(state_path)
        state.completed_batches.append(
            {"date": "20230916", "hours": ["15", "16", "17"], "completed_at": "2026-01-01T00:00:00Z"}
        )
        scheduler.save_state(state, state_path)
        runs_dir = tmp_path / "runs"
        captured = _mock_daily_cycle_success(monkeypatch, runs_dir=runs_dir)

        report = scheduler.cmd_run_batch_series(
            date="20230916",
            start_hour=15,
            end_hour=23,
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=runs_dir,
            daily_summary_dir=tmp_path / "daily",
        )

        assert captured == [[18, 19, 20], [21, 22, 23]]
        assert report.chunks_skipped == 1
        assert report.chunks_completed == 2
        assert any("already completed" in w for w in report.warnings)

    def test_series_report_includes_safety_fields_and_chunk_paths(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)
        runs_dir = tmp_path / "runs"
        _mock_daily_cycle_success(monkeypatch, runs_dir=runs_dir)

        report = scheduler.cmd_run_batch_series(
            date="20230916",
            start_hour=15,
            end_hour=17,
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=runs_dir,
            daily_summary_dir=tmp_path / "daily",
        )

        state = scheduler.load_state(state_path)
        saved = json.loads(Path(state.last_report_path).read_text(encoding="utf-8"))
        assert saved["paper_only"] is True
        assert saved["execution_enabled"] is False
        assert saved["command"] == "run-batch-series"
        assert saved["per_chunk_report_paths"]
        assert all(entry.get("json") for entry in saved["per_chunk_report_paths"])
        assert report.chunks_completed == 1
        series_json = Path(state.last_report_path)
        series_md = series_json.with_name(series_json.stem + ".md")
        assert series_json.is_file()
        assert series_md.is_file()


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
            chunk_size_hours=3,
            train_interval="15s",
            horizon=5,
            state_dir=state_dir,
            project_root=tmp_path,
        )
        assert plist_path.is_file()
        assert setup_path.is_file()
        plist_text = plist_path.read_text(encoding="utf-8")
        assert "com.hyperliquid.research.daily" in plist_text
        assert "auto-backfill-next" in plist_text
        assert "chunk-size-hours" in plist_text
        assert "train-interval" in plist_text
        assert "horizon" in plist_text
        assert "hyperliquid-research" in plist_text
        setup_text = setup_path.read_text(encoding="utf-8")
        assert "not installed automatically" in setup_text.lower()
        assert "auto-backfill-next" in setup_text


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


class TestFindNextIncompleteChunk:
    def test_selects_first_incomplete_chunk(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        result = scheduler.find_next_incomplete_chunk(
            scheduler.load_state(state_path), chunk_size_hours=3
        )
        assert result == ("20230916", ["9", "10", "11"])

    def test_skips_completed_chunks(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        state = scheduler.load_state(state_path)
        state.completed_batches.append(
            {"date": "20230916", "hours": ["9", "10", "11"], "completed_at": "2026-01-01T00:00:00Z"}
        )
        scheduler.save_state(state, state_path)
        result = scheduler.find_next_incomplete_chunk(
            scheduler.load_state(state_path), chunk_size_hours=3
        )
        assert result == ("20230916", ["12", "13", "14"])

    def test_respects_hour_window_not_zero(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        state = scheduler.load_state(state_path)
        for hour in range(9, 15):
            state.completed_batches.append(
                {
                    "date": "20230916",
                    "hours": [str(hour)],
                    "completed_at": "2026-01-01T00:00:00Z",
                }
            )
        scheduler.save_state(state, state_path)
        result = scheduler.find_next_incomplete_chunk(
            scheduler.load_state(state_path), chunk_size_hours=3
        )
        assert result is None


class TestBackwardsCompatibleState:
    def test_defaults_start_end_hour_when_missing(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        payload = {
            "historical_start_date": "20230916",
            "historical_end_date_target": "20230917",
            "next_historical_date": "20230916",
            "coins": ["BTC"],
        }
        state_path.write_text(json.dumps(payload), encoding="utf-8")
        state = scheduler.load_state(state_path)
        assert state.historical_start_hour == 0
        assert state.historical_end_hour == 23
        assert state.failed_batches == []
        assert state.autopilot.enabled is False


def _mock_auto_daily_cycle(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failures: list[str] | None = None,
    validation_warning: str | None = None,
) -> list[dict[str, object]]:
    captured: list[dict[str, object]] = []

    def fake_daily_cycle(**kwargs: object) -> scheduler.SchedulerReport:
        captured.append(dict(kwargs))
        hours = kwargs.get("historical_hours") or []
        report = scheduler.SchedulerReport(
            workflows_run=["daily-cycle", "resample-validate", "train-evaluate-historical"],
            historical_date=str(kwargs.get("date")),
            historical_hours=[str(h) for h in hours],
            train_interval=str(kwargs.get("train_interval")),
            horizon=int(kwargs.get("horizon") or 5),
            details={
                "resample_validate": {
                    "workflows_run": ["resample-validate"],
                    "warnings": [validation_warning] if validation_warning else [],
                    "failures": [],
                },
                "historical_backfill": {
                    "details": {
                        "orchestrator_batch": {
                            "download": {
                                "file_results": [{"status": "downloaded"}],
                            }
                        }
                    }
                },
                "scheduler_report_paths": {
                    "json": "/tmp/daily.json",
                    "markdown": "/tmp/daily.md",
                },
            },
        )
        if failures:
            report.failures.extend(failures)
        return report

    monkeypatch.setattr(scheduler, "cmd_daily_cycle", fake_daily_cycle)
    return captured


def _enable_autopilot(
    state_path: Path,
    *,
    daily_chunk_limit: int = 1,
    daily_file_limit: int = 9,
    ramp_every_successful_days: int = 2,
    max_daily_chunk_limit: int = 4,
    max_daily_file_limit: int = 36,
) -> None:
    scheduler.cmd_configure_autopilot(
        enable=True,
        daily_chunk_limit=daily_chunk_limit,
        daily_file_limit=daily_file_limit,
        ramp_every_successful_days=ramp_every_successful_days,
        max_daily_chunk_limit=max_daily_chunk_limit,
        max_daily_file_limit=max_daily_file_limit,
        state_path=state_path,
    )


def _mock_auto_backfill_next(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failures: list[str] | None = None,
    validation_warning: str | None = None,
    downloaded_files: int = 1,
) -> list[dict[str, object]]:
    captured: list[dict[str, object]] = []

    def fake_auto_backfill(**kwargs: object) -> scheduler.SchedulerReport:
        captured.append(dict(kwargs))
        hours = kwargs.get("chunk_size_hours")
        report = scheduler.SchedulerReport(
            workflows_run=["auto-backfill-next", "daily-cycle"],
            historical_date="20230916",
            historical_hours=["9", "10", "11"],
            train_interval=str(kwargs.get("train_interval")),
            horizon=int(kwargs.get("horizon") or 5),
            details={
                "chunk_size_hours": hours,
                "resample_validate": {
                    "warnings": [validation_warning] if validation_warning else [],
                    "failures": [],
                },
                "historical_backfill": {
                    "details": {
                        "orchestrator_batch": {
                            "download": {
                                "file_results": [{"status": "downloaded"}] * downloaded_files,
                            }
                        }
                    }
                },
                "scheduler_report_paths": {
                    "json": "/tmp/auto.json",
                    "markdown": "/tmp/auto.md",
                },
            },
        )
        if failures:
            report.failures.extend(failures)
        return report

    monkeypatch.setattr(scheduler, "cmd_auto_backfill_next", fake_auto_backfill)
    return captured


class TestConfigureAutopilot:
    def test_enables_settings(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        autopilot = scheduler.cmd_configure_autopilot(
            enable=True,
            daily_chunk_limit=2,
            daily_file_limit=18,
            ramp_enabled=True,
            ramp_every_successful_days=3,
            ramp_chunk_increment=2,
            max_daily_chunk_limit=6,
            max_daily_file_limit=54,
            state_path=state_path,
        )
        assert autopilot.enabled is True
        assert autopilot.daily_chunk_limit == 2
        assert autopilot.daily_file_limit == 18
        assert autopilot.ramp_every_successful_days == 3
        assert autopilot.started_at is not None
        state = scheduler.load_state(state_path)
        assert state.autopilot.enabled is True

    def test_configure_does_not_reset_usage_counters(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        _enable_autopilot(state_path)
        state = scheduler.load_state(state_path)
        state.autopilot.last_run_date = "20230916"
        state.autopilot.runs_today = 2
        state.autopilot.chunks_completed_today = 1
        state.autopilot.files_downloaded_today = 9
        scheduler.save_state(state, state_path)

        scheduler.cmd_configure_autopilot(
            daily_chunk_limit=2,
            state_path=state_path,
        )
        state = scheduler.load_state(state_path)
        assert state.autopilot.runs_today == 2
        assert state.autopilot.chunks_completed_today == 1
        assert state.autopilot.files_downloaded_today == 9
        assert state.autopilot.last_run_date == "20230916"

    def test_configure_reset_usage_clears_counters(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        _enable_autopilot(state_path)
        state = scheduler.load_state(state_path)
        state.autopilot.runs_today = 2
        state.autopilot.chunks_completed_today = 1
        state.autopilot.files_downloaded_today = 9
        scheduler.save_state(state, state_path)

        scheduler.cmd_configure_autopilot(reset_usage=True, state_path=state_path)
        state = scheduler.load_state(state_path)
        assert state.autopilot.runs_today == 0
        assert state.autopilot.chunks_completed_today == 0
        assert state.autopilot.files_downloaded_today == 0

    def test_pause_resume_disable(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        _enable_autopilot(state_path)
        scheduler.cmd_configure_autopilot(
            pause_reason="manual hold",
            state_path=state_path,
        )
        state = scheduler.load_state(state_path)
        assert state.autopilot.paused is True
        assert state.autopilot.pause_reason == "manual hold"

        scheduler.cmd_configure_autopilot(resume=True, state_path=state_path)
        state = scheduler.load_state(state_path)
        assert state.autopilot.paused is False
        assert state.autopilot.failed_runs_in_row == 0

        scheduler.cmd_configure_autopilot(disable=True, state_path=state_path)
        state = scheduler.load_state(state_path)
        assert state.autopilot.enabled is False


class TestAutopilotStatus:
    def test_reads_settings(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        _enable_autopilot(state_path)
        status = scheduler.cmd_autopilot_status(state_path=state_path)
        assert status["enabled"] is True
        assert status["daily_chunk_limit"] == 1
        assert status["daily_file_limit"] == 9
        assert status["ramp_enabled"] is True
        assert "autopilot-tick" in status["next_recommended_action"]


class TestAutopilotTick:
    def test_exits_when_disabled(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        report = scheduler.cmd_autopilot_tick(
            state_path=state_path,
            runs_dir=tmp_path / "runs",
        )
        assert report.action == "disabled"
        assert scheduler.AUTOPILOT_DISABLED_MESSAGE in report.warnings[0]

    def test_exits_when_paused(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        _enable_autopilot(state_path)
        scheduler.cmd_configure_autopilot(pause_reason="testing", state_path=state_path)
        report = scheduler.cmd_autopilot_tick(
            state_path=state_path,
            runs_dir=tmp_path / "runs",
        )
        assert report.action == "paused"
        assert "paused" in report.warnings[0].lower()

    def test_resets_daily_counters_on_new_date(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path)
        _mock_auto_backfill_next(monkeypatch)
        state = scheduler.load_state(state_path)
        state.autopilot.last_run_date = "20230915"
        state.autopilot.runs_today = 5
        state.autopilot.chunks_completed_today = 1
        state.autopilot.files_downloaded_today = 9
        scheduler.save_state(state, state_path)

        scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )
        state = scheduler.load_state(state_path)
        assert state.autopilot.last_run_date == "20230916"
        assert state.autopilot.runs_today == 1
        assert state.autopilot.chunks_completed_today == 1

    def test_respects_daily_chunk_limit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path, daily_chunk_limit=1)
        _mock_auto_backfill_next(monkeypatch)

        scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )
        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )
        assert report.action == "limit_skipped"

    def test_respects_daily_file_limit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path, daily_chunk_limit=4, daily_file_limit=9)
        _mock_auto_backfill_next(monkeypatch, downloaded_files=9)

        scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )
        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )
        assert report.action == "limit_skipped"

    def test_second_tick_same_day_skips_without_backfill(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path, daily_chunk_limit=1)
        captured = _mock_auto_backfill_next(monkeypatch)

        first = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )
        second = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )

        assert first.action == "chunk_completed"
        assert second.action == "limit_skipped"
        assert len(captured) == 1

    def test_runs_exactly_one_chunk_when_allowed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path)
        captured = _mock_auto_backfill_next(monkeypatch)

        scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )
        assert len(captured) == 1

    def test_successful_tick_increments_counters(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path)
        _mock_auto_backfill_next(monkeypatch, downloaded_files=3)

        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )
        assert report.action == "chunk_completed"
        state = scheduler.load_state(state_path)
        assert state.autopilot.runs_today == 1
        assert state.autopilot.chunks_completed_today == 1
        assert state.autopilot.files_downloaded_today == 3
        assert state.autopilot.failed_runs_in_row == 0
        assert state.autopilot.successful_days_in_row == 1

    def test_failed_tick_increments_failed_runs_and_does_not_ramp(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path)
        _mock_auto_backfill_next(monkeypatch, failures=["download failed"])

        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )
        assert report.action == "failed"
        state = scheduler.load_state(state_path)
        assert state.autopilot.chunks_completed_today == 0
        assert state.autopilot.failed_runs_in_row == 1
        assert state.autopilot.daily_chunk_limit == 1
        assert state.autopilot.successful_days_in_row == 0

    def test_three_failures_pauses_autopilot(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path)
        _mock_auto_backfill_next(monkeypatch, failures=["download failed"])

        for _ in range(3):
            scheduler.cmd_autopilot_tick(
                chunk_size_hours=3,
                state_path=state_path,
                runs_dir=tmp_path / "runs",
                today_override="20230916",
            )

        state = scheduler.load_state(state_path)
        assert state.autopilot.paused is True
        assert state.autopilot.pause_reason is not None
        assert state.autopilot.failed_runs_in_row >= 3

    def test_ramp_increases_limits_after_successful_days(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path, ramp_every_successful_days=2)
        _mock_auto_backfill_next(monkeypatch)

        scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )
        state = scheduler.load_state(state_path)
        assert state.autopilot.successful_days_in_row == 1
        assert state.autopilot.daily_chunk_limit == 1

        scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230917",
        )
        state = scheduler.load_state(state_path)
        assert state.autopilot.daily_chunk_limit == 2
        assert state.autopilot.daily_file_limit == 18
        assert state.autopilot.successful_days_in_row == 0

    def test_ramp_caps_at_max_limits(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        scheduler.cmd_configure_autopilot(
            enable=True,
            daily_chunk_limit=3,
            daily_file_limit=27,
            ramp_every_successful_days=1,
            ramp_chunk_increment=2,
            max_daily_chunk_limit=4,
            max_daily_file_limit=36,
            state_path=state_path,
        )
        _mock_auto_backfill_next(monkeypatch)

        scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )
        state = scheduler.load_state(state_path)
        assert state.autopilot.daily_chunk_limit == 4
        assert state.autopilot.daily_file_limit == 36

    def test_report_has_safety_fields(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path)
        _mock_auto_backfill_next(monkeypatch)
        runs_dir = tmp_path / "runs"

        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=runs_dir,
            today_override="20230916",
        )
        assert report.paper_only is True
        assert report.execution_enabled is False
        state = scheduler.load_state(state_path)
        saved = json.loads(Path(state.autopilot.last_autopilot_report_path).read_text(encoding="utf-8"))
        assert saved["paper_only"] is True
        assert saved["execution_enabled"] is False
        assert "secret_access_key" not in json.dumps(saved).lower()

    def test_file_limit_blocks_before_chunk_when_requested_files_exceed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path, daily_chunk_limit=4, daily_file_limit=9)
        captured = _mock_auto_backfill_next(monkeypatch)
        state = scheduler.load_state(state_path)
        state.autopilot.last_run_date = "20230916"
        state.autopilot.files_downloaded_today = 7
        scheduler.save_state(state, state_path)

        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )

        assert report.action == "limit_skipped"
        assert report.requested_files == 9
        assert len(captured) == 0

    def test_limit_skipped_report_includes_safety_and_limit_fields(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path, daily_chunk_limit=1)
        _mock_auto_backfill_next(monkeypatch)
        runs_dir = tmp_path / "runs"

        scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=runs_dir,
            today_override="20230916",
        )
        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=runs_dir,
            today_override="20230916",
        )

        assert report.action == "limit_skipped"
        assert report.paper_only is True
        assert report.execution_enabled is False
        assert report.daily_chunk_limit == 1
        assert report.daily_file_limit == 9
        assert report.chunks_completed_today == 1
        assert report.requested_files == 9
        state = scheduler.load_state(state_path)
        saved = json.loads(Path(state.autopilot.last_autopilot_report_path).read_text(encoding="utf-8"))
        assert saved["action"] == "limit_skipped"
        assert saved["paper_only"] is True
        assert saved["execution_enabled"] is False
        assert saved["requested_files"] == 9

    def test_successful_tick_persists_counters_to_state_json(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path)
        _mock_auto_backfill_next(monkeypatch, downloaded_files=3)

        scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )

        payload = json.loads(state_path.read_text(encoding="utf-8"))
        autopilot = payload["autopilot"]
        assert autopilot["runs_today"] == 1
        assert autopilot["chunks_completed_today"] == 1
        assert autopilot["files_downloaded_today"] == 3
        assert autopilot["last_run_date"] == "20230916"

    def test_failed_tick_does_not_increment_completed_or_file_counters(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path)
        _mock_auto_backfill_next(monkeypatch, failures=["download failed"])

        scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )

        payload = json.loads(state_path.read_text(encoding="utf-8"))
        autopilot = payload["autopilot"]
        assert autopilot["chunks_completed_today"] == 0
        assert autopilot["files_downloaded_today"] == 0
        assert autopilot["runs_today"] == 0
        assert autopilot["failed_runs_in_row"] == 1


class TestLaunchdAutopilot:
    def test_autopilot_template(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "scheduler"
        plist_path, setup_path = scheduler.cmd_write_launchd_template(
            run_time="02:00",
            coins=["BTC", "ETH", "SOL"],
            aws_profile="hyperliquid-research",
            requester_pays=True,
            chunk_size_hours=3,
            train_interval="15s",
            horizon=5,
            state_dir=state_dir,
            project_root=tmp_path,
            autopilot=True,
        )
        plist_text = plist_path.read_text(encoding="utf-8")
        setup_text = setup_path.read_text(encoding="utf-8")
        assert "autopilot-tick" in plist_text
        assert "auto-backfill-next" not in plist_text
        assert "com.hyperliquid.research.autopilot" in plist_text
        assert "configure-autopilot" in setup_text
        assert "AWS Budget" in setup_text

    def test_interval_hours_schedule(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "scheduler"
        plist_path, _ = scheduler.cmd_write_launchd_template(
            coins=["BTC"],
            state_dir=state_dir,
            project_root=tmp_path,
            autopilot=True,
            interval_hours=6,
        )
        plist_text = plist_path.read_text(encoding="utf-8")
        assert "StartInterval" in plist_text
        assert "<integer>21600</integer>" in plist_text


class TestLaunchdHourlyAutopilot:
    def test_writes_hourly_plist_and_setup(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "scheduler"
        plist_path, setup_path = scheduler.cmd_write_hourly_launchd_template(
            state_dir=state_dir,
            project_root=tmp_path,
        )
        assert plist_path.name == scheduler.LAUNCHD_PLIST_NAME_AUTOPILOT_HOURLY
        assert setup_path.name == scheduler.LAUNCHD_SETUP_MD_HOURLY
        assert plist_path.is_file()
        assert setup_path.is_file()

    def test_start_interval_default_3600(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "scheduler"
        plist_path, _ = scheduler.cmd_write_hourly_launchd_template(
            state_dir=state_dir,
            project_root=tmp_path,
        )
        plist_text = plist_path.read_text(encoding="utf-8")
        assert "<key>StartInterval</key>" in plist_text
        assert "<integer>3600</integer>" in plist_text

    def test_custom_interval_seconds(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "scheduler"
        plist_path, _ = scheduler.cmd_write_hourly_launchd_template(
            interval_seconds=7200,
            state_dir=state_dir,
            project_root=tmp_path,
        )
        plist_text = plist_path.read_text(encoding="utf-8")
        assert "<integer>7200</integer>" in plist_text

    def test_run_at_load_true_by_default(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "scheduler"
        plist_path, _ = scheduler.cmd_write_hourly_launchd_template(
            state_dir=state_dir,
            project_root=tmp_path,
        )
        plist_text = plist_path.read_text(encoding="utf-8")
        assert "<key>RunAtLoad</key>" in plist_text
        assert "<true/>" in plist_text

    def test_run_at_load_false(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "scheduler"
        plist_path, _ = scheduler.cmd_write_hourly_launchd_template(
            run_at_load=False,
            state_dir=state_dir,
            project_root=tmp_path,
        )
        plist_text = plist_path.read_text(encoding="utf-8")
        assert "<key>RunAtLoad</key>" not in plist_text

    def test_program_arguments_autopilot_tick(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "scheduler"
        plist_path, _ = scheduler.cmd_write_hourly_launchd_template(
            state_dir=state_dir,
            project_root=tmp_path,
        )
        plist_text = plist_path.read_text(encoding="utf-8")
        assert "bid_ask_research_scheduler.py" in plist_text
        assert "<string>autopilot-tick</string>" in plist_text
        assert "auto-backfill-next" not in plist_text

    def test_stdout_stderr_log_paths(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "scheduler"
        stdout_log = state_dir / "custom.out.log"
        stderr_log = state_dir / "custom.err.log"
        plist_path, _ = scheduler.cmd_write_hourly_launchd_template(
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            state_dir=state_dir,
            project_root=tmp_path,
        )
        plist_text = plist_path.read_text(encoding="utf-8")
        assert str(stdout_log) in plist_text
        assert str(stderr_log) in plist_text

    def test_setup_includes_daily_cap_warning(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "scheduler"
        _, setup_path = scheduler.cmd_write_hourly_launchd_template(
            state_dir=state_dir,
            project_root=tmp_path,
        )
        setup_text = setup_path.read_text(encoding="utf-8")
        assert "Daily caps protect the hourly schedule" in setup_text
        assert "daily-chunk-limit 8" in setup_text
        assert "daily-file-limit 24" in setup_text
        assert "max_daily_chunk_limit" in setup_text
        assert "max_daily_file_limit" in setup_text

    def test_setup_includes_unload_old_daily_plist(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "scheduler"
        _, setup_path = scheduler.cmd_write_hourly_launchd_template(
            state_dir=state_dir,
            project_root=tmp_path,
        )
        setup_text = setup_path.read_text(encoding="utf-8")
        assert (
            f"launchctl unload ~/Library/LaunchAgents/{scheduler.LAUNCHD_PLIST_NAME_AUTOPILOT}"
            in setup_text
        )
        assert "autopilot-status" in setup_text
        assert "tail -50" in setup_text
        assert "research-only" in setup_text.lower()
        assert "paper-only" in setup_text.lower()

    def test_custom_label(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "scheduler"
        plist_path, _ = scheduler.cmd_write_hourly_launchd_template(
            label="com.example.hourly",
            state_dir=state_dir,
            project_root=tmp_path,
        )
        plist_text = plist_path.read_text(encoding="utf-8")
        assert "<string>com.example.hourly</string>" in plist_text


class TestAutoBackfillNext:
    def test_selects_first_incomplete_chunk(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        captured = _mock_auto_daily_cycle(monkeypatch)

        report = scheduler.cmd_auto_backfill_next(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            daily_summary_dir=tmp_path / "daily",
        )

        assert captured[0]["historical_hours"] == [9, 10, 11]
        assert captured[0]["date"] == "20230916"
        assert captured[0]["defer_batch_completion"] is True
        assert report.historical_hours == ["9", "10", "11"]

    def test_skips_completed_chunks_via_state(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        state = scheduler.load_state(state_path)
        state.completed_batches.append(
            {"date": "20230916", "hours": ["9", "10", "11"], "completed_at": "2026-01-01T00:00:00Z"}
        )
        scheduler.save_state(state, state_path)
        captured = _mock_auto_daily_cycle(monkeypatch)

        scheduler.cmd_auto_backfill_next(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            daily_summary_dir=tmp_path / "daily",
        )

        assert captured[0]["historical_hours"] == [12, 13, 14]

    def test_marks_complete_on_success(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=11)
        _mock_auto_daily_cycle(monkeypatch)

        scheduler.cmd_auto_backfill_next(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            daily_summary_dir=tmp_path / "daily",
        )

        state = scheduler.load_state(state_path)
        assert len(state.completed_batches) == 1
        assert state.completed_batches[0]["hours"] == ["9", "10", "11"]
        assert state.completed_batches[0].get("report_path") == "/tmp/daily.json"

    def test_advances_date_after_all_chunks_complete(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start="20230916", end="20230917", start_hour=9, end_hour=11)
        _mock_auto_daily_cycle(monkeypatch)

        scheduler.cmd_auto_backfill_next(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            daily_summary_dir=tmp_path / "daily",
        )

        state = scheduler.load_state(state_path)
        assert state.next_historical_date == "20230917"
        assert "20230916" in state.completed_dates

    def test_exits_cleanly_when_range_complete(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start="20230916", end="20230916", start_hour=9, end_hour=11)
        state = scheduler.load_state(state_path)
        state.completed_batches.append(
            {"date": "20230916", "hours": ["9", "10", "11"], "completed_at": "2026-01-01T00:00:00Z"}
        )
        state.next_historical_date = None
        scheduler.save_state(state, state_path)
        captured = _mock_auto_daily_cycle(monkeypatch)

        report = scheduler.cmd_auto_backfill_next(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            daily_summary_dir=tmp_path / "daily",
        )

        assert captured == []
        assert any(scheduler.AUTO_BACKFILL_COMPLETE_MESSAGE in w for w in report.warnings)

    def test_does_not_mark_complete_on_validation_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=11)
        _mock_auto_daily_cycle(
            monkeypatch,
            validation_warning="BTC 1s validation failed: gaps detected",
        )

        scheduler.cmd_auto_backfill_next(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            daily_summary_dir=tmp_path / "daily",
        )

        state = scheduler.load_state(state_path)
        assert state.completed_batches == []
        assert len(state.failed_batches) == 1
        assert state.failed_batches[0]["hours"] == ["9", "10", "11"]

    def test_records_failed_batch_on_cycle_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=11)
        _mock_auto_daily_cycle(monkeypatch, failures=["download failed"])

        scheduler.cmd_auto_backfill_next(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            daily_summary_dir=tmp_path / "daily",
        )

        state = scheduler.load_state(state_path)
        assert state.completed_batches == []
        assert len(state.failed_batches) == 1
        assert "download failed" in state.failed_batches[0]["reason"]

    def test_report_has_safety_fields(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=11)
        _mock_auto_daily_cycle(monkeypatch)
        runs_dir = tmp_path / "runs"

        report = scheduler.cmd_auto_backfill_next(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=runs_dir,
            daily_summary_dir=tmp_path / "daily",
        )

        assert report.paper_only is True
        assert report.execution_enabled is False
        state = scheduler.load_state(state_path)
        saved = json.loads(Path(state.last_report_path).read_text(encoding="utf-8"))
        assert saved["paper_only"] is True
        assert saved["execution_enabled"] is False


class TestRepairStateFromExistingData:
    def _write_mock_historical_csv(
        self,
        data_root: Path,
        coin: str,
        date: str,
        hours: list[int],
    ) -> None:
        from bid_ask_schema import (
            BID_ASK_CSV_COLUMNS,
            BEST_ASK,
            BEST_BID,
            BID_SIZE,
            ASK_SIZE,
            TIMESTAMP_MS,
        )

        csv_dir = data_root / "historical_1s"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"{coin}_1s.csv"
        rows: list[dict[str, str]] = []
        for hour in hours:
            base_ms = int(
                __import__("datetime")
                .datetime.strptime(date, "%Y%m%d")
                .replace(hour=hour, tzinfo=__import__("datetime").timezone.utc)
                .timestamp()
                * 1000
            )
            rows.append(
                {
                    TIMESTAMP_MS: str(base_ms + 1000),
                    BEST_BID: "100.0",
                    BEST_ASK: "100.1",
                    BID_SIZE: "1.0",
                    ASK_SIZE: "1.0",
                    "coin": coin,
                    "source_session_id": "historical_archive",
                    "source_channel": "historical_l2Book",
                }
            )
        import csv

        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=BID_ASK_CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_marks_chunks_complete_when_data_validates(
        self,
        tmp_path: Path,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        data_root = tmp_path / "data"
        for coin in ["BTC", "ETH", "SOL"]:
            self._write_mock_historical_csv(data_root, coin, "20230916", [9, 10, 11, 12, 13, 14])

        report = scheduler.cmd_repair_state_from_existing_data(
            date="20230916",
            start_hour=9,
            end_hour=14,
            chunk_size_hours=3,
            data_root=data_root,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
        )

        state = scheduler.load_state(state_path)
        assert report.row_counts["chunks_repaired"] == 2
        assert len(state.completed_batches) == 2
        assert scheduler.REPAIR_INFERRED_WARNING in report.warnings
        assert report.paper_only is True
        assert report.execution_enabled is False
