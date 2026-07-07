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
    advance_csv: bool = True,
) -> list[dict[str, object]]:
    captured: list[dict[str, object]] = []

    def fake_daily_cycle(**kwargs: object) -> scheduler.SchedulerReport:
        captured.append(dict(kwargs))
        hours = kwargs.get("historical_hours") or []
        data_root = kwargs.get("data_root")
        coins = list(kwargs.get("coins") or ["BTC", "ETH", "SOL"])
        date = str(kwargs.get("date") or "20230916")
        if failures is None and advance_csv and isinstance(data_root, Path):
            for coin in coins:
                hour = int(hours[0]) if hours else 9
                _append_historical_csv(data_root, coin, date=date, hour=hour)

        report = scheduler.SchedulerReport(
            workflows_run=["daily-cycle", "resample-validate", "train-evaluate-historical"],
            historical_date=date,
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
                                "file_results": [
                                    {"status": "downloaded", "download_status": "downloaded"}
                                ],
                                "downloaded_count": 1,
                            },
                            "import": [
                                {
                                    "coin": coin,
                                    "import_status": "imported",
                                    "rows_written": 100,
                                }
                                for coin in coins
                            ],
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


def _append_historical_csv(
    data_root: Path,
    coin: str,
    *,
    date: str,
    hour: int,
) -> Path:
    from bid_ask_schema import (
        ASK_SIZE,
        BEST_ASK,
        BEST_BID,
        BID_ASK_CSV_COLUMNS,
        BID_SIZE,
        TIMESTAMP_MS,
    )

    csv_dir = data_root / "historical_1s"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{coin}_1s.csv"
    base_ms = int(
        __import__("datetime")
        .datetime.strptime(date, "%Y%m%d")
        .replace(hour=hour, tzinfo=__import__("datetime").timezone.utc)
        .timestamp()
        * 1000
    )
    row = {
        TIMESTAMP_MS: str(base_ms + 1000),
        BEST_BID: "100.0",
        BEST_ASK: "100.1",
        BID_SIZE: "1.0",
        ASK_SIZE: "1.0",
        "coin": coin,
        "source_session_id": "historical_archive",
        "source_channel": "historical_l2Book",
    }
    import csv

    write_header = not csv_path.is_file()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BID_ASK_CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return csv_path


def _mock_auto_backfill_next(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failures: list[str] | None = None,
    validation_warning: str | None = None,
    downloaded_files: int = 1,
    skipped_existing: int = 0,
    advance_csv: bool = True,
    inject_import_progress: bool = True,
    import_failures: list[str] | None = None,
) -> list[dict[str, object]]:
    captured: list[dict[str, object]] = []
    call_counter = {"n": 0}

    def _build_import_progress(coins: list[str]) -> dict[str, object]:
        if import_failures:
            return {
                "any_import_progress": False,
                "files_downloaded_new": downloaded_files,
                "files_skipped_existing": skipped_existing,
                "files_imported": 0,
                "files_requested": downloaded_files + skipped_existing,
                "import_failures": list(import_failures),
                "import_progress_by_coin": {coin: False for coin in coins},
                "rows_added_by_coin": {coin: 0 for coin in coins},
                "timestamp_advanced_by_coin": {coin: False for coin in coins},
                "pre_import_coverage": {},
                "post_import_coverage": {},
            }
        progressed = failures is None and advance_csv
        return {
            "any_import_progress": progressed,
            "files_downloaded_new": downloaded_files if progressed else 0,
            "files_skipped_existing": skipped_existing,
            "files_imported": downloaded_files if progressed else 0,
            "files_requested": downloaded_files + skipped_existing,
            "import_failures": [],
            "import_progress_by_coin": {coin: progressed for coin in coins},
            "rows_added_by_coin": {coin: (1 if progressed else 0) for coin in coins},
            "timestamp_advanced_by_coin": {coin: progressed for coin in coins},
            "pre_import_coverage": {},
            "post_import_coverage": {},
        }

    def fake_auto_backfill(**kwargs: object) -> scheduler.SchedulerReport:
        captured.append(dict(kwargs))
        data_root = kwargs.get("data_root")
        coins = list(kwargs.get("coins") or ["BTC", "ETH", "SOL"])
        if failures is None and advance_csv and isinstance(data_root, Path):
            call_counter["n"] += 1
            hour = 9 + call_counter["n"]
            for coin in coins:
                _append_historical_csv(data_root, coin, date="20230916", hour=hour)

        file_results: list[dict[str, str]] = []
        for _ in range(downloaded_files):
            file_results.append({"status": "downloaded", "download_status": "downloaded"})
        for _ in range(skipped_existing):
            file_results.append(
                {"status": "skipped_existing", "download_status": "skipped_existing"}
            )

        import_payload: list[dict[str, object]] = []
        if downloaded_files > 0 and failures is None and advance_csv:
            import_payload = [
                {"coin": coin, "import_status": "imported", "rows_written": 100}
                for coin in coins[: max(1, downloaded_files // max(1, len(coins)))]
            ]

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
                                "file_results": file_results,
                                "downloaded_count": downloaded_files,
                                "skipped_count": skipped_existing,
                            },
                            "import": import_payload,
                        }
                    }
                },
                "scheduler_report_paths": {
                    "json": "/tmp/auto.json",
                    "markdown": "/tmp/auto.md",
                },
            },
        )
        if inject_import_progress:
            report.details["import_progress"] = _build_import_progress(coins)
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


class TestAutopilotImportProgress:
    def test_chunk_completed_only_when_historical_1s_advances(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path)
        data_root = tmp_path / "data"
        _mock_auto_backfill_next(
            monkeypatch,
            downloaded_files=3,
            inject_import_progress=False,
            advance_csv=True,
        )

        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            data_root=data_root,
            today_override="20230916",
        )

        assert report.action == "chunk_completed"
        assert report.import_progress_by_coin.get("BTC") is True
        assert report.rows_added_by_coin.get("BTC", 0) > 0

    def test_no_import_progress_when_csv_unchanged(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path)
        data_root = tmp_path / "data"
        for coin in ["BTC", "ETH", "SOL"]:
            _write_historical_csv(data_root, coin, date="20230916", hour=12)
        _mock_auto_backfill_next(
            monkeypatch,
            downloaded_files=3,
            skipped_existing=0,
            advance_csv=False,
            inject_import_progress=False,
        )

        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            data_root=data_root,
            today_override="20230916",
        )

        assert report.action == "no_import_progress"
        state = scheduler.load_state(state_path)
        assert state.autopilot.chunks_completed_today == 0
        assert state.autopilot.files_downloaded_today == 0

    def test_skipped_existing_files_not_counted_as_new_downloads(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path)
        _mock_auto_backfill_next(
            monkeypatch,
            downloaded_files=0,
            skipped_existing=9,
            advance_csv=False,
        )

        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )

        assert report.action == "no_import_progress"
        assert report.files_downloaded_new == 0
        assert report.files_skipped_existing == 9

    def test_report_includes_pre_post_coverage_and_rows_added(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path)
        data_root = tmp_path / "data"
        _mock_auto_backfill_next(
            monkeypatch,
            downloaded_files=2,
            inject_import_progress=False,
            advance_csv=True,
        )

        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            data_root=data_root,
            today_override="20230916",
        )

        assert report.pre_import_coverage
        assert report.post_import_coverage
        assert "BTC" in report.rows_added_by_coin
        saved = json.loads(
            Path(report.details["scheduler_report_paths"]["json"]).read_text(encoding="utf-8")
        )
        assert "pre_import_coverage" in saved
        assert "rows_added_by_coin" in saved

    def test_files_downloaded_today_increments_only_for_new_files(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path)
        _mock_auto_backfill_next(monkeypatch, downloaded_files=3, skipped_existing=6)

        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )

        assert report.action == "chunk_completed"
        assert report.files_downloaded_new == 3
        assert report.files_skipped_existing == 6
        state = scheduler.load_state(state_path)
        assert state.autopilot.files_downloaded_today == 3

    def test_phase_mode_no_progress_does_not_consume_daily_chunk_quota(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        scheduler.cmd_configure_phase_backfill(
            enable=True,
            coins=["BTC", "ETH"],
            target_days=14,
            daily_file_limit=144,
            daily_chunk_limit=24,
            require_confirmation_text=scheduler.PHASE_BACKFILL_CONFIRMATION_TEXT,
            state_path=state_path,
        )
        _mock_auto_backfill_next(
            monkeypatch,
            downloaded_files=0,
            skipped_existing=6,
            advance_csv=False,
        )

        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )

        assert report.action == "no_import_progress"
        state = scheduler.load_state(state_path)
        assert state.autopilot.chunks_completed_today == 0
        assert state.autopilot.runs_today == 0

        _mock_auto_backfill_next(monkeypatch, downloaded_files=3)
        second = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )
        assert second.action == "chunk_completed"
        state = scheduler.load_state(state_path)
        assert state.autopilot.chunks_completed_today == 1

    def test_audit_recommends_importer_issue_for_no_import_progress(
        self,
        tmp_path: Path,
    ) -> None:
        runs_dir = tmp_path / "runs"
        _write_autopilot_report(
            runs_dir,
            action="no_import_progress",
            files_downloaded_today=0,
            chunks_completed_today=0,
        )

        audit = scheduler.cmd_phase_backfill_audit(
            coins=["BTC", "ETH"],
            data_root=tmp_path / "data",
            downloads_root=tmp_path / "downloads",
            reports_dir=runs_dir,
            candidate_reports_dir=tmp_path / "candidate",
            state_path=tmp_path / "missing-state.json",
        )

        assert audit["scheduler_progress"]["latest_action"] == "no_import_progress"
        assert audit["next_recommended_action"] == "investigate date/hour planning or importer path"

    def test_import_failed_action_on_importer_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=14)
        _enable_autopilot(state_path)
        _mock_auto_backfill_next(
            monkeypatch,
            import_failures=["BTC: import failed: corrupt jsonl"],
        )

        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )

        assert report.action == "import_failed"
        assert state_path
        state = scheduler.load_state(state_path)
        assert state.autopilot.chunks_completed_today == 0


class TestImportProgressHelpers:
    def test_compute_import_progress_delta(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        pre = scheduler._historical_1s_coverage_by_coin(["BTC"], data_root)
        _append_historical_csv(data_root, "BTC", date="20230916", hour=10)
        post = scheduler._historical_1s_coverage_by_coin(["BTC"], data_root)
        delta = scheduler._compute_import_progress_delta(pre, post)

        assert delta["rows_added_by_coin"]["BTC"] == 1
        assert delta["timestamp_advanced_by_coin"]["BTC"] is True
        assert delta["import_progress_by_coin"]["BTC"] is True
        assert delta["any_import_progress"] is True

    def test_summarize_batch_file_stats_uses_download_status(self) -> None:
        stats = scheduler._summarize_batch_file_stats(
            {
                "download": {
                    "file_results": [
                        {"download_status": "downloaded"},
                        {"download_status": "skipped_existing"},
                    ]
                },
                "import": [{"import_status": "imported", "rows_written": 50}],
            }
        )
        assert stats["files_downloaded_new"] == 1
        assert stats["files_skipped_existing"] == 1
        assert stats["files_imported"] == 1


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
            data_root=tmp_path / "data",
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
            data_root=tmp_path / "data",
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
            data_root=tmp_path / "data",
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
            data_root=tmp_path / "data",
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


def _write_historical_csv_at_timestamp(
    data_root: Path,
    coin: str,
    *,
    timestamp_ms: int,
) -> Path:
    from bid_ask_schema import (
        ASK_SIZE,
        BEST_ASK,
        BEST_BID,
        BID_ASK_CSV_COLUMNS,
        BID_SIZE,
        TIMESTAMP_MS,
    )

    csv_dir = data_root / "historical_1s"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{coin}_1s.csv"
    row = {
        TIMESTAMP_MS: str(timestamp_ms),
        BEST_BID: "100.0",
        BEST_ASK: "100.1",
        BID_SIZE: "1.0",
        ASK_SIZE: "1.0",
        "coin": coin,
        "source_session_id": "historical_archive",
        "source_channel": "historical_l2Book",
    }
    import csv

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BID_ASK_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)
    return csv_path


def _mark_hours_completed(
    state_path: Path,
    date: str,
    hours: list[int],
) -> None:
    state = scheduler.load_state(state_path)
    state.completed_batches.append(
        {
            "date": date,
            "hours": [str(h) for h in hours],
            "completed_at": "2026-01-01T00:00:00Z",
        }
    )
    scheduler.save_state(state, state_path)


# 2023-09-17T23:59:58.235Z
TS_20230917_235958Z = 1694995198235
# 2023-09-17T20:59:58Z — covers hours 0-20 but not 21-23
TS_20230917_205958Z = 1694984398000
# 2023-09-17T22:30:00Z — covers hour 21 but not 22-23
TS_20230917_223000Z = 1694989800000


class TestCoveredChunkAdvancement:
    def test_is_planned_chunk_fully_covered_by_latest_timestamp(
        self,
        tmp_path: Path,
    ) -> None:
        data_root = tmp_path / "data"
        for coin in ["BTC", "ETH"]:
            _write_historical_csv_at_timestamp(
                data_root,
                coin,
                timestamp_ms=TS_20230917_235958Z,
            )

        assert scheduler.is_planned_chunk_fully_covered_by_historical_1s(
            coins=["BTC", "ETH"],
            date="20230917",
            hours=["21", "22", "23"],
            data_root=data_root,
        )
        assert not scheduler.is_planned_chunk_fully_covered_by_historical_1s(
            coins=["BTC", "ETH"],
            date="20230918",
            hours=["0", "1", "2"],
            data_root=data_root,
        )

    def test_one_coin_lagging_prevents_skip(
        self,
        tmp_path: Path,
    ) -> None:
        data_root = tmp_path / "data"
        _write_historical_csv_at_timestamp(data_root, "BTC", timestamp_ms=TS_20230917_235958Z)
        _write_historical_csv_at_timestamp(data_root, "ETH", timestamp_ms=TS_20230917_205958Z)

        assert not scheduler.is_planned_chunk_fully_covered_by_historical_1s(
            coins=["BTC", "ETH"],
            date="20230917",
            hours=["21", "22", "23"],
            data_root=data_root,
        )

    def test_auto_backfill_skips_covered_final_chunk_and_advances(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(
            tmp_path,
            start="20230917",
            end="20230930",
            start_hour=0,
            end_hour=23,
        )
        _mark_hours_completed(state_path, "20230917", list(range(0, 21)))
        data_root = tmp_path / "data"
        for coin in ["BTC", "ETH"]:
            _write_historical_csv_at_timestamp(
                data_root,
                coin,
                timestamp_ms=TS_20230917_235958Z,
            )
        captured = _mock_auto_daily_cycle(monkeypatch)

        report = scheduler.cmd_auto_backfill_next(
            chunk_size_hours=3,
            coins=["BTC", "ETH"],
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            daily_summary_dir=tmp_path / "daily",
            data_root=data_root,
        )

        assert captured[0]["date"] == "20230918"
        assert captured[0]["historical_hours"] == [0, 1, 2]
        assert report.historical_date == "20230918"
        assert report.historical_hours == ["0", "1", "2"]
        selection = report.details["chunk_selection"]
        assert selection["original_planned_date"] == "20230917"
        assert selection["original_planned_hours"] == ["21", "22", "23"]
        assert selection["selected_date"] == "20230918"
        assert selection["selected_hours"] == ["0", "1", "2"]
        assert selection["selected_chunk_was_advanced"] is True
        assert len(selection["skipped_covered_chunks"]) == 1
        assert selection["reason"] == scheduler.CHUNK_ALREADY_COVERED_REASON

    def test_does_not_skip_uncovered_chunk(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(
            tmp_path,
            start="20230917",
            end="20230930",
            start_hour=0,
            end_hour=23,
        )
        _mark_hours_completed(state_path, "20230917", list(range(0, 21)))
        data_root = tmp_path / "data"
        for coin in ["BTC", "ETH"]:
            _write_historical_csv_at_timestamp(
                data_root,
                coin,
                timestamp_ms=TS_20230917_205958Z,
            )
        captured = _mock_auto_daily_cycle(monkeypatch)

        report = scheduler.cmd_auto_backfill_next(
            chunk_size_hours=3,
            coins=["BTC", "ETH"],
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            daily_summary_dir=tmp_path / "daily",
            data_root=data_root,
        )

        assert captured[0]["date"] == "20230917"
        assert captured[0]["historical_hours"] == [21, 22, 23]
        selection = report.details["chunk_selection"]
        assert selection["selected_chunk_was_advanced"] is False
        assert selection["skipped_covered_chunks"] == []

    def test_max_skip_loop_returns_no_uncovered_chunk_found(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(
            tmp_path,
            start="20230917",
            end="20230930",
            start_hour=0,
            end_hour=23,
        )
        data_root = tmp_path / "data"
        for coin in ["BTC", "ETH"]:
            _write_historical_csv_at_timestamp(
                data_root,
                coin,
                timestamp_ms=TS_20230917_235958Z,
            )
        monkeypatch.setattr(scheduler, "MAX_COVERED_CHUNK_SKIPS", 2)
        captured = _mock_auto_daily_cycle(monkeypatch)

        report = scheduler.cmd_auto_backfill_next(
            chunk_size_hours=3,
            coins=["BTC", "ETH"],
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            daily_summary_dir=tmp_path / "daily",
            data_root=data_root,
        )

        assert captured == []
        assert report.details["no_uncovered_chunk_found"] is True
        selection = report.details["chunk_selection"]
        assert len(selection["skipped_covered_chunks"]) == 2
        assert selection["selected_chunk_was_advanced"] is True

    def test_autopilot_tick_reports_skipped_covered_chunks(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(
            tmp_path,
            start="20230917",
            end="20230930",
            start_hour=0,
            end_hour=23,
        )
        _enable_autopilot(state_path)
        _mark_hours_completed(state_path, "20230917", list(range(0, 21)))
        data_root = tmp_path / "data"
        for coin in ["BTC", "ETH", "SOL"]:
            _write_historical_csv_at_timestamp(
                data_root,
                coin,
                timestamp_ms=TS_20230917_235958Z,
            )
        _mock_auto_daily_cycle(monkeypatch, advance_csv=True)

        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            coins=["BTC", "ETH"],
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            data_root=data_root,
            today_override="20230917",
        )

        assert report.details["skipped_covered_chunks"]
        assert report.details["selected_chunk_was_advanced"] is True
        assert report.historical_date == "20230918"

    def test_autopilot_tick_no_uncovered_chunk_found_action(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(
            tmp_path,
            start="20230917",
            end="20230930",
            start_hour=0,
            end_hour=23,
        )
        _enable_autopilot(state_path)
        data_root = tmp_path / "data"
        for coin in ["BTC", "ETH", "SOL"]:
            _write_historical_csv_at_timestamp(
                data_root,
                coin,
                timestamp_ms=TS_20230917_235958Z,
            )
        monkeypatch.setattr(scheduler, "MAX_COVERED_CHUNK_SKIPS", 1)

        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            coins=["BTC", "ETH"],
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            data_root=data_root,
            today_override="20230917",
        )

        assert report.action == "no_uncovered_chunk_found"
        assert scheduler.NO_UNCOVERED_CHUNK_NEXT_ACTION in report.next_recommended_action
        assert report.details["skipped_covered_chunks"]

    def test_autopilot_tick_cli_path_advances_covered_chunk(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(
            tmp_path,
            start="20230917",
            end="20230930",
            start_hour=0,
            end_hour=23,
        )
        _enable_autopilot(state_path)
        _mark_hours_completed(state_path, "20230917", list(range(0, 21)))
        data_root = tmp_path / "data"
        for coin in ["BTC", "ETH"]:
            _write_historical_csv_at_timestamp(
                data_root,
                coin,
                timestamp_ms=TS_20230917_235958Z,
            )
        captured = _mock_auto_daily_cycle(monkeypatch, advance_csv=True)

        exit_code = scheduler.main(
            [
                "autopilot-tick",
                "--coins",
                "BTC",
                "ETH",
                "--chunk-size-hours",
                "3",
                "--state-path",
                str(state_path),
                "--data-root",
                str(data_root),
                "--runs-dir",
                str(tmp_path / "runs"),
                "--daily-summary-dir",
                str(tmp_path / "daily"),
            ]
        )

        assert exit_code == 0
        assert captured
        assert captured[0]["date"] == "20230918"
        assert captured[0]["historical_hours"] == [0, 1, 2]
        report_path = scheduler.load_state(state_path).autopilot.last_autopilot_report_path
        assert report_path
        report_payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
        assert report_payload["details"]["selected_chunk_was_advanced"] is True
        assert report_payload["details"]["selected_date"] == "20230918"
        assert report_payload["details"]["selected_hours"] == ["0", "1", "2"]
        assert report_payload["details"]["coverage_check_enabled"] is True
        assert report_payload["details"]["coverage_source"] == "historical"

    def test_autopilot_tick_does_not_advance_when_eth_lags(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(
            tmp_path,
            start="20230917",
            end="20230930",
            start_hour=0,
            end_hour=23,
        )
        _enable_autopilot(state_path)
        _mark_hours_completed(state_path, "20230917", list(range(0, 21)))
        data_root = tmp_path / "data"
        _write_historical_csv_at_timestamp(
            data_root,
            "BTC",
            timestamp_ms=TS_20230917_235958Z,
        )
        _write_historical_csv_at_timestamp(
            data_root,
            "ETH",
            timestamp_ms=TS_20230917_223000Z,
        )
        captured = _mock_auto_daily_cycle(monkeypatch)

        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            coins=["BTC", "ETH"],
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            data_root=data_root,
            today_override="20230917",
        )

        assert captured
        assert captured[0]["date"] == "20230917"
        assert captured[0]["historical_hours"] == [21, 22, 23]
        assert report.details["selected_chunk_was_advanced"] is False
        assert report.details["selected_date"] == "20230917"
        assert report.details["selected_hours"] == ["21", "22", "23"]
        assert report.details["skipped_covered_chunks"] == []
        eth_hour_results = [
            entry
            for entry in report.details["coverage_hour_results"]
            if entry["coin"] == "ETH"
        ]
        assert any(entry["hour"] == "22" and entry["covered"] is False for entry in eth_hour_results)
        assert any(entry["hour"] == "23" and entry["covered"] is False for entry in eth_hour_results)


class TestPhaseBackfillEstimate:
    def test_btc_eth_14_days_at_144_files_per_day_with_chunk_limit_24(self) -> None:
        estimate = scheduler.compute_phase_backfill_estimate(
            target_days=14,
            coins=["BTC", "ETH"],
            daily_file_limit=144,
            daily_chunk_limit=24,
        )
        assert estimate["files_required"] == 672
        assert estimate["effective_daily_file_capacity"] == 144
        assert estimate["archive_days_per_real_day"] == pytest.approx(3.0)
        assert estimate["estimated_real_days_to_complete"] == pytest.approx(4.6666666667)
        assert estimate["warnings"] == []

    def test_btc_eth_14_days_at_144_files_per_day_with_chunk_limit_12(self) -> None:
        estimate = scheduler.compute_phase_backfill_estimate(
            target_days=14,
            coins=["BTC", "ETH"],
            daily_file_limit=144,
            daily_chunk_limit=12,
        )
        assert estimate["files_required"] == 672
        assert estimate["effective_daily_file_capacity"] == 72
        assert estimate["archive_days_per_real_day"] == pytest.approx(1.5)
        assert estimate["estimated_real_days_to_complete"] == pytest.approx(9.3333333333)
        assert scheduler.PHASE_CHUNK_LIMIT_BOTTLENECK_WARNING in estimate["warnings"]

    def test_dry_run_command_does_not_change_state(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        before = json.loads(state_path.read_text(encoding="utf-8"))
        estimate = scheduler.cmd_phase_backfill_estimate(
            coins=["BTC", "ETH"],
            target_days=14,
            daily_file_limit=144,
            daily_chunk_limit=24,
        )
        after = json.loads(state_path.read_text(encoding="utf-8"))
        assert before == after
        assert estimate["files_required"] == 672
        assert estimate["effective_daily_file_capacity"] == 144


class TestConfigurePhaseBackfill:
    CONFIRMATION = scheduler.PHASE_BACKFILL_CONFIRMATION_TEXT

    def test_enable_requires_exact_confirmation_text(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        with pytest.raises(ValueError, match="require-confirmation-text"):
            scheduler.cmd_configure_phase_backfill(
                enable=True,
                state_path=state_path,
            )
        with pytest.raises(ValueError, match="require-confirmation-text"):
            scheduler.cmd_configure_phase_backfill(
                enable=True,
                require_confirmation_text="wrong text",
                state_path=state_path,
            )

    def test_enable_stores_phase_config_and_focused_coins(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        result = scheduler.cmd_configure_phase_backfill(
            enable=True,
            phase_name="phase3_btc_eth_recent",
            coins=["BTC", "ETH"],
            target_days=14,
            require_confirmation_text=self.CONFIRMATION,
            state_path=state_path,
        )
        state = scheduler.load_state(state_path)
        assert state.autopilot.enabled is True
        assert state.autopilot.phase_backfill_enabled is True
        assert state.autopilot.phase_backfill_name == "phase3_btc_eth_recent"
        assert state.autopilot.focused_coins == ["BTC", "ETH"]
        assert state.autopilot.target_days == 14
        assert result["phase_backfill_enabled"] is True
        assert result["focused_coins"] == ["BTC", "ETH"]

    def test_enable_stores_aws_profile_and_requester_pays(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        result = scheduler.cmd_configure_phase_backfill(
            enable=True,
            require_confirmation_text=self.CONFIRMATION,
            state_path=state_path,
        )
        state = scheduler.load_state(state_path)
        assert state.autopilot.aws_profile == "hyperliquid-research"
        assert state.autopilot.requester_pays is True
        assert result["aws_profile"] == "hyperliquid-research"
        assert result["requester_pays"] is True

    def test_no_requester_pays_allowed_with_warning(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        result = scheduler.cmd_configure_phase_backfill(
            enable=True,
            requester_pays=False,
            require_confirmation_text=self.CONFIRMATION,
            state_path=state_path,
        )
        state = scheduler.load_state(state_path)
        assert state.autopilot.requester_pays is False
        assert scheduler.REQUESTER_PAYS_DISABLED_WARNING in result["warnings"]

    def test_limits_bounded_by_hard_caps(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        scheduler.cmd_configure_phase_backfill(
            enable=True,
            daily_file_limit=999,
            daily_chunk_limit=24,
            max_daily_file_limit=999,
            max_daily_chunk_limit=24,
            require_confirmation_text=self.CONFIRMATION,
            state_path=state_path,
        )
        state = scheduler.load_state(state_path)
        assert state.autopilot.daily_file_limit == scheduler.PHASE_BACKFILL_ABSOLUTE_MAX_DAILY_FILE_LIMIT
        assert state.autopilot.max_daily_file_limit == scheduler.PHASE_BACKFILL_ABSOLUTE_MAX_DAILY_FILE_LIMIT
        assert state.autopilot.daily_chunk_limit == 24
        assert state.autopilot.max_daily_chunk_limit == 24

    def test_accepts_daily_chunk_limit_24(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        result = scheduler.cmd_configure_phase_backfill(
            enable=True,
            daily_chunk_limit=24,
            max_daily_chunk_limit=24,
            require_confirmation_text=self.CONFIRMATION,
            state_path=state_path,
        )
        assert result["daily_chunk_limit"] == 24
        assert result["max_daily_chunk_limit"] == 24

    def test_rejects_daily_chunk_limit_25(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        with pytest.raises(ValueError, match="absolute max"):
            scheduler.cmd_configure_phase_backfill(
                enable=True,
                daily_chunk_limit=25,
                require_confirmation_text=self.CONFIRMATION,
                state_path=state_path,
            )

    def test_disable_restores_conservative_limits(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        scheduler.cmd_configure_phase_backfill(
            enable=True,
            require_confirmation_text=self.CONFIRMATION,
            state_path=state_path,
        )
        scheduler.cmd_configure_phase_backfill(
            disable=True,
            state_path=state_path,
        )
        state = scheduler.load_state(state_path)
        assert state.autopilot.phase_backfill_enabled is False
        assert state.autopilot.daily_chunk_limit == scheduler.CONSERVATIVE_DAILY_CHUNK_LIMIT
        assert state.autopilot.daily_file_limit == scheduler.CONSERVATIVE_DAILY_FILE_LIMIT
        assert state.autopilot.max_daily_chunk_limit == scheduler.CONSERVATIVE_MAX_DAILY_CHUNK_LIMIT
        assert state.autopilot.max_daily_file_limit == scheduler.CONSERVATIVE_MAX_DAILY_FILE_LIMIT

    def test_status_includes_phase_fields_and_eta(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        scheduler.cmd_configure_phase_backfill(
            enable=True,
            coins=["BTC", "ETH"],
            target_days=14,
            require_confirmation_text=self.CONFIRMATION,
            state_path=state_path,
        )
        status = scheduler.cmd_autopilot_status(state_path=state_path)
        assert status["phase_backfill_enabled"] is True
        assert status["phase_backfill_name"] == "phase3_btc_eth_recent"
        assert status["focused_coins"] == ["BTC", "ETH"]
        assert status["target_days"] == 14
        assert status["estimated_files_required"] == 672
        assert status["estimated_real_days_to_complete"] == pytest.approx(4.6666666667)
        assert status["effective_daily_file_capacity"] == 144
        assert status["daily_chunk_limit"] == 24
        assert status["daily_file_limit"] == 144
        assert status["requester_pays_warning"] == scheduler.PHASE_REQUESTER_PAYS_WARNING
        assert status["aws_profile"] == "hyperliquid-research"
        assert status["requester_pays"] is True
        assert status["warnings"] == []

    def test_status_does_not_expose_credentials(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        scheduler.cmd_configure_phase_backfill(
            enable=True,
            require_confirmation_text=self.CONFIRMATION,
            state_path=state_path,
        )
        status = scheduler.cmd_autopilot_status(state_path=state_path)
        combined = json.dumps(status).lower()
        assert "secret_access_key" not in combined
        assert "akia" not in combined
        assert status["aws_profile"] == "hyperliquid-research"

    def test_status_warns_when_chunk_limit_bottlenecks_file_limit(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        scheduler.cmd_configure_phase_backfill(
            enable=True,
            coins=["BTC", "ETH"],
            target_days=14,
            daily_chunk_limit=12,
            max_daily_chunk_limit=12,
            require_confirmation_text=self.CONFIRMATION,
            state_path=state_path,
        )
        status = scheduler.cmd_autopilot_status(state_path=state_path)
        assert status["effective_daily_file_capacity"] == 72
        assert status["estimated_real_days_to_complete"] == pytest.approx(9.3333333333)
        assert scheduler.PHASE_CHUNK_LIMIT_BOTTLENECK_WARNING in status["warnings"]

    def test_reset_usage_clears_counters(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        _enable_autopilot(state_path)
        state = scheduler.load_state(state_path)
        state.autopilot.runs_today = 3
        state.autopilot.chunks_completed_today = 2
        state.autopilot.files_downloaded_today = 12
        scheduler.save_state(state, state_path)

        scheduler.cmd_configure_phase_backfill(
            enable=True,
            reset_usage=True,
            require_confirmation_text=self.CONFIRMATION,
            state_path=state_path,
        )
        state = scheduler.load_state(state_path)
        assert state.autopilot.runs_today == 0
        assert state.autopilot.chunks_completed_today == 0
        assert state.autopilot.files_downloaded_today == 0

    def test_autopilot_tick_uses_focused_coins_when_phase_enabled(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=11)
        scheduler.cmd_configure_phase_backfill(
            enable=True,
            coins=["BTC", "ETH"],
            require_confirmation_text=self.CONFIRMATION,
            state_path=state_path,
        )
        captured: list[list[str]] = []

        def fake_auto_backfill(**kwargs: object) -> scheduler.SchedulerReport:
            coins_arg = kwargs.get("coins")
            if coins_arg:
                captured.append(list(coins_arg))
            return scheduler.SchedulerReport(
                workflows_run=["auto-backfill-next"],
                historical_date="20230916",
                historical_hours=["9", "10", "11"],
                details={
                    "historical_backfill": {
                        "details": {
                            "orchestrator_batch": {
                                "download": {"file_results": [{"status": "downloaded"}]},
                            }
                        }
                    }
                },
            )

        monkeypatch.setattr(scheduler, "cmd_auto_backfill_next", fake_auto_backfill)

        scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )
        assert captured == [["BTC", "ETH"]]

    def test_autopilot_tick_passes_aws_settings_to_backfill(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=15, end_hour=17)
        scheduler.cmd_configure_phase_backfill(
            enable=True,
            coins=["BTC", "ETH"],
            require_confirmation_text=self.CONFIRMATION,
            state_path=state_path,
        )
        captured: list[dict[str, object]] = []

        def fake_auto_backfill(**kwargs: object) -> scheduler.SchedulerReport:
            captured.append(dict(kwargs))
            return scheduler.SchedulerReport(
                workflows_run=["auto-backfill-next"],
                historical_date="20230917",
                historical_hours=["15", "16", "17"],
                aws_profile=str(kwargs.get("aws_profile")),
                requester_pays=bool(kwargs.get("requester_pays")),
                details={
                    "import_progress": {
                        "any_import_progress": True,
                        "files_downloaded_new": 6,
                        "files_skipped_existing": 0,
                        "files_failed": 0,
                        "import_failures": [],
                        "download_failures": [],
                    },
                    "historical_backfill": {
                        "details": {
                            "orchestrator_batch": {
                                "download": {
                                    "file_results": [
                                        {"download_status": "downloaded"},
                                    ],
                                },
                                "import": [
                                    {"coin": "BTC", "import_status": "imported", "rows_written": 1},
                                ],
                            }
                        }
                    },
                },
            )

        monkeypatch.setattr(scheduler, "cmd_auto_backfill_next", fake_auto_backfill)

        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230917",
        )

        assert captured
        assert captured[0]["aws_profile"] == "hyperliquid-research"
        assert captured[0]["requester_pays"] is True
        assert report.aws_profile == "hyperliquid-research"
        assert report.requester_pays is True

    def test_requester_pays_auth_required_is_download_failed_without_quota(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=15, end_hour=17)
        scheduler.cmd_configure_phase_backfill(
            enable=True,
            coins=["BTC", "ETH"],
            require_confirmation_text=self.CONFIRMATION,
            state_path=state_path,
        )
        state = scheduler.load_state(state_path)
        state.autopilot.chunks_completed_today = 0
        state.autopilot.files_downloaded_today = 0
        scheduler.save_state(state, state_path)

        error_type = "requester_pays_auth_required"
        s3_path = "s3://hyperliquid-archive/market_data/20230917/15/l2Book/BTC.lz4"

        def fake_auto_backfill(**kwargs: object) -> scheduler.SchedulerReport:
            download_failure = (
                f"BTC: download failed ({error_type}): {s3_path}"
            )
            return scheduler.SchedulerReport(
                workflows_run=["auto-backfill-next", "daily-cycle"],
                historical_date="20230917",
                historical_hours=["15", "16", "17"],
                aws_profile="hyperliquid-research",
                requester_pays=True,
                download_error_type=error_type,
                files_failed=1,
                failures=[download_failure],
                details={
                    "import_progress": {
                        "any_import_progress": False,
                        "files_downloaded_new": 0,
                        "files_skipped_existing": 0,
                        "files_failed": 1,
                        "download_error_type": error_type,
                        "download_failures": [download_failure],
                        "import_failures": [],
                    },
                    "historical_backfill": {
                        "details": {
                            "orchestrator_batch": {
                                "download": {
                                    "file_results": [
                                        {
                                            "coin": "BTC",
                                            "download_status": "failed",
                                            "download_error_type": error_type,
                                            "s3_path": s3_path,
                                        }
                                    ],
                                }
                            }
                        }
                    },
                },
            )

        monkeypatch.setattr(scheduler, "cmd_auto_backfill_next", fake_auto_backfill)

        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230917",
        )

        assert report.action == "download_failed"
        assert report.download_error_type == error_type
        assert report.files_failed == 1
        assert report.files_downloaded_new == 0
        assert "aws profile" in (report.next_recommended_action or "").lower()
        state = scheduler.load_state(state_path)
        assert state.autopilot.chunks_completed_today == 0
        assert state.autopilot.files_downloaded_today == 0

    def test_phase_mode_tick_preserves_daily_limits_when_ramp_would_apply(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path, start_hour=9, end_hour=11)
        scheduler.cmd_configure_phase_backfill(
            enable=True,
            coins=["BTC", "ETH"],
            target_days=14,
            require_confirmation_text=self.CONFIRMATION,
            state_path=state_path,
        )
        state = scheduler.load_state(state_path)
        state.autopilot.successful_days_in_row = 2
        state.autopilot.ramp_every_successful_days = 2
        scheduler.save_state(state, state_path)
        _mock_auto_backfill_next(monkeypatch, downloaded_files=6)

        report = scheduler.cmd_autopilot_tick(
            chunk_size_hours=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            today_override="20230916",
        )

        state = scheduler.load_state(state_path)
        assert state.autopilot.daily_file_limit == 144
        assert state.autopilot.daily_chunk_limit == 24
        assert report.daily_file_limit == 144
        assert report.daily_chunk_limit == 24
        assert report.effective_daily_file_capacity == 144
        assert report.focused_coins == ["BTC", "ETH"]
        assert report.phase_backfill_enabled is True
        assert report.phase_backfill_name == "phase3_btc_eth_recent"
        assert report.target_days == 14
        assert report.ramp_skipped_for_phase_backfill is True
        assert not any("Ramp applied" in warning for warning in report.warnings)

        status = scheduler.cmd_autopilot_status(
            state_path=state_path,
            runs_dir=tmp_path / "runs",
        )
        assert status["daily_file_limit"] == 144
        assert status["daily_chunk_limit"] == 24
        assert status["effective_daily_file_capacity"] == 144
        assert status["focused_coins"] == ["BTC", "ETH"]
        assert status["estimated_real_days_to_complete"] == pytest.approx(4.6666666667)

        saved = json.loads(Path(state.autopilot.last_autopilot_report_path).read_text(encoding="utf-8"))
        assert saved["focused_coins"] == ["BTC", "ETH"]
        assert saved["ramp_skipped_for_phase_backfill"] is True
        assert saved["daily_file_limit"] == 144
        assert saved["daily_chunk_limit"] == 24
        assert saved["effective_daily_file_capacity"] == 144

    def test_safety_flags_remain_paper_only(self, tmp_path: Path) -> None:
        state_path = _init_state(tmp_path)
        scheduler.cmd_configure_phase_backfill(
            enable=True,
            require_confirmation_text=self.CONFIRMATION,
            state_path=state_path,
        )
        report = scheduler.SchedulerReport(workflows_run=["configure-phase-backfill"])
        payload = report.to_dict()
        assert payload["paper_only"] is True
        assert payload["execution_enabled"] is False


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


def _write_historical_csv(
    data_root: Path,
    coin: str,
    *,
    date: str,
    hour: int,
) -> Path:
    from bid_ask_schema import (
        ASK_SIZE,
        BEST_ASK,
        BEST_BID,
        BID_ASK_CSV_COLUMNS,
        BID_SIZE,
        TIMESTAMP_MS,
    )

    csv_dir = data_root / "historical_1s"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{coin}_1s.csv"
    base_ms = int(
        __import__("datetime")
        .datetime.strptime(date, "%Y%m%d")
        .replace(hour=hour, tzinfo=__import__("datetime").timezone.utc)
        .timestamp()
        * 1000
    )
    row = {
        TIMESTAMP_MS: str(base_ms + 1000),
        BEST_BID: "100.0",
        BEST_ASK: "100.1",
        BID_SIZE: "1.0",
        ASK_SIZE: "1.0",
        "coin": coin,
        "source_session_id": "historical_archive",
        "source_channel": "historical_l2Book",
    }
    import csv

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BID_ASK_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)
    return csv_path


def _write_resampled_csv(
    data_root: Path,
    coin: str,
    interval: str,
    *,
    date: str,
    hour: int,
) -> Path:
    from bid_ask_resampler import RESAMPLED_COLUMNS

    csv_dir = data_root / f"historical_{interval}"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{coin}_{interval}.csv"
    base_ms = int(
        __import__("datetime")
        .datetime.strptime(date, "%Y%m%d")
        .replace(hour=hour, tzinfo=__import__("datetime").timezone.utc)
        .timestamp()
        * 1000
    )
    interval_ms = 15_000 if interval == "15s" else 60_000
    row = {
        "start_timestamp_ms": str(base_ms),
        "end_timestamp_ms": str(base_ms + interval_ms - 1),
        "timestamp_iso": "2023-09-16T12:00:01Z",
        "coin": coin,
        "open_mid": "100.05",
        "high_mid": "100.05",
        "low_mid": "100.05",
        "close_mid": "100.05",
        "avg_bid": "100.0",
        "avg_ask": "100.1",
        "avg_mid": "100.05",
        "avg_spread": "0.1",
        "avg_spread_bps": "10.0",
        "avg_bbo_imbalance": "0.0",
        "row_count": "15",
        "missing_seconds": "0",
    }
    import csv

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESAMPLED_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)
    return csv_path


def _write_autopilot_report(
    runs_dir: Path,
    *,
    action: str = "chunk_completed",
    files_downloaded_today: int = 3,
    chunks_completed_today: int = 1,
    coins: list[str] | None = None,
) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    report = scheduler.AutopilotReport(
        action=action,
        coins=coins or ["BTC", "ETH"],
        files_downloaded_today=files_downloaded_today,
        chunks_completed_today=chunks_completed_today,
        autopilot_snapshot={
            "failed_runs_in_row": 0,
            "paused": False,
            "files_downloaded_today": files_downloaded_today,
            "chunks_completed_today": chunks_completed_today,
        },
    )
    json_path, _ = scheduler.write_autopilot_report(report, runs_dir=runs_dir, label="autopilot")
    return json_path


def _write_candidate_report(
    reports_dir: Path,
    *,
    run_timestamp: str,
    filtered_paper_trades: int = 25,
    cumulative_net_return_bps: float = 12.5,
) -> Path:
    from bid_ask_candidate_validator import (
        build_candidate_config,
        build_candidate_validation_report,
        write_candidate_validation_json,
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    report = build_candidate_validation_report(
        {
            "filtered_paper_trades": filtered_paper_trades,
            "cumulative_net_return_bps": cumulative_net_return_bps,
            "beats_baseline_accuracy": True,
            "positive_after_costs": True,
            "stability_summary": {
                "concentration_warning": False,
                "sparse_stability_warning": False,
            },
        },
        config=build_candidate_config(),
        run_timestamp=run_timestamp,
    )
    return write_candidate_validation_json(report, reports_dir=reports_dir)


class TestPhaseBackfillAudit:
    def test_detects_stale_historical_1s(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        for coin in ["BTC", "ETH"]:
            _write_historical_csv(data_root, coin, date="20230916", hour=12)

        audit = scheduler.cmd_phase_backfill_audit(
            coins=["BTC", "ETH"],
            data_root=data_root,
            downloads_root=tmp_path / "downloads",
            reports_dir=tmp_path / "runs",
            candidate_reports_dir=tmp_path / "candidate",
            state_path=tmp_path / "missing-state.json",
        )

        assert audit["imported_historical_1s"]["BTC"]["freshness_status"] == "stale"
        assert audit["imported_historical_1s"]["ETH"]["freshness_status"] == "stale"
        assert audit["next_recommended_action"] == "check autopilot/download reports and AWS access"

    def test_detects_advanced_historical_1s(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        for coin in ["BTC", "ETH"]:
            _write_historical_csv(data_root, coin, date="20230917", hour=10)

        audit = scheduler.cmd_phase_backfill_audit(
            coins=["BTC", "ETH"],
            data_root=data_root,
            downloads_root=tmp_path / "downloads",
            reports_dir=tmp_path / "runs",
            candidate_reports_dir=tmp_path / "candidate",
            state_path=tmp_path / "missing-state.json",
        )

        assert audit["imported_historical_1s"]["BTC"]["freshness_status"] == "advancing"
        assert audit["imported_historical_1s"]["ETH"]["freshness_status"] == "advancing"

    def test_detects_resample_stale_while_1s_advanced(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        for coin in ["BTC", "ETH"]:
            _write_historical_csv(data_root, coin, date="20230917", hour=10)
            _write_resampled_csv(data_root, coin, "15s", date="20230916", hour=12)
            _write_resampled_csv(data_root, coin, "1m", date="20230916", hour=12)

        audit = scheduler.cmd_phase_backfill_audit(
            coins=["BTC", "ETH"],
            data_root=data_root,
            downloads_root=tmp_path / "downloads",
            reports_dir=tmp_path / "runs",
            candidate_reports_dir=tmp_path / "candidate",
            state_path=tmp_path / "missing-state.json",
        )

        assert audit["imported_historical_1s"]["BTC"]["freshness_status"] == "advancing"
        assert audit["next_recommended_action"] == "run historical resample"

    def test_detects_repeated_unchanged_candidate_reports(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        candidate_dir = tmp_path / "candidate"
        for coin in ["BTC", "ETH"]:
            _write_historical_csv(data_root, coin, date="20230917", hour=10)
            _write_resampled_csv(data_root, coin, "15s", date="20230917", hour=10)
            _write_resampled_csv(data_root, coin, "1m", date="20230917", hour=10)

        for index in range(3):
            _write_candidate_report(
                candidate_dir,
                run_timestamp=f"2026-07-0{index + 1}T08:00:00Z",
                filtered_paper_trades=25,
                cumulative_net_return_bps=12.5,
            )

        audit = scheduler.cmd_phase_backfill_audit(
            coins=["BTC", "ETH"],
            data_root=data_root,
            downloads_root=tmp_path / "downloads",
            reports_dir=tmp_path / "runs",
            candidate_reports_dir=candidate_dir,
            state_path=tmp_path / "missing-state.json",
        )

        assert audit["candidate_freshness"]["repeated_unchanged_warning"] is True
        assert audit["next_recommended_action"] == "rerun validator/robustness"
        assert any("unchanged" in warning.lower() for warning in audit.get("warnings", []))

    def test_recommends_import_issue_when_downloads_exist_but_csv_stale(
        self,
        tmp_path: Path,
    ) -> None:
        data_root = tmp_path / "data"
        downloads_root = tmp_path / "downloads"
        for coin in ["BTC", "ETH"]:
            _write_historical_csv(data_root, coin, date="20230916", hour=12)
            download_dir = downloads_root / "20230917" / "10"
            download_dir.mkdir(parents=True, exist_ok=True)
            (download_dir / f"{coin}.lz4").write_bytes(b"fake")

        audit = scheduler.cmd_phase_backfill_audit(
            coins=["BTC", "ETH"],
            data_root=data_root,
            downloads_root=downloads_root,
            reports_dir=tmp_path / "runs",
            candidate_reports_dir=tmp_path / "candidate",
            state_path=tmp_path / "missing-state.json",
        )

        assert audit["downloads"]["total_files"] == 2
        assert audit["imported_historical_1s"]["BTC"]["freshness_status"] == "stale"
        assert audit["next_recommended_action"] == "import step likely failing or not appending"

    def test_writes_json_and_markdown(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        json_path = tmp_path / "audit.json"
        markdown_path = tmp_path / "audit.md"

        audit = scheduler.cmd_phase_backfill_audit(
            coins=["BTC"],
            data_root=data_root,
            downloads_root=tmp_path / "downloads",
            reports_dir=tmp_path / "runs",
            candidate_reports_dir=tmp_path / "candidate",
            state_path=tmp_path / "missing-state.json",
            json_path=json_path,
            markdown_path=markdown_path,
        )

        saved = json.loads(json_path.read_text(encoding="utf-8"))
        markdown = markdown_path.read_text(encoding="utf-8")
        assert saved["command"] == "phase-backfill-audit"
        assert saved["paper_only"] is True
        assert saved["execution_enabled"] is False
        assert saved["real_execution_ready"] is False
        assert "Phase Backfill Audit" in markdown
        assert audit["paper_only"] is True

    def test_report_includes_safety_fields(self, tmp_path: Path) -> None:
        audit = scheduler.cmd_phase_backfill_audit(
            coins=["BTC", "ETH"],
            data_root=tmp_path / "data",
            downloads_root=tmp_path / "downloads",
            reports_dir=tmp_path / "runs",
            candidate_reports_dir=tmp_path / "candidate",
            state_path=tmp_path / "missing-state.json",
        )

        assert audit["paper_only"] is True
        assert audit["execution_enabled"] is False
        assert audit["real_execution_ready"] is False

    def test_reads_scheduler_progress_from_autopilot_report(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        report_path = _write_autopilot_report(
            runs_dir,
            action="chunk_completed",
            files_downloaded_today=6,
            chunks_completed_today=2,
        )

        audit = scheduler.cmd_phase_backfill_audit(
            coins=["BTC", "ETH"],
            data_root=tmp_path / "data",
            downloads_root=tmp_path / "downloads",
            reports_dir=runs_dir,
            candidate_reports_dir=tmp_path / "candidate",
            state_path=tmp_path / "missing-state.json",
        )

        progress = audit["scheduler_progress"]
        assert progress["latest_autopilot_report_path"] == str(report_path)
        assert progress["latest_action"] == "chunk_completed"
        assert progress["files_downloaded_today"] == 6
        assert progress["chunks_completed_today"] == 2

    def test_no_network_subprocess_or_aws_calls(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _blocked(*args: object, **kwargs: object) -> None:
            raise AssertionError("network/subprocess call attempted during audit")

        monkeypatch.setattr("urllib.request.urlopen", _blocked)
        monkeypatch.setattr("subprocess.run", _blocked)
        monkeypatch.setattr("subprocess.Popen", _blocked)
        monkeypatch.setattr("boto3.client", _blocked)

        scheduler.cmd_phase_backfill_audit(
            coins=["BTC"],
            data_root=tmp_path / "data",
            downloads_root=tmp_path / "downloads",
            reports_dir=tmp_path / "runs",
            candidate_reports_dir=tmp_path / "candidate",
            state_path=tmp_path / "missing-state.json",
        )


def _block_network_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network/subprocess call attempted")

    monkeypatch.setattr("urllib.request.urlopen", _blocked)
    monkeypatch.setattr("subprocess.run", _blocked)
    monkeypatch.setattr("subprocess.Popen", _blocked)
    monkeypatch.setattr("boto3.client", _blocked)


class TestResearchSession:
    def test_dry_run_produces_plan_without_actions(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)
        sessions_dir = tmp_path / "sessions"
        audit_calls: list[str] = []
        autopilot_calls: list[str] = []

        monkeypatch.setattr(
            scheduler,
            "cmd_phase_backfill_audit",
            lambda **kwargs: audit_calls.append("audit") or {"imported_historical_1s": {}},
        )
        monkeypatch.setattr(
            scheduler,
            "cmd_autopilot_tick",
            lambda **kwargs: autopilot_calls.append("tick") or scheduler.AutopilotReport(),
        )
        _block_network_subprocess(monkeypatch)

        session = scheduler.cmd_research_session(
            duration_hours=8,
            tick_interval_minutes=60,
            dry_run=True,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            sessions_dir=sessions_dir,
            data_root=tmp_path / "data",
            downloads_root=tmp_path / "downloads",
            candidate_reports_dir=tmp_path / "candidate",
            scheduler_state_dir=tmp_path / "scheduler_state",
        )

        assert session["dry_run"] is True
        assert session["ticks_planned"] == 8
        assert session["ticks_completed"] == 0
        assert audit_calls == []
        assert autopilot_calls == []
        assert session["planned_tick_actions"]
        report_json = sessions_dir / f"research_session_{session['session_id']}.json"
        assert report_json.is_file()

    def test_max_ticks_limits_loop(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)
        _enable_autopilot(state_path)
        _mock_auto_backfill_next(monkeypatch)
        tick_calls: list[int] = []

        def fake_execute_tick(**kwargs: object) -> dict[str, object]:
            tick_calls.append(int(kwargs["tick_index"]))
            return {
                "tick_index": kwargs["tick_index"],
                "started_at": "2026-07-07T00:00:00Z",
                "ended_at": "2026-07-07T00:01:00Z",
                "autopilot_action": "chunk_completed",
                "files_downloaded_new": 1,
                "files_skipped_existing": 0,
                "files_failed": 0,
                "rows_added_by_coin": {"BTC": 1},
                "timestamp_advanced_by_coin": {"BTC": True},
                "btc_latest_timestamp": None,
                "eth_latest_timestamp": None,
                "historical_1s_advanced": False,
                "candidate_trades": None,
                "candidate_cumulative_net_return_bps": None,
                "relaxed_trades": None,
                "relaxed_cumulative_net_return_bps": None,
                "resample_validate_ran": False,
                "warnings": [],
                "failures": [],
            }

        monkeypatch.setattr(scheduler, "_execute_research_session_tick", fake_execute_tick)
        _block_network_subprocess(monkeypatch)

        session = scheduler.cmd_research_session(
            duration_hours=8,
            tick_interval_minutes=60,
            max_ticks=2,
            run_candidate_tests=False,
            run_relaxed_candidate_test=False,
            update_dashboard_artifacts=False,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            sessions_dir=tmp_path / "sessions",
            data_root=tmp_path / "data",
            sleep_fn=lambda _seconds: None,
        )

        assert session["ticks_planned"] == 2
        assert session["ticks_completed"] == 2
        assert tick_calls == [0, 1]

    def test_session_report_includes_safety_flags(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)
        sessions_dir = tmp_path / "sessions"

        session = scheduler.cmd_research_session(
            duration_hours=1,
            tick_interval_minutes=60,
            dry_run=True,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            sessions_dir=sessions_dir,
            data_root=tmp_path / "data",
            scheduler_state_dir=tmp_path / "scheduler_state",
        )

        assert session["paper_only"] is True
        assert session["execution_enabled"] is False
        assert session["real_execution_ready"] is False
        saved = json.loads(
            (sessions_dir / f"research_session_{session['session_id']}.json").read_text(
                encoding="utf-8"
            )
        )
        assert saved["paper_only"] is True
        assert saved["execution_enabled"] is False
        assert saved["real_execution_ready"] is False

    def test_session_stops_if_autopilot_paused(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)
        _enable_autopilot(state_path)
        scheduler.cmd_configure_autopilot(pause_reason="testing", state_path=state_path)
        execute_calls: list[int] = []

        def fake_execute_tick(**kwargs: object) -> dict[str, object]:
            execute_calls.append(1)
            return {}

        monkeypatch.setattr(scheduler, "_execute_research_session_tick", fake_execute_tick)
        _block_network_subprocess(monkeypatch)

        session = scheduler.cmd_research_session(
            duration_hours=8,
            tick_interval_minutes=60,
            max_ticks=3,
            run_candidate_tests=False,
            run_relaxed_candidate_test=False,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            sessions_dir=tmp_path / "sessions",
            data_root=tmp_path / "data",
            sleep_fn=lambda _seconds: None,
        )

        assert session["ticks_completed"] == 0
        assert execute_calls == []
        assert session["stop_reason"] is not None
        assert "paused" in session["stop_reason"].lower()

    def test_session_stops_if_failed_runs_in_row_at_threshold(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)
        _enable_autopilot(state_path)
        state = scheduler.load_state(state_path)
        state.autopilot.failed_runs_in_row = scheduler.AUTOPILOT_FAILURE_PAUSE_THRESHOLD
        scheduler.save_state(state, state_path)
        execute_calls: list[int] = []

        def fake_execute_tick(**kwargs: object) -> dict[str, object]:
            execute_calls.append(1)
            return {}

        monkeypatch.setattr(scheduler, "_execute_research_session_tick", fake_execute_tick)
        _block_network_subprocess(monkeypatch)

        session = scheduler.cmd_research_session(
            duration_hours=8,
            tick_interval_minutes=60,
            max_ticks=3,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            sessions_dir=tmp_path / "sessions",
            data_root=tmp_path / "data",
            sleep_fn=lambda _seconds: None,
        )

        assert session["ticks_completed"] == 0
        assert execute_calls == []
        assert "failed_runs_in_row" in (session["stop_reason"] or "")

    def test_session_tick_records_autopilot_action_and_data_advancement(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)
        _enable_autopilot(state_path)
        _mock_auto_backfill_next(monkeypatch, advance_csv=True, downloaded_files=2)
        scheduler_state_dir = tmp_path / "scheduler_state"
        audit_json = scheduler_state_dir / "phase3_audit_latest.json"
        audit_md = scheduler_state_dir / "phase3_audit_latest.md"
        audit_calls: list[dict[str, object]] = []

        def fake_audit(**kwargs: object) -> dict[str, object]:
            audit_calls.append(dict(kwargs))
            ts_ms = 1_000 if len(audit_calls) <= 1 else 2_000
            audit_payload = {
                "imported_historical_1s": {
                    "BTC": {
                        "last_timestamp_ms": ts_ms,
                        "last_timestamp": f"{ts_ms}",
                    },
                    "ETH": {
                        "last_timestamp_ms": ts_ms,
                        "last_timestamp": f"{ts_ms}",
                    },
                }
            }
            json_path = kwargs.get("json_path")
            markdown_path = kwargs.get("markdown_path")
            if isinstance(json_path, Path):
                json_path.parent.mkdir(parents=True, exist_ok=True)
                json_path.write_text(json.dumps(audit_payload), encoding="utf-8")
            if isinstance(markdown_path, Path):
                markdown_path.parent.mkdir(parents=True, exist_ok=True)
                markdown_path.write_text("# audit\n", encoding="utf-8")
            return audit_payload

        monkeypatch.setattr(scheduler, "cmd_phase_backfill_audit", fake_audit)
        monkeypatch.setattr(
            orchestrator,
            "cmd_resample_validate",
            lambda **kwargs: orchestrator.OrchestratorReport(workflows_run=["resample-validate"]),
        )
        monkeypatch.setattr(
            scheduler.candidate_validator,
            "run_candidate_validation",
            lambda **kwargs: {"filtered_paper_trades": 12, "cumulative_net_return_bps": 4.5},
        )
        monkeypatch.setattr(
            scheduler.candidate_validator,
            "summarize_candidate_validation_reports",
            lambda **kwargs: "summary",
        )
        monkeypatch.setattr(
            scheduler,
            "_run_relaxed_candidate_b_validation",
            lambda **kwargs: {"filtered_paper_trades": 25, "cumulative_net_return_bps": 6.2},
        )
        _block_network_subprocess(monkeypatch)

        session = scheduler.cmd_research_session(
            duration_hours=1,
            tick_interval_minutes=60,
            max_ticks=1,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            sessions_dir=tmp_path / "sessions",
            data_root=tmp_path / "data",
            scheduler_state_dir=scheduler_state_dir,
            sleep_fn=lambda _seconds: None,
        )

        tick = session["per_tick_summaries"][0]
        assert tick["autopilot_action"] == "chunk_completed"
        assert tick["files_downloaded_new"] == 2
        assert tick["timestamp_advanced_by_coin"]["BTC"] is True
        assert tick["historical_1s_advanced"] is True
        assert tick["resample_validate_ran"] is True
        assert tick["candidate_trades"] == 12
        assert tick["relaxed_trades"] == 25
        assert audit_calls[0]["json_path"] == audit_json
        assert audit_calls[0]["markdown_path"] == audit_md
        assert audit_json.is_file()
        assert audit_md.is_file()

    def test_dashboard_audit_artifact_paths_are_used(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)
        scheduler_state_dir = tmp_path / "scheduler_state"
        expected_json, expected_md = scheduler._dashboard_phase_audit_paths(scheduler_state_dir)

        session = scheduler.cmd_research_session(
            duration_hours=1,
            tick_interval_minutes=60,
            dry_run=True,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            sessions_dir=tmp_path / "sessions",
            scheduler_state_dir=scheduler_state_dir,
        )

        assert session["dashboard_audit_paths"]["json"] == str(expected_json)
        assert session["dashboard_audit_paths"]["markdown"] == str(expected_md)
        assert expected_json.name == "phase3_audit_latest.json"
        assert expected_md.name == "phase3_audit_latest.md"

    def test_no_real_aws_network_or_subprocess_calls(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state_path = _init_state(tmp_path)

        session = scheduler.cmd_research_session(
            duration_hours=2,
            tick_interval_minutes=60,
            dry_run=True,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            sessions_dir=tmp_path / "sessions",
            data_root=tmp_path / "data",
            scheduler_state_dir=tmp_path / "scheduler_state",
        )
        assert session["dry_run"] is True

        _block_network_subprocess(monkeypatch)
        scheduler.cmd_research_session(
            duration_hours=2,
            tick_interval_minutes=60,
            dry_run=True,
            state_path=state_path,
            runs_dir=tmp_path / "runs",
            sessions_dir=tmp_path / "sessions",
            data_root=tmp_path / "data",
            scheduler_state_dir=tmp_path / "scheduler_state",
        )


class TestResearchSessionLaunchdTemplate:
    def test_writes_two_plists(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "launchd"
        morning_plist, evening_plist, setup_path = (
            scheduler.cmd_write_research_session_launchd_template(
                output_dir=output_dir,
                project_dir=tmp_path,
                python_path="/usr/bin/python3",
                use_caffeinate=False,
            )
        )

        assert morning_plist.is_file()
        assert evening_plist.is_file()
        assert setup_path.is_file()
        assert morning_plist.name == scheduler.LAUNCHD_RESEARCH_SESSION_MORNING_PLIST
        assert evening_plist.name == scheduler.LAUNCHD_RESEARCH_SESSION_EVENING_PLIST

        morning_text = morning_plist.read_text(encoding="utf-8")
        evening_text = evening_plist.read_text(encoding="utf-8")
        setup_text = setup_path.read_text(encoding="utf-8")

        assert "research-session" in morning_text
        assert "research-session" in evening_text
        assert "com.hyperliquid.research.session.morning" in morning_text
        assert "com.hyperliquid.research.session.evening" in evening_text
        assert "launchctl load" in setup_text
        assert "not installed automatically" in setup_text.lower()

    def test_darwin_defaults_to_caffeinate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(scheduler.platform, "system", lambda: "Darwin")
        output_dir = tmp_path / "launchd"
        morning_plist, _, setup_path = scheduler.cmd_write_research_session_launchd_template(
            output_dir=output_dir,
            project_dir=tmp_path,
            python_path="/usr/bin/python3",
        )

        morning_text = morning_plist.read_text(encoding="utf-8")
        setup_text = setup_path.read_text(encoding="utf-8")

        assert scheduler.DEFAULT_CAFFEINATE_PATH in morning_text
        assert "<string>-d</string>" in morning_text
        assert "<string>-i</string>" in morning_text
        assert "<string>-m</string>" in morning_text
        assert "<string>-s</string>" in morning_text
        assert "<string>-u</string>" in morning_text
        assert "caffeinate keeps the Mac awake" in setup_text
        assert "Manual caffeinated session" in setup_text

    def test_no_caffeinate_omits_caffeinate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(scheduler.platform, "system", lambda: "Darwin")
        output_dir = tmp_path / "launchd"
        morning_plist, _, setup_path = scheduler.cmd_write_research_session_launchd_template(
            output_dir=output_dir,
            project_dir=tmp_path,
            python_path="/usr/bin/python3",
            use_caffeinate=False,
        )

        morning_text = morning_plist.read_text(encoding="utf-8")
        setup_text = setup_path.read_text(encoding="utf-8")

        assert scheduler.DEFAULT_CAFFEINATE_PATH not in morning_text
        assert "<string>/usr/bin/python3</string>" in morning_text
        assert "Manual caffeinated session" not in setup_text

    def test_caffeinate_flags_split_into_program_arguments(
        self,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "launchd"
        morning_plist, _, _ = scheduler.cmd_write_research_session_launchd_template(
            output_dir=output_dir,
            project_dir=tmp_path,
            python_path="/usr/bin/python3",
            use_caffeinate=True,
            caffeinate_flags="-dimsu",
        )

        morning_text = morning_plist.read_text(encoding="utf-8")
        caffeinate_idx = morning_text.index(scheduler.DEFAULT_CAFFEINATE_PATH)
        research_idx = morning_text.index("research-session")
        caffeinate_section = morning_text[caffeinate_idx:research_idx]

        assert "<string>-d</string>" in caffeinate_section
        assert "<string>-i</string>" in caffeinate_section
        assert "<string>-m</string>" in caffeinate_section
        assert "<string>-s</string>" in caffeinate_section
        assert "<string>-u</string>" in caffeinate_section
        assert "<string>-dimsu</string>" not in morning_text

    def test_split_caffeinate_flags_helper(self) -> None:
        assert scheduler._split_caffeinate_flags("-dimsu") == ["-d", "-i", "-m", "-s", "-u"]
        assert scheduler._split_caffeinate_flags("-d -i") == ["-d", "-i"]
        assert scheduler._split_caffeinate_flags("--help") == ["--help"]

    def test_no_auto_load_behavior(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "launchd"
        _, _, setup_path = scheduler.cmd_write_research_session_launchd_template(
            output_dir=output_dir,
            project_dir=tmp_path,
            python_path="/usr/bin/python3",
            use_caffeinate=False,
        )

        setup_text = setup_path.read_text(encoding="utf-8").lower()
        assert "not installed automatically" in setup_text
        assert "launchctl load" in setup_text
        assert "launchctl bootstrap" not in setup_text
