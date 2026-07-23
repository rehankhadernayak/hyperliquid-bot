"""Focused offline tests for Candidate F milestone comparison."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from bid_ask_candidate_f_robustness_analyzer import CANDIDATE_F_FEATURE_SET
from bid_ask_candidate_f_milestone_comparator import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_ROBUSTNESS_DIR,
    DEFAULT_WALK_FORWARD_DIR,
    FROZEN_CONFIG,
    FROZEN_CONFIG_ID,
    SAFETY_NOTE,
    build_milestone_comparison_report,
    build_milestone_comparison_table,
    build_run_record,
    config_matches_frozen,
    dataset_group_key,
    discover_robustness_reports,
    discover_walk_forward_reports,
    extract_dataset_summary,
    find_matching_robustness_report,
    main,
    run_milestone_comparison,
    select_newest_valid_runs,
    write_comparison_json,
    write_comparison_markdown,
    _dominant_day_metrics,
)
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY

COMPARATOR_FILE = (
    Path(__file__).resolve().parent.parent / "bid_ask_candidate_f_milestone_comparator.py"
)


def _walk_forward_payload(
    *,
    stamp_suffix: str = "20260723T092114",
    accepted_trades_count: int = 17,
    completed_windows: int = 646,
    skipped_windows: int = 9,
    feature_set: str = CANDIDATE_F_FEATURE_SET,
    dataset_summary: dict | None = None,
    data_dir: str = "data/bid_ask_csv/historical_15s",
    source: str = "historical",
    accepted_trades_csv_path: str | None = None,
) -> dict:
    csv_path = (
        accepted_trades_csv_path
        or f"reports/bid_ask_walk_forward/accepted_trades_ETH_{stamp_suffix}.csv"
    )
    payload = {
        "generated_at": "2026-07-23T09:21:14Z",
        "coins": ["ETH"],
        "trade_classes": ["UP"],
        "feature_set": feature_set,
        "label_mode": "binary_tradeable_up",
        "horizon": 20,
        "interval": "15s",
        "min_confidence": 0.65,
        "min_edge_bps": 2.0,
        "cost_multiplier": 1.25,
        "max_spread_bps": 8.0,
        "accepted_trades_count": accepted_trades_count,
        "average_net_return_bps": 1.6911592413866594,
        "cumulative_net_return_bps": 28.74970710357321,
        "completed_windows": completed_windows,
        "skipped_windows": skipped_windows,
        "data_dir": data_dir,
        "data_root": "data/bid_ask_csv",
        "source": source,
        "paper_only": True,
        "execution_enabled": False,
        "accepted_trades_csv_path": csv_path,
        "class_trade_diagnostics": {
            "UP": {
                "win_rate_after_costs": 0.7647058823529411,
                "average_net_return_bps": 1.6911592413866594,
                "cumulative_net_return_bps": 28.74970710357321,
                "trades": accepted_trades_count,
            }
        },
    }
    if dataset_summary is not None:
        payload["dataset_summary"] = dataset_summary
    return payload


def _robustness_payload(
    *,
    stamp_suffix: str = "20260723T092119",
    accepted_trades_csv_path: str,
    verdict: str = "diagnostic_positive_sparse",
) -> dict:
    trade_analysis = {
        "trade_count": 17,
        "average_net_bps": 1.6911592413866594,
        "cumulative_net_bps": 28.74970710357321,
        "win_rate": 0.7647058823529411,
        "dominant_utc_hour": "14",
        "dominant_utc_hour_trade_fraction": 0.9411764705882353,
        "dominant_volatility_regime": "low_volatility_regime",
        "dominant_volatility_regime_trade_fraction": 0.6470588235294118,
        "dominant_spread_regime": "tight_spread_regime",
        "dominant_spread_regime_trade_fraction": 1.0,
        "by_day": {
            "2023-09-21": {
                "count": 1,
                "sum_net_bps": -2.683884428147218,
                "avg_net_bps": -2.683884428147218,
                "cumulative_net_bps": -2.683884428147218,
            },
            "2023-09-23": {
                "count": 16,
                "sum_net_bps": 31.43359153172043,
                "avg_net_bps": 1.964599470732527,
                "cumulative_net_bps": 31.43359153172043,
            },
        },
        "by_volatility_regime": {
            "low_volatility_regime": {"count": 11, "avg_net_bps": 3.34, "sum_net_bps": 36.78},
            "high_volatility_regime": {"count": 2, "avg_net_bps": -3.63, "sum_net_bps": -7.26},
        },
        "by_spread_regime": {
            "tight_spread_regime": {"count": 17, "avg_net_bps": 1.69, "sum_net_bps": 28.75},
        },
    }
    return {
        "generated_at": "2026-07-23T09:21:19Z",
        "paper_only": True,
        "execution_enabled": False,
        "candidate_f_summary": {
            "feature_set": CANDIDATE_F_FEATURE_SET,
            "best_config_id": FROZEN_CONFIG_ID,
        },
        "positive_configs": [{"config_id": FROZEN_CONFIG_ID, "feature_set": CANDIDATE_F_FEATURE_SET}],
        "config_analyses": [
            {
                "config_id": FROZEN_CONFIG_ID,
                "accepted_trades_csv_path": accepted_trades_csv_path,
                "matching_status": "matched",
                "verdict": verdict,
                "grid_metrics": {"feature_set": CANDIDATE_F_FEATURE_SET},
                "trade_analysis": trade_analysis,
            }
        ],
    }


def _write_walk_forward(
    walk_forward_dir: Path,
    *,
    stamp_suffix: str,
    **kwargs: object,
) -> Path:
    walk_forward_dir.mkdir(parents=True, exist_ok=True)
    path = walk_forward_dir / f"walk_forward_ETH_{stamp_suffix}.json"
    payload = _walk_forward_payload(stamp_suffix=stamp_suffix, **kwargs)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_robustness(
    robustness_dir: Path,
    *,
    stamp: str,
    accepted_trades_csv_path: str,
) -> Path:
    robustness_dir.mkdir(parents=True, exist_ok=True)
    path = robustness_dir / f"candidate_f_robustness_{stamp}.json"
    payload = _robustness_payload(
        stamp_suffix=stamp,
        accepted_trades_csv_path=accepted_trades_csv_path,
    )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def walk_forward_dir(tmp_path: Path) -> Path:
    return tmp_path / "walk_forward"


@pytest.fixture
def robustness_dir(tmp_path: Path) -> Path:
    return tmp_path / "robustness"


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "milestone_comparison"


class TestConfigMatching:
    def test_matches_frozen_candidate_f_config(self) -> None:
        assert config_matches_frozen(_walk_forward_payload()) is True

    def test_rejects_different_feature_set(self) -> None:
        payload = _walk_forward_payload(feature_set="microstructure")
        assert config_matches_frozen(payload) is False

    def test_rejects_different_thresholds(self) -> None:
        payload = _walk_forward_payload()
        payload["min_confidence"] = 0.70
        assert config_matches_frozen(payload) is False


class TestDatasetGrouping:
    def test_dataset_group_key_uses_timestamps_and_coverage(self) -> None:
        summary = {
            "first_timestamp_iso": "2023-09-16T00:00:00Z",
            "latest_timestamp_iso": "2023-09-28T23:59:45Z",
            "coverage_hours": 312.0,
        }
        assert dataset_group_key(summary) == (
            "2023-09-16T00:00:00Z",
            "2023-09-28T23:59:45Z",
            312.0,
        )

    def test_extract_dataset_summary_prefers_embedded_report_fields(self) -> None:
        report = _walk_forward_payload(
            dataset_summary={
                "first_timestamp_iso": "2023-09-16T00:00:00Z",
                "latest_timestamp_iso": "2023-09-20T00:00:00Z",
                "coverage_hours": 96.0,
                "csv_path": "data/bid_ask_csv/historical_15s/ETH_15s.csv",
                "row_count": 100,
            }
        )
        summary = extract_dataset_summary(report)
        assert summary["latest_timestamp_iso"] == "2023-09-20T00:00:00Z"
        assert summary["coverage_hours"] == 96.0


class TestDuplicateSelection:
    def test_select_newest_valid_run_for_same_dataset(self) -> None:
        dataset_summary = {
            "first_timestamp_iso": "2023-09-16T00:00:00Z",
            "latest_timestamp_iso": "2023-09-28T23:59:45Z",
            "coverage_hours": 312.0,
        }
        older = {
            "report_stamp": "20260723T085350",
            "walk_forward_report_path": "older.json",
            "dataset_summary": dataset_summary,
            "valid": True,
        }
        newer = {
            "report_stamp": "20260723T092114",
            "walk_forward_report_path": "newer.json",
            "dataset_summary": dataset_summary,
            "valid": True,
        }
        selected, duplicates = select_newest_valid_runs([older, newer])
        assert len(selected) == 1
        assert selected[0]["report_stamp"] == "20260723T092114"
        assert len(duplicates) == 1
        assert duplicates[0]["superseded_by_stamp"] == "20260723T092114"


class TestDominantDayMetrics:
    def test_excluding_dominant_day_recomputes_net_bps(self) -> None:
        metrics = _dominant_day_metrics(
            {
                "trade_count": 17,
                "cumulative_net_bps": 28.74970710357321,
                "by_day": {
                    "2023-09-21": {"count": 1, "sum_net_bps": -2.683884428147218},
                    "2023-09-23": {"count": 16, "sum_net_bps": 31.43359153172043},
                },
            }
        )
        assert metrics["dominant_day"] == "2023-09-23"
        assert metrics["dominant_day_trade_fraction"] == pytest.approx(16 / 17)
        excluding = metrics["excluding_dominant_day"]
        assert excluding["trade_count"] == 1
        assert excluding["cumulative_net_bps"] == pytest.approx(-2.683884428147218)


class TestDiscoveryAndMatching:
    def test_discover_walk_forward_reports_filters_config(
        self,
        walk_forward_dir: Path,
    ) -> None:
        _write_walk_forward(walk_forward_dir, stamp_suffix="20260723T092114")
        _write_walk_forward(
            walk_forward_dir,
            stamp_suffix="20260723T085208",
            feature_set="microstructure",
        )
        discovered = discover_walk_forward_reports(walk_forward_dir=walk_forward_dir)
        assert len(discovered) == 1
        assert discovered[0][1].name == "walk_forward_ETH_20260723T092114.json"

    def test_find_matching_robustness_report(
        self,
        walk_forward_dir: Path,
        robustness_dir: Path,
    ) -> None:
        csv_path = "reports/bid_ask_walk_forward/accepted_trades_ETH_20260723T092114.csv"
        wf = _walk_forward_payload(
            stamp_suffix="20260723T092114",
            accepted_trades_csv_path=csv_path,
        )
        _write_robustness(
            robustness_dir,
            stamp="20260723T092119",
            accepted_trades_csv_path=csv_path,
        )
        robustness_reports = discover_robustness_reports(robustness_dir=robustness_dir)
        match = find_matching_robustness_report(wf, robustness_reports)
        assert match is not None
        assert match[0] == "20260723T092119"


class TestReportWriters:
    def test_safety_fields_included(self, output_dir: Path) -> None:
        report = build_milestone_comparison_report(
            milestones=[],
            comparison_table=[],
            duplicates=[],
            discovered_walk_forward_count=0,
            discovered_robustness_count=0,
            generated_at="2026-07-23T12:00:00Z",
        )
        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False
        assert report["frozen_config_id"] == FROZEN_CONFIG_ID
        assert report["frozen_config"] == FROZEN_CONFIG
        assert report["safety_note"] == SAFETY_NOTE

        write_comparison_json(report, output_dir=output_dir, json_name="test.json")
        write_comparison_markdown(report, output_dir=output_dir, markdown_name="test.md")
        payload = json.loads((output_dir / "test.json").read_text(encoding="utf-8"))
        markdown = (output_dir / "test.md").read_text(encoding="utf-8")

        assert payload["feature_set"] == CANDIDATE_F_FEATURE_SET
        assert "## Chronological comparison table" in markdown
        assert "## Safety statement" in markdown

    def test_comparison_table_includes_required_columns(self) -> None:
        milestone = build_run_record(
            report_stamp="20260723T092114",
            walk_forward_path=Path("walk_forward.json"),
            walk_forward_report=_walk_forward_payload(),
            dataset_summary={
                "first_timestamp_iso": "2023-09-16T00:00:00Z",
                "latest_timestamp_iso": "2023-09-28T23:59:45Z",
                "coverage_hours": 312.0,
            },
            robustness_match=None,
        )
        table = build_milestone_comparison_table([milestone])
        row = table[0]
        assert row["accepted_trades_count"] == 17
        assert row["completed_windows"] == 646
        assert row["skipped_windows"] == 9
        assert row["robustness_verdict"] == "insufficient_matching_trades"


class TestEndToEnd:
    def test_run_milestone_comparison_deduplicates_and_writes_reports(
        self,
        walk_forward_dir: Path,
        robustness_dir: Path,
        output_dir: Path,
    ) -> None:
        dataset_summary = {
            "first_timestamp_iso": "2023-09-16T00:00:00Z",
            "latest_timestamp_iso": "2023-09-28T23:59:45Z",
            "coverage_hours": 312.0,
        }
        older_csv = "reports/bid_ask_walk_forward/accepted_trades_ETH_20260723T085350.csv"
        newer_csv = "reports/bid_ask_walk_forward/accepted_trades_ETH_20260723T092114.csv"
        _write_walk_forward(
            walk_forward_dir,
            stamp_suffix="20260723T085350",
            dataset_summary=dataset_summary,
            accepted_trades_csv_path=older_csv,
        )
        _write_walk_forward(
            walk_forward_dir,
            stamp_suffix="20260723T092114",
            dataset_summary=dataset_summary,
            accepted_trades_csv_path=newer_csv,
        )
        _write_walk_forward(
            walk_forward_dir,
            stamp_suffix="20260723T085208",
            completed_windows=0,
            dataset_summary={
                "first_timestamp_iso": "2023-09-16T00:00:00Z",
                "latest_timestamp_iso": "2023-09-20T00:00:00Z",
                "coverage_hours": 96.0,
            },
            data_dir="data/bid_ask_csv/combined_15s",
            source="combined",
        )
        _write_robustness(
            robustness_dir,
            stamp="20260723T085355",
            accepted_trades_csv_path=older_csv,
        )
        _write_robustness(
            robustness_dir,
            stamp="20260723T092119",
            accepted_trades_csv_path=newer_csv,
        )

        report = run_milestone_comparison(
            walk_forward_dir=walk_forward_dir,
            robustness_dir=robustness_dir,
            output_dir=output_dir,
            write_stamped=True,
        )

        assert report["milestone_count"] == 1
        assert len(report["duplicate_runs_superseded"]) == 1
        assert report["comparison_table"][0]["dataset_coverage_hours"] == 312.0
        assert report["comparison_table"][0]["robustness_verdict"] == "diagnostic_positive_sparse"
        assert Path(report["json_report_path"]).name == "candidate_f_milestone_comparison_latest.json"
        assert Path(report["markdown_report_path"]).name == "candidate_f_milestone_comparison_latest.md"
        assert len(list(output_dir.glob("candidate_f_milestone_comparison_202*.json"))) == 1


class TestCli:
    def test_main_runs(
        self,
        walk_forward_dir: Path,
        robustness_dir: Path,
        output_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_walk_forward(
            walk_forward_dir,
            stamp_suffix="20260723T092114",
            dataset_summary={
                "first_timestamp_iso": "2023-09-16T00:00:00Z",
                "latest_timestamp_iso": "2023-09-28T23:59:45Z",
                "coverage_hours": 312.0,
            },
        )
        exit_code = main(
            [
                "--walk-forward-dir",
                str(walk_forward_dir),
                "--robustness-dir",
                str(robustness_dir),
                "--output-dir",
                str(output_dir),
                "--no-stamped",
            ]
        )
        captured = capsys.readouterr().out
        assert exit_code == 0
        assert "milestone_count=" in captured
        assert SAFETY_NOTE in captured


class TestSafety:
    def test_forbidden_imports_absent(self) -> None:
        source = COMPARATOR_FILE.read_text(encoding="utf-8")
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

    def test_no_network_subprocess_or_shell(self) -> None:
        source = COMPARATOR_FILE.read_text(encoding="utf-8").lower()
        forbidden = (
            "requests.get",
            "requests.post",
            "urllib.request",
            "http.client",
            "aiohttp",
            "websocket",
            "boto3",
            "subprocess",
            "os.system",
            "popen",
        )
        for pattern in forbidden:
            assert pattern not in source

    def test_paper_only_constants(self) -> None:
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False

    def test_default_paths(self) -> None:
        assert DEFAULT_WALK_FORWARD_DIR == Path("reports/bid_ask_walk_forward")
        assert DEFAULT_ROBUSTNESS_DIR == Path("reports/candidate_f_robustness")
        assert DEFAULT_OUTPUT_DIR == Path("reports/candidate_f_milestone_comparison")
