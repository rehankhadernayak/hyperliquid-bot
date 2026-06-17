"""Offline tests for review package, capital readiness, and review verification."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from capital_readiness import (
    ELIGIBLE_FOR_TINY_SUPERVISED_PILOT_REVIEW,
    NOT_READY,
    NEEDS_MORE_EVIDENCE,
    ReadinessRules,
    assess_capital_readiness,
    format_assessment_report,
)
from config import VERSION
from review_data import (
    PytestRunSummary,
    contains_absolute_local_path,
    contains_sensitive_content,
    experiment_comparison_rows,
    latest_experiment,
    sanitize_text,
    session_summary_rows,
)
import review_documents as docs
from review_package import (
    generate_review_package,
    verify_manifest,
    verify_package_checksums,
)


@pytest.fixture
def sample_experiment() -> dict:
    return {
        "experiment_id": "test-exp-001",
        "created_at": "2026-06-17 00:00:00 UTC",
        "timestamp_count": 100,
        "row_count": 300,
        "input_data_hash": "abc123",
        "date_range": {"start": "2026-06-01 00:00:00 UTC", "end": "2026-06-17 00:00:00 UTC"},
        "split": {"development_timestamps": 70, "out_of_sample_timestamps": 30},
        "profile_results": {
            "contrarian_v1_test": {
                "profile": {"strategy_name": "contrarian_v1"},
                "development": {
                    "net_return_pct": 1.0,
                    "completed_trades": 20,
                    "maximum_drawdown_pct": 2.0,
                    "profit_factor": 1.5,
                    "fees_paid": 1.0,
                    "slippage_cost": 0.5,
                    "gross_profit": 5.0,
                    "trade_log": [],
                    "results_by_coin": {},
                },
                "out_of_sample": {
                    "net_return_pct": 0.5,
                    "completed_trades": 8,
                    "maximum_drawdown_pct": 3.0,
                    "profit_factor": 1.2,
                    "fees_paid": 0.5,
                    "slippage_cost": 0.2,
                    "gross_profit": 2.0,
                    "trade_log": [],
                    "results_by_coin": {"BTC": {"realized_pnl": 1.0, "completed_trades": 5}},
                },
                "combined": {
                    "net_return_pct": 0.8,
                    "completed_trades": 28,
                    "maximum_drawdown_pct": 3.0,
                    "profit_factor": 1.3,
                    "fees_paid": 1.5,
                    "slippage_cost": 0.7,
                },
                "evidence_assessment": "WEAK",
                "overfitting_warnings": ["Small OOS sample"],
            },
            "hold_baseline": {
                "profile": {"strategy_name": "hold_baseline"},
                "development": {"net_return_pct": 0.0, "completed_trades": 0},
                "out_of_sample": {"net_return_pct": 0.0, "completed_trades": 0},
                "combined": {"net_return_pct": 0.0, "completed_trades": 0},
                "evidence_assessment": "INSUFFICIENT_DATA",
                "overfitting_warnings": [],
            },
        },
    }


def test_version_is_review_release():
    assert VERSION == "9.0.0-review"


def test_executive_summary_deterministic(sample_experiment):
    summary = docs.generate_executive_summary(
        test_summary=PytestRunSummary(passed=True, total=100, failed=0, output=""),
        experiments=[sample_experiment],
        latest=sample_experiment,
        readiness_label=NEEDS_MORE_EVIDENCE,
        readiness_summary="Test summary",
    )
    assert "9.0.0-review" in summary
    assert "cannot place real trades" in summary
    assert "hold_baseline" in summary or "contrarian_v1_test" in summary


def test_evidence_summary_missing_data():
    text = docs.generate_evidence_summary(None)
    assert "Insufficient Data" in text


def test_evidence_summary_includes_hold_baseline(sample_experiment):
    text = docs.generate_evidence_summary(sample_experiment)
    assert "hold_baseline" in text
    assert "contrarian_v1_test" in text


def test_experiment_comparison_rows(sample_experiment):
    rows = experiment_comparison_rows(sample_experiment)
    assert len(rows) == 2
    profiles = {row["profile"] for row in rows}
    assert "hold_baseline" in profiles
    assert "contrarian_v1_test" in profiles


def test_session_summary_rows_empty():
    assert session_summary_rows([]) == []


def test_failure_scenarios_cover_required_topics():
    text = docs.generate_failure_scenarios()
    required = (
        "API unavailable",
        "Stale price",
        "Strategy overfitting",
        "Too few trades",
        "Future real-execution module",
    )
    for topic in required:
        assert topic in text


def test_advisor_questions_grouped():
    text = docs.generate_advisor_questions()
    assert "## Technical" in text
    assert "## Strategy" in text
    assert "## Risk" in text
    assert "## Legal" in text


def test_capital_readiness_not_ready_on_unexplained_trades():
    assessment = assess_capital_readiness(
        unexplained_trades=1,
        test_summary=PytestRunSummary(passed=True, total=100, failed=0, output=""),
    )
    assert assessment.label == NOT_READY
    assert any(c.name == "No unexplained trades" for c in assessment.failed_criteria)


def test_capital_readiness_not_ready_on_risk_violations():
    assessment = assess_capital_readiness(
        risk_violations=2,
        test_summary=PytestRunSummary(passed=True, total=100, failed=0, output=""),
    )
    assert assessment.label == NOT_READY


def test_capital_readiness_insufficient_evidence_blocks_progression():
    assessment = assess_capital_readiness(
        rules=ReadinessRules(min_completed_paper_trades=9999),
        test_summary=PytestRunSummary(passed=True, total=100, failed=0, output=""),
    )
    assert assessment.label in (NOT_READY, NEEDS_MORE_EVIDENCE)
    assert any(
        c.name == "Minimum paper trades" and not c.passed
        for c in assessment.failed_criteria
    )


def test_capital_readiness_no_authorization_language():
    assessment = assess_capital_readiness(
        test_summary=PytestRunSummary(passed=True, total=100, failed=0, output=""),
    )
    report = format_assessment_report(assessment)
    assert "approved for trading" not in report.lower()
    assert "authorized to trade" not in report.lower()
    assert "does NOT authorize" in report


def test_capital_readiness_highest_tier_requires_adviser():
    assessment = assess_capital_readiness(
        test_summary=PytestRunSummary(passed=True, total=100, failed=0, output=""),
    )
    if assessment.label == ELIGIBLE_FOR_TINY_SUPERVISED_PILOT_REVIEW:
        report = format_assessment_report(assessment)
        assert "adviser" in report.lower()


def test_review_package_generates_all_files():
    with tempfile.TemporaryDirectory() as tmp:
        package_dir = generate_review_package(
            Path(tmp) / "pkg-test",
            package_id="test-pkg-001",
            skip_tests=True,
        )
        expected = {
            "executive_summary.md",
            "system_architecture.md",
            "strategy_documentation.md",
            "risk_management.md",
            "evidence_summary.md",
            "experiment_comparison.csv",
            "session_summary.csv",
            "known_limitations.md",
            "failure_scenarios.md",
            "advisor_questions.md",
            "review_checklist.md",
            "manifest.json",
            "checksums.sha256",
        }
        actual = {p.name for p in package_dir.iterdir() if p.is_file()}
        assert expected.issubset(actual)


def test_manifest_valid():
    with tempfile.TemporaryDirectory() as tmp:
        package_dir = generate_review_package(
            Path(tmp) / "pkg-manifest",
            package_id="test-pkg-manifest",
            skip_tests=True,
        )
        assert verify_manifest(package_dir / "manifest.json")


def test_checksums_match():
    with tempfile.TemporaryDirectory() as tmp:
        package_dir = generate_review_package(
            Path(tmp) / "pkg-checksums",
            package_id="test-pkg-checksums",
            skip_tests=True,
        )
        assert verify_package_checksums(package_dir)


def test_package_excludes_sensitive_content():
    with tempfile.TemporaryDirectory() as tmp:
        package_dir = generate_review_package(
            Path(tmp) / "pkg-sensitive",
            package_id="test-pkg-sensitive",
            skip_tests=True,
        )
        for path in package_dir.iterdir():
            if path.suffix in (".md", ".json", ".csv"):
                content = path.read_text(encoding="utf-8")
                assert not contains_sensitive_content(content)


def test_package_excludes_absolute_paths():
    with tempfile.TemporaryDirectory() as tmp:
        package_dir = generate_review_package(
            Path(tmp) / "pkg-paths",
            package_id="test-pkg-paths",
            skip_tests=True,
        )
        for path in package_dir.iterdir():
            if path.suffix in (".md", ".json", ".csv"):
                content = path.read_text(encoding="utf-8")
                assert not contains_absolute_local_path(content)


def test_sanitize_text_removes_home_path():
    home = str(Path.home())
    if home and home != "/":
        result = sanitize_text(f"Data stored at {home}/hyperliquid-bot/data")
        assert home not in result


def test_manifest_contains_version():
    with tempfile.TemporaryDirectory() as tmp:
        package_dir = generate_review_package(
            Path(tmp) / "pkg-version",
            package_id="test-pkg-version",
            skip_tests=True,
        )
        manifest = json.loads((package_dir / "manifest.json").read_text())
        assert manifest["version"] == "9.0.0-review"


def test_dashboard_review_imports():
    import dashboard_review

    assert hasattr(dashboard_review, "render_review_mode")


def test_system_architecture_has_mermaid():
    text = docs.generate_system_architecture()
    assert "```mermaid" in text
    assert "Review package" in text


def test_strategy_docs_not_ai():
    text = docs.generate_strategy_documentation()
    assert "rule-based" in text.lower() or "not use trained" in text.lower()
    assert "contrarian_v1_test" in text
    assert "hold_baseline" in text


def test_latest_experiment_returns_most_recent(sample_experiment):
    older = {**sample_experiment, "experiment_id": "old", "created_at": "2026-01-01 00:00:00 UTC"}
    newer = {**sample_experiment, "experiment_id": "new", "created_at": "2026-06-17 00:00:00 UTC"}
    assert latest_experiment([older, newer])["experiment_id"] == "new"


def test_experiment_runner_has_no_network_imports():
    source = Path("experiment_runner.py").read_text(encoding="utf-8")
    assert "import requests" not in source
    assert "from requests" not in source
