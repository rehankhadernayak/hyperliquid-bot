"""Conservative capital-readiness assessment (descriptive only — never authorizes trading)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any

from config import EVIDENCE, EVIDENCE_INSUFFICIENT_DATA, EVIDENCE_PROMISING, VERSION
from review_data import (
    count_independent_sessions,
    count_market_days_from_experiment,
    count_paper_trades,
    latest_experiment,
    load_all_experiments,
    load_all_session_reports,
    run_test_summary,
)

NOT_READY = "NOT_READY"
NEEDS_MORE_EVIDENCE = "NEEDS_MORE_EVIDENCE"
READY_FOR_ADVISOR_REVIEW = "READY_FOR_ADVISOR_REVIEW"
ELIGIBLE_FOR_TINY_SUPERVISED_PILOT_REVIEW = "ELIGIBLE_FOR_TINY_SUPERVISED_PILOT_REVIEW"

AUTHORIZATION_PHRASES = (
    "approved for trading",
    "authorized to trade",
    "go live",
    "deploy capital",
    "automatic approval",
)


@dataclass
class ReadinessRules:
    """Configurable conservative thresholds."""

    min_completed_paper_trades: int = 20
    min_independent_sessions: int = 3
    min_market_days: int = 14
    min_oos_completed_trades: int = EVIDENCE.min_completed_trades_promising
    max_oos_drawdown_pct: float = EVIDENCE.max_drawdown_weak_pct
    min_profit_factor: float = EVIDENCE.min_profit_factor
    max_dev_oos_gap_pct: float = EVIDENCE.max_dev_oos_return_gap_pct
    min_oos_return_pct: float = 0.0
    require_tests_pass: bool = True
    require_promising_evidence: bool = True
    block_on_risk_violations: bool = True
    block_on_unexplained_trades: bool = True


@dataclass
class CriterionResult:
    name: str
    passed: bool
    detail: str


@dataclass
class ReadinessAssessment:
    label: str
    summary: str
    passed_criteria: list[CriterionResult] = field(default_factory=list)
    failed_criteria: list[CriterionResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "summary": self.summary,
            "passed_criteria": [
                {"name": c.name, "detail": c.detail} for c in self.passed_criteria
            ],
            "failed_criteria": [
                {"name": c.name, "detail": c.detail} for c in self.failed_criteria
            ],
            "warnings": self.warnings,
            "disclaimer": (
                "This assessment is descriptive only. The highest result means evidence "
                "may be presented to a qualified adult adviser for consideration. "
                "It does not authorize real trading."
            ),
        }


def assess_capital_readiness(
    *,
    rules: ReadinessRules | None = None,
    test_summary=None,
    experiments: list[dict[str, Any]] | None = None,
    risk_violations: int = 0,
    unexplained_trades: int = 0,
    health_check_passed: bool = True,
) -> ReadinessAssessment:
    """
    Return a conservative descriptive assessment.

    Never produces automatic approval to trade real capital.
    """
    rules = rules or ReadinessRules()
    if test_summary is None:
        test_summary = run_test_summary()
    if experiments is None:
        experiments = load_all_experiments()

    latest = latest_experiment(experiments)
    paper_trades = count_paper_trades()
    sessions = count_independent_sessions()
    market_days = count_market_days_from_experiment(latest)

    passed: list[CriterionResult] = []
    failed: list[CriterionResult] = []
    warnings: list[str] = []

    def record(name: str, ok: bool, detail: str) -> None:
        entry = CriterionResult(name=name, passed=ok, detail=detail)
        if ok:
            passed.append(entry)
        else:
            failed.append(entry)

    record(
        "Automated tests pass",
        test_summary.passed or not rules.require_tests_pass,
        f"{test_summary.total} tests, {test_summary.failed} failed",
    )
    record(
        "Health check passed",
        health_check_passed,
        "health_check.py must pass before review",
    )
    record(
        "No unexplained trades",
        unexplained_trades == 0 or not rules.block_on_unexplained_trades,
        f"{unexplained_trades} unexplained trade(s) detected",
    )
    record(
        "No risk-rule violations",
        risk_violations == 0 or not rules.block_on_risk_violations,
        f"{risk_violations} risk-rule violation(s) detected",
    )
    record(
        "Minimum paper trades",
        paper_trades >= rules.min_completed_paper_trades,
        f"{paper_trades} completed paper trades (need {rules.min_completed_paper_trades})",
    )
    record(
        "Minimum independent sessions",
        sessions >= rules.min_independent_sessions,
        f"{sessions} session(s) (need {rules.min_independent_sessions})",
    )
    record(
        "Minimum market days covered",
        market_days >= rules.min_market_days,
        f"{market_days} market day(s) (need {rules.min_market_days}; one week alone does not qualify)",
    )

    oos_metrics = _best_active_profile_oos(latest)
    oos_trades = oos_metrics.get("completed_trades", 0)
    oos_return = oos_metrics.get("net_return_pct", 0.0)
    oos_dd = oos_metrics.get("maximum_drawdown_pct", 100.0)
    oos_pf = _pf_value(oos_metrics.get("profit_factor"))
    evidence_label = oos_metrics.get("evidence_label", EVIDENCE_INSUFFICIENT_DATA)
    dev_oos_gap = oos_metrics.get("dev_oos_gap", 999.0)

    record(
        "Minimum OOS completed trades",
        oos_trades >= rules.min_oos_completed_trades,
        f"{oos_trades} OOS trades (need {rules.min_oos_completed_trades})",
    )
    record(
        "OOS return not sole qualifier",
        oos_trades >= rules.min_oos_completed_trades,
        "Positive return alone does not qualify — requires sufficient trade count",
    )
    record(
        "Win rate not sole qualifier",
        oos_trades >= rules.min_oos_completed_trades,
        "High win rate alone does not qualify — requires sufficient trade count",
    )
    record(
        "OOS maximum drawdown acceptable",
        oos_dd < rules.max_oos_drawdown_pct,
        f"OOS max drawdown {oos_dd:.2f}% (limit {rules.max_oos_drawdown_pct}%)",
    )
    record(
        "OOS profit factor acceptable",
        oos_pf is not None and oos_pf >= rules.min_profit_factor,
        f"OOS profit factor {oos_pf} (need ≥ {rules.min_profit_factor})",
    )
    record(
        "Development/OOS consistency",
        abs(dev_oos_gap) < rules.max_dev_oos_gap_pct,
        f"Dev/OOS return gap {dev_oos_gap:+.2f}% (limit {rules.max_dev_oos_gap_pct}%)",
    )
    record(
        "Evidence label acceptable",
        evidence_label
        not in (EVIDENCE_INSUFFICIENT_DATA, "REJECT", "UNSTABLE")
        if rules.require_promising_evidence
        else True,
        f"Best active profile evidence: {evidence_label}",
    )

    if evidence_label == EVIDENCE_PROMISING:
        passed.append(
            CriterionResult(
                name="Promising evidence label",
                passed=True,
                detail="At least one active profile has PROMISING evidence",
            )
        )
    else:
        failed.append(
            CriterionResult(
                name="Promising evidence label",
                passed=False,
                detail=f"No active profile has PROMISING evidence (best: {evidence_label})",
            )
        )

    if oos_return < rules.min_oos_return_pct:
        warnings.append(
            f"OOS return ({oos_return:+.2f}%) is below minimum ({rules.min_oos_return_pct}%)"
        )

    blocking = any(
        not c.passed
        for c in failed
        if c.name
        in (
            "No unexplained trades",
            "No risk-rule violations",
            "Automated tests pass",
            "Health check passed",
        )
    )

    label = _determine_label(
        blocking=blocking,
        failed=failed,
        passed=passed,
        rules=rules,
        oos_trades=oos_trades,
        evidence_label=evidence_label,
        market_days=market_days,
    )

    summary = _build_summary(label, passed, failed, warnings)
    assessment = ReadinessAssessment(
        label=label,
        summary=summary,
        passed_criteria=passed,
        failed_criteria=failed,
        warnings=warnings,
    )
    _assert_no_authorization_language(assessment)
    return assessment


def _best_active_profile_oos(latest: dict[str, Any] | None) -> dict[str, Any]:
    if latest is None:
        return {}

    best: dict[str, Any] = {}
    for profile_id, payload in latest.get("profile_results", {}).items():
        if profile_id == "hold_baseline":
            continue
        oos = payload.get("out_of_sample", {})
        dev = payload.get("development", {})
        entry = {
            "profile_id": profile_id,
            "completed_trades": oos.get("completed_trades", 0),
            "net_return_pct": oos.get("net_return_pct", 0.0),
            "maximum_drawdown_pct": oos.get("maximum_drawdown_pct", 0.0),
            "profit_factor": oos.get("profit_factor"),
            "evidence_label": payload.get("evidence_assessment", EVIDENCE_INSUFFICIENT_DATA),
            "dev_oos_gap": dev.get("net_return_pct", 0) - oos.get("net_return_pct", 0),
        }
        if not best or entry["completed_trades"] > best.get("completed_trades", 0):
            best = entry
    return best


def _pf_value(value: Any) -> float | None:
    if value is None:
        return None
    if value == "inf":
        return float("inf")
    return float(value)


def _determine_label(
    *,
    blocking: bool,
    failed: list[CriterionResult],
    passed: list[CriterionResult],
    rules: ReadinessRules,
    oos_trades: int,
    evidence_label: str,
    market_days: int,
) -> str:
    if blocking:
        return NOT_READY

    critical_failures = [
        c.name
        for c in failed
        if c.name
        in (
            "Minimum paper trades",
            "Minimum independent sessions",
            "Minimum market days covered",
            "Minimum OOS completed trades",
            "No unexplained trades",
            "No risk-rule violations",
        )
    ]
    if critical_failures:
        if oos_trades < rules.min_completed_paper_trades // 2:
            return NOT_READY
        return NEEDS_MORE_EVIDENCE

    core_passed = all(
        c.passed
        for c in passed
        if c.name
        in (
            "Minimum paper trades",
            "Minimum independent sessions",
            "Minimum market days covered",
            "Minimum OOS completed trades",
            "OOS maximum drawdown acceptable",
            "Development/OOS consistency",
            "Evidence label acceptable",
        )
    )

    if not core_passed or market_days < rules.min_market_days:
        return NEEDS_MORE_EVIDENCE

    if evidence_label == EVIDENCE_PROMISING and len(failed) <= 2:
        return ELIGIBLE_FOR_TINY_SUPERVISED_PILOT_REVIEW

    if len(failed) <= 4:
        return READY_FOR_ADVISOR_REVIEW

    return NEEDS_MORE_EVIDENCE


def _build_summary(
    label: str,
    passed: list[CriterionResult],
    failed: list[CriterionResult],
    warnings: list[str],
) -> str:
    lines = [
        f"Assessment: **{label}**",
        "",
        f"Passed criteria: {len(passed)} | Failed criteria: {len(failed)}",
        "",
    ]
    if label == ELIGIBLE_FOR_TINY_SUPERVISED_PILOT_REVIEW:
        lines.append(
            "Evidence may be presented to a qualified adult adviser for consideration "
            "of a tiny supervised pilot. This does **not** authorize real trading."
        )
    elif label == READY_FOR_ADVISOR_REVIEW:
        lines.append(
            "Sufficient structure exists for adviser review, but evidence gaps remain. "
            "Present materials for technical and risk review only."
        )
    elif label == NEEDS_MORE_EVIDENCE:
        lines.append(
            "Continue paper testing and evidence collection before adviser review."
        )
    else:
        lines.append(
            "System is not ready for capital consideration. Address blocking issues first."
        )

    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"- {w}")

    return "\n".join(lines)


def _assert_no_authorization_language(assessment: ReadinessAssessment) -> None:
    text = json.dumps(assessment.to_dict()).lower()
    for phrase in AUTHORIZATION_PHRASES:
        if phrase in text and phrase != "go live":
            raise RuntimeError(
                f"Capital readiness output must not contain authorization language: {phrase!r}"
            )


def format_assessment_report(assessment: ReadinessAssessment) -> str:
    lines = [
        f"Capital Readiness Assessment — {VERSION}",
        "=" * 60,
        f"Result: {assessment.label}",
        "",
        assessment.summary,
        "",
        "PASSED CRITERIA:",
    ]
    for c in assessment.passed_criteria:
        lines.append(f"  ✓ {c.name}: {c.detail}")
    lines.append("")
    lines.append("FAILED CRITERIA:")
    for c in assessment.failed_criteria:
        lines.append(f"  ✗ {c.name}: {c.detail}")
    if assessment.warnings:
        lines.append("")
        lines.append("WARNINGS:")
        for w in assessment.warnings:
            lines.append(f"  ! {w}")
    lines.extend(
        [
            "",
            "-" * 60,
            "DISCLAIMER: This assessment is descriptive only.",
            "The highest result means evidence may be presented to a qualified",
            "adult adviser for consideration. It does NOT authorize real trading.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Conservative capital-readiness assessment (descriptive only)."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output assessment as JSON",
    )
    args = parser.parse_args()

    assessment = assess_capital_readiness()

    if args.json:
        print(json.dumps(assessment.to_dict(), indent=2))
    else:
        print(format_assessment_report(assessment))

    return 0


if __name__ == "__main__":
    sys.exit(main())
