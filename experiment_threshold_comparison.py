"""
Direct threshold-comparison summary for contrarian entry experiments.

Compares wider-threshold variants against contrarian_v1_test. Labels are
descriptive only — no automatic winner is declared.
"""

from __future__ import annotations

from typing import Any

from config import EVIDENCE, EVIDENCE_INSUFFICIENT_DATA, EVIDENCE_REJECT

BASELINE_PROFILE = "contrarian_v1_test"
ENTRY_VARIANT_PROFILES = (
    "contrarian_v1_entry_medium",
    "contrarian_v1_entry_selective",
)

LABEL_MORE_SELECTIVE_BUT_INSUFFICIENT_DATA = "MORE_SELECTIVE_BUT_INSUFFICIENT_DATA"
LABEL_LOWER_COSTS_NO_EDGE = "LOWER_COSTS_NO_EDGE"
LABEL_POSSIBLE_IMPROVEMENT = "POSSIBLE_IMPROVEMENT"
LABEL_WORSE = "WORSE"
LABEL_INCONCLUSIVE = "INCONCLUSIVE"

THRESHOLD_COMPARISON_LABEL_RULES: dict[str, str] = {
    LABEL_MORE_SELECTIVE_BUT_INSUFFICIENT_DATA: (
        "OOS completed trades are below the minimum evidence threshold "
        f"({EVIDENCE.min_completed_trades}) or evidence is {EVIDENCE_INSUFFICIENT_DATA}, "
        "and the variant has fewer OOS completed trades than the baseline."
    ),
    LABEL_LOWER_COSTS_NO_EDGE: (
        "Total OOS costs (fees + slippage) are lower than the baseline, "
        "but OOS net return is not higher than the baseline."
    ),
    LABEL_POSSIBLE_IMPROVEMENT: (
        "OOS net return exceeds the baseline, OOS completed trades meet the "
        f"minimum evidence threshold ({EVIDENCE.min_completed_trades}), "
        "and evidence is not REJECT."
    ),
    LABEL_WORSE: (
        "Evidence is REJECT, or OOS net return is below the baseline while "
        "OOS profit factor is also worse than the baseline."
    ),
    LABEL_INCONCLUSIVE: (
        "Mixed or insufficient separation from the baseline — not enough "
        "evidence to classify as improvement, lower costs, worse, or "
        "insufficient data."
    ),
}


def _numeric_profit_factor(value: Any) -> float | None:
    if value is None:
        return None
    if value == "inf":
        return float("inf")
    return float(value)


def _total_costs(metrics: dict[str, Any]) -> float:
    return float(metrics.get("fees_paid", 0.0)) + float(metrics.get("slippage_cost", 0.0))


def classify_threshold_variant(
    baseline_oos: dict[str, Any],
    variant_oos: dict[str, Any],
    *,
    evidence_assessment: str,
) -> str:
    """
    Classify one wider-threshold variant against the baseline.

    Rules are evaluated in priority order (first match wins):

    1. MORE_SELECTIVE_BUT_INSUFFICIENT_DATA
       - evidence_assessment == INSUFFICIENT_DATA, OR
         variant OOS completed_trades < EVIDENCE.min_completed_trades
       - AND variant OOS completed_trades < baseline OOS completed_trades

    2. WORSE (evidence REJECT)
       - evidence_assessment == REJECT

    3. LOWER_COSTS_NO_EDGE
       - variant total OOS costs < baseline total OOS costs
       - variant OOS net_return_pct <= baseline OOS net_return_pct

    4. POSSIBLE_IMPROVEMENT
       - variant OOS net_return_pct > baseline OOS net_return_pct
       - variant OOS completed_trades >= EVIDENCE.min_completed_trades
       - evidence_assessment != REJECT

    5. WORSE (return and profit factor)
       - variant OOS net_return_pct < baseline OOS net_return_pct AND
         variant profit factor is worse than baseline (when both are known)

    6. INCONCLUSIVE — all remaining cases
    """
    baseline_trades = int(baseline_oos.get("completed_trades", 0))
    variant_trades = int(variant_oos.get("completed_trades", 0))
    baseline_return = float(baseline_oos.get("net_return_pct", 0.0))
    variant_return = float(variant_oos.get("net_return_pct", 0.0))
    baseline_costs = _total_costs(baseline_oos)
    variant_costs = _total_costs(variant_oos)

    baseline_pf = _numeric_profit_factor(baseline_oos.get("profit_factor"))
    variant_pf = _numeric_profit_factor(variant_oos.get("profit_factor"))

    insufficient_data = (
        evidence_assessment == EVIDENCE_INSUFFICIENT_DATA
        or variant_trades < EVIDENCE.min_completed_trades
    )
    if insufficient_data and variant_trades < baseline_trades:
        return LABEL_MORE_SELECTIVE_BUT_INSUFFICIENT_DATA

    if evidence_assessment == EVIDENCE_REJECT:
        return LABEL_WORSE

    if variant_costs < baseline_costs and variant_return <= baseline_return:
        return LABEL_LOWER_COSTS_NO_EDGE

    if (
        variant_return > baseline_return
        and variant_trades >= EVIDENCE.min_completed_trades
    ):
        return LABEL_POSSIBLE_IMPROVEMENT

    pf_worse = _profit_factor_worse(variant_pf, baseline_pf)
    if variant_return < baseline_return and pf_worse:
        return LABEL_WORSE

    return LABEL_INCONCLUSIVE


def _profit_factor_worse(variant: float | None, baseline: float | None) -> bool:
    if variant is None and baseline is None:
        return False
    if variant is None:
        return True
    if baseline is None:
        return False
    if variant == float("inf"):
        return False
    if baseline == float("inf"):
        return True
    return variant < baseline


def build_threshold_comparison(
    profile_results: dict[str, Any],
    *,
    baseline_profile: str = BASELINE_PROFILE,
) -> dict[str, Any] | None:
    """
    Build a concise comparison of entry-threshold variants vs the baseline.

    Returns None when the baseline profile is not part of the experiment.
    """
    if baseline_profile not in profile_results:
        return None

    baseline_payload = profile_results[baseline_profile]
    baseline_oos = baseline_payload.get("out_of_sample", {})
    baseline_combined = baseline_payload.get("combined", {})

    comparisons: dict[str, Any] = {}
    for variant_id in ENTRY_VARIANT_PROFILES:
        if variant_id not in profile_results:
            continue

        variant_payload = profile_results[variant_id]
        variant_oos = variant_payload.get("out_of_sample", {})
        variant_combined = variant_payload.get("combined", {})

        baseline_oos_trades = int(baseline_oos.get("completed_trades", 0))
        variant_oos_trades = int(variant_oos.get("completed_trades", 0))
        baseline_costs = _total_costs(baseline_oos)
        variant_costs = _total_costs(variant_oos)

        comparisons[variant_id] = {
            "label": classify_threshold_variant(
                baseline_oos,
                variant_oos,
                evidence_assessment=variant_payload.get(
                    "evidence_assessment", EVIDENCE_INSUFFICIENT_DATA
                ),
            ),
            "trade_count_reduction_vs_baseline": baseline_oos_trades - variant_oos_trades,
            "fee_reduction_vs_baseline": round(
                float(baseline_oos.get("fees_paid", 0.0))
                - float(variant_oos.get("fees_paid", 0.0)),
                2,
            ),
            "slippage_reduction_vs_baseline": round(
                float(baseline_oos.get("slippage_cost", 0.0))
                - float(variant_oos.get("slippage_cost", 0.0)),
                2,
            ),
            "total_cost_reduction_vs_baseline": round(baseline_costs - variant_costs, 2),
            "oos_return_difference_vs_baseline": round(
                float(variant_oos.get("net_return_pct", 0.0))
                - float(baseline_oos.get("net_return_pct", 0.0)),
                2,
            ),
            "profit_factor_difference_vs_baseline": _profit_factor_difference(
                variant_oos.get("profit_factor"),
                baseline_oos.get("profit_factor"),
            ),
            "maximum_drawdown_difference_vs_baseline": round(
                float(variant_oos.get("maximum_drawdown_pct", 0.0))
                - float(baseline_oos.get("maximum_drawdown_pct", 0.0)),
                2,
            ),
            "baseline_oos_completed_trades": baseline_oos_trades,
            "variant_oos_completed_trades": variant_oos_trades,
            "baseline_combined_completed_trades": int(
                baseline_combined.get("completed_trades", 0)
            ),
            "variant_combined_completed_trades": int(
                variant_combined.get("completed_trades", 0)
            ),
        }

    if not comparisons:
        return None

    return {
        "baseline_profile": baseline_profile,
        "label_rules": THRESHOLD_COMPARISON_LABEL_RULES,
        "comparisons": comparisons,
    }


def _profit_factor_difference(variant: Any, baseline: Any) -> float | str | None:
    variant_pf = _numeric_profit_factor(variant)
    baseline_pf = _numeric_profit_factor(baseline)
    if variant_pf is None and baseline_pf is None:
        return None
    if variant_pf == float("inf") or baseline_pf == float("inf"):
        if variant_pf == baseline_pf:
            return 0.0
        if variant_pf == float("inf"):
            return "inf"
        return "-inf"
    if variant_pf is None or baseline_pf is None:
        return None
    return round(variant_pf - baseline_pf, 2)
