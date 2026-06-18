"""
Exit-strategy comparison summary for cost-aware exit experiments.

Compares exit variants against contrarian_v1_medium_signal_exit. Labels are
descriptive only — no automatic winner is declared.
"""

from __future__ import annotations

from typing import Any

from config import EVIDENCE, EVIDENCE_INSUFFICIENT_DATA, EVIDENCE_REJECT, EVIDENCE_UNSTABLE

BASELINE_EXIT_PROFILE = "contrarian_v1_medium_signal_exit"

EXIT_VARIANT_PROFILES = (
    "contrarian_v1_entry_medium",
    "contrarian_v1_medium_tp25_sl25",
    "contrarian_v1_medium_tp35_sl25",
    "contrarian_v1_medium_tp50_sl30",
    "contrarian_v1_medium_tp35_sl25_maxhold",
)

LABEL_INSUFFICIENT_EXIT_DATA = "INSUFFICIENT_EXIT_DATA"
LABEL_LOWER_COSTS_NO_EDGE = "LOWER_COSTS_NO_EDGE"
LABEL_BETTER_NET_EXPECTANCY = "BETTER_NET_EXPECTANCY"
LABEL_LOWER_DRAWDOWN_ONLY = "LOWER_DRAWDOWN_ONLY"
LABEL_UNSTABLE = "UNSTABLE"
LABEL_WORSE = "WORSE"
LABEL_INCONCLUSIVE = "INCONCLUSIVE"

MAX_DRAWDOWN_MATERIAL_WORSE_PCT = 5.0

EXIT_COMPARISON_LABEL_RULES: dict[str, str] = {
    LABEL_INSUFFICIENT_EXIT_DATA: (
        f"OOS completed trades below minimum ({EVIDENCE.min_completed_trades}) "
        f"or evidence is {EVIDENCE_INSUFFICIENT_DATA}."
    ),
    LABEL_LOWER_COSTS_NO_EDGE: (
        "Total OOS costs are lower than baseline but OOS net return and "
        "average net return per trade are not higher."
    ),
    LABEL_BETTER_NET_EXPECTANCY: (
        f"OOS completed trades >= {EVIDENCE.min_completed_trades}, "
        "OOS average net return per trade and OOS return exceed baseline, "
        "drawdown is not materially worse, and results are not dominated "
        "by one trade or coin."
    ),
    LABEL_LOWER_DRAWDOWN_ONLY: (
        "Maximum drawdown is lower than baseline but net expectancy criteria "
        "for BETTER_NET_EXPECTANCY are not met."
    ),
    LABEL_UNSTABLE: (
        f"Evidence assessment is {EVIDENCE_UNSTABLE} or development/OOS "
        "return gap exceeds configured limits."
    ),
    LABEL_WORSE: (
        "Evidence is REJECT, or OOS return and average net return per trade "
        "are worse than baseline with no drawdown advantage."
    ),
    LABEL_INCONCLUSIVE: (
        "Mixed or insufficient separation from baseline."
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


def _dominated_by_single_trade_or_coin(oos_metrics: dict[str, Any]) -> bool:
    sell_trades = [
        trade
        for trade in oos_metrics.get("trade_log", [])
        if trade.get("action") == "SELL" and trade.get("realized_pnl", 0) != 0
    ]
    if not sell_trades:
        return False

    gross_wins = sum(t["realized_pnl"] for t in sell_trades if t["realized_pnl"] > 0)
    if gross_wins <= 0:
        return False

    max_trade_share = max(
        (t["realized_pnl"] / gross_wins) * 100 for t in sell_trades if t["realized_pnl"] > 0
    )
    if max_trade_share > EVIDENCE.max_single_trade_profit_share_pct:
        return True

    coin_pnl: dict[str, float] = {}
    for trade in sell_trades:
        coin_pnl[trade["coin"]] = coin_pnl.get(trade["coin"], 0.0) + trade["realized_pnl"]
    positive_coin_pnl = sum(v for v in coin_pnl.values() if v > 0)
    if positive_coin_pnl <= 0:
        return False
    max_coin_share = max((v / positive_coin_pnl) * 100 for v in coin_pnl.values() if v > 0)
    return max_coin_share > EVIDENCE.max_single_coin_profit_share_pct


def classify_exit_variant(
    baseline_oos: dict[str, Any],
    variant_oos: dict[str, Any],
    *,
    evidence_assessment: str,
    development_metrics: dict[str, Any],
) -> str:
    """
    Classify one exit variant against contrarian_v1_medium_signal_exit.

    Rules evaluated in priority order (first match wins).
    """
    variant_trades = int(variant_oos.get("completed_trades", 0))
    if (
        evidence_assessment == EVIDENCE_INSUFFICIENT_DATA
        or variant_trades < EVIDENCE.min_completed_trades
    ):
        return LABEL_INSUFFICIENT_EXIT_DATA

    if evidence_assessment == EVIDENCE_REJECT:
        return LABEL_WORSE

    if evidence_assessment == EVIDENCE_UNSTABLE:
        return LABEL_UNSTABLE

    baseline_return = float(baseline_oos.get("net_return_pct", 0.0))
    variant_return = float(variant_oos.get("net_return_pct", 0.0))
    baseline_avg_net = float(baseline_oos.get("average_net_return_per_trade", 0.0))
    variant_avg_net = float(variant_oos.get("average_net_return_per_trade", 0.0))
    baseline_dd = float(baseline_oos.get("maximum_drawdown_pct", 0.0))
    variant_dd = float(variant_oos.get("maximum_drawdown_pct", 0.0))
    baseline_costs = _total_costs(baseline_oos)
    variant_costs = _total_costs(variant_oos)

    drawdown_materially_worse = variant_dd > baseline_dd + MAX_DRAWDOWN_MATERIAL_WORSE_PCT
    drawdown_better = variant_dd < baseline_dd

    if (
        variant_return > baseline_return
        and variant_avg_net > baseline_avg_net
        and not drawdown_materially_worse
        and not _dominated_by_single_trade_or_coin(variant_oos)
    ):
        return LABEL_BETTER_NET_EXPECTANCY

    if variant_costs < baseline_costs and variant_return <= baseline_return:
        return LABEL_LOWER_COSTS_NO_EDGE

    if drawdown_better and variant_return <= baseline_return:
        return LABEL_LOWER_DRAWDOWN_ONLY

    dev_return = float(development_metrics.get("net_return_pct", 0.0))
    if abs(dev_return - variant_return) > EVIDENCE.max_dev_oos_return_gap_pct:
        return LABEL_UNSTABLE

    if variant_return < baseline_return and variant_avg_net < baseline_avg_net:
        return LABEL_WORSE

    return LABEL_INCONCLUSIVE


def build_exit_comparison(
    profile_results: dict[str, Any],
    *,
    baseline_profile: str = BASELINE_EXIT_PROFILE,
) -> dict[str, Any] | None:
    """Build exit comparison vs the signal-exit baseline."""
    if baseline_profile not in profile_results:
        return None

    baseline_payload = profile_results[baseline_profile]
    baseline_oos = baseline_payload.get("out_of_sample", {})
    baseline_combined = baseline_payload.get("combined", {})

    comparisons: dict[str, Any] = {}
    for variant_id in EXIT_VARIANT_PROFILES:
        if variant_id not in profile_results:
            continue
        if variant_id == baseline_profile:
            continue

        variant_payload = profile_results[variant_id]
        variant_oos = variant_payload.get("out_of_sample", {})
        variant_combined = variant_payload.get("combined", {})
        variant_dev = variant_payload.get("development", {})

        baseline_trades = int(baseline_oos.get("completed_trades", 0))
        variant_trades = int(variant_oos.get("completed_trades", 0))
        baseline_costs = _total_costs(baseline_oos)
        variant_costs = _total_costs(variant_oos)

        comparisons[variant_id] = {
            "label": classify_exit_variant(
                baseline_oos,
                variant_oos,
                evidence_assessment=variant_payload.get(
                    "evidence_assessment", EVIDENCE_INSUFFICIENT_DATA
                ),
                development_metrics=variant_dev,
            ),
            "oos_return_difference_vs_baseline": round(
                float(variant_oos.get("net_return_pct", 0.0))
                - float(baseline_oos.get("net_return_pct", 0.0)),
                2,
            ),
            "completed_trade_difference_vs_baseline": variant_trades - baseline_trades,
            "average_net_return_per_trade_difference_vs_baseline": round(
                float(variant_oos.get("average_net_return_per_trade", 0.0))
                - float(baseline_oos.get("average_net_return_per_trade", 0.0)),
                4,
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
            "total_cost_difference_vs_baseline": round(variant_costs - baseline_costs, 2),
            "average_holding_time_difference_vs_baseline": round(
                float(variant_oos.get("average_holding_seconds", 0.0))
                - float(baseline_oos.get("average_holding_seconds", 0.0)),
                2,
            ),
            "stop_loss_exit_difference_vs_baseline": int(
                variant_oos.get("stop_loss_exits", 0)
            )
            - int(baseline_oos.get("stop_loss_exits", 0)),
            "take_profit_exit_difference_vs_baseline": int(
                variant_oos.get("take_profit_exits", 0)
            )
            - int(baseline_oos.get("take_profit_exits", 0)),
            "baseline_oos_completed_trades": baseline_trades,
            "variant_oos_completed_trades": variant_trades,
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
        "label_rules": EXIT_COMPARISON_LABEL_RULES,
        "assessment_thresholds": {
            "min_completed_trades": EVIDENCE.min_completed_trades,
            "max_drawdown_material_worse_pct": MAX_DRAWDOWN_MATERIAL_WORSE_PCT,
            "max_single_trade_profit_share_pct": EVIDENCE.max_single_trade_profit_share_pct,
            "max_single_coin_profit_share_pct": EVIDENCE.max_single_coin_profit_share_pct,
            "max_dev_oos_return_gap_pct": EVIDENCE.max_dev_oos_return_gap_pct,
        },
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
