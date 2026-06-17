"""Evidence assessment for strategy experiment results."""

from __future__ import annotations

from typing import Any

from config import (
    EVIDENCE,
    EVIDENCE_INSUFFICIENT_DATA,
    EVIDENCE_PROMISING,
    EVIDENCE_REJECT,
    EVIDENCE_UNSTABLE,
    EVIDENCE_WEAK,
)


def assess_evidence(
    development: dict[str, Any],
    out_of_sample: dict[str, Any] | None,
    *,
    rules=EVIDENCE,
) -> str:
    """
    Label experiment evidence using configured rules.

    Never returns a label implying proven profitability.
    """
    oos = out_of_sample or {}
    completed = oos.get("completed_trades", development.get("completed_trades", 0))
    oos_return = oos.get("net_return_pct", 0.0)
    oos_drawdown = oos.get("maximum_drawdown_pct", development.get("maximum_drawdown_pct", 0.0))
    dev_return = development.get("net_return_pct", 0.0)
    pf = _numeric_profit_factor(oos.get("profit_factor", development.get("profit_factor")))

    if completed < rules.min_completed_trades:
        return EVIDENCE_INSUFFICIENT_DATA

    if oos_drawdown >= rules.max_drawdown_reject_pct:
        return EVIDENCE_REJECT

    gap = dev_return - oos_return
    if gap >= rules.max_dev_oos_return_gap_pct:
        return EVIDENCE_UNSTABLE

    fees_pct = _fees_pct_of_gross(oos)
    if fees_pct is not None and fees_pct >= rules.max_fees_of_gross_profit_pct:
        return EVIDENCE_WEAK

    if _single_trade_dominance(oos, rules.max_single_trade_profit_share_pct):
        return EVIDENCE_UNSTABLE

    if _single_coin_dominance(oos, rules.max_single_coin_profit_share_pct):
        return EVIDENCE_UNSTABLE

    if (
        completed >= rules.min_completed_trades_promising
        and oos_return >= rules.min_out_of_sample_return_pct
        and oos_drawdown < rules.max_drawdown_weak_pct
        and pf is not None
        and pf >= rules.min_profit_factor
        and gap < rules.max_dev_oos_return_gap_pct / 2
    ):
        return EVIDENCE_PROMISING

    if oos_return < 0 or (pf is not None and pf < 1.0):
        return EVIDENCE_REJECT

    return EVIDENCE_WEAK


def _numeric_profit_factor(value: Any) -> float | None:
    if value is None:
        return None
    if value == "inf":
        return float("inf")
    return float(value)


def _fees_pct_of_gross(metrics: dict[str, Any]) -> float | None:
    gross = metrics.get("gross_profit", 0.0)
    fees = metrics.get("fees_paid", 0.0)
    if gross <= 0:
        return None
    return (fees / gross) * 100


def _single_trade_dominance(metrics: dict[str, Any], threshold_pct: float) -> bool:
    gross = metrics.get("gross_profit", 0.0)
    if gross <= 0:
        return False

    sells = [
        trade
        for trade in metrics.get("trade_log", [])
        if trade.get("action") == "SELL" and trade.get("realized_pnl", 0) > 0
    ]
    if not sells:
        return False

    best = max(trade.get("realized_pnl", 0.0) for trade in sells)
    return (best / gross) * 100 >= threshold_pct


def _single_coin_dominance(metrics: dict[str, Any], threshold_pct: float) -> bool:
    gross = metrics.get("gross_profit", 0.0)
    if gross <= 0:
        return False

    by_coin = metrics.get("results_by_coin", {})
    if not by_coin:
        return False

    best_coin_pnl = max(coin_stats.get("realized_pnl", 0.0) for coin_stats in by_coin.values())
    if best_coin_pnl <= 0:
        return False

    return (best_coin_pnl / gross) * 100 >= threshold_pct
