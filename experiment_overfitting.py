"""Overfitting and data-leakage warnings for strategy experiments."""

from __future__ import annotations

from typing import Any

from config import EVIDENCE, ExperimentConfig


def collect_overfitting_warnings(
    *,
    development: dict[str, Any],
    out_of_sample: dict[str, Any] | None,
    combined: dict[str, Any],
    profile_count: int,
    split_identical: bool,
    used_future_data: bool,
    rules=EVIDENCE,
    experiment_config: ExperimentConfig | None = None,
) -> list[str]:
    """Return human-readable warnings — does not pick a winner."""
    warnings: list[str] = []
    oos = out_of_sample or {}

    completed = oos.get("completed_trades", combined.get("completed_trades", 0))
    if completed < rules.min_completed_trades:
        warnings.append(
            f"Too few completed trades in out-of-sample period ({completed} "
            f"< {rules.min_completed_trades})."
        )

    dev_return = development.get("net_return_pct", 0.0)
    oos_return = oos.get("net_return_pct", 0.0)
    gap = dev_return - oos_return
    if gap >= rules.max_dev_oos_return_gap_pct:
        warnings.append(
            f"Large development vs out-of-sample return drop "
            f"({dev_return:.2f}% dev vs {oos_return:.2f}% OOS)."
        )

    gross = oos.get("gross_profit", combined.get("gross_profit", 0.0))
    if gross > 0:
        best_trade = _best_trade_pnl(oos.get("trade_log", combined.get("trade_log", [])))
        if best_trade / gross * 100 >= rules.max_single_trade_profit_share_pct:
            warnings.append(
                "Most profit comes from a single trade — results may not generalize."
            )

        dominant_coin = _dominant_coin_share(oos.get("results_by_coin", {}), gross)
        if dominant_coin is not None and dominant_coin >= rules.max_single_coin_profit_share_pct:
            warnings.append(
                "Most profit comes from one coin — strategy may be coin-dependent."
            )

        fees_pct = (oos.get("fees_paid", 0.0) / gross) * 100
        if fees_pct >= rules.max_fees_of_gross_profit_pct:
            warnings.append(
                f"Fees consume {fees_pct:.1f}% of gross profit — edge may be fragile."
            )

    if profile_count > rules.max_parameter_combinations:
        warnings.append(
            f"Many profiles compared ({profile_count}) — risk of multiple-comparison bias."
        )

    if split_identical:
        warnings.append(
            "Development and out-of-sample periods are identical — no true holdout."
        )

    if used_future_data:
        warnings.append(
            "Potential look-ahead detected — evaluation may have accessed future data."
        )

    if experiment_config is not None and not oos:
        warnings.append(
            "No out-of-sample rows available — only development metrics are meaningful."
        )

    return warnings


def _best_trade_pnl(trades: list[dict[str, Any]]) -> float:
    best = 0.0
    for trade in trades:
        if trade.get("action") != "SELL":
            continue
        pnl = trade.get("realized_pnl", 0.0)
        if pnl > best:
            best = pnl
    return best


def _dominant_coin_share(by_coin: dict[str, dict[str, Any]], gross: float) -> float | None:
    if gross <= 0 or not by_coin:
        return None

    best = max(stats.get("realized_pnl", 0.0) for stats in by_coin.values())
    if best <= 0:
        return None
    return (best / gross) * 100
