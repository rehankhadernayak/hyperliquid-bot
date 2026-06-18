"""
Position-aware exit evaluation for offline strategy experiments.

Uses only data at or before the current timestamp — no future prices or rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import SignalThresholds
from exit_config import (
    EXIT_MODE_COST_AWARE,
    EXIT_MODE_SIGNAL_V1,
    EXIT_PRIORITY,
    EXIT_TYPE_HOLD,
    EXIT_TYPE_MAX_HOLD,
    EXIT_TYPE_SIGNAL_EXIT,
    EXIT_TYPE_STOP_LOSS,
    EXIT_TYPE_TAKE_PROFIT,
    ExitConfig,
)
from market_analyzer import PriceHistory, rolling_percent_change
from paper_trader import BUY_SIGNALS, SELL_SIGNALS
from signals import signal_from_change
from strategy_profiles import StrategyProfile
from strategy_types import PositionExitContext, StrategyExitResult


class FutureDataAccessError(RuntimeError):
    """Raised when exit evaluation would use data after the current timestamp."""


@dataclass(frozen=True)
class _ExitCandidate:
    exit_type: str
    reason: str
    indicators: dict[str, Any]


def _assert_no_future_history(
    price_history: PriceHistory,
    coin: str,
    now: float,
) -> None:
    for tick_time, _ in price_history.get(coin, []):
        if tick_time > now:
            raise FutureDataAccessError(
                f"Price history for {coin} contains tick at {tick_time} "
                f"after current time {now}"
            )


def _gross_return_pct(avg_entry_price: float, current_price: float) -> float:
    if avg_entry_price <= 0:
        return 0.0
    return ((current_price - avg_entry_price) / avg_entry_price) * 100.0


def _estimate_exit_costs(
    *,
    avg_entry_price: float,
    quantity: float,
    current_price: float,
    trading_fee_pct: float,
    slippage_pct: float,
) -> dict[str, float]:
    """Estimate round-trip costs without double-counting slippage."""
    position_value_mid = quantity * current_price
    slip_rate = slippage_pct / 100.0
    fee_rate = trading_fee_pct / 100.0

    execution_sell_price = current_price * (1.0 - slip_rate)
    sell_trade_value = quantity * execution_sell_price
    estimated_sell_fee = sell_trade_value * fee_rate
    estimated_sell_slippage = max((current_price - execution_sell_price) * quantity, 0.0)

    entry_notional = avg_entry_price * quantity
    estimated_buy_fee = entry_notional * fee_rate
    estimated_buy_slippage = entry_notional * slip_rate

    total_estimated_round_trip_cost = (
        estimated_buy_fee
        + estimated_buy_slippage
        + estimated_sell_fee
        + estimated_sell_slippage
    )

    gross_pnl = (execution_sell_price - avg_entry_price) * quantity
    estimated_net_pnl = gross_pnl - estimated_sell_fee
    estimated_net_return_pct = (
        (estimated_net_pnl / entry_notional) * 100.0 if entry_notional > 0 else 0.0
    )

    return {
        "estimated_sell_fee": estimated_sell_fee,
        "estimated_sell_slippage": estimated_sell_slippage,
        "estimated_buy_fee": estimated_buy_fee,
        "estimated_buy_slippage": estimated_buy_slippage,
        "total_estimated_round_trip_cost": total_estimated_round_trip_cost,
        "estimated_net_return_pct": estimated_net_return_pct,
        "execution_sell_price": execution_sell_price,
    }


def build_position_exit_context(
    *,
    coin: str,
    current_price: float,
    avg_entry_price: float,
    quantity: float,
    opened_at: float | None,
    entry_count: int,
    now: float,
    price_history: PriceHistory,
    trader_state: dict[str, Any],
    profile: StrategyProfile,
    lookback_seconds: int,
    thresholds: SignalThresholds,
) -> PositionExitContext:
    """Build the full context available to exit evaluation."""
    _assert_no_future_history(price_history, coin, now)

    position_value = quantity * current_price
    gross_return_pct = _gross_return_pct(avg_entry_price, current_price)
    costs = _estimate_exit_costs(
        avg_entry_price=avg_entry_price,
        quantity=quantity,
        current_price=current_price,
        trading_fee_pct=profile.trading_fee_pct,
        slippage_pct=profile.slippage_pct,
    )
    holding_time_seconds = (now - opened_at) if opened_at is not None else 0.0

    history = price_history.get(coin, [])
    change_1m = rolling_percent_change(
        history,
        current_price,
        now,
        lookback_seconds,
    )
    signal = signal_from_change(change_1m, thresholds) if change_1m is not None else "N/A"

    return PositionExitContext(
        coin=coin,
        current_price=current_price,
        avg_entry_price=avg_entry_price,
        quantity=quantity,
        position_value=position_value,
        unrealized_pnl_before_fees=(current_price - avg_entry_price) * quantity,
        gross_return_pct=gross_return_pct,
        estimated_sell_fee=costs["estimated_sell_fee"],
        estimated_sell_slippage=costs["estimated_sell_slippage"],
        total_estimated_round_trip_cost=costs["total_estimated_round_trip_cost"],
        estimated_net_return_pct=costs["estimated_net_return_pct"],
        opened_at=opened_at,
        holding_time_seconds=holding_time_seconds,
        entry_count=entry_count,
        now=now,
        price_history=price_history,
        trader_state=trader_state,
        profile=profile,
        change_1m=change_1m,
        current_signal=signal,
    )


def _detect_sparse_ambiguity(
    context: PositionExitContext,
    *,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
) -> bool:
    """True when both SL and TP could have triggered between sparse samples."""
    if stop_loss_pct is None or take_profit_pct is None:
        return False

    history = context.price_history.get(context.coin, [])
    if len(history) < 2:
        return False

    prev_price = history[-2][1]
    prev_gross = _gross_return_pct(context.avg_entry_price, prev_price)
    curr_gross = context.gross_return_pct

    low = min(prev_gross, curr_gross)
    high = max(prev_gross, curr_gross)
    return low <= -stop_loss_pct and high >= take_profit_pct


def _collect_cost_aware_candidates(
    context: PositionExitContext,
    exit_config: ExitConfig,
) -> tuple[list[_ExitCandidate], list[str]]:
    """Evaluate TP / SL / max-hold conditions."""
    triggered: list[_ExitCandidate] = []
    warnings: list[str] = []

    gross = context.gross_return_pct

    if exit_config.stop_loss_pct is not None and gross <= -exit_config.stop_loss_pct:
        triggered.append(
            _ExitCandidate(
                exit_type=EXIT_TYPE_STOP_LOSS,
                reason=(
                    f"gross return {gross:.4f}% at or below stop-loss "
                    f"-{exit_config.stop_loss_pct}% from average entry"
                ),
                indicators={
                    "gross_return_pct": gross,
                    "stop_loss_threshold_pct": -exit_config.stop_loss_pct,
                },
            )
        )

    if exit_config.take_profit_pct is not None and gross >= exit_config.take_profit_pct:
        triggered.append(
            _ExitCandidate(
                exit_type=EXIT_TYPE_TAKE_PROFIT,
                reason=(
                    f"gross return {gross:.4f}% at or above take-profit "
                    f"+{exit_config.take_profit_pct}% from average entry"
                ),
                indicators={
                    "gross_return_pct": gross,
                    "take_profit_threshold_pct": exit_config.take_profit_pct,
                },
            )
        )

    if (
        exit_config.max_hold_seconds is not None
        and context.holding_time_seconds >= exit_config.max_hold_seconds
    ):
        triggered.append(
            _ExitCandidate(
                exit_type=EXIT_TYPE_MAX_HOLD,
                reason=(
                    f"holding time {context.holding_time_seconds:.0f}s reached "
                    f"max {exit_config.max_hold_seconds}s"
                ),
                indicators={
                    "holding_time_seconds": context.holding_time_seconds,
                    "max_hold_seconds": exit_config.max_hold_seconds,
                },
            )
        )

    if _detect_sparse_ambiguity(
        context,
        stop_loss_pct=exit_config.stop_loss_pct,
        take_profit_pct=exit_config.take_profit_pct,
    ):
        warnings.append(
            f"{context.coin}: price may have crossed both stop-loss and take-profit "
            "between observations; applying STOP_LOSS priority (conservative rule)"
        )

    return triggered, warnings


def _signal_exit_candidate(context: PositionExitContext) -> _ExitCandidate | None:
    signal = context.current_signal
    if signal in SELL_SIGNALS:
        return _ExitCandidate(
            exit_type=EXIT_TYPE_SIGNAL_EXIT,
            reason="take-profit sell signal after short-term rise",
            indicators={"change_1m": context.change_1m, "signal": signal},
        )
    return None


def _select_highest_priority(
    candidates: list[_ExitCandidate],
) -> _ExitCandidate | None:
    if not candidates:
        return None
    priority_index = {name: idx for idx, name in enumerate(EXIT_PRIORITY)}
    return min(candidates, key=lambda c: priority_index.get(c.exit_type, 999))


def evaluate_position_exit(
    context: PositionExitContext,
    exit_config: ExitConfig,
    *,
    strategy_name: str,
    strategy_version: str,
) -> StrategyExitResult:
    """
    Evaluate all exit conditions and return the highest-priority action.

    When multiple conditions trigger on the same timestamp, all are recorded
    in triggered_conditions but only the highest-priority exit executes.
    """
    _assert_no_future_history(context.price_history, context.coin, context.now)

    candidates: list[_ExitCandidate] = []
    warnings: list[str] = []

    if exit_config.exit_mode == EXIT_MODE_COST_AWARE:
        cost_candidates, cost_warnings = _collect_cost_aware_candidates(context, exit_config)
        candidates.extend(cost_candidates)
        warnings.extend(cost_warnings)

    if exit_config.signal_exit_enabled or exit_config.exit_mode == EXIT_MODE_SIGNAL_V1:
        signal_candidate = _signal_exit_candidate(context)
        if signal_candidate is not None:
            candidates.append(signal_candidate)

    triggered_types = [c.exit_type for c in candidates]
    selected = _select_highest_priority(candidates)

    base_indicators: dict[str, Any] = {
        "gross_return_pct": context.gross_return_pct,
        "estimated_net_return_pct": context.estimated_net_return_pct,
        "holding_time_seconds": context.holding_time_seconds,
        "triggered_conditions": triggered_types,
    }

    if selected is None:
        return StrategyExitResult(
            action="HOLD",
            reason="no exit condition met",
            exit_type=EXIT_TYPE_HOLD,
            gross_return_pct=context.gross_return_pct,
            estimated_net_return_pct=context.estimated_net_return_pct,
            estimated_fee=context.estimated_sell_fee,
            estimated_slippage=context.estimated_sell_slippage,
            holding_time_seconds=context.holding_time_seconds,
            indicators={**base_indicators, **{"change_1m": context.change_1m}},
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            triggered_conditions=triggered_types,
            ambiguity_warnings=warnings,
        )

    return StrategyExitResult(
        action="SELL",
        reason=selected.reason,
        exit_type=selected.exit_type,
        gross_return_pct=context.gross_return_pct,
        estimated_net_return_pct=context.estimated_net_return_pct,
        estimated_fee=context.estimated_sell_fee,
        estimated_slippage=context.estimated_sell_slippage,
        holding_time_seconds=context.holding_time_seconds,
        indicators={**base_indicators, **selected.indicators},
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        triggered_conditions=triggered_types,
        ambiguity_warnings=warnings,
    )


def should_skip_signal_sell(exit_config: ExitConfig | None) -> bool:
    """Cost-aware profiles disable signal-only sells unless explicitly enabled."""
    if exit_config is None:
        return False
    if exit_config.exit_mode == EXIT_MODE_SIGNAL_V1:
        return False
    return not exit_config.signal_exit_enabled


def entry_signal_from_context(context: PositionExitContext) -> str | None:
    """Return a tradable buy signal from current context, if any."""
    signal = context.current_signal
    if signal in BUY_SIGNALS:
        return signal
    return None
