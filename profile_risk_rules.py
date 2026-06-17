"""
Profile-aware risk eligibility checks for experiment paper trading.

Mirrors risk_rules.py but uses limits from an immutable strategy profile
instead of global PAPER_TRADING settings.
"""

from __future__ import annotations

from paper_trader import Position
from strategy_types import TradingLimits


def _get_position(trader, coin: str) -> Position:
    return trader.positions.get(coin, Position())


def session_trade_block_reason(
    trader,
    coin: str,
    action: str,
    limits: TradingLimits,
) -> str | None:
    if trader.successful_trades_this_session < limits.max_successful_trades_per_session:
        return None

    if action == "SELL" and not _get_position(trader, coin).is_open:
        return None

    return "session trade limit reached"


def buy_block_reason(
    trader,
    coin: str,
    now: float,
    limits: TradingLimits,
) -> str | None:
    if trader.is_paused:
        return "equity protection triggered"

    session_reason = session_trade_block_reason(trader, coin, "BUY", limits)
    if session_reason is not None:
        return session_reason

    position = _get_position(trader, coin)

    if trader.cash_balance < limits.trade_size_usdc:
        return "insufficient fake cash"

    if (
        position.allocation_usdc + limits.trade_size_usdc
        > limits.max_allocation_per_coin_usdc
    ):
        return "per-coin allocation reached"

    if position.entry_count >= limits.max_entries_per_coin:
        return "maximum entries reached"

    if position.last_buy_time is not None:
        seconds_since_last_buy = now - position.last_buy_time
        if seconds_since_last_buy < limits.min_seconds_between_buys:
            return "cooldown active"

    if (
        trader.total_open_exposure() + limits.trade_size_usdc
        > limits.max_total_open_exposure_usdc
    ):
        return "total exposure reached"

    return None


def sell_block_reason(
    trader,
    coin: str,
    limits: TradingLimits,
) -> str | None:
    if trader.is_paused:
        return "equity protection triggered"

    return session_trade_block_reason(trader, coin, "SELL", limits)
