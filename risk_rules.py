"""
Shared buy/sell eligibility checks for paper trading.

Used by the read-only decision engine and the paper trader so recommendations
and simulated trades always agree on risk blocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from config import PAPER_TRADING

if TYPE_CHECKING:
    from paper_trader import PaperTrader, Position


class RiskCheckTrader(Protocol):
    """Minimal trader surface required for risk eligibility checks."""

    cash_balance: float
    is_paused: bool
    successful_trades_this_session: int
    positions: dict[str, Position]

    def total_open_exposure(self) -> float: ...

    def session_trade_limit(self) -> int: ...


def _get_position(trader: RiskCheckTrader, coin: str) -> Position:
    from paper_trader import Position

    return trader.positions.get(coin, Position())


def session_trade_block_reason(
    trader: RiskCheckTrader,
    coin: str,
    action: str,
) -> str | None:
    """Return a reason if session trade limits block the action."""
    if trader.successful_trades_this_session < trader.session_trade_limit():
        return None

    if action == "SELL" and not _get_position(trader, coin).is_open:
        return None

    return "session trade limit reached"


def buy_block_reason(trader: RiskCheckTrader, coin: str, now: float) -> str | None:
    """Return a reason if a PAPER BUY would be blocked, else None."""
    if trader.is_paused:
        return "equity protection triggered"

    session_reason = session_trade_block_reason(trader, coin, "BUY")
    if session_reason is not None:
        return session_reason

    position = _get_position(trader, coin)

    if trader.cash_balance < PAPER_TRADING.trade_size_usdc:
        return "insufficient fake cash"

    if (
        position.allocation_usdc + PAPER_TRADING.trade_size_usdc
        > PAPER_TRADING.max_allocation_per_coin_usdc
    ):
        return "per-coin allocation reached"

    if position.entry_count >= PAPER_TRADING.max_entries_per_coin:
        return "maximum entries reached"

    if position.last_buy_time is not None:
        seconds_since_last_buy = now - position.last_buy_time
        if seconds_since_last_buy < PAPER_TRADING.min_seconds_between_buys:
            return "cooldown active"

    if (
        trader.total_open_exposure() + PAPER_TRADING.trade_size_usdc
        > PAPER_TRADING.max_total_open_exposure_usdc
    ):
        return "total exposure reached"

    return None


def sell_block_reason(trader: RiskCheckTrader, coin: str) -> str | None:
    """Return a reason if a PAPER SELL would be blocked, else None."""
    if trader.is_paused:
        return "equity protection triggered"

    return session_trade_block_reason(trader, coin, "SELL")
