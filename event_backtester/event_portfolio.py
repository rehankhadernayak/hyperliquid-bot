"""Isolated portfolio simulation for event-driven backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from event_backtester.event_execution_simulator import ExecutionRequest, ExecutionResult
from paper_trader import Position
from profile_risk_rules import buy_block_reason
from strategy_profiles import StrategyProfile
from strategy_types import TradingLimits


def _ms_to_timestamp(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


@dataclass
class EventTradeRecord:
    timestamp: str
    timestamp_ms: int
    coin: str
    action: str
    signal: str
    fill_price: float
    quantity: float
    trade_value: float
    fee: float
    slippage: float
    trade_pnl: float
    cash_balance: float
    realized_pnl: float
    reference_bbo: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_ms": self.timestamp_ms,
            "coin": self.coin,
            "action": self.action,
            "signal": self.signal,
            "fill_price": self.fill_price,
            "quantity": self.quantity,
            "trade_value": self.trade_value,
            "fee": self.fee,
            "slippage": self.slippage,
            "trade_pnl": self.trade_pnl,
            "cash_balance": self.cash_balance,
            "realized_pnl": self.realized_pnl,
            "reference_bbo": self.reference_bbo,
        }


@dataclass
class BlockedSignalRecord:
    timestamp_ms: int
    coin: str
    signal: str
    side: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "coin": self.coin,
            "signal": self.signal,
            "side": self.side,
            "reason": self.reason,
        }


@dataclass
class EventPortfolio:
    """Simulated portfolio — no shorting, leverage, or margin in v1."""

    profile: StrategyProfile
    limits: TradingLimits
    cash_balance: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    total_slippage_cost: float = 0.0
    successful_trades_this_session: int = 0
    is_paused: bool = False
    trade_log: list[EventTradeRecord] = field(default_factory=list)
    blocked_signals: list[BlockedSignalRecord] = field(default_factory=list)
    events_processed: int = 0
    max_equity: float = 0.0
    exposure_timestamps: int = 0
    position_opened_at: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cash_balance = self.limits.starting_balance_usdc
        self.max_equity = self.cash_balance

    def _get_position(self, coin: str) -> Position:
        return self.positions.get(coin, Position())

    def total_open_exposure(self) -> float:
        return sum(pos.allocation_usdc for pos in self.positions.values() if pos.is_open)

    def unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        total = 0.0
        for coin, pos in self.positions.items():
            if not pos.is_open:
                continue
            price = current_prices.get(coin)
            if price is None:
                continue
            total += (price - pos.avg_entry_price) * pos.quantity
        return total

    def total_equity(self, current_prices: dict[str, float]) -> float:
        exposure = 0.0
        for coin, pos in self.positions.items():
            if not pos.is_open:
                continue
            price = current_prices.get(coin, pos.avg_entry_price)
            exposure += pos.quantity * price
        return self.cash_balance + exposure

    def update_equity_tracking(self, current_prices: dict[str, float]) -> None:
        equity = self.total_equity(current_prices)
        self.max_equity = max(self.max_equity, equity)
        if self.total_open_exposure() > 0:
            self.exposure_timestamps += 1

    def update_session_risk(self, current_prices: dict[str, float]) -> None:
        if self.is_paused:
            return
        equity = self.total_equity(current_prices)
        if equity < self.limits.min_equity_before_pause_usdc:
            self.is_paused = True

    def record_blocked(
        self,
        *,
        timestamp_ms: int,
        coin: str,
        signal: str,
        side: str,
        reason: str,
    ) -> None:
        self.blocked_signals.append(
            BlockedSignalRecord(
                timestamp_ms=timestamp_ms,
                coin=coin,
                signal=signal,
                side=side,
                reason=reason,
            )
        )

    def try_buy(
        self,
        *,
        coin: str,
        signal: str,
        execution: ExecutionResult,
        timestamp_ms: int,
    ) -> bool:
        if not execution.executed or execution.fill_price is None:
            reason = execution.blocked_reason or execution.reason
            self.record_blocked(
                timestamp_ms=timestamp_ms,
                coin=coin,
                signal=signal,
                side="BUY",
                reason=reason,
            )
            return False

        now = timestamp_ms / 1000.0
        block = buy_block_reason(self, coin, now, self.limits)
        if block is not None:
            self.record_blocked(
                timestamp_ms=timestamp_ms,
                coin=coin,
                signal=signal,
                side="BUY",
                reason=block,
            )
            return False

        fill_price = execution.fill_price
        quantity = self.limits.trade_size_usdc / fill_price
        trade_value = self.limits.trade_size_usdc
        total_cost = trade_value + execution.fee_amount

        if self.cash_balance < total_cost:
            self.record_blocked(
                timestamp_ms=timestamp_ms,
                coin=coin,
                signal=signal,
                side="BUY",
                reason="insufficient fake cash",
            )
            return False

        position = self._get_position(coin)
        combined_cost = (position.avg_entry_price * position.quantity) + trade_value
        new_quantity = position.quantity + quantity
        new_avg = combined_cost / new_quantity if new_quantity > 0 else fill_price

        self.cash_balance -= total_cost
        self.total_fees += execution.fee_amount
        self.total_slippage_cost += execution.slippage_amount
        self.successful_trades_this_session += 1

        if coin not in self.positions:
            self.positions[coin] = Position()
        pos = self.positions[coin]
        if pos.entry_count == 0:
            self.position_opened_at[coin] = now
        pos.quantity = new_quantity
        pos.avg_entry_price = new_avg
        pos.entry_count += 1
        pos.last_buy_time = now
        if pos.opened_at is None:
            pos.opened_at = now

        self.trade_log.append(
            EventTradeRecord(
                timestamp=_ms_to_timestamp(timestamp_ms),
                timestamp_ms=timestamp_ms,
                coin=coin,
                action="BUY",
                signal=signal,
                fill_price=fill_price,
                quantity=quantity,
                trade_value=trade_value,
                fee=execution.fee_amount,
                slippage=execution.slippage_amount,
                trade_pnl=0.0,
                cash_balance=self.cash_balance,
                realized_pnl=self.realized_pnl,
                reference_bbo=execution.reference_bbo,
            )
        )
        return True

    def try_sell(
        self,
        *,
        coin: str,
        signal: str,
        execution: ExecutionResult,
        timestamp_ms: int,
    ) -> bool:
        if not execution.executed or execution.fill_price is None:
            reason = execution.blocked_reason or execution.reason
            self.record_blocked(
                timestamp_ms=timestamp_ms,
                coin=coin,
                signal=signal,
                side="SELL",
                reason=reason,
            )
            return False

        position = self._get_position(coin)
        if not position.is_open:
            return False

        # Risk-reducing SELL: allow closing open positions even at session trade limit.
        block = None
        if self.is_paused:
            block = "equity protection triggered"
        if block is not None:
            self.record_blocked(
                timestamp_ms=timestamp_ms,
                coin=coin,
                signal=signal,
                side="SELL",
                reason=block,
            )
            return False

        fill_price = execution.fill_price
        quantity = position.quantity
        trade_value = quantity * fill_price
        fee = trade_value * (self.limits.trading_fee_pct / 100.0)
        slippage = execution.slippage_amount * quantity

        cost_basis = position.avg_entry_price * quantity
        trade_pnl = trade_value - fee - cost_basis

        self.cash_balance += trade_value - fee
        self.realized_pnl += trade_pnl
        self.total_fees += fee
        self.total_slippage_cost += slippage
        self.successful_trades_this_session += 1

        self.trade_log.append(
            EventTradeRecord(
                timestamp=_ms_to_timestamp(timestamp_ms),
                timestamp_ms=timestamp_ms,
                coin=coin,
                action="SELL",
                signal=signal,
                fill_price=fill_price,
                quantity=quantity,
                trade_value=trade_value,
                fee=fee,
                slippage=slippage,
                trade_pnl=trade_pnl,
                cash_balance=self.cash_balance,
                realized_pnl=self.realized_pnl,
                reference_bbo=execution.reference_bbo,
            )
        )

        self.positions[coin] = Position()
        self.position_opened_at.pop(coin, None)
        return True

    def drawdown(self, current_prices: dict[str, float]) -> float:
        equity = self.total_equity(current_prices)
        if self.max_equity <= 0:
            return 0.0
        return max(0.0, (self.max_equity - equity) / self.max_equity * 100.0)
