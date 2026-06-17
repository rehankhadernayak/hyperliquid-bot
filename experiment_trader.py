"""
Isolated paper trader for strategy experiments.

Never writes to live CSV files. Applies fee, slippage, and risk limits from
an immutable strategy profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import MARKET
from paper_trader import BUY_SIGNALS, SELL_SIGNALS, PaperTrader
from profile_risk_rules import buy_block_reason, sell_block_reason
from strategy_profiles import StrategyProfile
from strategy_types import ExperimentRunState, ProfileTradeRecord, TradingLimits


@dataclass
class ExperimentPaperTrader(PaperTrader):
    """Profile-aware trader used only during offline experiments."""

    profile: StrategyProfile | None = None
    limits: TradingLimits | None = None
    trade_log: list[ProfileTradeRecord] = field(default_factory=list)
    run_state: ExperimentRunState = field(default_factory=ExperimentRunState)
    total_slippage_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.profile is not None and self.limits is None:
            self.limits = self.profile.trading_limits()
        if self.limits is not None:
            self.cash_balance = self.limits.starting_balance_usdc

    def session_trade_limit(self) -> int:
        assert self.limits is not None
        return self.limits.max_successful_trades_per_session

    def execution_price(self, action: str, mid_price: float) -> float:
        assert self.limits is not None
        slip = self.limits.slippage_pct / 100.0
        if action == "BUY":
            return mid_price * (1 + slip)
        return mid_price * (1 - slip)

    def buy_fee(self, trade_value: float) -> float:
        assert self.limits is not None
        fee = trade_value * (self.limits.trading_fee_pct / 100.0)
        self.total_fees_paid += fee
        return fee

    def sell_fee(self, trade_value: float) -> float:
        assert self.limits is not None
        fee = trade_value * (self.limits.trading_fee_pct / 100.0)
        self.total_fees_paid += fee
        return fee

    def slippage_cost(self, action: str, mid_price: float, quantity: float) -> float:
        execution_price = self.execution_price(action, mid_price)
        if action == "BUY":
            return max((execution_price - mid_price) * quantity, 0.0)
        return max((mid_price - execution_price) * quantity, 0.0)

    def update_session_risk(self, current_prices: dict[str, float]) -> None:
        assert self.limits is not None
        if self.is_paused:
            return

        equity = self.total_equity(current_prices)
        if equity < self.limits.min_equity_before_pause_usdc:
            self.is_paused = True

    def record_exposure_tick(self, current_prices: dict[str, float]) -> None:
        self.run_state.total_timestamps += 1
        if self.total_open_exposure() > 0:
            self.run_state.exposure_timestamps += 1

    def process_signal(
        self,
        coin: str,
        signal: str,
        price: float,
        timestamp: str,
        now: float,
    ) -> None:
        if coin not in MARKET.watchlist:
            return

        if self.is_paused:
            return

        if signal in BUY_SIGNALS:
            self._profile_buy(coin, price, timestamp, signal, now)
        elif signal in SELL_SIGNALS:
            self._profile_sell(coin, price, timestamp, signal, now)

    def _profile_buy(
        self,
        coin: str,
        price: float,
        timestamp: str,
        signal: str,
        now: float,
    ) -> None:
        assert self.limits is not None
        block_reason = buy_block_reason(self, coin, now, self.limits)
        if block_reason is not None:
            return

        execution_price = self.execution_price("BUY", price)
        quantity = self.limits.trade_size_usdc / execution_price
        position = self._get_position(coin)

        total_cost = (
            (position.avg_entry_price * position.quantity) + self.limits.trade_size_usdc
        )
        position.quantity += quantity
        position.avg_entry_price = total_cost / position.quantity
        position.entry_count += 1
        position.last_buy_time = now

        buy_fee = self.buy_fee(self.limits.trade_size_usdc)
        slippage = self.slippage_cost("BUY", price, quantity)
        self.total_slippage_cost += slippage
        self.cash_balance -= self.limits.trade_size_usdc + buy_fee

        if coin not in self.run_state.open_position_ticks:
            self.run_state.open_position_ticks[coin] = []
        self.run_state.open_position_ticks[coin].append((now, quantity))

        self._append_trade(
            timestamp=timestamp,
            coin=coin,
            action="BUY",
            signal=signal,
            mid_price=price,
            execution_price=execution_price,
            quantity=quantity,
            trade_value=self.limits.trade_size_usdc,
            fee=buy_fee,
            slippage=slippage,
            realized_pnl=0.0,
            entry_count=position.entry_count,
        )
        self.successful_trades_this_session += 1

    def _profile_sell(
        self,
        coin: str,
        price: float,
        timestamp: str,
        signal: str,
        now: float,
    ) -> None:
        assert self.limits is not None
        position = self._get_position(coin)

        if not position.is_open:
            return

        block_reason = sell_block_reason(self, coin, self.limits)
        if block_reason is not None:
            return

        execution_price = self.execution_price("SELL", price)
        position_value = position.quantity * execution_price

        if position_value <= self.limits.trade_size_usdc:
            quantity = position.quantity
            trade_value = position_value
            entry_count = position.entry_count
        else:
            trade_value = self.limits.trade_size_usdc
            quantity = trade_value / execution_price
            entry_count = position.entry_count

            remaining_value = (position.quantity - quantity) * execution_price
            if remaining_value < self.limits.trade_size_usdc:
                quantity = position.quantity
                trade_value = position_value
                entry_count = position.entry_count

        trade_realized_pnl = (execution_price - position.avg_entry_price) * quantity
        sell_fee = self.sell_fee(trade_value)
        slippage = self.slippage_cost("SELL", price, quantity)
        self.total_slippage_cost += slippage
        self.realized_pnl += trade_realized_pnl
        self.cash_balance += trade_value - sell_fee

        position.quantity -= quantity
        if position.quantity <= 0 or position.quantity * execution_price < 0.01:
            position.quantity = 0.0
            position.avg_entry_price = 0.0
            position.entry_count = 0
            position.last_buy_time = None

        self._record_holding_time(coin, now, quantity)

        self._append_trade(
            timestamp=timestamp,
            coin=coin,
            action="SELL",
            signal=signal,
            mid_price=price,
            execution_price=execution_price,
            quantity=quantity,
            trade_value=trade_value,
            fee=sell_fee,
            slippage=slippage,
            realized_pnl=trade_realized_pnl,
            entry_count=entry_count,
        )
        self.successful_trades_this_session += 1

    def _record_holding_time(self, coin: str, now: float, quantity: float) -> None:
        open_ticks = self.run_state.open_position_ticks.get(coin, [])
        remaining = quantity
        while remaining > 0 and open_ticks:
            entry_time, entry_qty = open_ticks[0]
            matched = min(entry_qty, remaining)
            self.run_state.completed_holding_seconds.append(now - entry_time)
            remaining -= matched
            if matched >= entry_qty:
                open_ticks.pop(0)
            else:
                open_ticks[0] = (entry_time, entry_qty - matched)

    def _append_trade(
        self,
        *,
        timestamp: str,
        coin: str,
        action: str,
        signal: str,
        mid_price: float,
        execution_price: float,
        quantity: float,
        trade_value: float,
        fee: float,
        slippage: float,
        realized_pnl: float,
        entry_count: int,
    ) -> None:
        current_prices = {coin: mid_price}
        self.trade_log.append(
            ProfileTradeRecord(
                timestamp=timestamp,
                coin=coin,
                action=action,
                signal=signal,
                mid_price=mid_price,
                execution_price=execution_price,
                quantity=quantity,
                trade_value=trade_value,
                fee=fee,
                slippage=slippage,
                realized_pnl=realized_pnl,
                entry_count=entry_count,
                cash_balance=self.cash_balance,
                open_exposure=self.total_open_exposure(),
                equity=self.total_equity(current_prices),
            )
        )

    def force_close_all_positions(
        self, current_prices: dict[str, float], timestamp: str
    ) -> None:
        for coin in MARKET.watchlist:
            position = self._get_position(coin)
            if not position.is_open:
                continue

            price = current_prices.get(coin)
            if price is None:
                continue

            self._force_sell_full_position(coin, price, timestamp)

    def _force_sell_full_position(
        self, coin: str, price: float, timestamp: str
    ) -> None:
        assert self.limits is not None
        position = self._get_position(coin)
        if not position.is_open:
            return

        execution_price = self.execution_price("SELL", price)
        quantity = position.quantity
        trade_value = quantity * execution_price
        trade_realized_pnl = (execution_price - position.avg_entry_price) * quantity
        sell_fee = self.sell_fee(trade_value)
        slippage = self.slippage_cost("SELL", price, quantity)
        self.total_slippage_cost += slippage

        self.realized_pnl += trade_realized_pnl
        self.cash_balance += trade_value - sell_fee

        self._record_holding_time(coin, 0.0, quantity)

        position.quantity = 0.0
        position.avg_entry_price = 0.0
        position.entry_count = 0
        position.last_buy_time = None

        self._append_trade(
            timestamp=timestamp,
            coin=coin,
            action="SELL",
            signal="FORCED_CLOSE",
            mid_price=price,
            execution_price=execution_price,
            quantity=quantity,
            trade_value=trade_value,
            fee=sell_fee,
            slippage=slippage,
            realized_pnl=trade_realized_pnl,
            entry_count=0,
        )
