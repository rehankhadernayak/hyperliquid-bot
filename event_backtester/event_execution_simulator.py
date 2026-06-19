"""BBO-aware simulated execution for event-driven backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from event_backtester.market_state import CoinMarketState
from strategy_profiles import StrategyProfile


@dataclass(frozen=True)
class ExecutionRequest:
    coin: str
    side: str
    signal: str
    event_timestamp_ms: int
    received_timestamp_ms: int
    reference_price: float | None


@dataclass(frozen=True)
class ExecutionResult:
    executed: bool
    blocked: bool
    blocked_reason: str | None
    requested_side: str
    fill_side: str | None
    reference_bbo: dict[str, Any] | None
    fill_price: float | None
    slippage_amount: float
    fee_amount: float
    event_timestamp_ms: int
    received_timestamp_ms: int
    reason: str
    quantity: float = 0.0
    trade_value: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed": self.executed,
            "blocked": self.blocked,
            "blocked_reason": self.blocked_reason,
            "requested_side": self.requested_side,
            "fill_side": self.fill_side,
            "reference_bbo": self.reference_bbo,
            "fill_price": self.fill_price,
            "slippage_amount": self.slippage_amount,
            "fee_amount": self.fee_amount,
            "event_timestamp_ms": self.event_timestamp_ms,
            "received_timestamp_ms": self.received_timestamp_ms,
            "reason": self.reason,
            "quantity": self.quantity,
            "trade_value": self.trade_value,
        }


class EventExecutionSimulator:
    """Simulate fills using best bid/ask from current market state."""

    FILL_MODEL = "bid_ask_v1"

    def __init__(
        self,
        profile: StrategyProfile,
        *,
        bbo_stale_ms: int,
        allow_midpoint_fills: bool = False,
    ) -> None:
        self._profile = profile
        self._limits = profile.trading_limits()
        self._bbo_stale_ms = bbo_stale_ms
        self._allow_midpoint_fills = allow_midpoint_fills

    @property
    def fill_model(self) -> str:
        return self.FILL_MODEL

    def cost_assumptions(self) -> dict[str, float | str]:
        return {
            "trading_fee_pct": self._limits.trading_fee_pct,
            "slippage_pct": self._limits.slippage_pct,
            "fill_model": self.FILL_MODEL,
            "allow_midpoint_fills": self._allow_midpoint_fills,
        }

    def simulate(
        self,
        request: ExecutionRequest,
        market_state: CoinMarketState,
    ) -> ExecutionResult:
        stale_reason = self._bbo_stale_reason(market_state, request.event_timestamp_ms)
        if stale_reason is not None and not self._allow_midpoint_fills:
            return ExecutionResult(
                executed=False,
                blocked=True,
                blocked_reason=stale_reason,
                requested_side=request.side,
                fill_side=None,
                reference_bbo=self._reference_bbo(market_state),
                fill_price=None,
                slippage_amount=0.0,
                fee_amount=0.0,
                event_timestamp_ms=request.event_timestamp_ms,
                received_timestamp_ms=request.received_timestamp_ms,
                reason=f"execution blocked: {stale_reason}",
            )

        if request.side == "BUY":
            return self._simulate_buy(request, market_state)
        return self._simulate_sell(request, market_state)

    def _bbo_stale_reason(self, state: CoinMarketState, event_time_ms: int) -> str | None:
        if state.bbo_update_timestamp_ms is None:
            return "missing_or_stale_bbo"
        if state.best_bid_price is None or state.best_ask_price is None:
            return "missing_or_stale_bbo"
        age = event_time_ms - state.bbo_update_timestamp_ms
        if age < 0 or age > self._bbo_stale_ms:
            return "missing_or_stale_bbo"
        return None

    def _reference_bbo(self, state: CoinMarketState) -> dict[str, Any] | None:
        if state.best_bid_price is None or state.best_ask_price is None:
            return None
        return {
            "best_bid_price": state.best_bid_price,
            "best_bid_size": state.best_bid_size,
            "best_ask_price": state.best_ask_price,
            "best_ask_size": state.best_ask_size,
            "midpoint": state.midpoint,
            "spread_bps": state.spread_bps,
            "bbo_update_timestamp_ms": state.bbo_update_timestamp_ms,
        }

    def _simulate_buy(
        self,
        request: ExecutionRequest,
        state: CoinMarketState,
    ) -> ExecutionResult:
        if self._allow_midpoint_fills and state.midpoint is not None:
            base_price = state.midpoint
            ref_label = "midpoint"
        else:
            base_price = state.best_ask_price
            ref_label = "best_ask"

        if base_price is None or base_price <= 0:
            return self._blocked(request, state, "missing_or_stale_bbo")

        slip_pct = self._limits.slippage_pct / 100.0
        fill_price = base_price * (1 + slip_pct)
        quantity = self._limits.trade_size_usdc / fill_price
        trade_value = quantity * fill_price
        slippage_amount = max((fill_price - base_price) * quantity, 0.0)
        fee_amount = trade_value * (self._limits.trading_fee_pct / 100.0)

        return ExecutionResult(
            executed=True,
            blocked=False,
            blocked_reason=None,
            requested_side="BUY",
            fill_side="BUY",
            reference_bbo=self._reference_bbo(state),
            fill_price=fill_price,
            slippage_amount=slippage_amount,
            fee_amount=fee_amount,
            event_timestamp_ms=request.event_timestamp_ms,
            received_timestamp_ms=request.received_timestamp_ms,
            reason=f"BUY fill at {ref_label} plus slippage",
            quantity=quantity,
            trade_value=trade_value,
        )

    def _simulate_sell(
        self,
        request: ExecutionRequest,
        state: CoinMarketState,
    ) -> ExecutionResult:
        if self._allow_midpoint_fills and state.midpoint is not None:
            base_price = state.midpoint
            ref_label = "midpoint"
        else:
            base_price = state.best_bid_price
            ref_label = "best_bid"

        if base_price is None or base_price <= 0:
            return self._blocked(request, state, "missing_or_stale_bbo")

        slip_pct = self._limits.slippage_pct / 100.0
        fill_price = base_price * (1 - slip_pct)
        quantity = request.reference_price  # placeholder; portfolio sets actual qty
        # Quantity is determined by portfolio; simulator only computes price/costs per unit
        slippage_per_unit = max(base_price - fill_price, 0.0)
        fee_pct = self._limits.trading_fee_pct / 100.0

        return ExecutionResult(
            executed=True,
            blocked=False,
            blocked_reason=None,
            requested_side="SELL",
            fill_side="SELL",
            reference_bbo=self._reference_bbo(state),
            fill_price=fill_price,
            slippage_amount=slippage_per_unit,
            fee_amount=0.0,  # portfolio applies fee on trade_value
            event_timestamp_ms=request.event_timestamp_ms,
            received_timestamp_ms=request.received_timestamp_ms,
            reason=f"SELL fill at {ref_label} minus slippage",
            quantity=0.0,
            trade_value=0.0,
        )

    def compute_sell_costs(self, fill_price: float, quantity: float) -> tuple[float, float]:
        trade_value = fill_price * quantity
        fee = trade_value * (self._limits.trading_fee_pct / 100.0)
        return trade_value, fee

    def _blocked(
        self,
        request: ExecutionRequest,
        state: CoinMarketState,
        reason: str,
    ) -> ExecutionResult:
        return ExecutionResult(
            executed=False,
            blocked=True,
            blocked_reason=reason,
            requested_side=request.side,
            fill_side=None,
            reference_bbo=self._reference_bbo(state),
            fill_price=None,
            slippage_amount=0.0,
            fee_amount=0.0,
            event_timestamp_ms=request.event_timestamp_ms,
            received_timestamp_ms=request.received_timestamp_ms,
            reason=f"execution blocked: {reason}",
        )
