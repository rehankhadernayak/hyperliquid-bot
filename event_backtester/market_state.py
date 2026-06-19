"""Per-coin market state maintained from replay events (no future leakage)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from market_data.event_models import BboEvent, TradeEvent
from market_data.replay_events import ReplayEvent
from market_analyzer import PriceHistory, record_price_history


@dataclass
class CoinMarketState:
    """Latest known market state for one coin."""

    coin: str
    latest_trade_price: float | None = None
    latest_trade_timestamp_ms: int | None = None
    latest_bbo: BboEvent | None = None
    best_bid_price: float | None = None
    best_bid_size: float | None = None
    best_ask_price: float | None = None
    best_ask_size: float | None = None
    midpoint: float | None = None
    spread: float | None = None
    spread_bps: float | None = None
    last_update_timestamp_ms: int | None = None
    bbo_update_timestamp_ms: int | None = None
    recent_trades: list[dict[str, Any]] = field(default_factory=list)
    max_recent_trades: int = 50

    def to_dict(self) -> dict[str, Any]:
        return {
            "coin": self.coin,
            "latest_trade_price": self.latest_trade_price,
            "latest_trade_timestamp_ms": self.latest_trade_timestamp_ms,
            "best_bid_price": self.best_bid_price,
            "best_bid_size": self.best_bid_size,
            "best_ask_price": self.best_ask_price,
            "best_ask_size": self.best_ask_size,
            "midpoint": self.midpoint,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "last_update_timestamp_ms": self.last_update_timestamp_ms,
            "bbo_update_timestamp_ms": self.bbo_update_timestamp_ms,
            "recent_trade_count": len(self.recent_trades),
        }


class MarketStateManager:
    """Maintains per-coin state from replay cursor events only."""

    def __init__(self, coins: tuple[str, ...]) -> None:
        self._states: dict[str, CoinMarketState] = {
            coin: CoinMarketState(coin=coin) for coin in coins
        }
        self._price_history: PriceHistory = {}
        self._mid_history: PriceHistory = {}

    @property
    def price_history(self) -> PriceHistory:
        return self._price_history

    @property
    def mid_history(self) -> PriceHistory:
        return self._mid_history

    def get_state(self, coin: str) -> CoinMarketState | None:
        return self._states.get(coin)

    def all_states(self) -> dict[str, CoinMarketState]:
        return dict(self._states)

    def update_from_event(self, event: ReplayEvent) -> bool:
        """Update state from one replay event. Returns True if coin state changed."""
        coin = event.coin
        if not coin or coin not in self._states:
            return False

        state = self._states[coin]
        if event.original_event_type == "trade":
            return self._update_trade(state, event)
        if event.original_event_type == "bbo":
            return self._update_bbo(state, event)
        return False

    def _update_trade(self, state: CoinMarketState, event: ReplayEvent) -> bool:
        trade: TradeEvent = event.event
        now = event.primary_time_ms / 1000.0
        state.latest_trade_price = trade.price
        state.latest_trade_timestamp_ms = event.primary_time_ms
        state.last_update_timestamp_ms = event.primary_time_ms
        record_price_history(self._price_history, state.coin, trade.price, now)
        state.recent_trades.append(
            {
                "price": trade.price,
                "size": trade.size,
                "side": trade.side,
                "timestamp_ms": event.primary_time_ms,
                "trade_id": trade.trade_id,
            }
        )
        if len(state.recent_trades) > state.max_recent_trades:
            state.recent_trades = state.recent_trades[-state.max_recent_trades :]
        return True

    def _update_bbo(self, state: CoinMarketState, event: ReplayEvent) -> bool:
        bbo: BboEvent = event.event
        state.latest_bbo = bbo
        state.best_bid_price = bbo.best_bid_price
        state.best_bid_size = bbo.best_bid_size
        state.best_ask_price = bbo.best_ask_price
        state.best_ask_size = bbo.best_ask_size
        state.midpoint = (bbo.best_bid_price + bbo.best_ask_price) / 2.0
        state.spread = bbo.spread
        state.spread_bps = bbo.spread_bps
        state.bbo_update_timestamp_ms = event.primary_time_ms
        state.last_update_timestamp_ms = event.primary_time_ms
        now = event.primary_time_ms / 1000.0
        if state.midpoint is not None:
            record_price_history(self._mid_history, state.coin, state.midpoint, now)
        return True

    def bbo_available(self, coin: str, event_time_ms: int, stale_ms: int) -> bool:
        state = self._states.get(coin)
        if state is None or state.bbo_update_timestamp_ms is None:
            return False
        if state.best_bid_price is None or state.best_ask_price is None:
            return False
        age = event_time_ms - state.bbo_update_timestamp_ms
        return 0 <= age <= stale_ms

    def bbo_stale_reason(self, coin: str, event_time_ms: int, stale_ms: int) -> str | None:
        state = self._states.get(coin)
        if state is None or state.bbo_update_timestamp_ms is None:
            return "missing_or_stale_bbo"
        if state.best_bid_price is None or state.best_ask_price is None:
            return "missing_or_stale_bbo"
        age = event_time_ms - state.bbo_update_timestamp_ms
        if age < 0 or age > stale_ms:
            return "missing_or_stale_bbo"
        return None

    def current_prices(self) -> dict[str, float]:
        """Best available price per coin (trade price, else midpoint)."""
        prices: dict[str, float] = {}
        for coin, state in self._states.items():
            if state.latest_trade_price is not None:
                prices[coin] = state.latest_trade_price
            elif state.midpoint is not None:
                prices[coin] = state.midpoint
        return prices
