"""Event-based strategy adapter for contrarian_v1 profiles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from event_backtester.event_portfolio import EventPortfolio
from event_backtester.market_state import CoinMarketState, MarketStateManager
from exit_config import EXIT_TYPE_SIGNAL_EXIT
from exit_evaluator import build_position_exit_context, evaluate_position_exit
from market_data.replay_events import ReplayEvent
from paper_trader import BUY_SIGNALS, SELL_SIGNALS
from strategies import get_strategy
from strategy_profiles import StrategyProfile
from strategy_types import StrategyContext, StrategySignalResult


@dataclass(frozen=True)
class EventStrategySignal:
    """Structured signal returned by the event strategy adapter."""

    signal: str
    signal_strength: str
    reason: str
    strategy_name: str
    strategy_version: str
    profile_name: str
    indicators: dict[str, Any]
    lookback_data_available: bool
    event_timestamp_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "signal_strength": self.signal_strength,
            "reason": self.reason,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "profile_name": self.profile_name,
            "indicators": self.indicators,
            "lookback_data_available": self.lookback_data_available,
            "event_timestamp_ms": self.event_timestamp_ms,
        }


def _ms_to_timestamp(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


class EventStrategyAdapter:
    """
    Event-compatible adapter for contrarian_v1_medium_no_trend_filter.

    Reuses ContrarianV1Strategy.evaluate() and profile thresholds from the
    existing row-based experiment framework. Differences from row-based replay:

    - Evaluates on trade events (recorded trade prices), not CSV timestamp rows.
    - Uses event primary_time_ms for lookback windows instead of 10s polling ticks.
    - No trend filter (profile has TREND_FILTER_DISABLED).
    - No rebound confirmation or cost-aware exit (not in this profile).
    - Fills are handled separately via BBO-aware execution simulator.
    """

    def __init__(self, profile: StrategyProfile) -> None:
        self._profile = profile
        self._strategy = get_strategy(profile.strategy_name)

    @property
    def profile(self) -> StrategyProfile:
        return self._profile

    def evaluate_on_trade(
        self,
        *,
        event: ReplayEvent,
        market_state: MarketStateManager,
        portfolio: EventPortfolio,
    ) -> EventStrategySignal | None:
        """Evaluate strategy on a trade event. Returns None if coin/price unavailable."""
        coin = event.coin
        if not coin:
            return None

        state = market_state.get_state(coin)
        if state is None or state.latest_trade_price is None:
            return None

        price = state.latest_trade_price
        now = event.primary_time_ms / 1000.0
        timestamp = _ms_to_timestamp(event.primary_time_ms)

        # Position-aware exit (signal-only) when profile has exit_config
        exit_config = self._profile.exit_config
        position = portfolio._get_position(coin)
        if exit_config is not None and position.is_open:
            exit_context = build_position_exit_context(
                coin=coin,
                current_price=price,
                avg_entry_price=position.avg_entry_price,
                quantity=position.quantity,
                opened_at=position.opened_at,
                entry_count=position.entry_count,
                now=now,
                price_history=market_state.price_history,
                trader_state={
                    "cash_balance": portfolio.cash_balance,
                    "realized_pnl": portfolio.realized_pnl,
                    "successful_trades_this_session": portfolio.successful_trades_this_session,
                    "is_paused": portfolio.is_paused,
                },
                profile=self._profile,
                lookback_seconds=self._profile.lookback_seconds,
                thresholds=self._profile.thresholds,
            )
            exit_result = evaluate_position_exit(
                exit_context,
                exit_config,
                strategy_name=self._profile.strategy_name,
                strategy_version=self._profile.strategy_version,
            )
            if exit_result.action == "SELL":
                return EventStrategySignal(
                    signal=exit_result.indicators.get("signal", "UP"),
                    signal_strength="MEDIUM",
                    reason=exit_result.reason,
                    strategy_name=self._profile.strategy_name,
                    strategy_version=self._profile.strategy_version,
                    profile_name=self._profile.profile_id,
                    indicators=dict(exit_result.indicators),
                    lookback_data_available=True,
                    event_timestamp_ms=event.primary_time_ms,
                )

        if exit_config is not None:
            from market_analyzer import rolling_percent_change

            history = market_state.price_history.get(coin, [])
            change = rolling_percent_change(
                history,
                price,
                now,
                self._profile.lookback_seconds,
            )
            if change is None:
                return None

        context = StrategyContext(
            coin=coin,
            current_price=price,
            price_history=market_state.price_history,
            timestamp=timestamp,
            now=now,
            trader=portfolio,  # type: ignore[arg-type]
            thresholds=self._profile.thresholds,
            lookback_seconds=self._profile.lookback_seconds,
        )
        result = self._strategy.evaluate(context)
        return self._to_event_signal(result, event.primary_time_ms)

    def _to_event_signal(
        self,
        result: StrategySignalResult,
        event_timestamp_ms: int,
    ) -> EventStrategySignal:
        lookback_available = result.indicators.get("change_1m") is not None
        return EventStrategySignal(
            signal=result.signal,
            signal_strength=result.signal_strength,
            reason=result.reason,
            strategy_name=result.strategy_name,
            strategy_version=result.strategy_version,
            profile_name=self._profile.profile_id,
            indicators=dict(result.indicators),
            lookback_data_available=lookback_available,
            event_timestamp_ms=event_timestamp_ms,
        )

    def action_for_signal(self, signal: EventStrategySignal) -> str | None:
        if signal.signal in BUY_SIGNALS:
            return "BUY"
        if signal.signal in SELL_SIGNALS:
            return "SELL"
        return None

    def adapter_notes(self) -> list[str]:
        return [
            "Reuses ContrarianV1Strategy.evaluate() with profile thresholds.",
            "Evaluates on trade events only; BBO events update execution state.",
            "No trend filter (contrarian_v1_medium_no_trend_filter).",
            "Signal-only exits via SIGNAL_EXIT_V1 when position is open.",
            "Fills use bid/ask from BBO, not midpoint/last price.",
        ]
