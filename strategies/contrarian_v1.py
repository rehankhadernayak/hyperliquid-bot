"""
Contrarian percentage-change strategy (v1).

Generates BUY signals on short-term price declines and SELL signals on rises,
matching the existing live analyzer behavior when the same thresholds are used.
"""

from __future__ import annotations

from market_analyzer import WINDOW_1M_SECONDS, rolling_percent_change
from signals import signal_from_change
from strategy_types import StrategyContext, StrategySignalResult, signal_strength_from_label


class ContrarianV1Strategy:
    """Contrarian strategy based on rolling 1-minute percentage change."""

    name = "contrarian_v1"
    version = "1.0.0"

    def evaluate(self, context: StrategyContext) -> StrategySignalResult:
        if context.current_price is None:
            return StrategySignalResult(
                signal="N/A",
                signal_strength="NONE",
                reason="price data unavailable",
                indicators={"change_1m": None},
                strategy_name=self.name,
                strategy_version=self.version,
            )

        history = context.price_history.get(context.coin, [])
        lookback = context.lookback_seconds or WINDOW_1M_SECONDS
        change_1m = rolling_percent_change(
            history,
            context.current_price,
            context.now,
            lookback,
        )
        signal = signal_from_change(change_1m, context.thresholds)

        if signal == "N/A":
            reason = "not enough price history for lookback window"
        elif signal in ("DOWN", "STRONG_DOWN"):
            reason = "contrarian buy signal after short-term decline"
        elif signal in ("UP", "STRONG_UP"):
            reason = "take-profit sell signal after short-term rise"
        else:
            reason = "price change within flat band"

        return StrategySignalResult(
            signal=signal,
            signal_strength=signal_strength_from_label(signal),
            reason=reason,
            indicators={"change_1m": change_1m, "lookback_seconds": lookback},
            strategy_name=self.name,
            strategy_version=self.version,
        )
