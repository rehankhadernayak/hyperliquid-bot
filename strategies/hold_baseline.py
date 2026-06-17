"""Hold baseline strategy — never generates tradable signals."""

from __future__ import annotations

from strategy_types import StrategyContext, StrategySignalResult


class HoldBaselineStrategy:
    """Baseline that always returns FLAT (no simulated trades)."""

    name = "hold_baseline"
    version = "1.0.0"

    def evaluate(self, context: StrategyContext) -> StrategySignalResult:
        return StrategySignalResult(
            signal="FLAT",
            signal_strength="WEAK",
            reason="hold baseline — no trades",
            indicators={"change_1m": None},
            strategy_name=self.name,
            strategy_version=self.version,
        )
