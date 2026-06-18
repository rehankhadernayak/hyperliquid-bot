"""
Strategy interface types for offline paper-trading experiments.

No wallets, signing, or exchange execution — signal generation only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from config import SignalThresholds
from market_analyzer import PriceHistory
from paper_trader import PaperTrader

if TYPE_CHECKING:
    from strategy_profiles import StrategyProfile


@dataclass(frozen=True)
class StrategySignalResult:
    """Structured output from a strategy evaluation."""

    signal: str
    signal_strength: str
    reason: str
    indicators: dict[str, Any]
    strategy_name: str
    strategy_version: str


@dataclass(frozen=True)
class StrategyContext:
    """Inputs passed to a strategy on each evaluation tick."""

    coin: str
    current_price: float | None
    price_history: PriceHistory
    timestamp: str
    now: float
    trader: PaperTrader
    thresholds: SignalThresholds
    lookback_seconds: int


class Strategy(Protocol):
    """Reusable strategy interface for offline experiments."""

    name: str
    version: str

    def evaluate(self, context: StrategyContext) -> StrategySignalResult: ...


@dataclass(frozen=True)
class PositionExitContext:
    """Position and market state available for exit evaluation."""

    coin: str
    current_price: float
    avg_entry_price: float
    quantity: float
    position_value: float
    unrealized_pnl_before_fees: float
    gross_return_pct: float
    estimated_sell_fee: float
    estimated_sell_slippage: float
    total_estimated_round_trip_cost: float
    estimated_net_return_pct: float
    opened_at: float | None
    holding_time_seconds: float
    entry_count: int
    now: float
    price_history: PriceHistory
    trader_state: dict[str, Any]
    profile: StrategyProfile
    change_1m: float | None
    current_signal: str


@dataclass(frozen=True)
class StrategyExitResult:
    """Structured output from a position-aware exit evaluation."""

    action: str
    reason: str
    exit_type: str
    gross_return_pct: float
    estimated_net_return_pct: float
    estimated_fee: float
    estimated_slippage: float
    holding_time_seconds: float
    indicators: dict[str, Any]
    strategy_name: str
    strategy_version: str
    triggered_conditions: list[str] = field(default_factory=list)
    ambiguity_warnings: list[str] = field(default_factory=list)


class ExitStrategy(Protocol):
    """Strategy interface extension for position-aware exits."""

    name: str
    version: str

    def evaluate_exit(
        self, context: PositionExitContext, exit_config: Any
    ) -> StrategyExitResult: ...


def signal_strength_from_label(signal: str) -> str:
    """Map a signal name to a simple strength label."""
    if signal in ("STRONG_UP", "STRONG_DOWN"):
        return "STRONG"
    if signal in ("UP", "DOWN"):
        return "MODERATE"
    if signal == "FLAT":
        return "WEAK"
    return "NONE"


@dataclass(frozen=True)
class TradingLimits:
    """Execution and risk limits copied from an immutable strategy profile."""

    starting_balance_usdc: float
    trade_size_usdc: float
    max_allocation_per_coin_usdc: float
    max_entries_per_coin: int
    min_seconds_between_buys: int
    max_total_open_exposure_usdc: float
    max_successful_trades_per_session: int
    min_equity_before_pause_usdc: float
    trading_fee_pct: float
    slippage_pct: float
    close_positions_at_end: bool = True


@dataclass
class ProfileTradeRecord:
    """One simulated trade from an experiment run."""

    timestamp: str
    coin: str
    action: str
    signal: str
    mid_price: float
    execution_price: float
    quantity: float
    trade_value: float
    fee: float
    slippage: float
    realized_pnl: float
    entry_count: int
    cash_balance: float
    open_exposure: float
    equity: float
    exit_type: str | None = None
    gross_return_pct: float | None = None
    estimated_net_return_pct: float | None = None


@dataclass
class ExperimentRunState:
    """Mutable state tracked during one profile replay."""

    exposure_timestamps: int = 0
    total_timestamps: int = 0
    open_position_ticks: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    completed_holding_seconds: list[float] = field(default_factory=list)
    exit_ambiguity_warnings: list[str] = field(default_factory=list)
