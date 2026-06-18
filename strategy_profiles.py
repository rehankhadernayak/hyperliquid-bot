"""
Immutable strategy profiles for offline experiments.

Profiles are frozen at load time and must not be modified while an experiment
is running. The full profile snapshot is saved with every experiment result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from config import PAPER_TRADING, SIGNAL, SignalThresholds
from exit_config import ExitConfig, SIGNAL_EXIT_V1
from market_analyzer import WINDOW_1M_SECONDS
from profile_validation import validate_profile_fields, validate_thresholds
from strategy_types import TradingLimits


@dataclass(frozen=True)
class StrategyProfile:
    """Named, immutable configuration for one strategy experiment arm."""

    profile_id: str
    strategy_name: str
    strategy_version: str
    thresholds: SignalThresholds
    lookback_seconds: int
    cooldown_seconds: int
    trade_size_usdc: float
    trading_fee_pct: float
    slippage_pct: float
    starting_balance_usdc: float
    max_allocation_per_coin_usdc: float
    max_entries_per_coin: int
    max_total_open_exposure_usdc: float
    max_successful_trades_per_session: int
    min_equity_before_pause_usdc: float
    close_positions_at_end: bool = True
    exit_config: ExitConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable profile snapshot."""
        payload = asdict(self)
        payload["thresholds"] = asdict(self.thresholds)
        if self.exit_config is not None:
            payload["exit_config"] = self.exit_config.to_dict()
        else:
            payload["exit_config"] = None
        return payload

    def trading_limits(self) -> TradingLimits:
        """Return execution limits used by the experiment paper trader."""
        return TradingLimits(
            starting_balance_usdc=self.starting_balance_usdc,
            trade_size_usdc=self.trade_size_usdc,
            max_allocation_per_coin_usdc=self.max_allocation_per_coin_usdc,
            max_entries_per_coin=self.max_entries_per_coin,
            min_seconds_between_buys=self.cooldown_seconds,
            max_total_open_exposure_usdc=self.max_total_open_exposure_usdc,
            max_successful_trades_per_session=self.max_successful_trades_per_session,
            min_equity_before_pause_usdc=self.min_equity_before_pause_usdc,
            trading_fee_pct=self.trading_fee_pct,
            slippage_pct=self.slippage_pct,
            close_positions_at_end=self.close_positions_at_end,
        )


def _base_limits() -> dict[str, float | int]:
    return {
        "starting_balance_usdc": PAPER_TRADING.starting_balance_usdc,
        "trade_size_usdc": PAPER_TRADING.trade_size_usdc,
        "max_allocation_per_coin_usdc": PAPER_TRADING.max_allocation_per_coin_usdc,
        "max_entries_per_coin": PAPER_TRADING.max_entries_per_coin,
        "max_total_open_exposure_usdc": PAPER_TRADING.max_total_open_exposure_usdc,
        "max_successful_trades_per_session": 1000,
        "min_equity_before_pause_usdc": PAPER_TRADING.min_equity_before_pause_usdc,
        "trading_fee_pct": PAPER_TRADING.trading_fee_pct,
        "slippage_pct": PAPER_TRADING.slippage_pct,
        "lookback_seconds": WINDOW_1M_SECONDS,
        "cooldown_seconds": PAPER_TRADING.min_seconds_between_buys,
        "close_positions_at_end": True,
    }


STRATEGY_PROFILES: dict[str, StrategyProfile] = {
    "contrarian_v1_test": StrategyProfile(
        profile_id="contrarian_v1_test",
        strategy_name="contrarian_v1",
        strategy_version="1.0.0",
        thresholds=SIGNAL.test_mode,
        **_base_limits(),
    ),
    "contrarian_v1_production": StrategyProfile(
        profile_id="contrarian_v1_production",
        strategy_name="contrarian_v1",
        strategy_version="1.0.0",
        thresholds=SIGNAL.production,
        **_base_limits(),
    ),
    "hold_baseline": StrategyProfile(
        profile_id="hold_baseline",
        strategy_name="hold_baseline",
        strategy_version="1.0.0",
        thresholds=SIGNAL.production,
        **_base_limits(),
    ),
    "contrarian_v1_entry_medium": StrategyProfile(
        profile_id="contrarian_v1_entry_medium",
        strategy_name="contrarian_v1",
        strategy_version="1.0.0",
        thresholds=SignalThresholds(
            strong_up_pct=SIGNAL.test_mode.strong_up_pct,
            up_pct=SIGNAL.test_mode.up_pct,
            strong_down_pct=-0.25,
            down_pct=-0.15,
        ),
        **_base_limits(),
    ),
    "contrarian_v1_entry_selective": StrategyProfile(
        profile_id="contrarian_v1_entry_selective",
        strategy_name="contrarian_v1",
        strategy_version="1.0.0",
        thresholds=SignalThresholds(
            strong_up_pct=SIGNAL.test_mode.strong_up_pct,
            up_pct=SIGNAL.test_mode.up_pct,
            strong_down_pct=-0.40,
            down_pct=-0.25,
        ),
        **_base_limits(),
    ),
    "contrarian_v1_size_300_test": StrategyProfile(
        profile_id="contrarian_v1_size_300_test",
        strategy_name="contrarian_v1",
        strategy_version="1.0.0",
        thresholds=SIGNAL.test_mode,
        starting_balance_usdc=1000.0,
        trade_size_usdc=300.0,
        max_allocation_per_coin_usdc=300.0,
        max_entries_per_coin=1,
        max_total_open_exposure_usdc=600.0,
        max_successful_trades_per_session=PAPER_TRADING.max_successful_trades_per_session,
        min_equity_before_pause_usdc=PAPER_TRADING.min_equity_before_pause_usdc,
        trading_fee_pct=PAPER_TRADING.trading_fee_pct,
        slippage_pct=PAPER_TRADING.slippage_pct,
        lookback_seconds=WINDOW_1M_SECONDS,
        cooldown_seconds=PAPER_TRADING.min_seconds_between_buys,
        close_positions_at_end=True,
    ),
}

_MEDIUM_ENTRY_THRESHOLDS = SignalThresholds(
    strong_up_pct=SIGNAL.test_mode.strong_up_pct,
    up_pct=SIGNAL.test_mode.up_pct,
    strong_down_pct=-0.25,
    down_pct=-0.15,
)

_EXIT_PROFILE_LIMITS = _base_limits()

STRATEGY_PROFILES.update(
    {
        "contrarian_v1_medium_signal_exit": StrategyProfile(
            profile_id="contrarian_v1_medium_signal_exit",
            strategy_name="contrarian_v1",
            strategy_version="1.0.0",
            thresholds=_MEDIUM_ENTRY_THRESHOLDS,
            exit_config=SIGNAL_EXIT_V1,
            **_EXIT_PROFILE_LIMITS,
        ),
        "contrarian_v1_medium_tp25_sl25": StrategyProfile(
            profile_id="contrarian_v1_medium_tp25_sl25",
            strategy_name="contrarian_v1",
            strategy_version="1.0.0",
            thresholds=_MEDIUM_ENTRY_THRESHOLDS,
            exit_config=ExitConfig(
                exit_mode="cost_aware",
                take_profit_pct=0.25,
                stop_loss_pct=0.25,
                max_hold_seconds=None,
                signal_exit_enabled=False,
            ),
            **_EXIT_PROFILE_LIMITS,
        ),
        "contrarian_v1_medium_tp35_sl25": StrategyProfile(
            profile_id="contrarian_v1_medium_tp35_sl25",
            strategy_name="contrarian_v1",
            strategy_version="1.0.0",
            thresholds=_MEDIUM_ENTRY_THRESHOLDS,
            exit_config=ExitConfig(
                exit_mode="cost_aware",
                take_profit_pct=0.35,
                stop_loss_pct=0.25,
                max_hold_seconds=None,
                signal_exit_enabled=False,
            ),
            **_EXIT_PROFILE_LIMITS,
        ),
        "contrarian_v1_medium_tp50_sl30": StrategyProfile(
            profile_id="contrarian_v1_medium_tp50_sl30",
            strategy_name="contrarian_v1",
            strategy_version="1.0.0",
            thresholds=_MEDIUM_ENTRY_THRESHOLDS,
            exit_config=ExitConfig(
                exit_mode="cost_aware",
                take_profit_pct=0.50,
                stop_loss_pct=0.30,
                max_hold_seconds=None,
                signal_exit_enabled=False,
            ),
            **_EXIT_PROFILE_LIMITS,
        ),
        "contrarian_v1_medium_tp35_sl25_maxhold": StrategyProfile(
            profile_id="contrarian_v1_medium_tp35_sl25_maxhold",
            strategy_name="contrarian_v1",
            strategy_version="1.0.0",
            thresholds=_MEDIUM_ENTRY_THRESHOLDS,
            exit_config=ExitConfig(
                exit_mode="cost_aware",
                take_profit_pct=0.35,
                stop_loss_pct=0.25,
                max_hold_seconds=1800,
                signal_exit_enabled=False,
            ),
            **_EXIT_PROFILE_LIMITS,
        ),
    }
)


def get_profile(profile_id: str) -> StrategyProfile:
    """Return an immutable profile by id."""
    if profile_id not in STRATEGY_PROFILES:
        known = ", ".join(sorted(STRATEGY_PROFILES))
        raise KeyError(f"Unknown profile {profile_id!r}. Known profiles: {known}")
    profile = STRATEGY_PROFILES[profile_id]
    validate_profile_fields(profile)
    return profile


def _validate_all_profiles() -> None:
    for profile in STRATEGY_PROFILES.values():
        validate_profile_fields(profile)


_validate_all_profiles()


def list_profile_ids() -> list[str]:
    """Return sorted profile ids."""
    return sorted(STRATEGY_PROFILES)
