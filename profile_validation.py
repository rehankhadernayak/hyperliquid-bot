"""Validation for immutable strategy profile thresholds."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from config import SignalThresholds

from exit_config import ExitConfig, validate_exit_config


def validate_thresholds(thresholds: SignalThresholds) -> None:
    """
    Validate signal threshold ordering and numeric types.

    Rules:
      - strong_down_pct < down_pct < 0
      - 0 < up_pct < strong_up_pct
      - all values must be numeric (int or float, not bool)
    """
    for field in fields(thresholds):
        value = getattr(thresholds, field.name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Threshold {field.name!r} must be numeric, got {type(value).__name__}"
            )

    if not (thresholds.strong_down_pct < thresholds.down_pct < 0):
        raise ValueError(
            "Invalid down thresholds: require strong_down_pct < down_pct < 0 "
            f"(got strong_down_pct={thresholds.strong_down_pct}, "
            f"down_pct={thresholds.down_pct})"
        )

    if not (0 < thresholds.up_pct < thresholds.strong_up_pct):
        raise ValueError(
            "Invalid up thresholds: require 0 < up_pct < strong_up_pct "
            f"(got up_pct={thresholds.up_pct}, "
            f"strong_up_pct={thresholds.strong_up_pct})"
        )


def validate_profile_fields(profile: Any) -> None:
    """Validate threshold and numeric profile fields before experiment execution."""
    validate_thresholds(profile.thresholds)

    numeric_fields = (
        "trade_size_usdc",
        "trading_fee_pct",
        "slippage_pct",
        "starting_balance_usdc",
        "max_allocation_per_coin_usdc",
        "max_total_open_exposure_usdc",
        "min_equity_before_pause_usdc",
    )
    for name in numeric_fields:
        value = getattr(profile, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Profile field {name!r} must be numeric, got {type(value).__name__}"
            )

    integer_fields = (
        "lookback_seconds",
        "cooldown_seconds",
        "max_entries_per_coin",
        "max_successful_trades_per_session",
    )
    for name in integer_fields:
        value = getattr(profile, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"Profile field {name!r} must be an integer, got {type(value).__name__}"
            )

    if profile.exit_config is not None:
        validate_exit_config(profile.exit_config)
