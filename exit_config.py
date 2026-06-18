"""
Exit configuration for strategy experiment profiles.

Take-profit and stop-loss thresholds represent gross price movement from the
position's average simulated fill price (interpretation A). Reports show
expected and actual net return after fees and slippage separately.

Exit priority (deterministic, highest wins):
  1. STOP_LOSS
  2. TAKE_PROFIT
  3. MAX_HOLD
  4. SIGNAL_EXIT (when enabled)
  5. HOLD
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

EXIT_MODE_SIGNAL_V1 = "signal_exit_v1"
EXIT_MODE_COST_AWARE = "cost_aware"

EXIT_TYPE_STOP_LOSS = "STOP_LOSS"
EXIT_TYPE_TAKE_PROFIT = "TAKE_PROFIT"
EXIT_TYPE_MAX_HOLD = "MAX_HOLD"
EXIT_TYPE_SIGNAL_EXIT = "SIGNAL_EXIT"
EXIT_TYPE_HOLD = "HOLD"

EXIT_PRIORITY = (
    EXIT_TYPE_STOP_LOSS,
    EXIT_TYPE_TAKE_PROFIT,
    EXIT_TYPE_MAX_HOLD,
    EXIT_TYPE_SIGNAL_EXIT,
    EXIT_TYPE_HOLD,
)

THRESHOLD_SEMANTICS = (
    "take_profit_pct and stop_loss_pct are gross price movement from average "
    "entry (before exit costs). Net return after fees and slippage is reported "
    "separately and is not used to trigger exits."
)


@dataclass(frozen=True)
class ExitConfig:
    """Immutable exit rules attached to a strategy profile."""

    exit_mode: str
    take_profit_pct: float | None = None
    stop_loss_pct: float | None = None
    max_hold_seconds: int | None = None
    signal_exit_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["threshold_semantics"] = THRESHOLD_SEMANTICS
        payload["exit_priority"] = list(EXIT_PRIORITY)
        return payload


def validate_exit_config(config: ExitConfig) -> None:
    """Validate exit configuration fields."""
    if config.exit_mode not in (EXIT_MODE_SIGNAL_V1, EXIT_MODE_COST_AWARE):
        raise ValueError(
            f"exit_mode must be {EXIT_MODE_SIGNAL_V1!r} or {EXIT_MODE_COST_AWARE!r}, "
            f"got {config.exit_mode!r}"
        )

    for name in ("take_profit_pct", "stop_loss_pct"):
        value = getattr(config, name)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric, got {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    if config.max_hold_seconds is not None:
        if isinstance(config.max_hold_seconds, bool) or not isinstance(
            config.max_hold_seconds, int
        ):
            raise ValueError("max_hold_seconds must be an integer or None")
        if config.max_hold_seconds <= 0:
            raise ValueError("max_hold_seconds must be positive when set")

    if config.exit_mode == EXIT_MODE_COST_AWARE:
        if config.take_profit_pct is None and config.stop_loss_pct is None:
            if config.max_hold_seconds is None:
                raise ValueError(
                    "cost_aware exit requires at least one of take_profit_pct, "
                    "stop_loss_pct, or max_hold_seconds"
                )


SIGNAL_EXIT_V1 = ExitConfig(
    exit_mode=EXIT_MODE_SIGNAL_V1,
    take_profit_pct=None,
    stop_loss_pct=None,
    max_hold_seconds=None,
    signal_exit_enabled=True,
)
