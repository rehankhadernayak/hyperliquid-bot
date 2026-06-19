"""Configuration for event-driven backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from event_backtester.event_safety import ALLOWED_CHANNELS_V1, OUTPUT_SUBDIR

DEFAULT_COINS = ["BTC", "ETH", "SOL"]
DEFAULT_CHANNELS = ["trades", "bbo"]
DEFAULT_STRATEGY_PROFILE = "contrarian_v1_medium_no_trend_filter"
DEFAULT_STARTING_EQUITY = 1000.0
DEFAULT_BBO_STALE_MS = 10_000
DEFAULT_MIN_COMPLETED_TRADES = 3
DEFAULT_MIN_SESSION_DURATION_MS = 60_000
DEFAULT_MIN_BBO_AVAILABILITY_PCT = 50.0

SUPPORTED_STRATEGY_PROFILES = frozenset({DEFAULT_STRATEGY_PROFILE})


@dataclass(frozen=True)
class EventBacktestConfig:
    """Immutable configuration for one event backtest run."""

    replay_session_id: str | None = None
    source_recorder_session_id: str | None = None
    use_latest_replay: bool = False
    coins: tuple[str, ...] = tuple(DEFAULT_COINS)
    channels: tuple[str, ...] = tuple(DEFAULT_CHANNELS)
    strategy_profile_id: str = DEFAULT_STRATEGY_PROFILE
    starting_equity: float = DEFAULT_STARTING_EQUITY
    output_dir: Path = field(default_factory=lambda: Path("data") / OUTPUT_SUBDIR)
    max_events: int | None = None
    dry_run: bool = False
    bbo_stale_ms: int = DEFAULT_BBO_STALE_MS
    allow_midpoint_fills: bool = False
    min_completed_trades: int = DEFAULT_MIN_COMPLETED_TRADES
    min_session_duration_ms: int = DEFAULT_MIN_SESSION_DURATION_MS
    min_bbo_availability_pct: float = DEFAULT_MIN_BBO_AVAILABILITY_PCT
    market_stream_dir: Path = field(default_factory=lambda: Path("data/market_stream"))
    test_mode: bool = False

    def __post_init__(self) -> None:
        for channel in self.channels:
            if channel not in ALLOWED_CHANNELS_V1:
                raise ValueError(
                    f"Channel {channel!r} is not supported in event backtester v1. "
                    f"Allowed: {sorted(ALLOWED_CHANNELS_V1)}"
                )
        if self.strategy_profile_id not in SUPPORTED_STRATEGY_PROFILES:
            raise ValueError(
                f"Unsupported strategy profile: {self.strategy_profile_id!r}. "
                f"Supported: {sorted(SUPPORTED_STRATEGY_PROFILES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_session_id": self.replay_session_id,
            "source_recorder_session_id": self.source_recorder_session_id,
            "use_latest_replay": self.use_latest_replay,
            "coins": list(self.coins),
            "channels": list(self.channels),
            "strategy_profile_id": self.strategy_profile_id,
            "starting_equity": self.starting_equity,
            "output_dir": str(self.output_dir),
            "max_events": self.max_events,
            "dry_run": self.dry_run,
            "bbo_stale_ms": self.bbo_stale_ms,
            "allow_midpoint_fills": self.allow_midpoint_fills,
            "min_completed_trades": self.min_completed_trades,
            "min_session_duration_ms": self.min_session_duration_ms,
            "min_bbo_availability_pct": self.min_bbo_availability_pct,
        }
