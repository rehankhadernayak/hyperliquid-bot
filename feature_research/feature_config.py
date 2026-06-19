"""Configuration for feature research runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import DEFAULT_BACKTEST
from feature_research.feature_safety import ALLOWED_CHANNELS_V1, OUTPUT_SUBDIR

DEFAULT_COINS = ["BTC", "ETH", "SOL"]
DEFAULT_CHANNELS = ["trades", "bbo"]
DEFAULT_HORIZONS_SECONDS = [10, 30, 60, 300]
DEFAULT_BBO_STALE_MS = 10_000
DEFAULT_MIN_ROWS_FOR_SIGNAL = 50
DEFAULT_MAX_MISSING_RATE = 0.30
DEFAULT_QUANTILE_BUCKETS = 5

FEATURE_RESEARCH_SCHEMA_VERSION = "1.0.0"


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(config_dict).encode()).hexdigest()


@dataclass(frozen=True)
class FeatureResearchConfig:
    """Immutable configuration for one feature research run."""

    replay_session_id: str | None = None
    source_recorder_session_id: str | None = None
    use_latest_replay: bool = False
    coins: tuple[str, ...] = tuple(DEFAULT_COINS)
    channels: tuple[str, ...] = tuple(DEFAULT_CHANNELS)
    horizons_seconds: tuple[int, ...] = tuple(DEFAULT_HORIZONS_SECONDS)
    output_dir: Path = field(default_factory=lambda: Path("data") / OUTPUT_SUBDIR)
    max_events: int | None = None
    dry_run: bool = False
    bbo_stale_ms: int = DEFAULT_BBO_STALE_MS
    trading_fee_pct: float = DEFAULT_BACKTEST.trading_fee_pct
    slippage_pct: float = DEFAULT_BACKTEST.slippage_pct
    min_rows_for_signal: int = DEFAULT_MIN_ROWS_FOR_SIGNAL
    max_missing_rate: float = DEFAULT_MAX_MISSING_RATE
    quantile_buckets: int = DEFAULT_QUANTILE_BUCKETS
    market_stream_dir: Path = field(default_factory=lambda: Path("data/market_stream"))
    test_mode: bool = False

    def __post_init__(self) -> None:
        for channel in self.channels:
            if channel not in ALLOWED_CHANNELS_V1:
                raise ValueError(
                    f"Channel {channel!r} is not supported in feature research v1. "
                    f"Allowed: {sorted(ALLOWED_CHANNELS_V1)}"
                )

    @property
    def round_trip_cost_pct(self) -> float:
        """Estimated round-trip cost: buy fee + sell fee + slippage both ways."""
        return (2 * self.trading_fee_pct) + (2 * self.slippage_pct)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FEATURE_RESEARCH_SCHEMA_VERSION,
            "replay_session_id": self.replay_session_id,
            "source_recorder_session_id": self.source_recorder_session_id,
            "use_latest_replay": self.use_latest_replay,
            "coins": list(self.coins),
            "channels": list(self.channels),
            "horizons_seconds": list(self.horizons_seconds),
            "output_dir": str(self.output_dir),
            "max_events": self.max_events,
            "dry_run": self.dry_run,
            "bbo_stale_ms": self.bbo_stale_ms,
            "trading_fee_pct": self.trading_fee_pct,
            "slippage_pct": self.slippage_pct,
            "round_trip_cost_pct": self.round_trip_cost_pct,
            "min_rows_for_signal": self.min_rows_for_signal,
            "max_missing_rate": self.max_missing_rate,
            "quantile_buckets": self.quantile_buckets,
        }

    def config_hash(self) -> str:
        return compute_config_hash(self.to_dict())
