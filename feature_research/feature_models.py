"""Data models for feature research rows, labels, and analysis."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FutureDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"
    INSUFFICIENT_FUTURE_DATA = "INSUFFICIENT_FUTURE_DATA"


class SignalCandidate(str, Enum):
    NO_SIGNAL = "NO_SIGNAL"
    WEAK_SIGNAL = "WEAK_SIGNAL"
    POSSIBLE_SIGNAL_NEEDS_MORE_DATA = "POSSIBLE_SIGNAL_NEEDS_MORE_DATA"
    UNSTABLE_SIGNAL = "UNSTABLE_SIGNAL"
    REJECT = "REJECT"


FEATURE_COLUMN_NAMES = [
    "return_5s_pct",
    "return_10s_pct",
    "return_30s_pct",
    "return_60s_pct",
    "volatility_10s",
    "volatility_30s",
    "volatility_60s",
    "spread",
    "spread_bps",
    "spread_bps_percentile_recent",
    "bbo_available",
    "bbo_age_ms",
    "bid_size",
    "ask_size",
    "bbo_imbalance",
    "bbo_imbalance_change_10s",
    "bbo_imbalance_change_30s",
    "buy_trade_count_10s",
    "sell_trade_count_10s",
    "buy_volume_10s",
    "sell_volume_10s",
    "trade_imbalance_10s",
    "trade_imbalance_30s",
    "volume_burst_30s",
    "price_acceleration_10s",
    "price_acceleration_30s",
    "consecutive_up_ticks",
    "consecutive_down_ticks",
    "btc_return_30s",
    "eth_return_30s",
    "sol_return_30s",
    "coin_minus_btc_return_30s",
    "coin_minus_eth_return_30s",
]

LABEL_COLUMN_NAMES = [
    "future_return_10s_pct",
    "future_return_30s_pct",
    "future_return_60s_pct",
    "future_return_300s_pct",
    "future_up_after_costs_10s",
    "future_up_after_costs_30s",
    "future_up_after_costs_60s",
    "future_up_after_costs_300s",
    "future_down_after_costs_10s",
    "future_down_after_costs_30s",
    "future_down_after_costs_60s",
    "future_down_after_costs_300s",
    "future_abs_move_after_costs_10s",
    "future_abs_move_after_costs_30s",
    "future_abs_move_after_costs_60s",
    "future_abs_move_after_costs_300s",
    "future_direction_10s",
    "future_direction_30s",
    "future_direction_60s",
    "future_direction_300s",
    "max_favorable_move_pct",
    "max_adverse_move_pct",
    "time_to_max_favorable_ms",
    "time_to_max_adverse_ms",
]


def make_feature_row_id(replay_session_id: str, coin: str, replay_sequence: int) -> str:
    raw = f"{replay_session_id}|{coin}|{replay_sequence}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


@dataclass(frozen=True)
class FeatureRow:
    """Immutable feature row created at a trade decision point."""

    feature_row_id: str
    replay_session_id: str
    source_recorder_session_id: str
    coin: str
    event_timestamp_ms: int
    received_timestamp_ms: int
    latest_trade_price: float
    latest_midpoint: float | None
    best_bid: float | None
    best_ask: float | None
    spread: float | None
    spread_bps: float | None
    recent_trade_count: int
    recent_volume: float
    current_event_type: str
    source_replay_sequence: int
    feature_config_hash: str
    features: dict[str, Any] = field(default_factory=dict)
    missing_flags: dict[str, bool] = field(default_factory=dict)
    _frozen: bool = field(default=True, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_row_id": self.feature_row_id,
            "replay_session_id": self.replay_session_id,
            "source_recorder_session_id": self.source_recorder_session_id,
            "coin": self.coin,
            "event_timestamp_ms": self.event_timestamp_ms,
            "received_timestamp_ms": self.received_timestamp_ms,
            "latest_trade_price": self.latest_trade_price,
            "latest_midpoint": self.latest_midpoint,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "recent_trade_count": self.recent_trade_count,
            "recent_volume": self.recent_volume,
            "current_event_type": self.current_event_type,
            "source_replay_sequence": self.source_replay_sequence,
            "feature_config_hash": self.feature_config_hash,
            "features": dict(self.features),
            "missing_flags": dict(self.missing_flags),
        }


@dataclass(frozen=True)
class FeatureLabel:
    """Labels attached after feature rows are frozen (may use future data)."""

    feature_row_id: str
    labels: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"feature_row_id": self.feature_row_id, "labels": dict(self.labels)}


@dataclass
class LeakageCheckResult:
    """Result of no-future-leakage validation."""

    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "warnings": self.warnings,
        }
