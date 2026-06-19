"""Data models and documented thresholds for market data quality reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

QUALITY_REPORT_SCHEMA_VERSION = "1.0.0"

# --- Quality label thresholds (documented in README) ---

# Reconnect counts: GOOD <= 3, USABLE <= 10, above contributes to DEGRADED
MAX_RECONNECT_COUNT_GOOD = 3
MAX_RECONNECT_COUNT_USABLE = 10

# Malformed event rate (malformed / raw inspected)
MALFORMED_RATE_DEGRADED = 0.01
MALFORMED_RATE_UNUSABLE = 0.05

# Raw/normalized count mismatch tolerance vs manifest
RAW_COUNT_TOLERANCE = 0.02
NORMALIZED_COUNT_TOLERANCE = 0.05
NORMALIZED_TO_RAW_EXPANSION_MAX = 20.0

# Timestamp / gap thresholds (aligned with gap_detection.py)
RECEIVE_TIME_GAP_MS = 60_000
EXCHANGE_TIME_GAP_MS = 120_000
CHANNEL_STALENESS_MS = 120_000
CANDLE_INTERVAL_MS = 60_000

# Session duration guidance
MIN_SESSION_DURATION_RESEARCH_SEC = 300

# Duplicate exchange timestamps: warning above this rate per channel
DUPLICATE_TIMESTAMP_RATE_WARNING = 0.05

# Latency warnings (receive - exchange)
LATENCY_P95_WARNING_MS = 10_000
LATENCY_P99_WARNING_MS = 30_000
NEGATIVE_LATENCY_WARNING_COUNT = 1

# Backward exchange timestamps before DEGRADED
BACKWARD_TIMESTAMP_DEGRADED_COUNT = 10
BACKWARD_TIMESTAMP_CRITICAL_COUNT = 100

# Missing normalized file fraction before DEGRADED
MISSING_NORMALIZED_FRACTION_DEGRADED = 0.10

# Minimum raw events for research usability
MIN_RAW_EVENTS_RESEARCH = 100

# BBO stale period: no update for this many ms
BBO_STALE_MS = 120_000
L2_STALE_MS = 120_000

# Trade price jump vs recent mid (fraction)
TRADE_PRICE_JUMP_WARNING = 0.05

WATCH_COINS = ("BTC", "ETH", "SOL")
WATCH_CHANNELS = ("allMids", "trades", "bbo", "l2Book", "candle", "activeAssetCtx")

CHANNEL_TO_EVENT_TYPE = {
    "allMids": "mid",
    "trades": "trade",
    "bbo": "bbo",
    "l2Book": "l2_book",
    "candle": "candle",
    "activeAssetCtx": "asset_context",
}

EVENT_TYPE_TO_CHANNEL = {v: k for k, v in CHANNEL_TO_EVENT_TYPE.items()}


class QualityLabel(str, Enum):
    GOOD_FOR_RESEARCH = "GOOD_FOR_RESEARCH"
    USABLE_WITH_WARNINGS = "USABLE_WITH_WARNINGS"
    DEGRADED = "DEGRADED"
    UNUSABLE = "UNUSABLE"


class IssueSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


@dataclass
class IntegrityWarning:
    severity: str
    category: str
    message: str
    channel: str | None = None
    coin: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "channel": self.channel,
            "coin": self.coin,
            "details": self.details,
        }


@dataclass
class ChannelSummary:
    channel: str
    raw_event_count: int = 0
    normalized_event_count: int = 0
    first_received_timestamp_ms: int | None = None
    last_received_timestamp_ms: int | None = None
    observed_duration_ms: int | None = None
    messages_per_second: float | None = None
    exchange_timestamp_availability_pct: float | None = None
    duplicate_count: int = 0
    malformed_count: int = 0
    gap_count: int = 0
    stale_period_count: int = 0
    quality_label: str = QualityLabel.UNUSABLE.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "raw_event_count": self.raw_event_count,
            "normalized_event_count": self.normalized_event_count,
            "first_received_timestamp_ms": self.first_received_timestamp_ms,
            "last_received_timestamp_ms": self.last_received_timestamp_ms,
            "observed_duration_ms": self.observed_duration_ms,
            "messages_per_second": self.messages_per_second,
            "exchange_timestamp_availability_pct": self.exchange_timestamp_availability_pct,
            "duplicate_count": self.duplicate_count,
            "malformed_count": self.malformed_count,
            "gap_count": self.gap_count,
            "stale_period_count": self.stale_period_count,
            "quality_label": self.quality_label,
        }


@dataclass
class CoinSummary:
    coin: str
    raw_event_count: int = 0
    normalized_event_count: int = 0
    first_received_timestamp_ms: int | None = None
    last_received_timestamp_ms: int | None = None
    observed_duration_ms: int | None = None
    messages_per_second: float | None = None
    exchange_timestamp_availability_pct: float | None = None
    duplicate_count: int = 0
    malformed_count: int = 0
    gap_count: int = 0
    stale_period_count: int = 0
    quality_label: str = QualityLabel.UNUSABLE.value
    by_channel: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "coin": self.coin,
            "raw_event_count": self.raw_event_count,
            "normalized_event_count": self.normalized_event_count,
            "first_received_timestamp_ms": self.first_received_timestamp_ms,
            "last_received_timestamp_ms": self.last_received_timestamp_ms,
            "observed_duration_ms": self.observed_duration_ms,
            "messages_per_second": self.messages_per_second,
            "exchange_timestamp_availability_pct": self.exchange_timestamp_availability_pct,
            "duplicate_count": self.duplicate_count,
            "malformed_count": self.malformed_count,
            "gap_count": self.gap_count,
            "stale_period_count": self.stale_period_count,
            "quality_label": self.quality_label,
            "by_channel": self.by_channel,
        }


@dataclass
class LatencySummary:
    channel: str
    count: int = 0
    missing_timestamp_count: int = 0
    negative_latency_count: int = 0
    minimum_ms: float | None = None
    median_ms: float | None = None
    p90_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    maximum_ms: float | None = None
    latency_available: bool = True
    informational_only: bool = False
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "count": self.count,
            "missing_timestamp_count": self.missing_timestamp_count,
            "negative_latency_count": self.negative_latency_count,
            "minimum_ms": self.minimum_ms,
            "median_ms": self.median_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "maximum_ms": self.maximum_ms,
            "latency_available": self.latency_available,
            "informational_only": self.informational_only,
            "note": self.note,
        }


@dataclass
class UsabilityFlag:
    usable: bool
    reason: str

    def to_dict(self) -> dict[str, bool | str]:
        return {"usable": self.usable, "reason": self.reason}


@dataclass
class GapSummaryEntry:
    gap_type: str
    coin: str | None
    channel: str
    start_ms: int
    end_ms: int
    duration_ms: int
    certainty: str
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_type": self.gap_type,
            "coin": self.coin,
            "channel": self.channel,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "certainty": self.certainty,
            "severity": self.severity,
        }
