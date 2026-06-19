"""Deterministic ordering rules for market data replay."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from market_data.replay_events import ReplayEvent

ORDERING_RULE = (
    "1. Primary event time: exchange_timestamp_ms when available and safe; "
    "otherwise received_timestamp_ms (or candle open_time_ms with informational warning). "
    "2. Tie-breaker: received_timestamp_ms. "
    "3. Original local sequence number when available. "
    "4. Channel priority: trades, bbo, l2Book, activeAssetCtx, candle, allMids. "
    "5. Coin alphabetical order. "
    "6. Source file path. "
    "7. Source row index."
)

CHANNEL_PRIORITY: dict[str, int] = {
    "trades": 0,
    "bbo": 1,
    "l2Book": 2,
    "activeAssetCtx": 3,
    "candle": 4,
    "allMids": 5,
}

# Event types where exchange timestamp is unreliable or absent.
EXCHANGE_TIMESTAMP_UNSAFE: dict[str, str] = {
    "mid": "no_exchange_timestamp",
    "asset_context": "no_exchange_timestamp",
    "candle": "informational_candle_timestamp",
}


def primary_time_for_event(event: Any, event_type: str) -> tuple[int, str]:
    """Return (primary_time_ms, source_reason) without using future information."""
    received = getattr(event, "received_timestamp_ms", 0)

    if event_type == "candle":
        open_time = getattr(event, "open_time_ms", received)
        return open_time, EXCHANGE_TIMESTAMP_UNSAFE["candle"]

    if event_type in EXCHANGE_TIMESTAMP_UNSAFE:
        return received, EXCHANGE_TIMESTAMP_UNSAFE[event_type]

    exchange = getattr(event, "exchange_timestamp_ms", None)
    if exchange is not None:
        return exchange, "exchange_timestamp_ms"
    return received, "missing_exchange_timestamp"


def ordering_sort_key(record: ReplayEvent) -> tuple[Any, ...]:
    """Deterministic sort key for replay events."""
    channel_prio = CHANNEL_PRIORITY.get(record.channel, 99)
    local_seq = record.original_local_sequence if record.original_local_sequence is not None else -1
    coin = record.coin or ""

    return (
        record.primary_time_ms,
        record.received_timestamp_ms,
        local_seq,
        channel_prio,
        coin,
        record.source_file,
        record.source_row_index,
    )


def sort_events_deterministically(events: list[ReplayEvent]) -> list[ReplayEvent]:
    """Sort events in place using the documented ordering rule."""
    return sorted(events, key=ordering_sort_key)


def assign_replay_sequences(events: list[ReplayEvent]) -> list[ReplayEvent]:
    """Return new list with replay_sequence 1..N after ordering."""
    result: list[ReplayEvent] = []
    for idx, event in enumerate(events, start=1):
        result.append(
            ReplayEvent(
                replay_session_id=event.replay_session_id,
                source_recorder_session_id=event.source_recorder_session_id,
                original_event_type=event.original_event_type,
                coin=event.coin,
                exchange_timestamp_ms=event.exchange_timestamp_ms,
                received_timestamp_ms=event.received_timestamp_ms,
                replay_sequence=idx,
                source_file=event.source_file,
                source_row_index=event.source_row_index,
                original_local_sequence=event.original_local_sequence,
                payload_hash=event.payload_hash,
                event=event.event,
                replay_emitted_timestamp_ms=event.replay_emitted_timestamp_ms,
                primary_time_ms=event.primary_time_ms,
                primary_time_source=event.primary_time_source,
                channel=event.channel,
            )
        )
    return result


@dataclass
class OrderingAnalysis:
    events_ordered_by_exchange_timestamp: int = 0
    events_ordered_by_received_timestamp: int = 0
    tie_breaker_usage: dict[str, int] = field(default_factory=dict)
    missing_timestamp_counts: dict[str, int] = field(default_factory=dict)
    ordering_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events_ordered_by_exchange_timestamp": self.events_ordered_by_exchange_timestamp,
            "events_ordered_by_received_timestamp": self.events_ordered_by_received_timestamp,
            "tie_breaker_usage": dict(self.tie_breaker_usage),
            "missing_timestamp_counts": dict(self.missing_timestamp_counts),
            "ordering_warnings": list(self.ordering_warnings),
        }


def analyze_ordering(events: list[ReplayEvent]) -> OrderingAnalysis:
    """Analyze ordering decisions and emit warnings."""
    analysis = OrderingAnalysis()
    if not events:
        return analysis

    primary_sources = Counter(e.primary_time_source for e in events)
    analysis.events_ordered_by_exchange_timestamp = primary_sources.get("exchange_timestamp_ms", 0)
    analysis.events_ordered_by_received_timestamp = (
        primary_sources.get("missing_exchange_timestamp", 0)
        + primary_sources.get("no_exchange_timestamp", 0)
        + primary_sources.get("informational_candle_timestamp", 0)
    )

    for source, count in primary_sources.items():
        if source != "exchange_timestamp_ms":
            analysis.missing_timestamp_counts[source] = count

    # Duplicate primary timestamps
    ts_counts: Counter[int] = Counter(e.primary_time_ms for e in events)
    dup_ts = sum(1 for c in ts_counts.values() if c > 1)
    if dup_ts > 0:
        analysis.tie_breaker_usage["duplicate_primary_timestamp"] = dup_ts
        max_dup = max(ts_counts.values())
        if max_dup > 10:
            analysis.ordering_warnings.append(
                f"{max_dup} events share the same primary timestamp — tie-breakers applied heavily"
            )

    # Backward exchange timestamps within channel
    last_exchange_by_channel: dict[str, int] = {}
    backward_count = 0
    for event in events:
        if event.exchange_timestamp_ms is None:
            continue
        ch = event.channel
        prev = last_exchange_by_channel.get(ch)
        if prev is not None and event.exchange_timestamp_ms < prev:
            backward_count += 1
        last_exchange_by_channel[ch] = event.exchange_timestamp_ms

    if backward_count:
        analysis.ordering_warnings.append(
            f"Exchange timestamps moved backward {backward_count} time(s) within channels"
        )
        analysis.tie_breaker_usage["backward_exchange_timestamp"] = backward_count

    # Missing exchange timestamps by channel
    missing_by_channel: Counter[str] = Counter()
    for event in events:
        if event.primary_time_source in (
            "missing_exchange_timestamp",
            "no_exchange_timestamp",
        ):
            missing_by_channel[event.channel] += 1

    for ch, count in missing_by_channel.items():
        if count > 0:
            analysis.ordering_warnings.append(
                f"Channel {ch}: {count} event(s) ordered by received timestamp (missing exchange ts)"
            )

    received_fallback = primary_sources.get("missing_exchange_timestamp", 0)
    if events and received_fallback > len(events) * 0.5:
        analysis.ordering_warnings.append(
            "Ordering falls back heavily to received timestamps (>50% of events)"
        )

    candle_count = sum(1 for e in events if e.original_event_type == "candle")
    if candle_count:
        analysis.ordering_warnings.append(
            f"Candles included in event-time ordering ({candle_count} events) — "
            "timestamps are informational (open time), not message send time"
        )

    # Source file time order
    file_first_ts: dict[str, int] = {}
    for event in events:
        if event.source_file not in file_first_ts:
            file_first_ts[event.source_file] = event.primary_time_ms

    sorted_files = sorted(file_first_ts.items(), key=lambda x: x[1])
    path_order = sorted(file_first_ts.keys())
    if [f for f, _ in sorted_files] != path_order and len(file_first_ts) > 1:
        analysis.ordering_warnings.append(
            "Source files appear out of expected chronological order by path"
        )

    return analysis
