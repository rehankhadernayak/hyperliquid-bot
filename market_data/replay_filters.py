"""Non-destructive filters for market data replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from market_data.quality_models import CHANNEL_TO_EVENT_TYPE, QualityLabel
from market_data.replay_events import ReplayEvent

QUALITY_LABEL_RANK = {
    QualityLabel.UNUSABLE.value: 0,
    QualityLabel.DEGRADED.value: 1,
    QualityLabel.USABLE_WITH_WARNINGS.value: 2,
    QualityLabel.GOOD_FOR_RESEARCH.value: 3,
}


@dataclass
class ReplayFilters:
    coins: list[str] | None = None
    channels: list[str] | None = None
    start_timestamp_ms: int | None = None
    end_timestamp_ms: int | None = None
    max_events: int | None = None
    quality_label_minimum: str | None = None
    include_candles: bool = True
    include_l2_book: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "coins": self.coins,
            "channels": self.channels,
            "start_timestamp_ms": self.start_timestamp_ms,
            "end_timestamp_ms": self.end_timestamp_ms,
            "max_events": self.max_events,
            "quality_label_minimum": self.quality_label_minimum,
            "include_candles": self.include_candles,
            "include_l2_book": self.include_l2_book,
        }


def channels_to_event_types(channels: list[str] | None) -> set[str] | None:
    if channels is None:
        return None
    result: set[str] = set()
    for ch in channels:
        et = CHANNEL_TO_EVENT_TYPE.get(ch)
        if et:
            result.add(et)
    return result or None


def apply_filters(
    events: list[ReplayEvent],
    filters: ReplayFilters,
    session_quality_label: str | None = None,
) -> tuple[list[ReplayEvent], int]:
    """Apply filters without modifying source data. Returns (filtered, skipped_count)."""
    if filters.quality_label_minimum and session_quality_label:
        min_rank = QUALITY_LABEL_RANK.get(filters.quality_label_minimum, 0)
        session_rank = QUALITY_LABEL_RANK.get(session_quality_label, 0)
        if session_rank < min_rank:
            return [], len(events)

    allowed_types = channels_to_event_types(filters.channels)
    coin_set = set(filters.coins) if filters.coins else None

    filtered: list[ReplayEvent] = []
    skipped = 0

    for event in events:
        if not filters.include_candles and event.original_event_type == "candle":
            skipped += 1
            continue
        if not filters.include_l2_book and event.original_event_type == "l2_book":
            skipped += 1
            continue
        if allowed_types and event.original_event_type not in allowed_types:
            skipped += 1
            continue
        if coin_set and event.coin and event.coin not in coin_set:
            skipped += 1
            continue
        if filters.start_timestamp_ms is not None and event.primary_time_ms < filters.start_timestamp_ms:
            skipped += 1
            continue
        if filters.end_timestamp_ms is not None and event.primary_time_ms > filters.end_timestamp_ms:
            skipped += 1
            continue

        filtered.append(event)
        if filters.max_events is not None and len(filtered) >= filters.max_events:
            skipped += len(events) - len(filtered) - skipped
            break

    return filtered, skipped
