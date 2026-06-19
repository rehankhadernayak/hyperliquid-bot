"""Typed replay events and read-only consumer protocol."""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from market_data.event_models import (
    AssetContextEvent,
    BboEvent,
    CandleEvent,
    EVENT_TYPE_NAMES,
    MidPriceEvent,
    OrderBookEvent,
    TradeEvent,
)
from market_data.normalizer import dedup_key
from market_data.quality_models import EVENT_TYPE_TO_CHANNEL

logger = logging.getLogger(__name__)

REPLAY_EVENT_TYPES = (
    MidPriceEvent,
    TradeEvent,
    BboEvent,
    OrderBookEvent,
    CandleEvent,
    AssetContextEvent,
)

_EVENT_CLASS_BY_TYPE = {
    "mid": MidPriceEvent,
    "trade": TradeEvent,
    "bbo": BboEvent,
    "l2_book": OrderBookEvent,
    "candle": CandleEvent,
    "asset_context": AssetContextEvent,
}


@dataclass(frozen=True)
class ReplayEvent:
    """Immutable wrapper around a normalized market event for deterministic replay."""

    replay_session_id: str
    source_recorder_session_id: str
    original_event_type: str
    coin: str | None
    exchange_timestamp_ms: int | None
    received_timestamp_ms: int
    replay_sequence: int
    source_file: str
    source_row_index: int
    original_local_sequence: int | None
    payload_hash: str
    event: Any
    replay_emitted_timestamp_ms: int | None = None
    primary_time_ms: int = 0
    primary_time_source: str = "received_timestamp_ms"
    channel: str = ""

    def with_emitted(self, emitted_ms: int) -> ReplayEvent:
        return ReplayEvent(
            replay_session_id=self.replay_session_id,
            source_recorder_session_id=self.source_recorder_session_id,
            original_event_type=self.original_event_type,
            coin=self.coin,
            exchange_timestamp_ms=self.exchange_timestamp_ms,
            received_timestamp_ms=self.received_timestamp_ms,
            replay_sequence=self.replay_sequence,
            source_file=self.source_file,
            source_row_index=self.source_row_index,
            original_local_sequence=self.original_local_sequence,
            payload_hash=self.payload_hash,
            event=self.event,
            replay_emitted_timestamp_ms=emitted_ms,
            primary_time_ms=self.primary_time_ms,
            primary_time_source=self.primary_time_source,
            channel=self.channel,
        )


def event_type_name(event: Any) -> str:
    return EVENT_TYPE_NAMES.get(type(event).__name__, type(event).__name__.lower())


def channel_for_event_type(event_type: str) -> str:
    return EVENT_TYPE_TO_CHANNEL.get(event_type, event_type)


def compute_payload_hash(event: Any) -> str:
    return dedup_key(event)


def _parse_tuple_field(value: Any) -> tuple[tuple[float, float], ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(tuple(float(x) for x in pair) for pair in value)
    if isinstance(value, str):
        parsed = json.loads(value)
        return tuple(tuple(float(x) for x in pair) for pair in parsed)
    return ()


def _parse_extras(row: dict[str, Any]) -> dict[str, Any]:
    extras_json = row.pop("extras_json", None)
    if extras_json:
        try:
            return json.loads(extras_json)
        except (json.JSONDecodeError, TypeError):
            return {}
    return row.pop("extras", {}) or {}


def row_to_event(row: dict[str, Any], event_type: str) -> Any:
    """Rehydrate a normalized storage row into a typed event object."""
    cls = _EVENT_CLASS_BY_TYPE.get(event_type)
    if cls is None:
        raise ValueError(f"Unknown event type: {event_type}")

    data = dict(row)
    extras = _parse_extras(data)

    if event_type == "l2_book":
        data["bids"] = _parse_tuple_field(data.get("bids", ()))
        data["asks"] = _parse_tuple_field(data.get("asks", ()))

    allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {k: v for k, v in data.items() if k in allowed}
    if "extras" in allowed:
        kwargs["extras"] = extras
    return cls(**kwargs)


def build_replay_event(
    *,
    event: Any,
    event_type: str,
    replay_session_id: str,
    source_recorder_session_id: str,
    source_file: str,
    source_row_index: int,
    replay_sequence: int,
    original_local_sequence: int | None = None,
    primary_time_ms: int = 0,
    primary_time_source: str = "received_timestamp_ms",
) -> ReplayEvent:
    exchange_ts = getattr(event, "exchange_timestamp_ms", None)
    received_ts = getattr(event, "received_timestamp_ms", 0)
    coin = getattr(event, "coin", None)
    channel = channel_for_event_type(event_type)

    return ReplayEvent(
        replay_session_id=replay_session_id,
        source_recorder_session_id=source_recorder_session_id,
        original_event_type=event_type,
        coin=coin,
        exchange_timestamp_ms=exchange_ts,
        received_timestamp_ms=received_ts,
        replay_sequence=replay_sequence,
        source_file=source_file,
        source_row_index=source_row_index,
        original_local_sequence=original_local_sequence,
        payload_hash=compute_payload_hash(event),
        event=event,
        primary_time_ms=primary_time_ms,
        primary_time_source=primary_time_source,
        channel=channel,
    )


def stream_hash_entry(event: ReplayEvent) -> str:
    """Stable serialization of one event for deterministic stream hashing."""
    parts = [
        event.original_event_type,
        event.coin or "",
        str(event.exchange_timestamp_ms) if event.exchange_timestamp_ms is not None else "",
        str(event.received_timestamp_ms),
        event.payload_hash,
        str(event.replay_sequence),
        event.primary_time_source,
    ]
    return "|".join(parts)


def compute_stream_hash(events: list[ReplayEvent]) -> str:
    hasher = hashlib.sha256()
    for event in events:
        hasher.update(stream_hash_entry(event).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


@dataclass
class ReplayContext:
    replay_session_id: str
    source_recorder_session_id: str
    dry_run: bool
    speed: float
    filters: dict[str, Any] = field(default_factory=dict)
    ordering_rule: str = ""


@dataclass
class ReplaySummary:
    total_events: int
    emitted_count: int
    skipped_events: int
    stream_hash: str
    first_event_time_ms: int | None
    last_event_time_ms: int | None
    replay_duration_ms: int
    dry_run: bool


class MarketDataReplayConsumer(ABC):
    """Read-only consumer protocol for replay output."""

    @abstractmethod
    def on_start(self, context: ReplayContext) -> None:
        ...

    @abstractmethod
    def on_event(self, event: ReplayEvent) -> None:
        ...

    @abstractmethod
    def on_finish(self, summary: ReplaySummary) -> None:
        ...


class NullConsumer(MarketDataReplayConsumer):
    def on_start(self, context: ReplayContext) -> None:
        pass

    def on_event(self, event: ReplayEvent) -> None:
        pass

    def on_finish(self, summary: ReplaySummary) -> None:
        pass


class CountingConsumer(MarketDataReplayConsumer):
    def __init__(self) -> None:
        self.count = 0
        self.by_channel: dict[str, int] = {}
        self.by_coin: dict[str, int] = {}
        self.started = False
        self.finished = False

    def on_start(self, context: ReplayContext) -> None:
        self.started = True

    def on_event(self, event: ReplayEvent) -> None:
        self.count += 1
        ch = event.channel or event.original_event_type
        self.by_channel[ch] = self.by_channel.get(ch, 0) + 1
        coin = event.coin or "global"
        self.by_coin[coin] = self.by_coin.get(coin, 0) + 1

    def on_finish(self, summary: ReplaySummary) -> None:
        self.finished = True


class LoggingConsumer(MarketDataReplayConsumer):
    def __init__(self, max_log: int = 10) -> None:
        self._max_log = max_log
        self._logged = 0

    def on_start(self, context: ReplayContext) -> None:
        logger.info(
            "Replay started: session=%s source=%s dry_run=%s",
            context.replay_session_id,
            context.source_recorder_session_id,
            context.dry_run,
        )

    def on_event(self, event: ReplayEvent) -> None:
        if self._logged < self._max_log:
            logger.info(
                "Replay event #%s type=%s coin=%s primary_time=%s",
                event.replay_sequence,
                event.original_event_type,
                event.coin,
                event.primary_time_ms,
            )
            self._logged += 1

    def on_finish(self, summary: ReplaySummary) -> None:
        logger.info(
            "Replay finished: emitted=%s hash=%s",
            summary.emitted_count,
            summary.stream_hash[:16],
        )
