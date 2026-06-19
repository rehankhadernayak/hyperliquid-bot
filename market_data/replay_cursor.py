"""Replay cursor with no future-event leakage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from market_data.replay_events import ReplayEvent


@dataclass
class CursorPosition:
    index: int
    total: int
    replay_sequence: int | None
    primary_time_ms: int | None
    at_end: bool

    @property
    def progress_percentage(self) -> float:
        if self.total <= 0:
            return 100.0
        return min(100.0, (self.index / self.total) * 100.0)


class ReplayCursor:
    """Sequential cursor over ordered replay events.

    Only exposes the current position and the immediate next event via peek_event().
    No batch or window APIs that would leak future events to strategy code.
    """

    def __init__(self, events: list[ReplayEvent]) -> None:
        self._events = list(events)
        self._index = 0

    @property
    def total_events(self) -> int:
        return len(self._events)

    def reset(self) -> None:
        self._index = 0

    def at_end(self) -> bool:
        return self._index >= len(self._events)

    def current_position(self) -> CursorPosition:
        current = self._events[self._index] if self._index < len(self._events) else None
        return CursorPosition(
            index=self._index,
            total=len(self._events),
            replay_sequence=current.replay_sequence if current else None,
            primary_time_ms=current.primary_time_ms if current else None,
            at_end=self.at_end(),
        )

    def progress_percentage(self) -> float:
        return self.current_position().progress_percentage

    def peek_event(self) -> ReplayEvent | None:
        """Return the next event without advancing. Only one event ahead is visible."""
        if self.at_end():
            return None
        return self._events[self._index]

    def next_event(self) -> ReplayEvent | None:
        """Advance and return the next event."""
        if self.at_end():
            return None
        event = self._events[self._index]
        self._index += 1
        return event

    def seek_to_sequence(self, sequence: int) -> bool:
        """Seek to replay_sequence (1-based). Returns True if found."""
        for idx, event in enumerate(self._events):
            if event.replay_sequence == sequence:
                self._index = idx
                return True
        return False

    def seek_to_timestamp(self, timestamp_ms: int) -> bool:
        """Seek to first event with primary_time_ms >= timestamp_ms."""
        for idx, event in enumerate(self._events):
            if event.primary_time_ms >= timestamp_ms:
                self._index = idx
                return True
        self._index = len(self._events)
        return False

    def visible_event_count(self) -> int:
        """Events visible from current position (current peek only, not future window)."""
        return 0 if self.at_end() else 1

    def max_future_index(self) -> int:
        """Highest index that can be observed without advancing past peek."""
        if self.at_end():
            return self._index
        return self._index

    def events_ahead_count(self, max_lookahead: int = 1) -> int:
        """Count of events observable without consuming beyond peek (capped at one)."""
        if self.at_end():
            return 0
        allowed = min(max(1, max_lookahead), 1)
        return min(allowed, len(self._events) - self._index)
