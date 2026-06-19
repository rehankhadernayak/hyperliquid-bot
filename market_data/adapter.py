"""Compatibility adapter interface for future market data integration.

This phase does not connect the paper trader to WebSocket data.
The adapter provides a read-only interface for future historical replay
and live paper analysis integration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator


class MarketDataAdapter(ABC):
    """Abstract adapter for supplying normalized market events."""

    @abstractmethod
    def iter_events(
        self,
        event_types: list[str] | None = None,
        coins: list[str] | None = None,
    ) -> Iterator[Any]:
        """Yield normalized events from stored data."""

    @abstractmethod
    def event_count(self) -> int:
        """Return total number of available events."""


class StoredDataAdapter(MarketDataAdapter):
    """Read normalized events from the market_stream output directory."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir / "normalized"
        self._events: list[Any] = []

    def load(self) -> int:
        """Load events from JSONL files (Parquet loading deferred)."""
        import json

        count = 0
        if not self._output_dir.exists():
            return 0

        for jsonl_file in sorted(self._output_dir.rglob("*.jsonl")):
            with jsonl_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        self._events.append(json.loads(line))
                        count += 1
        return count

    def iter_events(
        self,
        event_types: list[str] | None = None,
        coins: list[str] | None = None,
    ) -> Iterator[Any]:
        for event in self._events:
            if event_types and event.get("event_type") not in event_types:
                continue
            if coins and event.get("coin") not in coins:
                continue
            yield event

    def event_count(self) -> int:
        return len(self._events)
