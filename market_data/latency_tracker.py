"""Latency tracking for market data messages."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyStats:
    """Rolling latency statistics for exchange-timestamped messages."""

    count: int = 0
    minimum_ms: float | None = None
    maximum_ms: float | None = None
    negative_count: int = 0
    missing_timestamp_count: int = 0
    _samples: list[float] = field(default_factory=list)
    _max_samples: int = 10000

    def record(self, exchange_timestamp_ms: int | None, received_timestamp_ms: int) -> None:
        if exchange_timestamp_ms is None:
            self.missing_timestamp_count += 1
            return

        latency = received_timestamp_ms - exchange_timestamp_ms
        self.count += 1
        self._samples.append(latency)
        if len(self._samples) > self._max_samples:
            self._samples = self._samples[-self._max_samples:]

        if latency < 0:
            self.negative_count += 1

        if self.minimum_ms is None or latency < self.minimum_ms:
            self.minimum_ms = latency
        if self.maximum_ms is None or latency > self.maximum_ms:
            self.maximum_ms = latency

    def percentile(self, p: float) -> float | None:
        if not self._samples:
            return None
        sorted_samples = sorted(self._samples)
        idx = int(math.ceil(p * len(sorted_samples))) - 1
        idx = max(0, min(idx, len(sorted_samples) - 1))
        return sorted_samples[idx]

    @property
    def median_ms(self) -> float | None:
        return self.percentile(0.5)

    @property
    def p95_ms(self) -> float | None:
        return self.percentile(0.95)

    @property
    def p99_ms(self) -> float | None:
        return self.percentile(0.99)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "minimum_ms": self.minimum_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "maximum_ms": self.maximum_ms,
            "negative_latency_count": self.negative_count,
            "missing_timestamp_count": self.missing_timestamp_count,
        }
