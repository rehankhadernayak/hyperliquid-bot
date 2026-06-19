"""Gap and staleness detection for market data streams."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GapCertainty(Enum):
    INFERRED = "INFERRED"
    CONFIRMED = "CONFIRMED"


@dataclass(frozen=True)
class GapRecord:
    gap_type: str
    coin: str | None
    channel: str
    started_at_ms: int
    ended_at_ms: int
    duration_ms: int
    certainty: str = GapCertainty.INFERRED.value
    details: dict[str, Any] = field(default_factory=dict)


class GapDetector:
    """Detect inferred gaps in market data streams."""

    def __init__(self, candle_interval_ms: int = 60000) -> None:
        self._candle_interval_ms = candle_interval_ms
        self._last_exchange_ts: dict[str, int] = {}
        self._last_receive_ts: dict[str, int] = {}
        self._last_candle_open: dict[str, int] = {}
        self._gaps: list[GapRecord] = []
        self._duplicate_timestamps: int = 0
        self._backward_timestamps: int = 0

    def _key(self, channel: str, coin: str | None) -> str:
        return f"{channel}:{coin or 'global'}"

    def record_reconnect_window(
        self,
        started_at_ms: int,
        ended_at_ms: int,
        coin: str | None = None,
        channel: str = "all",
    ) -> GapRecord:
        gap = GapRecord(
            gap_type="RECONNECT_WINDOW",
            coin=coin,
            channel=channel,
            started_at_ms=started_at_ms,
            ended_at_ms=ended_at_ms,
            duration_ms=ended_at_ms - started_at_ms,
            certainty=GapCertainty.INFERRED.value,
        )
        self._gaps.append(gap)
        return gap

    def record_exchange_timestamp(
        self,
        channel: str,
        coin: str | None,
        exchange_timestamp_ms: int,
        received_timestamp_ms: int,
    ) -> list[GapRecord]:
        new_gaps: list[GapRecord] = []
        key = self._key(channel, coin)

        prev_exchange = self._last_exchange_ts.get(key)
        if prev_exchange is not None:
            if exchange_timestamp_ms < prev_exchange:
                self._backward_timestamps += 1
                gap = GapRecord(
                    gap_type="BACKWARD_EXCHANGE_TIMESTAMP",
                    coin=coin,
                    channel=channel,
                    started_at_ms=prev_exchange,
                    ended_at_ms=exchange_timestamp_ms,
                    duration_ms=exchange_timestamp_ms - prev_exchange,
                    certainty=GapCertainty.INFERRED.value,
                )
                self._gaps.append(gap)
                new_gaps.append(gap)
            elif exchange_timestamp_ms == prev_exchange:
                self._duplicate_timestamps += 1

        self._last_exchange_ts[key] = exchange_timestamp_ms

        prev_receive = self._last_receive_ts.get(key)
        if prev_receive is not None:
            gap_ms = received_timestamp_ms - prev_receive
            if gap_ms > 60000:
                gap = GapRecord(
                    gap_type="RECEIVE_TIME_GAP",
                    coin=coin,
                    channel=channel,
                    started_at_ms=prev_receive,
                    ended_at_ms=received_timestamp_ms,
                    duration_ms=gap_ms,
                    certainty=GapCertainty.INFERRED.value,
                )
                self._gaps.append(gap)
                new_gaps.append(gap)

        self._last_receive_ts[key] = received_timestamp_ms
        return new_gaps

    def record_candle(self, coin: str, open_time_ms: int) -> list[GapRecord]:
        new_gaps: list[GapRecord] = []
        key = f"candle:{coin}"
        prev = self._last_candle_open.get(key)
        if prev is not None:
            expected = prev + self._candle_interval_ms
            if open_time_ms > expected:
                gap = GapRecord(
                    gap_type="MISSING_CANDLE_PERIOD",
                    coin=coin,
                    channel="candle",
                    started_at_ms=expected,
                    ended_at_ms=open_time_ms,
                    duration_ms=open_time_ms - expected,
                    certainty=GapCertainty.INFERRED.value,
                    details={"expected_interval_ms": self._candle_interval_ms},
                )
                self._gaps.append(gap)
                new_gaps.append(gap)
        self._last_candle_open[key] = open_time_ms
        return new_gaps

    def record_channel_staleness(
        self,
        channel: str,
        coin: str | None,
        last_message_at_ms: int,
        threshold_ms: int,
    ) -> GapRecord | None:
        now = int(time.time() * 1000)
        if now - last_message_at_ms > threshold_ms:
            gap = GapRecord(
                gap_type="CHANNEL_STALENESS",
                coin=coin,
                channel=channel,
                started_at_ms=last_message_at_ms,
                ended_at_ms=now,
                duration_ms=now - last_message_at_ms,
                certainty=GapCertainty.INFERRED.value,
            )
            self._gaps.append(gap)
            return gap
        return None

    @property
    def gaps(self) -> list[GapRecord]:
        return list(self._gaps)

    @property
    def gap_count(self) -> int:
        return len(self._gaps)

    def summary(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        for gap in self._gaps:
            by_type[gap.gap_type] = by_type.get(gap.gap_type, 0) + 1
        return {
            "total_gaps": len(self._gaps),
            "by_type": by_type,
            "duplicate_timestamps": self._duplicate_timestamps,
            "backward_timestamps": self._backward_timestamps,
            "gaps": [
                {
                    "gap_type": g.gap_type,
                    "coin": g.coin,
                    "channel": g.channel,
                    "started_at_ms": g.started_at_ms,
                    "ended_at_ms": g.ended_at_ms,
                    "duration_ms": g.duration_ms,
                    "certainty": g.certainty,
                }
                for g in self._gaps
            ],
        }
