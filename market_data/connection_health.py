"""Connection health tracking including heartbeats."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConnectionState(Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    STOPPED = "STOPPED"


@dataclass
class HeartbeatStats:
    ping_count: int = 0
    pong_count: int = 0
    timeout_count: int = 0
    last_ping_at_ms: int | None = None
    last_pong_at_ms: int | None = None
    last_round_trip_ms: float | None = None
    round_trip_samples: list[float] = field(default_factory=list)

    def record_ping(self, at_ms: int) -> None:
        self.ping_count += 1
        self.last_ping_at_ms = at_ms

    def record_pong(self, at_ms: int) -> float | None:
        self.pong_count += 1
        self.last_pong_at_ms = at_ms
        if self.last_ping_at_ms is not None:
            rtt = at_ms - self.last_ping_at_ms
            self.last_round_trip_ms = rtt
            self.round_trip_samples.append(rtt)
            return rtt
        return None

    def record_timeout(self) -> None:
        self.timeout_count += 1

    def to_dict(self) -> dict[str, Any]:
        samples = self.round_trip_samples
        median = None
        if samples:
            sorted_s = sorted(samples)
            median = sorted_s[len(sorted_s) // 2]
        return {
            "ping_count": self.ping_count,
            "pong_count": self.pong_count,
            "timeout_count": self.timeout_count,
            "last_ping_at_ms": self.last_ping_at_ms,
            "last_pong_at_ms": self.last_pong_at_ms,
            "last_round_trip_ms": self.last_round_trip_ms,
            "median_round_trip_ms": median,
        }


@dataclass
class ConnectionHealth:
    """Track WebSocket connection health metrics."""

    state: ConnectionState = ConnectionState.DISCONNECTED
    reconnect_count: int = 0
    reconnect_generation: int = 0
    connected_at_ms: int | None = None
    last_message_at_ms: int | None = None
    last_disconnect_reason: str | None = None
    last_disconnect_at_ms: int | None = None
    heartbeat: HeartbeatStats = field(default_factory=HeartbeatStats)
    channel_message_counts: dict[str, int] = field(default_factory=dict)
    malformed_count: int = 0
    unsupported_count: int = 0
    degraded: bool = False
    raw_dropped_count: int = 0
    normalized_dropped_count: int = 0

    def set_state(self, state: ConnectionState) -> None:
        self.state = state
        if state == ConnectionState.CONNECTED:
            self.connected_at_ms = int(time.time() * 1000)

    def record_message(self, channel: str) -> None:
        now = int(time.time() * 1000)
        self.last_message_at_ms = now
        self.channel_message_counts[channel] = self.channel_message_counts.get(channel, 0) + 1

    def record_disconnect(self, reason: str) -> None:
        self.last_disconnect_reason = reason
        self.last_disconnect_at_ms = int(time.time() * 1000)
        self.state = ConnectionState.DISCONNECTED

    def record_reconnect(self) -> None:
        self.reconnect_count += 1
        self.reconnect_generation += 1
        self.state = ConnectionState.RECONNECTING

    def uptime_seconds(self) -> float:
        if self.connected_at_ms is None:
            return 0.0
        return (int(time.time() * 1000) - self.connected_at_ms) / 1000.0

    def messages_per_second(self) -> float:
        uptime = self.uptime_seconds()
        if uptime <= 0:
            return 0.0
        total = sum(self.channel_message_counts.values())
        return total / uptime

    def summary(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "reconnect_count": self.reconnect_count,
            "reconnect_generation": self.reconnect_generation,
            "connected_at_ms": self.connected_at_ms,
            "last_message_at_ms": self.last_message_at_ms,
            "last_disconnect_reason": self.last_disconnect_reason,
            "uptime_seconds": self.uptime_seconds(),
            "messages_per_second": round(self.messages_per_second(), 2),
            "channel_message_counts": dict(self.channel_message_counts),
            "malformed_count": self.malformed_count,
            "unsupported_count": self.unsupported_count,
            "degraded": self.degraded,
            "raw_dropped_count": self.raw_dropped_count,
            "normalized_dropped_count": self.normalized_dropped_count,
            "heartbeat": self.heartbeat.to_dict(),
        }
