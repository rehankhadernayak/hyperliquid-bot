"""Replay clock for timed or fast-forward event emission."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

SleepFn = Callable[[float], None]


def _default_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


@dataclass
class ReplayClock:
    """Controls replay pacing with speed multiplier and optional event limits.

    In test mode (sleep_fn=None), no real sleeping occurs — delays are recorded only.
    """

    speed: float = 1.0
    start_timestamp_ms: int | None = None
    end_timestamp_ms: int | None = None
    max_events: int | None = None
    test_mode: bool = False
    sleep_fn: SleepFn | None = field(default=None, repr=False)

    _paused: bool = field(default=False, init=False)
    _events_emitted: int = field(default=0, init=False)
    _last_recorded_time_ms: int | None = field(default=None, init=False)
    _replay_wall_start_ms: int | None = field(default=None, init=False)
    _total_sleep_seconds: float = field(default=0.0, init=False)
    _step_mode: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.test_mode:
            self.sleep_fn = lambda _s: None
        elif self.sleep_fn is None:
            self.sleep_fn = _default_sleep

    @property
    def fast_mode(self) -> bool:
        return self.speed == 0

    @property
    def events_emitted(self) -> int:
        return self._events_emitted

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def total_sleep_seconds(self) -> float:
        return self._total_sleep_seconds

    def start(self, first_event_time_ms: int) -> None:
        self._replay_wall_start_ms = int(time.time() * 1000)
        self._last_recorded_time_ms = first_event_time_ms

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def enable_step_mode(self) -> None:
        self._step_mode = True
        self._paused = True

    def step_one_event(self) -> None:
        """Allow exactly one event through step mode."""
        self._paused = False
        self._step_mode = True

    def should_emit(self, event_time_ms: int) -> bool:
        if self._paused and not self._step_mode:
            return False
        if self.start_timestamp_ms is not None and event_time_ms < self.start_timestamp_ms:
            return False
        if self.end_timestamp_ms is not None and event_time_ms > self.end_timestamp_ms:
            return False
        if self.max_events is not None and self._events_emitted >= self.max_events:
            return False
        return True

    def wait_until(self, event_time_ms: int) -> None:
        """Wait until the event's recorded time, respecting speed multiplier."""
        if self.fast_mode or self.test_mode:
            return

        if self._last_recorded_time_ms is None:
            self._last_recorded_time_ms = event_time_ms
            return

        delta_ms = max(0, event_time_ms - self._last_recorded_time_ms)
        self._last_recorded_time_ms = event_time_ms

        if self.speed <= 0:
            return

        delay_sec = (delta_ms / 1000.0) / self.speed
        if delay_sec > 0 and self.sleep_fn is not None:
            self.sleep_fn(delay_sec)
            self._total_sleep_seconds += delay_sec

    def record_emission(self) -> None:
        self._events_emitted += 1
        if self._step_mode:
            self._paused = True

    def replay_wall_time_ms(self) -> int:
        if self._replay_wall_start_ms is None:
            return int(time.time() * 1000)
        if self.test_mode:
            return self._replay_wall_start_ms + int(self._total_sleep_seconds * 1000)
        return int(time.time() * 1000)

    def reset(self) -> None:
        self._paused = False
        self._events_emitted = 0
        self._last_recorded_time_ms = None
        self._replay_wall_start_ms = None
        self._total_sleep_seconds = 0.0
        self._step_mode = False
