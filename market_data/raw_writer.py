"""Append-only raw event storage with atomic file rotation."""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from market_data.recorder_config import SCHEMA_VERSION

logger = logging.getLogger(__name__)

SOURCE = "hyperliquid_websocket"


class RawWriter:
    """Write raw WebSocket messages to append-only JSONL files."""

    def __init__(
        self,
        output_dir: Path,
        recorder_session_id: str,
        network: str,
        queue_maxsize: int = 10000,
        flush_interval_seconds: float = 5.0,
    ) -> None:
        self._output_dir = output_dir / "raw"
        self._recorder_session_id = recorder_session_id
        self._network = network
        self._flush_interval = flush_interval_seconds
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=queue_maxsize)
        self._sequence = 0
        self._reconnect_generation = 0
        self._row_count = 0
        self._files_written: set[str] = set()
        self._current_file: Path | None = None
        self._current_handle: Any = None
        self._current_hour_key: str | None = None
        self._lock = threading.Lock()
        self._stopped = False
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def files_written(self) -> list[str]:
        return sorted(self._files_written)

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def reconnect_generation(self) -> int:
        return self._reconnect_generation

    def set_reconnect_generation(self, generation: int) -> None:
        self._reconnect_generation = generation

    def enqueue(
        self,
        channel: str,
        subscription: dict[str, Any] | None,
        raw_payload: dict[str, Any],
        exchange_timestamp_ms: int | None,
        received_timestamp_ms: int,
        received_timestamp_ns: int | None = None,
    ) -> bool:
        """Enqueue a raw event. Returns False if queue is full (backpressure)."""
        self._sequence += 1
        payload_str = json.dumps(raw_payload, sort_keys=True, separators=(",", ":"))
        payload_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

        record = {
            "schema_version": SCHEMA_VERSION,
            "recorder_session_id": self._recorder_session_id,
            "network": self._network,
            "channel": channel,
            "subscription": subscription,
            "exchange_timestamp_ms": exchange_timestamp_ms,
            "received_timestamp_ms": received_timestamp_ms,
            "received_timestamp_ns": received_timestamp_ns,
            "sequence": self._sequence,
            "raw_payload": raw_payload,
            "raw_payload_hash": payload_hash,
            "reconnect_generation": self._reconnect_generation,
            "source": SOURCE,
        }

        try:
            self._queue.put_nowait(record)
            return True
        except queue.Full:
            logger.warning("Raw event queue full — dropping event (session DEGRADED)")
            return False

    def _hour_key(self, timestamp_ms: int) -> str:
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d/events-%H")

    def _rotate_file(self, hour_key: str) -> None:
        if self._current_hour_key == hour_key and self._current_handle is not None:
            return

        if self._current_handle is not None:
            self._current_handle.flush()
            self._current_handle.close()
            self._current_handle = None

        date_part = hour_key.split("/")[0]
        file_name = hour_key.split("/")[1] + ".jsonl"
        dir_path = self._output_dir / date_part
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / file_name

        self._current_file = file_path
        self._current_handle = file_path.open("a", encoding="utf-8")
        self._current_hour_key = hour_key
        self._files_written.add(str(file_path))

    def _write_record(self, record: dict[str, Any]) -> None:
        with self._lock:
            ts = record["received_timestamp_ms"]
            hour_key = self._hour_key(ts)
            self._rotate_file(hour_key)
            line = json.dumps(record, separators=(",", ":"), default=str)
            self._current_handle.write(line + "\n")
            self._row_count += 1

    def _writer_loop(self) -> None:
        last_flush = time.monotonic()
        while True:
            try:
                record = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stopped:
                    self._flush()
                    break
                if time.monotonic() - last_flush >= self._flush_interval:
                    self._flush()
                    last_flush = time.monotonic()
                continue

            if record is None:
                self._flush()
                break

            self._write_record(record)
            if time.monotonic() - last_flush >= self._flush_interval:
                self._flush()
                last_flush = time.monotonic()

    def _flush(self) -> None:
        with self._lock:
            if self._current_handle is not None:
                self._current_handle.flush()

    def shutdown(self) -> None:
        self._stopped = True
        self._queue.put(None)
        self._writer_thread.join(timeout=30)
        self._flush()
        with self._lock:
            if self._current_handle is not None:
                self._current_handle.close()
                self._current_handle = None
