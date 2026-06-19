"""Normalized event storage with Parquet (preferred) or JSONL fallback."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from market_data.event_models import EVENT_TYPE_NAMES
from market_data.normalizer import dedup_key

logger = logging.getLogger(__name__)

_PARQUET_AVAILABLE = False
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _PARQUET_AVAILABLE = True
except ImportError:
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]


class NormalizedWriter:
    """Write normalized events to partitioned storage."""

    def __init__(
        self,
        output_dir: Path,
        queue_maxsize: int = 5000,
        batch_size: int = 100,
    ) -> None:
        self._output_dir = output_dir / "normalized"
        self._queue: queue.Queue[Any | None] = queue.Queue(maxsize=queue_maxsize)
        self._batch_size = batch_size
        self._row_count = 0
        self._duplicate_count = 0
        self._seen_keys: set[str] = set()
        self._files_written: set[str] = set()
        self._batches: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._stopped = False
        self._use_parquet = _PARQUET_AVAILABLE
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def duplicate_count(self) -> int:
        return self._duplicate_count

    @property
    def files_written(self) -> list[str]:
        return sorted(self._files_written)

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def using_parquet(self) -> bool:
        return self._use_parquet

    def enqueue(self, event: Any) -> bool:
        key = dedup_key(event)
        if key in self._seen_keys:
            self._duplicate_count += 1
            return True

        self._seen_keys.add(key)
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            logger.warning("Normalized event queue full — dropping normalized event")
            return False

    def _event_type_name(self, event: Any) -> str:
        class_name = type(event).__name__
        return EVENT_TYPE_NAMES.get(class_name, class_name.lower())

    def _partition_path(self, event_type: str, coin: str, timestamp_ms: int) -> Path:
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        date_str = dt.strftime("%Y-%m-%d")
        return self._output_dir / f"event_type={event_type}" / f"date={date_str}" / f"coin={coin}"

    def _event_to_row(self, event: Any) -> dict[str, Any]:
        row = asdict(event)
        extras = row.pop("extras", {})
        if extras:
            row["extras_json"] = json.dumps(extras, default=str)
        for key, val in row.items():
            if isinstance(val, tuple):
                row[key] = json.dumps(val)
        return row

    def _flush_batch(self, partition_key: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        parts = partition_key.split("/")
        partition_dir = self._output_dir
        for part in parts:
            partition_dir = partition_dir / part
        partition_dir.mkdir(parents=True, exist_ok=True)

        ts = int(time.time() * 1000)
        if self._use_parquet:
            file_path = partition_dir / f"events-{ts}.parquet"
            try:
                table = pa.Table.from_pylist(rows)
                pq.write_table(table, str(file_path))
                self._files_written.add(str(file_path))
                self._row_count += len(rows)
            except Exception:
                logger.warning("Parquet write failed, falling back to JSONL")
                self._use_parquet = False
                self._write_jsonl(partition_dir, ts, rows)
        else:
            self._write_jsonl(partition_dir, ts, rows)

    def _write_jsonl(self, partition_dir: Path, ts: int, rows: list[dict[str, Any]]) -> None:
        file_path = partition_dir / f"events-{ts}.jsonl"
        with file_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=str) + "\n")
        self._files_written.add(str(file_path))
        self._row_count += len(rows)

    def _writer_loop(self) -> None:
        pending: dict[str, list[dict[str, Any]]] = {}
        while True:
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stopped:
                    self._flush_pending(pending)
                    break
                self._flush_pending(pending)
                pending = {}
                continue

            if event is None:
                self._flush_pending(pending)
                break

            event_type = self._event_type_name(event)
            coin = getattr(event, "coin", "global")
            ts = getattr(event, "received_timestamp_ms", int(time.time() * 1000))
            partition_key = f"event_type={event_type}/date={datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d')}/coin={coin}"

            row = self._event_to_row(event)
            pending.setdefault(partition_key, []).append(row)

            if len(pending.get(partition_key, [])) >= self._batch_size:
                self._flush_batch(partition_key, pending.pop(partition_key, []))

        self._flush_pending(pending)

    def _flush_pending(self, pending: dict[str, list[dict[str, Any]]]) -> None:
        for key, rows in pending.items():
            if rows:
                self._flush_batch(key, rows)
        pending.clear()

    def shutdown(self) -> None:
        self._stopped = True
        self._queue.put(None)
        self._writer_thread.join(timeout=30)
