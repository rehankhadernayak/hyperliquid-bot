"""Read-only status file for the market data recorder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATUS_FILE_NAME = "recorder_status.json"


def status_path(output_dir: Path) -> Path:
    return output_dir / STATUS_FILE_NAME


def write_status(output_dir: Path, status: dict[str, Any]) -> None:
    path = status_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2, default=str)


def read_status(output_dir: Path) -> dict[str, Any] | None:
    path = status_path(output_dir)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_status_summary(status: dict[str, Any]) -> str:
    lines = [
        "Market Data Recorder Status",
        "=" * 40,
        f"Running:        {status.get('running', False)}",
        f"Session ID:     {status.get('recorder_session_id', '—')}",
        f"Uptime:         {status.get('uptime_seconds', 0):.1f}s",
        f"Connection:     {status.get('connection_state', '—')}",
        f"Reconnects:     {status.get('reconnect_count', 0)}",
        f"Last message:   {status.get('last_message_at_ms', '—')}",
        f"Subscriptions:  {status.get('active_subscriptions', 0)} active",
        f"Events/channel: {status.get('channel_message_counts', {})}",
        f"Queue sizes:    raw={status.get('raw_queue_size', 0)}, "
        f"normalized={status.get('normalized_queue_size', 0)}",
        f"Latency p50:    {status.get('latency_p50_ms', '—')} ms",
        f"Latency p95:    {status.get('latency_p95_ms', '—')} ms",
        f"Latency p99:    {status.get('latency_p99_ms', '—')} ms",
        f"Malformed:      {status.get('malformed_count', 0)}",
        f"Duplicates:     {status.get('duplicate_count', 0)}",
        f"Gaps detected:  {status.get('gap_count', 0)}",
        f"Output dir:     {status.get('output_dir', '—')}",
    ]
    return "\n".join(lines)
