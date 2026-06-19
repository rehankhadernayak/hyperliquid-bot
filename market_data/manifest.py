"""Recorder session manifest generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import VERSION
from market_data.recorder_config import SCHEMA_VERSION


def write_manifest(
    manifests_dir: Path,
    recorder_session_id: str,
    manifest_data: dict[str, Any],
) -> Path:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    path = manifests_dir / f"{recorder_session_id}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_data, handle, indent=2, default=str)
    return path


def build_manifest(
    recorder_session_id: str,
    config_dict: dict[str, Any],
    start_time_ms: int,
    end_time_ms: int | None,
    raw_row_count: int,
    normalized_row_count: int,
    duplicate_count: int,
    malformed_count: int,
    reconnect_count: int,
    heartbeat_stats: dict[str, Any],
    channel_stats: dict[str, Any],
    latency_stats: dict[str, Any],
    gap_stats: dict[str, Any],
    subscription_summary: dict[str, Any],
    files_written: list[str],
    shutdown_reason: str,
    configuration_hash: str,
    degraded: bool = False,
    raw_dropped: int = 0,
    normalized_dropped: int = 0,
) -> dict[str, Any]:
    return {
        "recorder_session_id": recorder_session_id,
        "start_time_ms": start_time_ms,
        "end_time_ms": end_time_ms,
        "network": config_dict.get("network"),
        "websocket_url": config_dict.get("websocket_url"),
        "coins": config_dict.get("coins"),
        "requested_subscriptions": config_dict.get("requested_subscriptions"),
        "acknowledged_subscriptions": subscription_summary.get("by_state", {}),
        "schema_version": SCHEMA_VERSION,
        "files_written": files_written,
        "raw_row_count": raw_row_count,
        "normalized_row_count": normalized_row_count,
        "duplicate_count": duplicate_count,
        "malformed_count": malformed_count,
        "reconnect_count": reconnect_count,
        "heartbeat_statistics": heartbeat_stats,
        "channel_statistics": channel_stats,
        "latency_statistics": latency_stats,
        "gap_statistics": gap_stats,
        "shutdown_reason": shutdown_reason,
        "configuration_hash": configuration_hash,
        "software_version": VERSION,
        "paper_only": True,
        "execution_enabled": False,
        "degraded": degraded,
        "raw_dropped_count": raw_dropped,
        "normalized_dropped_count": normalized_dropped,
    }
