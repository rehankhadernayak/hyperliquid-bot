"""Feature dataset serialization and deterministic hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from feature_research.feature_models import (
    FEATURE_COLUMN_NAMES,
    LABEL_COLUMN_NAMES,
    FeatureLabel,
    FeatureRow,
)


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def row_to_dataset_record(row: FeatureRow, labels: dict[str, Any] | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "feature_row_id": row.feature_row_id,
        "replay_session_id": row.replay_session_id,
        "source_recorder_session_id": row.source_recorder_session_id,
        "coin": row.coin,
        "event_timestamp_ms": row.event_timestamp_ms,
        "received_timestamp_ms": row.received_timestamp_ms,
        "latest_trade_price": row.latest_trade_price,
        "latest_midpoint": row.latest_midpoint,
        "best_bid": row.best_bid,
        "best_ask": row.best_ask,
        "spread": row.spread,
        "spread_bps": row.spread_bps,
        "recent_trade_count": row.recent_trade_count,
        "recent_volume": row.recent_volume,
        "current_event_type": row.current_event_type,
        "source_replay_sequence": row.source_replay_sequence,
        "feature_config_hash": row.feature_config_hash,
    }
    for name in FEATURE_COLUMN_NAMES:
        record[f"feature_{name}"] = row.features.get(name)
        record[f"feature_missing_{name}"] = row.missing_flags.get(name, False)
    if labels:
        for name in LABEL_COLUMN_NAMES:
            record[f"label_{name}"] = labels.get(name)
    return record


def build_dataset_records(
    rows: tuple[FeatureRow, ...],
    labels: list[FeatureLabel],
) -> list[dict[str, Any]]:
    label_map = {label.feature_row_id: label.labels for label in labels}
    return [row_to_dataset_record(row, label_map.get(row.feature_row_id)) for row in rows]


def compute_dataset_hash(
    *,
    stream_hash: str,
    config_hash: str,
    rows: tuple[FeatureRow, ...],
    labels: list[FeatureLabel],
    analysis_results: list[Any],
) -> str:
    row_entries = [_stable_json(row.to_dict()) for row in rows]
    label_entries = [_stable_json(label.to_dict()) for label in labels]
    analysis_entries = [
        _stable_json(
            {
                "feature_name": r.feature_name,
                "horizon_seconds": r.horizon_seconds,
                "correlation": r.correlation,
                "information_coefficient": r.information_coefficient,
                "candidate_signal": r.candidate_signal,
            }
        )
        for r in analysis_results
    ]
    payload = {
        "stream_hash": stream_hash,
        "config_hash": config_hash,
        "rows": row_entries,
        "labels": label_entries,
        "analysis": analysis_entries,
    }
    return hashlib.sha256(_stable_json(payload).encode()).hexdigest()


def write_dataset(
    records: list[dict[str, Any]],
    output_path: Path,
) -> tuple[Path, str]:
    """Write dataset as Parquet with JSONL fallback."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    format_used = "parquet"

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(records)
        pq.write_table(table, str(output_path))
    except Exception:
        format_used = "jsonl"
        jsonl_path = output_path.with_suffix(".jsonl")
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(_stable_json(record) + "\n")
        return jsonl_path, format_used

    return output_path, format_used
