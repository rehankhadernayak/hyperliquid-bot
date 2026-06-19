"""Offline tests for raw and normalized writers."""

from __future__ import annotations

import json
import time

import pytest

from market_data.event_models import TradeEvent
from market_data.normalized_writer import NormalizedWriter
from market_data.raw_writer import RawWriter


class TestRawWriter:
    def test_append_only_raw_events(self, tmp_path):
        writer = RawWriter(
            output_dir=tmp_path,
            recorder_session_id="test-session",
            network="mainnet",
            flush_interval_seconds=0.1,
        )
        payload = {"channel": "trades", "data": [{"coin": "BTC"}]}
        writer.enqueue(
            channel="trades",
            subscription={"type": "trades", "coin": "BTC"},
            raw_payload=payload,
            exchange_timestamp_ms=1000,
            received_timestamp_ms=2000,
        )
        writer.shutdown()
        assert writer.row_count == 1
        raw_files = list(tmp_path.rglob("*.jsonl"))
        assert len(raw_files) >= 1
        content = raw_files[0].read_text(encoding="utf-8")
        record = json.loads(content.strip())
        assert record["schema_version"] == "1.0.0"
        assert record["recorder_session_id"] == "test-session"
        assert record["source"] == "hyperliquid_websocket"
        assert record["sequence"] == 1
        assert "raw_payload_hash" in record

    def test_raw_events_never_mutated(self, tmp_path):
        writer = RawWriter(
            output_dir=tmp_path,
            recorder_session_id="test-session",
            network="mainnet",
            flush_interval_seconds=0.1,
        )
        payload = {"channel": "trades", "data": "original"}
        writer.enqueue(
            channel="trades",
            subscription=None,
            raw_payload=payload,
            exchange_timestamp_ms=None,
            received_timestamp_ms=1000,
        )
        writer.shutdown()
        raw_files = list(tmp_path.rglob("*.jsonl"))
        record = json.loads(raw_files[0].read_text(encoding="utf-8").strip())
        assert record["raw_payload"]["data"] == "original"

    def test_sequence_numbers_increment(self, tmp_path):
        writer = RawWriter(
            output_dir=tmp_path,
            recorder_session_id="test-session",
            network="mainnet",
            flush_interval_seconds=0.1,
        )
        for i in range(5):
            writer.enqueue(
                channel="trades",
                subscription=None,
                raw_payload={"i": i},
                exchange_timestamp_ms=None,
                received_timestamp_ms=1000 + i,
            )
        writer.shutdown()
        assert writer.row_count == 5

    def test_file_rotation_by_hour(self, tmp_path):
        writer = RawWriter(
            output_dir=tmp_path,
            recorder_session_id="test-session",
            network="mainnet",
            flush_interval_seconds=0.1,
        )
        writer.enqueue(
            channel="trades",
            subscription=None,
            raw_payload={"t": 1},
            exchange_timestamp_ms=None,
            received_timestamp_ms=int(time.time() * 1000),
        )
        writer.shutdown()
        assert (tmp_path / "raw").exists()


class TestNormalizedWriter:
    def test_write_and_dedup(self, tmp_path):
        writer = NormalizedWriter(
            output_dir=tmp_path,
            batch_size=1,
        )
        event = TradeEvent(
            coin="BTC",
            side="B",
            price=100.0,
            size=1.0,
            trade_id=1,
            transaction_hash="0xabc",
            exchange_timestamp_ms=1000,
            received_timestamp_ms=2000,
        )
        writer.enqueue(event)
        writer.enqueue(event)
        writer.shutdown()
        assert writer.duplicate_count == 1
        assert writer.row_count == 1

    def test_partitioned_output(self, tmp_path):
        writer = NormalizedWriter(
            output_dir=tmp_path,
            batch_size=1,
        )
        event = TradeEvent(
            coin="BTC",
            side="B",
            price=100.0,
            size=1.0,
            trade_id=1,
            transaction_hash="0xabc",
            exchange_timestamp_ms=1000,
            received_timestamp_ms=2000,
        )
        writer.enqueue(event)
        writer.shutdown()
        normalized_dir = tmp_path / "normalized"
        assert normalized_dir.exists()
        partitions = list(normalized_dir.rglob("*"))
        assert len(partitions) > 0

    def test_bounded_queue_backpressure(self, tmp_path):
        writer = NormalizedWriter(
            output_dir=tmp_path,
            queue_maxsize=2,
            batch_size=100,
        )
        event = TradeEvent(
            coin="BTC",
            side="B",
            price=100.0,
            size=1.0,
            trade_id=1,
            transaction_hash="0xabc",
            exchange_timestamp_ms=1000,
            received_timestamp_ms=2000,
        )
        for i in range(5):
            e = TradeEvent(
                coin="BTC",
                side="B",
                price=100.0 + i,
                size=1.0,
                trade_id=i,
                transaction_hash=f"0x{i}",
                exchange_timestamp_ms=1000 + i,
                received_timestamp_ms=2000,
            )
            result = writer.enqueue(e)
            if not result:
                break
        writer.shutdown()
