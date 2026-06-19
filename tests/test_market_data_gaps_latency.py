"""Offline tests for gap detection and latency tracking."""

from __future__ import annotations

from market_data.gap_detection import GapDetector, GapCertainty
from market_data.latency_tracker import LatencyStats


class TestGapDetection:
    def test_reconnect_window_gap(self):
        detector = GapDetector()
        gap = detector.record_reconnect_window(1000, 5000, coin="BTC", channel="trades")
        assert gap.gap_type == "RECONNECT_WINDOW"
        assert gap.duration_ms == 4000
        assert gap.certainty == GapCertainty.INFERRED.value

    def test_backward_timestamp(self):
        detector = GapDetector()
        detector.record_exchange_timestamp("trades", "BTC", 5000, 5001)
        detector.record_exchange_timestamp("trades", "BTC", 3000, 6000)
        assert detector._backward_timestamps == 1
        gaps = [g for g in detector.gaps if g.gap_type == "BACKWARD_EXCHANGE_TIMESTAMP"]
        assert len(gaps) == 1

    def test_duplicate_timestamp(self):
        detector = GapDetector()
        detector.record_exchange_timestamp("trades", "BTC", 5000, 5001)
        detector.record_exchange_timestamp("trades", "BTC", 5000, 5002)
        assert detector._duplicate_timestamps == 1

    def test_receive_time_gap(self):
        detector = GapDetector()
        detector.record_exchange_timestamp("trades", "BTC", 1000, 1000)
        detector.record_exchange_timestamp("trades", "BTC", 2000, 70000)
        gaps = [g for g in detector.gaps if g.gap_type == "RECEIVE_TIME_GAP"]
        assert len(gaps) == 1

    def test_candle_gap_detection(self):
        detector = GapDetector(candle_interval_ms=60000)
        detector.record_candle("BTC", 0)
        gaps = detector.record_candle("BTC", 120000)
        assert len(gaps) == 1
        assert gaps[0].gap_type == "MISSING_CANDLE_PERIOD"

    def test_gap_summary(self):
        detector = GapDetector()
        detector.record_reconnect_window(0, 1000)
        summary = detector.summary()
        assert summary["total_gaps"] == 1
        assert "RECONNECT_WINDOW" in summary["by_type"]


class TestLatencyTracker:
    def test_basic_latency(self):
        stats = LatencyStats()
        stats.record(1000, 1500)
        stats.record(1000, 2000)
        assert stats.count == 2
        assert stats.minimum_ms == 500
        assert stats.maximum_ms == 1000

    def test_negative_latency(self):
        stats = LatencyStats()
        stats.record(5000, 1000)
        assert stats.negative_count == 1
        assert stats.minimum_ms == -4000

    def test_missing_timestamp(self):
        stats = LatencyStats()
        stats.record(None, 1000)
        assert stats.missing_timestamp_count == 1
        assert stats.count == 0

    def test_percentiles(self):
        stats = LatencyStats()
        for i in range(100):
            stats.record(0, i * 10)
        assert stats.median_ms is not None
        assert stats.p95_ms is not None
        assert stats.p99_ms is not None
        assert stats.p95_ms > stats.median_ms

    def test_to_dict(self):
        stats = LatencyStats()
        stats.record(1000, 1500)
        d = stats.to_dict()
        assert d["count"] == 1
        assert d["negative_latency_count"] == 0
