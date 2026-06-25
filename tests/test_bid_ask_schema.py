"""Focused offline tests for bid/ask CSV schema helpers."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from bid_ask_schema import (
    BID_ASK_CSV_COLUMNS,
    FORBIDDEN_MODULE_NAMES,
    build_bid_ask_row,
    compute_bbo_imbalance,
    compute_mid_price,
    compute_spread,
    compute_spread_bps,
    timestamp_ms_to_iso,
    validate_bid_ask,
)

SCHEMA_FILE = Path(__file__).resolve().parent.parent / "bid_ask_schema.py"


class TestMidPrice:
    def test_mid_price(self) -> None:
        assert compute_mid_price(100.0, 102.0) == pytest.approx(101.0)

    def test_mid_price_when_bid_equals_ask(self) -> None:
        assert compute_mid_price(50.0, 50.0) == pytest.approx(50.0)


class TestSpread:
    def test_spread(self) -> None:
        assert compute_spread(63747.0, 63748.0) == pytest.approx(1.0)

    def test_zero_spread(self) -> None:
        assert compute_spread(100.0, 100.0) == pytest.approx(0.0)


class TestSpreadBps:
    def test_spread_bps(self) -> None:
        assert compute_spread_bps(63747.0, 63748.0) == pytest.approx(
            1.0 / 63747.5 * 10000.0
        )

    def test_zero_spread_bps(self) -> None:
        assert compute_spread_bps(100.0, 100.0) == pytest.approx(0.0)


class TestBboImbalance:
    def test_bbo_imbalance(self) -> None:
        assert compute_bbo_imbalance(3.0, 1.0) == pytest.approx(0.5)

    def test_missing_bid_size(self) -> None:
        assert compute_bbo_imbalance(None, 1.0) is None

    def test_missing_ask_size(self) -> None:
        assert compute_bbo_imbalance(1.0, None) is None

    def test_zero_total_size(self) -> None:
        assert compute_bbo_imbalance(0.0, 0.0) is None


class TestBidAskValidation:
    def test_ask_below_bid_invalid(self) -> None:
        with pytest.raises(ValueError, match="best_ask .* must be >= best_bid"):
            validate_bid_ask(102.0, 100.0)

    def test_compute_mid_price_rejects_invalid_quotes(self) -> None:
        with pytest.raises(ValueError):
            compute_mid_price(102.0, 100.0)


class TestTimestampIso:
    def test_timestamp_iso_conversion(self) -> None:
        assert timestamp_ms_to_iso(1_718_000_001_000) == "2024-06-10T06:13:21Z"


class TestBuildBidAskRow:
    def test_build_row_contains_all_columns(self) -> None:
        row = build_bid_ask_row(
            timestamp_ms=1_718_000_001_000,
            coin="BTC",
            best_bid=63747.0,
            best_ask=63748.0,
            bid_size=2.0,
            ask_size=1.0,
            source_session_id="session-1",
            source_channel="bbo",
        )
        assert list(row.keys()) == list(BID_ASK_CSV_COLUMNS)
        assert row["timestamp_ms"] == 1_718_000_001_000
        assert row["timestamp_iso"] == "2024-06-10T06:13:21Z"
        assert row["coin"] == "BTC"
        assert row["best_bid"] == pytest.approx(63747.0)
        assert row["best_ask"] == pytest.approx(63748.0)
        assert row["mid_price"] == pytest.approx(63747.5)
        assert row["spread"] == pytest.approx(1.0)
        assert row["spread_bps"] == pytest.approx(1.0 / 63747.5 * 10000.0)
        assert row["bid_size"] == pytest.approx(2.0)
        assert row["ask_size"] == pytest.approx(1.0)
        assert row["bbo_imbalance"] == pytest.approx(1.0 / 3.0)
        assert row["source_session_id"] == "session-1"
        assert row["source_channel"] == "bbo"

    def test_build_row_missing_sizes(self) -> None:
        row = build_bid_ask_row(
            timestamp_ms=1_718_000_001_000,
            coin="ETH",
            best_bid=3500.0,
            best_ask=3500.5,
        )
        assert row["bid_size"] is None
        assert row["ask_size"] is None
        assert row["bbo_imbalance"] is None


class TestSafety:
    def test_no_wallet_signing_order_execution_code(self) -> None:
        source = SCHEMA_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        lowered = source.lower()
        assert "requests" not in lowered
        assert "urllib" not in lowered
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES

    def test_paper_only_constants(self) -> None:
        from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY

        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False
