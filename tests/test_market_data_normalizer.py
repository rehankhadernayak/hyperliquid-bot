"""Offline tests for message normalization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from market_data.event_models import (
    AssetContextEvent,
    BboEvent,
    CandleEvent,
    MidPriceEvent,
    OrderBookEvent,
    TradeEvent,
)
from market_data.normalizer import dedup_key, normalize_message

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ws_messages.json"


def _load_fixtures() -> dict:
    with FIXTURES.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class TestNormalizer:
    @pytest.fixture(autouse=True)
    def fixtures(self):
        self.fixtures = _load_fixtures()
        self.received_ms = 1718000001000

    def test_normalize_trades(self):
        msg = self.fixtures["trade"]
        events = normalize_message("trades", msg["data"], self.received_ms)
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, TradeEvent)
        assert event.coin == "BTC"
        assert event.side == "B"
        assert event.price == 63747.5
        assert event.size == 0.1
        assert event.trade_id == 12345
        assert event.extras.get("extra_field") == "preserved"

    def test_normalize_bbo(self):
        msg = self.fixtures["bbo"]
        events = normalize_message("bbo", msg["data"], self.received_ms)
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, BboEvent)
        assert event.coin == "BTC"
        assert event.best_bid_price == 63747.0
        assert event.best_ask_price == 63748.0
        assert event.spread == 1.0
        assert event.extras.get("unknown_field") is True

    def test_normalize_l2_book(self):
        msg = self.fixtures["l2_book"]
        events = normalize_message("l2Book", msg["data"], self.received_ms)
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, OrderBookEvent)
        assert event.coin == "ETH"
        assert event.best_bid_price == 3500.0
        assert event.best_ask_price == 3500.5
        assert event.total_bid_depth == 15.0
        assert event.total_ask_depth == 11.0

    def test_normalize_candle(self):
        msg = self.fixtures["candle"]
        events = normalize_message("candle", msg["data"], self.received_ms)
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, CandleEvent)
        assert event.coin == "SOL"
        assert event.interval == "1m"
        assert event.open == 150.0
        assert event.volume == 1000.0
        assert event.trade_count == 250

    def test_normalize_asset_context(self):
        msg = self.fixtures["active_asset_ctx"]
        events = normalize_message("activeAssetCtx", msg["data"], self.received_ms)
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, AssetContextEvent)
        assert event.coin == "BTC"
        assert event.mark_price == 63747.5
        assert event.funding_rate == 0.0001
        assert event.extras.get("extra_ctx_field") == "kept"

    def test_normalize_all_mids(self):
        msg = self.fixtures["all_mids"]
        events = normalize_message("allMids", msg["data"], self.received_ms)
        assert len(events) == 3
        coins = {e.coin for e in events}
        assert coins == {"BTC", "ETH", "SOL"}
        assert all(isinstance(e, MidPriceEvent) for e in events)

    def test_trade_dedup_key(self):
        msg = self.fixtures["trade"]
        events = normalize_message("trades", msg["data"], self.received_ms)
        key1 = dedup_key(events[0])
        key2 = dedup_key(events[0])
        assert key1 == key2
        assert key1.startswith("trade:BTC:")

    def test_malformed_message_returns_empty(self):
        events = normalize_message("trades", "not_a_dict", self.received_ms)
        assert events == []

    def test_unknown_channel_returns_empty(self):
        events = normalize_message("unknownChannel", {"foo": "bar"}, self.received_ms)
        assert events == []

    def test_negative_latency_preserved(self):
        events = normalize_message(
            "trades",
            [{"coin": "BTC", "side": "B", "px": "100", "sz": "1", "time": 9999999999999, "hash": "x", "tid": 1}],
            1000,
        )
        assert len(events) == 1
        assert events[0].exchange_timestamp_ms > events[0].received_timestamp_ms
