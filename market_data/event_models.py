"""Immutable normalized event models for Hyperliquid market data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MidPriceEvent:
    coin: str
    mid_price: float
    exchange_timestamp_ms: int | None
    received_timestamp_ms: int
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TradeEvent:
    coin: str
    side: str
    price: float
    size: float
    trade_id: int | str
    transaction_hash: str
    exchange_timestamp_ms: int
    received_timestamp_ms: int
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BboEvent:
    coin: str
    best_bid_price: float
    best_bid_size: float
    best_ask_price: float
    best_ask_size: float
    spread: float
    spread_bps: float
    exchange_timestamp_ms: int | None
    received_timestamp_ms: int
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderBookEvent:
    coin: str
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]
    best_bid_price: float
    best_ask_price: float
    midpoint: float
    spread: float
    total_bid_depth: float
    total_ask_depth: float
    exchange_timestamp_ms: int | None
    received_timestamp_ms: int
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandleEvent:
    coin: str
    interval: str
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int
    received_timestamp_ms: int
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssetContextEvent:
    coin: str
    mark_price: float | None
    mid_price: float | None
    oracle_price: float | None
    funding_rate: float | None
    open_interest: float | None
    daily_notional_volume: float | None
    previous_day_price: float | None
    received_timestamp_ms: int
    extras: dict[str, Any] = field(default_factory=dict)


EVENT_TYPE_NAMES = {
    "MidPriceEvent": "mid",
    "TradeEvent": "trade",
    "BboEvent": "bbo",
    "OrderBookEvent": "l2_book",
    "CandleEvent": "candle",
    "AssetContextEvent": "asset_context",
}
