"""Message normalization from raw Hyperliquid WebSocket payloads."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from market_data.event_models import (
    AssetContextEvent,
    BboEvent,
    CandleEvent,
    MidPriceEvent,
    OrderBookEvent,
    TradeEvent,
)

logger = logging.getLogger(__name__)

KNOWN_TRADE_FIELDS = frozenset({"coin", "side", "px", "sz", "time", "hash", "tid"})
KNOWN_BBO_FIELDS = frozenset({"coin", "time", "bbo"})
KNOWN_L2_FIELDS = frozenset({"coin", "time", "levels"})
KNOWN_CANDLE_FIELDS = frozenset({"t", "T", "s", "i", "o", "h", "l", "c", "v", "n"})
KNOWN_CTX_FIELDS = frozenset({"coin", "ctx"})
KNOWN_CTX_INNER = frozenset(
    {"funding", "openInterest", "prevDayPx", "dayNtlVlm", "markPx", "midPx", "oraclePx"}
)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extras(data: dict[str, Any], known: frozenset[str]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if k not in known}


def normalize_message(
    channel: str,
    data: Any,
    received_timestamp_ms: int,
    l2_depth_levels: int = 20,
) -> list[Any]:
    """Normalize a WebSocket message into typed event objects."""
    events: list[Any] = []

    if channel == "allMids":
        events.extend(_normalize_all_mids(data, received_timestamp_ms))
    elif channel == "trades":
        events.extend(_normalize_trades(data, received_timestamp_ms))
    elif channel == "bbo":
        event = _normalize_bbo(data, received_timestamp_ms)
        if event:
            events.append(event)
    elif channel == "l2Book":
        event = _normalize_l2_book(data, received_timestamp_ms, l2_depth_levels)
        if event:
            events.append(event)
    elif channel == "candle":
        event = _normalize_candle(data, received_timestamp_ms)
        if event:
            events.append(event)
    elif channel == "activeAssetCtx":
        event = _normalize_asset_ctx(data, received_timestamp_ms)
        if event:
            events.append(event)

    return events


def _normalize_all_mids(data: Any, received_ms: int) -> list[MidPriceEvent]:
    events = []
    mids = data.get("mids", data) if isinstance(data, dict) else {}
    if not isinstance(mids, dict):
        return events
    for coin, price_str in mids.items():
        price = _safe_float(price_str)
        if price is not None:
            events.append(
                MidPriceEvent(
                    coin=coin,
                    mid_price=price,
                    exchange_timestamp_ms=None,
                    received_timestamp_ms=received_ms,
                )
            )
    return events


def _normalize_trades(data: Any, received_ms: int) -> list[TradeEvent]:
    events = []
    trades = data if isinstance(data, list) else [data]
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        coin = trade.get("coin", "")
        side = trade.get("side", "")
        price = _safe_float(trade.get("px"))
        size = _safe_float(trade.get("sz"))
        trade_id = trade.get("tid", 0)
        tx_hash = trade.get("hash", "")
        exchange_ts = trade.get("time")
        if price is None or size is None or exchange_ts is None:
            continue
        extras = _extras(trade, KNOWN_TRADE_FIELDS)
        events.append(
            TradeEvent(
                coin=coin,
                side=side,
                price=price,
                size=size,
                trade_id=trade_id,
                transaction_hash=tx_hash,
                exchange_timestamp_ms=int(exchange_ts),
                received_timestamp_ms=received_ms,
                extras=extras,
            )
        )
    return events


def _normalize_bbo(data: Any, received_ms: int) -> BboEvent | None:
    if not isinstance(data, dict):
        return None
    coin = data.get("coin", "")
    exchange_ts = data.get("time")
    bbo = data.get("bbo", [])
    if len(bbo) < 2:
        return None
    bid = bbo[0] if isinstance(bbo[0], dict) else {}
    ask = bbo[1] if isinstance(bbo[1], dict) else {}
    bid_px = _safe_float(bid.get("px"))
    bid_sz = _safe_float(bid.get("sz"))
    ask_px = _safe_float(ask.get("px"))
    ask_sz = _safe_float(ask.get("sz"))
    if bid_px is None or ask_px is None:
        return None
    bid_sz = bid_sz or 0.0
    ask_sz = ask_sz or 0.0
    spread = ask_px - bid_px
    mid = (bid_px + ask_px) / 2
    spread_bps = (spread / mid * 10000) if mid > 0 else 0.0
    extras = _extras(data, KNOWN_BBO_FIELDS)
    return BboEvent(
        coin=coin,
        best_bid_price=bid_px,
        best_bid_size=bid_sz,
        best_ask_price=ask_px,
        best_ask_size=ask_sz,
        spread=spread,
        spread_bps=spread_bps,
        exchange_timestamp_ms=int(exchange_ts) if exchange_ts else None,
        received_timestamp_ms=received_ms,
        extras=extras,
    )


def _normalize_l2_book(
    data: Any, received_ms: int, depth_levels: int
) -> OrderBookEvent | None:
    if not isinstance(data, dict):
        return None
    coin = data.get("coin", "")
    exchange_ts = data.get("time")
    levels = data.get("levels", [])
    if len(levels) < 2:
        return None
    bid_levels = levels[0] if isinstance(levels[0], list) else []
    ask_levels = levels[1] if isinstance(levels[1], list) else []

    bids: list[tuple[float, float]] = []
    for lvl in bid_levels[:depth_levels]:
        if isinstance(lvl, dict):
            px = _safe_float(lvl.get("px"))
            sz = _safe_float(lvl.get("sz"))
            if px is not None and sz is not None:
                bids.append((px, sz))

    asks: list[tuple[float, float]] = []
    for lvl in ask_levels[:depth_levels]:
        if isinstance(lvl, dict):
            px = _safe_float(lvl.get("px"))
            sz = _safe_float(lvl.get("sz"))
            if px is not None and sz is not None:
                asks.append((px, sz))

    if not bids or not asks:
        return None

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    midpoint = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    total_bid = sum(sz for _, sz in bids)
    total_ask = sum(sz for _, sz in asks)
    extras = _extras(data, KNOWN_L2_FIELDS)

    return OrderBookEvent(
        coin=coin,
        bids=tuple(bids),
        asks=tuple(asks),
        best_bid_price=best_bid,
        best_ask_price=best_ask,
        midpoint=midpoint,
        spread=spread,
        total_bid_depth=total_bid,
        total_ask_depth=total_ask,
        exchange_timestamp_ms=int(exchange_ts) if exchange_ts else None,
        received_timestamp_ms=received_ms,
        extras=extras,
    )


def _normalize_candle(data: Any, received_ms: int) -> CandleEvent | None:
    if not isinstance(data, dict):
        return None
    coin = data.get("s", "")
    interval = data.get("i", "")
    open_time = data.get("t")
    close_time = data.get("T")
    o = _safe_float(data.get("o"))
    h = _safe_float(data.get("h"))
    lo = _safe_float(data.get("l"))
    c = _safe_float(data.get("c"))
    v = _safe_float(data.get("v"))
    n = data.get("n", 0)
    if open_time is None or o is None:
        return None
    extras = _extras(data, KNOWN_CANDLE_FIELDS)
    return CandleEvent(
        coin=coin,
        interval=interval,
        open_time_ms=int(open_time),
        close_time_ms=int(close_time) if close_time else int(open_time),
        open=o,
        high=h or o,
        low=lo or o,
        close=c or o,
        volume=v or 0.0,
        trade_count=int(n),
        received_timestamp_ms=received_ms,
        extras=extras,
    )


def _normalize_asset_ctx(data: Any, received_ms: int) -> AssetContextEvent | None:
    if not isinstance(data, dict):
        return None
    coin = data.get("coin", "")
    ctx = data.get("ctx", {})
    if not isinstance(ctx, dict):
        ctx = {}
    extras_outer = _extras(data, KNOWN_CTX_FIELDS)
    extras_inner = _extras(ctx, KNOWN_CTX_INNER)
    extras = {**extras_outer, **extras_inner}

    return AssetContextEvent(
        coin=coin,
        mark_price=_safe_float(ctx.get("markPx")),
        mid_price=_safe_float(ctx.get("midPx")),
        oracle_price=_safe_float(ctx.get("oraclePx")),
        funding_rate=_safe_float(ctx.get("funding")),
        open_interest=_safe_float(ctx.get("openInterest")),
        daily_notional_volume=_safe_float(ctx.get("dayNtlVlm")),
        previous_day_price=_safe_float(ctx.get("prevDayPx")),
        received_timestamp_ms=received_ms,
        extras=extras,
    )


def dedup_key(event: Any) -> str:
    """Generate a deterministic deduplication key for a normalized event."""
    if isinstance(event, TradeEvent):
        return f"trade:{event.coin}:{event.exchange_timestamp_ms}:{event.trade_id}"

    payload = _event_to_dict(event)
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _event_to_dict(event: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(event)
