"""Pure schema and calculation helpers for bid/ask CSV rows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

PAPER_ONLY = True
EXECUTION_ENABLED = False

SAFETY_STATEMENT = (
    "Bid/ask schema helpers are read-only, paper-trading only, and research-only. "
    "No wallets, private keys, signing, leverage, margin, exchange actions, "
    "real orders, deposits, withdrawals, or execution endpoints are used."
)

FORBIDDEN_MODULE_NAMES = frozenset(
    {
        "wallet",
        "wallet_client",
        "exchange_client",
        "order_executor",
        "live_trader",
        "execution_engine",
        "signing",
    }
)

FORBIDDEN_SOURCE_PATTERNS = frozenset(
    {
        "private_key",
        "eth_account",
        "web3",
        "sign_transaction",
        "place_order",
        "execute_order",
        "submit_order",
    }
)

TIMESTAMP_MS = "timestamp_ms"
TIMESTAMP_ISO = "timestamp_iso"
COIN = "coin"
BEST_BID = "best_bid"
BEST_ASK = "best_ask"
MID_PRICE = "mid_price"
SPREAD = "spread"
SPREAD_BPS = "spread_bps"
BID_SIZE = "bid_size"
ASK_SIZE = "ask_size"
BBO_IMBALANCE = "bbo_imbalance"
SOURCE_SESSION_ID = "source_session_id"
SOURCE_CHANNEL = "source_channel"

BID_ASK_CSV_COLUMNS = (
    TIMESTAMP_MS,
    TIMESTAMP_ISO,
    COIN,
    BEST_BID,
    BEST_ASK,
    MID_PRICE,
    SPREAD,
    SPREAD_BPS,
    BID_SIZE,
    ASK_SIZE,
    BBO_IMBALANCE,
    SOURCE_SESSION_ID,
    SOURCE_CHANNEL,
)


def validate_bid_ask(best_bid: float, best_ask: float) -> None:
    """Raise ValueError when ask is below bid."""
    if best_ask < best_bid:
        raise ValueError(f"best_ask ({best_ask}) must be >= best_bid ({best_bid})")


def compute_mid_price(best_bid: float, best_ask: float) -> float:
    validate_bid_ask(best_bid, best_ask)
    return (best_bid + best_ask) / 2.0


def compute_spread(best_bid: float, best_ask: float) -> float:
    validate_bid_ask(best_bid, best_ask)
    return best_ask - best_bid


def compute_spread_bps(best_bid: float, best_ask: float) -> float:
    validate_bid_ask(best_bid, best_ask)
    spread = best_ask - best_bid
    mid = (best_bid + best_ask) / 2.0
    if mid <= 0:
        return 0.0
    return spread / mid * 10000.0


def compute_bbo_imbalance(
    bid_size: float | None,
    ask_size: float | None,
) -> float | None:
    if bid_size is None or ask_size is None:
        return None
    total = bid_size + ask_size
    if total <= 0:
        return None
    return (bid_size - ask_size) / total


def timestamp_ms_to_iso(timestamp_ms: int) -> str:
    dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def build_bid_ask_row(
    *,
    timestamp_ms: int,
    coin: str,
    best_bid: float,
    best_ask: float,
    bid_size: float | None = None,
    ask_size: float | None = None,
    source_session_id: str = "",
    source_channel: str = "",
) -> dict[str, Any]:
    validate_bid_ask(best_bid, best_ask)
    spread = compute_spread(best_bid, best_ask)
    mid_price = compute_mid_price(best_bid, best_ask)
    spread_bps = compute_spread_bps(best_bid, best_ask)
    return {
        TIMESTAMP_MS: timestamp_ms,
        TIMESTAMP_ISO: timestamp_ms_to_iso(timestamp_ms),
        COIN: coin,
        BEST_BID: best_bid,
        BEST_ASK: best_ask,
        MID_PRICE: mid_price,
        SPREAD: spread,
        SPREAD_BPS: spread_bps,
        BID_SIZE: bid_size,
        ASK_SIZE: ask_size,
        BBO_IMBALANCE: compute_bbo_imbalance(bid_size, ask_size),
        SOURCE_SESSION_ID: source_session_id,
        SOURCE_CHANNEL: source_channel,
    }
