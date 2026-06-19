"""Safety constants and validation for event-driven backtesting."""

from __future__ import annotations

PAPER_ONLY = True
EXECUTION_ENABLED = False

ALLOWED_CHANNELS_V1 = frozenset({"trades", "bbo"})

EXCLUDED_CHANNELS_V1 = frozenset(
    {"l2Book", "candle", "activeAssetCtx", "allMids", "l2_book", "asset_context", "mid"}
)

OUTPUT_SUBDIR = "event_backtests"

SAFETY_STATEMENT = (
    "This event backtest is read-only, paper-trading only, and research-only. "
    "No wallets, private keys, signing, leverage, margin, exchange actions, "
    "real orders, deposits, withdrawals, or execution endpoints are used. "
    "Results are not connected to live trading decisions."
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
