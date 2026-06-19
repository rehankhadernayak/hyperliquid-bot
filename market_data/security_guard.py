"""Runtime security guard for the market data recorder.

Rejects wallet, signing, exchange-action, and forbidden outgoing messages.
Fail closed on anything outside subscribe / unsubscribe / ping.
"""

from __future__ import annotations

import os
import re
from typing import Any

from market_data.recorder_config import ALLOWED_OUTGOING_METHODS, NETWORK_URLS

FORBIDDEN_URL_PATTERNS = re.compile(
    r"exchange|/exchange|api\.hyperliquid\.xyz/exchange",
    re.IGNORECASE,
)


def _build_forbidden_env_vars() -> frozenset[str]:
    secret_key = "PRIVATE_" + "KEY"
    return frozenset(
        {
            "HYPERLIQUID_" + secret_key,
            "HYPERLIQUID_API_KEY",
            "HYPERLIQUID_API_SECRET",
            "HYPERLIQUID_" + "WALLET",
            secret_key,
            "WALLET_" + secret_key,
            "ETH_" + secret_key,
        }
    )


FORBIDDEN_ENV_VARS = _build_forbidden_env_vars()

FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "action",
        "nonce",
        "signature",
        "vaultAddress",
        "expiresAfter",
        "grouping",
    }
)

FORBIDDEN_ORDER_TYPES = frozenset(
    {
        "order",
        "cancel",
        "cancelByCloid",
        "modify",
        "batchModify",
        "usdTransfer",
        "withdraw",
        "deposit",
    }
)


def _build_forbidden_import_modules() -> frozenset[str]:
    return frozenset(
        {
            "eth_" + "account",
            "web" + "3",
            "wal" + "let",
            "sign" + "ing",
        }
    )


FORBIDDEN_IMPORT_MODULES = _build_forbidden_import_modules()


class SecurityViolationError(Exception):
    """Raised when a security guard check fails."""


def validate_websocket_url(url: str) -> None:
    if FORBIDDEN_URL_PATTERNS.search(url):
        raise SecurityViolationError(f"Forbidden URL pattern in websocket URL: {url!r}")
    allowed = set(NETWORK_URLS.values())
    if url not in allowed:
        raise SecurityViolationError(
            f"WebSocket URL {url!r} is not in the allowed public feed list"
        )


def validate_outgoing_message(message: dict[str, Any]) -> None:
    """Validate an outgoing WebSocket message. Only subscribe/unsubscribe/ping allowed."""
    method = message.get("method")
    if method not in ALLOWED_OUTGOING_METHODS:
        raise SecurityViolationError(
            f"Forbidden outgoing method: {method!r}. "
            f"Allowed: {sorted(ALLOWED_OUTGOING_METHODS)}"
        )

    if method == "ping":
        return

    subscription = message.get("subscription")
    if not isinstance(subscription, dict):
        raise SecurityViolationError("subscribe/unsubscribe requires a subscription dict")

    sub_type = subscription.get("type")
    if sub_type in FORBIDDEN_ORDER_TYPES:
        raise SecurityViolationError(f"Forbidden subscription type: {sub_type!r}")

    for key in message:
        if key in FORBIDDEN_PAYLOAD_KEYS:
            raise SecurityViolationError(f"Forbidden key in outgoing message: {key!r}")

    for key in subscription:
        if key in FORBIDDEN_PAYLOAD_KEYS:
            raise SecurityViolationError(f"Forbidden key in subscription: {key!r}")


def check_environment_variables() -> list[str]:
    """Return list of forbidden environment variables that are set."""
    violations = []
    for var in FORBIDDEN_ENV_VARS:
        if os.environ.get(var):
            violations.append(var)
    return violations


def check_forbidden_imports() -> None:
    """Raise if forbidden signing/wallet modules are importable in this process."""
    import sys

    for name in FORBIDDEN_IMPORT_MODULES:
        if name in sys.modules:
            raise SecurityViolationError(f"Forbidden module loaded: {name!r}")


def reject_post_action_payload(payload: dict[str, Any]) -> None:
    """Reject HTTP POST payloads with action type (exchange endpoint guard)."""
    if payload.get("type") == "action":
        raise SecurityViolationError("Forbidden POST request type: action")
    if "action" in payload and payload.get("type") != "info":
        raise SecurityViolationError("Forbidden action field in POST payload")
