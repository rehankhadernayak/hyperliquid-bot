"""Offline tests for market data security guard."""

from __future__ import annotations

import pytest

from market_data.recorder_config import ALLOWED_OUTGOING_METHODS
from market_data.security_guard import (
    FORBIDDEN_ENV_VARS,
    reject_post_action_payload,
    SecurityViolationError,
    validate_outgoing_message,
    validate_websocket_url,
)


class TestSecurityGuard:
    def test_allowed_subscribe(self):
        msg = {"method": "subscribe", "subscription": {"type": "trades", "coin": "BTC"}}
        validate_outgoing_message(msg)

    def test_allowed_ping(self):
        validate_outgoing_message({"method": "ping"})

    def test_allowed_unsubscribe(self):
        msg = {"method": "unsubscribe", "subscription": {"type": "trades", "coin": "BTC"}}
        validate_outgoing_message(msg)

    def test_rejects_action_method(self):
        with pytest.raises(SecurityViolationError):
            validate_outgoing_message({"method": "post", "type": "action"})

    def test_rejects_order_subscription(self):
        with pytest.raises(SecurityViolationError):
            validate_outgoing_message(
                {"method": "subscribe", "subscription": {"type": "order", "coin": "BTC"}}
            )

    def test_rejects_cancel_subscription(self):
        with pytest.raises(SecurityViolationError):
            validate_outgoing_message(
                {"method": "subscribe", "subscription": {"type": "cancel"}}
            )

    def test_rejects_signature_in_message(self):
        with pytest.raises(SecurityViolationError):
            validate_outgoing_message(
                {"method": "subscribe", "subscription": {"type": "trades"}, "signature": "bad"}
            )

    def test_rejects_exchange_url(self):
        with pytest.raises(SecurityViolationError):
            validate_websocket_url("wss://api.hyperliquid.xyz/exchange")

    def test_allows_mainnet_ws_url(self):
        validate_websocket_url("wss://api.hyperliquid.xyz/ws")

    def test_rejects_post_action_payload(self):
        with pytest.raises(SecurityViolationError):
            reject_post_action_payload({"type": "action", "action": {"type": "order"}})

    def test_only_allowed_methods(self):
        assert ALLOWED_OUTGOING_METHODS == frozenset({"subscribe", "unsubscribe", "ping"})

    def test_forbidden_env_vars_defined(self):
        assert any("HYPERLIQUID" in v for v in FORBIDDEN_ENV_VARS)
        assert any(v.endswith("KEY") for v in FORBIDDEN_ENV_VARS)

    def test_no_wallet_imports_in_market_data(self):
        import ast
        from pathlib import Path

        from market_data.security_guard import FORBIDDEN_IMPORT_MODULES

        market_data_dir = Path(__file__).resolve().parent.parent / "market_data"
        for path in market_data_dir.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert alias.name.split(".")[0] not in FORBIDDEN_IMPORT_MODULES, (
                            f"Forbidden import in {path}: {alias.name}"
                        )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert node.module.split(".")[0] not in FORBIDDEN_IMPORT_MODULES, (
                        f"Forbidden import in {path}: {node.module}"
                    )
