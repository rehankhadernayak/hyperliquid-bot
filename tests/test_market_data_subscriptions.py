"""Offline tests for subscription management."""

from __future__ import annotations

import time

from market_data.recorder_config import RecorderConfig
from market_data.subscriptions import (
    build_subscribe_messages,
    build_subscriptions,
    SubscriptionState,
    SubscriptionTracker,
)


class TestSubscriptions:
    def test_build_subscriptions_default_coins(self):
        config = RecorderConfig(coins=("BTC", "ETH", "SOL"))
        subs = build_subscriptions(config)
        types = [s["type"] for s in subs]
        assert "allMids" in types
        assert subs.count({"type": "trades", "coin": "BTC"}) >= 1
        assert subs.count({"type": "bbo", "coin": "ETH"}) >= 1
        assert subs.count({"type": "l2Book", "coin": "SOL"}) >= 1
        assert subs.count({"type": "candle", "coin": "BTC", "interval": "1m"}) >= 1
        assert subs.count({"type": "activeAssetCtx", "coin": "BTC"}) >= 1

    def test_disable_l2(self):
        config = RecorderConfig(coins=("BTC"), disable_l2=True)
        subs = build_subscriptions(config)
        l2_subs = [s for s in subs if s.get("type") == "l2Book"]
        assert len(l2_subs) == 0

    def test_no_user_subscriptions(self):
        config = RecorderConfig()
        subs = build_subscriptions(config)
        forbidden_types = {
            "userEvents", "userFills", "openOrders",
            "clearinghouseState", "activeAssetData", "notifications",
        }
        for sub in subs:
            assert sub.get("type") not in forbidden_types

    def test_subscribe_messages_validated(self):
        config = RecorderConfig(coins=("BTC",))
        messages = build_subscribe_messages(config)
        for msg in messages:
            assert msg["method"] == "subscribe"
            assert "subscription" in msg

    def test_subscription_ack_tracking(self):
        config = RecorderConfig(coins=("BTC",))
        tracker = SubscriptionTracker(config)
        sub = {"type": "trades", "coin": "BTC"}
        tracker.mark_acknowledged(sub)
        summary = tracker.summary()
        assert summary["by_state"].get("ACTIVE", 0) >= 1

    def test_subscription_ack_timeout(self):
        config = RecorderConfig(
            coins=("BTC"),
            subscription_ack_timeout_seconds=0.001,
        )
        tracker = SubscriptionTracker(config)
        tracker.mark_requested({"type": "trades", "coin": "BTC"})
        time.sleep(0.01)
        warnings = tracker.check_ack_timeouts()
        assert len(warnings) >= 1

    def test_subscription_message_count(self):
        config = RecorderConfig(coins=("BTC",))
        tracker = SubscriptionTracker(config)
        tracker.record_message("trades", "BTC")
        tracker.record_message("trades", "BTC")
        summary = tracker.summary()
        trade_sub = [
            s for s in summary["subscriptions"]
            if s["subscription"].get("type") == "trades"
        ]
        assert trade_sub[0]["message_count"] == 2

    def test_subscription_states(self):
        assert SubscriptionState.REQUESTED.value == "REQUESTED"
        assert SubscriptionState.ACTIVE.value == "ACTIVE"
        assert SubscriptionState.FAILED.value == "FAILED"
