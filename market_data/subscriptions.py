"""Subscription management for Hyperliquid WebSocket feeds."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from market_data.recorder_config import RecorderConfig
from market_data.security_guard import validate_outgoing_message


class SubscriptionState(Enum):
    REQUESTED = "REQUESTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    ACTIVE = "ACTIVE"
    STALE = "STALE"
    FAILED = "FAILED"


GLOBAL_SUBSCRIPTION_TYPES = ("allMids",)

PER_COIN_SUBSCRIPTION_TYPES = (
    "trades",
    "bbo",
    "l2Book",
    "candle",
    "activeAssetCtx",
)


@dataclass
class SubscriptionRecord:
    subscription: dict[str, Any]
    state: SubscriptionState = SubscriptionState.REQUESTED
    requested_at_ms: int = 0
    acknowledged_at_ms: int | None = None
    last_message_at_ms: int | None = None
    message_count: int = 0


def subscription_key(subscription: dict[str, Any]) -> str:
    return json.dumps(subscription, sort_keys=True, separators=(",", ":"))


def build_subscriptions(config: RecorderConfig) -> list[dict[str, Any]]:
    """Build the list of subscription dicts for the configured coins."""
    subs: list[dict[str, Any]] = []

    subs.append({"type": "allMids"})

    for coin in config.coins:
        subs.append({"type": "trades", "coin": coin})
        subs.append({"type": "bbo", "coin": coin})
        if not config.disable_l2:
            subs.append({"type": "l2Book", "coin": coin})
        subs.append({"type": "candle", "coin": coin, "interval": config.candle_interval})
        subs.append({"type": "activeAssetCtx", "coin": coin})

    return subs


def build_subscribe_messages(config: RecorderConfig) -> list[dict[str, Any]]:
    """Build outgoing subscribe messages, validated by security guard."""
    messages = []
    for sub in build_subscriptions(config):
        msg = {"method": "subscribe", "subscription": sub}
        validate_outgoing_message(msg)
        messages.append(msg)
    return messages


class SubscriptionTracker:
    """Track subscription states and acknowledgements."""

    def __init__(self, config: RecorderConfig) -> None:
        self._ack_timeout_ms = int(config.subscription_ack_timeout_seconds * 1000)
        self._records: dict[str, SubscriptionRecord] = {}
        now = int(time.time() * 1000)
        for sub in build_subscriptions(config):
            key = subscription_key(sub)
            self._records[key] = SubscriptionRecord(
                subscription=sub,
                state=SubscriptionState.REQUESTED,
                requested_at_ms=now,
            )

    def mark_requested(self, subscription: dict[str, Any]) -> None:
        key = subscription_key(subscription)
        now = int(time.time() * 1000)
        if key in self._records:
            self._records[key].state = SubscriptionState.REQUESTED
            self._records[key].requested_at_ms = now
        else:
            self._records[key] = SubscriptionRecord(
                subscription=subscription,
                state=SubscriptionState.REQUESTED,
                requested_at_ms=now,
            )

    def mark_acknowledged(self, subscription: dict[str, Any]) -> None:
        key = subscription_key(subscription)
        now = int(time.time() * 1000)
        if key in self._records:
            rec = self._records[key]
            rec.state = SubscriptionState.ACKNOWLEDGED
            rec.acknowledged_at_ms = now
            rec.state = SubscriptionState.ACTIVE

    def record_message(self, channel: str, coin: str | None = None) -> None:
        now = int(time.time() * 1000)
        for rec in self._records.values():
            sub = rec.subscription
            sub_type = sub.get("type")
            if sub_type == channel or (channel == "l2Book" and sub_type == "l2Book"):
                if coin is not None and sub.get("coin") != coin:
                    continue
                if sub_type == "allMids" and channel == "allMids":
                    rec.last_message_at_ms = now
                    rec.message_count += 1
                elif sub.get("coin") == coin or sub_type == channel:
                    rec.last_message_at_ms = now
                    rec.message_count += 1
                    if rec.state in (
                        SubscriptionState.ACKNOWLEDGED,
                        SubscriptionState.REQUESTED,
                    ):
                        rec.state = SubscriptionState.ACTIVE

    def check_ack_timeouts(self) -> list[str]:
        """Return keys of subscriptions that missed acknowledgement."""
        now = int(time.time() * 1000)
        warnings = []
        for key, rec in self._records.items():
            if rec.state == SubscriptionState.REQUESTED:
                if now - rec.requested_at_ms > self._ack_timeout_ms:
                    rec.state = SubscriptionState.FAILED
                    warnings.append(key)
        return warnings

    def mark_stale(self, threshold_ms: int) -> list[str]:
        now = int(time.time() * 1000)
        stale = []
        for key, rec in self._records.items():
            if rec.state == SubscriptionState.ACTIVE and rec.last_message_at_ms:
                if now - rec.last_message_at_ms > threshold_ms:
                    rec.state = SubscriptionState.STALE
                    stale.append(key)
        return stale

    def summary(self) -> dict[str, Any]:
        by_state: dict[str, int] = {}
        details = []
        for key, rec in self._records.items():
            state_name = rec.state.value
            by_state[state_name] = by_state.get(state_name, 0) + 1
            details.append(
                {
                    "subscription": rec.subscription,
                    "state": state_name,
                    "message_count": rec.message_count,
                    "last_message_at_ms": rec.last_message_at_ms,
                }
            )
        return {"by_state": by_state, "subscriptions": details}

    @property
    def active_count(self) -> int:
        return sum(1 for r in self._records.values() if r.state == SubscriptionState.ACTIVE)
