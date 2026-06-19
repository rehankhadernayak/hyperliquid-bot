"""Hyperliquid WebSocket client with reconnection and heartbeat support."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any, Callable

from market_data.recorder_config import RecorderConfig
from market_data.security_guard import validate_outgoing_message, validate_websocket_url

logger = logging.getLogger(__name__)

HEARTBEAT_CHANNELS = frozenset({"pong"})
NON_MARKET_CHANNELS = frozenset({"pong", "subscriptionResponse", "error"})


class HyperliquidWebSocketClient:
    """Async WebSocket client for Hyperliquid public market data."""

    def __init__(
        self,
        config: RecorderConfig,
        on_message: Callable[[str, Any, int], None],
        on_disconnect: Callable[[str], None],
        on_connected: Callable[[], None],
    ) -> None:
        self._config = config
        self._on_message = on_message
        self._on_disconnect = on_disconnect
        self._on_connected = on_connected
        self._url = config.websocket_url
        validate_websocket_url(self._url)
        self._ws: Any = None
        self._running = False
        self._last_server_message_time: float = 0.0
        self._last_ping_time: float = 0.0
        self._awaiting_pong: bool = False
        self._reconnect_delay = config.reconnect_initial_delay_seconds
        self._stable_connection_start: float | None = None

    async def connect_and_run(
        self,
        subscribe_messages: list[dict[str, Any]],
    ) -> None:
        import ssl
        import certifi
        import websockets

        ssl_context = ssl.create_default_context(cafile=certifi.where())
        self._running = True
        while self._running:
            try:
                async with websockets.connect(self._url, ssl=ssl_context) as ws:
                    self._ws = ws
                    self._last_server_message_time = time.monotonic()
                    self._last_ping_time = 0.0
                    self._awaiting_pong = False
                    self._on_connected()

                    for msg in subscribe_messages:
                        await self._send(msg)

                    self._stable_connection_start = time.monotonic()
                    await self._receive_loop()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                logger.warning("WebSocket disconnected: %s", reason)
                self._on_disconnect(reason)
                if not self._running:
                    break
                await self._backoff_sleep()

    async def _backoff_sleep(self) -> None:
        delay = self._reconnect_delay
        jitter = random.uniform(0, delay * 0.3)
        total = min(delay + jitter, self._config.reconnect_max_delay_seconds)
        logger.info("Reconnecting in %.1f seconds...", total)
        await asyncio.sleep(total)
        self._reconnect_delay = min(
            self._reconnect_delay * 2,
            self._config.reconnect_max_delay_seconds,
        )

    def reset_backoff(self) -> None:
        if self._stable_connection_start is not None:
            stable_duration = time.monotonic() - self._stable_connection_start
            if stable_duration >= self._config.reconnect_stable_period_seconds:
                self._reconnect_delay = self._config.reconnect_initial_delay_seconds

    async def _send(self, message: dict[str, Any]) -> None:
        validate_outgoing_message(message)
        if self._ws is not None:
            await self._ws.send(json.dumps(message))

    async def send_ping(self) -> None:
        msg = {"method": "ping"}
        validate_outgoing_message(msg)
        self._last_ping_time = time.monotonic()
        self._awaiting_pong = True
        await self._send(msg)

    async def _receive_loop(self) -> None:
        while self._running and self._ws is not None:
            try:
                raw = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=self._config.heartbeat_interval_seconds,
                )
            except asyncio.TimeoutError:
                await self._handle_heartbeat_timeout()
                continue
            except Exception as exc:
                raise ConnectionError(f"Receive failed: {exc}") from exc

            self._last_server_message_time = time.monotonic()
            self._process_raw(raw)

    async def _handle_heartbeat_timeout(self) -> None:
        elapsed = time.monotonic() - self._last_server_message_time
        if elapsed >= self._config.heartbeat_interval_seconds:
            if self._awaiting_pong:
                raise ConnectionError("Pong timeout exceeded")
            await self.send_ping()

    def _process_raw(self, raw: str) -> None:
        received_ms = int(time.time() * 1000)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._on_message("__malformed__", raw, received_ms)
            return

        channel = payload.get("channel", "")
        data = payload.get("data")

        if channel == "pong":
            self._awaiting_pong = False
            self._on_message("pong", data, received_ms)
            return

        if channel in NON_MARKET_CHANNELS:
            self._on_message(channel, data, received_ms)
            return

        self._on_message(channel, payload, received_ms)

    def stop(self) -> None:
        self._running = False

    @property
    def is_awaiting_pong(self) -> bool:
        return self._awaiting_pong

    @property
    def last_ping_time(self) -> float:
        return self._last_ping_time
