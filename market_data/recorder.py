"""Main orchestrator for the Hyperliquid WebSocket market data recorder."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
import uuid
from pathlib import Path
from typing import Any

from market_data.connection_health import ConnectionHealth, ConnectionState
from market_data.gap_detection import GapDetector
from market_data.latency_tracker import LatencyStats
from market_data.manifest import build_manifest, write_manifest
from market_data.normalizer import normalize_message
from market_data.raw_writer import RawWriter
from market_data.normalized_writer import NormalizedWriter
from market_data.recorder_config import RecorderConfig
from market_data.security_guard import (
    check_environment_variables,
    check_forbidden_imports,
    SecurityViolationError,
)
from market_data.status import write_status
from market_data.subscriptions import (
    build_subscribe_messages,
    build_subscriptions,
    SubscriptionTracker,
)
from market_data.websocket_client import HyperliquidWebSocketClient, NON_MARKET_CHANNELS

logger = logging.getLogger(__name__)


class MarketDataRecorder:
    """Orchestrate WebSocket collection, normalization, and storage."""

    def __init__(self, config: RecorderConfig) -> None:
        self._config = config
        self._recorder_session_id = uuid.uuid4().hex
        self._start_time_ms = int(time.time() * 1000)
        self._shutdown_reason = "running"
        self._running = False

        env_violations = check_environment_variables()
        if env_violations:
            raise SecurityViolationError(
                f"Forbidden environment variables set: {env_violations}"
            )
        check_forbidden_imports()

        config.output_dir.mkdir(parents=True, exist_ok=True)

        self._health = ConnectionHealth()
        self._gap_detector = GapDetector(candle_interval_ms=60000)
        self._latency = LatencyStats()
        self._subscription_tracker = SubscriptionTracker(config)

        self._raw_writer = RawWriter(
            output_dir=config.output_dir,
            recorder_session_id=self._recorder_session_id,
            network=config.network,
            queue_maxsize=config.raw_queue_maxsize,
            flush_interval_seconds=config.raw_flush_interval_seconds,
        )
        self._normalized_writer: NormalizedWriter | None = None
        if not config.raw_only:
            self._normalized_writer = NormalizedWriter(
                output_dir=config.output_dir,
                queue_maxsize=config.normalized_queue_maxsize,
                batch_size=config.normalized_batch_size,
            )

        self._client: HyperliquidWebSocketClient | None = None
        self._disconnect_start_ms: int | None = None
        self._subscribe_messages = build_subscribe_messages(config)
        self._requested_subscriptions = build_subscriptions(config)

    @property
    def recorder_session_id(self) -> str:
        return self._recorder_session_id

    @property
    def health(self) -> ConnectionHealth:
        return self._health

    def _on_connected(self) -> None:
        self._health.set_state(ConnectionState.CONNECTED)
        if self._disconnect_start_ms is not None:
            now = int(time.time() * 1000)
            self._gap_detector.record_reconnect_window(
                self._disconnect_start_ms, now
            )
            self._disconnect_start_ms = None
        self._raw_writer.set_reconnect_generation(self._health.reconnect_generation)
        for sub in self._requested_subscriptions:
            self._subscription_tracker.mark_requested(sub)

    def _on_disconnect(self, reason: str) -> None:
        self._health.record_disconnect(reason)
        self._health.record_reconnect()
        self._disconnect_start_ms = int(time.time() * 1000)
        if self._client:
            self._client.reset_backoff()

    def _extract_coin(self, channel: str, payload: Any) -> str | None:
        if isinstance(payload, dict):
            if "coin" in payload:
                return payload["coin"]
            data = payload.get("data")
            if isinstance(data, dict) and "coin" in data:
                return data["coin"]
            if channel == "candle" and isinstance(data, dict):
                return data.get("s")
        return None

    def _extract_exchange_timestamp(self, channel: str, payload: Any) -> int | None:
        if not isinstance(payload, dict):
            return None
        data = payload.get("data", payload)
        if isinstance(data, dict):
            if "time" in data:
                return int(data["time"])
            if "t" in data:
                return int(data["t"])
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and "time" in first:
                return int(first["time"])
        return None

    def _handle_message(self, channel: str, payload: Any, received_ms: int) -> None:
        if channel == "__malformed__":
            self._health.malformed_count += 1
            return

        if channel == "pong":
            rtt = self._health.heartbeat.record_pong(received_ms)
            if rtt is not None:
                logger.debug("Pong received, RTT=%.1f ms", rtt)
            return

        if channel == "subscriptionResponse":
            if isinstance(payload, dict):
                sub = payload.get("subscription", payload)
                self._subscription_tracker.mark_acknowledged(sub)
            return

        if channel == "error":
            logger.warning("Server error: %s", payload)
            return

        self._health.record_message(channel)
        coin = self._extract_coin(channel, payload)
        self._subscription_tracker.record_message(channel, coin)

        exchange_ts = self._extract_exchange_timestamp(channel, payload)
        if exchange_ts is not None:
            self._latency.record(exchange_ts, received_ms)
            self._gap_detector.record_exchange_timestamp(
                channel, coin, exchange_ts, received_ms
            )

        subscription = None
        if coin:
            subscription = {"type": channel, "coin": coin}

        raw_payload = payload if isinstance(payload, dict) else {"raw": payload}
        if not self._raw_writer.enqueue(
            channel=channel,
            subscription=subscription,
            raw_payload=raw_payload,
            exchange_timestamp_ms=exchange_ts,
            received_timestamp_ms=received_ms,
        ):
            self._health.degraded = True
            self._health.raw_dropped_count += 1

        if self._normalized_writer and channel not in NON_MARKET_CHANNELS:
            data = raw_payload.get("data", raw_payload) if isinstance(raw_payload, dict) else raw_payload
            events = normalize_message(
                channel,
                data,
                received_ms,
                l2_depth_levels=self._config.l2_depth_levels,
            )
            from market_data.event_models import CandleEvent

            for event in events:
                if isinstance(event, CandleEvent):
                    self._gap_detector.record_candle(event.coin, event.open_time_ms)

                if not self._normalized_writer.enqueue(event):
                    self._health.normalized_dropped_count += 1

    def _write_status_file(self) -> None:
        status = {
            "running": self._running,
            "recorder_session_id": self._recorder_session_id,
            "uptime_seconds": self._health.uptime_seconds(),
            "connection_state": self._health.state.value,
            "reconnect_count": self._health.reconnect_count,
            "last_message_at_ms": self._health.last_message_at_ms,
            "active_subscriptions": self._subscription_tracker.active_count,
            "channel_message_counts": dict(self._health.channel_message_counts),
            "raw_queue_size": self._raw_writer.queue_size,
            "normalized_queue_size": (
                self._normalized_writer.queue_size if self._normalized_writer else 0
            ),
            "latency_p50_ms": self._latency.median_ms,
            "latency_p95_ms": self._latency.p95_ms,
            "latency_p99_ms": self._latency.p99_ms,
            "malformed_count": self._health.malformed_count,
            "duplicate_count": (
                self._normalized_writer.duplicate_count if self._normalized_writer else 0
            ),
            "gap_count": self._gap_detector.gap_count,
            "output_dir": str(self._config.output_dir),
            "heartbeat": self._health.heartbeat.to_dict(),
            "messages_per_second": self._health.messages_per_second(),
            "coins": list(self._config.coins),
        }
        write_status(self._config.output_dir, status)

    async def _status_update_loop(self) -> None:
        while self._running:
            self._subscription_tracker.check_ack_timeouts()
            stale_threshold = int(self._config.channel_staleness_threshold_seconds * 1000)
            self._subscription_tracker.mark_stale(stale_threshold)
            self._write_status_file()
            await asyncio.sleep(5)

    async def _duration_timer(self, duration_seconds: float) -> None:
        await asyncio.sleep(duration_seconds)
        self._shutdown_reason = "duration_limit"
        self.stop()

    async def run(self, duration_seconds: float | None = None) -> dict[str, Any]:
        self._running = True
        self._health.set_state(ConnectionState.CONNECTING)

        self._client = HyperliquidWebSocketClient(
            config=self._config,
            on_message=self._handle_message,
            on_disconnect=self._on_disconnect,
            on_connected=self._on_connected,
        )

        tasks = [
            asyncio.create_task(
                self._client.connect_and_run(self._subscribe_messages)
            ),
            asyncio.create_task(self._status_update_loop()),
        ]

        if duration_seconds is not None:
            tasks.append(asyncio.create_task(self._duration_timer(duration_seconds)))
        elif self._config.duration_seconds is not None:
            tasks.append(
                asyncio.create_task(
                    self._duration_timer(self._config.duration_seconds)
                )
            )

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self._shutdown_reason = "cancelled"
        finally:
            await self._shutdown()

        return self._build_summary()

    def stop(self) -> None:
        self._running = False
        if self._shutdown_reason == "running":
            self._shutdown_reason = "user_interrupt"
        if self._client:
            self._client.stop()

    async def _shutdown(self) -> None:
        self._health.set_state(ConnectionState.STOPPED)
        self._raw_writer.shutdown()
        if self._normalized_writer:
            self._normalized_writer.shutdown()

        end_time_ms = int(time.time() * 1000)
        all_files = list(self._raw_writer.files_written)
        if self._normalized_writer:
            all_files.extend(self._normalized_writer.files_written)

        config_dict = self._config.to_dict()
        config_dict["requested_subscriptions"] = self._requested_subscriptions

        manifest_data = build_manifest(
            recorder_session_id=self._recorder_session_id,
            config_dict=config_dict,
            start_time_ms=self._start_time_ms,
            end_time_ms=end_time_ms,
            raw_row_count=self._raw_writer.row_count,
            normalized_row_count=(
                self._normalized_writer.row_count if self._normalized_writer else 0
            ),
            duplicate_count=(
                self._normalized_writer.duplicate_count if self._normalized_writer else 0
            ),
            malformed_count=self._health.malformed_count,
            reconnect_count=self._health.reconnect_count,
            heartbeat_stats=self._health.heartbeat.to_dict(),
            channel_stats=dict(self._health.channel_message_counts),
            latency_stats=self._latency.to_dict(),
            gap_stats=self._gap_detector.summary(),
            subscription_summary=self._subscription_tracker.summary(),
            files_written=all_files,
            shutdown_reason=self._shutdown_reason,
            configuration_hash=self._config.configuration_hash(),
            degraded=self._health.degraded,
            raw_dropped=self._health.raw_dropped_count,
            normalized_dropped=self._health.normalized_dropped_count,
        )

        manifest_path = write_manifest(
            self._config.output_dir / "manifests",
            self._recorder_session_id,
            manifest_data,
        )

        self._write_status_file()
        status = {
            "running": False,
            "recorder_session_id": self._recorder_session_id,
            "manifest_path": str(manifest_path),
        }
        write_status(self._config.output_dir, status)

        logger.info("Manifest written: %s", manifest_path)

    def _build_summary(self) -> dict[str, Any]:
        return {
            "recorder_session_id": self._recorder_session_id,
            "raw_events": self._raw_writer.row_count,
            "normalized_events": (
                self._normalized_writer.row_count if self._normalized_writer else 0
            ),
            "channel_counts": dict(self._health.channel_message_counts),
            "reconnect_count": self._health.reconnect_count,
            "malformed_count": self._health.malformed_count,
            "duplicate_count": (
                self._normalized_writer.duplicate_count if self._normalized_writer else 0
            ),
            "latency_p50": self._latency.median_ms,
            "latency_p95": self._latency.p95_ms,
            "latency_p99": self._latency.p99_ms,
            "gaps_detected": self._gap_detector.gap_count,
            "shutdown_reason": self._shutdown_reason,
        }


def run_recorder(config: RecorderConfig) -> dict[str, Any]:
    """Synchronous entry point for the recorder."""
    recorder = MarketDataRecorder(config)

    loop = asyncio.new_event_loop()

    def signal_handler() -> None:
        recorder.stop()

    try:
        loop.add_signal_handler(signal.SIGINT, signal_handler)
        loop.add_signal_handler(signal.SIGTERM, signal_handler)
    except NotImplementedError:
        pass

    try:
        summary = loop.run_until_complete(recorder.run())
    finally:
        loop.close()

    return summary
