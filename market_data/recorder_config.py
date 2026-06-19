"""Configuration for the WebSocket market data recorder."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config import VERSION

SCHEMA_VERSION = "1.0.0"

NETWORK_URLS = {
    "mainnet": "wss://api.hyperliquid.xyz/ws",
}

DEFAULT_COINS = ("BTC", "ETH", "SOL")

ALLOWED_OUTGOING_METHODS = frozenset({"subscribe", "unsubscribe", "ping"})


@dataclass(frozen=True)
class RecorderConfig:
    """Immutable recorder configuration."""

    coins: tuple[str, ...] = DEFAULT_COINS
    network: str = "mainnet"
    output_dir: Path = Path("data/market_stream")
    duration_seconds: float | None = None
    log_level: str = "INFO"
    raw_only: bool = False
    disable_l2: bool = False
    candle_interval: str = "1m"
    heartbeat_interval_seconds: float = 30.0
    heartbeat_timeout_seconds: float = 10.0
    reconnect_initial_delay_seconds: float = 1.0
    reconnect_max_delay_seconds: float = 60.0
    reconnect_stable_period_seconds: float = 120.0
    subscription_ack_timeout_seconds: float = 10.0
    channel_staleness_threshold_seconds: float = 120.0
    raw_flush_interval_seconds: float = 5.0
    normalized_batch_size: int = 100
    raw_queue_maxsize: int = 10000
    normalized_queue_maxsize: int = 5000
    l2_depth_levels: int = 20
    schema_version: str = SCHEMA_VERSION

    @property
    def websocket_url(self) -> str:
        url = NETWORK_URLS.get(self.network)
        if url is None:
            raise ValueError(f"Unsupported network: {self.network!r}")
        return url

    def configuration_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        d["websocket_url"] = self.websocket_url
        d["software_version"] = VERSION
        d["paper_only"] = True
        d["execution_enabled"] = False
        return d
