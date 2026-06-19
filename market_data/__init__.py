"""
Read-only Hyperliquid WebSocket market data collection.

Paper-trading and research only — no wallets, signing, or order execution.
"""

from market_data.recorder_config import RecorderConfig, SCHEMA_VERSION

__all__ = ["RecorderConfig", "SCHEMA_VERSION"]
