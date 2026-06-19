#!/usr/bin/env python3
"""
Deterministic replay engine for recorded Hyperliquid WebSocket market data.

Paper-only, research-only. Reads manifest and normalized files; writes separate
replay reports only. Does not modify raw, normalized, or session data.

Usage:
    python3 market_data_replay.py --session-id <RECORDER_SESSION_ID>
    python3 market_data_replay.py --latest
    python3 market_data_replay.py --session-id <ID> --dry-run
    python3 market_data_replay.py --session-id <ID> --coins BTC ETH SOL
    python3 market_data_replay.py --session-id <ID> --channels trades bbo l2Book
    python3 market_data_replay.py --session-id <ID> --speed 1.0
    python3 market_data_replay.py --session-id <ID> --speed 0
    python3 market_data_replay.py --session-id <ID> --max-events 1000
    python3 market_data_replay.py --session-id <ID> --output-dir data/market_stream/replays
"""

from __future__ import annotations

import sys

from market_data.replay_cli import main

if __name__ == "__main__":
    sys.exit(main())
