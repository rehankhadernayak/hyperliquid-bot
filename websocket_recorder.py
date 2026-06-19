#!/usr/bin/env python3
"""
CLI entry point for the Hyperliquid WebSocket market data recorder.

Read-only public market data collection. No wallets, signing, or order execution.

Usage:
    python3 websocket_recorder.py --coins BTC ETH SOL
    python3 websocket_recorder.py --status
    python3 websocket_recorder.py --duration-seconds 300
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from market_data.recorder_config import DEFAULT_COINS, RecorderConfig
from market_data.recorder import run_recorder
from market_data.status import format_status_summary, read_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hyperliquid WebSocket market data recorder (read-only, paper-only)",
    )
    parser.add_argument(
        "--coins",
        nargs="+",
        default=list(DEFAULT_COINS),
        help="Coins to subscribe (default: BTC ETH SOL)",
    )
    parser.add_argument(
        "--network",
        default="mainnet",
        help="Network (default: mainnet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/market_stream"),
        help="Output directory (default: data/market_stream)",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Stop after this many seconds",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Only write raw events, skip normalization",
    )
    parser.add_argument(
        "--disable-l2",
        action="store_true",
        help="Disable l2Book subscription",
    )
    parser.add_argument(
        "--candle-interval",
        default="1m",
        help="Candle interval (default: 1m)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show recorder status and exit",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.status:
        output_dir = args.output_dir
        status = read_status(output_dir)
        if status is None:
            print(f"No recorder status found at {output_dir}")
            return 1
        print(format_status_summary(status))
        return 0

    config = RecorderConfig(
        coins=tuple(args.coins),
        network=args.network,
        output_dir=args.output_dir,
        duration_seconds=args.duration_seconds,
        log_level=args.log_level,
        raw_only=args.raw_only,
        disable_l2=args.disable_l2,
        candle_interval=args.candle_interval,
    )

    print("Hyperliquid WebSocket Market Data Recorder")
    print("Read-only public market data. No orders or wallet actions.")
    print(f"Coins: {', '.join(config.coins)}")
    print(f"Output: {config.output_dir}")
    print(f"WebSocket: {config.websocket_url}")
    print("-" * 50)

    summary = run_recorder(config)

    print("\n" + "=" * 50)
    print("Recorder Summary")
    print("=" * 50)
    print(f"Session ID:       {summary['recorder_session_id']}")
    print(f"Raw events:       {summary['raw_events']}")
    print(f"Normalized:       {summary['normalized_events']}")
    print(f"By channel:       {summary['channel_counts']}")
    print(f"Reconnects:       {summary['reconnect_count']}")
    print(f"Malformed:        {summary['malformed_count']}")
    print(f"Duplicates:       {summary['duplicate_count']}")
    print(f"Latency p50:      {summary['latency_p50']} ms")
    print(f"Latency p95:      {summary['latency_p95']} ms")
    print(f"Latency p99:      {summary['latency_p99']} ms")
    print(f"Gaps detected:    {summary['gaps_detected']}")
    print(f"Shutdown reason:  {summary['shutdown_reason']}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
