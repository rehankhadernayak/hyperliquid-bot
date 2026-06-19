"""CLI logic for market data replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from market_data.replay_engine import ReplayEngine, ReplayError
from market_data.replay_filters import ReplayFilters
from market_data.replay_manifest import find_latest_session_id


DEFAULT_OUTPUT_DIR = Path("data/market_stream/replays")
DEFAULT_MARKET_STREAM_DIR = Path("data/market_stream")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic replay of recorded Hyperliquid WebSocket market data. "
            "Paper-only, research-only — no execution."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--session-id",
        help="Recorder session ID to replay",
    )
    group.add_argument(
        "--latest",
        action="store_true",
        help="Replay the most recent recorder session",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Validate and summarize without streaming to consumers (default)",
    )
    parser.add_argument(
        "--emit",
        action="store_true",
        help="Stream events to consumer (disables dry-run)",
    )
    parser.add_argument(
        "--coins",
        nargs="+",
        help="Filter by coin symbols (e.g. BTC ETH SOL)",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        help="Filter by channels: trades bbo l2Book candle activeAssetCtx allMids",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier (0 = as fast as possible, 1.0 = recorded timing)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        help="Maximum number of events to replay",
    )
    parser.add_argument(
        "--start-timestamp-ms",
        type=int,
        help="Only replay events at or after this timestamp (ms)",
    )
    parser.add_argument(
        "--end-timestamp-ms",
        type=int,
        help="Only replay events at or before this timestamp (ms)",
    )
    parser.add_argument(
        "--exclude-candles",
        action="store_true",
        help="Exclude candle events",
    )
    parser.add_argument(
        "--exclude-l2-book",
        action="store_true",
        help="Exclude l2Book events",
    )
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help="Allow replay of UNUSABLE quality sessions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for replay reports (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--market-stream-dir",
        type=Path,
        default=DEFAULT_MARKET_STREAM_DIR,
        help=f"Market stream root (default: {DEFAULT_MARKET_STREAM_DIR})",
    )
    return parser


def run_replay(args: argparse.Namespace) -> int:
    market_stream_dir = args.market_stream_dir
    session_id = args.session_id
    if args.latest:
        session_id = find_latest_session_id(market_stream_dir)
        if not session_id:
            print("No recorder manifests found.", file=sys.stderr)
            return 1

    dry_run = not args.emit
    if args.dry_run:
        dry_run = True

    filters = ReplayFilters(
        coins=args.coins,
        channels=args.channels,
        start_timestamp_ms=args.start_timestamp_ms,
        end_timestamp_ms=args.end_timestamp_ms,
        max_events=args.max_events,
        include_candles=not args.exclude_candles,
        include_l2_book=not args.exclude_l2_book,
    )

    engine = ReplayEngine(market_stream_dir=market_stream_dir)
    try:
        result = engine.run(
            session_id,
            filters=filters,
            speed=args.speed,
            dry_run=dry_run,
            allow_degraded=args.allow_degraded,
            output_dir=args.output_dir,
        )
    except ReplayError as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 1

    report = result.report
    print(f"Replay session: {report['replay_session_id']}")
    print(f"  Source session: {report['source_recorder_session_id']}")
    print(f"  Quality label: {report.get('quality_label', '—')}")
    print(f"  Total events: {report['total_events']}")
    print(f"  Skipped: {report['skipped_events']}")
    print(f"  Stream hash: {report['stream_hash']}")
    print(f"  Dry run: {report['dry_run']}")
    print(f"  Speed: {report['speed']}")
    if report.get("ordering_warnings"):
        print(f"  Ordering warnings: {len(report['ordering_warnings'])}")
    for kind, path in result.report_paths.items():
        print(f"  {kind}: {path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_replay(args)
