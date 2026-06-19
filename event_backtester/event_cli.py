"""CLI for event-driven backtests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from event_backtester.backtest_config import (
    DEFAULT_CHANNELS,
    DEFAULT_COINS,
    DEFAULT_STARTING_EQUITY,
    DEFAULT_STRATEGY_PROFILE,
    EventBacktestConfig,
)
from event_backtester.event_backtest_engine import EventBacktestEngine, EventBacktestError
from event_backtester.event_safety import OUTPUT_SUBDIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Event-driven backtester v1 — deterministic replay, paper-only research.",
    )
    parser.add_argument("--replay-session-id", type=str, default=None)
    parser.add_argument("--source-recorder-session-id", type=str, default=None)
    parser.add_argument("--latest-replay", action="store_true")
    parser.add_argument("--coins", nargs="+", default=DEFAULT_COINS)
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    parser.add_argument("--strategy", type=str, default=DEFAULT_STRATEGY_PROFILE)
    parser.add_argument("--starting-equity", type=float, default=DEFAULT_STARTING_EQUITY)
    parser.add_argument("--output-dir", type=Path, default=Path("data") / OUTPUT_SUBDIR)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--market-stream-dir", type=Path, default=Path("data/market_stream"))
    return parser


def config_from_args(args: argparse.Namespace) -> EventBacktestConfig:
    return EventBacktestConfig(
        replay_session_id=args.replay_session_id,
        source_recorder_session_id=args.source_recorder_session_id,
        use_latest_replay=args.latest_replay,
        coins=tuple(args.coins),
        channels=tuple(args.channels),
        strategy_profile_id=args.strategy,
        starting_equity=args.starting_equity,
        output_dir=args.output_dir,
        max_events=args.max_events,
        dry_run=args.dry_run,
        market_stream_dir=args.market_stream_dir,
    )


def run_event_backtest(args: argparse.Namespace | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args=args)
    try:
        config = config_from_args(parsed)
        engine = EventBacktestEngine(config)
        result = engine.run()
    except EventBacktestError as exc:
        print(f"Event backtest error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Event backtest failed: {exc}", file=sys.stderr)
        return 1

    report = result.report
    print(f"Event backtest complete: {report['event_backtest_id']}")
    print(f"  Replay session: {report['replay_session_id']}")
    print(f"  Events processed: {report['events_processed']}")
    print(f"  Ending equity: ${report['ending_equity']:,.2f}")
    print(f"  Net return: {report['net_return_pct']:.2f}%")
    print(f"  Completed trades: {report['completed_trades']}")
    print(f"  Evidence label: {report['evidence_label']}")
    print(f"  Deterministic hash: {report['deterministic_backtest_hash']}")
    print(f"  JSON: {result.report_paths['json']}")
    print(f"  Markdown: {result.report_paths['markdown']}")
    return 0


def main() -> int:
    return run_event_backtest()
