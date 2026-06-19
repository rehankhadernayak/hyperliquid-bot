"""CLI for feature research."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from feature_research.feature_config import (
    DEFAULT_CHANNELS,
    DEFAULT_COINS,
    DEFAULT_HORIZONS_SECONDS,
    FeatureResearchConfig,
)
from feature_research.feature_engine import FeatureResearchEngine, FeatureResearchError
from feature_research.feature_safety import OUTPUT_SUBDIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature Research Engine v1 — deterministic replay, paper-only research.",
    )
    parser.add_argument("--replay-session-id", type=str, default=None)
    parser.add_argument("--latest-replay", action="store_true")
    parser.add_argument("--coins", nargs="+", default=DEFAULT_COINS)
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    parser.add_argument(
        "--horizons-seconds",
        nargs="+",
        type=int,
        default=DEFAULT_HORIZONS_SECONDS,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data") / OUTPUT_SUBDIR)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--market-stream-dir", type=Path, default=Path("data/market_stream"))
    return parser


def config_from_args(args: argparse.Namespace) -> FeatureResearchConfig:
    return FeatureResearchConfig(
        replay_session_id=args.replay_session_id,
        use_latest_replay=args.latest_replay,
        coins=tuple(args.coins),
        channels=tuple(args.channels),
        horizons_seconds=tuple(args.horizons_seconds),
        output_dir=args.output_dir,
        max_events=args.max_events,
        dry_run=args.dry_run,
        market_stream_dir=args.market_stream_dir,
    )


def run_feature_research(args: argparse.Namespace | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args=args)
    try:
        config = config_from_args(parsed)
        engine = FeatureResearchEngine(config)
        result = engine.run()
    except FeatureResearchError as exc:
        print(f"Feature research error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Feature research failed: {exc}", file=sys.stderr)
        return 1

    report = result.report
    print(f"Feature research complete: {report['feature_research_id']}")
    print(f"  Replay session: {report['replay_session_id']}")
    print(f"  Source recorder: {report['source_recorder_session_id']}")
    print(f"  Rows: {report['row_count']}")
    print(f"  Features: {report['feature_count']}")
    print(f"  Labels: {report['label_count']}")
    leakage = report.get("leakage_check") or {}
    print(f"  Leakage check: {'PASS' if leakage.get('passed') else 'FAIL'}")
    if report.get("deterministic_dataset_hash"):
        print(f"  Deterministic hash: {report['deterministic_dataset_hash']}")
    if result.report_paths:
        print(f"  JSON: {result.report_paths['json']}")
        print(f"  Markdown: {result.report_paths['markdown']}")
    if report.get("dataset_path"):
        print(f"  Dataset: {report['dataset_path']} ({report.get('dataset_format')})")
    possible = report.get("top_possible_signals") or []
    if possible:
        print(f"  Top possible signal: {possible[0].get('feature')} @ {possible[0].get('horizon_seconds')}s")
    return 0


def main() -> int:
    return run_feature_research()
