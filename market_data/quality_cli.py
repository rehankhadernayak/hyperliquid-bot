"""CLI logic for market data quality reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from market_data.quality_report import build_quality_report, write_quality_report
from market_data.replay_manifest import find_latest_session_id


DEFAULT_OUTPUT_DIR = Path("data/market_stream/quality_reports")
DEFAULT_MARKET_STREAM_DIR = Path("data/market_stream")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a read-only quality report for a WebSocket recorder session. "
            "Paper-only, research-only — no execution."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--session-id",
        help="Recorder session ID to inspect",
    )
    group.add_argument(
        "--latest",
        action="store_true",
        help="Inspect the most recent recorder session manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for report output (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--market-stream-dir",
        type=Path,
        default=DEFAULT_MARKET_STREAM_DIR,
        help=f"Market stream root (default: {DEFAULT_MARKET_STREAM_DIR})",
    )
    parser.add_argument(
        "--format",
        choices=["both", "json", "markdown"],
        default="both",
        help="Output format (default: both JSON and markdown)",
    )
    return parser


def run_quality_report(args: argparse.Namespace) -> int:
    market_stream_dir = args.market_stream_dir
    session_id = args.session_id
    if args.latest:
        session_id = find_latest_session_id(market_stream_dir)
        if not session_id:
            print("No recorder manifests found.", file=sys.stderr)
            return 1

    report = build_quality_report(session_id, market_stream_dir)
    write_json = args.format in ("both", "json")
    write_md = args.format in ("both", "markdown")
    paths = write_quality_report(
        report,
        args.output_dir,
        write_json=write_json,
        write_markdown=write_md,
    )

    if args.format == "json":
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"Quality report for session {session_id}")
        print(f"  Label: {report['quality_label']}")
        print(f"  Warnings: {report['warning_count']}")
        print(f"  Gaps: {report['gap_summary']['total_gaps']}")
        for kind, path in paths.items():
            print(f"  {kind}: {path}")

    return 0 if report["quality_label"] != "UNUSABLE" else 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_quality_report(args)
