"""Read-only research module for Hyperliquid historical L2 archive backfill planning.

This module documents what Hyperliquid provides and how a future backfill workflow
should behave. It does not download archive data, call private APIs, or perform
any trading or execution actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PAPER_ONLY = True
EXECUTION_ENABLED = False
DOWNLOAD_ENABLED = False

SAFETY_STATEMENT = (
    "Historical backfill research is read-only, paper-trading only, and research-only. "
    "No wallets, private keys, signing, leverage, margin, exchange actions, real orders, "
    "deposits, withdrawals, private APIs, user account data, or archive downloads are used."
)

ARCHIVE_BUCKET = "hyperliquid-archive"
ARCHIVE_MARKET_DATA_PREFIX = "market_data"
DEFAULT_L2_DATATYPE = "l2Book"

HISTORICAL_FACTS: tuple[str, ...] = (
    "Hyperliquid live WebSocket BBO provides best bid/ask updates.",
    "Hyperliquid historical archive provides L2 book snapshots under the archive market_data path.",
    "Historical archive is uploaded about once a month.",
    "Data may be missing and is not guaranteed timely.",
    "Hyperliquid API candleSnapshot only supports 1m+ intervals and most recent 5000 candles.",
    "All-time 1-second bid/ask is not directly available from a single public source.",
    "Best path: import historical L2 snapshots, extract top bid/ask, validate coverage, "
    "then convert to 1s/15s/1m CSV.",
)

RECOMMENDED_WORKFLOW: tuple[str, ...] = (
    "Inspect expected S3 paths with historical_l2_importer.inspect-path (no download).",
    "Download archive files manually or via a future dedicated downloader (requester-pays).",
    "Decompress .lz4 files locally with unlz4 or lz4.",
    "Import decompressed files with historical_l2_importer.import-local.",
    "Summarize coverage and write JSON reports under data/bid_ask_csv/reports/.",
    "Resample historical 1s CSV to 15s/1m using existing bid_ask_resampler tooling.",
)

DATA_LIMITATIONS: tuple[str, ...] = (
    "L2 archive snapshots are emitted about twice per second, not guaranteed every second.",
    "Archive coverage may have gaps; missing hours or coins are possible.",
    "candleSnapshot cannot reconstruct sub-minute BBO history beyond its 5000-candle window.",
    "Do not claim complete all-time 1-second BBO without validating imported coverage.",
)


@dataclass(frozen=True)
class ArchivePathExample:
    date: str
    hour: int
    coin: str
    datatype: str
    s3_path: str


def research_summary() -> dict[str, Any]:
    """Return structured research facts for docs and tooling."""
    return {
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "download_enabled": DOWNLOAD_ENABLED,
        "safety_statement": SAFETY_STATEMENT,
        "archive_bucket": ARCHIVE_BUCKET,
        "archive_market_data_prefix": ARCHIVE_MARKET_DATA_PREFIX,
        "default_l2_datatype": DEFAULT_L2_DATATYPE,
        "historical_facts": list(HISTORICAL_FACTS),
        "recommended_workflow": list(RECOMMENDED_WORKFLOW),
        "data_limitations": list(DATA_LIMITATIONS),
        "example_paths": [
            ArchivePathExample(
                date="20230916",
                hour=9,
                coin="SOL",
                datatype=DEFAULT_L2_DATATYPE,
                s3_path=(
                    "s3://hyperliquid-archive/market_data/20230916/9/l2Book/SOL.lz4"
                ),
            ).__dict__,
        ],
    }


def print_research_summary() -> None:
    """Print a human-readable research summary to stdout."""
    summary = research_summary()
    print("Hyperliquid Historical L2 Backfill Research")
    print("=" * 44)
    print(summary["safety_statement"])
    print()
    print("Historical facts:")
    for fact in summary["historical_facts"]:
        print(f"  - {fact}")
    print()
    print("Recommended workflow:")
    for step in summary["recommended_workflow"]:
        print(f"  - {step}")
    print()
    print("Data limitations:")
    for limitation in summary["data_limitations"]:
        print(f"  - {limitation}")
    print()
    print("Example archive path:")
    example = summary["example_paths"][0]
    print(f"  {example['s3_path']}")
