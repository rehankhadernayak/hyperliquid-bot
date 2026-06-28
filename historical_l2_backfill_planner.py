"""Offline planner for Hyperliquid historical L2 archive backfill manifests.

Plans which S3 archive files are needed for a date range and coin list. Does not
download, decompress, or call any network or execution endpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from historical_backfill_research import (
    ARCHIVE_BUCKET,
    ARCHIVE_MARKET_DATA_PREFIX,
    DATA_LIMITATIONS,
    DEFAULT_L2_DATATYPE,
    EXECUTION_ENABLED,
    PAPER_ONLY,
    SAFETY_STATEMENT,
)

DEFAULT_OUTPUT_DIR = Path("reports/historical_l2_backfill_plans")
DEFAULT_DOWNLOAD_ROOT = "downloads/hyperliquid"
HOURS_PER_DAY = 24

ARCHIVE_WARNING = (
    "Historical archive data may be missing or incomplete. Planned paths are "
    "expected locations only; availability is not guaranteed."
)

NOTHING_DOWNLOADED_WARNING = "No files were downloaded. This planner is offline only."


@dataclass
class BackfillPlanConfig:
    start_date: str
    end_date: str
    coins: list[str]
    datatype: str = DEFAULT_L2_DATATYPE
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)


def validate_date_yyyymmdd(value: str | int) -> str:
    """Validate YYYYMMDD format and that the date is real."""
    text = str(value).strip()
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"date must be YYYYMMDD, got {value!r}")
    year = int(text[0:4])
    month = int(text[4:6])
    day = int(text[6:8])
    try:
        date(year, month, day)
    except ValueError as exc:
        raise ValueError(f"invalid calendar date: {value!r}") from exc
    return text


def _normalize_coin(coin: str) -> str:
    text = coin.strip().upper()
    if not text:
        raise ValueError("coin must be non-empty")
    return text


def _normalize_hour(hour: int | str) -> str:
    value = int(hour)
    if value < 0 or value > 23:
        raise ValueError(f"hour must be 0-23, got {hour!r}")
    return str(value)


def iter_dates(start_date: str | int, end_date: str | int) -> Iterator[str]:
    """Yield inclusive dates as YYYYMMDD strings from start through end."""
    start = validate_date_yyyymmdd(start_date)
    end = validate_date_yyyymmdd(end_date)
    if end < start:
        raise ValueError(f"end_date {end!r} is before start_date {start!r}")

    current = date(int(start[0:4]), int(start[4:6]), int(start[6:8]))
    last = date(int(end[0:4]), int(end[4:6]), int(end[6:8]))
    while current <= last:
        yield current.strftime("%Y%m%d")
        current += timedelta(days=1)


def build_l2_s3_path(
    date_value: str | int,
    hour: int | str,
    coin: str,
    datatype: str = DEFAULT_L2_DATATYPE,
) -> str:
    """Return the expected Hyperliquid archive S3 path (no download)."""
    normalized_date = validate_date_yyyymmdd(date_value)
    normalized_hour = _normalize_hour(hour)
    normalized_coin = _normalize_coin(coin)
    datatype_name = datatype.strip() or DEFAULT_L2_DATATYPE
    return (
        f"s3://{ARCHIVE_BUCKET}/{ARCHIVE_MARKET_DATA_PREFIX}/"
        f"{normalized_date}/{normalized_hour}/{datatype_name}/{normalized_coin}.lz4"
    )


def build_local_download_path(date_value: str | int, hour: int | str, coin: str) -> str:
    """Return the expected local .lz4 download path."""
    normalized_date = validate_date_yyyymmdd(date_value)
    normalized_hour = _normalize_hour(hour)
    normalized_coin = _normalize_coin(coin)
    return f"{DEFAULT_DOWNLOAD_ROOT}/{normalized_date}/{normalized_hour}/{normalized_coin}.lz4"


def build_local_decompressed_path(date_value: str | int, hour: int | str, coin: str) -> str:
    """Return the expected local decompressed JSONL path."""
    normalized_date = validate_date_yyyymmdd(date_value)
    normalized_hour = _normalize_hour(hour)
    normalized_coin = _normalize_coin(coin)
    return f"{DEFAULT_DOWNLOAD_ROOT}/{normalized_date}/{normalized_hour}/{normalized_coin}.jsonl"


def _manifest_stamp(manifest: dict[str, Any]) -> str:
    coins_slug = "_".join(manifest.get("coins", []))
    return f"{manifest['start_date']}_{manifest['end_date']}_{coins_slug}"


def generate_backfill_manifest(config: BackfillPlanConfig) -> dict[str, Any]:
    """Build a planned manifest of archive files for the configured range."""
    start_date = validate_date_yyyymmdd(config.start_date)
    end_date = validate_date_yyyymmdd(config.end_date)
    if end_date < start_date:
        raise ValueError(f"end_date {end_date!r} is before start_date {start_date!r}")

    coins = [_normalize_coin(coin) for coin in config.coins]
    if not coins:
        raise ValueError("coins must contain at least one coin")

    dates = list(iter_dates(start_date, end_date))
    total_days = len(dates)
    total_hours = total_days * HOURS_PER_DAY
    estimated_files_by_coin = {coin: total_hours for coin in coins}

    files: list[dict[str, str]] = []
    for day in dates:
        for hour in range(HOURS_PER_DAY):
            hour_text = str(hour)
            for coin in coins:
                files.append(
                    {
                        "date": day,
                        "hour": hour_text,
                        "coin": coin,
                        "s3_path": build_l2_s3_path(day, hour_text, coin, config.datatype),
                        "local_lz4_path": build_local_download_path(day, hour_text, coin),
                        "local_jsonl_path": build_local_decompressed_path(day, hour_text, coin),
                        "status": "planned",
                    }
                )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "source": ARCHIVE_BUCKET,
        "warning": ARCHIVE_WARNING,
        "safety_statement": SAFETY_STATEMENT,
        "start_date": start_date,
        "end_date": end_date,
        "coins": coins,
        "datatype": config.datatype,
        "output_dir": str(config.output_dir),
        "total_days": total_days,
        "total_hours": total_hours,
        "total_files": len(files),
        "estimated_files_by_coin": estimated_files_by_coin,
        "files": files,
    }


def write_manifest_json(manifest: dict[str, Any]) -> Path:
    """Save manifest JSON under reports/historical_l2_backfill_plans/."""
    output_dir = Path(manifest.get("output_dir", DEFAULT_OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _manifest_stamp(manifest)
    output_path = output_dir / f"historical_l2_backfill_{stamp}.json"
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["json_report_path"] = str(output_path)
    return output_path


def write_manifest_markdown(manifest: dict[str, Any]) -> Path:
    """Save manifest Markdown under reports/historical_l2_backfill_plans/."""
    output_dir = Path(manifest.get("output_dir", DEFAULT_OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _manifest_stamp(manifest)
    output_path = output_dir / f"historical_l2_backfill_{stamp}.md"

    example_files = manifest.get("files", [])[:3]
    example_s3_paths = [entry["s3_path"] for entry in example_files]
    example_local_lz4 = [entry["local_lz4_path"] for entry in example_files]
    example_local_jsonl = [entry["local_jsonl_path"] for entry in example_files]

    estimated = manifest.get("estimated_files_by_coin", {})
    file_count_lines = [
        f"- total_days: {manifest.get('total_days', 0)}",
        f"- total_hours: {manifest.get('total_hours', 0)}",
        f"- total_files: {manifest.get('total_files', 0)}",
    ]
    for coin, count in estimated.items():
        file_count_lines.append(f"- {coin}: {count} files")

    limitation_lines = [f"- {item}" for item in DATA_LIMITATIONS]
    limitation_lines.append(f"- {manifest.get('warning', ARCHIVE_WARNING)}")

    lines = [
        "# Historical L2 Backfill Manifest",
        "",
        "## Summary",
        "",
        f"- generated_at: {manifest.get('generated_at', 'n/a')}",
        f"- source: {manifest.get('source', ARCHIVE_BUCKET)}",
        f"- start_date: {manifest.get('start_date')}",
        f"- end_date: {manifest.get('end_date')}",
        f"- coins: {', '.join(manifest.get('coins', []))}",
        f"- datatype: {manifest.get('datatype', DEFAULT_L2_DATATYPE)}",
        f"- paper_only: {manifest.get('paper_only')}",
        f"- execution_enabled: {manifest.get('execution_enabled')}",
        "",
        "## File counts",
        "",
        *file_count_lines,
        "",
        "## Example S3 paths",
        "",
    ]
    if example_s3_paths:
        lines.extend(f"- `{path}`" for path in example_s3_paths)
    else:
        lines.append("- (none planned)")

    lines.extend(["", "## Local paths", ""])
    if example_local_lz4:
        lines.append("Example `.lz4` download paths:")
        lines.extend(f"- `{path}`" for path in example_local_lz4)
        lines.append("")
        lines.append("Example decompressed `.jsonl` paths:")
        lines.extend(f"- `{path}`" for path in example_local_jsonl)
    else:
        lines.append("- (none planned)")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            *limitation_lines,
            "",
            "## Next manual steps",
            "",
            "1. Review this manifest and confirm the date range and coins.",
            "2. Use `historical_backfill_manual_guide.py` for one-hour manual commands.",
            "3. Download one `(date, hour, coin)` file manually with `aws s3 cp --no-sign-request`.",
            "4. Decompress locally with `lz4 -d` and import via `historical_l2_importer.py import-local`.",
            "5. Validate coverage with `historical_l2_importer.py summarize` before bulk download.",
            "",
            "## Research-only disclaimer",
            "",
            manifest.get("safety_statement", SAFETY_STATEMENT),
            "",
            NOTHING_DOWNLOADED_WARNING,
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    manifest["markdown_report_path"] = str(output_path)
    return output_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan Hyperliquid historical L2 archive files (offline, no downloads)."
    )
    parser.add_argument("--start-date", required=True, help="Start date YYYYMMDD (inclusive)")
    parser.add_argument("--end-date", required=True, help="End date YYYYMMDD (inclusive)")
    parser.add_argument(
        "--coins",
        nargs="+",
        required=True,
        help="Coins to plan, e.g. BTC ETH SOL",
    )
    parser.add_argument(
        "--datatype",
        default=DEFAULT_L2_DATATYPE,
        help=f"Archive datatype folder (default: {DEFAULT_L2_DATATYPE})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Report output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = BackfillPlanConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        coins=list(args.coins),
        datatype=args.datatype,
        output_dir=Path(args.output_dir),
    )
    manifest = generate_backfill_manifest(config)
    json_path = write_manifest_json(manifest)
    markdown_path = write_manifest_markdown(manifest)

    print("Historical L2 backfill manifest generated (offline planner).")
    print(f"JSON manifest: {json_path}")
    print(f"Markdown manifest: {markdown_path}")
    print(f"Date range: {manifest['start_date']} to {manifest['end_date']} (inclusive)")
    print(f"Coins: {', '.join(manifest['coins'])}")
    print(f"Total files planned: {manifest['total_files']}")
    print(f"Warning: {ARCHIVE_WARNING}")
    print(f"Warning: {NOTHING_DOWNLOADED_WARNING}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
