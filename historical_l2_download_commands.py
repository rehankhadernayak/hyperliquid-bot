"""Offline command generator for Hyperliquid historical L2 archive downloads.

Reads a backfill manifest and produces copyable shell commands for mkdir, S3 download,
decompression, and local import. Does not execute commands, call the network, or invoke
external shell tools.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from historical_backfill_research import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT

DEFAULT_MANIFEST_DIR = Path("reports/historical_l2_backfill_plans")
DEFAULT_OUTPUT_DIR = Path("reports/historical_l2_download_commands")

REQUIRED_FILE_FIELDS = (
    "s3_path",
    "local_lz4_path",
    "local_jsonl_path",
    "coin",
    "date",
    "hour",
)

GENERATED_ONLY_WARNING = (
    "Commands are generated only and are not executed by this tool. "
    "Run each command manually in your shell."
)


def discover_latest_manifest(manifest_dir: Path | str | None = None) -> Path:
    """Return the newest manifest JSON under reports/historical_l2_backfill_plans/."""
    root = Path(manifest_dir if manifest_dir is not None else DEFAULT_MANIFEST_DIR)
    if not root.is_dir():
        raise FileNotFoundError(f"manifest directory not found: {root}")

    candidates = sorted(root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no manifest JSON files found in {root}")

    return candidates[0]


def load_manifest(path: Path | str) -> dict[str, Any]:
    """Load and validate a backfill manifest JSON file."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("paper_only") is not True:
        raise ValueError("manifest paper_only must be true")
    if manifest.get("execution_enabled") is not False:
        raise ValueError("manifest execution_enabled must be false")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest files must be a non-empty list")

    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise ValueError(f"manifest files[{index}] must be an object")
        for field in REQUIRED_FILE_FIELDS:
            value = entry.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"manifest files[{index}] missing required field: {field}")

    manifest["_source_manifest_path"] = str(manifest_path.resolve())
    return manifest


def _selected_files(manifest: dict[str, Any], limit: int | None) -> list[dict[str, Any]]:
    files = manifest["files"]
    if limit is None:
        return list(files)
    if limit < 1:
        raise ValueError("limit must be at least 1 when provided")
    return list(files[:limit])


def build_mkdir_commands(manifest: dict[str, Any], *, limit: int | None = None) -> list[str]:
    """Return unique mkdir commands for downloads/hyperliquid/{date}/{hour} folders."""
    seen: set[tuple[str, str]] = set()
    commands: list[str] = []

    for entry in _selected_files(manifest, limit):
        key = (str(entry["date"]), str(entry["hour"]))
        if key in seen:
            continue
        seen.add(key)
        commands.append(f"mkdir -p downloads/hyperliquid/{entry['date']}/{entry['hour']}")

    return commands


def build_download_commands(manifest: dict[str, Any], *, limit: int | None = None) -> list[str]:
    """Return aws s3 cp commands for each planned .lz4 file."""
    return [
        f"aws s3 cp --no-sign-request {entry['s3_path']} {entry['local_lz4_path']}"
        for entry in _selected_files(manifest, limit)
    ]


def build_decompress_commands(manifest: dict[str, Any], *, limit: int | None = None) -> list[str]:
    """Return lz4 decompress commands for each planned .lz4 file."""
    return [
        f"lz4 -d {entry['local_lz4_path']} {entry['local_jsonl_path']}"
        for entry in _selected_files(manifest, limit)
    ]


def build_import_commands(manifest: dict[str, Any], *, limit: int | None = None) -> list[str]:
    """Return historical_l2_importer import-local commands for each file."""
    return [
        (
            "python3 historical_l2_importer.py import-local "
            f"--input {entry['local_jsonl_path']} --coin {entry['coin']}"
        )
        for entry in _selected_files(manifest, limit)
    ]


def build_command_plan(manifest: dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    """Build a structured command plan from a validated manifest."""
    selected = _selected_files(manifest, limit)
    mkdir_cmds = build_mkdir_commands(manifest, limit=limit)
    download_cmds = build_download_commands(manifest, limit=limit)
    decompress_cmds = build_decompress_commands(manifest, limit=limit)
    import_cmds = build_import_commands(manifest, limit=limit)

    total_commands = len(mkdir_cmds) + len(download_cmds) + len(decompress_cmds) + len(import_cmds)

    source_path = manifest.get("_source_manifest_path") or manifest.get("source_manifest_path", "")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "paper_only": True,
        "execution_enabled": False,
        "source_manifest_path": source_path,
        "limit": limit,
        "total_files": len(selected),
        "total_commands": total_commands,
        "warning": GENERATED_ONLY_WARNING,
        "command_sections": {
            "mkdir": mkdir_cmds,
            "download": download_cmds,
            "decompress": decompress_cmds,
            "import": import_cmds,
        },
        "files": selected,
    }


def _plan_stamp(plan: dict[str, Any]) -> str:
    source = plan.get("source_manifest_path", "")
    if source:
        stem = Path(source).stem
        if stem.startswith("historical_l2_backfill_"):
            return stem.replace("historical_l2_backfill_", "historical_l2_download_commands_", 1)
    limit = plan.get("limit")
    limit_suffix = f"_limit{limit}" if limit is not None else ""
    return f"historical_l2_download_commands{limit_suffix}"


def write_command_plan_json(
    plan: dict[str, Any],
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Save command plan JSON under reports/historical_l2_download_commands/."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / f"{_plan_stamp(plan)}.json"
    output_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    plan["json_report_path"] = str(output_path)
    return output_path


def write_command_plan_markdown(
    plan: dict[str, Any],
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Save command plan Markdown under reports/historical_l2_download_commands/."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / f"{_plan_stamp(plan)}.md"

    sections = plan.get("command_sections", {})
    mkdir_cmds = sections.get("mkdir", [])
    download_cmds = sections.get("download", [])
    decompress_cmds = sections.get("decompress", [])
    import_cmds = sections.get("import", [])

    def _command_block(commands: list[str]) -> list[str]:
        if not commands:
            return ["- (none)"]
        return [f"```bash\n{command}\n```" for command in commands]

    lines = [
        "# Historical L2 Download Commands",
        "",
        "## Summary",
        "",
        f"- generated_at: {plan.get('generated_at', 'n/a')}",
        f"- source_manifest_path: {plan.get('source_manifest_path', 'n/a')}",
        f"- limit: {plan.get('limit') if plan.get('limit') is not None else 'none (all files)'}",
        f"- total_files: {plan.get('total_files', 0)}",
        f"- total_commands: {plan.get('total_commands', 0)}",
        f"- paper_only: {plan.get('paper_only')}",
        f"- execution_enabled: {plan.get('execution_enabled')}",
        "",
        "## Important safety notes",
        "",
        f"- {GENERATED_ONLY_WARNING}",
        "- This tool does not download, decompress, import, or execute shell commands.",
        "- No wallets, keys, signing, orders, or execution endpoints are involved.",
        "",
        "## Required tools: aws, lz4",
        "",
        "- `aws` CLI for requester-pays S3 copy (`aws s3 cp --no-sign-request ...`)",
        "- `lz4` for decompressing archive files to JSONL",
        "",
        "Check local availability with:",
        "",
        "```bash",
        "python3 historical_backfill_tool_check.py",
        "```",
        "",
        "## mkdir commands",
        "",
    ]
    lines.extend(_command_block(mkdir_cmds))
    lines.extend(["", "## download commands", ""])
    lines.extend(_command_block(download_cmds))
    lines.extend(["", "## decompress commands", ""])
    lines.extend(_command_block(decompress_cmds))
    lines.extend(["", "## import commands", ""])
    lines.extend(_command_block(import_cmds))
    lines.extend(
        [
            "",
            "## Suggested staged workflow",
            "",
            "1. Start with a small sample: `python3 historical_l2_download_commands.py --latest --limit 3`",
            "2. Run the generated mkdir, download, decompress, and import commands manually.",
            "3. Validate coverage with `python3 historical_l2_importer.py summarize --coin COIN`.",
            "4. Expand to one full day (24 hours × coins) once the sample looks acceptable.",
            "5. Only then plan larger date ranges from the backfill manifest.",
            "",
            "## Research-only disclaimer",
            "",
            SAFETY_STATEMENT,
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    plan["markdown_report_path"] = str(output_path)
    return output_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate manual Hyperliquid historical L2 download/decompress/import commands "
            "from a backfill manifest (offline, no execution)."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--manifest",
        help="Path to a backfill manifest JSON file",
    )
    source.add_argument(
        "--latest",
        action="store_true",
        help="Use the newest manifest under reports/historical_l2_backfill_plans/",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Include only the first N files from the manifest (for safe small tests)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Command plan output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.latest:
        manifest_path = discover_latest_manifest()
    else:
        manifest_path = Path(args.manifest)

    manifest = load_manifest(manifest_path)
    plan = build_command_plan(manifest, limit=args.limit)
    json_path = write_command_plan_json(plan, output_dir=args.output_dir)
    markdown_path = write_command_plan_markdown(plan, output_dir=args.output_dir)

    print("Historical L2 download command plan generated (offline, not executed).")
    print(f"JSON command plan: {json_path}")
    print(f"Markdown command plan: {markdown_path}")
    print(f"Source manifest: {plan['source_manifest_path']}")
    print(f"Files included: {plan['total_files']}")
    print(f"Total commands: {plan['total_commands']}")
    print(f"Warning: {GENERATED_ONLY_WARNING}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
