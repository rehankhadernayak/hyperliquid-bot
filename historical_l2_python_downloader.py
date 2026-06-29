"""Python-native sample downloader for Hyperliquid historical L2 archive files.

Downloads a tiny number of planned archive files from public S3 using unsigned
access, then decompresses them with the lz4 Python library. Research-only;
no wallets, keys, signing, or trading.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import lz4.frame
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from historical_backfill_research import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT

DEFAULT_COMMAND_PLAN_DIR = Path("reports/historical_l2_download_commands")
DEFAULT_BACKFILL_PLAN_DIR = Path("reports/historical_l2_backfill_plans")
DEFAULT_REPORT_DIR = Path("reports/historical_l2_python_downloads")

DEFAULT_MAX_FILES = 1
DEFAULT_PROBE_MAX_FILES = 3
HARD_CAP_MAX_FILES = 3
HARD_CAP_MAX_FILES_ALLOW_MORE = 24

REQUIRED_FILE_FIELDS = (
    "s3_path",
    "local_lz4_path",
    "local_jsonl_path",
    "coin",
    "date",
    "hour",
)

PLAN_SEARCH_DIRS = (DEFAULT_COMMAND_PLAN_DIR, DEFAULT_BACKFILL_PLAN_DIR)


def discover_latest_command_plan(
    command_plan_dir: Path | str | None = None,
    backfill_plan_dir: Path | str | None = None,
) -> Path:
    """Return the newest JSON command plan or backfill plan."""
    search_dirs: list[Path] = []
    if command_plan_dir is not None:
        search_dirs.append(Path(command_plan_dir))
    elif backfill_plan_dir is not None:
        search_dirs.append(Path(backfill_plan_dir))
    else:
        search_dirs.extend(PLAN_SEARCH_DIRS)

    candidates: list[Path] = []
    for root in search_dirs:
        if not root.is_dir():
            continue
        candidates.extend(root.glob("*.json"))

    if not candidates:
        searched = ", ".join(str(d) for d in search_dirs)
        raise FileNotFoundError(f"no command or backfill plan JSON files found in: {searched}")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_plan(path: Path | str) -> dict[str, Any]:
    """Load and validate a command plan or backfill plan JSON file."""
    plan_path = Path(path)
    if not plan_path.is_file():
        raise FileNotFoundError(f"plan not found: {plan_path}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    if plan.get("paper_only") is not True:
        raise ValueError("plan paper_only must be true")
    if plan.get("execution_enabled") is not False:
        raise ValueError("plan execution_enabled must be false")

    files = plan.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("plan files must be a non-empty list")

    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise ValueError(f"plan files[{index}] must be an object")
        for field in REQUIRED_FILE_FIELDS:
            value = entry.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"plan files[{index}] missing required field: {field}")

    plan["_source_plan_path"] = str(plan_path.resolve())
    return plan


def parse_s3_path(s3_path: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    if not s3_path.startswith("s3://"):
        raise ValueError(f"invalid S3 path (must start with s3://): {s3_path}")

    remainder = s3_path[5:]
    if "/" not in remainder:
        raise ValueError(f"invalid S3 path (missing key): {s3_path}")

    bucket, key = remainder.split("/", 1)
    if not bucket or not key:
        raise ValueError(f"invalid S3 path (empty bucket or key): {s3_path}")

    return bucket, key


REQUESTER_PAYS_WARNING = (
    "Requester Pays may incur AWS charges. Use only with adult/mentor approval."
)


def create_s3_client(
    requester_pays: bool = False,
    aws_profile: str | None = None,
):
    """Create an S3 client for anonymous or authenticated Requester Pays access."""
    if not requester_pays:
        return boto3.client(
            "s3",
            config=Config(signature_version=UNSIGNED),
        )

    if aws_profile:
        session = boto3.Session(profile_name=aws_profile)
    else:
        session = boto3.Session()
    return session.client("s3")


def create_unsigned_s3_client():
    """Create a boto3 S3 client with unsigned (no-signature) configuration."""
    return create_s3_client(requester_pays=False)


def file_entry_from_s3_path(s3_path: str) -> dict[str, Any]:
    """Build a minimal file entry from an archive S3 path (no manifest required)."""
    _, key = parse_s3_path(s3_path)
    parts = key.split("/")
    if len(parts) < 5 or parts[0] != "market_data" or parts[3] != "l2Book":
        raise ValueError(f"cannot parse archive path metadata from: {s3_path}")

    coin = Path(parts[4]).stem
    if not coin:
        raise ValueError(f"cannot parse coin from archive path: {s3_path}")

    return {
        "s3_path": s3_path,
        "coin": coin,
        "date": parts[1],
        "hour": parts[2],
        "local_lz4_path": "",
        "local_jsonl_path": "",
    }


def _get_object_kwargs(
    bucket: str,
    key: str,
    *,
    requester_pays: bool = False,
    range_header: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key}
    if range_header is not None:
        kwargs["Range"] = range_header
    if requester_pays:
        kwargs["RequestPayer"] = "requester"
    return kwargs


def _aws_access_report_fields(
    requester_pays: bool,
    aws_profile: str | None,
) -> dict[str, Any]:
    return {
        "requester_pays": requester_pays,
        "aws_profile_used": (aws_profile if aws_profile else "default_chain") if requester_pays else None,
    }


def _aws_access_warnings(
    requester_pays: bool,
    existing: list[str] | None = None,
) -> list[str]:
    warnings = list(existing or [])
    if requester_pays and REQUESTER_PAYS_WARNING not in warnings:
        warnings.append(REQUESTER_PAYS_WARNING)
    return warnings


def classify_download_error(exc: BaseException) -> str:
    """Classify an S3 download failure into a stable error category."""
    if isinstance(exc, ClientError):
        error_code = exc.response.get("Error", {}).get("Code", "")
        error_message = exc.response.get("Error", {}).get("Message", "")
        http_status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if error_code in {"403", "AccessDenied", "Forbidden"} or http_status == 403:
            combined = f"{error_message} {exc}".lower()
            if "requester pays" in combined or "please authenticate" in combined:
                return "requester_pays_auth_required"
            return "forbidden"
        if error_code in {"404", "NoSuchKey", "NotFound"} or http_status == 404:
            return "not_found"
    if isinstance(exc, (BotoCoreError, ConnectionError, TimeoutError, OSError)):
        return "network_error"
    return "unknown_error"


def select_plan_files(
    files: list[dict[str, Any]],
    *,
    coin: str | None = None,
    date: str | None = None,
    hour: str | None = None,
) -> list[dict[str, Any]]:
    """Filter plan file entries by optional coin/date/hour selectors."""
    selected = list(files)
    if coin is not None:
        selected = [entry for entry in selected if str(entry.get("coin")) == coin]
    if date is not None:
        selected = [entry for entry in selected if str(entry.get("date")) == date]
    if hour is not None:
        selected = [entry for entry in selected if str(entry.get("hour")) == hour]
    return selected


def _build_import_command(file_entry: dict[str, Any]) -> str:
    return (
        "python3 historical_l2_importer.py import-local "
        f"--input {file_entry['local_jsonl_path']} --coin {file_entry['coin']}"
    )


def download_one_file(
    client,
    file_entry: dict[str, Any],
    *,
    overwrite: bool = False,
    requester_pays: bool = False,
) -> dict[str, Any]:
    """Download one .lz4 file from S3 to local_lz4_path via get_object streaming."""
    local_path = Path(file_entry["local_lz4_path"])
    result: dict[str, Any] = {
        "coin": file_entry["coin"],
        "date": file_entry["date"],
        "hour": file_entry["hour"],
        "s3_path": file_entry["s3_path"],
        "local_lz4_path": file_entry["local_lz4_path"],
        "download_status": "pending",
        "download_error": None,
        "download_error_type": None,
    }

    if local_path.is_file() and not overwrite:
        result["download_status"] = "skipped_existing"
        return result

    try:
        bucket, key = parse_s3_path(file_entry["s3_path"])
        local_path.parent.mkdir(parents=True, exist_ok=True)
        response = client.get_object(
            **_get_object_kwargs(bucket, key, requester_pays=requester_pays)
        )
        body = response["Body"]
        try:
            with local_path.open("wb") as handle:
                for chunk in body.iter_chunks():
                    if chunk:
                        handle.write(chunk)
        finally:
            body.close()
        result["download_status"] = "downloaded"
    except Exception as exc:  # noqa: BLE001 - capture per-file failures in report
        result["download_status"] = "failed"
        result["download_error_type"] = classify_download_error(exc)
        result["download_error"] = str(exc)

    return result


def decompress_one_file(
    file_entry: dict[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Decompress one .lz4 file to .jsonl using lz4.frame."""
    lz4_path = Path(file_entry["local_lz4_path"])
    jsonl_path = Path(file_entry["local_jsonl_path"])
    result: dict[str, Any] = {
        "coin": file_entry["coin"],
        "date": file_entry["date"],
        "hour": file_entry["hour"],
        "local_lz4_path": file_entry["local_lz4_path"],
        "local_jsonl_path": file_entry["local_jsonl_path"],
        "decompress_status": "pending",
        "decompress_error": None,
    }

    if jsonl_path.is_file() and not overwrite:
        result["decompress_status"] = "skipped_existing"
        return result

    if not lz4_path.is_file():
        result["decompress_status"] = "failed"
        result["decompress_error"] = f"source lz4 file not found: {lz4_path}"
        return result

    try:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        compressed = lz4_path.read_bytes()
        decompressed = lz4.frame.decompress(compressed)
        jsonl_path.write_bytes(decompressed)
        result["decompress_status"] = "decompressed"
    except Exception as exc:  # noqa: BLE001 - capture per-file failures in report
        result["decompress_status"] = "failed"
        result["decompress_error"] = str(exc)

    return result


def resolve_max_files(
    max_files: int | None,
    *,
    allow_more: bool = False,
    default: int = DEFAULT_MAX_FILES,
) -> int:
    """Apply default and hard caps to max_files."""
    requested = default if max_files is None else max_files
    if requested < 1:
        raise ValueError("max_files must be at least 1")

    cap = HARD_CAP_MAX_FILES_ALLOW_MORE if allow_more else HARD_CAP_MAX_FILES
    if requested > cap:
        if allow_more:
            raise ValueError(
                f"max_files {requested} exceeds hard cap of {HARD_CAP_MAX_FILES_ALLOW_MORE} "
                "even with --allow-more"
            )
        raise ValueError(
            f"max_files {requested} exceeds hard cap of {HARD_CAP_MAX_FILES}; "
            "pass --allow-more to raise the cap (max 24)"
        )

    return requested


def resolve_probe_max_files(
    max_files: int | None,
    *,
    allow_more: bool = False,
) -> int:
    """Apply default and hard caps to probe max_files."""
    return resolve_max_files(
        max_files,
        allow_more=allow_more,
        default=DEFAULT_PROBE_MAX_FILES,
    )


def probe_one_file(
    client,
    file_entry: dict[str, Any],
    *,
    requester_pays: bool = False,
) -> dict[str, Any]:
    """Lightweight S3 access check using a small Range request (no full download)."""
    result: dict[str, Any] = {
        "coin": file_entry["coin"],
        "date": file_entry["date"],
        "hour": file_entry["hour"],
        "s3_path": file_entry["s3_path"],
        "accessible": False,
        "error_class": "none",
        "error_message": None,
        "bytes_read": 0,
    }

    try:
        bucket, key = parse_s3_path(file_entry["s3_path"])
        response = client.get_object(
            **_get_object_kwargs(
                bucket,
                key,
                requester_pays=requester_pays,
                range_header="bytes=0-1023",
            )
        )
        body = response["Body"]
        try:
            data = body.read()
            result["bytes_read"] = len(data)
            result["accessible"] = True
        finally:
            body.close()
    except Exception as exc:  # noqa: BLE001 - capture per-file failures in report
        result["accessible"] = False
        result["error_class"] = classify_download_error(exc)
        result["error_message"] = str(exc)

    return result


def run_availability_probe(
    plan: dict[str, Any],
    *,
    max_files: int = DEFAULT_PROBE_MAX_FILES,
    client=None,
    coin: str | None = None,
    date: str | None = None,
    hour: str | None = None,
    requester_pays: bool = False,
    aws_profile: str | None = None,
) -> dict[str, Any]:
    """Probe the first N matching plan files for S3 accessibility without full downloads."""
    filtered = select_plan_files(plan["files"], coin=coin, date=date, hour=hour)
    selected = list(filtered[:max_files])
    source_plan_path = plan.get("_source_plan_path") or plan.get("source_manifest_path", "")

    if client is None:
        client = create_s3_client(requester_pays=requester_pays, aws_profile=aws_profile)

    results: list[dict[str, Any]] = []
    warnings = _aws_access_warnings(
        requester_pays,
        [
            "This tool probes only a tiny sample; it does not verify all-time archive coverage.",
            "Archive data may be missing or incomplete; accessibility is not guaranteed.",
        ],
    )

    accessible_count = 0
    failed_count = 0

    for entry in selected:
        probe_result = probe_one_file(client, entry, requester_pays=requester_pays)
        results.append(probe_result)
        if probe_result["accessible"]:
            accessible_count += 1
        else:
            failed_count += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "paper_only": True,
        "execution_enabled": False,
        "mode": "availability_probe",
        "source_plan_path": source_plan_path,
        "requested_files": max_files,
        "probed_count": len(selected),
        "accessible_count": accessible_count,
        "failed_count": failed_count,
        "selector": {
            "coin": coin,
            "date": date,
            "hour": hour,
        },
        "results": results,
        "warnings": warnings,
        **_aws_access_report_fields(requester_pays, aws_profile),
    }


def run_probe_path(
    s3_path: str,
    *,
    client=None,
    requester_pays: bool = False,
    aws_profile: str | None = None,
) -> dict[str, Any]:
    """Probe a single archive S3 path without requiring a manifest."""
    file_entry = file_entry_from_s3_path(s3_path)

    if client is None:
        client = create_s3_client(requester_pays=requester_pays, aws_profile=aws_profile)

    probe_result = probe_one_file(client, file_entry, requester_pays=requester_pays)
    warnings = _aws_access_warnings(
        requester_pays,
        [
            "Single-path probe; no manifest or bulk coverage check performed.",
            "Archive data may be missing or incomplete; accessibility is not guaranteed.",
        ],
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "paper_only": True,
        "execution_enabled": False,
        "mode": "availability_probe",
        "source_plan_path": None,
        "requested_files": 1,
        "probed_count": 1,
        "accessible_count": 1 if probe_result["accessible"] else 0,
        "failed_count": 0 if probe_result["accessible"] else 1,
        "selector": {
            "coin": file_entry["coin"],
            "date": file_entry["date"],
            "hour": file_entry["hour"],
            "probe_path": s3_path,
        },
        "results": [probe_result],
        "warnings": warnings,
        **_aws_access_report_fields(requester_pays, aws_profile),
    }


def run_sample_download(
    plan: dict[str, Any],
    *,
    max_files: int = DEFAULT_MAX_FILES,
    overwrite: bool = False,
    client=None,
    coin: str | None = None,
    date: str | None = None,
    hour: str | None = None,
    requester_pays: bool = False,
    aws_profile: str | None = None,
) -> dict[str, Any]:
    """Download and decompress the first N files from a validated plan."""
    filtered = select_plan_files(plan["files"], coin=coin, date=date, hour=hour)
    selected = list(filtered[:max_files])
    source_plan_path = plan.get("_source_plan_path") or plan.get("source_manifest_path", "")

    if client is None:
        client = create_s3_client(requester_pays=requester_pays, aws_profile=aws_profile)

    file_results: list[dict[str, Any]] = []
    warnings = _aws_access_warnings(
        requester_pays,
        [
            "This tool downloads only a tiny sample; it does not support all-time bulk downloads.",
            "Archive data may be missing or incomplete; availability is not guaranteed.",
        ],
    )
    next_import_commands: list[str] = []

    downloaded_count = 0
    decompressed_count = 0
    skipped_count = 0
    failed_count = 0

    for entry in selected:
        download_result = download_one_file(
            client,
            entry,
            overwrite=overwrite,
            requester_pays=requester_pays,
        )
        dl_status = download_result["download_status"]
        local_lz4_path = Path(entry["local_lz4_path"])

        if dl_status != "downloaded" and not local_lz4_path.is_file():
            decompress_result = {
                "decompress_status": "skipped",
                "decompress_error": "download_failed",
            }
        else:
            decompress_result = decompress_one_file(entry, overwrite=overwrite)

        combined = {
            **download_result,
            "decompress_status": decompress_result["decompress_status"],
            "decompress_error": decompress_result.get("decompress_error"),
            "local_jsonl_path": entry["local_jsonl_path"],
        }
        file_results.append(combined)

        dc_status = decompress_result["decompress_status"]
        file_failed = False

        if dl_status == "downloaded":
            downloaded_count += 1
        elif dl_status == "skipped_existing":
            skipped_count += 1
        elif dl_status == "failed":
            file_failed = True

        if dc_status == "decompressed":
            decompressed_count += 1
            next_import_commands.append(_build_import_command(entry))
        elif dc_status == "skipped_existing":
            skipped_count += 1
            next_import_commands.append(_build_import_command(entry))
        elif dc_status == "failed":
            file_failed = True

        if file_failed:
            failed_count += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "paper_only": True,
        "execution_enabled": False,
        "source_plan_path": source_plan_path,
        "requested_files": max_files,
        "selected_files": len(selected),
        "selector": {
            "coin": coin,
            "date": date,
            "hour": hour,
        },
        "downloaded_count": downloaded_count,
        "decompressed_count": decompressed_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "file_results": file_results,
        "warnings": warnings,
        "next_import_commands": next_import_commands,
        **_aws_access_report_fields(requester_pays, aws_profile),
    }


def _report_stamp(report: dict[str, Any]) -> str:
    ts = report.get("generated_at", datetime.now(timezone.utc).isoformat())
    safe_ts = ts.replace(":", "").replace("-", "").replace(".", "")
    prefix = (
        "historical_l2_python_probe"
        if report.get("mode") == "availability_probe"
        else "historical_l2_python_download"
    )
    return f"{prefix}_{safe_ts}"


def write_report_json(
    report: dict[str, Any],
    output_dir: Path | str = DEFAULT_REPORT_DIR,
) -> Path:
    """Save download report JSON under reports/historical_l2_python_downloads/."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / f"{_report_stamp(report)}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["json_report_path"] = str(output_path)
    return output_path


def write_report_markdown(
    report: dict[str, Any],
    output_dir: Path | str = DEFAULT_REPORT_DIR,
) -> Path:
    """Save download or probe report Markdown under reports/historical_l2_python_downloads/."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / f"{_report_stamp(report)}.md"

    if report.get("mode") == "availability_probe":
        return _write_probe_report_markdown(report, output_path)

    failures: list[str] = []
    for entry in report.get("file_results", []):
        if entry.get("download_status") == "failed":
            failures.append(
                f"- download failed ({entry.get('coin')} {entry.get('date')} hour {entry.get('hour')}): "
                f"{entry.get('download_error', 'unknown error')}"
            )
        if entry.get("decompress_status") == "failed":
            failures.append(
                f"- decompress failed ({entry.get('coin')} {entry.get('date')} hour {entry.get('hour')}): "
                f"{entry.get('decompress_error', 'unknown error')}"
            )

    file_lines: list[str] = []
    for entry in report.get("file_results", []):
        file_lines.append(
            f"- coin={entry.get('coin')} date={entry.get('date')} hour={entry.get('hour')}: "
            f"s3_path={entry.get('s3_path')}, "
            f"download={entry.get('download_status')}, decompress={entry.get('decompress_status')}"
        )

    import_lines = report.get("next_import_commands") or ["- (none — decompress files first)"]
    if import_lines and not import_lines[0].startswith("-") and not import_lines[0].startswith("```"):
        import_lines = [f"```bash\n{cmd}\n```" for cmd in import_lines]

    warning_lines = [f"- {w}" for w in report.get("warnings", [])]

    lines = [
        "# Historical L2 Python Download Report",
        "",
        "## Summary",
        "",
        f"- generated_at: {report.get('generated_at', 'n/a')}",
        f"- source_plan_path: {report.get('source_plan_path', 'n/a')}",
        f"- requested_files: {report.get('requested_files', 0)}",
        f"- downloaded_count: {report.get('downloaded_count', 0)}",
        f"- decompressed_count: {report.get('decompressed_count', 0)}",
        f"- skipped_count: {report.get('skipped_count', 0)}",
        f"- failed_count: {report.get('failed_count', 0)}",
        f"- paper_only: {report.get('paper_only')}",
        f"- execution_enabled: {report.get('execution_enabled')}",
        f"- requester_pays: {report.get('requester_pays', False)}",
        f"- aws_profile_used: {report.get('aws_profile_used')}",
        "",
        "## File results",
        "",
        *(file_lines or ["- (none)"]),
        "",
        "## Failures",
        "",
        *(failures or ["- (none)"]),
        "",
        "## Next import commands",
        "",
        *import_lines,
        "",
        "## Limitations",
        "",
        *warning_lines,
        *(
            ["- Uses authenticated AWS access with Requester Pays; charges may apply."]
            if report.get("requester_pays")
            else ["- Uses unsigned public S3 access only; no AWS access keys."]
        ),
        "- Hard cap: 3 files without --allow-more, 24 with --allow-more.",
        "- Does not support all-time or bulk archive downloads.",
        "",
        "## Research-only disclaimer",
        "",
        SAFETY_STATEMENT,
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    report["markdown_report_path"] = str(output_path)
    return output_path


def _write_probe_report_markdown(report: dict[str, Any], output_path: Path) -> Path:
    failures: list[str] = []
    for entry in report.get("results", []):
        if not entry.get("accessible"):
            failures.append(
                f"- probe failed ({entry.get('coin')} {entry.get('date')} hour {entry.get('hour')}): "
                f"{entry.get('error_class')} — {entry.get('error_message', 'unknown error')}"
            )

    result_lines: list[str] = []
    for entry in report.get("results", []):
        accessibility = "accessible" if entry.get("accessible") else "inaccessible"
        result_lines.append(
            f"- coin={entry.get('coin')} date={entry.get('date')} hour={entry.get('hour')}: "
            f"s3_path={entry.get('s3_path')}, "
            f"accessibility={accessibility}, error_class={entry.get('error_class')}, "
            f"bytes_read={entry.get('bytes_read', 0)}"
        )

    warning_lines = [f"- {w}" for w in report.get("warnings", [])]

    lines = [
        "# Historical L2 Archive Availability Probe Report",
        "",
        "## Summary",
        "",
        f"- generated_at: {report.get('generated_at', 'n/a')}",
        f"- source_plan_path: {report.get('source_plan_path', 'n/a')}",
        f"- requested_files: {report.get('requested_files', 0)}",
        f"- probed_count: {report.get('probed_count', 0)}",
        f"- accessible_count: {report.get('accessible_count', 0)}",
        f"- failed_count: {report.get('failed_count', 0)}",
        f"- paper_only: {report.get('paper_only')}",
        f"- execution_enabled: {report.get('execution_enabled')}",
        f"- requester_pays: {report.get('requester_pays', False)}",
        f"- aws_profile_used: {report.get('aws_profile_used')}",
        "",
        "## Probe results",
        "",
        *(result_lines or ["- (none)"]),
        "",
        "## Failures",
        "",
        *(failures or ["- (none)"]),
        "",
        "## Limitations",
        "",
        *warning_lines,
        *(
            ["- Uses authenticated AWS access with Requester Pays; charges may apply."]
            if report.get("requester_pays")
            else ["- Uses unsigned public S3 access only; no AWS access keys."]
        ),
        "- Range probe reads at most 1024 bytes per file; no full download.",
        "- Hard cap: 3 files without --allow-more, 24 with --allow-more.",
        "",
        "## Research-only disclaimer",
        "",
        SAFETY_STATEMENT,
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    report["markdown_report_path"] = str(output_path)
    return output_path


def _print_probe_result(entry: dict[str, Any]) -> None:
    accessibility = "accessible" if entry.get("accessible") else "inaccessible"
    print(
        f"Probed file: coin={entry.get('coin')} date={entry.get('date')} "
        f"hour={entry.get('hour')} s3_path={entry.get('s3_path')} "
        f"accessibility={accessibility} error_class={entry.get('error_class')}"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a tiny sample of Hyperliquid historical L2 archive files "
            "using Python-native unsigned S3 access and lz4 decompression."
        )
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--plan",
        help="Path to a command plan or backfill plan JSON file",
    )
    source.add_argument(
        "--latest",
        action="store_true",
        help="Use the newest plan under reports/historical_l2_download_commands/ or backfill plans",
    )
    source.add_argument(
        "--probe-path",
        help="Probe a single archive S3 path directly (no manifest required)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Probe S3 accessibility with a small Range request instead of downloading",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help=(
            f"Number of files to process "
            f"(download default: {DEFAULT_MAX_FILES}, probe default: {DEFAULT_PROBE_MAX_FILES}; "
            f"hard cap: {HARD_CAP_MAX_FILES})"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local .lz4 and .jsonl files",
    )
    parser.add_argument(
        "--allow-more",
        action="store_true",
        help=f"Raise hard cap to {HARD_CAP_MAX_FILES_ALLOW_MORE} files (still no all-time downloads)",
    )
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help=f"Report output directory (default: {DEFAULT_REPORT_DIR})",
    )
    parser.add_argument(
        "--coin",
        help="Select only plan files matching this coin (e.g. SOL)",
    )
    parser.add_argument(
        "--date",
        help="Select only plan files matching this date (YYYYMMDD, e.g. 20230916)",
    )
    parser.add_argument(
        "--hour",
        help="Select only plan files matching this hour (e.g. 9)",
    )
    parser.add_argument(
        "--requester-pays",
        action="store_true",
        help=(
            "Use authenticated AWS access with Requester Pays for S3 get_object "
            "(may incur AWS charges)"
        ),
    )
    parser.add_argument(
        "--aws-profile",
        metavar="PROFILE",
        help="AWS profile name for Requester Pays access (default: boto3 credential chain)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    probe_mode = args.probe or args.probe_path is not None
    requester_pays = args.requester_pays
    aws_profile = args.aws_profile

    if args.probe_path:
        if args.probe and (args.plan or args.latest):
            print("Error: --probe-path cannot be combined with --plan or --latest", file=sys.stderr)
            return 1
        try:
            report = run_probe_path(
                args.probe_path,
                requester_pays=requester_pays,
                aws_profile=aws_profile,
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    elif probe_mode:
        if not args.plan and not args.latest:
            print("Error: --probe requires --plan or --latest", file=sys.stderr)
            return 1
        if args.plan and args.latest:
            print("Error: --plan and --latest are mutually exclusive", file=sys.stderr)
            return 1

        try:
            max_files = resolve_probe_max_files(args.max_files, allow_more=args.allow_more)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        if args.latest:
            plan_path = discover_latest_command_plan()
        else:
            plan_path = Path(args.plan)

        try:
            plan = load_plan(plan_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        report = run_availability_probe(
            plan,
            max_files=max_files,
            coin=args.coin,
            date=args.date,
            hour=args.hour,
            requester_pays=requester_pays,
            aws_profile=aws_profile,
        )
    else:
        if not args.plan and not args.latest:
            print("Error: --plan or --latest is required", file=sys.stderr)
            return 1
        if args.plan and args.latest:
            print("Error: --plan and --latest are mutually exclusive", file=sys.stderr)
            return 1

        try:
            max_files = resolve_max_files(args.max_files, allow_more=args.allow_more)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        if args.latest:
            plan_path = discover_latest_command_plan()
        else:
            plan_path = Path(args.plan)

        try:
            plan = load_plan(plan_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        report = run_sample_download(
            plan,
            max_files=max_files,
            overwrite=args.overwrite,
            coin=args.coin,
            date=args.date,
            hour=args.hour,
            requester_pays=requester_pays,
            aws_profile=aws_profile,
        )

    json_path = write_report_json(report, output_dir=args.report_dir)
    markdown_path = write_report_markdown(report, output_dir=args.report_dir)

    if probe_mode:
        print("Historical L2 archive availability probe complete (research-only).")
        print(f"JSON report: {json_path}")
        print(f"Markdown report: {markdown_path}")
        print(f"Probed: {report['probed_count']}")
        print(f"Accessible: {report['accessible_count']}")
        print(f"Failed: {report['failed_count']}")
        for entry in report.get("results", []):
            _print_probe_result(entry)
        return 0 if report["failed_count"] == 0 else 1

    print("Historical L2 Python sample download complete (research-only).")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    print(f"Downloaded: {report['downloaded_count']}")
    print(f"Decompressed: {report['decompressed_count']}")
    print(f"Failed: {report['failed_count']}")
    for entry in report.get("file_results", []):
        print(
            f"Selected file: coin={entry.get('coin')} date={entry.get('date')} "
            f"hour={entry.get('hour')} s3_path={entry.get('s3_path')} "
            f"download_status={entry.get('download_status')} "
            f"decompress_status={entry.get('decompress_status')}"
        )
    if report["next_import_commands"]:
        print("Next import commands:")
        for cmd in report["next_import_commands"]:
            print(f"  {cmd}")
    return 0 if report["failed_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
