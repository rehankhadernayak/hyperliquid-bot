"""Integrity checks and quarantine helpers for local Hyperliquid L2 archive files.

Research-only; does not download or place orders.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lz4.frame

from historical_l2_importer import parse_l2_snapshot_line

DEFAULT_QUARANTINE_ROOT = Path("downloads/hyperliquid/quarantine")

_VALID_INT_TOKEN = re.compile(r"^-?\d+$")


@dataclass
class ArchiveIntegrityReport:
    ok: bool
    lz4_path: str | None = None
    jsonl_path: str | None = None
    reasons: list[str] = field(default_factory=list)
    lines_scanned: int = 0
    malformed_line_count: int = 0
    null_byte_count: int = 0
    lz4_decompress_ok: bool | None = None
    jsonl_matches_lz4: bool | None = None
    truncated_tail: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArchiveQuarantineReport:
    quarantined: bool
    quarantine_dir: str | None = None
    moved_paths: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _count_null_bytes(data: bytes) -> int:
    return data.count(b"\x00")


def _last_non_empty_line_is_valid_jsonl(data: bytes) -> bool:
    text = data.decode("utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    last = lines[-1].strip()
    if not last.startswith("{"):
        return False
    try:
        json.loads(last)
    except json.JSONDecodeError:
        return False
    return True


def inspect_jsonl_bytes(
    data: bytes,
    *,
    max_lines_to_scan: int = 400,
) -> ArchiveIntegrityReport:
    """Scan decompressed JSONL bytes for corruption signatures."""
    reasons: list[str] = []
    null_byte_count = _count_null_bytes(data)
    if null_byte_count:
        reasons.append(f"jsonl contains {null_byte_count} null byte(s)")

    if not _last_non_empty_line_is_valid_jsonl(data):
        reasons.append("jsonl tail appears truncated or incomplete")
        truncated_tail = True
    else:
        truncated_tail = False

    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    malformed = 0
    scanned = 0
    for line in lines:
        if not line.strip():
            continue
        scanned += 1
        if scanned > max_lines_to_scan:
            break
        if "\x00" in line:
            malformed += 1
            continue
        snapshot = parse_l2_snapshot_line(line)
        if snapshot is None:
            malformed += 1

    if scanned == 0:
        reasons.append("jsonl contains no non-empty lines")

    report = ArchiveIntegrityReport(
        ok=not reasons and malformed == 0 and scanned > 0,
        jsonl_path=None,
        reasons=reasons,
        lines_scanned=scanned,
        malformed_line_count=malformed,
        null_byte_count=null_byte_count,
        truncated_tail=truncated_tail,
    )
    if malformed:
        report.reasons.append(f"{malformed} malformed line(s) in first {scanned} non-empty line(s)")
        report.ok = False
    return report


def inspect_local_archive_pair(
    *,
    lz4_path: Path | str | None,
    jsonl_path: Path | str | None,
    max_lines_to_scan: int = 400,
) -> ArchiveIntegrityReport:
    """Inspect local .lz4 and/or .jsonl archive artifacts."""
    lz4_file = Path(lz4_path) if lz4_path else None
    jsonl_file = Path(jsonl_path) if jsonl_path else None
    reasons: list[str] = []
    lz4_decompress_ok: bool | None = None
    jsonl_matches_lz4: bool | None = None
    jsonl_report: ArchiveIntegrityReport | None = None
    decompressed: bytes | None = None

    if lz4_file and lz4_file.is_file():
        try:
            decompressed = lz4.frame.decompress(lz4_file.read_bytes())
            lz4_decompress_ok = True
        except Exception as exc:  # noqa: BLE001
            lz4_decompress_ok = False
            reasons.append(f"lz4 decompress failed: {exc}")

    if jsonl_file and jsonl_file.is_file():
        jsonl_bytes = jsonl_file.read_bytes()
        jsonl_report = inspect_jsonl_bytes(jsonl_bytes, max_lines_to_scan=max_lines_to_scan)
        if not jsonl_report.ok:
            reasons.extend(jsonl_report.reasons)
        if decompressed is not None:
            jsonl_matches_lz4 = jsonl_bytes == decompressed
            if not jsonl_matches_lz4:
                reasons.append("jsonl content does not match decompressed lz4 payload")
    elif jsonl_file:
        reasons.append(f"jsonl file missing: {jsonl_file}")

    if decompressed is not None and (not jsonl_file or not jsonl_file.is_file()):
        inner = inspect_jsonl_bytes(decompressed, max_lines_to_scan=max_lines_to_scan)
        if not inner.ok:
            reasons.extend(inner.reasons)
        jsonl_report = inner

    ok = not reasons
    if jsonl_report and jsonl_report.malformed_line_count and ok:
        ok = False

    return ArchiveIntegrityReport(
        ok=ok,
        lz4_path=str(lz4_file) if lz4_file else None,
        jsonl_path=str(jsonl_file) if jsonl_file else None,
        reasons=reasons,
        lines_scanned=jsonl_report.lines_scanned if jsonl_report else 0,
        malformed_line_count=jsonl_report.malformed_line_count if jsonl_report else 0,
        null_byte_count=jsonl_report.null_byte_count if jsonl_report else 0,
        lz4_decompress_ok=lz4_decompress_ok,
        jsonl_matches_lz4=jsonl_matches_lz4,
        truncated_tail=jsonl_report.truncated_tail if jsonl_report else False,
    )


def quarantine_local_archive_files(
    *,
    lz4_path: Path | str | None,
    jsonl_path: Path | str | None,
    reasons: list[str],
    quarantine_root: Path | str = DEFAULT_QUARANTINE_ROOT,
    coin: str | None = None,
    date: str | None = None,
    hour: str | None = None,
    paths_to_move: list[Path | str] | None = None,
) -> ArchiveQuarantineReport:
    """Move corrupt archive artifacts into a timestamped quarantine directory."""
    if not reasons:
        return ArchiveQuarantineReport(quarantined=False, reasons=[])

    stamp = _utc_stamp()
    label_parts = [part for part in (date, hour, coin) if part]
    label = "_".join(label_parts) if label_parts else "archive"
    target_dir = Path(quarantine_root) / f"{stamp}_{label}"
    target_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[Path]
    if paths_to_move:
        candidates = [Path(path) for path in paths_to_move]
    else:
        candidates = [Path(path) for path in (lz4_path, jsonl_path) if path]

    moved: list[str] = []
    for source in candidates:
        if not source.is_file():
            continue
        destination = target_dir / source.name
        source.replace(destination)
        moved.append(str(destination.resolve()))

    return ArchiveQuarantineReport(
        quarantined=bool(moved),
        quarantine_dir=str(target_dir.resolve()) if moved else None,
        moved_paths=moved,
        reasons=list(reasons),
    )


def ensure_local_archive_integrity(
    file_entry: dict[str, Any],
    *,
    quarantine_root: Path | str = DEFAULT_QUARANTINE_ROOT,
) -> dict[str, Any]:
    """Inspect archive files; quarantine and mark missing when corrupt."""
    lz4_path = file_entry.get("local_lz4_path")
    jsonl_path = file_entry.get("local_jsonl_path")
    integrity = inspect_local_archive_pair(lz4_path=lz4_path, jsonl_path=jsonl_path)
    result: dict[str, Any] = {
        "integrity_ok": integrity.ok,
        "integrity": integrity.to_dict(),
        "quarantine": None,
    }
    if integrity.ok:
        return result
    paths_to_move: list[Path] = []
    lz4_file = Path(lz4_path) if lz4_path else None
    jsonl_file = Path(jsonl_path) if jsonl_path else None
    if integrity.lz4_decompress_ok is False and lz4_file and lz4_file.is_file():
        paths_to_move.append(lz4_file)
        if jsonl_file and jsonl_file.is_file():
            paths_to_move.append(jsonl_file)
    elif jsonl_file and jsonl_file.is_file() and (
        integrity.null_byte_count
        or integrity.truncated_tail
        or integrity.jsonl_matches_lz4 is False
    ):
        paths_to_move.append(jsonl_file)
    else:
        for candidate in (lz4_file, jsonl_file):
            if candidate and candidate.is_file():
                paths_to_move.append(candidate)

    quarantine = quarantine_local_archive_files(
        lz4_path=lz4_path,
        jsonl_path=jsonl_path,
        reasons=integrity.reasons,
        quarantine_root=quarantine_root,
        coin=str(file_entry.get("coin") or ""),
        date=str(file_entry.get("date") or ""),
        hour=str(file_entry.get("hour") or ""),
        paths_to_move=paths_to_move,
    )
    result["quarantine"] = quarantine.to_dict()
    return result
