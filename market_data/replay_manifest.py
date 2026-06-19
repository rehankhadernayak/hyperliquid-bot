"""Load recorder session manifests and classify output files for quality inspection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ManifestFileEntry:
    path: Path
    exists: bool
    size_bytes: int = 0
    category: str = "unknown"
    readable: bool = False
    row_count: int | None = None
    error: str | None = None


@dataclass
class SessionManifest:
    session_id: str
    manifest_path: Path | None
    manifest_data: dict[str, Any] | None
    raw_files: list[ManifestFileEntry] = field(default_factory=list)
    normalized_files: list[ManifestFileEntry] = field(default_factory=list)
    other_files: list[ManifestFileEntry] = field(default_factory=list)
    load_error: str | None = None

    @property
    def exists(self) -> bool:
        return self.manifest_data is not None

    @property
    def coins(self) -> list[str]:
        if not self.manifest_data:
            return []
        return list(self.manifest_data.get("coins") or [])

    @property
    def start_time_ms(self) -> int | None:
        if not self.manifest_data:
            return None
        return self.manifest_data.get("start_time_ms")

    @property
    def end_time_ms(self) -> int | None:
        if not self.manifest_data:
            return None
        return self.manifest_data.get("end_time_ms")

    @property
    def gap_statistics(self) -> dict[str, Any]:
        if not self.manifest_data:
            return {}
        return dict(self.manifest_data.get("gap_statistics") or {})

    @property
    def raw_row_count_manifest(self) -> int:
        if not self.manifest_data:
            return 0
        return int(self.manifest_data.get("raw_row_count") or 0)

    @property
    def normalized_row_count_manifest(self) -> int:
        if not self.manifest_data:
            return 0
        return int(self.manifest_data.get("normalized_row_count") or 0)


def _classify_path(path_str: str) -> str:
    normalized = path_str.replace("\\", "/")
    if "/raw/" in normalized or normalized.endswith(".jsonl"):
        return "raw"
    if "/normalized/" in normalized or normalized.endswith(".parquet"):
        return "normalized"
    return "other"


def _resolve_path(path_str: str, repo_root: Path | None) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if repo_root is not None:
        return repo_root / path
    return path


def load_session_manifest(
    session_id: str,
    market_stream_dir: Path,
    repo_root: Path | None = None,
) -> SessionManifest:
    """Load manifest and inspect listed files without modifying them."""
    manifest_path = market_stream_dir / "manifests" / f"{session_id}.json"
    result = SessionManifest(
        session_id=session_id,
        manifest_path=manifest_path if manifest_path.exists() else None,
        manifest_data=None,
    )

    if not manifest_path.exists():
        result.load_error = f"Manifest not found: {manifest_path}"
        return result

    try:
        with manifest_path.open(encoding="utf-8") as handle:
            result.manifest_data = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        result.load_error = f"Manifest unreadable: {exc}"
        return result

    files_written = result.manifest_data.get("files_written") or []
    for path_str in files_written:
        category = _classify_path(str(path_str))
        resolved = _resolve_path(str(path_str), repo_root)
        entry = _inspect_file(resolved, category)
        if category == "raw":
            result.raw_files.append(entry)
        elif category == "normalized":
            result.normalized_files.append(entry)
        else:
            result.other_files.append(entry)

    return result


def _inspect_file(path: Path, category: str) -> ManifestFileEntry:
    entry = ManifestFileEntry(path=path, exists=path.exists(), category=category)
    if not entry.exists:
        entry.error = "file missing"
        return entry

    try:
        entry.size_bytes = path.stat().st_size
    except OSError as exc:
        entry.error = f"stat failed: {exc}"
        return entry

    if category == "normalized" and path.suffix == ".parquet":
        entry.row_count = _parquet_row_count(path)
        entry.readable = entry.row_count is not None
        if not entry.readable:
            entry.error = "parquet unreadable"
    elif category == "raw":
        entry.readable = path.is_file() and path.stat().st_size >= 0
    else:
        entry.readable = True

    return entry


def _parquet_row_count(path: Path) -> int | None:
    try:
        import pyarrow.parquet as pq

        metadata = pq.read_metadata(path)
        return metadata.num_rows
    except Exception:
        return None


def find_latest_session_id(market_stream_dir: Path) -> str | None:
    """Return session id from the newest manifest by mtime."""
    manifests_dir = market_stream_dir / "manifests"
    if not manifests_dir.exists():
        return None

    manifests = sorted(
        manifests_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not manifests:
        return None
    return manifests[0].stem


def list_quality_reports(quality_reports_dir: Path) -> list[dict[str, Any]]:
    """List available quality report JSON files sorted newest first."""
    if not quality_reports_dir.exists():
        return []

    reports: list[dict[str, Any]] = []
    for path in sorted(quality_reports_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            reports.append(
                {
                    "session_id": data.get("recorder_session_id", path.stem),
                    "report_id": data.get("report_id"),
                    "generated_at": data.get("generated_at"),
                    "quality_label": data.get("quality_label"),
                    "json_path": str(path),
                    "markdown_path": str(path.with_suffix(".md")),
                }
            )
        except (json.JSONDecodeError, OSError):
            continue
    return reports
