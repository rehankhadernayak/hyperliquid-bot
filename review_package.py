#!/usr/bin/env python3
"""
Review package generator for supervised adviser review.

Creates a timestamped directory under review_packages/ with documentation,
CSVs, manifest, and checksums. Read-only — no network, no live data writes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from capital_readiness import assess_capital_readiness, format_assessment_report
from config import DATA, VERSION
import review_documents as docs
from review_data import (
    REVIEW_PACKAGES_DIR,
    contains_absolute_local_path,
    contains_sensitive_content,
    experiment_comparison_rows,
    get_dependency_versions,
    get_git_info,
    get_python_version,
    hash_config_module,
    hash_file,
    latest_experiment,
    load_all_experiments,
    load_all_session_reports,
    package_id_now,
    relative_path,
    run_test_summary,
    sanitize_text,
    session_summary_rows,
)

PACKAGE_FILES = (
    "executive_summary.md",
    "system_architecture.md",
    "strategy_documentation.md",
    "risk_management.md",
    "evidence_summary.md",
    "experiment_comparison.csv",
    "session_summary.csv",
    "known_limitations.md",
    "failure_scenarios.md",
    "advisor_questions.md",
    "review_checklist.md",
    "manifest.json",
    "checksums.sha256",
)


def generate_review_package(
    output_dir: Path | None = None,
    *,
    package_id: str | None = None,
    skip_tests: bool = False,
) -> Path:
    """
    Generate a complete review package and return its directory path.

    Does not modify live trading CSV files.
    """
    package_id = package_id or package_id_now()
    package_dir = output_dir or (REVIEW_PACKAGES_DIR / package_id)
    package_dir.mkdir(parents=True, exist_ok=True)

    test_summary = run_test_summary() if not skip_tests else run_test_summary(quick=True)
    experiments = load_all_experiments()
    sessions = load_all_session_reports()
    latest = latest_experiment(experiments)
    readiness = assess_capital_readiness(
        test_summary=test_summary,
        experiments=experiments,
    )

    file_contents: dict[str, str] = {}

    file_contents["executive_summary.md"] = sanitize_text(
        docs.generate_executive_summary(
            test_summary=test_summary,
            experiments=experiments,
            latest=latest,
            readiness_label=readiness.label,
            readiness_summary=readiness.summary,
        )
    )
    file_contents["system_architecture.md"] = docs.generate_system_architecture()
    file_contents["strategy_documentation.md"] = docs.generate_strategy_documentation()
    file_contents["risk_management.md"] = docs.generate_risk_management()
    file_contents["evidence_summary.md"] = docs.generate_evidence_summary(latest)
    file_contents["known_limitations.md"] = docs.generate_known_limitations()
    file_contents["failure_scenarios.md"] = docs.generate_failure_scenarios()
    file_contents["advisor_questions.md"] = docs.generate_advisor_questions()
    file_contents["review_checklist.md"] = docs.generate_review_checklist()

    if latest:
        comparison = experiment_comparison_rows(latest)
        file_contents["experiment_comparison.csv"] = _dataframe_to_csv(
            pd.DataFrame(comparison)
        )
    else:
        file_contents["experiment_comparison.csv"] = "experiment_id,profile,evidence_label\n"

    session_rows = session_summary_rows(sessions)
    if session_rows:
        file_contents["session_summary.csv"] = _dataframe_to_csv(pd.DataFrame(session_rows))
    else:
        file_contents["session_summary.csv"] = "session_id,mode,completed_trades\n"

    git = get_git_info()
    manifest = _build_manifest(
        package_id=package_id,
        package_dir=package_dir,
        test_summary=test_summary,
        experiments=experiments,
        sessions=sessions,
        git=git,
        readiness=readiness,
        files_to_include=list(file_contents.keys()),
    )
    file_contents["manifest.json"] = json.dumps(manifest, indent=2)

    for filename, content in file_contents.items():
        if filename == "manifest.json":
            continue
        _validate_export_content(filename, content)
        (package_dir / filename).write_text(content, encoding="utf-8")

    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(file_contents["manifest.json"], encoding="utf-8")
    _validate_export_content("manifest.json", file_contents["manifest.json"])

    checksums = _generate_checksums(package_dir, exclude={"checksums.sha256"})
    (package_dir / "checksums.sha256").write_text(checksums, encoding="utf-8")

    return package_dir


def _build_manifest(
    *,
    package_id: str,
    package_dir: Path,
    test_summary,
    experiments: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    git,
    readiness,
    files_to_include: list[str],
) -> dict[str, Any]:
    experiment_ids = [e.get("experiment_id", "") for e in experiments]
    experiment_hashes = {
        e.get("experiment_id", ""): e.get("input_data_hash", "")
        for e in experiments
    }
    session_ids = [s.get("session_id", "") for s in sessions]

    return {
        "package_id": package_id,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "version": VERSION,
        "git_commit": git.commit_hash,
        "git_branch": git.branch,
        "git_clean": git.is_clean,
        "python_version": get_python_version(),
        "dependency_versions": get_dependency_versions(),
        "configuration_hashes": {
            "config.py": hash_config_module(),
        },
        "experiment_ids": experiment_ids,
        "experiment_input_data_hashes": experiment_hashes,
        "session_ids": session_ids,
        "files_included": sorted(files_to_include) + ["manifest.json", "checksums.sha256"],
        "files_excluded": [
            "data/paper_trades.csv (raw history)",
            "data/price_log.csv (raw history)",
            "data/equity_log.csv (raw history)",
            ".env",
            "secrets",
        ],
        "test_result_summary": {
            "passed": test_summary.passed,
            "total": test_summary.total,
            "failed": test_summary.failed,
        },
        "health_check_summary": {
            "note": "Run python3 review_check.py for full verification",
        },
        "capital_readiness": readiness.label,
        "safety_statement": (
            "This release cannot place real trades and is intended only for "
            "supervised technical and risk review."
        ),
    }


def _dataframe_to_csv(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def _generate_checksums(package_dir: Path, *, exclude: set[str]) -> str:
    lines: list[str] = []
    for path in sorted(package_dir.iterdir()):
        if not path.is_file() or path.name in exclude:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    return "\n".join(lines) + "\n"


def _validate_export_content(filename: str, content: str) -> None:
    if contains_sensitive_content(content):
        raise RuntimeError(f"Sensitive content detected in {filename}")
    if contains_absolute_local_path(content):
        raise RuntimeError(f"Absolute local path detected in {filename}")


def verify_package_checksums(package_dir: Path) -> bool:
    """Verify checksums.sha256 against package files."""
    checksums_path = package_dir / "checksums.sha256"
    if not checksums_path.exists():
        return False

    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected_hash, filename = line.split("  ", 1)
        file_path = package_dir / filename
        if not file_path.exists():
            return False
        actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if actual != expected_hash:
            return False
    return True


def verify_manifest(manifest_path: Path) -> bool:
    """Validate manifest.json structure."""
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return False

    required = (
        "package_id",
        "generated_at",
        "version",
        "git_commit",
        "files_included",
        "test_result_summary",
        "safety_statement",
    )
    return all(key in manifest for key in required)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a supervised review package (read-only, paper trading only)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: review_packages/<package_id>/)",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running pytest during package generation (faster, for dev only)",
    )
    args = parser.parse_args()

    print(f"Generating review package — {VERSION}")
    print("Read-only paper system — no wallet, signing, or real orders.")
    print("-" * 60)

    try:
        package_dir = generate_review_package(
            output_dir=args.output_dir,
            skip_tests=args.skip_tests,
        )
    except Exception as exc:
        print(f"Package generation failed: {exc}")
        return 1

    print(f"Package created: {relative_path(package_dir)}")
    print("")
    print("Contents:")
    for name in sorted(PACKAGE_FILES):
        path = package_dir / name
        if path.exists():
            print(f"  {name} ({path.stat().st_size:,} bytes)")

    print("")
    print("Verify with: python3 review_check.py")
    print(
        "Historical and paper results do not guarantee future performance."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
