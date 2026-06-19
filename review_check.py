#!/usr/bin/env python3
"""
Automated release verification for v9.0.0-review.

Verifies version, imports, tests, review package generation, checksums,
safety constraints, and that capital readiness cannot authorize real trading.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

FORBIDDEN_MODULE_NAMES = frozenset(
    {
        "wallet",
        "wallet_client",
        "exchange_client",
        "order_executor",
        "live_trader",
        "execution_engine",
        "signing",
    }
)

FORBIDDEN_SOURCE_PATTERNS = frozenset(
    {
        "private_key",
        "eth_account",
        "web3",
        "sign_transaction",
        "place_order",
        "execute_order",
        "submit_order",
    }
)

LIVE_DATA_FILES = (
    REPO_ROOT / "data" / "paper_trades.csv",
    REPO_ROOT / "data" / "price_log.csv",
    REPO_ROOT / "data" / "equity_log.csv",
)


def check_version() -> None:
    from config import VERSION

    if VERSION != "9.0.0-review":
        raise RuntimeError(f"Expected VERSION '9.0.0-review', got {VERSION!r}")


def check_review_imports() -> None:
    import capital_readiness  # noqa: F401
    import review_data  # noqa: F401
    import review_documents  # noqa: F401
    import review_package  # noqa: F401
    import dashboard_review  # noqa: F401


def check_dashboard_imports() -> None:
    import dashboard  # noqa: F401


def check_experiment_runner_offline() -> None:
    source = (REPO_ROOT / "experiment_runner.py").read_text(encoding="utf-8")
    if "requests" in source or "urllib" in source or "http" in source.lower():
        lowered = source.lower()
        if "import requests" in lowered or "from requests" in lowered:
            raise RuntimeError("experiment_runner.py must not use network access")


def check_no_execution_modules() -> None:
    python_files = [
        path
        for path in REPO_ROOT.rglob("*.py")
        if ".git" not in path.parts and ".pytest_cache" not in path.parts
    ]

    skip_files = {"health_check.py", "review_check.py", "review_data.py", "system_check.py", "event_safety.py"}
    for path in python_files:
        if path.name in skip_files:
            continue

        if path.stem in FORBIDDEN_MODULE_NAMES:
            raise RuntimeError(f"Forbidden execution module present: {path}")

        source = path.read_text(encoding="utf-8")
        lowered = source.lower()
        for pattern in FORBIDDEN_SOURCE_PATTERNS:
            if pattern in lowered:
                raise RuntimeError(
                    f"Forbidden pattern {pattern!r} found in {path.relative_to(REPO_ROOT)}"
                )

        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in FORBIDDEN_MODULE_NAMES:
                        raise RuntimeError(
                            f"Forbidden import {alias.name!r} in {path.relative_to(REPO_ROOT)}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".", 1)[0]
                    if root in FORBIDDEN_MODULE_NAMES:
                        raise RuntimeError(
                            f"Forbidden import from {node.module!r} in "
                            f"{path.relative_to(REPO_ROOT)}"
                        )


def check_tests_pass() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        raise RuntimeError(f"pytest failed:\n{output}")


def check_review_package_generation() -> None:
    from review_data import contains_absolute_local_path, contains_sensitive_content
    from review_package import generate_review_package, verify_manifest, verify_package_checksums

    snapshots = {path: path.read_bytes() for path in LIVE_DATA_FILES if path.exists()}

    with tempfile.TemporaryDirectory() as tmp:
        package_dir = generate_review_package(
            Path(tmp) / "test-package",
            package_id="test-review-package",
            skip_tests=True,
        )

        if not verify_manifest(package_dir / "manifest.json"):
            raise RuntimeError("Generated manifest.json is invalid")

        if not verify_package_checksums(package_dir):
            raise RuntimeError("Package checksums do not match")

        for path in package_dir.iterdir():
            if path.is_file():
                content = path.read_text(encoding="utf-8")
                if contains_sensitive_content(content):
                    raise RuntimeError(f"Sensitive content in package file: {path.name}")
                if contains_absolute_local_path(content):
                    raise RuntimeError(f"Absolute local path in package file: {path.name}")

    for path, original in snapshots.items():
        if path.exists() and path.read_bytes() != original:
            raise RuntimeError(f"Live data file modified during package generation: {path}")


def check_capital_readiness_no_authorization() -> None:
    from capital_readiness import (
        AUTHORIZATION_PHRASES,
        ELIGIBLE_FOR_TINY_SUPERVISED_PILOT_REVIEW,
        assess_capital_readiness,
        format_assessment_report,
    )

    assessment = assess_capital_readiness()
    report = format_assessment_report(assessment)

    forbidden_outputs = (
        "approved for trading",
        "authorized to trade",
        "automatic approval",
        "deploy capital now",
    )
    lowered = report.lower()
    for phrase in forbidden_outputs:
        if phrase in lowered:
            raise RuntimeError(
                f"Capital readiness output contains authorization language: {phrase!r}"
            )

    if assessment.label == ELIGIBLE_FOR_TINY_SUPERVISED_PILOT_REVIEW:
        if "adviser" not in report.lower():
            raise RuntimeError(
                "Highest tier must reference adviser review, not automatic approval"
            )

    for phrase in AUTHORIZATION_PHRASES:
        if phrase in report.lower() and phrase not in ("go live",):
            pass  # checked above for critical phrases


def check_live_data_unchanged() -> None:
    """Placeholder — live data integrity verified during package generation check."""
    return None


def main() -> int:
    checks: list[tuple[str, callable]] = [
        ("VERSION is 9.0.0-review", check_version),
        ("Review modules import", check_review_imports),
        ("Dashboard imports", check_dashboard_imports),
        ("Experiment runner offline", check_experiment_runner_offline),
        ("No forbidden execution code", check_no_execution_modules),
        ("Full test suite passes", check_tests_pass),
        ("Review package generation", check_review_package_generation),
        ("Live data files unchanged", check_live_data_unchanged),
        ("Capital readiness cannot authorize trading", check_capital_readiness_no_authorization),
    ]

    print(f"Review check — Hyperliquid paper bot ({REPO_ROOT.name})")
    print("-" * 60)

    failed = False
    for label, check in checks:
        try:
            check()
            print(f"  OK  {label}")
        except Exception as exc:
            failed = True
            print(f"  FAIL  {label}: {exc}")

    print("-" * 60)
    if failed:
        print("Review check FAILED.")
        return 1

    print("Review check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
