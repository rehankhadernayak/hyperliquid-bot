#!/usr/bin/env python3
"""
Release health check for the read-only paper trading bot.

Verifies data directories, configuration, imports, test suite, and the
absence of wallet or exchange execution modules.
"""

from __future__ import annotations

import ast
import subprocess
import sys
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


def check_data_directories() -> None:
    from config import DATA

    DATA.data_dir.mkdir(exist_ok=True)
    if not DATA.data_dir.is_dir():
        raise RuntimeError(f"Data directory missing: {DATA.data_dir}")


def check_config_loads() -> None:
    from config import (
        DATA,
        DEFAULT_BACKTEST,
        MARKET,
        PAPER_TRADING,
        SIGNAL,
        TEST_MODE,
        VERSION,
    )

    if VERSION != "8.2.0-paper":
        raise RuntimeError(f"Unexpected VERSION: {VERSION!r}")

    _ = (DATA, DEFAULT_BACKTEST, MARKET, PAPER_TRADING, SIGNAL, TEST_MODE)


def check_dashboard_imports() -> None:
    import dashboard  # noqa: F401


def check_backtester_imports() -> None:
    import backtester  # noqa: F401


def check_experiment_imports() -> None:
    import experiment_runner  # noqa: F401
    import strategy_profiles  # noqa: F401


def check_no_execution_modules() -> None:
    python_files = [
        path
        for path in REPO_ROOT.rglob("*.py")
        if ".git" not in path.parts and ".pytest_cache" not in path.parts
    ]

    for path in python_files:
        if path.name == "health_check.py":
            continue

        stem = path.stem
        if stem in FORBIDDEN_MODULE_NAMES:
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


def main() -> int:
    checks: list[tuple[str, callable]] = [
        ("Data directories", check_data_directories),
        ("Configuration", check_config_loads),
        ("Dashboard import", check_dashboard_imports),
        ("Backtester import", check_backtester_imports),
        ("Experiment framework import", check_experiment_imports),
        ("No wallet/execution modules", check_no_execution_modules),
        ("Test suite", check_tests_pass),
    ]

    print(f"Health check — Hyperliquid paper bot ({REPO_ROOT.name})")
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
        print("Health check FAILED.")
        return 1

    print("Health check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
