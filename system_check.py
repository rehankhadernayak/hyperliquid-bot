#!/usr/bin/env python3
"""
Framework-level systems check — read-only validations, no network.

Runs health/review checks, module imports, report readability, safety scans,
and offline test suite.
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

SCAN_PACKAGES = ("market_data", "event_backtester", "feature_research")


def _run_script(script: str) -> None:
    result = subprocess.run(
        [sys.executable, script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        raise RuntimeError(f"{script} failed:\n{output}")


def check_health_check() -> None:
    _run_script("health_check.py")


def check_review_check() -> None:
    _run_script("review_check.py")


def check_market_data_imports() -> None:
    import market_data.replay_engine  # noqa: F401
    import market_data.quality_report  # noqa: F401


def check_event_backtester_imports() -> None:
    import event_backtester  # noqa: F401
    import event_backtester.event_backtest_engine  # noqa: F401


def check_feature_research_imports() -> None:
    import feature_research  # noqa: F401
    import feature_research.feature_engine  # noqa: F401


def check_latest_recorder_manifest() -> None:
    from dashboard_data import load_recorder_manifests

    manifests = load_recorder_manifests()
    if not manifests:
        return
    latest = manifests[0]
    if not latest.get("recorder_session_id"):
        raise RuntimeError("Latest recorder manifest missing session id")


def check_latest_quality_report() -> None:
    from dashboard_data import load_quality_reports, load_quality_report

    reports = load_quality_reports()
    if not reports:
        return
    session_id = reports[0].get("recorder_session_id") or reports[0].get("session_id")
    if session_id and load_quality_report(session_id) is None:
        raise RuntimeError(f"Quality report not readable for {session_id}")


def check_latest_replay_report() -> None:
    from dashboard_data import load_replay_report, load_replay_reports

    reports = load_replay_reports()
    if not reports:
        return
    replay_id = reports[0]["replay_session_id"]
    if load_replay_report(replay_id) is None:
        raise RuntimeError(f"Replay report not readable for {replay_id}")


def check_no_execution_flags_in_reports() -> None:
    from dashboard_data import load_event_backtest_reports, load_feature_research_reports, load_replay_reports, load_quality_reports

    for loader, id_key in (
        (load_quality_reports, "recorder_session_id"),
        (load_replay_reports, "replay_session_id"),
        (load_event_backtest_reports, "event_backtest_id"),
        (load_feature_research_reports, "feature_research_id"),
    ):
        for meta in loader():
            report_id = meta.get(id_key) or meta.get("session_id")
            if meta.get("execution_enabled") is True:
                raise RuntimeError(f"execution_enabled=True in {report_id}")


def check_no_wallet_signing_in_packages() -> None:
    for package in SCAN_PACKAGES:
        pkg_dir = REPO_ROOT / package.replace(".", "/")
        if not pkg_dir.exists():
            continue
        for path in pkg_dir.rglob("*.py"):
            if path.name in {"event_safety.py", "feature_safety.py"}:
                continue
            stem = path.stem
            if stem in FORBIDDEN_MODULE_NAMES:
                raise RuntimeError(f"Forbidden module: {path}")
            source = path.read_text(encoding="utf-8").lower()
            for pattern in FORBIDDEN_SOURCE_PATTERNS:
                if pattern in source:
                    raise RuntimeError(f"Forbidden pattern {pattern!r} in {path}")
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".", 1)[0] in FORBIDDEN_MODULE_NAMES:
                            raise RuntimeError(f"Forbidden import in {path}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.split(".", 1)[0] in FORBIDDEN_MODULE_NAMES:
                        raise RuntimeError(f"Forbidden import in {path}")


def check_tests_pass() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pytest failed:\n{result.stdout + result.stderr}")


def check_dashboard_loaders() -> None:
    import dashboard_data

    dashboard_data.load_price_log()
    dashboard_data.load_paper_trades()
    dashboard_data.load_equity_log()
    dashboard_data.load_backtest_results()
    dashboard_data.load_quality_reports()
    dashboard_data.load_replay_reports()
    dashboard_data.load_event_backtest_reports()
    dashboard_data.load_feature_research_reports()
    dashboard_data.load_recorder_manifests()
    dashboard_data.load_recorder_status()


def main() -> int:
    checks: list[tuple[str, callable]] = [
        ("health_check.py", check_health_check),
        ("review_check.py", check_review_check),
        ("market_data imports", check_market_data_imports),
        ("event_backtester imports", check_event_backtester_imports),
        ("feature_research imports", check_feature_research_imports),
        ("latest recorder manifest", check_latest_recorder_manifest),
        ("latest quality report", check_latest_quality_report),
        ("latest replay report", check_latest_replay_report),
        ("no execution_enabled in reports", check_no_execution_flags_in_reports),
        ("no wallet/signing in packages", check_no_wallet_signing_in_packages),
        ("pytest suite", check_tests_pass),
        ("dashboard data loaders", check_dashboard_loaders),
    ]

    print(f"System check — Hyperliquid paper bot ({REPO_ROOT.name})")
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
        print("System check FAILED.")
        return 1

    print("System check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
