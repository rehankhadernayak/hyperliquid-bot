"""Shared data loading and sanitization for review package generation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import DATA, VERSION
import dashboard_data

REPO_ROOT = Path(__file__).resolve().parent
REVIEW_PACKAGES_DIR = REPO_ROOT / "review_packages"

SENSITIVE_PATTERNS = (
    r"\.env\b",
    r"private_key",
    r"seed_phrase",
    r"mnemonic",
    r"api_secret",
    r"password",
)

ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?:/[A-Za-z0-9._-]+)+|"
    r"(?:[A-Za-z]:\\(?:[^\\/\s\"']+\\)*[^\\/\s\"']*)"
)


@dataclass(frozen=True)
class GitInfo:
    commit_hash: str
    branch: str
    is_clean: bool


@dataclass(frozen=True)
class PytestRunSummary:
    passed: bool
    total: int
    failed: int
    output: str


# Backward-compatible alias for internal imports
TestSummary = PytestRunSummary


def sanitize_text(text: str) -> str:
    """Remove absolute paths and common home-directory usernames from exported text."""
    sanitized = text
    home = Path.home()
    if str(home) in sanitized:
        sanitized = sanitized.replace(str(home), "~")
    user = os.environ.get("USER") or os.environ.get("USERNAME") or ""
    if user and len(user) > 1:
        sanitized = re.sub(rf"/Users/{re.escape(user)}", "~", sanitized)
        sanitized = re.sub(rf"/home/{re.escape(user)}", "~", sanitized)
    sanitized = ABSOLUTE_PATH_PATTERN.sub("<path>", sanitized)
    return sanitized


def relative_path(path: Path, base: Path = REPO_ROOT) -> str:
    """Return a repo-relative path string for manifests."""
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.name


def hash_file(path: Path) -> str:
    """Return SHA-256 hex digest of file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_config_module() -> str:
    """Return SHA-256 of config.py contents."""
    return hash_file(REPO_ROOT / "config.py")


def get_git_info() -> GitInfo:
    """Return current git commit, branch, and working-tree cleanliness."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return GitInfo(commit_hash=commit, branch=branch, is_clean=not bool(status))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return GitInfo(commit_hash="unknown", branch="unknown", is_clean=False)


def get_python_version() -> str:
    return sys.version.split()[0]


def get_dependency_versions() -> dict[str, str]:
    """Return installed versions of declared dependencies."""
    deps = ("requests", "python-dotenv", "streamlit", "pandas", "pytest")
    versions: dict[str, str] = {}
    for name in deps:
        try:
            import importlib.metadata as metadata

            versions[name] = metadata.version(name)
        except Exception:
            versions[name] = "unknown"
    return versions


def run_test_summary(*, quick: bool = False) -> TestSummary:
    """Run pytest and return pass/fail summary."""
    args = [sys.executable, "-m", "pytest", "-q"]
    if quick:
        args.append("--collect-only")
    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    output = (result.stdout + result.stderr).strip()
    if quick:
        match = re.search(r"(\d+) tests? collected", output)
        total = int(match.group(1)) if match else 0
        return TestSummary(passed=True, total=total, failed=0, output=output)

    match = re.search(r"(\d+) passed", output)
    total = int(match.group(1)) if match else 0
    fail_match = re.search(r"(\d+) failed", output)
    failed = int(fail_match.group(1)) if fail_match else 0
    return TestSummary(
        passed=result.returncode == 0,
        total=total,
        failed=failed,
        output=output,
    )


def load_all_experiments() -> list[dict[str, Any]]:
    """Load all experiment JSON files sorted by created_at."""
    experiments: list[dict[str, Any]] = []
    if not DATA.experiments_dir.exists():
        return experiments

    for path in sorted(DATA.experiments_dir.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["_source_path"] = relative_path(path)
            experiments.append(payload)
        except (json.JSONDecodeError, OSError):
            continue

    experiments.sort(key=lambda item: item.get("created_at", ""))
    return experiments


def load_all_session_reports() -> list[dict[str, Any]]:
    """Load all session report JSON files."""
    reports: list[dict[str, Any]] = []
    if not DATA.session_reports_dir.exists():
        return reports

    for path in sorted(DATA.session_reports_dir.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["_source_path"] = relative_path(path)
            reports.append(payload)
        except (json.JSONDecodeError, OSError):
            continue
    return reports


def latest_experiment(experiments: list[dict[str, Any]]) -> dict[str, Any] | None:
    return experiments[-1] if experiments else None


def experiment_comparison_rows(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    """Build flat comparison rows for CSV export."""
    rows: list[dict[str, Any]] = []
    for profile_id, payload in experiment.get("profile_results", {}).items():
        dev = payload.get("development", {})
        oos = payload.get("out_of_sample", {})
        combined = payload.get("combined", {})
        rows.append(
            {
                "experiment_id": experiment.get("experiment_id"),
                "profile": profile_id,
                "strategy": payload.get("profile", {}).get("strategy_name"),
                "evidence_label": payload.get("evidence_assessment"),
                "dev_return_pct": dev.get("net_return_pct"),
                "oos_return_pct": oos.get("net_return_pct"),
                "combined_return_pct": combined.get("net_return_pct"),
                "dev_completed_trades": dev.get("completed_trades"),
                "oos_completed_trades": oos.get("completed_trades"),
                "combined_completed_trades": combined.get("completed_trades"),
                "oos_max_drawdown_pct": oos.get("maximum_drawdown_pct"),
                "oos_profit_factor": oos.get("profit_factor"),
                "oos_fees_paid": oos.get("fees_paid"),
                "oos_slippage_cost": oos.get("slippage_cost"),
                "overfitting_warning_count": len(payload.get("overfitting_warnings") or []),
            }
        )
    return rows


def session_summary_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build flat session summary rows for CSV export."""
    rows: list[dict[str, Any]] = []
    for report in reports:
        rows.append(
            {
                "session_id": report.get("session_id"),
                "mode": report.get("mode"),
                "starting_equity": report.get("starting_equity"),
                "ending_equity": report.get("ending_equity"),
                "total_return_pct": report.get("total_return_pct"),
                "realized_pnl": report.get("realized_pnl"),
                "fees_paid": report.get("fees_paid"),
                "completed_trades": report.get("completed_trades"),
                "win_rate_pct": report.get("win_rate_pct"),
                "profit_factor": report.get("profit_factor"),
                "maximum_drawdown_pct": report.get("maximum_drawdown_pct"),
            }
        )
    return rows


def count_paper_trades() -> int:
    trades = dashboard_data.load_paper_trades()
    if trades is None or trades.empty:
        return 0
    normalized = dashboard_data.normalize_paper_trades(trades)
    return int(normalized[normalized["action"] == "SELL"].shape[0])


def count_independent_sessions() -> int:
    trades = dashboard_data.load_paper_trades()
    if trades is None or trades.empty:
        return len(load_all_session_reports())
    normalized = dashboard_data.normalize_paper_trades(trades)
    sessions = set(normalized.get("session_id", pd.Series(dtype=str)).dropna().astype(str))
    sessions.update(report.get("session_id", "") for report in load_all_session_reports())
    sessions.discard("")
    return len(sessions)


def count_market_days() -> int:
    prices = dashboard_data.load_price_log()
    if prices.empty:
        return 0
    return int(prices["timestamp"].dt.date.nunique())


def count_market_days_from_experiment(experiment: dict[str, Any] | None) -> int:
    if experiment is None:
        return count_market_days()
    date_range = experiment.get("date_range", {})
    start = date_range.get("start", "")
    end = date_range.get("end", "")
    if not start or not end:
        return count_market_days()
    try:
        start_dt = datetime.strptime(start.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
        return max(1, (end_dt.date() - start_dt.date()).days + 1)
    except ValueError:
        return count_market_days()


def package_id_now() -> str:
    return datetime.now(timezone.utc).strftime("review-%Y%m%d-%H%M%S")


def contains_sensitive_content(text: str) -> bool:
    """Detect likely secrets, not documentation that mentions excluded file types."""
    lowered = text.lower()
    secret_markers = (
        "private_key=",
        "api_secret=",
        "seed_phrase=",
        "mnemonic=",
        "password=",
    )
    for marker in secret_markers:
        if marker in lowered:
            return True

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("-"):
            continue
        if re.search(r"(?:^|[/\s])\.env(?:\s|$|/)", stripped):
            return True
        if re.search(r"private_key|seed_phrase|api_secret", stripped, re.IGNORECASE):
            if "no " not in stripped.lower() and "forbidden" not in stripped.lower():
                return True
    return False


def contains_absolute_local_path(text: str) -> bool:
    home = str(Path.home())
    if home in text and home != "~":
        return True
    user = os.environ.get("USER") or ""
    if user and f"/Users/{user}" in text:
        return True
    if user and f"/home/{user}" in text:
        return True
    if re.search(r"/Users/[A-Za-z0-9._-]+/", text):
        return True
    return False
