"""Shared pytest fixtures for offline, read-only tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import dashboard_data
import equity_logger
import paper_trader
import session_report
from config import DATA, DataPaths
from signals import apply_signal_thresholds

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_PAPER_TRADES = REPO_ROOT / "data" / "paper_trades.csv"
LIVE_PRICE_LOG = REPO_ROOT / "data" / "price_log.csv"
LIVE_EQUITY_LOG = REPO_ROOT / "data" / "equity_log.csv"


@pytest.fixture(autouse=True)
def reset_signal_thresholds_after_test():
    """Restore production signal thresholds so tests do not leak state."""
    yield
    apply_signal_thresholds(test_mode=False)


@pytest.fixture(autouse=True)
def protect_live_data_files():
    """Fail if tests modify live CSV data files."""
    snapshots: dict[Path, bytes] = {}

    for path in (LIVE_PAPER_TRADES, LIVE_PRICE_LOG, LIVE_EQUITY_LOG):
        if path.exists():
            snapshots[path] = path.read_bytes()

    yield

    for path, original_content in snapshots.items():
        if path.exists():
            assert path.read_bytes() == original_content, (
                f"Live data file was modified during tests: {path}"
            )


@pytest.fixture
def temp_data_paths(tmp_path, monkeypatch):
    """Redirect paper trader and equity CSV output to a temporary directory."""
    data_dir = tmp_path / "data"
    paths = DataPaths(
        data_dir=data_dir,
        price_log_csv=data_dir / "price_log.csv",
        paper_trades_csv=data_dir / "paper_trades.csv",
        backtest_results_json=data_dir / "backtest_results.json",
        recommendations_json=data_dir / "recommendations.json",
        equity_log_csv=data_dir / "equity_log.csv",
        session_configs_dir=data_dir / "session_configs",
        session_reports_dir=data_dir / "session_reports",
        experiments_dir=data_dir / "experiments",
    )
    monkeypatch.setattr(paper_trader, "DATA", paths)
    monkeypatch.setattr(equity_logger, "DATA", paths)
    monkeypatch.setattr(dashboard_data, "DATA", paths)
    monkeypatch.setattr(session_report, "DATA", paths)
    return paths


@pytest.fixture
def silent_trade_output(monkeypatch):
    """Suppress paper trade print output during tests."""
    monkeypatch.setattr(paper_trader, "print_paper_trade_proposal", lambda **_kwargs: None)
    monkeypatch.setattr(paper_trader, "print_skipped_trade_proposal", lambda **_kwargs: None)
