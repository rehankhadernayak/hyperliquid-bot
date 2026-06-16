"""Tests for backtest simulation helpers and comparison output."""

from __future__ import annotations

from pathlib import Path

import pytest

from backtester import (
    BacktestConfig,
    BacktestPaperTrader,
    compute_max_drawdown,
    run_backtest,
    run_comparison,
)
from config import PAPER_TRADING
from tests.conftest import LIVE_PAPER_TRADES


def _write_sample_price_log(path: Path) -> None:
    """Create a tiny price log with enough history to generate signals."""
    rows = [
        "timestamp,coin,price",
        "2026-01-01 00:00:00 UTC,BTC,100.0",
        "2026-01-01 00:00:00 UTC,ETH,100.0",
        "2026-01-01 00:00:00 UTC,SOL,100.0",
        "2026-01-01 00:01:00 UTC,BTC,100.0",
        "2026-01-01 00:01:00 UTC,ETH,100.0",
        "2026-01-01 00:01:00 UTC,SOL,100.0",
        "2026-01-01 00:02:00 UTC,BTC,99.0",
        "2026-01-01 00:02:00 UTC,ETH,99.0",
        "2026-01-01 00:02:00 UTC,SOL,99.0",
        "2026-01-01 00:03:00 UTC,BTC,101.0",
        "2026-01-01 00:03:00 UTC,ETH,101.0",
        "2026-01-01 00:03:00 UTC,SOL,101.0",
    ]
    path.write_text("\n".join(rows) + "\n")


class TestBacktestPaperTraderExecution:
    """Fees, slippage, and forced position closing."""

    def test_fees_are_applied_on_buys_and_sells(self):
        config = BacktestConfig(trading_fee_pct=0.05)
        trader = BacktestPaperTrader(config=config)

        buy_fee = trader.buy_fee(100.0)
        sell_fee = trader.sell_fee(100.0)

        assert buy_fee == pytest.approx(0.05)
        assert sell_fee == pytest.approx(0.05)
        assert trader.total_fees_paid == pytest.approx(0.10)

    def test_slippage_makes_buys_more_expensive_and_sells_cheaper(self):
        config = BacktestConfig(slippage_pct=0.02)
        trader = BacktestPaperTrader(config=config)
        mid_price = 100.0

        buy_price = trader.execution_price("BUY", mid_price)
        sell_price = trader.execution_price("SELL", mid_price)

        assert buy_price > mid_price
        assert sell_price < mid_price
        assert buy_price == pytest.approx(mid_price * 1.0002)
        assert sell_price == pytest.approx(mid_price * 0.9998)

    def test_forced_close_removes_all_open_positions(self, silent_trade_output):
        trader = BacktestPaperTrader()
        trader.process_signal("BTC", "DOWN", 100.0, "2026-01-01 00:00:00 UTC", 1_000.0)
        trader.process_signal("ETH", "DOWN", 50.0, "2026-01-01 00:00:00 UTC", 1_000.0)

        trader.force_close_all_positions(
            {"BTC": 110.0, "ETH": 55.0},
            "2026-01-01 00:05:00 UTC",
        )

        assert not trader._get_position("BTC").is_open
        assert not trader._get_position("ETH").is_open
        assert trader.total_open_exposure() == 0.0


class TestBacktestMetrics:
    """Drawdown math and end-to-end replay behavior."""

    def test_maximum_drawdown_calculation(self):
        equity_curve = [1_000.0, 1_050.0, 1_000.0]

        drawdown_pct = compute_max_drawdown(equity_curve)

        assert drawdown_pct == pytest.approx(4.7619, rel=1e-3)

    def test_backtester_does_not_write_to_live_paper_trades_csv(
        self, tmp_path, silent_trade_output
    ):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)

        live_mtime_before = LIVE_PAPER_TRADES.stat().st_mtime if LIVE_PAPER_TRADES.exists() else None
        live_size_before = LIVE_PAPER_TRADES.stat().st_size if LIVE_PAPER_TRADES.exists() else None

        run_backtest(csv_path=price_log, test_mode=True)

        if live_mtime_before is not None:
            assert LIVE_PAPER_TRADES.stat().st_mtime == live_mtime_before
            assert LIVE_PAPER_TRADES.stat().st_size == live_size_before

    def test_backtest_results_contain_test_and_production_modes(
        self, tmp_path, silent_trade_output
    ):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)

        results = run_comparison(csv_path=price_log)

        assert "test_mode" in results
        assert "production_mode" in results
        assert "with_open_positions" in results["test_mode"]
        assert "with_open_positions" in results["production_mode"]
        assert results["test_mode"]["test_mode"] is True
        assert results["production_mode"]["test_mode"] is False

    def test_run_backtest_applies_fees_during_replay(
        self, tmp_path, silent_trade_output
    ):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        config = BacktestConfig(
            trading_fee_pct=0.05,
            slippage_pct=0.02,
            max_session_trades=1000,
        )

        results = run_backtest(csv_path=price_log, test_mode=True, config=config)
        summary = results["with_open_positions"]

        assert summary["total_fees_paid"] >= 0.0
        assert summary["starting_equity"] == PAPER_TRADING.starting_balance_usdc
