"""Tests for v8.1 forward-testing analytics."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import dashboard_data
import equity_logger
import session
import session_report
from config import PAPER_TRADING
from paper_trader import LEGACY_CSV_COLUMNS, PaperTrader


def _buy(
    trader: PaperTrader,
    *,
    coin: str = "BTC",
    price: float = 100.0,
    now: float = 1_000.0,
    timestamp: str = "2026-01-01 00:00:00 UTC",
) -> None:
    trader.process_signal(coin, "DOWN", price, timestamp, now)


def _sell(
    trader: PaperTrader,
    *,
    coin: str = "BTC",
    price: float = 100.0,
    timestamp: str = "2026-01-01 00:01:00 UTC",
    now: float = 1_100.0,
) -> None:
    trader.process_signal(coin, "UP", price, timestamp, now)


class TestSessionTracking:
    def test_generate_session_id_is_unique(self):
        ids = {session.generate_session_id() for _ in range(20)}
        assert len(ids) == 20
        assert all(len(value) == 32 for value in ids)

    def test_session_config_snapshot_records_mode_and_costs(self):
        snapshot = session.session_config_snapshot(test_mode=True)

        assert snapshot["test_mode"] is True
        assert snapshot["mode"] == "TEST_MODE"
        assert snapshot["costs"]["trading_fee_pct"] == PAPER_TRADING.trading_fee_pct
        assert snapshot["costs"]["slippage_pct"] == PAPER_TRADING.slippage_pct
        assert snapshot["paper_trading"]["starting_balance_usdc"] == 1000.0

    def test_paper_trade_csv_includes_session_fields(
        self, temp_data_paths, silent_trade_output
    ):
        trader = PaperTrader(session_id="abc123", mode="TEST_MODE")
        _buy(trader, price=100.0)

        content = temp_data_paths.paper_trades_csv.read_text()
        header = content.splitlines()[0]
        row = content.splitlines()[1]

        assert "session_id" in header
        assert "abc123" in row
        assert "TEST_MODE" in row
        assert "DOWN" in row


class TestLivePaperCosts:
    def test_fees_and_slippage_match_backtest_rules(
        self, temp_data_paths, silent_trade_output
    ):
        trader = PaperTrader()
        mid_price = 100.0

        buy_price = trader.execution_price("BUY", mid_price)
        sell_price = trader.execution_price("SELL", mid_price)
        buy_fee = trader.buy_fee(100.0)
        sell_fee = trader.sell_fee(100.0)

        assert buy_price == pytest.approx(mid_price * 1.0002)
        assert sell_price == pytest.approx(mid_price * 0.9998)
        assert buy_fee == pytest.approx(0.05)
        assert sell_fee == pytest.approx(0.05)
        assert trader.total_fees_paid == pytest.approx(0.10)

    def test_buy_reduces_cash_by_trade_size_and_fee(
        self, temp_data_paths, silent_trade_output
    ):
        trader = PaperTrader()
        _buy(trader, price=100.0)

        expected_fee = 100.0 * (PAPER_TRADING.trading_fee_pct / 100.0)
        assert trader.cash_balance == pytest.approx(
            PAPER_TRADING.starting_balance_usdc - PAPER_TRADING.trade_size_usdc - expected_fee
        )
        assert trader.total_fees_paid == pytest.approx(expected_fee)


class TestEquityLogging:
    def test_append_equity_snapshot_creates_csv(
        self, temp_data_paths, silent_trade_output
    ):
        trader = PaperTrader(session_id="session-1", mode="TEST_MODE")
        equity_logger.append_equity_snapshot(
            timestamp="2026-01-01 00:00:00 UTC",
            session_id="session-1",
            mode="TEST_MODE",
            paper_trader=trader,
            current_prices={"BTC": 100.0},
        )

        assert temp_data_paths.equity_log_csv.exists()
        content = temp_data_paths.equity_log_csv.read_text()
        assert "session_id" in content.splitlines()[0]
        assert "session-1" in content

    def test_append_equity_snapshot_handles_missing_file(
        self, temp_data_paths, silent_trade_output
    ):
        trader = PaperTrader(session_id="session-2", mode="production")
        assert not temp_data_paths.equity_log_csv.exists()

        equity_logger.append_equity_snapshot(
            timestamp="2026-01-01 00:00:00 UTC",
            session_id="session-2",
            mode="production",
            paper_trader=trader,
            current_prices={},
        )

        df = pd.read_csv(temp_data_paths.equity_log_csv)
        assert len(df) == 1
        assert df.loc[0, "mode"] == "production"


class TestLegacyCsvCompatibility:
    def test_normalize_paper_trades_fills_missing_columns(self, tmp_path):
        legacy_csv = tmp_path / "paper_trades.csv"
        legacy_csv.write_text(
            "\n".join(
                [
                    ",".join(LEGACY_CSV_COLUMNS),
                    "2026-01-01 00:00:00 UTC,BTC,BUY,100.0,1.0,100.00,900.00,0.00",
                    "2026-01-01 00:01:00 UTC,BTC,SELL,110.0,1.0,110.00,1010.00,10.00",
                ]
            )
            + "\n"
        )

        df = dashboard_data.normalize_paper_trades(pd.read_csv(legacy_csv))

        assert df.loc[0, "session_id"] == ""
        assert df.loc[0, "fee"] == 0.0
        assert df.loc[0, "trade_pnl"] == 0.0
        assert df.loc[1, "trade_pnl"] == pytest.approx(10.0)
        assert df.loc[1, "cumulative_realized_pnl"] == pytest.approx(10.0)

    def test_migrate_legacy_paper_trades_csv(self, temp_data_paths, silent_trade_output):
        temp_data_paths.data_dir.mkdir(parents=True, exist_ok=True)
        temp_data_paths.paper_trades_csv.write_text(
            ",".join(LEGACY_CSV_COLUMNS) + "\n"
            "2026-01-01 00:00:00 UTC,BTC,BUY,100.0,1.0,100.00,900.00,0.00\n"
        )

        trader = PaperTrader(session_id="new-session", mode="TEST_MODE")
        _buy(trader, price=100.0)

        lines = temp_data_paths.paper_trades_csv.read_text().splitlines()
        assert "session_id" in lines[0]
        assert any("new-session" in line for line in lines[1:])


class TestDashboardDataHelpers:
    def test_compute_trade_stats_and_profit_factor(self):
        trades = pd.DataFrame(
            [
                {
                    "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
                    "coin": "BTC",
                    "action": "BUY",
                    "trade_pnl": 0.0,
                    "fee": 0.05,
                    "cumulative_realized_pnl": 0.0,
                    "session_id": "s1",
                },
                {
                    "timestamp": datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
                    "coin": "BTC",
                    "action": "SELL",
                    "trade_pnl": 12.0,
                    "fee": 0.05,
                    "cumulative_realized_pnl": 12.0,
                    "session_id": "s1",
                },
                {
                    "timestamp": datetime(2026, 1, 1, 2, tzinfo=timezone.utc),
                    "coin": "ETH",
                    "action": "SELL",
                    "trade_pnl": -4.0,
                    "fee": 0.05,
                    "cumulative_realized_pnl": 8.0,
                    "session_id": "s1",
                },
            ]
        )

        stats = dashboard_data.compute_trade_stats(trades)
        pf = dashboard_data.compute_profit_factor(trades)

        assert stats["total_buys"] == 1
        assert stats["total_sells"] == 2
        assert stats["completed_trades"] == 2
        assert stats["win_rate_pct"] == 50.0
        assert stats["realized_pnl"] == 8.0
        assert stats["total_fees_paid"] == pytest.approx(0.15)
        assert pf == pytest.approx(12.0 / 4.0)

    def test_compute_drawdown_series(self):
        equity = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-01-01 00:00:00 UTC",
                        "2026-01-01 00:01:00 UTC",
                        "2026-01-01 00:02:00 UTC",
                    ],
                    utc=True,
                ),
                "total_equity": [1000.0, 1050.0, 1000.0],
            }
        )

        drawdown = dashboard_data.compute_drawdown_series(equity)
        assert drawdown["drawdown_pct"].iloc[0] == pytest.approx(0.0)
        assert drawdown["drawdown_pct"].iloc[-1] == pytest.approx(
            ((1050.0 - 1000.0) / 1050.0) * 100
        )


class TestSessionReports:
    def test_build_and_save_session_report(
        self, temp_data_paths, silent_trade_output, monkeypatch
    ):
        monkeypatch.setattr(dashboard_data, "DATA", temp_data_paths)

        trader = PaperTrader(session_id="report-session", mode="TEST_MODE")
        _buy(trader, price=100.0, timestamp="2026-01-01 00:00:00 UTC")
        _sell(trader, price=110.0, timestamp="2026-01-01 00:01:00 UTC")

        equity_logger.append_equity_snapshot(
            timestamp="2026-01-01 00:00:00 UTC",
            session_id="report-session",
            mode="TEST_MODE",
            paper_trader=trader,
            current_prices={"BTC": 100.0},
        )
        equity_logger.append_equity_snapshot(
            timestamp="2026-01-01 00:01:00 UTC",
            session_id="report-session",
            mode="TEST_MODE",
            paper_trader=trader,
            current_prices={"BTC": 110.0},
        )

        report = session_report.build_session_report("report-session")
        path = session_report.save_session_report("report-session", report)

        assert report["session_id"] == "report-session"
        assert report["mode"] == "TEST_MODE"
        assert report["total_buys"] == 1
        assert report["total_sells"] == 1
        assert report["completed_trades"] == 1
        assert report["fees_paid"] >= 0.0
        assert "results_by_coin" in report
        assert path.exists()
