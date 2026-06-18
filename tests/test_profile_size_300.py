"""Tests for the contrarian_v1_size_300_test paper-trading profile."""

from __future__ import annotations

from pathlib import Path

import pytest

import dashboard_data
import session
import session_report
from config import PAPER_TRADING, SIGNAL
from paper_trader import PaperTrader
from risk_rules import buy_block_reason, sell_block_reason
from session_report import SessionShutdownContext
from strategy_profiles import STRATEGY_PROFILES, get_profile


PROFILE_ID = "contrarian_v1_size_300_test"
BTC_TEST_PRICE = 64_500.0


def _profile_trader(**kwargs) -> PaperTrader:
    profile = get_profile(PROFILE_ID)
    defaults = {"session_id": "size-300-test", "mode": "TEST_MODE"}
    defaults.update(kwargs)
    return PaperTrader(profile=profile, **defaults)


def _buy(
    trader: PaperTrader,
    *,
    coin: str = "BTC",
    price: float = BTC_TEST_PRICE,
    now: float = 1_000.0,
    timestamp: str = "2026-01-01 00:00:00 UTC",
) -> None:
    trader.process_signal(coin, "DOWN", price, timestamp, now)


def _sell(
    trader: PaperTrader,
    *,
    coin: str = "BTC",
    price: float = BTC_TEST_PRICE,
    timestamp: str = "2026-01-01 00:01:00 UTC",
    now: float = 1_100.0,
) -> None:
    trader.process_signal(coin, "UP", price, timestamp, now)


class TestProfileConfiguration:
    def test_profile_values(self):
        profile = get_profile(PROFILE_ID)
        limits = profile.trading_limits()

        assert profile.profile_id == PROFILE_ID
        assert profile.strategy_name == "contrarian_v1"
        assert limits.starting_balance_usdc == 1000.0
        assert limits.trade_size_usdc == 300.0
        assert limits.max_allocation_per_coin_usdc == 300.0
        assert limits.max_total_open_exposure_usdc == 600.0
        assert limits.max_entries_per_coin == 1
        assert limits.trading_fee_pct == PAPER_TRADING.trading_fee_pct
        assert limits.slippage_pct == PAPER_TRADING.slippage_pct
        assert limits.min_seconds_between_buys == PAPER_TRADING.min_seconds_between_buys
        assert (
            limits.max_successful_trades_per_session
            == PAPER_TRADING.max_successful_trades_per_session
        )
        assert profile.thresholds == SIGNAL.test_mode

    def test_standard_profiles_unchanged(self):
        test_profile = get_profile("contrarian_v1_test")
        production_profile = get_profile("contrarian_v1_production")

        assert test_profile.trade_size_usdc == PAPER_TRADING.trade_size_usdc
        assert production_profile.trade_size_usdc == PAPER_TRADING.trade_size_usdc
        assert test_profile.max_entries_per_coin == PAPER_TRADING.max_entries_per_coin
        assert (
            production_profile.max_entries_per_coin
            == PAPER_TRADING.max_entries_per_coin
        )


class TestProfileTradeExecution:
    def test_trade_size_is_300(self, temp_data_paths, silent_trade_output):
        trader = _profile_trader()
        assert trader.trading_limits().trade_size_usdc == 300.0

    def test_btc_quantity_uses_fill_price_with_slippage(
        self, temp_data_paths, silent_trade_output
    ):
        trader = _profile_trader()
        execution_price = trader.execution_price("BUY", BTC_TEST_PRICE)
        expected_quantity = 300.0 / execution_price

        _buy(trader, price=BTC_TEST_PRICE)

        position = trader.positions["BTC"]
        assert position.quantity == pytest.approx(expected_quantity)
        assert expected_quantity == pytest.approx(300.0 / (BTC_TEST_PRICE * 1.0002))

    def test_one_entry_reaches_per_coin_cap(self, temp_data_paths, silent_trade_output):
        trader = _profile_trader()
        _buy(trader, price=BTC_TEST_PRICE)

        assert trader.positions["BTC"].allocation_usdc == pytest.approx(300.0, rel=1e-4)
        assert buy_block_reason(trader, "BTC", 10_000.0) == "per-coin allocation reached"

    def test_second_same_coin_buy_blocked(self, temp_data_paths, silent_trade_output):
        trader = _profile_trader()
        _buy(trader, price=BTC_TEST_PRICE, now=1_000.0)

        cash_before = trader.cash_balance
        _buy(trader, price=BTC_TEST_PRICE, now=10_000.0)

        assert trader.positions["BTC"].entry_count == 1
        assert trader.cash_balance == cash_before
        assert buy_block_reason(trader, "BTC", 10_000.0) == "per-coin allocation reached"

    def test_two_different_300_positions_allowed(
        self, temp_data_paths, silent_trade_output
    ):
        trader = _profile_trader()
        _buy(trader, coin="BTC", price=BTC_TEST_PRICE, now=1_000.0)
        _buy(trader, coin="ETH", price=3_000.0, now=2_000.0)

        assert trader.positions["BTC"].is_open
        assert trader.positions["ETH"].is_open
        assert trader.total_open_exposure() == pytest.approx(600.0, rel=1e-4)

    def test_third_position_blocked_by_total_exposure(
        self, temp_data_paths, silent_trade_output
    ):
        trader = _profile_trader()
        _buy(trader, coin="BTC", price=BTC_TEST_PRICE, now=1_000.0)
        _buy(trader, coin="ETH", price=3_000.0, now=2_000.0)

        cash_before = trader.cash_balance
        _buy(trader, coin="SOL", price=150.0, now=3_000.0)

        assert "SOL" not in trader.positions or not trader.positions["SOL"].is_open
        assert trader.cash_balance == cash_before
        assert buy_block_reason(trader, "SOL", 3_000.0) == "total exposure reached"

    def test_sell_allowed_at_cap(self, temp_data_paths, silent_trade_output):
        trader = _profile_trader()
        _buy(trader, price=BTC_TEST_PRICE)

        assert buy_block_reason(trader, "BTC", 10_000.0) == "per-coin allocation reached"
        assert sell_block_reason(trader, "BTC") is None

        cash_before = trader.cash_balance
        _sell(trader, price=BTC_TEST_PRICE + 500.0)

        assert trader.cash_balance > cash_before

    def test_cash_cannot_become_negative(self, temp_data_paths, silent_trade_output):
        trader = _profile_trader()
        trader.cash_balance = 250.0

        _buy(trader, price=BTC_TEST_PRICE)

        assert trader.cash_balance == 250.0
        assert "BTC" not in trader.positions or not trader.positions["BTC"].is_open


class TestProfileSessionReporting:
    def test_session_report_includes_profile_section(
        self, temp_data_paths, silent_trade_output, monkeypatch
    ):
        monkeypatch.setattr(dashboard_data, "DATA", temp_data_paths)
        monkeypatch.setattr(session_report, "DATA", temp_data_paths)

        trader = _profile_trader(session_id="report-session")
        _buy(trader, price=BTC_TEST_PRICE)

        profile_payload = get_profile(PROFILE_ID).to_dict()
        shutdown = SessionShutdownContext(
            shutdown_reason="keyboard_interrupt",
            started_at=trader.started_at,
            ended_at=trader.started_at + 60.0,
            current_prices={"BTC": BTC_TEST_PRICE, "ETH": 3_000.0, "SOL": 150.0},
            strategy_profile=PROFILE_ID,
            profile=profile_payload,
        )

        report = session_report.build_session_report(
            "report-session",
            shutdown=shutdown,
            paper_trader=trader,
        )

        profile_section = report["profile_session"]
        assert profile_section["profile_name"] == PROFILE_ID
        assert profile_section["trade_size_usdc"] == 300.0
        assert profile_section["max_allocation_per_coin_usdc"] == 300.0
        assert profile_section["max_total_open_exposure_usdc"] == 600.0
        assert profile_section["starting_equity"] == 1000.0
        assert profile_section["exposure_increasing_entries"] == 1
        assert profile_section["risk_reducing_exits"] == 0
        assert profile_section["total_executions"] == 1
        assert len(profile_section["quantities_bought"]) == 1
        assert profile_section["quantities_bought"][0]["coin"] == "BTC"
        assert profile_section["accounting_valid"] is True
        assert profile_section["leverage_enabled"] is False
        assert profile_section["margin_enabled"] is False
        assert profile_section["paper_only"] is True

    def test_session_config_snapshot_includes_profile(self):
        snapshot = session.session_config_snapshot(profile_id=PROFILE_ID)

        assert snapshot["strategy_profile"] == PROFILE_ID
        assert snapshot["profile"]["trade_size_usdc"] == 300.0
        assert snapshot["paper_only"] is True
        assert snapshot["leverage_enabled"] is False


class TestNoLiveDataModification:
    def test_profile_trades_use_temp_files_only(
        self, temp_data_paths, silent_trade_output
    ):
        repo_root = Path(__file__).resolve().parent.parent
        live_trades = repo_root / "data" / "paper_trades.csv"
        before = live_trades.read_bytes() if live_trades.exists() else None

        trader = _profile_trader()
        _buy(trader, price=BTC_TEST_PRICE)

        assert temp_data_paths.paper_trades_csv.exists()
        if before is not None:
            assert live_trades.read_bytes() == before
        else:
            assert not live_trades.exists()

    def test_profile_registered_without_modifying_existing_profiles(self):
        assert PROFILE_ID in STRATEGY_PROFILES
        assert "contrarian_v1_test" in STRATEGY_PROFILES
        assert "contrarian_v1_production" in STRATEGY_PROFILES
