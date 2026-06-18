"""Tests for simulated paper trading rules and state updates."""

import pytest

from config import PAPER_TRADING
from paper_trader import PaperTrader, Position


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


class TestPaperTraderBasics:
    """Core buy/sell behavior and position tracking."""

    def test_starting_cash_matches_config(
        self, temp_data_paths, silent_trade_output
    ):
        trader = PaperTrader()

        assert trader.cash_balance == PAPER_TRADING.starting_balance_usdc

    def test_buy_reduces_cash_and_adds_quantity(
        self, temp_data_paths, silent_trade_output
    ):
        trader = PaperTrader()
        price = 200.0
        execution_price = trader.execution_price("BUY", price)
        buy_fee = PAPER_TRADING.trade_size_usdc * (PAPER_TRADING.trading_fee_pct / 100.0)

        _buy(trader, price=price)

        position = trader.positions["BTC"]
        expected_quantity = PAPER_TRADING.trade_size_usdc / execution_price

        assert trader.cash_balance == pytest.approx(
            PAPER_TRADING.starting_balance_usdc
            - PAPER_TRADING.trade_size_usdc
            - buy_fee
        )
        assert position.quantity == pytest.approx(expected_quantity)
        assert position.avg_entry_price == pytest.approx(execution_price)

    def test_multiple_buys_calculate_weighted_average_entry(
        self, temp_data_paths, silent_trade_output
    ):
        trader = PaperTrader()

        _buy(trader, price=100.0, now=1_000.0)
        _buy(trader, price=200.0, now=1_100.0)

        position = trader.positions["BTC"]
        first_execution = trader.execution_price("BUY", 100.0)
        second_execution = trader.execution_price("BUY", 200.0)
        first_quantity = PAPER_TRADING.trade_size_usdc / first_execution
        second_quantity = PAPER_TRADING.trade_size_usdc / second_execution
        expected_avg = (
            (first_execution * first_quantity) + PAPER_TRADING.trade_size_usdc
        ) / (first_quantity + second_quantity)

        assert position.quantity == pytest.approx(first_quantity + second_quantity)
        assert position.avg_entry_price == pytest.approx(expected_avg)
        assert position.entry_count == 2

    def test_sell_updates_cash_and_realized_pnl(
        self, temp_data_paths, silent_trade_output
    ):
        trader = PaperTrader()

        _buy(trader, price=100.0)
        position = trader.positions["BTC"]
        quantity_before_sell = position.quantity
        avg_entry_before_sell = position.avg_entry_price
        sell_execution_price = trader.execution_price("SELL", 110.0)
        _sell(trader, price=110.0)

        expected_realized_pnl = (
            (sell_execution_price - avg_entry_before_sell) * quantity_before_sell
        )
        buy_fee = PAPER_TRADING.trade_size_usdc * (PAPER_TRADING.trading_fee_pct / 100.0)
        trade_value = sell_execution_price * quantity_before_sell
        sell_fee = trade_value * (PAPER_TRADING.trading_fee_pct / 100.0)

        assert trader.realized_pnl == pytest.approx(expected_realized_pnl)
        assert trader.cash_balance == pytest.approx(
            PAPER_TRADING.starting_balance_usdc
            - PAPER_TRADING.trade_size_usdc
            - buy_fee
            + trade_value
            - sell_fee
        )
        assert not trader.positions["BTC"].is_open

    def test_full_close_resets_position_state(
        self, temp_data_paths, silent_trade_output
    ):
        trader = PaperTrader()

        _buy(trader, price=100.0, now=1_000.0)
        _buy(trader, price=100.0, now=1_100.0)
        _sell(trader, price=100.0, now=1_200.0)
        _sell(trader, price=100.0, now=1_300.0)

        position = trader.positions["BTC"]

        assert position.quantity == 0.0
        assert position.avg_entry_price == 0.0
        assert position.entry_count == 0
        assert position.last_buy_time is None
        assert position.allocation_usdc == 0.0


class TestPaperTraderRiskRules:
    """Risk limits that block or pause paper trades."""

    def test_insufficient_cash_blocks_buy(
        self, temp_data_paths, silent_trade_output
    ):
        trader = PaperTrader()
        trader.cash_balance = PAPER_TRADING.trade_size_usdc - 1.0

        _buy(trader, price=100.0)

        assert "BTC" not in trader.positions or not trader.positions["BTC"].is_open
        assert trader.cash_balance == PAPER_TRADING.trade_size_usdc - 1.0

    def test_per_coin_allocation_limit_blocks_buy(
        self, temp_data_paths, silent_trade_output
    ):
        trader = PaperTrader()
        max_buys = int(
            PAPER_TRADING.max_allocation_per_coin_usdc // PAPER_TRADING.trade_size_usdc
        )

        for index in range(max_buys):
            _buy(trader, price=100.0, now=1_000.0 + (index * 100))

        cash_before_blocked_buy = trader.cash_balance
        _buy(trader, price=100.0, now=10_000.0)

        position = trader.positions["BTC"]
        assert position.entry_count == max_buys
        assert trader.cash_balance == cash_before_blocked_buy

    def test_maximum_entries_blocks_buy(
        self, temp_data_paths, silent_trade_output
    ):
        trader = PaperTrader()

        for index in range(PAPER_TRADING.max_entries_per_coin):
            _buy(trader, price=100.0, now=1_000.0 + (index * 100))

        cash_before_blocked_buy = trader.cash_balance
        _buy(trader, price=100.0, now=10_000.0)

        assert trader.positions["BTC"].entry_count == PAPER_TRADING.max_entries_per_coin
        assert trader.cash_balance == cash_before_blocked_buy

    def test_cooldown_blocks_repeated_buys(
        self, temp_data_paths, silent_trade_output
    ):
        trader = PaperTrader()

        _buy(trader, price=100.0, now=1_000.0)

        cash_after_first_buy = trader.cash_balance
        _buy(trader, price=100.0, now=1_000.0 + PAPER_TRADING.min_seconds_between_buys - 1)

        assert trader.positions["BTC"].entry_count == 1
        assert trader.cash_balance == cash_after_first_buy

    def test_total_exposure_limit_blocks_buy(
        self, temp_data_paths, silent_trade_output
    ):
        trader = PaperTrader()
        max_buys = int(
            PAPER_TRADING.max_total_open_exposure_usdc // PAPER_TRADING.trade_size_usdc
        )
        coins = ("BTC", "ETH", "SOL")

        for index in range(max_buys):
            coin = coins[index % len(coins)]
            _buy(trader, coin=coin, price=100.0, now=1_000.0 + (index * 100))

        cash_before_blocked_buy = trader.cash_balance
        _buy(trader, coin="SOL", price=100.0, now=10_000.0)

        assert trader.total_open_exposure() == pytest.approx(
            PAPER_TRADING.max_total_open_exposure_usdc
        )
        assert trader.cash_balance == cash_before_blocked_buy

    def test_session_trade_limit_blocks_buy_allows_sell(
        self, temp_data_paths, silent_trade_output
    ):
        trader = PaperTrader()
        trader.successful_trades_this_session = trader.session_trade_limit()

        cash_before = trader.cash_balance
        _buy(trader, price=100.0)

        assert trader.cash_balance == cash_before

        trader.positions["BTC"] = Position(
            quantity=1.0,
            avg_entry_price=100.0,
            entry_count=1,
        )
        cash_with_open_position = trader.cash_balance

        _sell(trader, price=110.0)

        assert not trader.positions["BTC"].is_open
        assert trader.cash_balance > cash_with_open_position
        assert trader.successful_trades_this_session == trader.session_trade_limit() + 1

    def test_equity_protection_pauses_trading(
        self, temp_data_paths, silent_trade_output
    ):
        trader = PaperTrader()
        trader.cash_balance = PAPER_TRADING.min_equity_before_pause_usdc - 1.0

        trader.update_session_risk({"BTC": 100.0})

        assert trader.is_paused is True

        _buy(trader, price=100.0)

        assert trader.cash_balance == PAPER_TRADING.min_equity_before_pause_usdc - 1.0
        assert "BTC" not in trader.positions or not trader.positions["BTC"].is_open
