"""Tests for read-only recommendation and blocking logic."""

import pytest

from config import PAPER_TRADING
from decision_engine import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    RECOMMENDATION_BLOCKED,
    RECOMMENDATION_BUY,
    RECOMMENDATION_HOLD,
    RECOMMENDATION_SELL,
    build_decisions,
    decide_for_coin,
)
from risk_rules import buy_block_reason
from market_snapshot import MarketRow
from paper_trader import PaperTrader, Position


def _row(
    *,
    coin: str = "BTC",
    signal: str = "FLAT",
    current_price: float = 100.0,
    change_1m: float | None = 0.0,
    change_5m: float | None = 0.0,
) -> MarketRow:
    return MarketRow(
        coin=coin,
        current_price=current_price,
        change_since_start=0.0,
        change_1m=change_1m,
        change_5m=change_5m,
        signal=signal,
    )


class TestRecommendationRules:
    """Signal-to-recommendation mapping with risk checks passing."""

    def test_down_signal_recommends_paper_buy(self):
        trader = PaperTrader()
        decision = decide_for_coin(
            coin="BTC",
            signal="DOWN",
            current_price=100.0,
            change_since_start=0.0,
            change_1m=-0.30,
            change_5m=-0.40,
            trader=trader,
            now=1_000.0,
        )

        assert decision.recommendation == RECOMMENDATION_BUY
        assert decision.confidence == CONFIDENCE_MEDIUM

    def test_strong_down_signal_has_high_confidence_buy(self):
        trader = PaperTrader()
        decision = decide_for_coin(
            coin="BTC",
            signal="STRONG_DOWN",
            current_price=100.0,
            change_since_start=0.0,
            change_1m=-0.60,
            change_5m=-0.80,
            trader=trader,
            now=1_000.0,
        )

        assert decision.recommendation == RECOMMENDATION_BUY
        assert decision.confidence == CONFIDENCE_HIGH

    def test_up_signal_with_position_recommends_paper_sell(self):
        trader = PaperTrader()
        trader.positions["BTC"] = Position(
            quantity=1.0,
            avg_entry_price=100.0,
            entry_count=1,
        )

        decision = decide_for_coin(
            coin="BTC",
            signal="UP",
            current_price=110.0,
            change_since_start=1.0,
            change_1m=0.30,
            change_5m=0.40,
            trader=trader,
            now=1_000.0,
        )

        assert decision.recommendation == RECOMMENDATION_SELL
        assert decision.current_position_value == pytest.approx(110.0)

    def test_up_signal_without_position_holds(self):
        trader = PaperTrader()
        decision = decide_for_coin(
            coin="BTC",
            signal="UP",
            current_price=110.0,
            change_since_start=1.0,
            change_1m=0.30,
            change_5m=0.40,
            trader=trader,
            now=1_000.0,
        )

        assert decision.recommendation == RECOMMENDATION_HOLD
        assert "no open position to sell" in decision.primary_reason

    def test_flat_signal_holds(self):
        trader = PaperTrader()
        decision = decide_for_coin(
            coin="ETH",
            signal="FLAT",
            current_price=50.0,
            change_since_start=0.0,
            change_1m=0.01,
            change_5m=0.02,
            trader=trader,
            now=1_000.0,
        )

        assert decision.recommendation == RECOMMENDATION_HOLD
        assert decision.confidence == CONFIDENCE_LOW


class TestBlockingRules:
    """Risk rules that turn actionable signals into BLOCKED."""

    def test_insufficient_cash_blocks_buy(self):
        trader = PaperTrader()
        trader.cash_balance = PAPER_TRADING.trade_size_usdc - 1.0

        decision = decide_for_coin(
            coin="BTC",
            signal="DOWN",
            current_price=100.0,
            change_since_start=0.0,
            change_1m=-0.30,
            change_5m=-0.30,
            trader=trader,
            now=1_000.0,
        )

        assert decision.recommendation == RECOMMENDATION_BLOCKED
        assert decision.primary_reason == "insufficient fake cash"

    def test_cooldown_blocks_buy(self):
        trader = PaperTrader()
        trader.positions["BTC"] = Position(
            quantity=1.0,
            avg_entry_price=100.0,
            entry_count=1,
            last_buy_time=1_000.0,
        )

        decision = decide_for_coin(
            coin="BTC",
            signal="DOWN",
            current_price=100.0,
            change_since_start=0.0,
            change_1m=-0.30,
            change_5m=-0.30,
            trader=trader,
            now=1_000.0 + PAPER_TRADING.min_seconds_between_buys - 1,
        )

        assert decision.recommendation == RECOMMENDATION_BLOCKED
        assert decision.primary_reason == "cooldown active"

    def test_maximum_entries_blocks_buy(self):
        trader = PaperTrader()
        trader.positions["BTC"] = Position(
            quantity=1.0,
            avg_entry_price=100.0,
            entry_count=PAPER_TRADING.max_entries_per_coin,
        )

        decision = decide_for_coin(
            coin="BTC",
            signal="STRONG_DOWN",
            current_price=100.0,
            change_since_start=0.0,
            change_1m=-0.60,
            change_5m=-0.60,
            trader=trader,
            now=10_000.0,
        )

        assert decision.recommendation == RECOMMENDATION_BLOCKED
        assert decision.primary_reason == "maximum entries reached"
        assert decision.confidence == CONFIDENCE_MEDIUM

    def test_per_coin_allocation_blocks_buy(self):
        trader = PaperTrader()
        trader.positions["BTC"] = Position(
            quantity=PAPER_TRADING.max_allocation_per_coin_usdc / 100.0,
            avg_entry_price=100.0,
            entry_count=1,
        )

        decision = decide_for_coin(
            coin="BTC",
            signal="DOWN",
            current_price=100.0,
            change_since_start=0.0,
            change_1m=-0.30,
            change_5m=-0.30,
            trader=trader,
            now=10_000.0,
        )

        assert decision.recommendation == RECOMMENDATION_BLOCKED
        assert decision.primary_reason == "per-coin allocation reached"

    def test_total_exposure_blocks_buy(self):
        trader = PaperTrader()
        max_buys = int(
            PAPER_TRADING.max_total_open_exposure_usdc // PAPER_TRADING.trade_size_usdc
        )
        coins = ("BTC", "ETH", "SOL")

        for index in range(max_buys):
            coin = coins[index % len(coins)]
            position = trader.positions.get(coin, Position())
            add_quantity = PAPER_TRADING.trade_size_usdc / 100.0
            total_cost = position.avg_entry_price * position.quantity + PAPER_TRADING.trade_size_usdc
            position.quantity += add_quantity
            position.avg_entry_price = total_cost / position.quantity
            position.entry_count += 1
            trader.positions[coin] = position

        decision = decide_for_coin(
            coin="SOL",
            signal="DOWN",
            current_price=100.0,
            change_since_start=0.0,
            change_1m=-0.30,
            change_5m=-0.30,
            trader=trader,
            now=10_000.0,
        )

        assert decision.recommendation == RECOMMENDATION_BLOCKED
        assert decision.primary_reason == "total exposure reached"

    def test_session_trade_limit_blocks_buy_allows_sell(self):
        trader = PaperTrader()
        trader.successful_trades_this_session = trader.session_trade_limit()
        trader.positions["BTC"] = Position(
            quantity=1.0,
            avg_entry_price=100.0,
            entry_count=1,
        )

        buy_decision = decide_for_coin(
            coin="BTC",
            signal="DOWN",
            current_price=100.0,
            change_since_start=0.0,
            change_1m=-0.30,
            change_5m=-0.30,
            trader=trader,
            now=10_000.0,
        )
        sell_decision = decide_for_coin(
            coin="BTC",
            signal="UP",
            current_price=110.0,
            change_since_start=1.0,
            change_1m=0.30,
            change_5m=0.30,
            trader=trader,
            now=10_000.0,
        )

        assert buy_decision.recommendation == RECOMMENDATION_BLOCKED
        assert buy_decision.primary_reason == "session trade limit reached"
        assert sell_decision.recommendation == RECOMMENDATION_SELL
        assert sell_decision.primary_reason == "UP signal — paper sell with open position"

    def test_equity_protection_blocks_buy(self):
        trader = PaperTrader()
        trader.is_paused = True

        decision = decide_for_coin(
            coin="BTC",
            signal="DOWN",
            current_price=100.0,
            change_since_start=0.0,
            change_1m=-0.30,
            change_5m=-0.30,
            trader=trader,
            now=10_000.0,
        )

        assert decision.recommendation == RECOMMENDATION_BLOCKED
        assert decision.primary_reason == "equity protection triggered"

    def test_buy_block_reason_reports_insufficient_cash(self):
        trader = PaperTrader()
        trader.cash_balance = 0.0

        assert buy_block_reason(trader, "BTC", now=1_000.0) == "insufficient fake cash"


class TestDecisionSummary:
    """Session summary counts from build_decisions."""

    def test_summary_counts_active_blocked_and_positions(self):
        trader = PaperTrader()
        trader.positions["BTC"] = Position(
            quantity=1.0,
            avg_entry_price=100.0,
            entry_count=1,
        )
        trader.successful_trades_this_session = 2

        rows = [
            _row(coin="BTC", signal="DOWN", change_1m=-0.30),
            _row(coin="ETH", signal="UP", current_price=50.0, change_1m=0.30),
            _row(coin="SOL", signal="FLAT", current_price=20.0, change_1m=0.0),
        ]
        prices = {"BTC": 100.0, "ETH": 50.0, "SOL": 20.0}

        decisions, summary = build_decisions(rows, trader, prices, now=10_000.0)

        assert len(decisions) == 3
        assert summary.active_recommendations == 1
        assert summary.blocked_recommendations == 0
        assert summary.open_positions == 1
        assert summary.session_trades == 2
        assert summary.total_equity is not None
        assert summary.current_drawdown_pct == pytest.approx(0.0)
