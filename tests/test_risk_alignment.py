"""Tests that decision recommendations and paper trades share risk rules."""

import copy

import pytest

from config import PAPER_TRADING
from decision_engine import (
    RECOMMENDATION_BLOCKED,
    RECOMMENDATION_BUY,
    RECOMMENDATION_HOLD,
    RECOMMENDATION_SELL,
    decide_for_coin,
)
from paper_trader import PaperTrader, Position
from risk_rules import buy_block_reason, sell_block_reason


def _decide(
    trader: PaperTrader,
    *,
    coin: str = "BTC",
    signal: str = "DOWN",
    price: float = 100.0,
    now: float = 10_000.0,
):
    change = -0.30 if signal in ("DOWN", "STRONG_DOWN") else 0.30
    return decide_for_coin(
        coin=coin,
        signal=signal,
        current_price=price,
        change_since_start=0.0,
        change_1m=change,
        change_5m=change,
        trader=trader,
        now=now,
    )


def _paper_block_reason(
    trader: PaperTrader,
    coin: str,
    signal: str,
    now: float,
) -> str | None:
    if signal in ("STRONG_DOWN", "DOWN"):
        return buy_block_reason(trader, coin, now)
    if signal in ("STRONG_UP", "UP"):
        position = trader.positions.get(coin, Position())
        if not position.is_open:
            return None
        return sell_block_reason(trader, coin)
    return None


def _snapshot_trader(trader: PaperTrader) -> dict:
    return {
        "cash_balance": trader.cash_balance,
        "realized_pnl": trader.realized_pnl,
        "successful_trades_this_session": trader.successful_trades_this_session,
        "is_paused": trader.is_paused,
        "positions": copy.deepcopy(trader.positions),
    }


def _apply_paper_signal(
    trader: PaperTrader,
    *,
    coin: str,
    signal: str,
    price: float,
    now: float,
) -> None:
    trader.process_signal(
        coin,
        signal,
        price,
        "2026-01-01 00:00:00 UTC",
        now,
    )


def _trader_with_open_btc() -> PaperTrader:
    trader = PaperTrader()
    trader.positions["BTC"] = Position(
        quantity=1.0,
        avg_entry_price=100.0,
        entry_count=1,
    )
    return trader


def _trader_at_total_exposure() -> PaperTrader:
    trader = PaperTrader()
    max_buys = int(
        PAPER_TRADING.max_total_open_exposure_usdc // PAPER_TRADING.trade_size_usdc
    )
    coins = ("BTC", "ETH", "SOL")

    for index in range(max_buys):
        coin = coins[index % len(coins)]
        position = trader.positions.get(coin, Position())
        add_quantity = PAPER_TRADING.trade_size_usdc / 100.0
        total_cost = (
            position.avg_entry_price * position.quantity + PAPER_TRADING.trade_size_usdc
        )
        position.quantity += add_quantity
        position.avg_entry_price = total_cost / position.quantity
        position.entry_count += 1
        trader.positions[coin] = position

    return trader


class TestRecommendationPaperTraderAlignment:
    """Recommendations and paper trades must agree on allow/block."""

    @pytest.mark.parametrize(
        ("trader_factory", "coin", "signal", "price", "now"),
        [
            (PaperTrader, "BTC", "DOWN", 100.0, 10_000.0),
            (PaperTrader, "BTC", "STRONG_DOWN", 100.0, 10_000.0),
            (_trader_with_open_btc, "BTC", "UP", 110.0, 10_000.0),
            (_trader_with_open_btc, "BTC", "STRONG_UP", 110.0, 10_000.0),
        ],
    )
    def test_allowed_recommendation_executes_paper_trade(
        self,
        temp_data_paths,
        silent_trade_output,
        trader_factory,
        coin,
        signal,
        price,
        now,
    ):
        trader = trader_factory()
        decision = _decide(trader, coin=coin, signal=signal, price=price, now=now)

        assert decision.recommendation in (RECOMMENDATION_BUY, RECOMMENDATION_SELL)
        assert _paper_block_reason(trader, coin, signal, now) is None

        before = _snapshot_trader(trader)
        _apply_paper_signal(trader, coin=coin, signal=signal, price=price, now=now)
        after = _snapshot_trader(trader)

        assert after != before
        assert (
            after["successful_trades_this_session"]
            == before["successful_trades_this_session"] + 1
        )

    @pytest.mark.parametrize(
        ("setup", "coin", "signal", "price", "now", "expected_reason"),
        [
            (
                lambda t: setattr(t, "cash_balance", 0.0),
                "BTC",
                "DOWN",
                100.0,
                10_000.0,
                "insufficient fake cash",
            ),
            (
                lambda t: setattr(t, "is_paused", True),
                "BTC",
                "DOWN",
                100.0,
                10_000.0,
                "equity protection triggered",
            ),
            (
                lambda t: t.positions.update(
                    {
                        "BTC": Position(
                            quantity=1.0,
                            avg_entry_price=100.0,
                            entry_count=1,
                            last_buy_time=1_000.0,
                        )
                    }
                ),
                "BTC",
                "DOWN",
                100.0,
                1_000.0 + PAPER_TRADING.min_seconds_between_buys - 1,
                "cooldown active",
            ),
            (
                lambda t: t.positions.update(
                    {
                        "BTC": Position(
                            quantity=1.0,
                            avg_entry_price=100.0,
                            entry_count=PAPER_TRADING.max_entries_per_coin,
                        )
                    }
                ),
                "BTC",
                "STRONG_DOWN",
                100.0,
                10_000.0,
                "maximum entries reached",
            ),
            (
                lambda t: t.positions.update(
                    {
                        "BTC": Position(
                            quantity=PAPER_TRADING.max_allocation_per_coin_usdc / 100.0,
                            avg_entry_price=100.0,
                            entry_count=1,
                        )
                    }
                ),
                "BTC",
                "DOWN",
                100.0,
                10_000.0,
                "per-coin allocation reached",
            ),
            (
                lambda t: (
                    setattr(
                        t,
                        "successful_trades_this_session",
                        t.session_trade_limit(),
                    )
                ),
                "BTC",
                "DOWN",
                100.0,
                10_000.0,
                "session trade limit reached",
            ),
            (
                lambda t: (
                    setattr(
                        t,
                        "successful_trades_this_session",
                        t.session_trade_limit(),
                    ),
                    t.positions.update(
                        {
                            "BTC": Position(
                                quantity=1.0,
                                avg_entry_price=100.0,
                                entry_count=1,
                            )
                        }
                    ),
                ),
                "BTC",
                "UP",
                110.0,
                10_000.0,
                "session trade limit reached",
            ),
        ],
    )
    def test_blocked_recommendation_matches_paper_trader_reason(
        self,
        temp_data_paths,
        silent_trade_output,
        setup,
        coin,
        signal,
        price,
        now,
        expected_reason,
    ):
        trader = PaperTrader()
        setup(trader)
        decision = _decide(trader, coin=coin, signal=signal, price=price, now=now)

        assert decision.recommendation == RECOMMENDATION_BLOCKED
        assert decision.primary_reason == expected_reason
        assert _paper_block_reason(trader, coin, signal, now) == expected_reason

        before = _snapshot_trader(trader)
        _apply_paper_signal(trader, coin=coin, signal=signal, price=price, now=now)
        after = _snapshot_trader(trader)

        assert after == before

    def test_total_exposure_block_is_aligned(
        self,
        temp_data_paths,
        silent_trade_output,
    ):
        trader = _trader_at_total_exposure()
        decision = _decide(trader, coin="SOL", signal="DOWN", price=100.0, now=10_000.0)

        assert decision.recommendation == RECOMMENDATION_BLOCKED
        assert decision.primary_reason == "total exposure reached"
        assert buy_block_reason(trader, "SOL", 10_000.0) == "total exposure reached"

        before = _snapshot_trader(trader)
        _apply_paper_signal(
            trader, coin="SOL", signal="DOWN", price=100.0, now=10_000.0
        )
        assert _snapshot_trader(trader) == before

    def test_hold_recommendation_does_not_trade_without_position(
        self,
        temp_data_paths,
        silent_trade_output,
    ):
        trader = PaperTrader()
        decision = _decide(trader, coin="BTC", signal="UP", price=110.0, now=10_000.0)

        assert decision.recommendation == RECOMMENDATION_HOLD

        cash_before = trader.cash_balance
        trades_before = trader.successful_trades_this_session
        _apply_paper_signal(trader, coin="BTC", signal="UP", price=110.0, now=10_000.0)

        assert trader.cash_balance == cash_before
        assert trader.successful_trades_this_session == trades_before
