"""Tests for session reporting, session limits, lookback, and blocked proposals."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import dashboard_data
import equity_logger
import session
import session_report
from config import PAPER_TRADING
from decision_engine import RECOMMENDATION_BLOCKED, RECOMMENDATION_SELL, decide_for_coin
from market_analyzer import (
    WINDOW_5M_SECONDS,
    record_price_history,
    rolling_percent_change,
)
from paper_trader import PaperTrader, Position, print_skipped_trade_proposal
from risk_rules import (
    SESSION_TRADE_LIMIT_REASON,
    buy_block_reason,
    is_exposure_increasing_action,
    is_risk_reducing_sell,
    sell_block_reason,
    session_trade_block_reason,
)
from session_report import SessionShutdownContext, validate_accounting_identity


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
    price: float = 110.0,
    timestamp: str = "2026-01-01 00:01:00 UTC",
    now: float = 1_100.0,
) -> None:
    trader.process_signal(coin, "UP", price, timestamp, now)


class TestSessionLimitExitPolicy:
    def test_buy_blocked_at_session_cap(self):
        trader = PaperTrader()
        trader.successful_trades_this_session = trader.session_trade_limit()

        assert buy_block_reason(trader, "BTC", 10_000.0) == SESSION_TRADE_LIMIT_REASON
        assert session_trade_block_reason(trader, "BTC", "BUY") == SESSION_TRADE_LIMIT_REASON

    def test_exposure_increasing_action_blocked_at_cap(self):
        trader = PaperTrader()
        trader.successful_trades_this_session = trader.session_trade_limit()

        assert is_exposure_increasing_action(trader, "BTC", "BUY") is True
        assert buy_block_reason(trader, "BTC", 10_000.0) == SESSION_TRADE_LIMIT_REASON

    def test_partial_sell_allowed_at_cap(self, temp_data_paths, silent_trade_output):
        trader = PaperTrader()
        trader.successful_trades_this_session = trader.session_trade_limit()
        trader.positions["BTC"] = Position(
            quantity=3.0,
            avg_entry_price=100.0,
            entry_count=2,
        )

        assert is_risk_reducing_sell(trader, "BTC") is True
        assert sell_block_reason(trader, "BTC") is None

        cash_before = trader.cash_balance
        _sell(trader, price=110.0)

        assert trader.positions["BTC"].is_open
        assert trader.cash_balance > cash_before

    def test_full_close_sell_allowed_at_cap(self, temp_data_paths, silent_trade_output):
        trader = PaperTrader()
        trader.successful_trades_this_session = trader.session_trade_limit()
        trader.positions["BTC"] = Position(
            quantity=0.5,
            avg_entry_price=100.0,
            entry_count=1,
        )

        assert sell_block_reason(trader, "BTC") is None
        _sell(trader, price=110.0)
        assert not trader.positions["BTC"].is_open

    def test_recommendation_and_execution_share_session_limit_reason(
        self,
        temp_data_paths,
        silent_trade_output,
    ):
        trader = PaperTrader()
        trader.successful_trades_this_session = trader.session_trade_limit()

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
        assert decision.primary_reason == SESSION_TRADE_LIMIT_REASON
        assert buy_block_reason(trader, "BTC", 10_000.0) == SESSION_TRADE_LIMIT_REASON

        cash_before = trader.cash_balance
        _buy(trader, price=100.0)
        assert trader.cash_balance == cash_before

    def test_sell_recommendation_allowed_at_cap(
        self,
        temp_data_paths,
        silent_trade_output,
    ):
        trader = PaperTrader()
        trader.successful_trades_this_session = trader.session_trade_limit()
        trader.positions["BTC"] = Position(
            quantity=1.0,
            avg_entry_price=100.0,
            entry_count=1,
        )

        decision = decide_for_coin(
            coin="BTC",
            signal="UP",
            current_price=110.0,
            change_since_start=0.0,
            change_1m=0.30,
            change_5m=0.30,
            trader=trader,
            now=10_000.0,
        )

        assert decision.recommendation == RECOMMENDATION_SELL
        assert sell_block_reason(trader, "BTC") is None


class TestAccountingReconciliation:
    def test_accounting_identity_holds_for_closed_session(
        self, temp_data_paths, silent_trade_output, monkeypatch
    ):
        monkeypatch.setattr(dashboard_data, "DATA", temp_data_paths)
        monkeypatch.setattr(session_report, "DATA", temp_data_paths)

        trader = PaperTrader(session_id="acct-session", mode="TEST_MODE")
        _buy(trader, price=100.0, timestamp="2026-01-01 00:00:00 UTC")
        _sell(trader, price=110.0, timestamp="2026-01-01 00:01:00 UTC")

        equity_logger.append_equity_snapshot(
            timestamp="2026-01-01 00:00:00 UTC",
            session_id="acct-session",
            mode="TEST_MODE",
            paper_trader=trader,
            current_prices={"BTC": 100.0},
        )
        equity_logger.append_equity_snapshot(
            timestamp="2026-01-01 00:01:00 UTC",
            session_id="acct-session",
            mode="TEST_MODE",
            paper_trader=trader,
            current_prices={"BTC": 110.0},
        )

        report = session_report.build_session_report("acct-session")

        assert report["accounting_valid"] is True
        assert validate_accounting_identity(
            starting_equity=report["starting_equity"],
            ending_equity=report["ending_equity"],
            realized_pnl=report["realized_pnl"],
            unrealized_pnl=report["unrealized_pnl"],
            fees_paid=report["fees_paid"],
        )

    def test_slippage_reported_separately_from_realized_pnl(
        self, temp_data_paths, silent_trade_output, monkeypatch
    ):
        monkeypatch.setattr(dashboard_data, "DATA", temp_data_paths)
        monkeypatch.setattr(session_report, "DATA", temp_data_paths)

        trader = PaperTrader(session_id="slip-session", mode="TEST_MODE")
        _buy(trader, price=100.0)
        _sell(trader, price=110.0)

        report = session_report.build_session_report("slip-session")

        assert report["slippage_cost"] > 0.0
        assert report["gross_realized_pnl"] == pytest.approx(
            report["realized_pnl"] + report["slippage_cost"]
        )


class TestShutdownReporting:
    def test_open_positions_preserved_in_shutdown_report(
        self, temp_data_paths, silent_trade_output, monkeypatch
    ):
        monkeypatch.setattr(dashboard_data, "DATA", temp_data_paths)

        trader = PaperTrader(session_id="shutdown-session", mode="TEST_MODE")
        _buy(trader, price=100.0)

        shutdown = SessionShutdownContext(
            shutdown_reason="keyboard_interrupt",
            started_at=trader.started_at,
            ended_at=trader.started_at + 120.0,
            current_prices={"BTC": 105.0, "ETH": 50.0, "SOL": 20.0},
            strategy_profile="contrarian_v1",
            configuration_hash="abc123",
            blocked_proposals_total=trader.blocked_proposals_total,
            blocked_proposals_by_reason=dict(trader.blocked_proposals_by_reason),
        )

        report = session_report.build_session_report(
            "shutdown-session",
            shutdown=shutdown,
            paper_trader=trader,
        )

        assert report["shutdown_reason"] == "keyboard_interrupt"
        assert len(report["open_positions_at_shutdown"]) == 1
        position = report["open_positions_at_shutdown"][0]
        assert position["coin"] == "BTC"
        assert position["quantity"] > 0
        assert position["average_entry"] > 0
        assert position["final_market_price"] == 105.0
        assert position["position_value"] > 0
        assert position["entry_count"] == 1
        assert report["positions_force_closed"] is False
        assert report["force_close_reason"] is None

    def test_force_close_method_closes_open_positions(
        self, temp_data_paths, silent_trade_output
    ):
        trader = PaperTrader(session_id="force-close", mode="TEST_MODE")
        trader.positions["BTC"] = Position(
            quantity=0.5,
            avg_entry_price=100.0,
            entry_count=1,
        )

        trader.force_close_all_positions(
            {"BTC": 110.0},
            "2026-01-01 00:02:00 UTC",
        )

        assert not trader.positions["BTC"].is_open

    def test_force_close_on_shutdown_defaults_to_false(self):
        assert PAPER_TRADING.force_close_on_shutdown is False


class TestBlockedProposalTracking:
    def test_blocked_counters_increment_without_suppressing_records(
        self, temp_data_paths, silent_trade_output
    ):
        trader = PaperTrader()
        trader.cash_balance = 0.0

        _buy(trader, price=100.0)
        _buy(trader, price=100.0, now=1_100.0, timestamp="2026-01-01 00:02:00 UTC")

        assert trader.blocked_proposals_total == 2
        assert trader.blocked_proposals_by_reason["insufficient fake cash"] == 2

    def test_blocked_message_throttling(self, monkeypatch):
        from dataclasses import replace

        import paper_trader as paper_trader_module

        throttled = replace(PAPER_TRADING, blocked_message_throttle_seconds=60)
        monkeypatch.setattr(paper_trader_module, "PAPER_TRADING", throttled)

        throttle_state: dict[str, float] = {}
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            print_skipped_trade_proposal(
                timestamp="t1",
                coin="BTC",
                action="BUY",
                price=100.0,
                signal="DOWN",
                reason="session trade limit reached",
                throttle_state=throttle_state,
            )
            print_skipped_trade_proposal(
                timestamp="t2",
                coin="BTC",
                action="BUY",
                price=100.0,
                signal="DOWN",
                reason="session trade limit reached",
                throttle_state=throttle_state,
            )

        output = buffer.getvalue()
        assert output.count("SKIPPED PAPER TRADE PROPOSAL") == 1

    def test_blocked_counts_in_session_report(
        self, temp_data_paths, silent_trade_output, monkeypatch
    ):
        monkeypatch.setattr(dashboard_data, "DATA", temp_data_paths)
        monkeypatch.setattr(session_report, "DATA", temp_data_paths)

        trader = PaperTrader(session_id="blocked-session", mode="TEST_MODE")
        trader.successful_trades_this_session = trader.session_trade_limit()
        trader.record_blocked_proposal("session trade limit reached")
        trader.record_blocked_proposal("session trade limit reached")

        shutdown = SessionShutdownContext(
            shutdown_reason="keyboard_interrupt",
            blocked_proposals_total=trader.blocked_proposals_total,
            blocked_proposals_by_reason=dict(trader.blocked_proposals_by_reason),
        )

        report = session_report.build_session_report(
            "blocked-session",
            shutdown=shutdown,
            paper_trader=trader,
        )

        assert report["blocked_proposals_total"] == 2
        assert report["blocked_proposals_by_reason"]["session trade limit reached"] == 2
        assert report["session_trade_limit_reached"] is True


class TestFiveMinuteLookback:
    def test_nearest_earlier_sample_within_tolerance(self):
        now = 1_000.0
        history = [(now - 305.0, 100.0), (now - 10.0, 105.0)]

        change = rolling_percent_change(history, 105.0, now, WINDOW_5M_SECONDS)

        assert change == pytest.approx(5.0)

    def test_returns_none_when_insufficient_history(self):
        now = 1_000.0
        history = [(now - 50.0, 100.0)]

        assert rolling_percent_change(history, 100.0, now, WINDOW_5M_SECONDS) is None

    def test_pruning_boundary_does_not_cause_intermittent_na(self):
        now = 1_000.0
        history: dict[str, list[tuple[float, float]]] = {}

        for offset in range(320, -1, -10):
            record_price_history(history, "BTC", 100.0 + offset * 0.01, now - offset)

        change = rolling_percent_change(
            history["BTC"],
            history["BTC"][-1][1],
            now,
            WINDOW_5M_SECONDS,
        )

        assert change is not None

    def test_duplicate_timestamps_use_latest_at_or_before_target(self):
        now = 1_000.0
        target = now - WINDOW_5M_SECONDS
        history = [
            (target - 20.0, 100.0),
            (target, 101.0),
            (target, 102.0),
            (now - 10.0, 110.0),
        ]

        change = rolling_percent_change(history, 110.0, now, WINDOW_5M_SECONDS)

        assert change == pytest.approx(((110.0 - 102.0) / 102.0) * 100)


class TestNoLiveDataModification:
    def test_temp_data_paths_do_not_touch_live_csv(self, temp_data_paths, silent_trade_output):
        repo_root = Path(__file__).resolve().parent.parent
        live_trades = repo_root / "data" / "paper_trades.csv"
        before = live_trades.read_bytes() if live_trades.exists() else None

        trader = PaperTrader(session_id="temp-only", mode="TEST_MODE")
        _buy(trader, price=100.0)

        assert temp_data_paths.paper_trades_csv.exists()
        if before is not None:
            assert live_trades.read_bytes() == before
        else:
            assert not live_trades.exists()

    def test_configuration_hash_is_stable(self):
        snapshot = session.session_config_snapshot(test_mode=True)
        first = session.configuration_hash(snapshot)
        second = session.configuration_hash(snapshot)
        assert first == second
        assert len(first) == 64
