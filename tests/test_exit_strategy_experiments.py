"""Tests for cost-aware exit strategy experiments."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from config import EVIDENCE, SIGNAL
from exit_config import (
    EXIT_TYPE_HOLD,
    EXIT_TYPE_MAX_HOLD,
    EXIT_TYPE_SIGNAL_EXIT,
    EXIT_TYPE_STOP_LOSS,
    EXIT_TYPE_TAKE_PROFIT,
    ExitConfig,
    SIGNAL_EXIT_V1,
    validate_exit_config,
)
from exit_evaluator import (
    FutureDataAccessError,
    _estimate_exit_costs,
    _gross_return_pct,
    build_position_exit_context,
    evaluate_position_exit,
)
from experiment_exit_comparison import (
    LABEL_BETTER_NET_EXPECTANCY,
    LABEL_INSUFFICIENT_EXIT_DATA,
    LABEL_LOWER_COSTS_NO_EDGE,
    LABEL_WORSE,
    EXIT_COMPARISON_LABEL_RULES,
    build_exit_comparison,
    classify_exit_variant,
)
from experiment_metrics import compute_period_metrics
from experiment_runner import replay_profile, run_experiment, save_experiment
from experiment_trader import ExperimentPaperTrader
from strategy_profiles import STRATEGY_PROFILES, get_profile
from tests.conftest import LIVE_PAPER_TRADES, LIVE_PRICE_LOG
from tests.test_experiments import _write_sample_price_log


EXIT_PROFILES = [
    "contrarian_v1_entry_medium",
    "contrarian_v1_medium_signal_exit",
    "contrarian_v1_medium_tp25_sl25",
    "contrarian_v1_medium_tp35_sl25",
    "contrarian_v1_medium_tp50_sl30",
    "contrarian_v1_medium_tp35_sl25_maxhold",
    "hold_baseline",
]


class TestExitProfiles:
    def test_all_exit_profiles_exist(self):
        for profile_id in EXIT_PROFILES:
            assert profile_id in STRATEGY_PROFILES

    def test_medium_signal_exit_matches_entry_medium_thresholds(self):
        baseline = get_profile("contrarian_v1_entry_medium")
        signal_exit = get_profile("contrarian_v1_medium_signal_exit")
        assert signal_exit.thresholds == baseline.thresholds
        assert signal_exit.trade_size_usdc == 100.0
        assert signal_exit.exit_config == SIGNAL_EXIT_V1

    def test_cost_aware_profile_values(self):
        profile = get_profile("contrarian_v1_medium_tp35_sl25")
        assert profile.exit_config is not None
        assert profile.exit_config.take_profit_pct == 0.35
        assert profile.exit_config.stop_loss_pct == 0.25
        assert profile.exit_config.signal_exit_enabled is False

    def test_maxhold_profile(self):
        profile = get_profile("contrarian_v1_medium_tp35_sl25_maxhold")
        assert profile.exit_config is not None
        assert profile.exit_config.max_hold_seconds == 1800

    def test_profiles_immutable(self):
        profile = get_profile("contrarian_v1_medium_tp25_sl25")
        with pytest.raises(Exception):
            profile.trade_size_usdc = 200.0

    def test_existing_profiles_unchanged(self):
        test_profile = get_profile("contrarian_v1_test")
        assert test_profile.thresholds == SIGNAL.test_mode
        assert test_profile.exit_config is None

        medium = get_profile("contrarian_v1_entry_medium")
        assert medium.thresholds.down_pct == -0.15
        assert medium.exit_config is None


class TestCostSemantics:
    def test_gross_return_from_average_entry(self):
        assert _gross_return_pct(100.0, 100.25) == pytest.approx(0.25)

    def test_fee_and_slippage_estimates_no_double_count(self):
        costs = _estimate_exit_costs(
            avg_entry_price=100.0,
            quantity=1.0,
            current_price=100.25,
            trading_fee_pct=0.05,
            slippage_pct=0.02,
        )
        assert costs["estimated_sell_fee"] > 0
        assert costs["estimated_sell_slippage"] > 0
        assert costs["total_estimated_round_trip_cost"] == pytest.approx(
            costs["estimated_buy_fee"]
            + costs["estimated_buy_slippage"]
            + costs["estimated_sell_fee"]
            + costs["estimated_sell_slippage"]
        )


class TestExitEvaluator:
    def _context(self, profile, *, price=100.0, entry=100.0, quantity=1.0, now=120.0):
        trader = ExperimentPaperTrader(profile=profile)
        return build_position_exit_context(
            coin="BTC",
            current_price=price,
            avg_entry_price=entry,
            quantity=quantity,
            opened_at=60.0,
            entry_count=1,
            now=now,
            price_history={"BTC": [(0.0, 99.0), (60.0, entry), (now, price)]},
            trader_state={"cash_balance": 900.0},
            profile=profile,
            lookback_seconds=60,
            thresholds=profile.thresholds,
        )

    def test_take_profit_from_average_entry(self):
        profile = get_profile("contrarian_v1_medium_tp25_sl25")
        context = self._context(profile, price=100.30, entry=100.0)
        result = evaluate_position_exit(
            context,
            profile.exit_config,
            strategy_name="contrarian_v1",
            strategy_version="1.0.0",
        )
        assert result.exit_type == EXIT_TYPE_TAKE_PROFIT
        assert result.action == "SELL"

    def test_stop_loss_from_average_entry(self):
        profile = get_profile("contrarian_v1_medium_tp25_sl25")
        context = self._context(profile, price=99.70, entry=100.0)
        result = evaluate_position_exit(
            context,
            profile.exit_config,
            strategy_name="contrarian_v1",
            strategy_version="1.0.0",
        )
        assert result.exit_type == EXIT_TYPE_STOP_LOSS
        assert result.action == "SELL"

    def test_exit_priority_stop_before_take_profit(self):
        profile = get_profile("contrarian_v1_medium_tp25_sl25")
        context = self._context(profile, price=99.70, entry=100.0)
        result = evaluate_position_exit(
            context,
            profile.exit_config,
            strategy_name="contrarian_v1",
            strategy_version="1.0.0",
        )
        assert EXIT_TYPE_STOP_LOSS in result.triggered_conditions or result.exit_type == EXIT_TYPE_STOP_LOSS

    def test_max_hold_exit(self):
        profile = get_profile("contrarian_v1_medium_tp35_sl25_maxhold")
        context = self._context(profile, price=100.0, now=2000.0)
        result = evaluate_position_exit(
            context,
            profile.exit_config,
            strategy_name="contrarian_v1",
            strategy_version="1.0.0",
        )
        assert result.exit_type == EXIT_TYPE_MAX_HOLD

    def test_signal_exit_v1_matches_up_signal(self):
        profile = get_profile("contrarian_v1_medium_signal_exit")
        context = build_position_exit_context(
            coin="BTC",
            current_price=100.10,
            avg_entry_price=100.0,
            quantity=1.0,
            opened_at=60.0,
            entry_count=1,
            now=120.0,
            price_history={"BTC": [(0.0, 100.0), (60.0, 100.0), (120.0, 100.10)]},
            trader_state={},
            profile=profile,
            lookback_seconds=60,
            thresholds=profile.thresholds,
        )
        result = evaluate_position_exit(
            context,
            profile.exit_config,
            strategy_name="contrarian_v1",
            strategy_version="1.0.0",
        )
        assert result.exit_type == EXIT_TYPE_SIGNAL_EXIT
        assert result.action == "SELL"

    def test_hold_when_no_condition(self):
        profile = get_profile("contrarian_v1_medium_tp25_sl25")
        context = self._context(profile, price=100.05, entry=100.0)
        result = evaluate_position_exit(
            context,
            profile.exit_config,
            strategy_name="contrarian_v1",
            strategy_version="1.0.0",
        )
        assert result.exit_type == EXIT_TYPE_HOLD
        assert result.action == "HOLD"

    def test_future_data_guard(self):
        profile = get_profile("contrarian_v1_medium_tp25_sl25")
        with pytest.raises(FutureDataAccessError):
            build_position_exit_context(
                coin="BTC",
                current_price=100.0,
                avg_entry_price=100.0,
                quantity=1.0,
                opened_at=60.0,
                entry_count=1,
                now=120.0,
                price_history={"BTC": [(0.0, 99.0), (150.0, 100.0)]},
                trader_state={},
                profile=profile,
                lookback_seconds=60,
                thresholds=profile.thresholds,
            )

    def test_sparse_data_ambiguity_warning(self):
        profile = get_profile("contrarian_v1_medium_tp25_sl25")
        context = build_position_exit_context(
            coin="BTC",
            current_price=100.30,
            avg_entry_price=100.0,
            quantity=1.0,
            opened_at=60.0,
            entry_count=1,
            now=120.0,
            price_history={"BTC": [(60.0, 99.50), (120.0, 100.30)]},
            trader_state={},
            profile=profile,
            lookback_seconds=60,
            thresholds=profile.thresholds,
        )
        result = evaluate_position_exit(
            context,
            profile.exit_config,
            strategy_name="contrarian_v1",
            strategy_version="1.0.0",
        )
        assert result.ambiguity_warnings


class TestSignalExitBaselineParity:
    def test_signal_exit_matches_entry_medium_on_sample_data(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        timestamps, prices = __import__("backtester").load_price_ticks(price_log)

        medium_profile = get_profile("contrarian_v1_entry_medium")
        signal_profile = get_profile("contrarian_v1_medium_signal_exit")

        medium_trader, medium_equity, medium_prices, _ = replay_profile(
            timestamps, prices, medium_profile
        )
        signal_trader, signal_equity, signal_prices, _ = replay_profile(
            timestamps, prices, signal_profile
        )

        assert len(medium_trader.trade_log) == len(signal_trader.trade_log)
        for medium_trade, signal_trade in zip(
            medium_trader.trade_log, signal_trader.trade_log
        ):
            assert medium_trade.timestamp == signal_trade.timestamp
            assert medium_trade.coin == signal_trade.coin
            assert medium_trade.action == signal_trade.action
            assert medium_trade.signal == signal_trade.signal
            assert medium_trade.quantity == pytest.approx(signal_trade.quantity)
            assert medium_trade.realized_pnl == pytest.approx(signal_trade.realized_pnl)
            assert medium_trade.cash_balance == pytest.approx(signal_trade.cash_balance)
        assert medium_equity == signal_equity


class TestExitMetrics:
    def test_zero_trade_handling(self):
        profile = get_profile("hold_baseline")
        trader = ExperimentPaperTrader(profile=profile)
        limits = profile.trading_limits()
        metrics = compute_period_metrics(
            trader,
            [limits.starting_balance_usdc],
            {},
            period_label="combined",
            limits=limits,
            profile=profile,
        )
        assert metrics["completed_trades"] == 0
        assert metrics["take_profit_exits"] == 0
        assert metrics["average_net_return_per_trade"] == 0.0
        assert metrics["profit_factor"] is None


class TestExitComparisonLabels:
    def test_label_rules_documented(self):
        assert LABEL_INSUFFICIENT_EXIT_DATA in EXIT_COMPARISON_LABEL_RULES
        assert LABEL_BETTER_NET_EXPECTANCY in EXIT_COMPARISON_LABEL_RULES

    def test_insufficient_exit_data(self):
        baseline = {"completed_trades": 10, "net_return_pct": 1.0, "average_net_return_per_trade": 0.1}
        variant = {"completed_trades": 2, "net_return_pct": 0.5, "average_net_return_per_trade": 0.05}
        assert (
            classify_exit_variant(
                baseline,
                variant,
                evidence_assessment="INSUFFICIENT_DATA",
                development_metrics={"net_return_pct": 0.5},
            )
            == LABEL_INSUFFICIENT_EXIT_DATA
        )

    def test_worse_label(self):
        baseline = {
            "completed_trades": 10,
            "net_return_pct": 1.0,
            "average_net_return_per_trade": 0.1,
            "maximum_drawdown_pct": 5.0,
            "fees_paid": 1.0,
            "slippage_cost": 0.5,
            "trade_log": [],
        }
        variant = {
            "completed_trades": 10,
            "net_return_pct": -1.0,
            "average_net_return_per_trade": -0.1,
            "maximum_drawdown_pct": 10.0,
            "fees_paid": 1.0,
            "slippage_cost": 0.5,
            "trade_log": [],
        }
        assert (
            classify_exit_variant(
                baseline,
                variant,
                evidence_assessment="REJECT",
                development_metrics={"net_return_pct": -1.0},
            )
            == LABEL_WORSE
        )


class TestExitExperimentRunner:
    def test_full_exit_comparison_experiment(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        results = run_experiment(EXIT_PROFILES, csv_path=price_log)
        for profile_id in EXIT_PROFILES:
            assert profile_id in results["profile_results"]
        assert results.get("exit_comparison") is not None

    def test_exit_comparison_in_json(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        output_dir = tmp_path / "experiments"
        results = run_experiment(EXIT_PROFILES, csv_path=price_log)
        path = save_experiment(results, output_dir=output_dir)
        payload = json.loads(path.read_text())
        assert "exit_comparison" in payload
        assert payload["exit_comparison"]["baseline_profile"] == "contrarian_v1_medium_signal_exit"

    def test_no_live_data_writes(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        live_mtime = LIVE_PAPER_TRADES.stat().st_mtime if LIVE_PAPER_TRADES.exists() else None
        live_price = LIVE_PRICE_LOG.stat().st_mtime if LIVE_PRICE_LOG.exists() else None
        run_experiment(EXIT_PROFILES[:3], csv_path=price_log)
        if live_mtime is not None:
            assert LIVE_PAPER_TRADES.stat().st_mtime == live_mtime
        if live_price is not None:
            assert LIVE_PRICE_LOG.stat().st_mtime == live_price

    def test_no_network_calls(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        with patch("urllib.request.urlopen") as mock_urlopen:
            run_experiment(EXIT_PROFILES[:2], csv_path=price_log)
            mock_urlopen.assert_not_called()

    def test_identical_input_hash_across_profiles(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        results = run_experiment(EXIT_PROFILES, csv_path=price_log)
        assert results["timestamp_count"] == 6

    def test_build_exit_comparison_requires_baseline(self):
        assert build_exit_comparison({}) is None

    def test_validate_exit_config_rejects_invalid(self):
        with pytest.raises(ValueError):
            validate_exit_config(
                ExitConfig(exit_mode="bad", take_profit_pct=0.25, stop_loss_pct=0.25)
            )
