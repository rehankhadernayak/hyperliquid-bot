"""Tests for wider contrarian entry-threshold experiment profiles."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from config import EVIDENCE, SIGNAL
from experiment_metrics import compute_period_metrics
from experiment_runner import run_experiment, save_experiment
from experiment_threshold_comparison import (
    LABEL_INCONCLUSIVE,
    LABEL_LOWER_COSTS_NO_EDGE,
    LABEL_MORE_SELECTIVE_BUT_INSUFFICIENT_DATA,
    LABEL_POSSIBLE_IMPROVEMENT,
    LABEL_WORSE,
    THRESHOLD_COMPARISON_LABEL_RULES,
    build_threshold_comparison,
    classify_threshold_variant,
)
from experiment_trader import ExperimentPaperTrader
from profile_validation import validate_thresholds
from signals import signal_from_change
from strategies.contrarian_v1 import ContrarianV1Strategy
from strategy_profiles import STRATEGY_PROFILES, get_profile
from strategy_types import ProfileTradeRecord, StrategyContext
from tests.conftest import LIVE_PAPER_TRADES, LIVE_PRICE_LOG
from tests.test_experiments import _write_sample_price_log


class TestContrarianV1TestThresholdsUnchanged:
    def test_test_profile_thresholds_match_signal_test_mode(self):
        profile = get_profile("contrarian_v1_test")
        assert profile.thresholds == SIGNAL.test_mode
        assert profile.thresholds.down_pct == -0.05
        assert profile.thresholds.strong_down_pct == -0.10
        assert profile.thresholds.up_pct == 0.05
        assert profile.thresholds.strong_up_pct == 0.10

    def test_trade_size_unchanged(self):
        profile = get_profile("contrarian_v1_test")
        assert profile.trade_size_usdc == 100.0


class TestEntryThresholdProfiles:
    def test_medium_profile_values(self):
        profile = get_profile("contrarian_v1_entry_medium")
        assert profile.thresholds.down_pct == -0.15
        assert profile.thresholds.strong_down_pct == -0.25
        assert profile.thresholds.up_pct == SIGNAL.test_mode.up_pct
        assert profile.thresholds.strong_up_pct == SIGNAL.test_mode.strong_up_pct
        assert profile.trade_size_usdc == 100.0

    def test_selective_profile_values(self):
        profile = get_profile("contrarian_v1_entry_selective")
        assert profile.thresholds.down_pct == -0.25
        assert profile.thresholds.strong_down_pct == -0.40
        assert profile.thresholds.up_pct == SIGNAL.test_mode.up_pct
        assert profile.thresholds.strong_up_pct == SIGNAL.test_mode.strong_up_pct
        assert profile.trade_size_usdc == 100.0

    def test_other_settings_match_test_profile(self):
        baseline = get_profile("contrarian_v1_test")
        for profile_id in ("contrarian_v1_entry_medium", "contrarian_v1_entry_selective"):
            variant = get_profile(profile_id)
            assert variant.cooldown_seconds == baseline.cooldown_seconds
            assert variant.trading_fee_pct == baseline.trading_fee_pct
            assert variant.slippage_pct == baseline.slippage_pct
            assert variant.max_allocation_per_coin_usdc == baseline.max_allocation_per_coin_usdc
            assert variant.max_total_open_exposure_usdc == baseline.max_total_open_exposure_usdc
            assert variant.lookback_seconds == baseline.lookback_seconds

    def test_profiles_are_immutable(self):
        profile = get_profile("contrarian_v1_entry_medium")
        with pytest.raises(Exception):
            profile.trade_size_usdc = 200.0


class TestThresholdValidation:
    def test_valid_thresholds_pass(self):
        validate_thresholds(SIGNAL.test_mode)
        validate_thresholds(get_profile("contrarian_v1_entry_medium").thresholds)
        validate_thresholds(get_profile("contrarian_v1_entry_selective").thresholds)

    def test_invalid_down_order_fails(self):
        thresholds = replace(SIGNAL.test_mode, strong_down_pct=-0.05, down_pct=-0.10)
        with pytest.raises(ValueError, match="strong_down_pct < down_pct < 0"):
            validate_thresholds(thresholds)

    def test_invalid_up_order_fails(self):
        thresholds = replace(SIGNAL.test_mode, up_pct=0.50, strong_up_pct=0.25)
        with pytest.raises(ValueError, match="0 < up_pct < strong_up_pct"):
            validate_thresholds(thresholds)

    def test_non_numeric_threshold_fails(self):
        thresholds = replace(SIGNAL.test_mode, down_pct="bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="must be numeric"):
            validate_thresholds(thresholds)

    def test_invalid_profile_fails_before_execution(self):
        bad_profile = replace(
            get_profile("contrarian_v1_test"),
            thresholds=replace(SIGNAL.test_mode, down_pct=0.05),
        )
        with pytest.raises(ValueError):
            from profile_validation import validate_profile_fields

            validate_profile_fields(bad_profile)


class TestWiderThresholdSignalCounts:
    def test_wider_thresholds_produce_fewer_or_equal_buy_signals(self):
        strategy = ContrarianV1Strategy()
        test_profile = get_profile("contrarian_v1_test")
        medium_profile = get_profile("contrarian_v1_entry_medium")
        selective_profile = get_profile("contrarian_v1_entry_selective")
        trader = ExperimentPaperTrader(profile=test_profile)

        changes = [-0.06, -0.11, -0.16, -0.26, -0.41, 0.06, 0.11, 0.0]
        test_buys = 0
        medium_buys = 0
        selective_buys = 0

        for change in changes:
            for profile, counter_name in (
                (test_profile, "test"),
                (medium_profile, "medium"),
                (selective_profile, "selective"),
            ):
                context = StrategyContext(
                    coin="BTC",
                    current_price=100.0,
                    price_history={
                        "BTC": [(0.0, 100.0 / (1 + change / 100)), (60.0, 100.0)]
                    },
                    timestamp="2026-01-01 00:01:00 UTC",
                    now=60.0,
                    trader=trader,
                    thresholds=profile.thresholds,
                    lookback_seconds=60,
                )
                signal = strategy.evaluate(context).signal
                assert signal == signal_from_change(change, profile.thresholds)
                if signal in ("DOWN", "STRONG_DOWN"):
                    if counter_name == "test":
                        test_buys += 1
                    elif counter_name == "medium":
                        medium_buys += 1
                    else:
                        selective_buys += 1

        assert medium_buys <= test_buys
        assert selective_buys <= medium_buys


class TestCostsPctOfGrossProfit:
    def test_zero_gross_profit_returns_na(self):
        profile = get_profile("hold_baseline")
        trader = ExperimentPaperTrader(profile=profile)
        limits = profile.trading_limits()
        metrics = compute_period_metrics(
            trader,
            [limits.starting_balance_usdc],
            {},
            period_label="combined",
            limits=limits,
        )
        assert metrics["costs_pct_of_gross_profit"] == "N/A"

    def test_negative_gross_profit_returns_na(self):
        profile = get_profile("contrarian_v1_test")
        trader = ExperimentPaperTrader(profile=profile)
        limits = profile.trading_limits()
        trader.trade_log = [
            ProfileTradeRecord(
                timestamp="t",
                coin="BTC",
                action="SELL",
                signal="UP",
                mid_price=100.0,
                execution_price=100.0,
                quantity=1.0,
                trade_value=100.0,
                fee=1.0,
                slippage=0.5,
                realized_pnl=-5.0,
                entry_count=1,
                cash_balance=995.0,
                open_exposure=0.0,
                equity=995.0,
            )
        ]
        trader.total_fees_paid = 1.0
        trader.total_slippage_cost = 0.5
        metrics = compute_period_metrics(
            trader,
            [1000.0, 995.0],
            {"BTC": 100.0},
            period_label="combined",
            limits=limits,
        )
        assert metrics["costs_pct_of_gross_profit"] == "N/A"

    def test_positive_gross_profit_returns_percentage(self):
        profile = get_profile("contrarian_v1_test")
        trader = ExperimentPaperTrader(profile=profile)
        limits = profile.trading_limits()
        trader.trade_log = [
            ProfileTradeRecord(
                timestamp="t",
                coin="BTC",
                action="SELL",
                signal="UP",
                mid_price=100.0,
                execution_price=100.0,
                quantity=1.0,
                trade_value=100.0,
                fee=1.0,
                slippage=0.5,
                realized_pnl=10.0,
                entry_count=1,
                cash_balance=1010.0,
                open_exposure=0.0,
                equity=1010.0,
            )
        ]
        trader.total_fees_paid = 1.0
        trader.total_slippage_cost = 0.5
        trader.realized_pnl = 10.0
        metrics = compute_period_metrics(
            trader,
            [1000.0, 1010.0],
            {"BTC": 100.0},
            period_label="combined",
            limits=limits,
        )
        assert metrics["costs_pct_of_gross_profit"] == pytest.approx(15.0)


class TestThresholdComparisonLabels:
    def test_label_rules_documented(self):
        assert LABEL_MORE_SELECTIVE_BUT_INSUFFICIENT_DATA in THRESHOLD_COMPARISON_LABEL_RULES
        assert LABEL_LOWER_COSTS_NO_EDGE in THRESHOLD_COMPARISON_LABEL_RULES
        assert LABEL_POSSIBLE_IMPROVEMENT in THRESHOLD_COMPARISON_LABEL_RULES
        assert LABEL_WORSE in THRESHOLD_COMPARISON_LABEL_RULES
        assert LABEL_INCONCLUSIVE in THRESHOLD_COMPARISON_LABEL_RULES

    def test_more_selective_insufficient_data_label(self):
        baseline = {"completed_trades": 10, "net_return_pct": 1.0, "fees_paid": 2.0, "slippage_cost": 1.0}
        variant = {"completed_trades": 2, "net_return_pct": 0.5, "fees_paid": 0.5, "slippage_cost": 0.2}
        label = classify_threshold_variant(
            baseline,
            variant,
            evidence_assessment="INSUFFICIENT_DATA",
        )
        assert label == LABEL_MORE_SELECTIVE_BUT_INSUFFICIENT_DATA

    def test_lower_costs_no_edge_label(self):
        baseline = {
            "completed_trades": 10,
            "net_return_pct": 1.0,
            "fees_paid": 5.0,
            "slippage_cost": 2.0,
            "profit_factor": 1.2,
        }
        variant = {
            "completed_trades": 8,
            "net_return_pct": 0.5,
            "fees_paid": 2.0,
            "slippage_cost": 1.0,
            "profit_factor": 1.1,
        }
        label = classify_threshold_variant(
            baseline,
            variant,
            evidence_assessment="WEAK",
        )
        assert label == LABEL_LOWER_COSTS_NO_EDGE

    def test_possible_improvement_label(self):
        baseline = {
            "completed_trades": 10,
            "net_return_pct": 0.5,
            "fees_paid": 2.0,
            "slippage_cost": 1.0,
            "profit_factor": 1.0,
        }
        variant = {
            "completed_trades": 8,
            "net_return_pct": 1.5,
            "fees_paid": 2.0,
            "slippage_cost": 1.0,
            "profit_factor": 1.3,
        }
        label = classify_threshold_variant(
            baseline,
            variant,
            evidence_assessment="WEAK",
        )
        assert label == LABEL_POSSIBLE_IMPROVEMENT

    def test_worse_label_on_reject(self):
        baseline = {"completed_trades": 10, "net_return_pct": 1.0, "fees_paid": 2.0, "slippage_cost": 1.0}
        variant = {"completed_trades": 8, "net_return_pct": 0.5, "fees_paid": 1.0, "slippage_cost": 0.5}
        label = classify_threshold_variant(
            baseline,
            variant,
            evidence_assessment="REJECT",
        )
        assert label == LABEL_WORSE


class TestEntryThresholdExperimentRunner:
    def test_identical_input_data_across_profiles(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        profiles = [
            "contrarian_v1_test",
            "contrarian_v1_entry_medium",
            "contrarian_v1_entry_selective",
            "hold_baseline",
        ]
        results = run_experiment(profiles, csv_path=price_log)
        assert results["timestamp_count"] == 6
        for profile_id in profiles:
            assert profile_id in results["profile_results"]

    def test_identical_fees_and_slippage_across_profiles(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        results = run_experiment(
            ["contrarian_v1_test", "contrarian_v1_entry_medium"],
            csv_path=price_log,
        )
        baseline = results["profile_results"]["contrarian_v1_test"]["profile"]
        medium = results["profile_results"]["contrarian_v1_entry_medium"]["profile"]
        assert baseline["trading_fee_pct"] == medium["trading_fee_pct"]
        assert baseline["slippage_pct"] == medium["slippage_pct"]

    def test_threshold_comparison_in_experiment_json(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        output_dir = tmp_path / "experiments"
        profiles = [
            "contrarian_v1_test",
            "contrarian_v1_entry_medium",
            "contrarian_v1_entry_selective",
            "hold_baseline",
        ]
        results = run_experiment(profiles, csv_path=price_log)
        path = save_experiment(results, output_dir=output_dir)
        payload = json.loads(path.read_text())

        assert "threshold_comparison" in payload
        comparison = payload["threshold_comparison"]
        assert comparison["baseline_profile"] == "contrarian_v1_test"
        assert "contrarian_v1_entry_medium" in comparison["comparisons"]
        assert "contrarian_v1_entry_selective" in comparison["comparisons"]
        assert "label_rules" in comparison

    def test_no_live_data_writes(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)

        live_mtime_before = LIVE_PAPER_TRADES.stat().st_mtime if LIVE_PAPER_TRADES.exists() else None
        live_price_before = LIVE_PRICE_LOG.stat().st_mtime if LIVE_PRICE_LOG.exists() else None

        run_experiment(
            [
                "contrarian_v1_test",
                "contrarian_v1_entry_medium",
                "contrarian_v1_entry_selective",
                "hold_baseline",
            ],
            csv_path=price_log,
        )

        if live_mtime_before is not None:
            assert LIVE_PAPER_TRADES.stat().st_mtime == live_mtime_before
        if live_price_before is not None:
            assert LIVE_PRICE_LOG.stat().st_mtime == live_price_before

    def test_no_network_calls(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)

        with patch("urllib.request.urlopen") as mock_urlopen:
            run_experiment(["contrarian_v1_test", "contrarian_v1_entry_medium"], csv_path=price_log)
            mock_urlopen.assert_not_called()

    def test_wider_thresholds_reduce_or_equal_trades(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        results = run_experiment(
            [
                "contrarian_v1_test",
                "contrarian_v1_entry_medium",
                "contrarian_v1_entry_selective",
            ],
            csv_path=price_log,
        )
        test_buys = results["profile_results"]["contrarian_v1_test"]["combined"]["total_buys"]
        medium_buys = results["profile_results"]["contrarian_v1_entry_medium"]["combined"]["total_buys"]
        selective_buys = results["profile_results"]["contrarian_v1_entry_selective"]["combined"]["total_buys"]
        assert medium_buys <= test_buys
        assert selective_buys <= medium_buys

    def test_build_threshold_comparison_returns_none_without_baseline(self):
        profile_results = {
            "hold_baseline": {
                "out_of_sample": {"completed_trades": 0},
                "combined": {"completed_trades": 0},
                "evidence_assessment": "INSUFFICIENT_DATA",
            }
        }
        assert build_threshold_comparison(profile_results) is None

    def test_required_profiles_exist(self):
        assert "contrarian_v1_entry_medium" in STRATEGY_PROFILES
        assert "contrarian_v1_entry_selective" in STRATEGY_PROFILES
