"""Tests for the v8.2 strategy experiment framework."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import EVIDENCE_INSUFFICIENT_DATA, EVIDENCE_REJECT, SIGNAL
from experiment_evidence import assess_evidence
from experiment_metrics import compute_period_metrics
from experiment_overfitting import collect_overfitting_warnings
from experiment_runner import hash_input_file, replay_profile, run_experiment, save_experiment
from experiment_split import split_timestamps_chronologically, walk_forward_folds
from experiment_trader import ExperimentPaperTrader
from signals import apply_signal_thresholds, signal_from_1m_change, signal_from_change
from strategies.contrarian_v1 import ContrarianV1Strategy
from strategy_profiles import STRATEGY_PROFILES, StrategyProfile, get_profile
from strategy_types import StrategyContext
from tests.conftest import LIVE_PAPER_TRADES, LIVE_PRICE_LOG


def _write_sample_price_log(path: Path, *, rows: list[str] | None = None) -> None:
    default_rows = [
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
        "2026-01-01 00:04:00 UTC,BTC,98.0",
        "2026-01-01 00:04:00 UTC,ETH,98.0",
        "2026-01-01 00:04:00 UTC,SOL,98.0",
        "2026-01-01 00:05:00 UTC,BTC,102.0",
        "2026-01-01 00:05:00 UTC,ETH,102.0",
        "2026-01-01 00:05:00 UTC,SOL,102.0",
    ]
    path.write_text("\n".join(rows or default_rows) + "\n")


class TestContrarianStrategyBehavior:
    """Existing contrarian threshold behavior remains unchanged."""

    @pytest.fixture(autouse=True)
    def use_production_thresholds(self):
        apply_signal_thresholds(test_mode=False)

    def test_signal_from_1m_change_matches_signal_from_change(self):
        changes = [None, 0.0, 0.26, 0.51, -0.26, -0.51]
        for change in changes:
            assert signal_from_1m_change(change) == signal_from_change(
                change, SIGNAL.production
            )

    def test_contrarian_v1_strategy_matches_legacy_signals(self):
        strategy = ContrarianV1Strategy()
        profile = get_profile("contrarian_v1_production")
        trader = ExperimentPaperTrader(profile=profile)

        for change in (0.26, -0.26, 0.51, -0.51, 0.0):
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
            result = strategy.evaluate(context)
            assert result.signal == signal_from_change(change, profile.thresholds)
            assert result.strategy_name == "contrarian_v1"
            assert result.strategy_version == "1.0.0"


class TestStrategyProfiles:
    def test_profiles_are_immutable(self):
        profile = get_profile("contrarian_v1_test")
        with pytest.raises(Exception):
            profile.profile_id = "changed"

    def test_required_profiles_exist(self):
        assert "contrarian_v1_test" in STRATEGY_PROFILES
        assert "contrarian_v1_production" in STRATEGY_PROFILES
        assert "hold_baseline" in STRATEGY_PROFILES

    def test_profile_to_dict_is_json_serializable(self):
        payload = get_profile("contrarian_v1_test").to_dict()
        json.dumps(payload)


class TestChronologicalSplit:
    def test_split_preserves_order_and_ratio(self):
        timestamps = [f"2026-01-01 00:{idx:02d}:00 UTC" for idx in range(10)]
        split = split_timestamps_chronologically(timestamps, development_pct=0.7)

        assert split.development == tuple(timestamps[:7])
        assert split.out_of_sample == tuple(timestamps[7:])
        assert list(split.combined) == timestamps

    def test_walk_forward_has_no_future_data_in_development(self):
        timestamps = [f"t{idx}" for idx in range(20)]
        folds = walk_forward_folds(
            timestamps,
            min_development_timestamps=5,
            evaluation_window=3,
        )
        assert folds
        for fold in folds:
            dev_end = timestamps.index(fold.development[-1])
            eval_start = timestamps.index(fold.evaluation[0])
            assert eval_start == dev_end + 1 or eval_start > dev_end


class TestExperimentTraderIsolation:
    def test_isolated_traders_do_not_share_state(self):
        profile = get_profile("contrarian_v1_test")
        trader_a = ExperimentPaperTrader(profile=profile)
        trader_b = ExperimentPaperTrader(profile=profile)

        trader_a.cash_balance -= 50.0
        assert trader_b.cash_balance == profile.starting_balance_usdc

    def test_fees_and_slippage_use_profile_settings(self):
        profile = get_profile("contrarian_v1_test")
        trader = ExperimentPaperTrader(profile=profile)

        buy_fee = trader.buy_fee(100.0)
        assert buy_fee == pytest.approx(100.0 * (profile.trading_fee_pct / 100.0))

        buy_price = trader.execution_price("BUY", 100.0)
        assert buy_price > 100.0


class TestExperimentMetrics:
    def test_zero_trade_metrics_are_safe(self):
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

        assert metrics["completed_trades"] == 0
        assert metrics["profit_factor"] is None
        assert metrics["win_rate_pct"] == 0.0
        assert metrics["return_per_completed_trade"] == 0.0

    def test_profit_factor_without_losses_is_not_misleading_number(self):
        profile = get_profile("contrarian_v1_test")
        trader = ExperimentPaperTrader(profile=profile)
        limits = profile.trading_limits()
        from strategy_types import ProfileTradeRecord

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
                fee=0.0,
                slippage=0.0,
                realized_pnl=10.0,
                entry_count=1,
                cash_balance=1000.0,
                open_exposure=0.0,
                equity=1010.0,
            )
        ]
        trader.realized_pnl = 10.0

        metrics = compute_period_metrics(
            trader,
            [1000.0, 1010.0],
            {"BTC": 100.0},
            period_label="combined",
            limits=limits,
        )
        assert metrics["profit_factor"] == "inf"


class TestEvidenceAndOverfitting:
    def test_insufficient_data_label(self):
        dev = {"completed_trades": 0, "net_return_pct": 0.0, "maximum_drawdown_pct": 0.0}
        oos = {"completed_trades": 1, "net_return_pct": 0.0, "maximum_drawdown_pct": 0.0}
        assert assess_evidence(dev, oos) == EVIDENCE_INSUFFICIENT_DATA

    def test_reject_on_high_drawdown(self):
        dev = {"completed_trades": 20, "net_return_pct": 5.0, "maximum_drawdown_pct": 5.0}
        oos = {
            "completed_trades": 20,
            "net_return_pct": 1.0,
            "maximum_drawdown_pct": 30.0,
            "gross_profit": 10.0,
            "fees_paid": 1.0,
            "trade_log": [],
            "results_by_coin": {},
            "profit_factor": 0.5,
        }
        assert assess_evidence(dev, oos) == EVIDENCE_REJECT

    def test_overfitting_warnings_include_low_trade_count(self):
        warnings = collect_overfitting_warnings(
            development={"net_return_pct": 1.0},
            out_of_sample={"completed_trades": 1, "net_return_pct": 0.0, "gross_profit": 0.0},
            combined={"completed_trades": 1},
            profile_count=2,
            split_identical=False,
            used_future_data=False,
        )
        assert any("Too few completed trades" in warning for warning in warnings)


class TestExperimentRunner:
    def test_identical_input_data_hash_across_profiles(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        expected_hash = hash_input_file(price_log)

        results = run_experiment(
            ["contrarian_v1_test", "hold_baseline"],
            csv_path=price_log,
        )
        assert results["input_data_hash"] == expected_hash

    def test_profiles_use_same_timestamps(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)

        results = run_experiment(
            ["contrarian_v1_test", "contrarian_v1_production"],
            csv_path=price_log,
        )
        assert results["timestamp_count"] == 6

    def test_hold_baseline_makes_no_trades(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)

        results = run_experiment(["hold_baseline"], csv_path=price_log)
        combined = results["profile_results"]["hold_baseline"]["combined"]
        assert combined["total_buys"] == 0
        assert combined["total_sells"] == 0

    def test_experiment_json_output_structure(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        output_dir = tmp_path / "experiments"

        results = run_experiment(["contrarian_v1_test"], csv_path=price_log)
        path = save_experiment(results, output_dir=output_dir)

        assert path.exists()
        payload = json.loads(path.read_text())
        assert payload["experiment_id"] == results["experiment_id"]
        assert "profile_results" in payload
        assert "profile" in payload["profile_results"]["contrarian_v1_test"]
        assert "development" in payload["profile_results"]["contrarian_v1_test"]
        assert "out_of_sample" in payload["profile_results"]["contrarian_v1_test"]
        assert "combined" in payload["profile_results"]["contrarian_v1_test"]

    def test_experiment_does_not_write_live_files(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)

        live_mtime_before = LIVE_PAPER_TRADES.stat().st_mtime if LIVE_PAPER_TRADES.exists() else None
        live_price_before = LIVE_PRICE_LOG.stat().st_mtime if LIVE_PRICE_LOG.exists() else None

        run_experiment(["contrarian_v1_test"], csv_path=price_log)

        if live_mtime_before is not None:
            assert LIVE_PAPER_TRADES.stat().st_mtime == live_mtime_before
        if live_price_before is not None:
            assert LIVE_PRICE_LOG.stat().st_mtime == live_price_before

    def test_oos_period_replay_does_not_use_future_prices(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)

        from backtester import load_price_ticks

        timestamps, prices_by_timestamp = load_price_ticks(price_log)
        split = split_timestamps_chronologically(timestamps, development_pct=0.5)
        profile = get_profile("contrarian_v1_test")

        dev_trader, _, _, _ = replay_profile(
            list(split.development), prices_by_timestamp, profile
        )
        full_trader, _, _, _ = replay_profile(timestamps, prices_by_timestamp, profile)

        assert len(dev_trader.trade_log) <= len(full_trader.trade_log)

    def test_profile_snapshot_saved_in_results(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)

        results = run_experiment(["contrarian_v1_test"], csv_path=price_log)
        saved = results["profile_results"]["contrarian_v1_test"]["profile"]
        assert saved["profile_id"] == "contrarian_v1_test"
        assert saved["strategy_name"] == "contrarian_v1"

    def test_contrarian_test_and_production_can_differ(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)

        results = run_experiment(
            ["contrarian_v1_test", "contrarian_v1_production"],
            csv_path=price_log,
        )
        test_trades = results["profile_results"]["contrarian_v1_test"]["combined"]["total_buys"]
        prod_trades = results["profile_results"]["contrarian_v1_production"]["combined"]["total_buys"]
        assert test_trades >= prod_trades
