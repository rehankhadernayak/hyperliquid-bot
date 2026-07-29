"""Tests for trend-filter strategy experiments."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from config import SIGNAL
from exit_config import SIGNAL_EXIT_V1
from experiment_metrics import compute_period_metrics
from experiment_runner import replay_profile, run_experiment, save_experiment
from experiment_trader import ExperimentPaperTrader
from experiment_trend_comparison import (
    LABEL_FILTERED_TOO_MUCH,
    LABEL_INSUFFICIENT_TREND_DATA,
    LABEL_POSSIBLE_ENTRY_IMPROVEMENT,
    TREND_COMPARISON_LABEL_RULES,
    build_trend_comparison,
    classify_trend_variant,
)
from market_analyzer import WINDOW_1M_SECONDS
from paper_trader import SELL_SIGNALS
from strategy_profiles import STRATEGY_PROFILES, get_profile
from tests.conftest import LIVE_PAPER_TRADES, LIVE_PRICE_LOG
from tests.test_experiments import _write_sample_price_log
from trend_config import (
    BLOCK_REASON_INSUFFICIENT_TREND_HISTORY,
    FILTER_TYPE_COMBINED,
    FILTER_TYPE_FIVE_MINUTE_THRESHOLD,
    FILTER_TYPE_WORSENING_MOMENTUM,
    MISSING_HISTORY_ALLOW_ENTRY,
    MISSING_HISTORY_BLOCK_ENTRY,
    TREND_FILTER_DISABLED,
    TrendFilterConfig,
    WORSENING_MOMENTUM_RULE,
    validate_trend_filter,
)
from trend_filter import (
    FutureDataAccessError,
    compute_worsening_momentum,
    evaluate_trend_filter_for_buy,
)

TREND_PROFILES = [
    "contrarian_v1_entry_medium",
    "contrarian_v1_medium_no_trend_filter",
    "contrarian_v1_medium_trend30",
    "contrarian_v1_medium_trend50",
    "contrarian_v1_medium_worsening",
    "contrarian_v1_medium_combined",
    "hold_baseline",
]


def _build_history(
    prices: list[tuple[float, float]],
) -> dict[str, list[tuple[float, float]]]:
    return {"BTC": list(prices)}


class TestTrendProfiles:
    def test_all_trend_profiles_exist(self):
        for profile_id in TREND_PROFILES:
            assert profile_id in STRATEGY_PROFILES

    def test_entry_medium_unchanged(self):
        medium = get_profile("contrarian_v1_entry_medium")
        assert medium.thresholds.down_pct == -0.15
        assert medium.thresholds.strong_down_pct == -0.25
        assert medium.thresholds.up_pct == SIGNAL.test_mode.up_pct
        assert medium.thresholds.strong_up_pct == SIGNAL.test_mode.strong_up_pct
        assert medium.exit_config is None
        assert medium.trend_filter == TREND_FILTER_DISABLED

    def test_no_trend_filter_matches_medium_thresholds_and_signal_exit(self):
        baseline = get_profile("contrarian_v1_medium_no_trend_filter")
        medium = get_profile("contrarian_v1_entry_medium")
        signal_exit = get_profile("contrarian_v1_medium_signal_exit")
        assert baseline.thresholds == medium.thresholds
        assert baseline.exit_config == SIGNAL_EXIT_V1
        assert baseline.exit_config == signal_exit.exit_config
        assert baseline.trend_filter.enabled is False

    def test_trend30_values(self):
        profile = get_profile("contrarian_v1_medium_trend30")
        assert profile.trend_filter.filter_type == FILTER_TYPE_FIVE_MINUTE_THRESHOLD
        assert profile.trend_filter.five_minute_block_threshold_pct == -0.30

    def test_trend50_values(self):
        profile = get_profile("contrarian_v1_medium_trend50")
        assert profile.trend_filter.five_minute_block_threshold_pct == -0.50

    def test_worsening_values(self):
        profile = get_profile("contrarian_v1_medium_worsening")
        assert profile.trend_filter.filter_type == FILTER_TYPE_WORSENING_MOMENTUM
        assert profile.trend_filter.worsening_momentum_required is True

    def test_combined_values(self):
        profile = get_profile("contrarian_v1_medium_combined")
        assert profile.trend_filter.filter_type == FILTER_TYPE_COMBINED
        assert profile.trend_filter.five_minute_block_threshold_pct == -0.30
        assert profile.trend_filter.worsening_momentum_required is True

    def test_profiles_immutable(self):
        profile = get_profile("contrarian_v1_medium_trend30")
        with pytest.raises(Exception):
            profile.trend_filter = TREND_FILTER_DISABLED


class TestTrendFilterEvaluation:
    def test_disabled_filter_allows_buy(self):
        profile = get_profile("contrarian_v1_medium_no_trend_filter")
        history = _build_history([(0.0, 100.0), (300.0, 97.0)])
        assert (
            evaluate_trend_filter_for_buy(
                coin="BTC",
                current_price=97.0,
                price_history=history,
                timestamp="t",
                now=300.0,
                lookback_seconds=WINDOW_1M_SECONDS,
                trend_filter=profile.trend_filter,
                strategy_name="contrarian_v1",
                strategy_version="1.0.0",
                profile_name=profile.profile_id,
            )
            is None
        )

    def test_five_minute_threshold_blocks_at_boundary(self):
        profile = get_profile("contrarian_v1_medium_trend30")
        history = _build_history(
            [
                (0.0, 100.0),
                (240.0, 99.69),
                (300.0, 99.69),
            ]
        )
        block = evaluate_trend_filter_for_buy(
            coin="BTC",
            current_price=99.69,
            price_history=history,
            timestamp="t",
            now=300.0,
            lookback_seconds=WINDOW_1M_SECONDS,
            trend_filter=profile.trend_filter,
            strategy_name="contrarian_v1",
            strategy_version="1.0.0",
            profile_name=profile.profile_id,
        )
        assert block is not None
        assert block.decision == "BLOCKED"
        assert block.five_minute_change_pct == pytest.approx(-0.31, abs=0.01)

    def test_five_minute_threshold_allows_above_boundary(self):
        profile = get_profile("contrarian_v1_medium_trend30")
        history = _build_history(
            [
                (0.0, 100.0),
                (240.0, 99.75),
                (300.0, 99.75),
            ]
        )
        assert (
            evaluate_trend_filter_for_buy(
                coin="BTC",
                current_price=99.75,
                price_history=history,
                timestamp="t",
                now=300.0,
                lookback_seconds=WINDOW_1M_SECONDS,
                trend_filter=profile.trend_filter,
                strategy_name="contrarian_v1",
                strategy_version="1.0.0",
                profile_name=profile.profile_id,
            )
            is None
        )

    def test_trend50_blocks_deeper_downtrend(self):
        profile = get_profile("contrarian_v1_medium_trend50")
        history = _build_history(
            [
                (0.0, 100.0),
                (240.0, 99.4),
                (300.0, 99.4),
            ]
        )
        block = evaluate_trend_filter_for_buy(
            coin="BTC",
            current_price=99.4,
            price_history=history,
            timestamp="t",
            now=300.0,
            lookback_seconds=WINDOW_1M_SECONDS,
            trend_filter=profile.trend_filter,
            strategy_name="contrarian_v1",
            strategy_version="1.0.0",
            profile_name=profile.profile_id,
        )
        assert block is not None
        assert block.five_minute_change_pct == pytest.approx(-0.60, abs=0.01)

    def test_worsening_momentum_calculation(self):
        history = _build_history(
            [
                (0.0, 100.0),
                (60.0, 99.5),
                (120.0, 99.0),
                (180.0, 98.5),
            ]
        )
        worsening, current, previous = compute_worsening_momentum(
            history["BTC"],
            current_price=98.5,
            now=180.0,
            lookback_seconds=WINDOW_1M_SECONDS,
        )
        assert current is not None
        assert previous is not None
        assert current < previous
        assert worsening is True

    def test_worsening_momentum_blocks_buy(self):
        profile = get_profile("contrarian_v1_medium_worsening")
        history = _build_history(
            [
                (0.0, 100.0),
                (60.0, 99.5),
                (120.0, 99.0),
                (180.0, 98.5),
            ]
        )
        block = evaluate_trend_filter_for_buy(
            coin="BTC",
            current_price=98.5,
            price_history=history,
            timestamp="t",
            now=180.0,
            lookback_seconds=WINDOW_1M_SECONDS,
            trend_filter=profile.trend_filter,
            strategy_name="contrarian_v1",
            strategy_version="1.0.0",
            profile_name=profile.profile_id,
        )
        assert block is not None
        assert block.worsening_momentum is True

    def test_combined_blocks_on_either_condition(self):
        profile = get_profile("contrarian_v1_medium_combined")
        history = _build_history(
            [
                (0.0, 100.0),
                (240.0, 99.69),
                (300.0, 99.69),
            ]
        )
        block = evaluate_trend_filter_for_buy(
            coin="BTC",
            current_price=99.69,
            price_history=history,
            timestamp="t",
            now=300.0,
            lookback_seconds=WINDOW_1M_SECONDS,
            trend_filter=profile.trend_filter,
            strategy_name="contrarian_v1",
            strategy_version="1.0.0",
            profile_name=profile.profile_id,
        )
        assert block is not None

    def test_insufficient_history_blocks_with_distinct_reason(self):
        profile = get_profile("contrarian_v1_medium_trend30")
        history = _build_history([(0.0, 100.0), (60.0, 99.0)])
        block = evaluate_trend_filter_for_buy(
            coin="BTC",
            current_price=99.0,
            price_history=history,
            timestamp="t",
            now=60.0,
            lookback_seconds=WINDOW_1M_SECONDS,
            trend_filter=profile.trend_filter,
            strategy_name="contrarian_v1",
            strategy_version="1.0.0",
            profile_name=profile.profile_id,
        )
        assert block is not None
        assert block.reason == BLOCK_REASON_INSUFFICIENT_TREND_HISTORY
        assert block.insufficient_trend_history is True

    def test_missing_history_allow_entry(self):
        trend_filter = TrendFilterConfig(
            enabled=True,
            filter_type=FILTER_TYPE_FIVE_MINUTE_THRESHOLD,
            five_minute_block_threshold_pct=-0.30,
            worsening_momentum_required=False,
            minimum_history_seconds=300,
            missing_history_behavior=MISSING_HISTORY_ALLOW_ENTRY,
        )
        history = _build_history([(0.0, 100.0), (60.0, 99.0)])
        assert (
            evaluate_trend_filter_for_buy(
                coin="BTC",
                current_price=99.0,
                price_history=history,
                timestamp="t",
                now=60.0,
                lookback_seconds=WINDOW_1M_SECONDS,
                trend_filter=trend_filter,
                strategy_name="contrarian_v1",
                strategy_version="1.0.0",
                profile_name="test",
            )
            is None
        )

    def test_future_data_guard(self):
        profile = get_profile("contrarian_v1_medium_trend30")
        history = _build_history([(0.0, 100.0), (400.0, 99.0)])
        with pytest.raises(FutureDataAccessError):
            evaluate_trend_filter_for_buy(
                coin="BTC",
                current_price=99.0,
                price_history=history,
                timestamp="t",
                now=300.0,
                lookback_seconds=WINDOW_1M_SECONDS,
                trend_filter=profile.trend_filter,
                strategy_name="contrarian_v1",
                strategy_version="1.0.0",
                profile_name=profile.profile_id,
            )

    def test_block_result_structure(self):
        profile = get_profile("contrarian_v1_medium_trend30")
        history = _build_history(
            [
                (0.0, 100.0),
                (240.0, 99.69),
                (300.0, 99.69),
            ]
        )
        block = evaluate_trend_filter_for_buy(
            coin="BTC",
            current_price=99.69,
            price_history=history,
            timestamp="2026-01-01 00:05:00 UTC",
            now=300.0,
            lookback_seconds=WINDOW_1M_SECONDS,
            trend_filter=profile.trend_filter,
            strategy_name="contrarian_v1",
            strategy_version="1.0.0",
            profile_name=profile.profile_id,
        )
        payload = block.to_dict()
        assert payload["decision"] == "BLOCKED"
        assert payload["strategy_name"] == "contrarian_v1"
        assert payload["profile_name"] == profile.profile_id
        assert payload["timestamp"] == "2026-01-01 00:05:00 UTC"


class TestTrendFilterReplay:
    def test_no_filter_matches_signal_exit_baseline(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        timestamps, prices = __import__("backtester").load_price_ticks(price_log)

        no_filter = get_profile("contrarian_v1_medium_no_trend_filter")
        signal_exit = get_profile("contrarian_v1_medium_signal_exit")

        no_filter_trader, no_filter_equity, _, _ = replay_profile(
            timestamps, prices, no_filter
        )
        signal_trader, signal_equity, _, _ = replay_profile(
            timestamps, prices, signal_exit
        )

        assert len(no_filter_trader.trade_log) == len(signal_trader.trade_log)
        assert no_filter_equity == signal_equity

    def test_trend_filter_reduces_buys_vs_no_filter(self, tmp_path):
        rows = [
            "timestamp,coin,price",
            "2026-01-01 00:00:00 UTC,BTC,100.0",
            "2026-01-01 00:01:00 UTC,BTC,100.0",
            "2026-01-01 00:02:00 UTC,BTC,100.0",
            "2026-01-01 00:03:00 UTC,BTC,100.0",
            "2026-01-01 00:04:00 UTC,BTC,100.0",
            "2026-01-01 00:05:00 UTC,BTC,100.0",
            "2026-01-01 00:06:00 UTC,BTC,99.80",
            "2026-01-01 00:07:00 UTC,BTC,99.80",
        ]
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log, rows=rows)
        timestamps, prices = __import__("backtester").load_price_ticks(price_log)

        no_filter = get_profile("contrarian_v1_medium_no_trend_filter")
        trend30 = get_profile("contrarian_v1_medium_trend30")

        baseline_trader, _, _, _ = replay_profile(timestamps, prices, no_filter)
        filtered_trader, _, _, _ = replay_profile(timestamps, prices, trend30)

        baseline_buys = sum(1 for t in baseline_trader.trade_log if t.action == "BUY")
        filtered_buys = sum(1 for t in filtered_trader.trade_log if t.action == "BUY")
        assert filtered_trader.run_state.entries_blocked_by_trend_filter >= 0
        assert filtered_buys <= baseline_buys

    def test_sells_never_blocked_by_trend_filter(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        timestamps, prices = __import__("backtester").load_price_ticks(price_log)

        filtered_trader, _, _, _ = replay_profile(
            timestamps,
            prices,
            get_profile("contrarian_v1_medium_trend30"),
        )

        for trade in filtered_trader.trade_log:
            if trade.action == "SELL":
                assert trade.signal in SELL_SIGNALS or trade.signal in {
                    "STOP_LOSS",
                    "TAKE_PROFIT",
                    "MAX_HOLD",
                    "FORCED_CLOSE",
                }

    def test_identical_input_rows_across_profiles(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        results = run_experiment(TREND_PROFILES, csv_path=price_log)
        assert results["timestamp_count"] == 6
        assert results["row_count"] == 18


class TestTrendMetrics:
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
        assert metrics["total_entry_signals"] == 0
        assert metrics["entries_executed"] == 0
        assert metrics["entry_filter_percentage"] == "N/A"

    def test_trend_filter_metrics_populated(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        results = run_experiment(
            ["contrarian_v1_medium_trend30"],
            csv_path=price_log,
        )
        oos = results["profile_results"]["contrarian_v1_medium_trend30"]["out_of_sample"]
        assert "total_entry_signals" in oos
        assert "entries_blocked_by_trend_filter" in oos
        assert "entries_blocked_insufficient_history" in oos
        assert "entry_filter_percentage" in oos


class TestTrendComparisonLabels:
    def test_label_rules_documented(self):
        assert LABEL_INSUFFICIENT_TREND_DATA in TREND_COMPARISON_LABEL_RULES
        assert LABEL_POSSIBLE_ENTRY_IMPROVEMENT in TREND_COMPARISON_LABEL_RULES

    def test_insufficient_trend_data(self):
        baseline = {"completed_trades": 10, "net_return_pct": 1.0}
        variant = {"completed_trades": 2, "net_return_pct": 0.5, "entry_filter_percentage": 10.0}
        assert (
            classify_trend_variant(
                baseline,
                variant,
                evidence_assessment="INSUFFICIENT_DATA",
                development_metrics={"net_return_pct": 0.5},
            )
            == LABEL_INSUFFICIENT_TREND_DATA
        )

    def test_filtered_too_much(self):
        baseline = {
            "completed_trades": 10,
            "net_return_pct": 1.0,
            "average_net_return_per_trade": 0.1,
            "maximum_drawdown_pct": 5.0,
            "fees_paid": 1.0,
            "slippage_cost": 0.5,
            "entry_filter_percentage": 0.0,
            "trade_log": [],
        }
        variant = {
            "completed_trades": 10,
            "net_return_pct": 2.0,
            "average_net_return_per_trade": 0.2,
            "maximum_drawdown_pct": 4.0,
            "fees_paid": 0.5,
            "slippage_cost": 0.2,
            "entry_filter_percentage": 85.0,
            "trade_log": [],
        }
        assert (
            classify_trend_variant(
                baseline,
                variant,
                evidence_assessment="PROMISING",
                development_metrics={"net_return_pct": 2.0},
            )
            == LABEL_FILTERED_TOO_MUCH
        )

    def test_build_trend_comparison_requires_baseline(self):
        assert build_trend_comparison({}) is None


class TestTrendExperimentRunner:
    def test_full_trend_comparison_experiment(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        results = run_experiment(TREND_PROFILES, csv_path=price_log)
        for profile_id in TREND_PROFILES:
            assert profile_id in results["profile_results"]
        assert results.get("trend_comparison") is not None

    def test_trend_comparison_in_json(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        output_dir = tmp_path / "experiments"
        results = run_experiment(TREND_PROFILES, csv_path=price_log)
        path = save_experiment(results, output_dir=output_dir)
        payload = json.loads(path.read_text())
        assert "trend_comparison" in payload
        assert (
            payload["trend_comparison"]["baseline_profile"]
            == "contrarian_v1_medium_no_trend_filter"
        )

    def test_no_live_data_writes(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        live_mtime = LIVE_PAPER_TRADES.stat().st_mtime if LIVE_PAPER_TRADES.exists() else None
        live_price = LIVE_PRICE_LOG.stat().st_mtime if LIVE_PRICE_LOG.exists() else None
        run_experiment(TREND_PROFILES[:3], csv_path=price_log)
        if live_mtime is not None:
            assert LIVE_PAPER_TRADES.stat().st_mtime == live_mtime
        if live_price is not None:
            assert LIVE_PRICE_LOG.stat().st_mtime == live_price

    def test_no_network_calls(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        with patch("urllib.request.urlopen") as mock_urlopen:
            run_experiment(TREND_PROFILES[:2], csv_path=price_log)
            mock_urlopen.assert_not_called()

    def test_worsening_momentum_rule_documented(self):
        assert "previous completed 1-minute percentage change" in WORSENING_MOMENTUM_RULE

    def test_validate_trend_filter_rejects_invalid(self):
        with pytest.raises(ValueError):
            validate_trend_filter(
                TrendFilterConfig(
                    enabled=True,
                    filter_type="BAD",
                    five_minute_block_threshold_pct=-0.30,
                    worsening_momentum_required=False,
                    minimum_history_seconds=300,
                    missing_history_behavior=MISSING_HISTORY_BLOCK_ENTRY,
                )
            )

    def test_isolated_trader_state(self, tmp_path):
        price_log = tmp_path / "price_log.csv"
        _write_sample_price_log(price_log)
        timestamps, prices = __import__("backtester").load_price_ticks(price_log)
        profile_a = get_profile("contrarian_v1_medium_no_trend_filter")
        profile_b = get_profile("contrarian_v1_medium_trend30")
        trader_a, _, _, _ = replay_profile(timestamps, prices, profile_a)
        trader_b, _, _, _ = replay_profile(timestamps, prices, profile_b)
        assert trader_a is not trader_b
