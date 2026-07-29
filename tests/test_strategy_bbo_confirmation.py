"""Focused offline tests for contrarian_bbo_confirmation_v1."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from bbo_confirmation_config import (
    BBO_CONFIRMATION_LIGHT,
    BBO_CONFIRMATION_MEDIUM,
    BBO_CONFIRMATION_REBOUND,
    BBO_CONFIRMATION_STRONG,
    validate_bbo_confirmation,
)
from config import SIGNAL
from event_backtester.backtest_config import (
    EventBacktestConfig,
    SUPPORTED_STRATEGY_PROFILES,
    resolve_strategy_profile,
)
from event_backtester.event_strategy_adapter import EventStrategyAdapter
from exit_config import ExitConfig
from paper_trader import PaperTrader
from strategies import STRATEGY_REGISTRY, get_strategy
from strategies.contrarian_bbo_confirmation_v1 import (
    SIGNAL_BLOCKED,
    SIGNAL_HOLD,
    SIGNAL_PAPER_BUY,
    SIGNAL_PAPER_SELL,
    ContrarianBboConfirmationV1Strategy,
)
from strategy_profiles import get_profile
from strategy_types import StrategyContext

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_FILE = REPO_ROOT / "strategies" / "contrarian_bbo_confirmation_v1.py"


def _context(
    *,
    price: float = 100.0,
    history: list[tuple[float, float]] | None = None,
    now: float = 1000.0,
    bbo_confirmation=BBO_CONFIRMATION_MEDIUM,
    spread_bps: float | None = 10.0,
    bbo_imbalance: float | None = 0.15,
    bbo_available: bool = True,
    buy_trades: int = 5,
    sell_trades: int = 2,
    cooldown_remaining: float | None = None,
    position_is_open: bool = False,
    exit_config=None,
) -> StrategyContext:
    if history is None:
        history = [
            (now - 30, 101.5),
            (now - 15, 101.0),
            (now - 10, 100.5),
            (now - 5, 99.0),
        ]
    trader = PaperTrader()
    return StrategyContext(
        coin="BTC",
        current_price=price,
        price_history={"BTC": history},
        timestamp="2026-06-18 12:00:00 UTC",
        now=now,
        trader=trader,
        thresholds=SIGNAL.test_mode,
        lookback_seconds=60,
        profile_name="contrarian_bbo_medium",
        bbo_confirmation=bbo_confirmation,
        spread_bps=spread_bps,
        bbo_imbalance=bbo_imbalance,
        bbo_imbalance_change_10s=0.02,
        bbo_available=bbo_available,
        trade_imbalance_10s=0.1,
        buy_trade_count_10s=buy_trades,
        sell_trade_count_10s=sell_trades,
        recent_volume_10s=1.5,
        cooldown_remaining=cooldown_remaining,
        position_is_open=position_is_open,
        exit_config=exit_config,
    )


class TestStrategyRegistration:
    def test_strategy_in_registry(self) -> None:
        assert "contrarian_bbo_confirmation_v1" in STRATEGY_REGISTRY
        strategy = get_strategy("contrarian_bbo_confirmation_v1")
        assert strategy.name == "contrarian_bbo_confirmation_v1"
        assert strategy.version == "1.0.0"

    def test_profiles_registered(self) -> None:
        for profile_id in (
            "contrarian_bbo_light",
            "contrarian_bbo_medium",
            "contrarian_bbo_strong",
            "contrarian_bbo_rebound",
        ):
            profile = get_profile(profile_id)
            assert profile.strategy_name == "contrarian_bbo_confirmation_v1"
            assert profile.bbo_confirmation is not None

    def test_profile_immutability(self) -> None:
        profile = get_profile("contrarian_bbo_medium")
        with pytest.raises(Exception):
            profile.trade_size_usdc = 999.0  # type: ignore[misc]

    def test_bbo_config_immutability(self) -> None:
        with pytest.raises(Exception):
            BBO_CONFIRMATION_MEDIUM.min_bbo_imbalance = 0.99  # type: ignore[misc]

    def test_event_backtest_profiles_supported(self) -> None:
        for profile_id in (
            "contrarian_bbo_light",
            "contrarian_bbo_medium",
            "contrarian_bbo_strong",
            "contrarian_bbo_rebound",
        ):
            assert profile_id in SUPPORTED_STRATEGY_PROFILES
            EventBacktestConfig(strategy_profile_id=profile_id, test_mode=True)

    def test_strategy_name_alias_resolves(self) -> None:
        assert resolve_strategy_profile("contrarian_bbo_confirmation_v1") == "contrarian_bbo_medium"
        config = EventBacktestConfig(
            strategy_profile_id="contrarian_bbo_confirmation_v1",
            test_mode=True,
        )
        assert config.strategy_profile_id == "contrarian_bbo_medium"


class TestMissingData:
    def test_hold_when_price_missing(self) -> None:
        strategy = ContrarianBboConfirmationV1Strategy()
        ctx = _context(price=100.0)
        ctx = StrategyContext(
            coin=ctx.coin,
            current_price=None,
            price_history=ctx.price_history,
            timestamp=ctx.timestamp,
            now=ctx.now,
            trader=ctx.trader,
            thresholds=ctx.thresholds,
            lookback_seconds=ctx.lookback_seconds,
            bbo_confirmation=BBO_CONFIRMATION_MEDIUM,
        )
        result = strategy.evaluate(ctx)
        assert result.signal == SIGNAL_HOLD

    def test_blocked_when_bbo_missing(self) -> None:
        strategy = ContrarianBboConfirmationV1Strategy()
        ctx = _context(bbo_available=False)
        result = strategy.evaluate(ctx)
        assert result.signal == SIGNAL_BLOCKED
        assert result.indicators["blocked_reason"] == "missing_or_stale_bbo"

    def test_blocked_when_spread_missing(self) -> None:
        strategy = ContrarianBboConfirmationV1Strategy()
        ctx = _context(spread_bps=None)
        result = strategy.evaluate(ctx)
        assert result.signal == SIGNAL_BLOCKED
        assert result.indicators["blocked_reason"] == "missing_spread"

    def test_blocked_when_bbo_imbalance_missing(self) -> None:
        strategy = ContrarianBboConfirmationV1Strategy()
        ctx = _context(bbo_imbalance=None)
        result = strategy.evaluate(ctx)
        assert result.signal == SIGNAL_BLOCKED
        assert result.indicators["blocked_reason"] == "missing_bbo_imbalance"

    def test_missing_bbo_config_blocks(self) -> None:
        strategy = ContrarianBboConfirmationV1Strategy()
        ctx = _context(bbo_confirmation=None)
        result = strategy.evaluate(ctx)
        assert result.signal == SIGNAL_BLOCKED


class TestSpreadFilter:
    def test_blocks_when_spread_too_wide(self) -> None:
        strategy = ContrarianBboConfirmationV1Strategy()
        ctx = _context(spread_bps=50.0, bbo_confirmation=BBO_CONFIRMATION_MEDIUM)
        result = strategy.evaluate(ctx)
        assert result.signal == SIGNAL_BLOCKED
        assert result.indicators["blocked_reason"] == "spread_too_wide"


class TestBboImbalance:
    def test_blocks_when_imbalance_below_threshold(self) -> None:
        strategy = ContrarianBboConfirmationV1Strategy()
        ctx = _context(bbo_imbalance=0.02, bbo_confirmation=BBO_CONFIRMATION_MEDIUM)
        result = strategy.evaluate(ctx)
        assert result.signal == SIGNAL_BLOCKED
        assert result.indicators["blocked_reason"] == "bbo_imbalance_below_threshold"


class TestEntrySignal:
    def test_paper_buy_when_all_gates_pass(self) -> None:
        strategy = ContrarianBboConfirmationV1Strategy()
        ctx = _context(
            price=98.0,
            history=[
                (970.0, 101.0),
                (985.0, 100.5),
                (990.0, 100.0),
                (995.0, 99.5),
                (998.0, 99.0),
            ],
            now=1000.0,
            bbo_imbalance=0.20,
            spread_bps=8.0,
        )
        result = strategy.evaluate(ctx)
        assert result.signal == SIGNAL_PAPER_BUY
        assert result.indicators["return_5s_pct"] is not None
        assert result.indicators["bbo_imbalance"] == pytest.approx(0.20)
        assert result.indicators["spread_bps"] == pytest.approx(8.0)

    def test_rebound_profile_blocks_without_rebound(self) -> None:
        strategy = ContrarianBboConfirmationV1Strategy()
        ctx = _context(
            price=98.02,
            history=[(985.0, 100.0), (995.0, 99.0), (999.0, 98.0)],
            now=1000.0,
            bbo_confirmation=BBO_CONFIRMATION_REBOUND,
        )
        result = strategy.evaluate(ctx)
        assert result.signal == SIGNAL_BLOCKED
        assert result.indicators["blocked_reason"] == "rebound_not_confirmed"


class TestExitSignal:
    def test_paper_sell_on_rise(self) -> None:
        strategy = ContrarianBboConfirmationV1Strategy()
        ctx = _context(
            price=102.0,
            history=[(970.0, 98.0), (990.0, 100.0), (998.0, 101.0)],
            now=1000.0,
            position_is_open=True,
        )
        result = strategy.evaluate(ctx)
        assert result.signal == SIGNAL_PAPER_SELL

    def test_fixed_tp_sl_exposed_in_indicators(self) -> None:
        strategy = ContrarianBboConfirmationV1Strategy()
        exit_cfg = ExitConfig(
            exit_mode="cost_aware",
            take_profit_pct=0.35,
            stop_loss_pct=0.25,
            signal_exit_enabled=False,
        )
        ctx = _context(position_is_open=True, exit_config=exit_cfg)
        result = strategy.evaluate(ctx)
        assert result.signal == SIGNAL_HOLD
        assert result.indicators["planned_exit_mode"] == "fixed_tp_sl"
        assert result.indicators["planned_take_profit_pct"] == 0.35


class TestCooldown:
    def test_cooldown_blocks_entry(self) -> None:
        strategy = ContrarianBboConfirmationV1Strategy()
        ctx = _context(cooldown_remaining=30.0)
        result = strategy.evaluate(ctx)
        assert result.signal == SIGNAL_BLOCKED
        assert result.indicators["blocked_reason"] == "cooldown_active"


class TestIndicators:
    def test_indicators_in_result(self) -> None:
        strategy = ContrarianBboConfirmationV1Strategy()
        result = strategy.evaluate(_context())
        for key in (
            "return_5s_pct",
            "return_10s_pct",
            "return_30s_pct",
            "bbo_imbalance",
            "spread_bps",
            "buy_trade_count_10s",
            "sell_trade_count_10s",
            "recent_volume_10s",
        ):
            assert key in result.indicators
        assert result.strategy_name == "contrarian_bbo_confirmation_v1"


class TestEventAdapter:
    def test_adapter_loads_bbo_strategy(self) -> None:
        profile = get_profile("contrarian_bbo_medium")
        adapter = EventStrategyAdapter(profile)
        assert adapter.profile.strategy_name == "contrarian_bbo_confirmation_v1"
        notes = adapter.adapter_notes()
        assert any("BBO confirmation" in n for n in notes)

    def test_paper_buy_maps_to_buy_action(self) -> None:
        from event_backtester.event_strategy_adapter import EventStrategySignal

        adapter = EventStrategyAdapter(get_profile("contrarian_bbo_medium"))
        signal = EventStrategySignal(
            signal=SIGNAL_PAPER_BUY,
            signal_strength="MODERATE",
            reason="test",
            strategy_name="contrarian_bbo_confirmation_v1",
            strategy_version="1.0.0",
            profile_name="contrarian_bbo_medium",
            indicators={},
            lookback_data_available=True,
            event_timestamp_ms=1_000_000,
        )
        assert adapter.action_for_signal(signal) == "BUY"


class TestSafety:
    def test_no_wallet_or_signing_in_strategy(self) -> None:
        source = STRATEGY_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        joined = source.lower()
        for pattern in ("private_key", "wallet", "sign(", "execute_order", "requests."):
            assert pattern not in joined
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "requests" not in alias.name

    def test_bbo_config_validation(self) -> None:
        validate_bbo_confirmation(BBO_CONFIRMATION_LIGHT)
        validate_bbo_confirmation(BBO_CONFIRMATION_STRONG)
        with pytest.raises(ValueError):
            validate_bbo_confirmation(
                BBO_CONFIRMATION_LIGHT.__class__(
                    dip_lookback_seconds=10,
                    dip_threshold_pct=0.5,
                    rebound_lookback_seconds=5,
                    rebound_threshold_pct=0.0,
                    min_bbo_imbalance=0.1,
                    max_spread_bps=20.0,
                )
            )


class TestExperimentComparison:
    def test_build_bbo_comparison_structure(self) -> None:
        from experiment_bbo_comparison import build_bbo_comparison

        baseline = {
            "profile_name": "contrarian_v1_medium_no_trend_filter",
            "strategy_name": "contrarian_v1",
            "net_return_pct": 1.0,
            "completed_trades": 10,
            "win_rate": 55.0,
            "profit_factor": 1.2,
            "maximum_drawdown_pct": 5.0,
            "total_fees": 2.0,
            "total_slippage_cost": 1.0,
            "average_net_return_per_trade": 0.1,
            "blocked_signals": 3,
            "blocked_signals_summary": {"total_blocked": 3, "blocked_reasons": {"spread_too_wide": 2}},
            "results_by_coin": {"BTC": {"buys": 5, "sells": 5}},
        }
        variant = {
            **baseline,
            "profile_name": "contrarian_bbo_medium",
            "strategy_name": "contrarian_bbo_confirmation_v1",
            "net_return_pct": 2.0,
            "blocked_signals_summary": {
                "total_blocked": 8,
                "blocked_reasons": {"bbo_imbalance_below_threshold": 5},
            },
        }
        comparison = build_bbo_comparison(
            {
                "contrarian_v1_medium_no_trend_filter": baseline,
                "contrarian_bbo_medium": variant,
            }
        )
        assert comparison["baseline_profile"] == "contrarian_v1_medium_no_trend_filter"
        assert len(comparison["bbo_variants"]) == 4
        assert comparison["no_auto_winner"] is True
        assert comparison["out_of_sample_available"] is False


class TestEventBacktestSmokeMode:
    @pytest.fixture
    def smoke_stream(self, tmp_path: Path):
        import json

        from market_data.quality_models import QualityLabel
        from tests.test_event_backtester import (
            _write_manifest,
            _write_parquet,
            _write_quality,
            _bbo_row,
            _trade_row,
        )

        session_id = "fixture_recorder_session"
        _write_quality(tmp_path, session_id)
        trade_path = _write_parquet(
            tmp_path,
            "trade",
            "BTC",
            [
                _trade_row("BTC", 1_000_000, 100.0, 1),
                _trade_row("BTC", 1_000_500, 99.5, 2),
                _trade_row("BTC", 1_001_000, 99.0, 3),
                _trade_row("BTC", 1_061_000, 100.5, 4),
                _trade_row("BTC", 1_061_500, 101.0, 5),
            ],
        )
        bbo_path = _write_parquet(
            tmp_path,
            "bbo",
            "BTC",
            [
                _bbo_row("BTC", 999_900, 99.0, 101.0),
                _bbo_row("BTC", 1_000_000, 99.0, 101.0),
                _bbo_row("BTC", 1_000_500, 98.5, 100.5),
                _bbo_row("BTC", 1_001_000, 98.0, 100.0),
                _bbo_row("BTC", 1_061_000, 100.0, 102.0),
                _bbo_row("BTC", 1_061_500, 100.5, 102.5),
            ],
        )
        _write_manifest(tmp_path, session_id, [str(trade_path), str(bbo_path)])
        replay_id = "fixture_replay_session"
        replays_dir = tmp_path / "replays"
        replays_dir.mkdir(parents=True, exist_ok=True)
        replay_report = {
            "replay_session_id": replay_id,
            "source_recorder_session_id": session_id,
            "quality_label": QualityLabel.GOOD_FOR_RESEARCH.value,
            "stream_hash": "placeholder",
            "filters": {"coins": ["BTC"], "channels": ["trades", "bbo"]},
            "total_events": 11,
            "event_counts_by_coin": {"BTC": 11},
            "paper_only": True,
            "execution_enabled": False,
        }
        (replays_dir / f"{replay_id}.json").write_text(json.dumps(replay_report), encoding="utf-8")
        return tmp_path, session_id, replay_id

    def test_quick_maps_to_bounded_settings(self) -> None:
        from event_backtester.backtest_config import QUICK_MAX_EVENTS, QUICK_PROGRESS_EVERY
        from event_backtester.event_cli import build_parser, config_from_args

        args = build_parser().parse_args(["--quick", "--replay-session-id", "abc"])
        config = config_from_args(args)
        assert config.quick is True
        assert config.max_events == QUICK_MAX_EVENTS
        assert config.progress_every == QUICK_PROGRESS_EVERY
        assert config.smoke_test_mode is True

    def test_max_events_stops_after_limit(self, smoke_stream) -> None:
        from event_backtester.backtest_config import EventBacktestConfig
        from event_backtester.event_backtest_engine import EventBacktestEngine

        tmp_path, _session_id, replay_id = smoke_stream
        config = EventBacktestConfig(
            replay_session_id=replay_id,
            coins=("BTC",),
            channels=("trades", "bbo"),
            strategy_profile_id="contrarian_bbo_medium",
            max_events=3,
            output_dir=tmp_path / "event_backtests",
            market_stream_dir=tmp_path,
            test_mode=True,
        )
        result = EventBacktestEngine(config).run()
        assert result.events_processed == 3
        assert result.report["smoke_test"] is True
        assert result.report["truncated"] is True
        assert result.report["reason"] == "max_events"

    def test_max_seconds_stops_after_limit(self, smoke_stream) -> None:
        from event_backtester.backtest_config import EventBacktestConfig
        from event_backtester.event_backtest_engine import EventBacktestEngine

        tmp_path, _session_id, replay_id = smoke_stream
        config = EventBacktestConfig(
            replay_session_id=replay_id,
            coins=("BTC",),
            channels=("trades", "bbo"),
            strategy_profile_id="contrarian_bbo_medium",
            max_seconds=0,
            output_dir=tmp_path / "event_backtests",
            market_stream_dir=tmp_path,
            test_mode=True,
        )
        result = EventBacktestEngine(config).run()
        assert result.events_processed >= 1
        assert result.report["smoke_test"] is True
        assert result.report["truncated"] is True
        assert result.report["reason"] == "max_seconds"

    def test_progress_printing_does_not_crash(self, smoke_stream, capsys) -> None:
        from event_backtester.backtest_config import EventBacktestConfig
        from event_backtester.event_backtest_engine import EventBacktestEngine

        tmp_path, _session_id, replay_id = smoke_stream
        config = EventBacktestConfig(
            replay_session_id=replay_id,
            coins=("BTC",),
            channels=("trades", "bbo"),
            strategy_profile_id="contrarian_bbo_medium",
            max_events=4,
            progress_every=2,
            output_dir=tmp_path / "event_backtests",
            market_stream_dir=tmp_path,
            test_mode=True,
        )
        EventBacktestEngine(config).run()
        output = capsys.readouterr().out
        assert "Processed 2 events..." in output
        assert "Processed 4 events..." in output

    def test_report_includes_smoke_fields(self, smoke_stream) -> None:
        from event_backtester.backtest_config import EventBacktestConfig
        from event_backtester.event_backtest_engine import EventBacktestEngine

        tmp_path, _session_id, replay_id = smoke_stream
        config = EventBacktestConfig(
            replay_session_id=replay_id,
            coins=("BTC",),
            channels=("trades", "bbo"),
            strategy_profile_id="contrarian_bbo_medium",
            max_events=2,
            output_dir=tmp_path / "event_backtests",
            market_stream_dir=tmp_path,
            test_mode=True,
        )
        result = EventBacktestEngine(config).run()
        report = result.report
        assert report["smoke_test"] is True
        assert report["truncated"] is True
        assert report["max_events"] == 2
        assert report["processed_events"] == 2
        assert report["reason"] == "max_events"
        assert "must not be used as evidence of profitability" in (report.get("smoke_test_warning") or "")

    def test_compare_reports_labels_smoke_reports(self, tmp_path: Path) -> None:
        from experiment_bbo_comparison import LABEL_SMOKE_INCOMPLETE, build_bbo_comparison

        baseline = {
            "profile_name": "contrarian_v1_medium_no_trend_filter",
            "net_return_pct": 1.0,
            "completed_trades": 10,
            "total_fees": 1.0,
            "total_slippage_cost": 0.5,
            "maximum_drawdown_pct": 2.0,
            "profit_factor": 1.1,
        }
        smoke_variant = {
            "profile_name": "contrarian_bbo_medium",
            "net_return_pct": 5.0,
            "completed_trades": 10,
            "total_fees": 1.0,
            "total_slippage_cost": 0.5,
            "maximum_drawdown_pct": 2.0,
            "profit_factor": 2.0,
            "smoke_test": True,
            "truncated": True,
            "reason": "max_events",
            "smoke_test_warning": "Smoke-test results are incomplete and must not be used as evidence of profitability.",
        }
        comparison = build_bbo_comparison(
            {
                "contrarian_v1_medium_no_trend_filter": baseline,
                "contrarian_bbo_medium": smoke_variant,
            }
        )
        medium = next(v for v in comparison["bbo_variants"] if v["profile_name"] == "contrarian_bbo_medium")
        assert medium["smoke_test"] is True
        assert medium["comparison_eligible"] is False
        assert medium["comparison_label"] == LABEL_SMOKE_INCOMPLETE
        assert "contrarian_bbo_medium" in comparison["smoke_reports_excluded_from_ranking"]

    def test_no_wallet_signing_or_network_in_event_backtester(self) -> None:
        from event_backtester.event_safety import FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS

        skip = {"event_safety.py"}
        for path in (REPO_ROOT / "event_backtester").rglob("*.py"):
            if path.name in skip:
                continue
            source = path.read_text(encoding="utf-8").lower()
            for pattern in FORBIDDEN_SOURCE_PATTERNS:
                assert pattern not in source, f"{pattern} in {path.name}"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert alias.name.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        assert node.module.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES
