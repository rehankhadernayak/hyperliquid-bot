"""Offline tests for event-driven backtester v1."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from event_backtester.backtest_config import EventBacktestConfig
from event_backtester.event_backtest_engine import EventBacktestEngine, EventBacktestError
from event_backtester.event_cli import build_parser, config_from_args
from event_backtester.event_execution_simulator import EventExecutionSimulator, ExecutionRequest
from event_backtester.event_metrics import (
    EVIDENCE_INSUFFICIENT,
    assign_evidence_label,
    compute_completed_trade_metrics,
)
from event_backtester.event_portfolio import EventPortfolio
from event_backtester.event_report import (
    EVENT_BACKTEST_SCHEMA_VERSION,
    compute_deterministic_backtest_hash,
    list_event_backtest_reports,
    render_markdown_report,
    write_event_backtest_report,
)
from event_backtester.event_safety import (
    ALLOWED_CHANNELS_V1,
    FORBIDDEN_MODULE_NAMES,
    FORBIDDEN_SOURCE_PATTERNS,
    PAPER_ONLY,
)
from event_backtester.market_state import CoinMarketState, MarketStateManager
from market_data.event_models import BboEvent, TradeEvent
from market_data.quality_models import QualityLabel
from market_data.replay_cursor import ReplayCursor
from market_data.replay_engine import ReplayEngine
from market_data.replay_events import build_replay_event
from market_data.replay_filters import ReplayFilters
from strategy_profiles import get_profile

REPO_ROOT = Path(__file__).resolve().parent.parent
EVENT_BACKTESTER_DIR = REPO_ROOT / "event_backtester"


def _write_manifest(tmp_path: Path, session_id: str, files_written: list[str]) -> None:
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "recorder_session_id": session_id,
        "start_time_ms": 1_000_000,
        "end_time_ms": 1_100_000,
        "network": "mainnet",
        "coins": ["BTC"],
        "schema_version": "1.0.0",
        "files_written": files_written,
        "raw_row_count": 0,
        "normalized_row_count": 0,
        "paper_only": True,
        "execution_enabled": False,
    }
    (manifests_dir / f"{session_id}.json").write_text(json.dumps(data), encoding="utf-8")


def _write_quality(tmp_path: Path, session_id: str, label: str = QualityLabel.GOOD_FOR_RESEARCH.value) -> None:
    reports_dir = tmp_path / "quality_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": "1.0.0",
        "recorder_session_id": session_id,
        "quality_label": label,
        "data_integrity_warnings": [],
        "gap_summary": {"total_gaps": 0},
    }
    (reports_dir / f"{session_id}.json").write_text(json.dumps(data), encoding="utf-8")


def _write_parquet(tmp_path: Path, event_type: str, coin: str, rows: list[dict]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    partition = (
        tmp_path / "normalized" / f"event_type={event_type}" / "date=2026-06-18" / f"coin={coin}"
    )
    partition.mkdir(parents=True, exist_ok=True)
    path = partition / f"events-{event_type}-{coin}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), str(path))
    return path


def _trade_row(coin: str, recv_ms: int, price: float, trade_id: int = 1) -> dict:
    return {
        "coin": coin,
        "side": "B",
        "price": price,
        "size": 0.1,
        "trade_id": trade_id,
        "transaction_hash": "0xabc",
        "exchange_timestamp_ms": recv_ms,
        "received_timestamp_ms": recv_ms,
    }


def _bbo_row(coin: str, recv_ms: int, bid: float = 99.0, ask: float = 101.0) -> dict:
    return {
        "coin": coin,
        "best_bid_price": bid,
        "best_bid_size": 1.0,
        "best_ask_price": ask,
        "best_ask_size": 1.0,
        "spread": ask - bid,
        "spread_bps": ((ask - bid) / ((bid + ask) / 2)) * 10000,
        "exchange_timestamp_ms": recv_ms,
        "received_timestamp_ms": recv_ms,
    }


def _replay_event(
    event,
    event_type: str,
    recv_ms: int,
    seq: int,
    session_id: str = "recorder1",
    replay_id: str = "replay1",
) -> object:
    return build_replay_event(
        event=event,
        event_type=event_type,
        replay_session_id=replay_id,
        source_recorder_session_id=session_id,
        source_file="test.parquet",
        source_row_index=seq,
        replay_sequence=seq,
        primary_time_ms=recv_ms,
        primary_time_source="received_timestamp_ms",
    )


@pytest.fixture
def fixture_stream(tmp_path: Path) -> tuple[Path, str, str]:
    """Build a minimal recorder session with trades + BBO for BTC."""
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
        "paper_only": True,
        "execution_enabled": False,
    }
    (replays_dir / f"{replay_id}.json").write_text(json.dumps(replay_report), encoding="utf-8")
    return tmp_path, session_id, replay_id


class TestCLI:
    def test_argument_parsing(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--replay-session-id",
                "abc123",
                "--coins",
                "BTC",
                "ETH",
                "--channels",
                "trades",
                "bbo",
                "--strategy",
                "contrarian_v1_medium_no_trend_filter",
                "--dry-run",
            ]
        )
        config = config_from_args(args)
        assert config.replay_session_id == "abc123"
        assert config.coins == ("BTC", "ETH")
        assert config.dry_run is True

    def test_unsupported_channel_rejected(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            EventBacktestConfig(channels=("candle",))


class TestMarketState:
    def test_trade_update(self) -> None:
        mgr = MarketStateManager(("BTC",))
        trade = TradeEvent(
            coin="BTC",
            side="B",
            price=100.0,
            size=0.1,
            trade_id=1,
            transaction_hash="0x",
            exchange_timestamp_ms=1000,
            received_timestamp_ms=1000,
        )
        event = _replay_event(trade, "trade", 1000, 1)
        assert mgr.update_from_event(event)
        state = mgr.get_state("BTC")
        assert state is not None
        assert state.latest_trade_price == 100.0

    def test_bbo_update(self) -> None:
        mgr = MarketStateManager(("BTC",))
        bbo = BboEvent(
            coin="BTC",
            best_bid_price=99.0,
            best_bid_size=1.0,
            best_ask_price=101.0,
            best_ask_size=1.0,
            spread=2.0,
            spread_bps=20.0,
            exchange_timestamp_ms=1000,
            received_timestamp_ms=1000,
        )
        event = _replay_event(bbo, "bbo", 1000, 1)
        assert mgr.update_from_event(event)
        state = mgr.get_state("BTC")
        assert state is not None
        assert state.midpoint == 100.0
        assert state.spread_bps == 20.0

    def test_stale_bbo_detection(self) -> None:
        mgr = MarketStateManager(("BTC",))
        bbo = BboEvent(
            coin="BTC",
            best_bid_price=99.0,
            best_bid_size=1.0,
            best_ask_price=101.0,
            best_ask_size=1.0,
            spread=2.0,
            spread_bps=20.0,
            exchange_timestamp_ms=1000,
            received_timestamp_ms=1000,
        )
        mgr.update_from_event(_replay_event(bbo, "bbo", 1000, 1))
        assert mgr.bbo_available("BTC", 2500, stale_ms=2000)
        assert mgr.bbo_stale_reason("BTC", 5000, stale_ms=2000) == "missing_or_stale_bbo"

    def test_missing_bbo(self) -> None:
        mgr = MarketStateManager(("BTC",))
        assert mgr.bbo_stale_reason("BTC", 1000, stale_ms=5000) == "missing_or_stale_bbo"


class TestExecutionSimulator:
    def setup_method(self) -> None:
        self.profile = get_profile("contrarian_v1_medium_no_trend_filter")
        self.sim = EventExecutionSimulator(self.profile, bbo_stale_ms=10_000)

    def _state(self, bid: float = 99.0, ask: float = 101.0, ts: int = 1000) -> CoinMarketState:
        state = CoinMarketState(coin="BTC")
        state.best_bid_price = bid
        state.best_ask_price = ask
        state.best_bid_size = 1.0
        state.best_ask_size = 1.0
        state.midpoint = (bid + ask) / 2
        state.bbo_update_timestamp_ms = ts
        return state

    def test_buy_fill_at_ask_plus_slippage(self) -> None:
        state = self._state()
        req = ExecutionRequest("BTC", "BUY", "DOWN", 1000, 1000, 100.0)
        result = self.sim.simulate(req, state)
        assert result.executed
        assert result.fill_price == pytest.approx(101.0 * 1.0002, rel=1e-4)

    def test_sell_fill_at_bid_minus_slippage(self) -> None:
        state = self._state()
        req = ExecutionRequest("BTC", "SELL", "UP", 1000, 1000, 100.0)
        result = self.sim.simulate(req, state)
        assert result.executed
        assert result.fill_price == pytest.approx(99.0 * 0.9998, rel=1e-4)

    def test_fee_on_buy(self) -> None:
        state = self._state()
        req = ExecutionRequest("BTC", "BUY", "DOWN", 1000, 1000, 100.0)
        result = self.sim.simulate(req, state)
        assert result.fee_amount > 0

    def test_missing_bbo_blocks(self) -> None:
        state = CoinMarketState(coin="BTC")
        req = ExecutionRequest("BTC", "BUY", "DOWN", 1000, 1000, 100.0)
        result = self.sim.simulate(req, state)
        assert result.blocked
        assert result.blocked_reason == "missing_or_stale_bbo"

    def test_no_midpoint_fills_by_default(self) -> None:
        state = CoinMarketState(coin="BTC")
        state.midpoint = 100.0
        state.bbo_update_timestamp_ms = 1000
        req = ExecutionRequest("BTC", "BUY", "DOWN", 1000, 1000, 100.0)
        result = self.sim.simulate(req, state)
        assert result.blocked


class TestPortfolio:
    def setup_method(self) -> None:
        self.profile = get_profile("contrarian_v1_medium_no_trend_filter")
        self.portfolio = EventPortfolio(profile=self.profile, limits=self.profile.trading_limits())

    def test_no_shorting(self) -> None:
        pos = self.portfolio._get_position("BTC")
        assert not pos.is_open

    def test_buy_block_insufficient_cash(self) -> None:
        self.portfolio.cash_balance = 1.0
        from event_backtester.event_execution_simulator import ExecutionResult

        execution = ExecutionResult(
            executed=True,
            blocked=False,
            blocked_reason=None,
            requested_side="BUY",
            fill_side="BUY",
            reference_bbo=None,
            fill_price=100.0,
            slippage_amount=0.0,
            fee_amount=0.05,
            event_timestamp_ms=1000,
            received_timestamp_ms=1000,
            reason="test",
            quantity=1.0,
            trade_value=100.0,
        )
        assert not self.portfolio.try_buy(coin="BTC", signal="DOWN", execution=execution, timestamp_ms=1000)

    def test_risk_reducing_sell_after_trade_limit(self) -> None:
        limits = self.profile.trading_limits()
        self.portfolio.successful_trades_this_session = limits.max_successful_trades_per_session
        from event_backtester.event_execution_simulator import ExecutionResult

        # Open position manually
        from paper_trader import Position

        pos = Position()
        pos.quantity = 1.0
        pos.avg_entry_price = 100.0
        pos.entry_count = 1
        pos.opened_at = 1.0
        self.portfolio.positions["BTC"] = pos
        execution = ExecutionResult(
            executed=True,
            blocked=False,
            blocked_reason=None,
            requested_side="SELL",
            fill_side="SELL",
            reference_bbo={"best_bid_price": 99.0},
            fill_price=99.0,
            slippage_amount=0.01,
            fee_amount=0.0,
            event_timestamp_ms=2000,
            received_timestamp_ms=2000,
            reason="test",
        )
        assert self.portfolio.try_sell(coin="BTC", signal="UP", execution=execution, timestamp_ms=2000)


class TestNoFutureLeakage:
    def test_cursor_peek_only_one_event(self) -> None:
        events = []
        for i in range(5):
            trade = TradeEvent(
                coin="BTC",
                side="B",
                price=100.0 + i,
                size=0.1,
                trade_id=i,
                transaction_hash="0x",
                exchange_timestamp_ms=1000 + i,
                received_timestamp_ms=1000 + i,
            )
            events.append(_replay_event(trade, "trade", 1000 + i, i + 1))
        cursor = ReplayCursor(events)
        assert cursor.events_ahead_count() == 1
        peek = cursor.peek_event()
        assert peek is not None
        assert peek.replay_sequence == 1
        nxt = cursor.next_event()
        assert nxt is not None
        assert cursor.peek_event().replay_sequence == 2  # type: ignore[union-attr]

    def test_strategy_cannot_access_cursor_internals(self) -> None:
        source = (EVENT_BACKTESTER_DIR / "event_strategy_adapter.py").read_text(encoding="utf-8")
        assert "ReplayCursor" not in source
        assert "peek_event" not in source


class TestDeterministicHash:
    def test_repeat_run_same_hash(self) -> None:
        profile = get_profile("contrarian_v1_medium_no_trend_filter")
        payload = {
            "stream_hash": "abc",
            "profile_dict": profile.to_dict(),
            "strategy_version": profile.strategy_version,
            "cost_assumptions": {"trading_fee_pct": 0.05},
            "fill_model": "bid_ask_v1",
            "coins": ["BTC"],
            "channels": ["trades", "bbo"],
            "trade_log": [],
            "blocked_signals": [],
            "final_equity": 1000.0,
        }
        h1 = compute_deterministic_backtest_hash(**payload)
        h2 = compute_deterministic_backtest_hash(**payload)
        assert h1 == h2


class TestEngineIntegration:
    def test_dry_run_writes_report(self, fixture_stream: tuple, tmp_path: Path) -> None:
        stream_dir, _session_id, replay_id = fixture_stream
        output_dir = tmp_path / "event_backtests"
        config = EventBacktestConfig(
            replay_session_id=replay_id,
            coins=("BTC",),
            channels=("trades", "bbo"),
            output_dir=output_dir,
            dry_run=True,
            market_stream_dir=stream_dir,
            test_mode=True,
            min_completed_trades=0,
            min_session_duration_ms=0,
        )
        result = EventBacktestEngine(config).run("test_dry_run_id")
        assert result.report["dry_run"] is True
        assert result.report_paths["json"].exists()
        assert result.report_paths["markdown"].exists()
        assert result.report["paper_only"] is True
        assert result.report["execution_enabled"] is False

    def test_full_backtest_offline(self, fixture_stream: tuple, tmp_path: Path) -> None:
        stream_dir, _session_id, replay_id = fixture_stream
        output_dir = tmp_path / "event_backtests"
        config = EventBacktestConfig(
            replay_session_id=replay_id,
            coins=("BTC",),
            channels=("trades", "bbo"),
            output_dir=output_dir,
            dry_run=False,
            market_stream_dir=stream_dir,
            test_mode=True,
            min_completed_trades=0,
            min_session_duration_ms=0,
            min_bbo_availability_pct=0,
        )
        result = EventBacktestEngine(config).run("test_full_id")
        assert result.report["schema_version"] == EVENT_BACKTEST_SCHEMA_VERSION
        assert result.report["events_processed"] > 0

    def test_determinism_repeat(self, fixture_stream: tuple, tmp_path: Path) -> None:
        stream_dir, _session_id, replay_id = fixture_stream
        output_dir = tmp_path / "event_backtests"
        base = dict(
            replay_session_id=replay_id,
            coins=("BTC",),
            channels=("trades", "bbo"),
            output_dir=output_dir,
            dry_run=False,
            market_stream_dir=stream_dir,
            test_mode=True,
            min_completed_trades=0,
            min_session_duration_ms=0,
            min_bbo_availability_pct=0,
        )
        r1 = EventBacktestEngine(EventBacktestConfig(**base)).run("det1")
        r2 = EventBacktestEngine(EventBacktestConfig(**base)).run("det2")
        assert r1.deterministic_backtest_hash == r2.deterministic_backtest_hash


class TestReports:
    def test_json_and_markdown_output(self, tmp_path: Path) -> None:
        report = {
            "schema_version": EVENT_BACKTEST_SCHEMA_VERSION,
            "event_backtest_id": "test123",
            "replay_session_id": "replay1",
            "source_recorder_session_id": "rec1",
            "strategy_name": "contrarian_v1",
            "strategy_version": "1.0.0",
            "profile_name": "contrarian_v1_medium_no_trend_filter",
            "coins": ["BTC"],
            "channels": ["trades", "bbo"],
            "starting_equity": 1000.0,
            "ending_equity": 1000.0,
            "net_return_pct": 0.0,
            "completed_trades": 0,
            "blocked_signals": 0,
            "blocked_reasons": {},
            "evidence_label": EVIDENCE_INSUFFICIENT,
            "warnings": [],
            "limitations": [],
            "paper_only": PAPER_ONLY,
            "execution_enabled": False,
            "deterministic_backtest_hash": "abc",
            "stream_hash": "def",
        }
        paths = write_event_backtest_report(report, tmp_path)
        assert paths["json"].exists()
        md = paths["markdown"].read_text(encoding="utf-8")
        assert "Event-Driven Backtest Report" in md
        listed = list_event_backtest_reports(tmp_path)
        assert len(listed) == 1


class TestDashboardLoader:
    def test_load_event_backtest_reports_empty(self, tmp_path: Path, monkeypatch) -> None:
        import dashboard_data

        monkeypatch.setattr(dashboard_data, "EVENT_BACKTESTS_DIR", tmp_path / "event_backtests")
        assert dashboard_data.load_event_backtest_reports() == []


class TestSafety:
    def test_no_forbidden_patterns_in_event_backtester(self) -> None:
        skip = {"event_safety.py"}
        for path in EVENT_BACKTESTER_DIR.rglob("*.py"):
            if path.name in skip:
                continue
            source = path.read_text(encoding="utf-8").lower()
            for pattern in FORBIDDEN_SOURCE_PATTERNS:
                assert pattern not in source, f"{pattern} in {path}"

    def test_no_network_in_engine(self) -> None:
        source = (EVENT_BACKTESTER_DIR / "event_backtest_engine.py").read_text(encoding="utf-8")
        assert "requests" not in source
        assert "urllib" not in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 0)[0] not in FORBIDDEN_MODULE_NAMES

    def test_evidence_defaults_insufficient(self) -> None:
        label = assign_evidence_label(
            completed_trades=0,
            session_duration_ms=1000,
            bbo_availability_pct=10.0,
            quality_label="DEGRADED",
            results_by_coin={},
            min_completed_trades=3,
            min_session_duration_ms=60_000,
            min_bbo_availability_pct=50.0,
            net_return_pct=0.0,
            profit_factor=None,
            max_drawdown_pct=0.0,
        )
        assert label == EVIDENCE_INSUFFICIENT

    def test_zero_trade_metrics(self) -> None:
        metrics = compute_completed_trade_metrics([])
        assert metrics.completed_trades == 0
        assert metrics.profit_factor is None


class TestEventFiltering:
    def test_trades_and_bbo_only(self, fixture_stream: tuple) -> None:
        stream_dir, session_id, replay_id = fixture_stream
        engine = ReplayEngine(market_stream_dir=stream_dir, test_mode=True)
        session = engine.load_session(session_id)
        filters = ReplayFilters(coins=["BTC"], channels=["trades", "bbo"], include_candles=False, include_l2_book=False)
        events, _, _ = engine.prepare_events(session, filters, replay_id)
        types = {e.original_event_type for e in events}
        assert types <= {"trade", "bbo"}


class TestLiveDataProtection:
    def test_output_only_under_event_backtests(self, fixture_stream: tuple, tmp_path: Path) -> None:
        stream_dir, _session_id, replay_id = fixture_stream
        live_replay = stream_dir / "replays" / f"{replay_id}.json"
        mtime_before = live_replay.stat().st_mtime
        config = EventBacktestConfig(
            replay_session_id=replay_id,
            coins=("BTC",),
            channels=("trades", "bbo"),
            output_dir=tmp_path / "event_backtests",
            dry_run=True,
            market_stream_dir=stream_dir,
            test_mode=True,
            min_completed_trades=0,
            min_session_duration_ms=0,
        )
        EventBacktestEngine(config).run()
        assert live_replay.stat().st_mtime == mtime_before


class TestExistingFrameworkUnchanged:
    def test_experiment_runner_still_imports(self) -> None:
        import experiment_runner  # noqa: F401

    def test_paper_trader_unchanged(self) -> None:
        from paper_trader import PaperTrader

        trader = PaperTrader()
        assert trader.cash_balance > 0
