"""Event-driven backtest engine — consumes replay engine output."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from event_backtester.backtest_config import EventBacktestConfig
from event_backtester.event_execution_simulator import EventExecutionSimulator, ExecutionRequest
from event_backtester.event_metrics import (
    assign_evidence_label,
    compute_bbo_stats,
    compute_completed_trade_metrics,
    compute_holding_times,
    compute_results_by_coin,
    summarize_blocked_signals,
)
from event_backtester.event_portfolio import EventPortfolio
from event_backtester.event_report import (
    build_event_backtest_report,
    compute_deterministic_backtest_hash,
    new_event_backtest_id,
    write_event_backtest_report,
)
from event_backtester.event_strategy_adapter import EventStrategyAdapter
from event_backtester.market_state import MarketStateManager
from market_data.replay_cursor import ReplayCursor
from market_data.replay_engine import ReplayEngine
from market_data.replay_events import compute_stream_hash
from market_data.replay_filters import ReplayFilters
from market_data.replay_report import list_replay_reports
from strategy_profiles import get_profile


class EventBacktestError(Exception):
    """Raised when event backtest cannot proceed safely."""


@dataclass
class EventBacktestResult:
    report: dict[str, Any]
    report_paths: dict[str, Path]
    events_processed: int
    deterministic_backtest_hash: str


@dataclass
class _RunStats:
    bbo_events: int = 0
    trade_events: int = 0
    bbo_available_at_trade: int = 0
    stale_bbo_count: int = 0
    missing_bbo_count: int = 0
    first_event_ms: int | None = None
    last_event_ms: int | None = None
    warnings: list[str] = field(default_factory=list)


def _load_replay_report(replay_session_id: str, replays_dir: Path) -> dict[str, Any]:
    path = replays_dir / f"{replay_session_id}.json"
    if not path.exists():
        raise EventBacktestError(f"Replay report not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_replay_session(config: EventBacktestConfig) -> tuple[str, dict[str, Any] | None]:
    replays_dir = config.market_stream_dir / "replays"

    if config.replay_session_id:
        report = _load_replay_report(config.replay_session_id, replays_dir)
        return config.replay_session_id, report

    if config.use_latest_replay:
        reports = list_replay_reports(replays_dir)
        if not reports:
            raise EventBacktestError("No replay reports found for --latest-replay")
        replay_id = reports[0]["replay_session_id"]
        return replay_id, _load_replay_report(replay_id, replays_dir)

    if config.source_recorder_session_id:
        return uuid.uuid4().hex, None

    raise EventBacktestError(
        "Provide --replay-session-id, --source-recorder-session-id, or --latest-replay"
    )


def _build_filters(config: EventBacktestConfig, replay_report: dict[str, Any] | None) -> ReplayFilters:
    replay_filters = (replay_report or {}).get("filters") or {}
    return ReplayFilters(
        coins=list(config.coins),
        channels=list(config.channels),
        max_events=config.max_events or replay_filters.get("max_events"),
        include_candles=False,
        include_l2_book=False,
    )


def _comparison_section(profile_name: str, strategy_name: str) -> dict[str, str]:
    return {
        "row_based_profile_name": profile_name,
        "event_based_strategy_name": strategy_name,
        "dataset_source_difference": (
            "Row-based experiments replay CSV price_log timestamp rows; "
            "event-based backtests replay recorded WebSocket trades and BBO streams."
        ),
        "row_based_backtest_type": "timestamp-row replay",
        "event_based_backtest_type": "streamed market-data replay",
        "results_expectation": (
            "Results are not expected to match exactly between frameworks."
        ),
        "fill_model_difference": (
            "Event-based fills use bid/ask plus slippage; "
            "row-based experiments use midpoint/last price plus slippage."
        ),
        "correctness_claim": (
            "Neither framework is claimed more correct without further evidence."
        ),
    }


class EventBacktestEngine:
    """Runs event-driven backtests over deterministic replay streams."""

    def __init__(self, config: EventBacktestConfig) -> None:
        self._config = config
        self._replay_engine = ReplayEngine(
            market_stream_dir=config.market_stream_dir,
            test_mode=config.test_mode,
        )

    def run(self, event_backtest_id: str | None = None) -> EventBacktestResult:
        config = self._config
        backtest_id = event_backtest_id or new_event_backtest_id()

        replay_session_id, replay_report = _resolve_replay_session(config)
        filters = _build_filters(config, replay_report)

        if replay_report:
            source_recorder_id = replay_report["source_recorder_session_id"]
        else:
            source_recorder_id = config.source_recorder_session_id
            if not source_recorder_id:
                raise EventBacktestError("source_recorder_session_id required for internal replay")

        session = self._replay_engine.load_session(source_recorder_id)
        events, _skipped, session = self._replay_engine.prepare_events(
            session,
            filters,
            replay_session_id=replay_session_id,
        )

        stream_hash = compute_stream_hash(events)
        if replay_report and replay_report.get("stream_hash") and not config.test_mode:
            expected = replay_report["stream_hash"]
            if expected != stream_hash:
                # Filter differences may change hash; warn but continue
                pass

        quality_report = session.quality_report
        quality_label = (quality_report or {}).get("quality_label")
        quality_path = str(session.quality_report_path) if session.quality_report_path else None

        profile = get_profile(config.strategy_profile_id)
        profile_dict = profile.to_dict()
        limits = replace(profile.trading_limits(), starting_balance_usdc=config.starting_equity)

        market_state = MarketStateManager(config.coins)
        portfolio = EventPortfolio(profile=profile, limits=limits)
        adapter = EventStrategyAdapter(profile)
        simulator = EventExecutionSimulator(
            profile,
            bbo_stale_ms=config.bbo_stale_ms,
            allow_midpoint_fills=config.allow_midpoint_fills,
        )

        stats = _RunStats()
        cursor = ReplayCursor(events)

        if config.dry_run:
            return self._dry_run_result(
                backtest_id=backtest_id,
                replay_session_id=replay_session_id,
                source_recorder_id=source_recorder_id,
                quality_path=quality_path,
                quality_label=quality_label,
                profile=profile,
                profile_dict=profile_dict,
                adapter=adapter,
                simulator=simulator,
                cursor=cursor,
                market_state=market_state,
                stream_hash=stream_hash,
                stats=stats,
                config=config,
            )

        while not cursor.at_end():
            event = cursor.next_event()
            if event is None:
                break

            portfolio.events_processed += 1
            if stats.first_event_ms is None:
                stats.first_event_ms = event.primary_time_ms
            stats.last_event_ms = event.primary_time_ms

            changed = market_state.update_from_event(event)

            if event.original_event_type == "bbo":
                stats.bbo_events += 1
            elif event.original_event_type == "trade":
                stats.trade_events += 1
                coin = event.coin
                if coin and market_state.bbo_available(coin, event.primary_time_ms, config.bbo_stale_ms):
                    stats.bbo_available_at_trade += 1
                elif coin:
                    reason = market_state.bbo_stale_reason(coin, event.primary_time_ms, config.bbo_stale_ms)
                    if reason == "missing_or_stale_bbo":
                        if market_state.get_state(coin) and market_state.get_state(coin).bbo_update_timestamp_ms:
                            stats.stale_bbo_count += 1
                        else:
                            stats.missing_bbo_count += 1

                signal = adapter.evaluate_on_trade(
                    event=event,
                    market_state=market_state,
                    portfolio=portfolio,
                )
                if signal is not None:
                    action = adapter.action_for_signal(signal)
                    if action is not None and changed:
                        state = market_state.get_state(coin)  # type: ignore[arg-type]
                        request = ExecutionRequest(
                            coin=coin,  # type: ignore[arg-type]
                            side=action,
                            signal=signal.signal,
                            event_timestamp_ms=event.primary_time_ms,
                            received_timestamp_ms=event.received_timestamp_ms,
                            reference_price=state.latest_trade_price if state else None,
                        )
                        execution = simulator.simulate(request, state)  # type: ignore[arg-type]
                        if action == "BUY":
                            portfolio.try_buy(
                                coin=coin,  # type: ignore[arg-type]
                                signal=signal.signal,
                                execution=execution,
                                timestamp_ms=event.primary_time_ms,
                            )
                        else:
                            portfolio.try_sell(
                                coin=coin,  # type: ignore[arg-type]
                                signal=signal.signal,
                                execution=execution,
                                timestamp_ms=event.primary_time_ms,
                            )

            prices = market_state.current_prices()
            portfolio.update_session_risk(prices)
            portfolio.update_equity_tracking(prices)

        return self._build_result(
            backtest_id=backtest_id,
            replay_session_id=replay_session_id,
            source_recorder_id=source_recorder_id,
            quality_path=quality_path,
            quality_label=quality_label,
            profile=profile,
            profile_dict=profile_dict,
            adapter=adapter,
            simulator=simulator,
            portfolio=portfolio,
            market_state=market_state,
            stream_hash=stream_hash,
            stats=stats,
            dry_run=False,
        )

    def _dry_run_result(
        self,
        *,
        backtest_id: str,
        replay_session_id: str,
        source_recorder_id: str,
        quality_path: str | None,
        quality_label: str | None,
        profile: Any,
        profile_dict: dict[str, Any],
        adapter: EventStrategyAdapter,
        simulator: EventExecutionSimulator,
        cursor: ReplayCursor,
        market_state: MarketStateManager,
        stream_hash: str,
        stats: _RunStats,
        config: EventBacktestConfig,
    ) -> EventBacktestResult:
        usable_trades = 0
        while not cursor.at_end():
            event = cursor.next_event()
            if event is None:
                break
            stats.first_event_ms = stats.first_event_ms or event.primary_time_ms
            stats.last_event_ms = event.primary_time_ms
            market_state.update_from_event(event)
            if event.original_event_type == "bbo":
                stats.bbo_events += 1
            elif event.original_event_type == "trade":
                stats.trade_events += 1
                usable_trades += 1
                coin = event.coin
                if coin and market_state.bbo_available(coin, event.primary_time_ms, config.bbo_stale_ms):
                    stats.bbo_available_at_trade += 1

        stats.warnings.append("Dry-run mode: no trades simulated")

        empty_portfolio = EventPortfolio(
            profile=profile,
            limits=replace(profile.trading_limits(), starting_balance_usdc=config.starting_equity),
        )
        empty_portfolio.events_processed = cursor.total_events

        return self._build_result(
            backtest_id=backtest_id,
            replay_session_id=replay_session_id,
            source_recorder_id=source_recorder_id,
            quality_path=quality_path,
            quality_label=quality_label,
            profile=profile,
            profile_dict=profile_dict,
            adapter=adapter,
            simulator=simulator,
            portfolio=empty_portfolio,
            market_state=market_state,
            stream_hash=stream_hash,
            stats=stats,
            dry_run=True,
        )

    def _build_result(
        self,
        *,
        backtest_id: str,
        replay_session_id: str,
        source_recorder_id: str,
        quality_path: str | None,
        quality_label: str | None,
        profile: Any,
        profile_dict: dict[str, Any],
        adapter: EventStrategyAdapter,
        simulator: EventExecutionSimulator,
        portfolio: EventPortfolio,
        market_state: MarketStateManager,
        stream_hash: str,
        stats: _RunStats,
        dry_run: bool,
    ) -> EventBacktestResult:
        config = self._config
        prices = market_state.current_prices()
        ending_equity = portfolio.total_equity(prices)
        unrealized = portfolio.unrealized_pnl(prices)
        trade_metrics = compute_completed_trade_metrics(portfolio.trade_log)
        blocked_summary = summarize_blocked_signals(portfolio.blocked_signals)
        results_by_coin = compute_results_by_coin(portfolio.trade_log)
        bbo_stats = compute_bbo_stats(
            events_processed=portfolio.events_processed or (stats.trade_events + stats.bbo_events),
            bbo_events=stats.bbo_events,
            trade_events=stats.trade_events,
            bbo_available_at_trade=stats.bbo_available_at_trade,
            stale_bbo_count=stats.stale_bbo_count,
            missing_bbo_count=stats.missing_bbo_count,
        )

        session_duration_ms = 0
        if stats.first_event_ms is not None and stats.last_event_ms is not None:
            session_duration_ms = stats.last_event_ms - stats.first_event_ms

        exposure_pct = 0.0
        if portfolio.events_processed > 0:
            exposure_pct = (portfolio.exposure_timestamps / portfolio.events_processed) * 100.0

        max_dd = portfolio.drawdown(prices)
        net_return = (
            ((ending_equity - config.starting_equity) / config.starting_equity * 100.0)
            if config.starting_equity
            else 0.0
        )

        evidence = assign_evidence_label(
            completed_trades=trade_metrics.completed_trades,
            session_duration_ms=session_duration_ms,
            bbo_availability_pct=bbo_stats.availability_pct,
            quality_label=quality_label,
            results_by_coin=results_by_coin,
            min_completed_trades=config.min_completed_trades,
            min_session_duration_ms=config.min_session_duration_ms,
            min_bbo_availability_pct=config.min_bbo_availability_pct,
            net_return_pct=net_return,
            profit_factor=trade_metrics.profit_factor,
            max_drawdown_pct=max_dd,
        )

        limitations = [
            "v1 supports trades and BBO channels only.",
            "No shorting, leverage, or margin.",
            "Smoke datasets (~300 seconds) produce few completed trades.",
            "BBO staleness threshold may block fills during sparse BBO coverage.",
            "Results are research-only and not trading authorization.",
        ]
        limitations.extend(adapter.adapter_notes())

        warnings = list(stats.warnings)
        if quality_label not in ("GOOD_FOR_RESEARCH", "USABLE_WITH_WARNINGS"):
            warnings.append(f"Quality label is {quality_label!r}")
        if session_duration_ms < config.min_session_duration_ms:
            warnings.append("Session duration is shorter than recommended minimum")
        if bbo_stats.availability_pct < config.min_bbo_availability_pct:
            warnings.append("BBO availability is below recommended threshold")

        trade_log_dicts = [t.to_dict() for t in portfolio.trade_log]
        blocked_dicts = [b.to_dict() for b in portfolio.blocked_signals]

        det_hash = compute_deterministic_backtest_hash(
            stream_hash=stream_hash,
            profile_dict=profile_dict,
            strategy_version=profile.strategy_version,
            cost_assumptions=simulator.cost_assumptions(),
            fill_model=simulator.fill_model,
            coins=list(config.coins),
            channels=list(config.channels),
            trade_log=trade_log_dicts,
            blocked_signals=blocked_dicts,
            final_equity=ending_equity,
        )

        report = build_event_backtest_report(
            event_backtest_id=backtest_id,
            replay_session_id=replay_session_id,
            source_recorder_session_id=source_recorder_id,
            quality_report_path=quality_path,
            quality_label=quality_label,
            strategy_name=profile.strategy_name,
            strategy_version=profile.strategy_version,
            profile_name=profile.profile_id,
            profile_configuration=profile_dict,
            coins=list(config.coins),
            channels=list(config.channels),
            starting_equity=config.starting_equity,
            ending_equity=ending_equity,
            realized_pnl=portfolio.realized_pnl,
            unrealized_pnl=unrealized,
            total_fees=portfolio.total_fees,
            total_slippage_cost=portfolio.total_slippage_cost,
            gross_pnl_before_costs=trade_metrics.gross_pnl_before_costs,
            completed_trades=trade_metrics.completed_trades,
            buys=trade_metrics.buys,
            sells=trade_metrics.sells,
            blocked_signals=blocked_summary["total_blocked"],
            blocked_reasons=blocked_summary["blocked_reasons"],
            win_rate=trade_metrics.win_rate,
            average_win=trade_metrics.average_win,
            average_loss=trade_metrics.average_loss,
            average_net_return_per_trade=trade_metrics.average_net_return_per_trade,
            profit_factor=trade_metrics.profit_factor,
            maximum_drawdown=max_dd,
            exposure_time_pct=exposure_pct,
            average_holding_time_seconds=compute_holding_times(portfolio.trade_log),
            results_by_coin=results_by_coin,
            events_processed=portfolio.events_processed,
            bbo_availability_pct=bbo_stats.availability_pct,
            stale_bbo_count=bbo_stats.stale_bbo_count,
            missing_bbo_count=bbo_stats.missing_bbo_count,
            stream_hash=stream_hash,
            deterministic_backtest_hash=det_hash,
            warnings=warnings,
            limitations=limitations,
            evidence_label=evidence,
            dry_run=dry_run,
            adapter_notes=adapter.adapter_notes(),
            comparison_section=_comparison_section(profile.profile_id, profile.strategy_name),
        )

        report_paths = write_event_backtest_report(report, config.output_dir)
        return EventBacktestResult(
            report=report,
            report_paths=report_paths,
            events_processed=portfolio.events_processed,
            deterministic_backtest_hash=det_hash,
        )
