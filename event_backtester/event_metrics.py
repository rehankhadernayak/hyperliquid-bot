"""Metrics and evidence labels for event-driven backtests."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from event_backtester.event_portfolio import EventPortfolio, EventTradeRecord
from event_backtester.market_state import MarketStateManager

EVIDENCE_INSUFFICIENT = "INSUFFICIENT_EVENT_DATA"
EVIDENCE_TECHNICAL_PASS_WEAK = "TECHNICAL_PASS_STRATEGY_WEAK"
EVIDENCE_PROMISING = "PROMISING_BUT_NEEDS_MORE_DATA"
EVIDENCE_UNSTABLE = "UNSTABLE"
EVIDENCE_REJECT = "REJECT"


@dataclass
class BboStats:
    total_events: int = 0
    bbo_events: int = 0
    trade_events: int = 0
    bbo_available_at_trade: int = 0
    stale_bbo_count: int = 0
    missing_bbo_count: int = 0

    @property
    def availability_pct(self) -> float:
        if self.trade_events <= 0:
            return 100.0 if self.bbo_events > 0 else 0.0
        return (self.bbo_available_at_trade / self.trade_events) * 100.0


@dataclass
class CompletedTradeMetrics:
    completed_trades: int = 0
    buys: int = 0
    sells: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    average_net_return_per_trade: float = 0.0
    profit_factor: float | None = None
    gross_pnl_before_costs: float = 0.0


def compute_bbo_stats(
    *,
    events_processed: int,
    bbo_events: int,
    trade_events: int,
    bbo_available_at_trade: int,
    stale_bbo_count: int,
    missing_bbo_count: int,
) -> BboStats:
    return BboStats(
        total_events=events_processed,
        bbo_events=bbo_events,
        trade_events=trade_events,
        bbo_available_at_trade=bbo_available_at_trade,
        stale_bbo_count=stale_bbo_count,
        missing_bbo_count=missing_bbo_count,
    )


def _pair_completed_trades(trade_log: list[EventTradeRecord]) -> list[tuple[EventTradeRecord, EventTradeRecord]]:
    """Pair BUY followed by SELL per coin for round-trip metrics."""
    open_buys: dict[str, EventTradeRecord] = {}
    pairs: list[tuple[EventTradeRecord, EventTradeRecord]] = []
    for trade in trade_log:
        if trade.action == "BUY":
            open_buys[trade.coin] = trade
        elif trade.action == "SELL" and trade.coin in open_buys:
            pairs.append((open_buys.pop(trade.coin), trade))
    return pairs


def compute_completed_trade_metrics(trade_log: list[EventTradeRecord]) -> CompletedTradeMetrics:
    buys = sum(1 for t in trade_log if t.action == "BUY")
    sells = sum(1 for t in trade_log if t.action == "SELL")
    pairs = _pair_completed_trades(trade_log)

    net_returns: list[float] = []
    gross_pnls: list[float] = []
    for buy, sell in pairs:
        cost_basis = buy.trade_value + buy.fee + buy.slippage
        proceeds = sell.trade_value - sell.fee - sell.slippage
        gross = sell.trade_value - buy.trade_value
        net = proceeds - cost_basis
        gross_pnls.append(gross)
        if cost_basis > 0:
            net_returns.append((net / cost_basis) * 100.0)

    wins = [r for r in net_returns if r > 0]
    losses = [r for r in net_returns if r <= 0]

    win_rate = (len(wins) / len(net_returns) * 100.0) if net_returns else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    avg_net = sum(net_returns) / len(net_returns) if net_returns else 0.0

    gross_wins = sum(r for r in net_returns if r > 0)
    gross_losses = abs(sum(r for r in net_returns if r <= 0))
    profit_factor: float | None = None
    if gross_losses > 0:
        profit_factor = gross_wins / gross_losses
    elif gross_wins > 0:
        profit_factor = float("inf")

    return CompletedTradeMetrics(
        completed_trades=len(pairs),
        buys=buys,
        sells=sells,
        wins=len(wins),
        losses=len(losses),
        win_rate=win_rate,
        average_win=avg_win,
        average_loss=avg_loss,
        average_net_return_per_trade=avg_net,
        profit_factor=profit_factor if profit_factor != float("inf") else None,
        gross_pnl_before_costs=sum(gross_pnls),
    )


def compute_holding_times(
    trade_log: list[EventTradeRecord],
) -> float:
    pairs = _pair_completed_trades(trade_log)
    if not pairs:
        return 0.0
    durations = [(sell.timestamp_ms - buy.timestamp_ms) / 1000.0 for buy, sell in pairs]
    return sum(durations) / len(durations)


def compute_results_by_coin(trade_log: list[EventTradeRecord]) -> dict[str, Any]:
    by_coin: dict[str, dict[str, Any]] = {}
    for trade in trade_log:
        entry = by_coin.setdefault(
            trade.coin,
            {"buys": 0, "sells": 0, "realized_pnl": 0.0, "fees": 0.0, "slippage": 0.0},
        )
        if trade.action == "BUY":
            entry["buys"] += 1
        else:
            entry["sells"] += 1
            entry["realized_pnl"] += trade.trade_pnl
        entry["fees"] += trade.fee
        entry["slippage"] += trade.slippage
    return by_coin


def assign_evidence_label(
    *,
    completed_trades: int,
    session_duration_ms: int,
    bbo_availability_pct: float,
    quality_label: str | None,
    results_by_coin: dict[str, Any],
    min_completed_trades: int,
    min_session_duration_ms: int,
    min_bbo_availability_pct: float,
    net_return_pct: float,
    profit_factor: float | None,
    max_drawdown_pct: float,
) -> str:
    """Conservative evidence label — defaults to INSUFFICIENT_EVENT_DATA."""
    reasons_insufficient = []

    if completed_trades < min_completed_trades:
        reasons_insufficient.append("completed_trades_below_minimum")
    if session_duration_ms < min_session_duration_ms:
        reasons_insufficient.append("session_duration_too_short")
    if bbo_availability_pct < min_bbo_availability_pct:
        reasons_insufficient.append("bbo_availability_poor")
    if quality_label not in ("GOOD_FOR_RESEARCH", "USABLE_WITH_WARNINGS"):
        reasons_insufficient.append("quality_label_not_acceptable")

    trade_counts = [
        (coin, data.get("buys", 0) + data.get("sells", 0))
        for coin, data in results_by_coin.items()
    ]
    if trade_counts:
        total = sum(c for _, c in trade_counts)
        if total > 0:
            dominant = max(trade_counts, key=lambda x: x[1])
            if dominant[1] / total > 0.8 and len(trade_counts) > 1:
                reasons_insufficient.append("one_coin_dominates_trades")

    if reasons_insufficient:
        return EVIDENCE_INSUFFICIENT

    if net_return_pct < -5.0 or max_drawdown_pct > 20.0:
        return EVIDENCE_REJECT

    if profit_factor is not None and profit_factor < 0.5:
        return EVIDENCE_REJECT

    if completed_trades < min_completed_trades * 3:
        return EVIDENCE_TECHNICAL_PASS_WEAK

    if profit_factor is not None and 0.8 <= profit_factor <= 1.2:
        return EVIDENCE_UNSTABLE

    if net_return_pct > 0 and (profit_factor or 0) >= 1.0:
        return EVIDENCE_PROMISING

    return EVIDENCE_TECHNICAL_PASS_WEAK


def summarize_blocked_signals(blocked: list[Any]) -> dict[str, Any]:
    reasons = Counter()
    for record in blocked:
        reasons[getattr(record, "reason", "unknown")] += 1
    return {
        "total_blocked": len(blocked),
        "blocked_reasons": dict(sorted(reasons.items(), key=lambda x: -x[1])),
    }
