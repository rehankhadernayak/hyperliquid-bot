"""Performance metrics for strategy experiment runs."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from backtester import compute_max_drawdown, summarize_by_coin
from config import MARKET
from experiment_trader import ExperimentPaperTrader
from strategy_types import ProfileTradeRecord, TradingLimits


def _profit_factor(sell_trades: list[ProfileTradeRecord]) -> float | None:
    gross_wins = sum(
        trade.realized_pnl for trade in sell_trades if trade.realized_pnl > 0
    )
    gross_losses = abs(
        sum(trade.realized_pnl for trade in sell_trades if trade.realized_pnl < 0)
    )
    if gross_losses == 0:
        return None if gross_wins == 0 else float("inf")
    return gross_wins / gross_losses


def _longest_losing_sequence(sell_trades: list[ProfileTradeRecord]) -> int:
    longest = 0
    current = 0
    for trade in sell_trades:
        if trade.realized_pnl < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _serialize_trades(trades: list[ProfileTradeRecord]) -> list[dict[str, Any]]:
    return [asdict(trade) for trade in trades]


def _results_by_coin_from_trades(
    trades: list[ProfileTradeRecord],
) -> dict[str, dict[str, Any]]:
    legacy = [
        {
            "coin": trade.coin,
            "action": trade.action,
            "realized_pnl": trade.realized_pnl,
        }
        for trade in trades
    ]
    base = summarize_by_coin(legacy)
    for coin in MARKET.watchlist:
        coin_trades = [trade for trade in trades if trade.coin == coin]
        base[coin]["fees_paid"] = round(sum(trade.fee for trade in coin_trades), 2)
        base[coin]["slippage_cost"] = round(
            sum(trade.slippage for trade in coin_trades), 2
        )
    return base


def compute_period_metrics(
    trader: ExperimentPaperTrader,
    equity_curve: list[float],
    current_prices: dict[str, float],
    *,
    period_label: str,
    limits: TradingLimits,
) -> dict[str, Any]:
    """Compute the full metric set for one evaluation period."""
    sell_trades = [trade for trade in trader.trade_log if trade.action == "SELL"]
    buy_trades = [trade for trade in trader.trade_log if trade.action == "BUY"]

    winning = [trade for trade in sell_trades if trade.realized_pnl > 0]
    losing = [trade for trade in sell_trades if trade.realized_pnl < 0]
    closed_trades = len(winning) + len(losing)

    ending_equity = trader.total_equity(current_prices)
    starting_equity = limits.starting_balance_usdc
    net_return_pct = (
        ((ending_equity - starting_equity) / starting_equity) * 100
        if starting_equity
        else 0.0
    )

    gross_profit = sum(trade.realized_pnl for trade in sell_trades if trade.realized_pnl > 0)
    pf = _profit_factor(sell_trades)
    exposure_time_pct = 0.0
    if trader.run_state.total_timestamps > 0:
        exposure_time_pct = (
            trader.run_state.exposure_timestamps / trader.run_state.total_timestamps
        ) * 100

    avg_holding_seconds = 0.0
    if trader.run_state.completed_holding_seconds:
        avg_holding_seconds = sum(trader.run_state.completed_holding_seconds) / len(
            trader.run_state.completed_holding_seconds
        )

    return_per_trade = 0.0
    if closed_trades:
        return_per_trade = trader.realized_pnl / closed_trades

    return {
        "period_label": period_label,
        "starting_equity": round(starting_equity, 2),
        "ending_equity": round(ending_equity, 2),
        "net_return_pct": round(net_return_pct, 2),
        "realized_pnl": round(trader.realized_pnl, 2),
        "fees_paid": round(trader.total_fees_paid, 2),
        "slippage_cost": round(trader.total_slippage_cost, 2),
        "gross_profit": round(gross_profit, 2),
        "total_buys": len(buy_trades),
        "total_sells": len(sell_trades),
        "completed_trades": closed_trades,
        "winning_closed_trades": len(winning),
        "losing_closed_trades": len(losing),
        "win_rate_pct": round((len(winning) / closed_trades) * 100, 2)
        if closed_trades
        else 0.0,
        "average_win": round(
            sum(trade.realized_pnl for trade in winning) / len(winning), 2
        )
        if winning
        else 0.0,
        "average_loss": round(
            sum(trade.realized_pnl for trade in losing) / len(losing), 2
        )
        if losing
        else 0.0,
        "profit_factor": _format_profit_factor(pf),
        "maximum_drawdown_pct": round(compute_max_drawdown(equity_curve), 2),
        "exposure_time_pct": round(exposure_time_pct, 2),
        "longest_losing_sequence": _longest_losing_sequence(sell_trades),
        "average_holding_seconds": round(avg_holding_seconds, 2),
        "return_per_completed_trade": round(return_per_trade, 2),
        "results_by_coin": _results_by_coin_from_trades(trader.trade_log),
        "equity_curve": [round(value, 2) for value in equity_curve],
        "trade_log": _serialize_trades(trader.trade_log),
        "paused_during_period": trader.is_paused,
    }


def _format_profit_factor(value: float | None) -> float | str | None:
    if value is None:
        return None
    if value == float("inf"):
        return "inf"
    return round(value, 2)
