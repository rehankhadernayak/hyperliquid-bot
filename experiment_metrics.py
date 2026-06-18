"""Performance metrics for strategy experiment runs."""

from __future__ import annotations

from dataclasses import asdict
from statistics import median
from typing import Any

from backtester import compute_max_drawdown, summarize_by_coin
from config import MARKET
from exit_config import (
    EXIT_TYPE_MAX_HOLD,
    EXIT_TYPE_SIGNAL_EXIT,
    EXIT_TYPE_STOP_LOSS,
    EXIT_TYPE_TAKE_PROFIT,
)
from experiment_trader import ExperimentPaperTrader
from strategy_profiles import StrategyProfile
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


def _exit_type_counts(sell_trades: list[ProfileTradeRecord]) -> dict[str, int]:
    counts = {
        "take_profit_exits": 0,
        "stop_loss_exits": 0,
        "max_hold_exits": 0,
        "signal_exits": 0,
    }
    for trade in sell_trades:
        exit_type = trade.exit_type
        if exit_type == EXIT_TYPE_TAKE_PROFIT:
            counts["take_profit_exits"] += 1
        elif exit_type == EXIT_TYPE_STOP_LOSS:
            counts["stop_loss_exits"] += 1
        elif exit_type == EXIT_TYPE_MAX_HOLD:
            counts["max_hold_exits"] += 1
        elif exit_type == EXIT_TYPE_SIGNAL_EXIT or exit_type is None:
            counts["signal_exits"] += 1
    return counts


def _exit_type_percentages(counts: dict[str, int], closed_trades: int) -> dict[str, float]:
    if closed_trades <= 0:
        return {key: 0.0 for key in counts}
    return {key: round((value / closed_trades) * 100, 2) for key, value in counts.items()}


def _average_gross_and_net_returns(
    sell_trades: list[ProfileTradeRecord],
) -> tuple[float, float]:
    closed = [t for t in sell_trades if t.realized_pnl != 0]
    if not closed:
        return 0.0, 0.0

    gross_returns: list[float] = []
    net_returns: list[float] = []
    for trade in closed:
        entry_notional = trade.execution_price * trade.quantity
        if trade.gross_return_pct is not None:
            gross_returns.append(trade.gross_return_pct)
        elif trade.mid_price > 0 and trade.quantity > 0:
            gross_returns.append(
                (trade.realized_pnl / (trade.mid_price * trade.quantity)) * 100
            )
        if trade.estimated_net_return_pct is not None:
            net_returns.append(trade.estimated_net_return_pct)
        elif entry_notional > 0:
            net_returns.append(((trade.realized_pnl - trade.fee) / entry_notional) * 100)

    avg_gross = sum(gross_returns) / len(gross_returns) if gross_returns else 0.0
    avg_net = sum(net_returns) / len(net_returns) if net_returns else 0.0
    return avg_gross, avg_net


def _trades_never_reaching_targets(
    sell_trades: list[ProfileTradeRecord],
    profile: StrategyProfile | None,
) -> int:
    if profile is None or profile.exit_config is None:
        return 0
    exit_config = profile.exit_config
    if exit_config.take_profit_pct is None and exit_config.stop_loss_pct is None:
        return 0

    count = 0
    for trade in sell_trades:
        if trade.exit_type in (EXIT_TYPE_TAKE_PROFIT, EXIT_TYPE_STOP_LOSS):
            continue
        if trade.exit_type in (EXIT_TYPE_SIGNAL_EXIT, EXIT_TYPE_MAX_HOLD, None):
            count += 1
    return count


def compute_period_metrics(
    trader: ExperimentPaperTrader,
    equity_curve: list[float],
    current_prices: dict[str, float],
    *,
    period_label: str,
    limits: TradingLimits,
    profile: StrategyProfile | None = None,
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
    holding_times = [t for t in trader.run_state.completed_holding_seconds if t >= 0]
    if holding_times:
        avg_holding_seconds = sum(holding_times) / len(holding_times)

    return_per_trade = 0.0
    if closed_trades:
        return_per_trade = trader.realized_pnl / closed_trades

    total_costs = trader.total_fees_paid + trader.total_slippage_cost
    costs_pct_of_gross_profit = _costs_pct_of_gross_profit(gross_profit, total_costs)

    exit_counts = _exit_type_counts(sell_trades)
    exit_pcts = _exit_type_percentages(exit_counts, closed_trades)
    avg_gross_return, avg_net_return = _average_gross_and_net_returns(sell_trades)

    median_holding = median(holding_times) if holding_times else 0.0
    longest_holding = max(holding_times) if holding_times else 0.0

    return {
        "period_label": period_label,
        "starting_equity": round(starting_equity, 2),
        "ending_equity": round(ending_equity, 2),
        "net_return_pct": round(net_return_pct, 2),
        "realized_pnl": round(trader.realized_pnl, 2),
        "fees_paid": round(trader.total_fees_paid, 2),
        "slippage_cost": round(trader.total_slippage_cost, 2),
        "gross_profit": round(gross_profit, 2),
        "costs_pct_of_gross_profit": costs_pct_of_gross_profit,
        "total_costs": round(total_costs, 2),
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
        "average_gross_return_per_trade": round(avg_gross_return, 4),
        "average_net_return_per_trade": round(avg_net_return, 4),
        "profit_factor": _format_profit_factor(pf),
        "maximum_drawdown_pct": round(compute_max_drawdown(equity_curve), 2),
        "exposure_time_pct": round(exposure_time_pct, 2),
        "longest_losing_sequence": _longest_losing_sequence(sell_trades),
        "average_holding_seconds": round(avg_holding_seconds, 2),
        "median_holding_seconds": round(median_holding, 2),
        "longest_holding_seconds": round(longest_holding, 2),
        "return_per_completed_trade": round(return_per_trade, 2),
        "take_profit_exits": exit_counts["take_profit_exits"],
        "stop_loss_exits": exit_counts["stop_loss_exits"],
        "max_hold_exits": exit_counts["max_hold_exits"],
        "signal_exits": exit_counts["signal_exits"],
        "take_profit_exit_pct": exit_pcts["take_profit_exits"],
        "stop_loss_exit_pct": exit_pcts["stop_loss_exits"],
        "max_hold_exit_pct": exit_pcts["max_hold_exits"],
        "signal_exit_pct": exit_pcts["signal_exits"],
        "trades_never_reaching_targets": _trades_never_reaching_targets(
            sell_trades, profile
        ),
        "exit_ambiguity_warnings": list(dict.fromkeys(trader.run_state.exit_ambiguity_warnings)),
        "results_by_coin": _results_by_coin_from_trades(trader.trade_log),
        "equity_curve": [round(value, 2) for value in equity_curve],
        "trade_log": _serialize_trades(trader.trade_log),
        "paused_during_period": trader.is_paused,
    }


def _costs_pct_of_gross_profit(gross_profit: float, total_costs: float) -> float | str | None:
    if gross_profit <= 0:
        return "N/A"
    return round((total_costs / gross_profit) * 100, 2)


def _format_profit_factor(value: float | None) -> float | str | None:
    if value is None:
        return None
    if value == float("inf"):
        return "inf"
    return round(value, 2)
