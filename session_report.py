"""Generate forward-testing session reports from logged paper trades and equity."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtester import compute_max_drawdown
from config import DATA, PAPER_TRADING
from dashboard_data import (
    compute_trade_stats,
    load_equity_log,
    load_paper_trades,
    normalize_paper_trades,
    pnl_by_coin,
)
from risk_rules import session_trade_limit_reached

# Slippage is embedded in fill prices (execution_price vs mid). Realized and
# unrealized PnL therefore already reflect slippage. Fees are deducted from cash
# separately and are NOT included in realized_pnl.
#
# gross_realized_pnl adds slippage back out for reporting (mid-equivalent closed PnL).
# ending_equity identity (validated): starting_equity + realized_pnl + unrealized_pnl - fees_paid
ACCOUNTING_FORMULA = (
    "ending_equity = starting_equity + realized_pnl + unrealized_pnl - fees_paid "
    "(slippage embedded in realized/unrealized via execution prices; "
    "gross_realized_pnl = realized_pnl + slippage_cost for mid-equivalent view)"
)


@dataclass
class OpenPositionSnapshot:
    """One open paper position preserved at shutdown."""

    coin: str
    quantity: float
    average_entry: float
    final_market_price: float
    position_value: float
    unrealized_pnl: float
    entry_count: int


@dataclass
class SessionShutdownContext:
    """Runtime context captured when a live analyzer session ends."""

    shutdown_reason: str = "unknown"
    started_at: float | None = None
    ended_at: float | None = None
    current_prices: dict[str, float] = field(default_factory=dict)
    strategy_profile: str = ""
    configuration_hash: str = ""
    session_trade_limit: int = PAPER_TRADING.max_successful_trades_per_session
    blocked_proposals_total: int = 0
    blocked_proposals_by_reason: dict[str, int] = field(default_factory=dict)
    positions_force_closed: bool = False
    force_close_reason: str | None = None


def _profit_factor(trades: pd.DataFrame) -> float | None:
    if trades.empty or "trade_pnl" not in trades.columns:
        return None

    sell_trades = trades[trades["action"] == "SELL"]
    if sell_trades.empty:
        return None

    gross_wins = sell_trades.loc[sell_trades["trade_pnl"] > 0, "trade_pnl"].sum()
    gross_losses = abs(sell_trades.loc[sell_trades["trade_pnl"] < 0, "trade_pnl"].sum())

    if gross_losses == 0:
        return None if gross_wins == 0 else float("inf")
    return gross_wins / gross_losses


def _iso_timestamp(unix_time: float | None) -> str | None:
    if unix_time is None:
        return None
    return datetime.fromtimestamp(unix_time, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def build_open_position_snapshots(
    paper_trader,
    current_prices: dict[str, float],
) -> list[dict]:
    """Build open-position records for session reports."""
    from config import MARKET
    from paper_trader import Position

    snapshots: list[dict] = []

    for coin in MARKET.watchlist:
        position = paper_trader.positions.get(coin, Position())
        current_price = current_prices.get(coin)

        if not position.is_open or current_price is None:
            continue

        position_value = position.quantity * current_price
        unrealized = (current_price - position.avg_entry_price) * position.quantity

        snapshots.append(
            OpenPositionSnapshot(
                coin=coin,
                quantity=round(position.quantity, 8),
                average_entry=round(position.avg_entry_price, 2),
                final_market_price=round(current_price, 2),
                position_value=round(position_value, 2),
                unrealized_pnl=round(unrealized, 2),
                entry_count=position.entry_count,
            ).__dict__
        )

    return snapshots


def validate_accounting_identity(
    *,
    starting_equity: float,
    ending_equity: float,
    realized_pnl: float,
    unrealized_pnl: float,
    fees_paid: float,
    tolerance: float = 0.02,
) -> bool:
    """
    Validate ending equity against PnL components.

    Slippage is not subtracted here because it is already embedded in
    realized_pnl and unrealized_pnl through execution fill prices.
    """
    expected = starting_equity + realized_pnl + unrealized_pnl - fees_paid
    return abs(expected - ending_equity) <= tolerance


def build_session_report(
    session_id: str,
    *,
    shutdown: SessionShutdownContext | None = None,
    paper_trader=None,
) -> dict:
    """Calculate analytics for one paper-trading session."""
    trades_df = load_paper_trades()
    equity_df = load_equity_log()

    if trades_df is None and paper_trader is None:
        raise ValueError("No paper trades file found.")

    trades = (
        normalize_paper_trades(trades_df)
        if trades_df is not None and not trades_df.empty
        else pd.DataFrame()
    )
    if trades.empty or "session_id" not in trades.columns:
        session_trades = trades.copy()
    else:
        session_trades = trades[trades["session_id"] == session_id].copy()

    session_equity = pd.DataFrame()
    if equity_df is not None and not equity_df.empty:
        session_equity = equity_df[equity_df["session_id"] == session_id].copy()

    mode = "unknown"
    if not session_trades.empty and "mode" in session_trades.columns:
        mode = str(session_trades["mode"].iloc[0])
    elif not session_equity.empty and "mode" in session_equity.columns:
        mode = str(session_equity["mode"].iloc[0])
    elif paper_trader is not None and paper_trader.mode:
        mode = paper_trader.mode

    starting_equity = PAPER_TRADING.starting_balance_usdc
    ending_equity = starting_equity
    unrealized_pnl = 0.0
    fees_paid = 0.0
    final_open_exposure = 0.0

    if not session_equity.empty:
        session_equity = session_equity.sort_values("timestamp")
        ending_equity = float(session_equity["total_equity"].iloc[-1])
        unrealized_pnl = float(session_equity["unrealized_pnl"].iloc[-1])
        fees_paid = float(session_equity["fees_paid"].iloc[-1])
        final_open_exposure = float(session_equity["open_exposure"].iloc[-1])
    elif paper_trader is not None and shutdown and shutdown.current_prices:
        starting_equity = PAPER_TRADING.starting_balance_usdc
        unrealized_pnl = paper_trader.unrealized_pnl(shutdown.current_prices)
        fees_paid = paper_trader.total_fees_paid
        ending_equity = paper_trader.total_equity(shutdown.current_prices)
        final_open_exposure = paper_trader.total_open_exposure()

    if not session_trades.empty and fees_paid == 0.0:
        fees_paid = float(session_trades["fee"].sum())

    slippage_cost = (
        float(session_trades["slippage"].sum())
        if not session_trades.empty and "slippage" in session_trades.columns
        else 0.0
    )
    if paper_trader is not None and slippage_cost == 0.0:
        slippage_cost = paper_trader.total_slippage_cost

    realized_pnl = float(session_trades["trade_pnl"].sum()) if not session_trades.empty else 0.0
    if not session_trades.empty and "cumulative_realized_pnl" in session_trades.columns:
        realized_pnl = float(session_trades["cumulative_realized_pnl"].iloc[-1])
    elif paper_trader is not None:
        realized_pnl = paper_trader.realized_pnl

    gross_realized_pnl = realized_pnl + slippage_cost
    net_pnl_after_costs = realized_pnl + unrealized_pnl - fees_paid

    total_return_pct = 0.0
    if starting_equity > 0:
        total_return_pct = ((ending_equity - starting_equity) / starting_equity) * 100

    equity_curve = (
        session_equity.sort_values("timestamp")["total_equity"].tolist()
        if not session_equity.empty
        else [starting_equity, ending_equity]
    )

    stats = compute_trade_stats(session_trades)
    pf = _profit_factor(session_trades)

    session_trade_limit = (
        shutdown.session_trade_limit
        if shutdown is not None
        else PAPER_TRADING.max_successful_trades_per_session
    )
    successful_trades = 0
    if paper_trader is not None:
        successful_trades = paper_trader.successful_trades_this_session
    elif not session_trades.empty:
        successful_trades = len(session_trades)
    elif not session_equity.empty:
        successful_trades = int(session_equity["session_trades"].iloc[-1])

    limit_reached = successful_trades >= session_trade_limit
    if paper_trader is not None:
        limit_reached = session_trade_limit_reached(paper_trader)

    blocked_total = shutdown.blocked_proposals_total if shutdown else 0
    blocked_by_reason = shutdown.blocked_proposals_by_reason if shutdown else {}

    open_positions_at_shutdown: list[dict] = []
    if shutdown and paper_trader is not None:
        open_positions_at_shutdown = build_open_position_snapshots(
            paper_trader,
            shutdown.current_prices,
        )
        if open_positions_at_shutdown:
            final_open_exposure = paper_trader.total_open_exposure()

    started_at = _iso_timestamp(shutdown.started_at if shutdown else None)
    ended_at = _iso_timestamp(shutdown.ended_at if shutdown else None)
    runtime_seconds = None
    if shutdown and shutdown.started_at is not None and shutdown.ended_at is not None:
        runtime_seconds = round(shutdown.ended_at - shutdown.started_at, 1)

    accounting_valid = validate_accounting_identity(
        starting_equity=starting_equity,
        ending_equity=ending_equity,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        fees_paid=fees_paid,
    )

    report = {
        "session_id": session_id,
        "mode": mode,
        "started_at": started_at,
        "ended_at": ended_at,
        "runtime_seconds": runtime_seconds,
        "shutdown_reason": shutdown.shutdown_reason if shutdown else None,
        "strategy_profile": shutdown.strategy_profile if shutdown else "",
        "configuration_hash": shutdown.configuration_hash if shutdown else "",
        "accounting_formula": ACCOUNTING_FORMULA,
        "accounting_valid": accounting_valid,
        "starting_equity": round(starting_equity, 2),
        "ending_equity": round(ending_equity, 2),
        "total_return_pct": round(total_return_pct, 2),
        "gross_realized_pnl": round(gross_realized_pnl, 2),
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "fees_paid": round(fees_paid, 2),
        "slippage_cost": round(slippage_cost, 2),
        "net_pnl_after_costs": round(net_pnl_after_costs, 2),
        "final_open_exposure": round(final_open_exposure, 2),
        "session_trade_limit": session_trade_limit,
        "session_trade_limit_reached": limit_reached,
        "blocked_proposals_total": blocked_total,
        "blocked_proposals_by_reason": blocked_by_reason,
        "open_positions_at_shutdown": open_positions_at_shutdown,
        "positions_force_closed": (
            shutdown.positions_force_closed if shutdown else False
        ),
        "force_close_reason": shutdown.force_close_reason if shutdown else None,
        "total_buys": stats["total_buys"],
        "total_sells": stats["total_sells"],
        "completed_trades": stats["completed_trades"],
        "winning_trades": stats["winning_trades"],
        "losing_trades": stats["losing_trades"],
        "win_rate_pct": stats["win_rate_pct"],
        "average_win": stats["average_win"],
        "average_loss": stats["average_loss"],
        "profit_factor": round(pf, 2) if pf is not None and pf != float("inf") else pf,
        "maximum_drawdown_pct": round(compute_max_drawdown(equity_curve), 2),
        "results_by_coin": pnl_by_coin(session_trades),
    }

    return report


def save_session_report(
    session_id: str,
    report: dict | None = None,
    output_dir: Path = DATA.session_reports_dir,
    *,
    shutdown: SessionShutdownContext | None = None,
    paper_trader=None,
) -> Path:
    """Write a session report JSON file and return its path."""
    payload = report
    if payload is None:
        payload = build_session_report(
            session_id,
            shutdown=shutdown,
            paper_trader=paper_trader,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{session_id}.json"

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)

    return path


def _json_default(value: object) -> object:
    if value == float("inf"):
        return "inf"
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def list_session_ids() -> list[str]:
    """Return session IDs found in paper trades or equity logs."""
    ids: set[str] = set()

    trades = load_paper_trades()
    if trades is not None and "session_id" in trades.columns:
        ids.update(trades["session_id"].dropna().astype(str).tolist())

    equity = load_equity_log()
    if equity is not None and "session_id" in equity.columns:
        ids.update(equity["session_id"].dropna().astype(str).tolist())

    return sorted(ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a forward-testing session report (read-only)."
    )
    parser.add_argument("session_id", help="Session ID to summarize")
    args = parser.parse_args(argv)

    try:
        path = save_session_report(args.session_id)
    except ValueError as exc:
        print(f"Session report failed: {exc}")
        return 1

    print(f"Session report saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
