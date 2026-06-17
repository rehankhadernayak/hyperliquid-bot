"""Generate forward-testing session reports from logged paper trades and equity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from backtester import compute_max_drawdown
from config import DATA, PAPER_TRADING
from dashboard_data import (
    compute_cumulative_realized_pnl,
    compute_trade_stats,
    load_equity_log,
    load_paper_trades,
    normalize_paper_trades,
    pnl_by_coin,
)


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


def build_session_report(session_id: str) -> dict:
    """Calculate analytics for one paper-trading session."""
    trades_df = load_paper_trades()
    equity_df = load_equity_log()

    if trades_df is None:
        raise ValueError("No paper trades file found.")

    trades = normalize_paper_trades(trades_df)
    session_trades = trades[trades["session_id"] == session_id].copy()

    session_equity = pd.DataFrame()
    if equity_df is not None and not equity_df.empty:
        session_equity = equity_df[equity_df["session_id"] == session_id].copy()

    mode = "unknown"
    if not session_trades.empty and "mode" in session_trades.columns:
        mode = str(session_trades["mode"].iloc[0])
    elif not session_equity.empty and "mode" in session_equity.columns:
        mode = str(session_equity["mode"].iloc[0])

    starting_equity = PAPER_TRADING.starting_balance_usdc
    ending_equity = starting_equity
    unrealized_pnl = 0.0
    fees_paid = 0.0

    if not session_equity.empty:
        session_equity = session_equity.sort_values("timestamp")
        starting_equity = float(session_equity["total_equity"].iloc[0])
        ending_equity = float(session_equity["total_equity"].iloc[-1])
        unrealized_pnl = float(session_equity["unrealized_pnl"].iloc[-1])
        fees_paid = float(session_equity["fees_paid"].iloc[-1])
    elif not session_trades.empty:
        fees_paid = float(session_trades["fee"].sum())

    realized_pnl = float(session_trades["trade_pnl"].sum()) if not session_trades.empty else 0.0
    if not session_trades.empty and "cumulative_realized_pnl" in session_trades.columns:
        realized_pnl = float(session_trades["cumulative_realized_pnl"].iloc[-1])

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

    return {
        "session_id": session_id,
        "mode": mode,
        "starting_equity": round(starting_equity, 2),
        "ending_equity": round(ending_equity, 2),
        "total_return_pct": round(total_return_pct, 2),
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "fees_paid": round(fees_paid, 2),
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


def save_session_report(
    session_id: str,
    report: dict | None = None,
    output_dir: Path = DATA.session_reports_dir,
) -> Path:
    """Write a session report JSON file and return its path."""
    payload = report if report is not None else build_session_report(session_id)
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
