"""Append-only equity snapshots for forward-testing analytics."""

from __future__ import annotations

import csv

from config import DATA
from paper_trader import PaperTrader, ensure_data_dir

EQUITY_COLUMNS = (
    "timestamp",
    "session_id",
    "mode",
    "cash",
    "open_position_value",
    "realized_pnl",
    "unrealized_pnl",
    "fees_paid",
    "total_equity",
    "open_exposure",
    "session_trades",
    "paused",
)


def append_equity_snapshot(
    *,
    timestamp: str,
    session_id: str,
    mode: str,
    paper_trader: PaperTrader,
    current_prices: dict[str, float],
) -> None:
    """Append one equity row for the current analyzer cycle."""
    ensure_data_dir()

    open_position_value = 0.0
    for coin, position in paper_trader.positions.items():
        if not position.is_open:
            continue
        current_price = current_prices.get(coin)
        if current_price is None:
            continue
        open_position_value += position.quantity * current_price

    unrealized = paper_trader.unrealized_pnl(current_prices)
    total_equity = paper_trader.total_equity(current_prices)

    write_header = not DATA.equity_log_csv.exists()

    with DATA.equity_log_csv.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        if write_header:
            writer.writerow(EQUITY_COLUMNS)

        writer.writerow(
            [
                timestamp,
                session_id,
                mode,
                f"{paper_trader.cash_balance:.2f}",
                f"{open_position_value:.2f}",
                f"{paper_trader.realized_pnl:.2f}",
                f"{unrealized:.2f}",
                f"{paper_trader.total_fees_paid:.2f}",
                f"{total_equity:.2f}",
                f"{paper_trader.total_open_exposure():.2f}",
                str(paper_trader.successful_trades_this_session),
                "1" if paper_trader.is_paused else "0",
            ]
        )
