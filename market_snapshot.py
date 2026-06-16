"""
Read-only market snapshot layer.

Builds a structured dictionary from in-memory state (prices, history, paper trader).
Does not fetch prices, sign transactions, or place real orders.
"""

from dataclasses import asdict, dataclass
import json
import time
from datetime import datetime, timezone

from config import DATA, MARKET, PAPER_TRADING
from decision_engine import (
    CoinDecision,
    DecisionSessionSummary,
    build_decisions,
)
from market_analyzer import (
    WINDOW_1M_SECONDS,
    WINDOW_5M_SECONDS,
    PriceHistory,
    percent_change,
    price_to_float,
    record_price_history,
    rolling_percent_change,
)
from paper_trader import PaperTrader, Position
from signals import signal_from_1m_change


@dataclass
class MarketRow:
    """One coin's market analysis row."""

    coin: str
    current_price: float | None
    change_since_start: float | None
    change_1m: float | None
    change_5m: float | None
    signal: str


@dataclass
class PaperTraderSummary:
    """Paper trading account summary."""

    cash_balance: float
    realized_pnl: float
    unrealized_pnl: float
    total_equity: float
    open_exposure: float
    paused: bool
    pause_reason: str | None
    session_trades: int
    max_session_trades: int


@dataclass
class OpenPositionRow:
    """One open paper position."""

    coin: str
    quantity: float
    average_entry: float
    current_price: float
    unrealized_pnl: float


@dataclass
class MarketSnapshot:
    """Full read-only snapshot for one loop iteration."""

    timestamp: str
    test_mode: bool
    market_rows: list[MarketRow]
    paper_trader: PaperTraderSummary | None
    open_positions: list[OpenPositionRow]
    decisions: list[CoinDecision]
    decision_summary: DecisionSessionSummary | None

    def to_dict(self) -> dict:
        """Return the snapshot as a plain dictionary."""
        return asdict(self)


def current_timestamp() -> str:
    """Return the current UTC time as a readable string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def build_market_rows(
    mids: dict[str, str],
    start_prices: dict[str, float],
    price_history: PriceHistory,
    now: float | None = None,
) -> tuple[list[MarketRow], dict[str, float]]:
    """
    Build market rows and record price history for each coin.

    Returns the rows and a map of coin -> current price for coins with valid data.
    """
    if now is None:
        now = time.time()

    rows: list[MarketRow] = []
    current_prices: dict[str, float] = {}

    for coin in MARKET.watchlist:
        price_str = mids.get(coin)

        if price_str is None:
            rows.append(
                MarketRow(
                    coin=coin,
                    current_price=None,
                    change_since_start=None,
                    change_1m=None,
                    change_5m=None,
                    signal="N/A",
                )
            )
            continue

        try:
            current_price = price_to_float(price_str)
        except ValueError:
            rows.append(
                MarketRow(
                    coin=coin,
                    current_price=None,
                    change_since_start=None,
                    change_1m=None,
                    change_5m=None,
                    signal="N/A",
                )
            )
            continue

        current_prices[coin] = current_price
        record_price_history(price_history, coin, current_price, now)

        start_price = start_prices.get(coin)
        history = price_history.get(coin, [])

        change_since_start = (
            percent_change(start_price, current_price) if start_price is not None else None
        )
        change_1m = rolling_percent_change(history, current_price, now, WINDOW_1M_SECONDS)
        change_5m = rolling_percent_change(history, current_price, now, WINDOW_5M_SECONDS)
        signal = signal_from_1m_change(change_1m)

        rows.append(
            MarketRow(
                coin=coin,
                current_price=current_price,
                change_since_start=change_since_start,
                change_1m=change_1m,
                change_5m=change_5m,
                signal=signal,
            )
        )

    return rows, current_prices


def build_paper_trader_summary(
    paper_trader: PaperTrader,
    current_prices: dict[str, float],
) -> PaperTraderSummary:
    """Build the paper trader summary from current trader state."""
    pause_reason = None
    if paper_trader.is_paused:
        pause_reason = "equity protection"

    return PaperTraderSummary(
        cash_balance=paper_trader.cash_balance,
        realized_pnl=paper_trader.realized_pnl,
        unrealized_pnl=paper_trader.unrealized_pnl(current_prices),
        total_equity=paper_trader.total_equity(current_prices),
        open_exposure=paper_trader.total_open_exposure(),
        paused=paper_trader.is_paused,
        pause_reason=pause_reason,
        session_trades=paper_trader.successful_trades_this_session,
        max_session_trades=PAPER_TRADING.max_successful_trades_per_session,
    )


def build_open_positions(
    paper_trader: PaperTrader,
    current_prices: dict[str, float],
) -> list[OpenPositionRow]:
    """Build open position rows for each coin with an active position."""
    positions: list[OpenPositionRow] = []

    for coin in MARKET.watchlist:
        position = paper_trader.positions.get(coin, Position())
        current_price = current_prices.get(coin)

        if not position.is_open or current_price is None:
            continue

        coin_unrealized = (current_price - position.avg_entry_price) * position.quantity

        positions.append(
            OpenPositionRow(
                coin=coin,
                quantity=position.quantity,
                average_entry=position.avg_entry_price,
                current_price=current_price,
                unrealized_pnl=coin_unrealized,
            )
        )

    return positions


def build_snapshot(
    mids: dict[str, str],
    start_prices: dict[str, float],
    price_history: PriceHistory,
    test_mode: bool,
    paper_trader: PaperTrader | None = None,
    timestamp: str | None = None,
    now: float | None = None,
    market_rows: list[MarketRow] | None = None,
) -> MarketSnapshot:
    """
    Build a full market snapshot dictionary for one loop iteration.

    Market rows are always included. Paper trader sections are included when
    a PaperTrader instance is provided.

    Pass market_rows to reuse rows from an earlier build in the same loop
    (avoids recording duplicate price history entries).
    """
    if timestamp is None:
        timestamp = current_timestamp()

    if market_rows is None:
        market_rows, current_prices = build_market_rows(
            mids, start_prices, price_history, now=now
        )
    else:
        current_prices = current_prices_from_rows(market_rows)

    paper_summary = None
    open_positions: list[OpenPositionRow] = []
    decisions: list[CoinDecision] = []
    decision_summary: DecisionSessionSummary | None = None

    trader_for_decisions = paper_trader if paper_trader is not None else PaperTrader()

    if market_rows:
        decisions, decision_summary = build_decisions(
            market_rows,
            trader_for_decisions,
            current_prices,
            now if now is not None else time.time(),
        )

    if paper_trader is not None:
        paper_summary = build_paper_trader_summary(paper_trader, current_prices)
        open_positions = build_open_positions(paper_trader, current_prices)

    return MarketSnapshot(
        timestamp=timestamp,
        test_mode=test_mode,
        market_rows=market_rows,
        paper_trader=paper_summary,
        open_positions=open_positions,
        decisions=decisions,
        decision_summary=decision_summary,
    )


def format_price(price: float) -> str:
    """Format a price for display (e.g. $95,000.00)."""
    return f"${price:,.2f}"


def format_change(change_pct: float) -> str:
    """Format a percentage change with sign (e.g. +0.21% or -0.35%)."""
    sign = "+" if change_pct >= 0 else ""
    return f"{sign}{change_pct:.2f}%"


def format_change_or_na(change_pct: float | None) -> str:
    """Format a percentage change, or N/A when there is not enough history."""
    if change_pct is None:
        return "N/A"
    return format_change(change_pct)


def format_usdc(amount: float) -> str:
    """Format a USDC amount for display."""
    sign = "+" if amount >= 0 else ""
    return f"{sign}${amount:,.2f}"


def format_quantity(quantity: float) -> str:
    """Format a coin quantity for display."""
    if quantity >= 1:
        return f"{quantity:.6f}"
    return f"{quantity:.8f}"


def print_snapshot(snapshot: MarketSnapshot) -> None:
    """Print market analysis and paper trading summary from a snapshot."""
    print(f"\n[{snapshot.timestamp}]")
    print("Hyperliquid market analysis:")
    print("-" * 78)
    print(
        f"{'Coin':<6} {'Current':>14} {'Since Start':>12} "
        f"{'1-Min':>10} {'5-Min':>10} {'Signal':>12}"
    )
    print("-" * 78)

    for row in snapshot.market_rows:
        if row.current_price is None:
            print(f"{row.coin:<6} {'not available':>14}")
            continue

        if row.change_since_start is None:
            print(
                f"{row.coin:<6} {format_price(row.current_price):>14} "
                f"{'(recording)':>12} {'N/A':>10} {'N/A':>10} {'N/A':>12}"
            )
            continue

        print(
            f"{row.coin:<6} {format_price(row.current_price):>14} "
            f"{format_change_or_na(row.change_since_start):>12} "
            f"{format_change_or_na(row.change_1m):>10} "
            f"{format_change_or_na(row.change_5m):>10} {row.signal:>12}"
        )

    if snapshot.paper_trader is not None:
        _print_paper_trader_section(snapshot.paper_trader, snapshot.open_positions)

    if snapshot.decisions:
        _print_decisions_section(snapshot.decisions, snapshot.decision_summary)


def _print_paper_trader_section(
    summary: PaperTraderSummary,
    open_positions: list[OpenPositionRow],
) -> None:
    """Print paper trading summary and open positions from snapshot data."""
    if summary.paused and summary.pause_reason:
        paused_label = f"YES ({summary.pause_reason})"
    elif summary.paused:
        paused_label = "YES"
    else:
        paused_label = "NO"

    print()
    print("Paper trading summary:")
    print("-" * 78)
    print(f"  Status:          Paused: {paused_label}")
    print(f"  Cash balance:    ${summary.cash_balance:,.2f}")
    print(f"  Realized PnL:    {format_usdc(summary.realized_pnl)}")
    print(f"  Unrealized PnL:  {format_usdc(summary.unrealized_pnl)}")
    print(f"  Total equity:    ${summary.total_equity:,.2f}")
    print(f"  Open exposure:   ${summary.open_exposure:,.2f}")
    print(
        f"  Session trades:  {summary.session_trades}"
        f" / {summary.max_session_trades}"
    )
    print()
    print(
        f"  {'Coin':<6} {'Quantity':>14} {'Avg Entry':>14} "
        f"{'Current':>14} {'Unrealized':>12}"
    )
    print("  " + "-" * 74)

    if not open_positions:
        print("  (no open positions)")
    else:
        for position in open_positions:
            print(
                f"  {position.coin:<6} {format_quantity(position.quantity):>14} "
                f"${position.average_entry:>12,.2f} "
                f"${position.current_price:>12,.2f} "
                f"{format_usdc(position.unrealized_pnl):>12}"
            )

    print("-" * 78)


def _print_decisions_section(
    decisions: list[CoinDecision],
    summary: DecisionSessionSummary | None,
) -> None:
    """Print read-only recommendations and a short session summary."""
    print()
    print("Current recommendations (read-only, no orders placed):")
    print("-" * 78)
    print(
        f"{'Coin':<6} {'Recommendation':<12} {'Signal':>12} {'Strength':>10} "
        f"{'Risk':<28} {'Confidence':>10}"
    )
    print("-" * 78)

    for decision in decisions:
        risk_short = decision.risk_status
        if len(risk_short) > 28:
            risk_short = risk_short[:25] + "..."

        print(
            f"{decision.coin:<6} {decision.recommendation:<12} "
            f"{decision.signal:>12} {decision.signal_strength:>10} "
            f"{risk_short:<28} {decision.confidence:>10}"
        )
        print(f"         Reason: {decision.primary_reason}")

    if summary is not None:
        print()
        print("Recommendation session summary:")
        print(
            f"  Active: {summary.active_recommendations}  "
            f"Blocked: {summary.blocked_recommendations}  "
            f"Open positions: {summary.open_positions}  "
            f"Session trades: {summary.session_trades}"
        )
        if summary.total_equity is not None:
            print(f"  Total equity: ${summary.total_equity:,.2f}")
        if summary.current_drawdown_pct is not None:
            print(f"  Current drawdown: {summary.current_drawdown_pct:.2f}%")

    print("-" * 78)


def save_recommendations_json(snapshot: MarketSnapshot) -> None:
    """Write the latest recommendations snapshot for the dashboard."""
    DATA.data_dir.mkdir(exist_ok=True)

    payload = snapshot.to_dict()

    with DATA.recommendations_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def current_prices_from_rows(market_rows: list[MarketRow]) -> dict[str, float]:
    """Extract a coin -> price map from market rows."""
    return {
        row.coin: row.current_price
        for row in market_rows
        if row.current_price is not None
    }


def tradable_signals_from_rows(
    market_rows: list[MarketRow],
) -> list[tuple[str, str, float]]:
    """Return (coin, signal, price) tuples for rows ready to process."""
    signals: list[tuple[str, str, float]] = []

    for row in market_rows:
        if row.current_price is None or row.change_since_start is None:
            continue

        signals.append((row.coin, row.signal, row.current_price))

    return signals
