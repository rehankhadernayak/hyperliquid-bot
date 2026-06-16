"""
Read-only backtesting using historical prices from data/price_log.csv.

Replays saved prices in timestamp order and runs the same signal logic and
paper trading rules as the live analyzer — without calling Hyperliquid,
using wallets, or placing real orders.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from market_analyzer import PriceHistory, WATCHLIST, update_start_prices
from market_snapshot import build_market_rows, tradable_signals_from_rows
from paper_trader import (
    MIN_EQUITY_BEFORE_PAUSE_USDC,
    PaperTrader,
    STARTING_BALANCE_USDC,
)
import paper_trader as paper_trader_module
from signals import apply_signal_thresholds

# Match main.py: lower thresholds trigger more paper trades during testing.
TEST_MODE = True

PRICE_LOG_CSV = Path("data") / "price_log.csv"
RESULTS_JSON = Path("data") / "backtest_results.json"


@dataclass
class BacktestPaperTrader(PaperTrader):
    """
    Separate paper trader for backtests.

    Does not write to data/paper_trades.csv or print per-trade messages
    during replay, so live paper trading state stays untouched.
    """

    trade_log: list[dict] = field(default_factory=list)

    def update_session_risk(self, current_prices: dict[str, float]) -> None:
        """Pause on low equity without printing (summary is shown at the end)."""
        if self.is_paused:
            return

        equity = self.total_equity(current_prices)
        if equity < MIN_EQUITY_BEFORE_PAUSE_USDC:
            self.is_paused = True

    def _log_trade(
        self,
        timestamp: str,
        coin: str,
        action: str,
        price: float,
        quantity: float,
        trade_value: float,
        trade_realized_pnl: float,
        signal: str,
    ) -> None:
        self.trade_log.append(
            {
                "timestamp": timestamp,
                "coin": coin,
                "action": action,
                "price": price,
                "quantity": quantity,
                "trade_value": trade_value,
                "realized_pnl": trade_realized_pnl,
                "signal": signal,
            }
        )
        self.successful_trades_this_session += 1


def parse_timestamp(timestamp_str: str) -> float:
    """Convert a CSV timestamp string into a Unix time (seconds)."""
    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S UTC").replace(
        tzinfo=timezone.utc
    )
    return dt.timestamp()


def load_price_ticks(
    csv_path: Path,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """
    Read price_log.csv and group prices by timestamp.

    Returns timestamps in file order and a map:
        timestamp -> {coin: price}
    """
    prices_by_timestamp: dict[str, dict[str, float]] = defaultdict(dict)
    timestamps_in_order: list[str] = []
    seen_timestamps: set[str] = set()

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            timestamp = row["timestamp"]
            coin = row["coin"]
            price = float(row["price"])

            prices_by_timestamp[timestamp][coin] = price

            if timestamp not in seen_timestamps:
                seen_timestamps.add(timestamp)
                timestamps_in_order.append(timestamp)

    return timestamps_in_order, prices_by_timestamp


def prices_to_mids(prices: dict[str, float]) -> dict[str, str]:
    """Convert float prices into the string mids format used by market_analyzer."""
    return {coin: str(price) for coin, price in prices.items()}


def compute_max_drawdown(equity_curve: list[float]) -> float:
    """
    Return maximum drawdown as a positive percentage.

    Example: peak $1,050, trough $1,000 -> drawdown is about 4.76%.
    """
    if not equity_curve:
        return 0.0

    peak = equity_curve[0]
    max_drawdown_pct = 0.0

    for equity in equity_curve:
        if equity > peak:
            peak = equity

        if peak <= 0:
            continue

        drawdown_pct = ((peak - equity) / peak) * 100
        if drawdown_pct > max_drawdown_pct:
            max_drawdown_pct = drawdown_pct

    return max_drawdown_pct


def summarize_by_coin(trade_log: list[dict]) -> dict[str, dict]:
    """Build per-coin trade stats from the backtest trade log."""
    by_coin: dict[str, dict] = {}

    for coin in WATCHLIST:
        coin_trades = [trade for trade in trade_log if trade["coin"] == coin]
        coin_sells = [trade for trade in coin_trades if trade["action"] == "SELL"]

        winning = sum(1 for trade in coin_sells if trade["realized_pnl"] > 0)
        losing = sum(1 for trade in coin_sells if trade["realized_pnl"] < 0)
        closed = winning + losing

        by_coin[coin] = {
            "total_trades": len(coin_trades),
            "buy_trades": sum(1 for trade in coin_trades if trade["action"] == "BUY"),
            "sell_trades": len(coin_sells),
            "realized_pnl": round(sum(trade["realized_pnl"] for trade in coin_sells), 2),
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate_pct": round((winning / closed) * 100, 2) if closed else 0.0,
        }

    return by_coin


def _silent_skip(**_kwargs) -> None:
    """No-op replacement for skip messages during replay."""


def run_backtest(
    csv_path: Path = PRICE_LOG_CSV,
    test_mode: bool = TEST_MODE,
) -> dict:
    """
    Replay historical prices and return backtest metrics.

    Uses a fresh BacktestPaperTrader instance so live paper trading files
    and state are never modified.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Price log not found: {csv_path}")

    apply_signal_thresholds(test_mode)

    timestamps, prices_by_timestamp = load_price_ticks(csv_path)
    if not timestamps:
        raise ValueError(f"No price rows found in {csv_path}")

    original_skip_printer = paper_trader_module.print_skipped_trade_proposal
    paper_trader_module.print_skipped_trade_proposal = _silent_skip

    paper_trader = BacktestPaperTrader()
    start_prices: dict[str, float] = {}
    price_history: PriceHistory = {}
    equity_curve: list[float] = [STARTING_BALANCE_USDC]

    try:
        for timestamp in timestamps:
            coin_prices = prices_by_timestamp[timestamp]
            mids = prices_to_mids(coin_prices)
            now = parse_timestamp(timestamp)

            update_start_prices(start_prices, mids)

            market_rows, current_prices = build_market_rows(
                mids, start_prices, price_history, now=now
            )

            paper_trader.update_session_risk(current_prices)

            for coin, signal, current_price in tradable_signals_from_rows(market_rows):
                paper_trader.process_signal(
                    coin, signal, current_price, timestamp, now
                )

            equity_curve.append(paper_trader.total_equity(current_prices))
    finally:
        paper_trader_module.print_skipped_trade_proposal = original_skip_printer

    ending_equity = equity_curve[-1]
    total_return_pct = (
        ((ending_equity - STARTING_BALANCE_USDC) / STARTING_BALANCE_USDC) * 100
    )

    sell_trades = [
        trade for trade in paper_trader.trade_log if trade["action"] == "SELL"
    ]
    winning_trades = sum(1 for trade in sell_trades if trade["realized_pnl"] > 0)
    losing_trades = sum(1 for trade in sell_trades if trade["realized_pnl"] < 0)
    closed_trades = winning_trades + losing_trades

    results = {
        "source_file": str(csv_path),
        "test_mode": test_mode,
        "timestamps_replayed": len(timestamps),
        "starting_equity": STARTING_BALANCE_USDC,
        "ending_equity": round(ending_equity, 2),
        "total_return_pct": round(total_return_pct, 2),
        "realized_pnl": round(paper_trader.realized_pnl, 2),
        "unrealized_pnl_at_end": round(
            paper_trader.unrealized_pnl(current_prices), 2
        ),
        "total_trades": len(paper_trader.trade_log),
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate_pct": round((winning_trades / closed_trades) * 100, 2)
        if closed_trades
        else 0.0,
        "max_drawdown_pct": round(compute_max_drawdown(equity_curve), 2),
        "paused_during_backtest": paper_trader.is_paused,
        "results_by_coin": summarize_by_coin(paper_trader.trade_log),
    }

    return results


def save_results(results: dict, output_path: Path = RESULTS_JSON) -> None:
    """Write backtest results to a JSON file."""
    output_path.parent.mkdir(exist_ok=True)

    with output_path.open("w") as json_file:
        json.dump(results, json_file, indent=2)


def print_results(results: dict) -> None:
    """Print a readable backtest summary to the terminal."""
    print()
    print("=" * 78)
    print("BACKTEST RESULTS")
    print("=" * 78)
    print(f"  Source:              {results['source_file']}")
    print(f"  Test mode:           {'ON' if results['test_mode'] else 'OFF'}")
    print(f"  Timestamps replayed: {results['timestamps_replayed']:,}")
    print()
    print("  Performance")
    print("  " + "-" * 74)
    print(f"  Starting equity:     ${results['starting_equity']:,.2f}")
    print(f"  Ending equity:       ${results['ending_equity']:,.2f}")
    print(f"  Total return:        {results['total_return_pct']:+.2f}%")
    print(f"  Realized PnL:        ${results['realized_pnl']:+,.2f}")
    print(f"  Unrealized PnL:      ${results['unrealized_pnl_at_end']:+,.2f}")
    print(f"  Max drawdown:        {results['max_drawdown_pct']:.2f}%")
    print()
    print("  Trades")
    print("  " + "-" * 74)
    print(f"  Total trades:        {results['total_trades']}")
    print(f"  Winning trades:      {results['winning_trades']}")
    print(f"  Losing trades:       {results['losing_trades']}")
    print(f"  Win rate:            {results['win_rate_pct']:.2f}%")
    print(f"  Paused during run:   {'YES' if results['paused_during_backtest'] else 'NO'}")
    print()
    print("  Results by coin")
    print("  " + "-" * 74)
    print(
        f"  {'Coin':<6} {'Trades':>7} {'Buys':>6} {'Sells':>6} "
        f"{'Realized PnL':>14} {'Win rate':>10}"
    )
    print("  " + "-" * 74)

    for coin, stats in results["results_by_coin"].items():
        print(
            f"  {coin:<6} {stats['total_trades']:>7} {stats['buy_trades']:>6} "
            f"{stats['sell_trades']:>6} ${stats['realized_pnl']:>+12,.2f} "
            f"{stats['win_rate_pct']:>9.2f}%"
        )

    print("=" * 78)
    print()


def main() -> None:
    """Run the backtest, save results, and print a summary."""
    print("Starting read-only backtest...")
    print(f"  Price log: {PRICE_LOG_CSV}")
    print(f"  TEST_MODE: {'ON' if TEST_MODE else 'OFF'}")

    try:
        results = run_backtest()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nBacktest failed: {exc}")
        sys.exit(1)

    save_results(results)
    print_results(results)
    print(f"Results saved to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
