"""
Read-only backtesting using historical prices from data/price_log.csv.

Replays saved prices in timestamp order and runs the same signal logic and
paper trading rules as the live analyzer — without calling Hyperliquid,
using wallets, or placing real orders.
"""

from __future__ import annotations

import copy
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

PRICE_LOG_CSV = Path("data") / "price_log.csv"
RESULTS_JSON = Path("data") / "backtest_results.json"

# Backtest-only defaults (live PaperTrader limits are unchanged).
DEFAULT_BACKTEST_MAX_SESSION_TRADES = 1000

# Small, realistic simulation costs applied to every BUY and SELL fill.
# 0.05% taker fee and 0.02% slippage per side are typical for liquid perps.
DEFAULT_TRADING_FEE_PCT = 0.05
DEFAULT_SLIPPAGE_PCT = 0.02


@dataclass
class BacktestConfig:
    """Settings used only during backtests — does not affect live paper trading."""

    max_session_trades: int = DEFAULT_BACKTEST_MAX_SESSION_TRADES
    trading_fee_pct: float = DEFAULT_TRADING_FEE_PCT
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT
    close_positions_at_end: bool = True


@dataclass
class BacktestPaperTrader(PaperTrader):
    """
    Separate paper trader for backtests.

    Does not write to data/paper_trades.csv or print per-trade messages
    during replay, so live paper trading state stays untouched.
    """

    config: BacktestConfig = field(default_factory=BacktestConfig)
    trade_log: list[dict] = field(default_factory=list)
    total_fees_paid: float = 0.0

    def session_trade_limit(self) -> int:
        return self.config.max_session_trades

    def execution_price(self, action: str, mid_price: float) -> float:
        slip = self.config.slippage_pct / 100.0
        if action == "BUY":
            return mid_price * (1 + slip)
        return mid_price * (1 - slip)

    def buy_fee(self, trade_value: float) -> float:
        fee = trade_value * (self.config.trading_fee_pct / 100.0)
        self.total_fees_paid += fee
        return fee

    def sell_fee(self, trade_value: float) -> float:
        fee = trade_value * (self.config.trading_fee_pct / 100.0)
        self.total_fees_paid += fee
        return fee

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

    def force_close_all_positions(
        self, current_prices: dict[str, float], timestamp: str
    ) -> None:
        """Close every open position at the final available prices."""
        for coin in WATCHLIST:
            position = self._get_position(coin)
            if not position.is_open:
                continue

            price = current_prices.get(coin)
            if price is None:
                continue

            self._force_sell_full_position(coin, price, timestamp)

    def _force_sell_full_position(
        self, coin: str, price: float, timestamp: str
    ) -> None:
        """Sell an entire open position, bypassing session trade limits."""
        position = self._get_position(coin)
        if not position.is_open:
            return

        execution_price = self.execution_price("SELL", price)
        quantity = position.quantity
        trade_value = quantity * execution_price
        trade_realized_pnl = (execution_price - position.avg_entry_price) * quantity
        sell_fee = self.sell_fee(trade_value)

        self.realized_pnl += trade_realized_pnl
        self.cash_balance += trade_value - sell_fee

        position.quantity = 0.0
        position.avg_entry_price = 0.0
        position.entry_count = 0
        position.last_buy_time = None

        self._log_trade(
            timestamp=timestamp,
            coin=coin,
            action="SELL",
            price=execution_price,
            quantity=quantity,
            trade_value=trade_value,
            trade_realized_pnl=trade_realized_pnl,
            signal="FORCED_CLOSE",
        )


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
        coin_buys = [trade for trade in coin_trades if trade["action"] == "BUY"]

        winning = sum(1 for trade in coin_sells if trade["realized_pnl"] > 0)
        losing = sum(1 for trade in coin_sells if trade["realized_pnl"] < 0)
        closed = winning + losing
        win_pnls = [trade["realized_pnl"] for trade in coin_sells if trade["realized_pnl"] > 0]
        loss_pnls = [trade["realized_pnl"] for trade in coin_sells if trade["realized_pnl"] < 0]

        by_coin[coin] = {
            "total_buys": len(coin_buys),
            "total_sells": len(coin_sells),
            "completed_trades": closed,
            "realized_pnl": round(sum(trade["realized_pnl"] for trade in coin_sells), 2),
            "winning_closed_trades": winning,
            "losing_closed_trades": losing,
            "win_rate_pct": round((winning / closed) * 100, 2) if closed else 0.0,
            "average_win": round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0.0,
            "average_loss": round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0.0,
        }

    return by_coin


def _profit_factor(sell_trades: list[dict]) -> float | None:
    gross_wins = sum(trade["realized_pnl"] for trade in sell_trades if trade["realized_pnl"] > 0)
    gross_losses = abs(
        sum(trade["realized_pnl"] for trade in sell_trades if trade["realized_pnl"] < 0)
    )
    if gross_losses == 0:
        return None if gross_wins == 0 else float("inf")
    return gross_wins / gross_losses


def build_mode_results(
    paper_trader: BacktestPaperTrader,
    equity_curve: list[float],
    current_prices: dict[str, float],
    *,
    label: str,
) -> dict:
    """Compute the full metric set for one strategy mode."""
    sell_trades = [trade for trade in paper_trader.trade_log if trade["action"] == "SELL"]
    buy_trades = [trade for trade in paper_trader.trade_log if trade["action"] == "BUY"]

    winning = [trade for trade in sell_trades if trade["realized_pnl"] > 0]
    losing = [trade for trade in sell_trades if trade["realized_pnl"] < 0]
    closed_trades = len(winning) + len(losing)

    unrealized = paper_trader.unrealized_pnl(current_prices)
    ending_equity = paper_trader.total_equity(current_prices)
    total_pnl = paper_trader.realized_pnl + unrealized
    total_return_pct = (
        ((ending_equity - STARTING_BALANCE_USDC) / STARTING_BALANCE_USDC) * 100
    )

    avg_win = sum(trade["realized_pnl"] for trade in winning) / len(winning) if winning else 0.0
    avg_loss = sum(trade["realized_pnl"] for trade in losing) / len(losing) if losing else 0.0
    pf = _profit_factor(sell_trades)

    return {
        "label": label,
        "starting_equity": STARTING_BALANCE_USDC,
        "ending_equity": round(ending_equity, 2),
        "realized_pnl": round(paper_trader.realized_pnl, 2),
        "unrealized_pnl": round(unrealized, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "total_buys": len(buy_trades),
        "total_sells": len(sell_trades),
        "completed_trades": closed_trades,
        "winning_closed_trades": len(winning),
        "losing_closed_trades": len(losing),
        "win_rate_pct": round((len(winning) / closed_trades) * 100, 2) if closed_trades else 0.0,
        "average_win": round(avg_win, 2),
        "average_loss": round(avg_loss, 2),
        "profit_factor": round(pf, 2) if pf is not None and pf != float("inf") else pf,
        "maximum_drawdown_pct": round(compute_max_drawdown(equity_curve), 2),
        "final_cash": round(paper_trader.cash_balance, 2),
        "final_open_exposure": round(paper_trader.total_open_exposure(), 2),
        "total_fees_paid": round(paper_trader.total_fees_paid, 2),
        "paused_during_backtest": paper_trader.is_paused,
        "results_by_coin": summarize_by_coin(paper_trader.trade_log),
    }


def _silent_skip(**_kwargs) -> None:
    """No-op replacement for skip messages during replay."""


def _replay(
    timestamps: list[str],
    prices_by_timestamp: dict[str, dict[str, float]],
    paper_trader: BacktestPaperTrader,
) -> tuple[list[float], dict[str, float], str]:
    """Run the replay loop and return equity curve, final prices, last timestamp."""
    start_prices: dict[str, float] = {}
    price_history: PriceHistory = {}
    equity_curve: list[float] = [STARTING_BALANCE_USDC]
    current_prices: dict[str, float] = {}
    last_timestamp = timestamps[-1]

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
        last_timestamp = timestamp

    return equity_curve, current_prices, last_timestamp


def run_backtest(
    csv_path: Path = PRICE_LOG_CSV,
    test_mode: bool = True,
    config: BacktestConfig | None = None,
) -> dict:
    """
    Replay historical prices and return backtest metrics for one strategy mode.

    Uses a fresh BacktestPaperTrader instance so live paper trading files
    and state are never modified.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Price log not found: {csv_path}")

    backtest_config = config or BacktestConfig()
    apply_signal_thresholds(test_mode)

    timestamps, prices_by_timestamp = load_price_ticks(csv_path)
    if not timestamps:
        raise ValueError(f"No price rows found in {csv_path}")

    original_skip_printer = paper_trader_module.print_skipped_trade_proposal
    paper_trader_module.print_skipped_trade_proposal = _silent_skip

    paper_trader = BacktestPaperTrader(config=backtest_config)

    try:
        equity_curve, current_prices, last_timestamp = _replay(
            timestamps, prices_by_timestamp, paper_trader
        )
    finally:
        paper_trader_module.print_skipped_trade_proposal = original_skip_printer

    label = "test_mode" if test_mode else "production"
    with_open_positions = build_mode_results(
        paper_trader, equity_curve, current_prices, label=label
    )

    after_forced_close = None
    if backtest_config.close_positions_at_end:
        trader_closed = copy.deepcopy(paper_trader)
        trader_closed.force_close_all_positions(current_prices, last_timestamp)
        closed_equity_curve = equity_curve + [
            trader_closed.total_equity(current_prices)
        ]
        after_forced_close = build_mode_results(
            trader_closed, closed_equity_curve, current_prices, label=label
        )

    return {
        "test_mode": test_mode,
        "with_open_positions": with_open_positions,
        "after_forced_close": after_forced_close,
    }


def run_comparison(
    csv_path: Path = PRICE_LOG_CSV,
    config: BacktestConfig | None = None,
) -> dict:
    """Run both TEST_MODE and production thresholds and return a full comparison."""
    backtest_config = config or BacktestConfig()

    if not csv_path.exists():
        raise FileNotFoundError(f"Price log not found: {csv_path}")

    timestamps, _ = load_price_ticks(csv_path)
    if not timestamps:
        raise ValueError(f"No price rows found in {csv_path}")

    test_results = run_backtest(csv_path, test_mode=True, config=backtest_config)
    production_results = run_backtest(
        csv_path, test_mode=False, config=backtest_config
    )

    return {
        "source_file": str(csv_path),
        "timestamps_replayed": len(timestamps),
        "simulation": {
            "max_session_trades": backtest_config.max_session_trades,
            "trading_fee_pct": backtest_config.trading_fee_pct,
            "slippage_pct": backtest_config.slippage_pct,
            "close_positions_at_end": backtest_config.close_positions_at_end,
        },
        "test_mode": test_results,
        "production_mode": production_results,
    }


def save_results(results: dict, output_path: Path = RESULTS_JSON) -> None:
    """Write backtest results to a JSON file."""
    output_path.parent.mkdir(exist_ok=True)

    with output_path.open("w") as json_file:
        json.dump(results, json_file, indent=2, allow_nan=False, default=_json_default)


def _json_default(value: object) -> object:
    if value == float("inf"):
        return "inf"
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _format_profit_factor(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == float("inf"):
        return "inf"
    return f"{value:.2f}"


def _print_mode_block(mode_results: dict, heading: str) -> None:
    """Print one strategy mode summary (prefers forced-close view when available)."""
    open_pos = mode_results["with_open_positions"]
    forced = mode_results.get("after_forced_close")
    display = forced if forced is not None else open_pos

    print()
    print(f"  {heading}")
    print("  " + "-" * 74)

    if forced is not None:
        print("  (after closing all open positions at final prices)")
        print()

    print(f"  Starting equity:     ${display['starting_equity']:,.2f}")
    print(f"  Ending equity:       ${display['ending_equity']:,.2f}")
    print(f"  Total return:        {display['total_return_pct']:+.2f}%")
    print(f"  Realized PnL:        ${display['realized_pnl']:+,.2f}")
    print(f"  Unrealized PnL:      ${display['unrealized_pnl']:+,.2f}")
    print(f"  Total PnL:           ${display['total_pnl']:+,.2f}")
    print(f"  Max drawdown:        {display['maximum_drawdown_pct']:.2f}%")
    print(f"  Fees paid:           ${display['total_fees_paid']:,.2f}")
    print()
    print(f"  Buys: {display['total_buys']}  |  Sells: {display['total_sells']}  |  "
          f"Completed: {display['completed_trades']}")
    print(f"  Wins: {display['winning_closed_trades']}  |  Losses: {display['losing_closed_trades']}  |  "
          f"Win rate: {display['win_rate_pct']:.1f}%")
    print(f"  Avg win: ${display['average_win']:+.2f}  |  Avg loss: ${display['average_loss']:+.2f}  |  "
          f"Profit factor: {_format_profit_factor(display['profit_factor'])}")
    print(f"  Final cash: ${display['final_cash']:,.2f}  |  "
          f"Open exposure: ${display['final_open_exposure']:,.2f}")

    if forced is not None and open_pos["final_open_exposure"] > 0:
        print()
        print("  Before forced close (open positions kept):")
        print(f"    Ending equity: ${open_pos['ending_equity']:,.2f}  |  "
              f"Unrealized: ${open_pos['unrealized_pnl']:+,.2f}  |  "
              f"Open exposure: ${open_pos['final_open_exposure']:,.2f}")


def print_results(results: dict) -> None:
    """Print a readable comparison summary to the terminal."""
    sim = results["simulation"]

    print()
    print("=" * 78)
    print("BACKTEST COMPARISON")
    print("=" * 78)
    print(f"  Source:              {results['source_file']}")
    print(f"  Timestamps replayed: {results['timestamps_replayed']:,}")
    print(f"  Max session trades:  {sim['max_session_trades']:,}")
    print(f"  Trading fee:         {sim['trading_fee_pct']:.2f}% per side")
    print(f"  Slippage:            {sim['slippage_pct']:.2f}% per side")

    _print_mode_block(results["test_mode"], "TEST_MODE thresholds")
    _print_mode_block(results["production_mode"], "Production thresholds")

    print()
    print("=" * 78)
    print()


def main() -> None:
    """Run both strategy modes, save results, and print a summary."""
    print("Starting read-only backtest comparison...")
    print(f"  Price log: {PRICE_LOG_CSV}")
    print(f"  Modes:     TEST_MODE + Production")
    print(f"  Max trades per session: {DEFAULT_BACKTEST_MAX_SESSION_TRADES:,}")

    try:
        results = run_comparison()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nBacktest failed: {exc}")
        sys.exit(1)

    save_results(results)
    print_results(results)
    print(f"Full results saved to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
