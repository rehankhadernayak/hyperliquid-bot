"""
Simulated paper trading based on market signals.

Uses a fake USDC balance and tracks positions for BTC, ETH, and SOL.
No wallet, signing, or real exchange orders — read-only simulation only.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

from market_analyzer import WATCHLIST

# Starting fake account settings
STARTING_BALANCE_USDC = 1000.0
TRADE_SIZE_USDC = 100.0

# Where to save paper trade history
DATA_DIR = Path("data")
PAPER_TRADES_CSV = DATA_DIR / "paper_trades.csv"
CSV_COLUMNS = (
    "timestamp",
    "coin",
    "action",
    "price",
    "quantity",
    "trade_value",
    "cash_balance",
    "realized_pnl",
)

# Signals that trigger fake trades
BUY_SIGNALS = frozenset({"STRONG_DOWN", "DOWN"})
SELL_SIGNALS = frozenset({"STRONG_UP", "UP"})


@dataclass
class Position:
    """A fake open position for one coin."""

    quantity: float = 0.0
    avg_entry_price: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.quantity > 0


@dataclass
class PaperTrader:
    """
    Tracks fake cash, positions, and PnL.

    BUY on DOWN / STRONG_DOWN signals (contrarian: buy when price is falling).
    SELL on UP / STRONG_UP signals when a position is already open.
    Each trade uses TRADE_SIZE_USDC worth of notional.
    """

    cash_balance: float = STARTING_BALANCE_USDC
    realized_pnl: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)

    def process_signal(self, coin: str, signal: str, price: float, timestamp: str) -> None:
        """Apply one signal for a coin at the current price."""
        if coin not in WATCHLIST:
            return

        if signal in BUY_SIGNALS:
            self._fake_buy(coin, price, timestamp, signal)
        elif signal in SELL_SIGNALS:
            self._fake_sell(coin, price, timestamp, signal)

    def unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        """Sum unrealized PnL across all open positions."""
        total = 0.0

        for coin, position in self.positions.items():
            if not position.is_open:
                continue

            current_price = current_prices.get(coin)
            if current_price is None:
                continue

            total += (current_price - position.avg_entry_price) * position.quantity

        return total

    def _get_position(self, coin: str) -> Position:
        if coin not in self.positions:
            self.positions[coin] = Position()
        return self.positions[coin]

    def _fake_buy(self, coin: str, price: float, timestamp: str, signal: str) -> None:
        if self.cash_balance < TRADE_SIZE_USDC:
            return

        quantity = TRADE_SIZE_USDC / price
        position = self._get_position(coin)

        # Weighted average entry when adding to an existing position
        total_cost = (position.avg_entry_price * position.quantity) + TRADE_SIZE_USDC
        position.quantity += quantity
        position.avg_entry_price = total_cost / position.quantity

        self.cash_balance -= TRADE_SIZE_USDC

        self._log_trade(
            timestamp=timestamp,
            coin=coin,
            action="BUY",
            price=price,
            quantity=quantity,
            trade_value=TRADE_SIZE_USDC,
            trade_realized_pnl=0.0,
            signal=signal,
        )

    def _fake_sell(self, coin: str, price: float, timestamp: str, signal: str) -> None:
        position = self._get_position(coin)

        if not position.is_open:
            return

        position_value = position.quantity * price

        if position_value <= TRADE_SIZE_USDC:
            quantity = position.quantity
            trade_value = position_value
        else:
            trade_value = TRADE_SIZE_USDC
            quantity = trade_value / price

            remaining_value = (position.quantity - quantity) * price
            if remaining_value < TRADE_SIZE_USDC:
                quantity = position.quantity
                trade_value = position_value

        trade_realized_pnl = (price - position.avg_entry_price) * quantity
        self.realized_pnl += trade_realized_pnl
        self.cash_balance += trade_value

        position.quantity -= quantity
        if position.quantity <= 0 or position.quantity * price < 0.01:
            position.quantity = 0.0
            position.avg_entry_price = 0.0

        self._log_trade(
            timestamp=timestamp,
            coin=coin,
            action="SELL",
            price=price,
            quantity=quantity,
            trade_value=trade_value,
            trade_realized_pnl=trade_realized_pnl,
            signal=signal,
        )

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
        print_paper_trade_proposal(
            timestamp=timestamp,
            coin=coin,
            action=action,
            price=price,
            trade_value=trade_value,
            signal=signal,
        )

        ensure_data_dir()

        write_header = not PAPER_TRADES_CSV.exists()

        with PAPER_TRADES_CSV.open("a", newline="") as csv_file:
            writer = csv.writer(csv_file)

            if write_header:
                writer.writerow(CSV_COLUMNS)

            writer.writerow(
                [
                    timestamp,
                    coin,
                    action,
                    f"{price:.8f}",
                    f"{quantity:.8f}",
                    f"{trade_value:.2f}",
                    f"{self.cash_balance:.2f}",
                    f"{trade_realized_pnl:.2f}",
                ]
            )


def print_paper_trade_proposal(
    timestamp: str,
    coin: str,
    action: str,
    price: float,
    trade_value: float,
    signal: str,
) -> None:
    """Print a readable paper trade proposal when a simulated trade executes."""
    print()
    print("  >> PAPER TRADE PROPOSAL")
    print(f"     Timestamp:   {timestamp}")
    print(f"     Coin:        {coin}")
    print(f"     Action:      {action}")
    print(f"     Price:       ${price:,.2f}")
    print(f"     Trade value: ${trade_value:,.2f}")
    print(f"     Signal:      {signal}")
    print()


def ensure_data_dir() -> None:
    """Create the data folder if it does not exist."""
    DATA_DIR.mkdir(exist_ok=True)


def format_usdc(amount: float) -> str:
    """Format a USDC amount for display."""
    sign = "+" if amount >= 0 else ""
    return f"{sign}${amount:,.2f}"


def format_quantity(quantity: float) -> str:
    """Format a coin quantity for display."""
    if quantity >= 1:
        return f"{quantity:.6f}"
    return f"{quantity:.8f}"


def print_paper_trading_summary(
    paper_trader: PaperTrader,
    current_prices: dict[str, float],
) -> None:
    """Print cash, PnL, and open positions below the market analysis table."""
    unrealized = paper_trader.unrealized_pnl(current_prices)
    total_equity = paper_trader.cash_balance + sum(
        paper_trader.positions[coin].quantity * current_prices[coin]
        for coin in WATCHLIST
        if coin in current_prices and paper_trader.positions.get(coin, Position()).is_open
    )

    print()
    print("Paper trading summary:")
    print("-" * 78)
    print(f"  Cash balance:    ${paper_trader.cash_balance:,.2f}")
    print(f"  Realized PnL:    {format_usdc(paper_trader.realized_pnl)}")
    print(f"  Unrealized PnL:  {format_usdc(unrealized)}")
    print(f"  Total equity:    ${total_equity:,.2f}")
    print()
    print(
        f"  {'Coin':<6} {'Quantity':>14} {'Avg Entry':>14} "
        f"{'Current':>14} {'Unrealized':>12}"
    )
    print("  " + "-" * 74)

    open_positions = False

    for coin in WATCHLIST:
        position = paper_trader.positions.get(coin, Position())
        current_price = current_prices.get(coin)

        if not position.is_open or current_price is None:
            continue

        open_positions = True
        coin_unrealized = (current_price - position.avg_entry_price) * position.quantity

        print(
            f"  {coin:<6} {format_quantity(position.quantity):>14} "
            f"${position.avg_entry_price:>12,.2f} ${current_price:>12,.2f} "
            f"{format_usdc(coin_unrealized):>12}"
        )

    if not open_positions:
        print("  (no open positions)")

    print("-" * 78)
