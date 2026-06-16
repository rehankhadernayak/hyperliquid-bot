"""
Simulated paper trading based on market signals.

Uses a fake USDC balance and tracks positions for BTC, ETH, and SOL.
No wallet, signing, or real exchange orders — read-only simulation only.
"""

import csv
from dataclasses import dataclass, field

from config import DATA, MARKET, PAPER_TRADING
from risk_rules import buy_block_reason, sell_block_reason
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
    entry_count: int = 0
    last_buy_time: float | None = None

    @property
    def is_open(self) -> bool:
        return self.quantity > 0

    @property
    def allocation_usdc(self) -> float:
        """USDC cost basis currently tied up in this position."""
        return self.quantity * self.avg_entry_price


@dataclass
class PaperTrader:
    """
    Tracks fake cash, positions, and PnL.

    BUY on DOWN / STRONG_DOWN signals (contrarian: buy when price is falling).
    SELL on UP / STRONG_UP signals when a position is already open.
    Each trade uses PAPER_TRADING.trade_size_usdc worth of notional.
    """

    cash_balance: float = PAPER_TRADING.starting_balance_usdc
    realized_pnl: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    successful_trades_this_session: int = 0
    is_paused: bool = False
    _equity_pause_announced: bool = False

    def total_open_exposure(self) -> float:
        """Sum of cost basis across all open positions (USDC tied up)."""
        return sum(
            position.allocation_usdc
            for position in self.positions.values()
            if position.is_open
        )

    def total_equity(self, current_prices: dict[str, float]) -> float:
        """Cash balance plus current market value of all open positions."""
        positions_value = 0.0

        for coin, position in self.positions.items():
            if not position.is_open:
                continue

            current_price = current_prices.get(coin)
            if current_price is None:
                continue

            positions_value += position.quantity * current_price

        return self.cash_balance + positions_value

    def update_session_risk(self, current_prices: dict[str, float]) -> None:
        """Check equity and pause paper trading if it falls below the session floor."""
        if self.is_paused:
            return

        equity = self.total_equity(current_prices)
        if equity < PAPER_TRADING.min_equity_before_pause_usdc:
            self.is_paused = True

            if not self._equity_pause_announced:
                print()
                print("  >> PAPER TRADING PAUSED")
                print("     Reason:      equity protection triggered")
                print(f"     Total equity: ${equity:,.2f}")
                print(f"     Minimum:      ${PAPER_TRADING.min_equity_before_pause_usdc:,.2f}")
                print()
                self._equity_pause_announced = True

    def process_signal(
        self,
        coin: str,
        signal: str,
        price: float,
        timestamp: str,
        now: float,
    ) -> None:
        """Apply one signal for a coin at the current price."""
        if coin not in MARKET.watchlist:
            return

        if self.is_paused:
            return

        if signal in BUY_SIGNALS:
            self._fake_buy(coin, price, timestamp, signal, now)
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

    def session_trade_limit(self) -> int:
        """Maximum successful trades allowed this session (subclasses may override)."""
        return PAPER_TRADING.max_successful_trades_per_session

    def execution_price(self, action: str, mid_price: float) -> float:
        """Price used for fills; subclasses may add slippage."""
        return mid_price

    def buy_fee(self, trade_value: float) -> float:
        """Fee charged on a BUY notional (subclasses may override)."""
        return 0.0

    def sell_fee(self, trade_value: float) -> float:
        """Fee charged on SELL proceeds (subclasses may override)."""
        return 0.0

    def _fake_buy(
        self,
        coin: str,
        price: float,
        timestamp: str,
        signal: str,
        now: float,
    ) -> None:
        block_reason = buy_block_reason(self, coin, now)
        if block_reason is not None:
            print_skipped_trade_proposal(
                timestamp=timestamp,
                coin=coin,
                action="BUY",
                price=price,
                signal=signal,
                reason=block_reason,
            )
            return

        execution_price = self.execution_price("BUY", price)
        quantity = PAPER_TRADING.trade_size_usdc / execution_price
        position = self._get_position(coin)

        # Weighted average entry when adding to an existing position
        total_cost = (
            (position.avg_entry_price * position.quantity) + PAPER_TRADING.trade_size_usdc
        )
        position.quantity += quantity
        position.avg_entry_price = total_cost / position.quantity
        position.entry_count += 1
        position.last_buy_time = now

        buy_fee = self.buy_fee(PAPER_TRADING.trade_size_usdc)
        self.cash_balance -= PAPER_TRADING.trade_size_usdc + buy_fee

        self._log_trade(
            timestamp=timestamp,
            coin=coin,
            action="BUY",
            price=execution_price,
            quantity=quantity,
            trade_value=PAPER_TRADING.trade_size_usdc,
            trade_realized_pnl=0.0,
            signal=signal,
        )

    def _fake_sell(self, coin: str, price: float, timestamp: str, signal: str) -> None:
        position = self._get_position(coin)

        if not position.is_open:
            return

        block_reason = sell_block_reason(self, coin)
        if block_reason is not None:
            print_skipped_trade_proposal(
                timestamp=timestamp,
                coin=coin,
                action="SELL",
                price=price,
                signal=signal,
                reason=block_reason,
            )
            return

        execution_price = self.execution_price("SELL", price)
        position_value = position.quantity * execution_price

        if position_value <= PAPER_TRADING.trade_size_usdc:
            quantity = position.quantity
            trade_value = position_value
        else:
            trade_value = PAPER_TRADING.trade_size_usdc
            quantity = trade_value / execution_price

            remaining_value = (position.quantity - quantity) * execution_price
            if remaining_value < PAPER_TRADING.trade_size_usdc:
                quantity = position.quantity
                trade_value = position_value

        trade_realized_pnl = (execution_price - position.avg_entry_price) * quantity
        sell_fee = self.sell_fee(trade_value)
        self.realized_pnl += trade_realized_pnl
        self.cash_balance += trade_value - sell_fee

        position.quantity -= quantity
        if position.quantity <= 0 or position.quantity * execution_price < 0.01:
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

        self.successful_trades_this_session += 1

        ensure_data_dir()

        write_header = not DATA.paper_trades_csv.exists()

        with DATA.paper_trades_csv.open("a", newline="") as csv_file:
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


def print_skipped_trade_proposal(
    timestamp: str,
    coin: str,
    action: str,
    price: float,
    signal: str,
    reason: str,
) -> None:
    """Print a readable skip message when risk rules block a fake trade."""
    print()
    print("  >> SKIPPED PAPER TRADE PROPOSAL")
    print(f"     Timestamp:   {timestamp}")
    print(f"     Coin:        {coin}")
    print(f"     Action:      {action} (skipped)")
    print(f"     Price:       ${price:,.2f}")
    print(f"     Trade value: ${PAPER_TRADING.trade_size_usdc:,.2f}")
    print(f"     Signal:      {signal}")
    print(f"     Reason:      {reason}")
    print()


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
    DATA.data_dir.mkdir(exist_ok=True)


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
    total_equity = paper_trader.total_equity(current_prices)
    open_exposure = paper_trader.total_open_exposure()
    paused_label = "YES (equity protection)" if paper_trader.is_paused else "NO"

    print()
    print("Paper trading summary:")
    print("-" * 78)
    print(f"  Status:          Paused: {paused_label}")
    print(f"  Cash balance:    ${paper_trader.cash_balance:,.2f}")
    print(f"  Realized PnL:    {format_usdc(paper_trader.realized_pnl)}")
    print(f"  Unrealized PnL:  {format_usdc(unrealized)}")
    print(f"  Total equity:    ${total_equity:,.2f}")
    print(f"  Open exposure:   ${open_exposure:,.2f}")
    print(
        f"  Session trades:  {paper_trader.successful_trades_this_session}"
        f" / {PAPER_TRADING.max_successful_trades_per_session}"
    )
    print()
    print(
        f"  {'Coin':<6} {'Quantity':>14} {'Avg Entry':>14} "
        f"{'Current':>14} {'Unrealized':>12}"
    )
    print("  " + "-" * 74)

    open_positions = False

    for coin in MARKET.watchlist:
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
