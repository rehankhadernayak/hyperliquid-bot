"""
Read-only Hyperliquid market analyzer.

Fetches public mid prices from Hyperliquid's Info API every 10 seconds,
tracks each coin's first observed price, keeps recent price history in memory,
and shows rolling 1-minute and 5-minute percentage changes with signals.

No wallet, signing, or trading involved.
"""

import sys
import time
from datetime import datetime, timezone

from data_logger import append_prices_to_csv
from hyperliquid_client import fetch_all_mids, handle_request_error
from market_analyzer import (
    REFRESH_INTERVAL_SECONDS,
    WINDOW_1M_SECONDS,
    WINDOW_5M_SECONDS,
    WATCHLIST,
    PriceHistory,
    percent_change,
    price_to_float,
    record_price_history,
    rolling_percent_change,
    update_start_prices,
)
from signals import SIGNAL_STRONG_UP_PCT, SIGNAL_UP_PCT, signal_from_1m_change


def current_timestamp() -> str:
    """Return the current UTC time as a readable string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


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


def print_market_analysis(
    mids: dict[str, str],
    start_prices: dict[str, float],
    price_history: PriceHistory,
    timestamp: str | None = None,
) -> str:
    """Print current price, rolling changes, and signals for BTC, ETH, and SOL."""
    if timestamp is None:
        timestamp = current_timestamp()

    now = time.time()

    print(f"\n[{timestamp}]")
    print("Hyperliquid market analysis:")
    print("-" * 78)
    print(
        f"{'Coin':<6} {'Current':>14} {'Since Start':>12} "
        f"{'1-Min':>10} {'5-Min':>10} {'Signal':>12}"
    )
    print("-" * 78)

    for coin in WATCHLIST:
        price_str = mids.get(coin)

        if price_str is None:
            print(f"{coin:<6} {'not available':>14}")
            continue

        try:
            current_price = price_to_float(price_str)
        except ValueError:
            print(f"{coin:<6} invalid price ({price_str!r})")
            continue

        record_price_history(price_history, coin, current_price, now)

        start_price = start_prices.get(coin)
        history = price_history.get(coin, [])

        change_since_start = (
            percent_change(start_price, current_price) if start_price is not None else None
        )
        change_1m = rolling_percent_change(history, current_price, now, WINDOW_1M_SECONDS)
        change_5m = rolling_percent_change(history, current_price, now, WINDOW_5M_SECONDS)
        signal = signal_from_1m_change(change_1m)

        if start_price is None:
            print(
                f"{coin:<6} {format_price(current_price):>14} "
                f"{'(recording)':>12} {'N/A':>10} {'N/A':>10} {'N/A':>12}"
            )
            continue

        print(
            f"{coin:<6} {format_price(current_price):>14} "
            f"{format_change_or_na(change_since_start):>12} "
            f"{format_change_or_na(change_1m):>10} "
            f"{format_change_or_na(change_5m):>10} {signal:>12}"
        )

    return timestamp


def run_analyzer() -> None:
    """
    Fetch prices, track history, and print analysis in a loop.

    Runs until the user presses Ctrl+C.
    """
    print("Starting Hyperliquid market analyzer (Ctrl+C to stop)...")
    print(
        "Signals use 1-minute rolling change "
        f"(±{SIGNAL_UP_PCT}% / ±{SIGNAL_STRONG_UP_PCT}% thresholds)"
    )

    # First observed price for each coin during this run
    start_prices: dict[str, float] = {}
    # Recent (timestamp, price) history for rolling window calculations
    price_history: PriceHistory = {}

    while True:
        try:
            mids = fetch_all_mids()
            update_start_prices(start_prices, mids)
            timestamp = print_market_analysis(mids, start_prices, price_history)
            append_prices_to_csv(mids, timestamp)
        except Exception as exc:
            print(f"\n[{current_timestamp()}]")
            handle_request_error(exc)

        # Wait before the next update (Ctrl+C can interrupt this sleep)
        time.sleep(REFRESH_INTERVAL_SECONDS)


def main() -> None:
    try:
        run_analyzer()
    except KeyboardInterrupt:
        print("\nAnalyzer stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
