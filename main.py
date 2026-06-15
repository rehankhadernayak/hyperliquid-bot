"""
Read-only Hyperliquid market analyzer.

Fetches public mid prices from Hyperliquid's Info API every 10 seconds,
tracks each coin's first observed price, and shows percentage change
with a simple UP / DOWN / FLAT signal.

No wallet, signing, or trading involved.
"""

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# Hyperliquid's public read-only Info API (mainnet)
INFO_API_URL = "https://api.hyperliquid.xyz/info"

# How often to refresh prices (seconds)
REFRESH_INTERVAL_SECONDS = 10

# Coins we want to display from the allMids response
WATCHLIST = ("BTC", "ETH", "SOL")

# Signal threshold: move beyond this % from start price triggers UP or DOWN
SIGNAL_THRESHOLD_PCT = 0.25

# Where to save price history
DATA_DIR = Path("data")
PRICE_LOG_CSV = DATA_DIR / "price_log.csv"
CSV_COLUMNS = ("timestamp", "coin", "price")


def fetch_all_mids() -> dict[str, str]:
    """
    Request mid prices for all coins from the Info API.

    Returns a dict like {"BTC": "43250.5", "ETH": "2650.25", ...}
    """
    payload = {"type": "allMids"}

    # POST JSON to the public info endpoint (read-only, no auth needed)
    response = requests.post(
        INFO_API_URL,
        json=payload,
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, dict):
        raise ValueError("Unexpected response: expected a JSON object of coin prices")

    return data


def price_to_float(price_str: str) -> float:
    """Convert a price string from the API (e.g. '43250.5') into a float."""
    return float(price_str)


def current_timestamp() -> str:
    """Return the current UTC time as a readable string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def percent_change(start_price: float, current_price: float) -> float:
    """Calculate percentage change from the start price to the current price."""
    if start_price == 0:
        return 0.0
    return ((current_price - start_price) / start_price) * 100


def signal_from_change(change_pct: float) -> str:
    """
    Return a simple signal based on percentage change from start price.

    UP   if change is greater than +0.25%
    DOWN if change is less than -0.25%
    FLAT otherwise
    """
    if change_pct > SIGNAL_THRESHOLD_PCT:
        return "UP"
    if change_pct < -SIGNAL_THRESHOLD_PCT:
        return "DOWN"
    return "FLAT"


def update_start_prices(start_prices: dict[str, float], mids: dict[str, str]) -> None:
    """
    Record the first price we see for each coin during this run.

    Once a start price is stored for a coin, it is not overwritten.
    """
    for coin in WATCHLIST:
        if coin in start_prices:
            continue

        price_str = mids.get(coin)
        if price_str is None:
            continue

        try:
            start_prices[coin] = price_to_float(price_str)
        except ValueError:
            continue


def ensure_data_dir() -> None:
    """Create the data folder if it does not exist."""
    DATA_DIR.mkdir(exist_ok=True)


def append_prices_to_csv(mids: dict[str, str], timestamp: str) -> None:
    """Append watchlist mid prices to data/price_log.csv."""
    ensure_data_dir()

    write_header = not PRICE_LOG_CSV.exists()

    with PRICE_LOG_CSV.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        if write_header:
            writer.writerow(CSV_COLUMNS)

        for coin in WATCHLIST:
            price_str = mids.get(coin)

            if price_str is None:
                continue

            try:
                price = price_to_float(price_str)
                writer.writerow([timestamp, coin, price])
            except ValueError:
                continue


def format_price(price: float) -> str:
    """Format a price for display (e.g. $95,000.00)."""
    return f"${price:,.2f}"


def format_change(change_pct: float) -> str:
    """Format a percentage change with sign (e.g. +0.21% or -0.35%)."""
    sign = "+" if change_pct >= 0 else ""
    return f"{sign}{change_pct:.2f}%"


def print_market_analysis(
    mids: dict[str, str],
    start_prices: dict[str, float],
    timestamp: str | None = None,
) -> str:
    """Print current price, start price, change, and signal for BTC, ETH, and SOL."""
    if timestamp is None:
        timestamp = current_timestamp()

    print(f"\n[{timestamp}]")
    print("Hyperliquid market analysis:")
    print("-" * 62)
    print(f"{'Coin':<6} {'Current':>14} {'Start':>14} {'Change':>10} {'Signal':>8}")
    print("-" * 62)

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

        start_price = start_prices.get(coin)

        if start_price is None:
            # First tick: we just recorded the start price but have no prior baseline yet
            print(
                f"{coin:<6} {format_price(current_price):>14} "
                f"{'(recording)':>14} {'—':>10} {'—':>8}"
            )
            continue

        change_pct = percent_change(start_price, current_price)
        signal = signal_from_change(change_pct)

        print(
            f"{coin:<6} {format_price(current_price):>14} "
            f"{format_price(start_price):>14} {format_change(change_pct):>10} {signal:>8}"
        )

    return timestamp


def handle_request_error(exc: Exception) -> None:
    """Print a friendly error message for a failed API request."""
    if isinstance(exc, requests.exceptions.Timeout):
        print("Error: request timed out. Will retry on the next interval.")
    elif isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else "unknown"
        print(f"Error: HTTP {status} from Hyperliquid API. Will retry on the next interval.")
    elif isinstance(exc, requests.exceptions.RequestException):
        print(f"Error: network request failed ({exc}). Will retry on the next interval.")
    elif isinstance(exc, ValueError):
        print(f"Error: {exc}. Will retry on the next interval.")
    else:
        print(f"Error: {exc}. Will retry on the next interval.")


def run_analyzer() -> None:
    """
    Fetch prices, track start prices, and print analysis in a loop.

    Runs until the user presses Ctrl+C.
    """
    print("Starting Hyperliquid market analyzer (Ctrl+C to stop)...")
    print(f"Signal threshold: ±{SIGNAL_THRESHOLD_PCT}% from first observed price")

    # First observed price for each coin during this run
    start_prices: dict[str, float] = {}

    while True:
        try:
            mids = fetch_all_mids()
            update_start_prices(start_prices, mids)
            timestamp = print_market_analysis(mids, start_prices)
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
