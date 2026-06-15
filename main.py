"""
Read-only Hyperliquid market analyzer.

Fetches public mid prices from Hyperliquid's Info API every 10 seconds,
tracks each coin's first observed price, keeps recent price history in memory,
and shows rolling 1-minute and 5-minute percentage changes with signals.

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

# Rolling window lengths for short-term change calculations
WINDOW_1M_SECONDS = 60
WINDOW_5M_SECONDS = 300

# Coins we want to display from the allMids response
WATCHLIST = ("BTC", "ETH", "SOL")

# Signal thresholds based on approximate 1-minute percentage change
SIGNAL_STRONG_UP_PCT = 0.50
SIGNAL_UP_PCT = 0.25
SIGNAL_STRONG_DOWN_PCT = -0.50
SIGNAL_DOWN_PCT = -0.25

# Where to save price history
DATA_DIR = Path("data")
PRICE_LOG_CSV = DATA_DIR / "price_log.csv"
CSV_COLUMNS = ("timestamp", "coin", "price")

# Each coin's in-memory history is a list of (unix_time, price) tuples
PriceHistory = dict[str, list[tuple[float, float]]]


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


def percent_change(old_price: float, current_price: float) -> float:
    """Calculate percentage change from an older price to the current price."""
    if old_price == 0:
        return 0.0
    return ((current_price - old_price) / old_price) * 100


def signal_from_1m_change(change_1m: float | None) -> str:
    """
    Return a signal based on approximate 1-minute percentage change.

    STRONG_UP   if 1-minute change > +0.50%
    UP          if 1-minute change > +0.25%
    STRONG_DOWN if 1-minute change < -0.50%
    DOWN        if 1-minute change < -0.25%
    FLAT        otherwise
    N/A         when there is not enough history yet
    """
    if change_1m is None:
        return "N/A"

    if change_1m > SIGNAL_STRONG_UP_PCT:
        return "STRONG_UP"
    if change_1m > SIGNAL_UP_PCT:
        return "UP"
    if change_1m < SIGNAL_STRONG_DOWN_PCT:
        return "STRONG_DOWN"
    if change_1m < SIGNAL_DOWN_PCT:
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


def record_price_history(
    price_history: PriceHistory,
    coin: str,
    price: float,
    now: float,
) -> None:
    """Append the latest price and drop entries older than the 5-minute window."""
    if coin not in price_history:
        price_history[coin] = []

    price_history[coin].append((now, price))

    # Keep a little extra beyond 5 minutes so lookups stay accurate
    cutoff = now - WINDOW_5M_SECONDS - REFRESH_INTERVAL_SECONDS
    price_history[coin] = [(t, p) for t, p in price_history[coin] if t >= cutoff]


def price_at_or_before(history: list[tuple[float, float]], target_time: float) -> float | None:
    """
    Find the most recent stored price at or before target_time.

    History entries are stored in time order (oldest first).
    """
    reference_price = None

    for entry_time, entry_price in history:
        if entry_time <= target_time:
            reference_price = entry_price
        else:
            break

    return reference_price


def rolling_percent_change(
    history: list[tuple[float, float]],
    current_price: float,
    now: float,
    window_seconds: int,
) -> float | None:
    """
    Calculate approximate percentage change over a rolling time window.

    Returns None when we do not yet have a price old enough for the window.
    """
    if not history:
        return None

    target_time = now - window_seconds

    # The oldest entry must be at or before the window start
    if history[0][0] > target_time:
        return None

    reference_price = price_at_or_before(history, target_time)
    if reference_price is None:
        return None

    return percent_change(reference_price, current_price)


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
