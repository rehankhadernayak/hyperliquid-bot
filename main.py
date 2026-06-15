"""
Read-only Hyperliquid market scanner.

Fetches public mid prices from Hyperliquid's Info API every 10 seconds.
No wallet, signing, or trading involved.
"""

import sys
import time
from datetime import datetime, timezone

import requests

# Hyperliquid's public read-only Info API (mainnet)
INFO_API_URL = "https://api.hyperliquid.xyz/info"

# How often to refresh prices (seconds)
REFRESH_INTERVAL_SECONDS = 10

# Coins we want to display from the allMids response
WATCHLIST = ("BTC", "ETH", "SOL")


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


def print_watchlist_prices(mids: dict[str, str]) -> None:
    """Print mid prices for BTC, ETH, and SOL with a timestamp."""
    print(f"\n[{current_timestamp()}]")
    print("Hyperliquid mid prices:")
    print("-" * 28)

    for coin in WATCHLIST:
        price_str = mids.get(coin)

        if price_str is None:
            print(f"{coin}: not available")
            continue

        try:
            price = price_to_float(price_str)
            print(f"{coin}: ${price:,.2f}")
        except ValueError:
            print(f"{coin}: invalid price ({price_str!r})")


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


def run_scanner() -> None:
    """
    Fetch and print watchlist prices in a loop.

    Runs until the user presses Ctrl+C.
    """
    print("Starting Hyperliquid market scanner (Ctrl+C to stop)...")

    while True:
        try:
            mids = fetch_all_mids()
            print_watchlist_prices(mids)
        except Exception as exc:
            print(f"\n[{current_timestamp()}]")
            handle_request_error(exc)

        # Wait before the next update (Ctrl+C can interrupt this sleep)
        time.sleep(REFRESH_INTERVAL_SECONDS)


def main() -> None:
    try:
        run_scanner()
    except KeyboardInterrupt:
        print("\nScanner stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
