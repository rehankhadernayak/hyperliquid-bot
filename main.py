"""
Read-only Hyperliquid market scanner.

Fetches public mid prices from Hyperliquid's Info API.
No wallet, signing, or trading involved.
"""

import sys

import requests

# Hyperliquid's public read-only Info API (mainnet)
INFO_API_URL = "https://api.hyperliquid.xyz/info"

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


def print_watchlist_prices(mids: dict[str, str]) -> None:
    """Print mid prices for BTC, ETH, and SOL when they are present."""
    print("Hyperliquid mid prices:")
    print("-" * 28)

    for coin in WATCHLIST:
        price = mids.get(coin)

        if price is None:
            print(f"{coin}: not available")
        else:
            print(f"{coin}: ${price}")


def main() -> None:
    try:
        mids = fetch_all_mids()
        print_watchlist_prices(mids)
    except requests.exceptions.Timeout:
        print("Error: request timed out. Check your internet connection and try again.")
        sys.exit(1)
    except requests.exceptions.HTTPError as exc:
        print(f"Error: HTTP {exc.response.status_code} from Hyperliquid API.")
        sys.exit(1)
    except requests.exceptions.RequestException as exc:
        print(f"Error: network request failed ({exc}).")
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
