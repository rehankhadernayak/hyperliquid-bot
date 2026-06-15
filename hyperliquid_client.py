"""
Read-only Hyperliquid Info API client.

Fetches public market data only — no wallet, signing, or trading.
"""

import requests

# Hyperliquid's public read-only Info API (mainnet)
INFO_API_URL = "https://api.hyperliquid.xyz/info"


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
