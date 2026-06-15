"""
Price history tracking and percentage change calculations.
"""

# Rolling window lengths for short-term change calculations
WINDOW_1M_SECONDS = 60
WINDOW_5M_SECONDS = 300

# How often the main loop refreshes prices (used when trimming history)
REFRESH_INTERVAL_SECONDS = 10

# Coins we want to track from the allMids response
WATCHLIST = ("BTC", "ETH", "SOL")

# Each coin's in-memory history is a list of (unix_time, price) tuples
PriceHistory = dict[str, list[tuple[float, float]]]


def price_to_float(price_str: str) -> float:
    """Convert a price string from the API (e.g. '43250.5') into a float."""
    return float(price_str)


def percent_change(old_price: float, current_price: float) -> float:
    """Calculate percentage change from an older price to the current price."""
    if old_price == 0:
        return 0.0
    return ((current_price - old_price) / old_price) * 100


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
