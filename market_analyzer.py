"""
Price history tracking and percentage change calculations.
"""

from config import MARKET

# Rolling window lengths for short-term change calculations
WINDOW_1M_SECONDS = 60
WINDOW_5M_SECONDS = 300

# Allow the nearest stored sample to be this many seconds after the ideal
# lookback timestamp. Matches one polling interval by default.
LOOKBACK_TOLERANCE_SECONDS = MARKET.polling_interval_seconds

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
    for coin in MARKET.watchlist:
        if coin in start_prices:
            continue

        price_str = mids.get(coin)
        if price_str is None:
            continue

        try:
            start_prices[coin] = price_to_float(price_str)
        except ValueError:
            continue


def history_retention_seconds() -> int:
    """How long in-memory price history is kept beyond the 5-minute window."""
    # Keep two polling intervals beyond the longest window so pruning does not
    # remove the reference sample needed for 5-minute lookbacks.
    return WINDOW_5M_SECONDS + (2 * MARKET.polling_interval_seconds)


def record_price_history(
    price_history: PriceHistory,
    coin: str,
    price: float,
    now: float,
) -> None:
    """Append the latest price and drop entries older than the retention window."""
    if coin not in price_history:
        price_history[coin] = []

    price_history[coin].append((now, price))

    cutoff = now - history_retention_seconds()
    price_history[coin] = [(t, p) for t, p in price_history[coin] if t >= cutoff]


def price_at_or_before(
    history: list[tuple[float, float]],
    target_time: float,
) -> tuple[float, float] | None:
    """
    Find the most recent stored (time, price) at or before target_time.

    History entries are stored in time order (oldest first).
    """
    reference: tuple[float, float] | None = None

    for entry_time, entry_price in history:
        if entry_time <= target_time:
            reference = (entry_time, entry_price)
        else:
            break

    return reference


def rolling_percent_change(
    history: list[tuple[float, float]],
    current_price: float,
    now: float,
    window_seconds: int,
    *,
    tolerance_seconds: int | None = None,
) -> float | None:
    """
    Calculate approximate percentage change over a rolling time window.

    Uses the nearest valid sample at or before ``now - window_seconds``.
    When no sample exists at or before the target, the oldest stored sample
    may be used if it falls within ``tolerance_seconds`` after the target.

    Returns None when insufficient history genuinely exists.
    """
    if not history:
        return None

    tolerance = (
        LOOKBACK_TOLERANCE_SECONDS
        if tolerance_seconds is None
        else tolerance_seconds
    )
    target_time = now - window_seconds

    reference = price_at_or_before(history, target_time)
    if reference is None:
        oldest_time, oldest_price = history[0]
        if oldest_time - target_time <= tolerance:
            reference = (oldest_time, oldest_price)
        else:
            return None

    reference_time, reference_price = reference
    if reference_time - target_time > tolerance:
        return None

    return percent_change(reference_price, current_price)
