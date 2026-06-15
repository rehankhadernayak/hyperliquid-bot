"""
CSV logging for price history.
"""

import csv
from pathlib import Path

from market_analyzer import WATCHLIST, price_to_float

# Where to save price history
DATA_DIR = Path("data")
PRICE_LOG_CSV = DATA_DIR / "price_log.csv"
CSV_COLUMNS = ("timestamp", "coin", "price")


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
