"""
CSV logging for price history.
"""

import csv

from config import DATA, MARKET
from market_analyzer import price_to_float

CSV_COLUMNS = ("timestamp", "coin", "price")


def ensure_data_dir() -> None:
    """Create the data folder if it does not exist."""
    DATA.data_dir.mkdir(exist_ok=True)


def append_prices_to_csv(mids: dict[str, str], timestamp: str) -> None:
    """Append watchlist mid prices to data/price_log.csv."""
    ensure_data_dir()

    write_header = not DATA.price_log_csv.exists()

    with DATA.price_log_csv.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        if write_header:
            writer.writerow(CSV_COLUMNS)

        for coin in MARKET.watchlist:
            price_str = mids.get(coin)

            if price_str is None:
                continue

            try:
                price = price_to_float(price_str)
                writer.writerow([timestamp, coin, price])
            except ValueError:
                continue
