"""
Read-only Hyperliquid market analyzer.

Fetches public mid prices from Hyperliquid's Info API every 10 seconds,
tracks each coin's first observed price, keeps recent price history in memory,
and shows rolling 1-minute and 5-minute percentage changes with signals.
Also runs a read-only paper trading simulation (no real orders).

No wallet, signing, or live trading involved.
"""

import sys
import time

from data_logger import append_prices_to_csv
from hyperliquid_client import fetch_all_mids, handle_request_error
from market_analyzer import (
    REFRESH_INTERVAL_SECONDS,
    PriceHistory,
    update_start_prices,
)
from market_snapshot import (
    build_snapshot,
    current_prices_from_rows,
    current_timestamp,
    print_snapshot,
    tradable_signals_from_rows,
)
from paper_trader import (
    MAX_SUCCESSFUL_TRADES_PER_SESSION,
    MAX_TOTAL_OPEN_EXPOSURE_USDC,
    MIN_EQUITY_BEFORE_PAUSE_USDC,
    MIN_SECONDS_BETWEEN_BUYS,
    PaperTrader,
)
import signals
from signals import apply_signal_thresholds

# Set True to use lower signal thresholds and trigger paper trades more often.
TEST_MODE = True


def run_analyzer() -> None:
    """
    Fetch prices, track history, and print analysis in a loop.

    Runs until the user presses Ctrl+C.
    """
    apply_signal_thresholds(TEST_MODE)

    print("Starting Hyperliquid market analyzer (Ctrl+C to stop)...")
    print(f"TEST_MODE: {'ON' if TEST_MODE else 'OFF'}")
    print(
        "Signals use 1-minute rolling change "
        f"(±{signals.SIGNAL_UP_PCT}% / ±{signals.SIGNAL_STRONG_UP_PCT}% thresholds)"
    )
    print("Paper trading: fake $1,000 USDC balance, $100 per simulated trade")
    print(
        "Per-coin limits: max $300/coin, max 3 entries/coin, "
        f"{MIN_SECONDS_BETWEEN_BUYS}s buy cooldown per coin"
    )
    print(
        "Session limits: max "
        f"${MAX_TOTAL_OPEN_EXPOSURE_USDC:,.0f} total exposure, max "
        f"{MAX_SUCCESSFUL_TRADES_PER_SESSION} trades/session, "
        f"pause below ${MIN_EQUITY_BEFORE_PAUSE_USDC:,.0f} equity"
    )

    # First observed price for each coin during this run
    start_prices: dict[str, float] = {}
    # Recent (timestamp, price) history for rolling window calculations
    price_history: PriceHistory = {}
    paper_trader = PaperTrader()

    while True:
        try:
            mids = fetch_all_mids()
            update_start_prices(start_prices, mids)

            now = time.time()
            timestamp = current_timestamp()

            # Build market snapshot, then run paper trading from snapshot signals
            snapshot = build_snapshot(
                mids,
                start_prices,
                price_history,
                TEST_MODE,
                paper_trader=None,
                timestamp=timestamp,
                now=now,
            )

            current_prices = current_prices_from_rows(snapshot.market_rows)
            paper_trader.update_session_risk(current_prices)

            for coin, signal, current_price in tradable_signals_from_rows(
                snapshot.market_rows
            ):
                paper_trader.process_signal(
                    coin, signal, current_price, timestamp, now
                )

            # Refresh paper trader fields on the same market rows for display
            snapshot = build_snapshot(
                mids,
                start_prices,
                price_history,
                TEST_MODE,
                paper_trader=paper_trader,
                timestamp=timestamp,
                now=now,
                market_rows=snapshot.market_rows,
            )

            print_snapshot(snapshot)
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
