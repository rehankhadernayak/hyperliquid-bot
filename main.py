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

from config import MARKET, PAPER_TRADING, TEST_MODE
from data_logger import append_prices_to_csv
from hyperliquid_client import fetch_all_mids, handle_request_error
from market_analyzer import PriceHistory, update_start_prices
from market_snapshot import (
    build_snapshot,
    current_prices_from_rows,
    current_timestamp,
    print_snapshot,
    save_recommendations_json,
    tradable_signals_from_rows,
)
from paper_trader import PaperTrader
import signals
from signals import apply_signal_thresholds


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
        f"{PAPER_TRADING.min_seconds_between_buys}s buy cooldown per coin"
    )
    print(
        "Session limits: max "
        f"${PAPER_TRADING.max_total_open_exposure_usdc:,.0f} total exposure, max "
        f"{PAPER_TRADING.max_successful_trades_per_session} trades/session, "
        f"pause below ${PAPER_TRADING.min_equity_before_pause_usdc:,.0f} equity"
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
            save_recommendations_json(snapshot)
            append_prices_to_csv(mids, timestamp)
        except Exception as exc:
            print(f"\n[{current_timestamp()}]")
            handle_request_error(exc)

        # Wait before the next update (Ctrl+C can interrupt this sleep)
        time.sleep(MARKET.polling_interval_seconds)


def main() -> None:
    try:
        run_analyzer()
    except KeyboardInterrupt:
        print("\nAnalyzer stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
