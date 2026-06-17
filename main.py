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

from config import DATA, MARKET, PAPER_TRADING, TEST_MODE, VERSION
from data_logger import append_prices_to_csv
from equity_logger import append_equity_snapshot
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
import session
import session_report
import signals
from signals import apply_signal_thresholds


def run_analyzer(session_id: str, mode: str) -> None:
    """
    Fetch prices, track history, and print analysis in a loop.

    Runs until the user presses Ctrl+C.
    """
    print("Starting Hyperliquid market analyzer (Ctrl+C to stop)...")
    print(f"Version: {VERSION}")
    print(f"Session ID: {session_id}")
    print(f"TEST_MODE: {'ON' if TEST_MODE else 'OFF'} ({mode})")
    print(
        "Signals use 1-minute rolling change "
        f"(±{signals.SIGNAL_UP_PCT}% / ±{signals.SIGNAL_STRONG_UP_PCT}% thresholds)"
    )
    print("Paper trading: fake $1,000 USDC balance, $100 per simulated trade")
    print(
        f"Costs: {PAPER_TRADING.trading_fee_pct:.2f}% fee and "
        f"{PAPER_TRADING.slippage_pct:.2f}% slippage per side"
    )
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

    start_prices: dict[str, float] = {}
    price_history: PriceHistory = {}
    paper_trader = PaperTrader(session_id=session_id, mode=mode)

    while True:
        try:
            mids = fetch_all_mids()
            update_start_prices(start_prices, mids)

            now = time.time()
            timestamp = current_timestamp()

            snapshot = build_snapshot(
                mids,
                start_prices,
                price_history,
                TEST_MODE,
                paper_trader=None,
                timestamp=timestamp,
                now=now,
                session_id=session_id,
            )

            current_prices = current_prices_from_rows(snapshot.market_rows)
            paper_trader.update_session_risk(current_prices)

            for coin, signal, current_price in tradable_signals_from_rows(
                snapshot.market_rows
            ):
                paper_trader.process_signal(
                    coin, signal, current_price, timestamp, now
                )

            snapshot = build_snapshot(
                mids,
                start_prices,
                price_history,
                TEST_MODE,
                paper_trader=paper_trader,
                timestamp=timestamp,
                now=now,
                market_rows=snapshot.market_rows,
                session_id=session_id,
            )

            print_snapshot(snapshot)
            save_recommendations_json(snapshot)
            append_equity_snapshot(
                timestamp=timestamp,
                session_id=session_id,
                mode=mode,
                paper_trader=paper_trader,
                current_prices=current_prices,
            )
            append_prices_to_csv(mids, timestamp)
        except Exception as exc:
            print(f"\n[{current_timestamp()}]")
            handle_request_error(exc)

        time.sleep(MARKET.polling_interval_seconds)


def main() -> None:
    apply_signal_thresholds(TEST_MODE)

    session_id = session.generate_session_id()
    mode = session.mode_label(TEST_MODE)
    config_snapshot = session.session_config_snapshot(test_mode=TEST_MODE)
    session.save_session_config(session_id, config_snapshot, DATA.session_configs_dir)

    try:
        run_analyzer(session_id, mode)
    except KeyboardInterrupt:
        print("\nAnalyzer stopped.")
        try:
            report_path = session_report.save_session_report(session_id)
            print(f"Session report saved to {report_path}")
        except ValueError:
            pass
        sys.exit(0)


if __name__ == "__main__":
    main()
