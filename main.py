"""
Read-only Hyperliquid market analyzer.

Fetches public mid prices from Hyperliquid's Info API every 10 seconds,
tracks each coin's first observed price, keeps recent price history in memory,
and shows rolling 1-minute and 5-minute percentage changes with signals.
Also runs a read-only paper trading simulation (no real orders).

No wallet, signing, or live trading involved.
"""

import argparse
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
from session_report import SessionShutdownContext
from signals import apply_signal_thresholds
from strategy_profiles import get_profile, list_profile_ids


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only Hyperliquid market analyzer with paper trading."
    )
    parser.add_argument(
        "--profile",
        choices=list_profile_ids(),
        help="Immutable strategy profile for this paper-trading session.",
    )
    return parser.parse_args(argv)


def _print_startup_banner(
    *,
    session_id: str,
    mode: str,
    config_snapshot: dict,
    profile_id: str | None,
) -> None:
    print("Starting Hyperliquid market analyzer (Ctrl+C to stop)...")
    print(f"Version: {VERSION}")
    print(f"Session ID: {session_id}")
    print(f"TEST_MODE: {'ON' if config_snapshot['test_mode'] else 'OFF'} ({mode})")

    if profile_id is not None:
        profile = get_profile(profile_id)
        limits = profile.trading_limits()
        print(f"Selected profile: {profile_id}")
        print(f"Starting equity: ${limits.starting_balance_usdc:,.2f}")
        print(f"Trade size: ${limits.trade_size_usdc:,.2f}")
        print(
            f"Per-coin allocation limit: ${limits.max_allocation_per_coin_usdc:,.2f}"
        )
        print(f"Total exposure limit: ${limits.max_total_open_exposure_usdc:,.2f}")
        print(f"Fee rate: {limits.trading_fee_pct:.2f}%")
        print(f"Slippage rate: {limits.slippage_pct:.2f}%")
        print("Leverage: disabled")
        print("Paper-only status: enabled (no wallets, signing, or real orders)")
    else:
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
        print("Leverage: disabled")
        print("Paper-only status: enabled (no wallets, signing, or real orders)")

    if PAPER_TRADING.force_close_on_shutdown:
        print("Shutdown policy: open positions will be force-closed on exit.")
    else:
        print(
            "Shutdown policy: open positions are preserved in the session report "
            "(no force-close)."
        )


def run_analyzer(
    session_id: str,
    mode: str,
    config_snapshot: dict,
    *,
    profile_id: str | None = None,
) -> None:
    """
    Fetch prices, track history, and print analysis in a loop.

    Runs until the user presses Ctrl+C.
    """
    _print_startup_banner(
        session_id=session_id,
        mode=mode,
        config_snapshot=config_snapshot,
        profile_id=profile_id,
    )

    profile = get_profile(profile_id) if profile_id is not None else None
    paper_trader = PaperTrader(
        session_id=session_id,
        mode=mode,
        profile=profile,
    )

    start_prices: dict[str, float] = {}
    price_history: PriceHistory = {}
    last_current_prices: dict[str, float] = {}

    try:
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
                    config_snapshot["test_mode"],
                    paper_trader=None,
                    timestamp=timestamp,
                    now=now,
                    session_id=session_id,
                )

                current_prices = current_prices_from_rows(snapshot.market_rows)
                if current_prices:
                    last_current_prices = current_prices

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
                    config_snapshot["test_mode"],
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
    except KeyboardInterrupt:
        shutdown_time = time.time()
        shutdown = SessionShutdownContext(
            shutdown_reason="keyboard_interrupt",
            started_at=paper_trader.started_at,
            ended_at=shutdown_time,
            current_prices=last_current_prices,
            strategy_profile=config_snapshot.get("strategy_profile", ""),
            configuration_hash=session.configuration_hash(config_snapshot),
            session_trade_limit=paper_trader.session_trade_limit(),
            blocked_proposals_total=paper_trader.blocked_proposals_total,
            blocked_proposals_by_reason=dict(paper_trader.blocked_proposals_by_reason),
            profile=config_snapshot.get("profile"),
        )

        positions_force_closed = False
        force_close_reason = None

        if PAPER_TRADING.force_close_on_shutdown and last_current_prices:
            open_before = sum(
                1 for position in paper_trader.positions.values() if position.is_open
            )
            if open_before > 0:
                force_close_reason = "force_close_on_shutdown enabled"
                paper_trader.force_close_all_positions(
                    last_current_prices,
                    current_timestamp(),
                )
                positions_force_closed = True
                append_equity_snapshot(
                    timestamp=current_timestamp(),
                    session_id=session_id,
                    mode=mode,
                    paper_trader=paper_trader,
                    current_prices=last_current_prices,
                )

        shutdown.positions_force_closed = positions_force_closed
        shutdown.force_close_reason = force_close_reason

        print("\nAnalyzer stopped.")
        if paper_trader.blocked_proposals_total > 0:
            paper_trader.maybe_print_blocked_summary(shutdown_time)

        open_positions = session_report.build_open_position_snapshots(
            paper_trader,
            last_current_prices,
        )
        if open_positions and not positions_force_closed:
            print("Open positions preserved at shutdown:")
            for position in open_positions:
                print(
                    f"  {position['coin']}: qty={position['quantity']}, "
                    f"entry=${position['average_entry']:,.2f}, "
                    f"price=${position['final_market_price']:,.2f}, "
                    f"value=${position['position_value']:,.2f}, "
                    f"unrealized={position['unrealized_pnl']:+,.2f}, "
                    f"entries={position['entry_count']}"
                )

        try:
            report_path = session_report.save_session_report(
                session_id,
                shutdown=shutdown,
                paper_trader=paper_trader,
            )
            print(f"Session report saved to {report_path}")
        except ValueError:
            pass
        sys.exit(0)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    profile_id = args.profile

    active_test_mode = session.resolve_test_mode(profile_id=profile_id)
    apply_signal_thresholds(active_test_mode)

    session_id = session.generate_session_id()
    mode = session.mode_label(active_test_mode)
    config_snapshot = session.session_config_snapshot(
        test_mode=active_test_mode,
        profile_id=profile_id,
    )
    session.save_session_config(session_id, config_snapshot, DATA.session_configs_dir)

    run_analyzer(session_id, mode, config_snapshot, profile_id=profile_id)


if __name__ == "__main__":
    main()
