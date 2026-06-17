"""Read-only Review Mode helpers for the Streamlit dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from capital_readiness import assess_capital_readiness, format_assessment_report
from config import MARKET, PAPER_TRADING, TEST_MODE, VERSION
from review_data import load_all_experiments, latest_experiment, run_test_summary
from review_documents import (
    generate_advisor_questions,
    generate_known_limitations,
    generate_system_architecture,
)
import review_documents as docs
from strategy_profiles import list_profile_ids


def render_review_mode_banner() -> None:
    st.warning(
        "**Review-only paper system — no wallet, signing, leverage, or real-order capability.** "
        "Export review packages from the command line only: `python3 review_package.py`"
    )


def render_review_mode(
    *,
    load_experiment,
    list_experiments_fn,
) -> None:
    """Render the read-only Review Mode page."""
    render_review_mode_banner()

    st.subheader("Review Mode")
    st.caption(f"Version `{VERSION}` — supervised technical and risk review")

    col1, col2, col3 = st.columns(3)
    col1.metric("Version", VERSION)
    col2.metric("Safety", "Paper-only")
    col3.metric("Mode", "TEST_MODE" if TEST_MODE else "production")

    st.markdown("#### Safety Status")
    st.success(
        "No wallet, signing, leverage, deposits, withdrawals, or exchange execution. "
        "This dashboard cannot place real trades or export review packages."
    )

    st.markdown("#### Architecture")
    with st.expander("System pipeline", expanded=False):
        st.markdown(generate_system_architecture())

    st.markdown("#### Health & Tests")
    test_summary = run_test_summary(quick=True)
    health_cols = st.columns(3)
    health_cols[0].metric("Tests collected", test_summary.total)
    health_cols[1].metric(
        "Health check",
        "Run `python3 review_check.py`",
    )
    health_cols[2].metric(
        "Review package",
        "Run `python3 review_package.py`",
    )
    st.caption(
        "For full verification run `python3 review_check.py` and `python3 -m pytest` "
        "from the command line."
    )

    st.markdown("#### Strategy Profiles")
    for profile_id in list_profile_ids():
        st.markdown(f"- `{profile_id}`")
    st.caption(f"Monitored coins: {', '.join(MARKET.watchlist)}")

    st.markdown("#### Latest Experiment Comparison")
    experiments = load_all_experiments()
    latest = latest_experiment(experiments)
    if latest is None:
        st.info("No experiments available. Run `python3 experiment_runner.py` first.")
    else:
        st.caption(
            f"Experiment `{latest.get('experiment_id', 'unknown')}` | "
            f"{latest.get('timestamp_count', 0):,} timestamps"
        )
        rows = []
        for profile_id, payload in latest.get("profile_results", {}).items():
            oos = payload.get("out_of_sample", {})
            rows.append(
                {
                    "Profile": profile_id,
                    "Evidence": payload.get("evidence_assessment"),
                    "OOS Return (%)": oos.get("net_return_pct"),
                    "OOS Trades": oos.get("completed_trades"),
                    "OOS Max DD (%)": oos.get("maximum_drawdown_pct"),
                    "Warnings": len(payload.get("overfitting_warnings") or []),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        st.markdown("#### Overfitting Warnings")
        for profile_id, payload in latest.get("profile_results", {}).items():
            warnings = payload.get("overfitting_warnings") or []
            if warnings:
                st.markdown(f"**{profile_id}:**")
                for warning in warnings:
                    st.warning(warning)

    st.markdown("#### Session Analytics")
    st.caption(
        f"Paper balance: ${PAPER_TRADING.starting_balance_usdc:,.2f} | "
        f"Trade size: ${PAPER_TRADING.trade_size_usdc:,.2f} | "
        f"Session limit: {PAPER_TRADING.max_successful_trades_per_session} | "
        f"Max exposure: ${PAPER_TRADING.max_total_open_exposure_usdc:,.2f}"
    )

    st.markdown("#### Risk Limits")
    risk_rows = [
        ("Starting balance", f"${PAPER_TRADING.starting_balance_usdc:,.2f}"),
        ("Trade size", f"${PAPER_TRADING.trade_size_usdc:,.2f}"),
        ("Max allocation/coin", f"${PAPER_TRADING.max_allocation_per_coin_usdc:,.2f}"),
        ("Max entries/coin", str(PAPER_TRADING.max_entries_per_coin)),
        ("Buy cooldown", f"{PAPER_TRADING.min_seconds_between_buys}s"),
        ("Max total exposure", f"${PAPER_TRADING.max_total_open_exposure_usdc:,.2f}"),
        ("Session trade limit", str(PAPER_TRADING.max_successful_trades_per_session)),
        ("Equity pause", f"${PAPER_TRADING.min_equity_before_pause_usdc:,.2f}"),
        ("Trading fee", f"{PAPER_TRADING.trading_fee_pct}%"),
        ("Slippage", f"{PAPER_TRADING.slippage_pct}%"),
    ]
    st.dataframe(
        pd.DataFrame(risk_rows, columns=["Limit", "Value"]),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Capital Readiness Assessment")
    assessment = assess_capital_readiness()
    st.info(f"**{assessment.label}** — descriptive only, does not authorize trading.")
    with st.expander("Full assessment details"):
        st.text(format_assessment_report(assessment))

    st.markdown("#### Known Limitations")
    with st.expander("Known limitations"):
        st.markdown(generate_known_limitations())

    st.markdown("#### Adviser Questions")
    with st.expander("Questions for qualified adviser"):
        st.markdown(generate_advisor_questions())

    st.markdown("#### Evidence Summary")
    with st.expander("Evidence summary"):
        st.markdown(docs.generate_evidence_summary(latest))

    st.caption(
        "Historical and paper results do not guarantee future performance. "
        "This release cannot place real trades."
    )
