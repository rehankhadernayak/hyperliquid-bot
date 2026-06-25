"""Read-only Review Mode helpers for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from capital_readiness import (
    ReadinessRules,
    assess_capital_readiness,
    format_assessment_report,
)
from config import MARKET, PAPER_TRADING, TEST_MODE, VERSION
import dashboard_data
import dashboard_live
import dashboard_market_archive
from review_data import (
    PytestRunSummary,
    load_all_experiments,
    latest_experiment,
    run_test_summary,
)
from review_documents import (
    generate_advisor_questions,
    generate_known_limitations,
    generate_system_architecture,
)
import review_documents as docs
from strategy_profiles import list_profile_ids

_REVIEW_TEST_SUMMARY_KEY = "review_mode_test_summary"


def _cached_test_summary() -> PytestRunSummary | None:
    summary = st.session_state.get(_REVIEW_TEST_SUMMARY_KEY)
    return summary if isinstance(summary, PytestRunSummary) else None


def _placeholder_test_summary() -> PytestRunSummary:
    return PytestRunSummary(
        passed=False,
        total=0,
        failed=0,
        output="Review checks not run yet.",
    )


def _render_health_and_tests_section() -> PytestRunSummary | None:
    """Lazy pytest metadata — only collected when the user clicks the button."""
    st.markdown("#### Latest Tests Summary")

    test_summary = _cached_test_summary()
    if test_summary is None:
        st.info("Review checks not run yet.")
        st.caption(
            "Click **Run Review Checks** to collect pytest metadata "
            "(collect-only, no test execution)."
        )
    else:
        st.success(f"**{test_summary.total}** tests collected (collect-only metadata).")
        if test_summary.output:
            with st.expander("Collect output", expanded=False):
                st.text(test_summary.output[:2000])

    if st.button("Run Review Checks", key="review_mode_run_checks"):
        try:
            summary = run_test_summary(quick=True)
            st.session_state[_REVIEW_TEST_SUMMARY_KEY] = summary
            st.rerun()
        except Exception as error:
            st.error("Review checks failed to run")
            st.exception(error)

    health_cols = st.columns(3)
    health_cols[0].metric(
        "Tests collected",
        test_summary.total if test_summary else "—",
    )
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
    return test_summary


def _render_data_pipeline_readiness() -> None:
    try:
        archive_index = dashboard_market_archive.build_market_archive_index()
    except Exception as error:
        st.warning("Data pipeline readiness unavailable.")
        st.caption(str(error))
        st.info("No local market archive index found yet. Recorder output will appear here when available.")
        return

    session_id = None
    try:
        status = dashboard_data.load_recorder_status()
        session_id = (status or {}).get("recorder_session_id")
    except Exception:
        session_id = None

    try:
        replay_count = len(dashboard_data.load_replay_reports())
        feature_count = len(dashboard_data.load_feature_research_reports())
        backtest_count = len(dashboard_data.load_event_backtest_reports())
    except Exception:
        replay_count = feature_count = backtest_count = 0

    readiness = dashboard_live.build_research_readiness(
        archive_index,
        session_id=session_id,
        replay_count=replay_count,
        feature_count=feature_count,
        backtest_count=backtest_count,
    )

    labels = readiness.get("labels") or {}
    if not labels:
        st.info("Research readiness labels unavailable — local pipeline metadata not found yet.")
        return

    for label, ready in labels.items():
        if ready:
            st.success(label)
        else:
            st.info(f"{label} — not ready")

    st.caption(readiness.get("explanation", dashboard_live.DATA_QUALITY_EXPLANATION))
    next_cmd = readiness.get("next_recommended_command")
    if next_cmd:
        st.code(next_cmd, language="bash")


def _render_strategy_readiness() -> None:
    try:
        reports = dashboard_data.load_event_backtest_reports()
    except Exception as error:
        st.warning("Strategy readiness data unavailable.")
        st.caption(str(error))
        st.info("No event backtest reports found.")
        return

    if not reports:
        st.info("No event backtest reports found.")
        return

    latest = reports[0]
    scols = st.columns(4)
    scols[0].metric("Latest strategy", latest.get("strategy_name", "—"))
    scols[1].metric("Profile", (latest.get("profile_name") or "—")[:20])
    scols[2].metric("Evidence", latest.get("evidence_label", "—"))
    scols[3].metric("Completed trades", latest.get("completed_trades", 0))

    st.markdown("**Strategy profiles**")
    for profile_id in list_profile_ids():
        st.markdown(f"- `{profile_id}`")
    st.caption(f"Monitored coins: {', '.join(MARKET.watchlist)}")


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
    st.session_state["review_mode_loaded"] = True
    st.info("Review Mode loaded.")
    try:
        render_review_mode_banner()
    except Exception as error:
        st.warning("Review Mode banner unavailable.")
        st.caption(str(error))

    try:
        _render_review_mode_content(
            load_experiment=load_experiment,
            list_experiments_fn=list_experiments_fn,
        )
    except Exception as error:
        st.error("Review Mode failed to load")
        st.exception(error)
        st.info("Review Mode placeholders remain available below while sections recover.")

    st.markdown("#### Review Mode Placeholders")
    st.caption(
        "Read-only review summary. Export full packages with "
        "`python3 review_package.py` from the command line."
    )
    st.write(f"Version: `{VERSION}` | Safety: paper-only | Experiments on disk: display-only")


def _render_review_mode_content(
    *,
    load_experiment,
    list_experiments_fn,
) -> None:
    st.subheader("Review Mode")

    st.markdown("#### Project Version / Status")
    try:
        col1, col2, col3 = st.columns(3)
        col1.metric("Version", VERSION)
        col2.metric("Safety", "Paper-only")
        col3.metric("Mode", "TEST_MODE" if TEST_MODE else "production")
        st.caption(f"Version `{VERSION}` — supervised technical and risk review")
    except Exception as error:
        st.warning("Project version section unavailable.")
        st.caption(str(error))

    st.markdown("#### Safety Audit")
    try:
        st.success(
            "No wallet, signing, leverage, deposits, withdrawals, or exchange execution. "
            "This dashboard cannot place real trades or export review packages."
        )
    except Exception as error:
        st.warning("Safety audit section unavailable.")
        st.caption(str(error))

    st.markdown("#### Architecture Summary")
    try:
        with st.expander("System pipeline", expanded=False):
            st.markdown(generate_system_architecture())
    except Exception as error:
        st.warning("Architecture summary unavailable.")
        st.caption(str(error))

    try:
        _render_health_and_tests_section()
    except Exception as error:
        st.warning("Latest tests summary unavailable.")
        st.caption(str(error))

    st.markdown("#### Data Pipeline Readiness")
    try:
        _render_data_pipeline_readiness()
    except Exception as error:
        st.warning("Data pipeline readiness unavailable.")
        st.caption(str(error))

    st.markdown("#### Strategy Readiness")
    try:
        _render_strategy_readiness()
    except Exception as error:
        st.warning("Strategy readiness unavailable.")
        st.caption(str(error))

    st.markdown("#### Latest Experiment Comparison")
    experiments: list[dict] = []
    latest = None
    try:
        experiments = load_all_experiments()
        latest = latest_experiment(experiments)
    except Exception as error:
        st.warning("Experiment review data unavailable.")
        st.caption(str(error))

    if latest is None:
        st.info("No experiments available. Run `python3 experiment_runner.py` first.")
    else:
        try:
            st.caption(
                f"Experiment `{latest.get('experiment_id', 'unknown')}` | "
                f"{latest.get('timestamp_count', 0):,} timestamps"
            )
            rows = []
            for profile_id, payload in (latest.get("profile_results") or {}).items():
                oos = payload.get("out_of_sample") or {}
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
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                st.info("Latest experiment loaded but contains no profile rows.")

            with st.expander("Overfitting warnings", expanded=False):
                for profile_id, payload in (latest.get("profile_results") or {}).items():
                    warnings = payload.get("overfitting_warnings") or []
                    if warnings:
                        st.markdown(f"**{profile_id}:**")
                        for warning in warnings:
                            st.warning(warning)
        except Exception as error:
            st.warning("Experiment comparison table unavailable.")
            st.caption(str(error))

    st.markdown("#### Known Limitations")
    try:
        with st.expander("Known limitations", expanded=False):
            st.markdown(generate_known_limitations())
    except Exception as error:
        st.warning("Known limitations section unavailable.")
        st.caption(str(error))

    st.markdown("#### Capital Readiness Assessment")
    try:
        test_summary = _cached_test_summary()
        if test_summary is None:
            st.warning("Capital readiness uses placeholder data until review checks are run.")
            assessment = assess_capital_readiness(
                test_summary=_placeholder_test_summary(),
                experiments=experiments,
                rules=ReadinessRules(require_tests_pass=False),
            )
        else:
            try:
                assessment = assess_capital_readiness(
                    test_summary=test_summary,
                    experiments=experiments,
                )
            except Exception as error:
                st.warning("Capital readiness assessment unavailable.")
                st.caption(str(error))
                assessment = assess_capital_readiness(
                    test_summary=_placeholder_test_summary(),
                    experiments=experiments,
                    rules=ReadinessRules(require_tests_pass=False),
                )
        st.info(f"**{assessment.label}** — descriptive only, does not authorize trading.")
        with st.expander("Full assessment details", expanded=False):
            st.text(format_assessment_report(assessment))
    except Exception as error:
        st.warning("Capital readiness section unavailable.")
        st.caption(str(error))

    try:
        with st.expander("Session analytics & risk limits", expanded=False):
            st.caption(
                f"Paper balance: ${PAPER_TRADING.starting_balance_usdc:,.2f} | "
                f"Trade size: ${PAPER_TRADING.trade_size_usdc:,.2f} | "
                f"Session limit: {PAPER_TRADING.max_successful_trades_per_session} | "
                f"Max exposure: ${PAPER_TRADING.max_total_open_exposure_usdc:,.2f}"
            )
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
    except Exception as error:
        st.warning("Session analytics section unavailable.")
        st.caption(str(error))

    try:
        with st.expander("Questions for qualified adviser", expanded=False):
            st.markdown(generate_advisor_questions())
    except Exception as error:
        st.warning("Adviser questions section unavailable.")
        st.caption(str(error))

    try:
        with st.expander("Evidence summary", expanded=False):
            st.markdown(docs.generate_evidence_summary(latest))
    except Exception as error:
        st.warning("Evidence summary unavailable.")
        st.caption(str(error))

    st.caption(
        "Historical and paper results do not guarantee future performance. "
        "This release cannot place real trades."
    )
