"""
Read-only Streamlit dashboard for the paper trading bot.

Displays decision-focused recommendations, session state, price history,
paper trade results, and backtest summaries from local CSV/JSON files.
No wallet, signing, or real exchange access.
"""

from datetime import date

import pandas as pd
import streamlit as st

from config import DATA, MARKET, PAPER_TRADING, VERSION
import dashboard_data
import dashboard_review


@st.cache_data(ttl=30)
def load_price_log() -> pd.DataFrame:
    return dashboard_data.load_price_log()


@st.cache_data(ttl=30)
def load_paper_trades() -> pd.DataFrame | None:
    return dashboard_data.load_paper_trades()


@st.cache_data(ttl=30)
def load_equity_log() -> pd.DataFrame | None:
    return dashboard_data.load_equity_log()


@st.cache_data(ttl=30)
def load_backtest_results() -> dict | None:
    return dashboard_data.load_backtest_results()


@st.cache_data(ttl=30)
def list_experiments() -> list[str]:
    return dashboard_data.list_experiments()


@st.cache_data(ttl=30)
def load_experiment(experiment_id: str) -> dict | None:
    return dashboard_data.load_experiment(experiment_id)


@st.cache_data(ttl=30)
def load_recommendations() -> dict | None:
    return dashboard_data.load_recommendations()


def _format_money(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:,.2f}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}%"


def _format_change(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def _format_profit_factor(value: float | None) -> str:
    if value is None:
        return "—"
    if value == float("inf"):
        return "inf"
    return f"{value:.2f}"


def _recommendation_color(recommendation: str) -> str:
    colors = {
        "PAPER BUY": "#2ecc71",
        "PAPER SELL": "#3498db",
        "HOLD": "#95a5a6",
        "BLOCKED": "#e74c3c",
    }
    return colors.get(recommendation, "#95a5a6")


def _supporting_reason(decision: dict) -> str:
    reasons = decision.get("supporting_reasons") or []
    if not reasons:
        return "—"
    return reasons[0]


def _mode_badge(snapshot: dict | None) -> str:
    if snapshot is None:
        return "Unknown mode"
    if snapshot.get("test_mode"):
        return "TEST_MODE"
    return "production"


def _render_session_summary(snapshot: dict | None) -> None:
    st.subheader("Session Summary")

    if snapshot is None:
        st.info(
            "No session snapshot yet. Run `python3 main.py` to generate "
            "`data/recommendations.json`."
        )
        return

    paper = snapshot.get("paper_trader")
    decision_summary = snapshot.get("decision_summary") or {}

    if paper is None:
        st.info("Session snapshot has no paper trading summary.")
        return

    mode_label = paper.get("mode") or _mode_badge(snapshot)
    st.caption(f"Active mode: **{mode_label}** | Version: `{VERSION}`")

    paused_label = "YES"
    if paper.get("paused"):
        reason = paper.get("pause_reason")
        if reason:
            paused_label = f"YES ({reason})"
    else:
        paused_label = "NO"

    session_trades = paper.get("session_trades", 0)
    max_session_trades = paper.get(
        "max_session_trades",
        PAPER_TRADING.max_successful_trades_per_session,
    )

    cols = st.columns(8)
    cols[0].metric("Total Equity", _format_money(paper.get("total_equity")))
    cols[1].metric("Realized PnL", _format_money(paper.get("realized_pnl")))
    cols[2].metric("Unrealized PnL", _format_money(paper.get("unrealized_pnl")))
    cols[3].metric("Fees Paid", _format_money(paper.get("fees_paid")))
    cols[4].metric("Open Exposure", _format_money(paper.get("open_exposure")))
    cols[5].metric(
        "Open Positions",
        decision_summary.get("open_positions", "—"),
    )
    cols[6].metric("Session Trades", f"{session_trades} / {max_session_trades}")
    cols[7].metric("Paused", paused_label)

    open_positions = decision_summary.get("open_positions", 0)
    if session_trades >= max_session_trades and open_positions:
        st.warning(
            "**SESSION TRADE LIMIT REACHED**\n\n"
            "New exposure is blocked.\n\n"
            "Risk-reducing exits remain allowed.\n\n"
            f"Open positions: {open_positions}"
        )

    session_id = paper.get("session_id") or snapshot.get("session_id", "")
    st.caption(
        f"Session ID: `{session_id or 'unknown'}` | "
        f"Last updated: {snapshot.get('timestamp', 'unknown')}"
    )


def _render_recommendation_card(decision: dict) -> None:
    coin = decision["coin"]
    recommendation = decision["recommendation"]
    color = _recommendation_color(recommendation)

    st.markdown(
        f"### {coin}  \n"
        f"<span style='color:{color}; font-size: 1.35em; font-weight: 700;'>"
        f"{recommendation}</span>",
        unsafe_allow_html=True,
    )

    st.markdown(f"**Signal:** {decision['signal']} ({decision['signal_strength']})")
    st.markdown(f"**Risk:** {decision['risk_status']}")
    st.markdown(f"**Reason:** {decision['primary_reason']}")
    st.markdown(f"**Supporting:** {_supporting_reason(decision)}")
    st.markdown(
        f"**Confidence:** {decision['confidence']} "
        f"(signal strength only — not a profit probability)"
    )

    detail_cols = st.columns(4)
    detail_cols[0].metric("Price", _format_money(decision.get("current_price")))
    detail_cols[1].metric("1-Min Change", _format_change(decision.get("change_1m")))
    detail_cols[2].metric("5-Min Change", _format_change(decision.get("change_5m")))
    detail_cols[3].metric(
        "Position Value",
        _format_money(decision.get("current_position_value")),
    )


def _render_current_recommendations(snapshot: dict | None) -> None:
    st.subheader("Current Recommendations")
    st.caption(
        "Read-only simulated guidance from the decision engine. "
        "No wallet, signing, or real orders."
    )

    if snapshot is None:
        st.info(
            "No recommendations yet. Run `python3 main.py` to generate "
            "`data/recommendations.json`."
        )
        return

    decisions = snapshot.get("decisions", [])
    decisions_by_coin = {item["coin"]: item for item in decisions}

    card_cols = st.columns(len(MARKET.watchlist))
    for col, coin in zip(card_cols, MARKET.watchlist):
        with col:
            with st.container(border=True):
                decision = decisions_by_coin.get(coin)
                if decision is None:
                    st.info(f"No recommendation for {coin}")
                else:
                    _render_recommendation_card(decision)


def _render_filters(
    trades: pd.DataFrame | None,
    equity: pd.DataFrame | None,
    snapshot: dict | None,
) -> tuple[str | None, date | None, date | None, str | None]:
    normalized = (
        dashboard_data.normalize_paper_trades(trades)
        if trades is not None
        else pd.DataFrame()
    )
    sessions = dashboard_data.available_sessions(normalized, equity)

    current_session = ""
    if snapshot is not None:
        paper = snapshot.get("paper_trader") or {}
        current_session = paper.get("session_id") or snapshot.get("session_id", "")

    filter_cols = st.columns(3)
    session_options = ["All sessions"] + sessions
    default_session_index = 0
    if current_session and current_session in sessions:
        default_session_index = session_options.index(current_session)

    selected_session = filter_cols[0].selectbox(
        "Session filter",
        session_options,
        index=default_session_index,
    )

    min_date = None
    max_date = None
    if not normalized.empty:
        min_date = normalized["timestamp"].min().date()
        max_date = normalized["timestamp"].max().date()
    elif equity is not None and not equity.empty:
        min_date = equity["timestamp"].min().date()
        max_date = equity["timestamp"].max().date()

    start_date = filter_cols[1].date_input(
        "Start date",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
    )
    end_date = filter_cols[2].date_input(
        "End date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )

    session_filter = None if selected_session == "All sessions" else selected_session
    return session_filter, start_date, end_date, current_session or None


def _render_performance_section(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    session_filter: str | None,
    start_date: date | None,
    end_date: date | None,
    current_session: str | None,
) -> None:
    st.subheader("Forward-Testing Analytics")

    filtered_trades = dashboard_data.filter_trades_by_date(
        dashboard_data.filter_trades_by_session(trades, session_filter),
        start_date,
        end_date,
    )
    filtered_equity = dashboard_data.filter_equity_by_date(
        dashboard_data.filter_equity_by_session(equity, session_filter),
        start_date,
        end_date,
    )

    current_trades = filtered_trades
    current_equity = filtered_equity
    if current_session:
        current_trades = dashboard_data.filter_trades_by_session(trades, current_session)
        current_equity = dashboard_data.filter_equity_by_session(equity, current_session)

    all_time_stats = dashboard_data.compute_trade_stats(filtered_trades)
    current_stats = dashboard_data.compute_trade_stats(current_trades)
    all_time_equity = dashboard_data.equity_summary(filtered_equity)
    current_equity_summary = dashboard_data.equity_summary(current_equity)

    tab_current, tab_all_time = st.tabs(["Current session", "All-time (filtered)"])

    for tab, stats, equity_stats, label in (
        (tab_current, current_stats, current_equity_summary, "Current session"),
        (tab_all_time, all_time_stats, all_time_equity, "Filtered results"),
    ):
        with tab:
            st.markdown(f"**{label}**")
            metric_cols = st.columns(6)
            metric_cols[0].metric("Ending Equity", _format_money(equity_stats["ending_equity"]))
            metric_cols[1].metric("Total Return", _format_pct(equity_stats["total_return_pct"]))
            metric_cols[2].metric("Realized PnL", _format_money(stats["realized_pnl"]))
            metric_cols[3].metric("Fees Paid", _format_money(stats["total_fees_paid"]))
            metric_cols[4].metric("Completed Trades", str(stats["completed_trades"]))
            metric_cols[5].metric("Win Rate", _format_pct(stats["win_rate_pct"]))

            detail_cols = st.columns(4)
            detail_cols[0].metric("Average Win", _format_money(stats["average_win"]))
            detail_cols[1].metric("Average Loss", _format_money(stats["average_loss"]))
            detail_cols[2].metric(
                "Profit Factor",
                _format_profit_factor(dashboard_data.compute_profit_factor(
                    dashboard_data.filter_trades_by_session(trades, session_filter)
                    if label == "Filtered results"
                    else current_trades
                )),
            )
            detail_cols[3].metric("Buys / Sells", f"{stats['total_buys']} / {stats['total_sells']}")

    chart_equity = filtered_equity if session_filter else current_equity
    if chart_equity.empty and not filtered_equity.empty:
        chart_equity = filtered_equity

    if not chart_equity.empty:
        ordered_equity = chart_equity.sort_values("timestamp")
        st.markdown("**Equity curve**")
        st.line_chart(
            ordered_equity.set_index("timestamp")[["total_equity"]],
            width="stretch",
        )

        drawdown = dashboard_data.compute_drawdown_series(chart_equity)
        st.markdown("**Drawdown over time**")
        st.line_chart(
            drawdown.set_index("timestamp")[["drawdown_pct"]],
            width="stretch",
        )

    if not filtered_trades.empty:
        cumulative = dashboard_data.compute_cumulative_realized_pnl(filtered_trades)
        st.markdown("**Cumulative realized PnL**")
        st.line_chart(
            cumulative.set_index("timestamp")[["cumulative_realized_pnl"]],
            width="stretch",
        )

        coin_stats = dashboard_data.pnl_by_coin(filtered_trades)
        if coin_stats:
            coin_rows = [
                {
                    "Coin": coin,
                    "Realized PnL": values["realized_pnl"],
                    "Completed trades": values["completed_trades"],
                    "Win rate (%)": values["win_rate_pct"],
                    "Fees paid": values["fees_paid"],
                }
                for coin, values in coin_stats.items()
            ]
            st.markdown("**PnL by coin**")
            st.dataframe(pd.DataFrame(coin_rows), width="stretch", hide_index=True)

    with st.expander("Raw equity log"):
        if chart_equity.empty:
            st.info("No equity rows for the selected filters.")
        else:
            st.dataframe(
                chart_equity.sort_values("timestamp", ascending=False),
                width="stretch",
                hide_index=True,
            )

    with st.expander("Full paper trades table"):
        if filtered_trades.empty:
            st.info("No paper trades for the selected filters.")
        else:
            st.dataframe(
                filtered_trades.sort_values("timestamp", ascending=False),
                width="stretch",
                hide_index=True,
            )


def _render_price_section(prices: pd.DataFrame) -> None:
    if prices.empty:
        st.warning(
            "No price data yet. Run the market scanner first to populate "
            "`data/price_log.csv`."
        )
        return

    st.subheader("Price Charts")
    chart_tabs = st.tabs(list(MARKET.watchlist))

    for tab, coin in zip(chart_tabs, MARKET.watchlist):
        coin_prices = prices[prices["coin"] == coin].sort_values("timestamp")
        with tab:
            if coin_prices.empty:
                st.info(f"No {coin} data yet.")
            else:
                chart_data = coin_prices.set_index("timestamp")[["price"]]
                st.line_chart(chart_data, width="stretch")

    with st.expander("Raw price history"):
        recent = prices.sort_values("timestamp", ascending=False)
        st.dataframe(recent, width="stretch", hide_index=True)


def _mode_summary_rows(block: dict) -> list[tuple[str, str]]:
    total_trades = block["total_buys"] + block["total_sells"]
    return [
        ("Ending equity (after forced close)", _format_money(block["ending_equity"])),
        ("Total return", _format_pct(block["total_return_pct"])),
        ("Realized PnL", _format_money(block["realized_pnl"])),
        ("Unrealized PnL", _format_money(block["unrealized_pnl"])),
        ("Total PnL", _format_money(block["total_pnl"])),
        ("Total trades", str(total_trades)),
        ("Buys", str(block["total_buys"])),
        ("Sells", str(block["total_sells"])),
        ("Completed trades", str(block["completed_trades"])),
        ("Win rate", _format_pct(block["win_rate_pct"])),
        ("Average win", _format_money(block["average_win"])),
        ("Average loss", _format_money(block["average_loss"])),
        ("Profit factor", _format_profit_factor(block["profit_factor"])),
        ("Maximum drawdown", _format_pct(block["maximum_drawdown_pct"])),
        ("Fees paid", _format_money(block["total_fees_paid"])),
        ("Final cash", _format_money(block["final_cash"])),
        ("Final open exposure", _format_money(block["final_open_exposure"])),
    ]


def _coin_results_table(mode_label: str, block: dict) -> pd.DataFrame:
    rows = []
    for coin, stats in block["results_by_coin"].items():
        rows.append(
            {
                "Mode": mode_label,
                "Coin": coin,
                "Buys": stats["total_buys"],
                "Sells": stats["total_sells"],
                "Completed trades": stats["completed_trades"],
                "Realized PnL": stats["realized_pnl"],
                "Win rate (%)": stats["win_rate_pct"],
                "Average win": stats["average_win"],
                "Average loss": stats["average_loss"],
            }
        )
    return pd.DataFrame(rows)


def _render_experiment_analysis() -> None:
    st.subheader("Experiment Analysis")
    st.caption(
        "Historical and paper results do not guarantee future performance."
    )

    experiment_ids = list_experiments()
    if not experiment_ids:
        st.info(
            "No experiments yet. Run "
            "`python3 experiment_runner.py --profiles contrarian_v1_test "
            "contrarian_v1_production hold_baseline` "
            "to generate results under `data/experiments/`."
        )
        return

    selected_id = st.selectbox(
        "Experiment",
        experiment_ids,
        index=len(experiment_ids) - 1,
    )
    experiment = load_experiment(selected_id)
    if experiment is None:
        st.warning("Selected experiment file could not be loaded.")
        return

    st.caption(
        f"Experiment `{selected_id}` | Created: {experiment.get('created_at', 'unknown')} | "
        f"Timestamps: {experiment.get('timestamp_count', '—')} | "
        f"Hash: `{str(experiment.get('input_data_hash', ''))[:12]}…`"
    )

    split = experiment.get("split", {})
    dev_pct = experiment.get("configuration", {}).get("development_pct", 0.7)
    st.markdown(
        f"**Split:** development {split.get('development_timestamps', '—')} timestamps "
        f"({dev_pct * 100:.0f}%) | "
        f"out-of-sample {split.get('out_of_sample_timestamps', '—')} timestamps"
    )

    profile_results = experiment.get("profile_results", {})
    comparison_rows = []
    for profile_id, payload in profile_results.items():
        dev = payload.get("development", {})
        oos = payload.get("out_of_sample", {})
        comparison_rows.append(
            {
                "Profile": profile_id,
                "Evidence": payload.get("evidence_assessment", "—"),
                "Dev return (%)": dev.get("net_return_pct"),
                "OOS return (%)": oos.get("net_return_pct"),
                "Combined return (%)": payload.get("combined", {}).get("net_return_pct"),
                "OOS completed trades": oos.get("completed_trades"),
                "OOS max drawdown (%)": oos.get("maximum_drawdown_pct"),
                "OOS profit factor": oos.get("profit_factor"),
                "OOS fees": oos.get("fees_paid"),
                "OOS slippage": oos.get("slippage_cost"),
            }
        )

    st.markdown("**Profile comparison**")
    st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)

    selected_profile = st.selectbox(
        "Profile detail",
        list(profile_results.keys()),
    )
    detail = profile_results.get(selected_profile, {})

    warn_cols = st.columns(2)
    with warn_cols[0]:
        st.markdown("**Evidence assessment**")
        st.info(detail.get("evidence_assessment", "—"))
    with warn_cols[1]:
        warnings = detail.get("overfitting_warnings") or []
        st.markdown("**Overfitting warnings**")
        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("No overfitting warnings recorded.")

    period_tabs = st.tabs(["Development", "Out-of-sample", "Combined"])
    for tab, key in zip(
        period_tabs,
        ("development", "out_of_sample", "combined"),
    ):
        with tab:
            metrics = detail.get(key, {})
            metric_cols = st.columns(6)
            metric_cols[0].metric("Net return", _format_pct(metrics.get("net_return_pct")))
            metric_cols[1].metric("Realized PnL", _format_money(metrics.get("realized_pnl")))
            metric_cols[2].metric("Fees", _format_money(metrics.get("fees_paid")))
            metric_cols[3].metric("Slippage", _format_money(metrics.get("slippage_cost")))
            metric_cols[4].metric("Completed trades", str(metrics.get("completed_trades", 0)))
            metric_cols[5].metric("Win rate", _format_pct(metrics.get("win_rate_pct")))

            detail_cols = st.columns(4)
            detail_cols[0].metric("Max drawdown", _format_pct(metrics.get("maximum_drawdown_pct")))
            detail_cols[1].metric(
                "Profit factor",
                _format_profit_factor(metrics.get("profit_factor")),
            )
            detail_cols[2].metric(
                "Exposure time",
                _format_pct(metrics.get("exposure_time_pct")),
            )
            detail_cols[3].metric(
                "Return / trade",
                _format_money(metrics.get("return_per_completed_trade")),
            )

            equity_curve = metrics.get("equity_curve") or []
            if equity_curve:
                st.markdown("**Equity curve**")
                st.line_chart(pd.Series(equity_curve, name="equity"))

                peak = pd.Series(equity_curve).cummax()
                drawdown = ((peak - pd.Series(equity_curve)) / peak.replace(0, pd.NA)) * 100
                drawdown = drawdown.fillna(0.0)
                st.markdown("**Drawdown curve**")
                st.line_chart(drawdown.rename("drawdown_pct"))

            by_coin = metrics.get("results_by_coin") or {}
            if by_coin:
                coin_rows = [
                    {
                        "Coin": coin,
                        "Realized PnL": stats.get("realized_pnl"),
                        "Completed trades": stats.get("completed_trades"),
                        "Win rate (%)": stats.get("win_rate_pct"),
                        "Fees": stats.get("fees_paid"),
                        "Slippage": stats.get("slippage_cost"),
                    }
                    for coin, stats in by_coin.items()
                ]
                st.markdown("**Per-coin results**")
                st.dataframe(pd.DataFrame(coin_rows), width="stretch", hide_index=True)

    with st.expander("Full profile configuration"):
        st.json(detail.get("profile", {}))

    with st.expander("Trade-level results (combined period)"):
        trades = detail.get("combined", {}).get("trade_log") or []
        if trades:
            st.dataframe(pd.DataFrame(trades), width="stretch", hide_index=True)
        else:
            st.info("No trades in the combined period.")


def _render_backtest_results() -> None:
    st.subheader("Backtest Results")
    st.caption(
        "Backtest results are simulated and do not guarantee future performance."
    )

    results = load_backtest_results()
    if results is None:
        st.info("Run python3 backtester.py to generate backtest results.")
        return

    test_block = results["test_mode"]["after_forced_close"]
    production_block = results["production_mode"]["after_forced_close"]

    max_drawdown = max(
        test_block["maximum_drawdown_pct"],
        production_block["maximum_drawdown_pct"],
    )

    summary_cols = st.columns(5)
    summary_cols[0].metric(
        "Test-Mode Return",
        _format_pct(test_block["total_return_pct"]),
    )
    summary_cols[1].metric(
        "Production Return",
        _format_pct(production_block["total_return_pct"]),
    )
    summary_cols[2].metric("Maximum Drawdown", _format_pct(max_drawdown))
    summary_cols[3].metric(
        "Fees Paid (Test)",
        _format_money(test_block["total_fees_paid"]),
        help="Test-mode simulation fees. See details for production.",
    )
    summary_cols[4].metric(
        "Completed Trades (Test)",
        str(test_block["completed_trades"]),
        help="Test-mode completed round trips. See details for production.",
    )

    with st.expander("Detailed backtest tables"):
        test_rows = _mode_summary_rows(test_block)
        production_rows = _mode_summary_rows(production_block)

        comparison_rows = [
            {
                "Metric": label,
                "TEST_MODE": test_value,
                "Production": production_value,
            }
            for (label, test_value), (_, production_value) in zip(
                test_rows, production_rows
            )
        ]

        st.markdown("**Mode comparison**")
        st.dataframe(
            pd.DataFrame(comparison_rows),
            width="stretch",
            hide_index=True,
        )

        st.markdown("**Per-coin results**")
        coin_tables = pd.concat(
            [
                _coin_results_table("TEST_MODE", test_block),
                _coin_results_table("Production", production_block),
            ],
            ignore_index=True,
        )
        st.dataframe(coin_tables, width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Paper Trading Dashboard", layout="wide")
    st.title("Paper Trading Dashboard")
    st.caption(
        "Decision-focused, read-only view of recommendations, session state, "
        "forward-testing analytics, and simulated trades."
    )

    tab_dashboard, tab_review = st.tabs(["Dashboard", "Review Mode"])

    with tab_review:
        dashboard_review.render_review_mode(
            load_experiment=load_experiment,
            list_experiments_fn=list_experiments,
        )

    with tab_dashboard:
        _render_dashboard()


def _render_dashboard() -> None:
    snapshot = load_recommendations()

    _render_session_summary(snapshot)
    st.divider()
    _render_current_recommendations(snapshot)
    st.divider()

    raw_trades = load_paper_trades()
    equity = load_equity_log()
    normalized_trades = (
        dashboard_data.normalize_paper_trades(raw_trades)
        if raw_trades is not None
        else pd.DataFrame()
    )
    equity_df = equity if equity is not None else pd.DataFrame()

    session_filter, start_date, end_date, current_session = _render_filters(
        raw_trades,
        equity,
        snapshot,
    )

    if normalized_trades.empty and equity_df.empty:
        st.info(
            "No forward-testing data yet. Run `python3 main.py` to populate "
            "`data/paper_trades.csv` and `data/equity_log.csv`."
        )
    else:
        _render_performance_section(
            normalized_trades,
            equity_df,
            session_filter,
            start_date,
            end_date,
            current_session,
        )

    st.divider()
    prices = load_price_log()
    _render_price_section(prices)

    st.divider()
    _render_experiment_analysis()

    st.divider()
    _render_backtest_results()


if __name__ == "__main__":
    main()
