"""
Read-only Streamlit dashboard for the paper trading bot.

Displays decision-focused recommendations, session state, price history,
paper trade results, and backtest summaries from local CSV/JSON files.
No wallet, signing, or real exchange access.
"""

import json

import pandas as pd
import streamlit as st

from config import DATA, MARKET, PAPER_TRADING


@st.cache_data(ttl=30)
def load_price_log() -> pd.DataFrame:
    """Load price history from CSV."""
    if not DATA.price_log_csv.exists():
        return pd.DataFrame(columns=["timestamp", "coin", "price"])

    df = pd.read_csv(DATA.price_log_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.dropna(subset=["price"])


@st.cache_data(ttl=30)
def load_paper_trades() -> pd.DataFrame | None:
    """Load paper trades from CSV, or None if the file does not exist."""
    if not DATA.paper_trades_csv.exists():
        return None

    df = pd.read_csv(DATA.paper_trades_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ("price", "quantity", "trade_value", "cash_balance", "realized_pnl"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=30)
def load_backtest_results() -> dict | None:
    """Load backtest summary JSON, or None if the file does not exist."""
    if not DATA.backtest_results_json.exists():
        return None

    with DATA.backtest_results_json.open(encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(ttl=30)
def load_recommendations() -> dict | None:
    """Load the latest read-only recommendations snapshot, if available."""
    if not DATA.recommendations_json.exists():
        return None

    with DATA.recommendations_json.open(encoding="utf-8") as handle:
        return json.load(handle)


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

    cols = st.columns(7)
    cols[0].metric("Total Equity", _format_money(paper.get("total_equity")))
    cols[1].metric("Realized PnL", _format_money(paper.get("realized_pnl")))
    cols[2].metric("Unrealized PnL", _format_money(paper.get("unrealized_pnl")))
    cols[3].metric("Open Exposure", _format_money(paper.get("open_exposure")))
    cols[4].metric(
        "Open Positions",
        decision_summary.get("open_positions", "—"),
    )
    cols[5].metric("Session Trades", f"{session_trades} / {max_session_trades}")
    cols[6].metric("Paused", paused_label)

    st.caption(f"Last updated: {snapshot.get('timestamp', 'unknown')}")


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


def _render_paper_trades_section() -> None:
    st.subheader("Paper Trades")

    trades = load_paper_trades()
    if trades is None:
        st.info(
            "No paper trades yet. Run the paper trader to create "
            "`data/paper_trades.csv`."
        )
        return

    trade_count = len(trades)
    total_pnl = trades["realized_pnl"].sum()

    metric_cols = st.columns(2)
    metric_cols[0].metric("Number of Trades", trade_count)
    metric_cols[1].metric("Total Realized PnL", f"${total_pnl:,.2f}")

    with st.expander("Full paper trades table"):
        st.dataframe(
            trades.sort_values("timestamp", ascending=False),
            width="stretch",
            hide_index=True,
        )


def _format_profit_factor(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


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
        "and simulated trades."
    )

    snapshot = load_recommendations()

    _render_session_summary(snapshot)
    st.divider()
    _render_current_recommendations(snapshot)
    st.divider()

    prices = load_price_log()
    _render_price_section(prices)

    st.divider()
    _render_paper_trades_section()

    st.divider()
    _render_backtest_results()


if __name__ == "__main__":
    main()
