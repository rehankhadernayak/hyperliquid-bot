"""
Read-only Streamlit dashboard for the paper trading bot.

Displays price history, paper trade results, and backtest summaries from
local CSV/JSON files. No wallet, signing, or real exchange access.
"""

import json

import pandas as pd
import streamlit as st

from config import DATA, MARKET


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


def _format_money(value: float) -> str:
    return f"${value:,.2f}"


def _format_pct(value: float) -> str:
    return f"{value:.2f}%"


def _format_profit_factor(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _mode_summary_rows(block: dict) -> list[tuple[str, str]]:
    """Turn one backtest mode block into labeled rows for display."""
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
    """Build a per-coin table for one backtest mode."""
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

    test_rows = _mode_summary_rows(test_block)
    production_rows = _mode_summary_rows(production_block)

    comparison_rows = [
        {
            "Metric": label,
            "TEST_MODE": test_value,
            "Production": production_value,
        }
        for (label, test_value), (_, production_value) in zip(test_rows, production_rows)
    ]

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
    st.caption("Read-only view of logged prices and simulated trades.")

    prices = load_price_log()

    if prices.empty:
        st.warning(
            "No price data yet. Run the market scanner first to populate "
            "`data/price_log.csv`."
        )
    else:
        latest = (
            prices.sort_values("timestamp")
            .groupby("coin", as_index=False)
            .last()
            .set_index("coin")
        )

        st.subheader("Latest Prices")
        cols = st.columns(len(MARKET.watchlist))
        for col, coin in zip(cols, MARKET.watchlist):
            with col:
                if coin in latest.index:
                    price = latest.loc[coin, "price"]
                    ts = latest.loc[coin, "timestamp"]
                    st.metric(coin, f"${price:,.2f}", help=f"Last update: {ts}")
                else:
                    st.metric(coin, "—")

        st.subheader("Recent Price History")
        recent = prices.sort_values("timestamp", ascending=False).head(30)
        st.dataframe(recent, width="stretch", hide_index=True)

        st.subheader("Price Charts")
        chart_cols = st.columns(len(MARKET.watchlist))
        for col, coin in zip(chart_cols, MARKET.watchlist):
            coin_prices = prices[prices["coin"] == coin].sort_values("timestamp")
            with col:
                st.markdown(f"**{coin}**")
                if coin_prices.empty:
                    st.info(f"No {coin} data yet.")
                else:
                    chart_data = coin_prices.set_index("timestamp")[["price"]]
                    st.line_chart(chart_data, width="stretch")

    st.divider()
    st.subheader("Paper Trades")

    trades = load_paper_trades()
    if trades is None:
        st.info(
            "No paper trades yet. Run the paper trader to create "
            "`data/paper_trades.csv`."
        )
    else:
        trade_count = len(trades)
        total_pnl = trades["realized_pnl"].sum()

        metric_cols = st.columns(2)
        metric_cols[0].metric("Number of Trades", trade_count)
        metric_cols[1].metric("Total Realized PnL", f"${total_pnl:,.2f}")

        st.dataframe(
            trades.sort_values("timestamp", ascending=False),
            width="stretch",
            hide_index=True,
        )

    st.divider()
    _render_backtest_results()


if __name__ == "__main__":
    main()
