"""
Read-only Streamlit dashboard for the paper trading bot.

Displays price history and paper trade results from local CSV files.
No wallet, signing, or real exchange access.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path("data")
PRICE_LOG_CSV = DATA_DIR / "price_log.csv"
PAPER_TRADES_CSV = DATA_DIR / "paper_trades.csv"
COINS = ("BTC", "ETH", "SOL")


@st.cache_data(ttl=30)
def load_price_log() -> pd.DataFrame:
    """Load price history from CSV."""
    if not PRICE_LOG_CSV.exists():
        return pd.DataFrame(columns=["timestamp", "coin", "price"])

    df = pd.read_csv(PRICE_LOG_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.dropna(subset=["price"])


@st.cache_data(ttl=30)
def load_paper_trades() -> pd.DataFrame | None:
    """Load paper trades from CSV, or None if the file does not exist."""
    if not PAPER_TRADES_CSV.exists():
        return None

    df = pd.read_csv(PAPER_TRADES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ("price", "quantity", "trade_value", "cash_balance", "realized_pnl"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


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
        cols = st.columns(len(COINS))
        for col, coin in zip(cols, COINS):
            with col:
                if coin in latest.index:
                    price = latest.loc[coin, "price"]
                    ts = latest.loc[coin, "timestamp"]
                    st.metric(coin, f"${price:,.2f}", help=f"Last update: {ts}")
                else:
                    st.metric(coin, "—")

        st.subheader("Recent Price History")
        recent = prices.sort_values("timestamp", ascending=False).head(30)
        st.dataframe(recent, use_container_width=True, hide_index=True)

        st.subheader("Price Charts")
        chart_cols = st.columns(len(COINS))
        for col, coin in zip(chart_cols, COINS):
            coin_prices = prices[prices["coin"] == coin].sort_values("timestamp")
            with col:
                st.markdown(f"**{coin}**")
                if coin_prices.empty:
                    st.info(f"No {coin} data yet.")
                else:
                    chart_data = coin_prices.set_index("timestamp")[["price"]]
                    st.line_chart(chart_data, use_container_width=True)

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
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
