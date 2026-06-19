"""Read-only data loading and analytics helpers for the dashboard and reports."""

from __future__ import annotations

from datetime import date

import pandas as pd

from config import DATA, PAPER_TRADING

LEGACY_PAPER_TRADE_COLUMNS = (
    "timestamp",
    "coin",
    "action",
    "price",
    "quantity",
    "trade_value",
    "cash_balance",
    "realized_pnl",
)

PAPER_TRADE_COLUMNS = (
    "timestamp",
    "session_id",
    "mode",
    "coin",
    "action",
    "signal",
    "price",
    "quantity",
    "trade_value",
    "entry_count",
    "fee",
    "slippage",
    "trade_pnl",
    "cumulative_realized_pnl",
    "cash_balance",
    "realized_pnl",
)


def load_price_log() -> pd.DataFrame:
    """Load price history from CSV."""
    if not DATA.price_log_csv.exists():
        return pd.DataFrame(columns=["timestamp", "coin", "price"])

    df = pd.read_csv(DATA.price_log_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.dropna(subset=["price"])


def load_paper_trades() -> pd.DataFrame | None:
    """Load paper trades from CSV, or None if the file does not exist."""
    if not DATA.paper_trades_csv.exists():
        return None

    return pd.read_csv(DATA.paper_trades_csv)


def load_equity_log() -> pd.DataFrame | None:
    """Load equity snapshots from CSV, or None if the file does not exist."""
    if not DATA.equity_log_csv.exists():
        return None

    df = pd.read_csv(DATA.equity_log_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    numeric_cols = (
        "cash",
        "open_position_value",
        "realized_pnl",
        "unrealized_pnl",
        "fees_paid",
        "total_equity",
        "open_exposure",
        "session_trades",
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "paused" in df.columns:
        df["paused"] = df["paused"].astype(str).isin(("1", "True", "true", "yes"))

    return df


def load_backtest_results() -> dict | None:
    """Load backtest summary JSON, or None if the file does not exist."""
    if not DATA.backtest_results_json.exists():
        return None

    import json

    with DATA.backtest_results_json.open(encoding="utf-8") as handle:
        return json.load(handle)


def list_experiments() -> list[str]:
    """Return sorted experiment ids from data/experiments/."""
    if not DATA.experiments_dir.exists():
        return []

    return sorted(
        path.stem
        for path in DATA.experiments_dir.glob("*.json")
        if path.is_file()
    )


def load_experiment(experiment_id: str) -> dict | None:
    """Load one experiment JSON by id, or None if missing."""
    path = DATA.experiments_dir / f"{experiment_id}.json"
    if not path.exists():
        return None

    import json

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_recommendations() -> dict | None:
    """Load the latest recommendations snapshot, if available."""
    if not DATA.recommendations_json.exists():
        return None

    import json

    with DATA.recommendations_json.open(encoding="utf-8") as handle:
        return json.load(handle)


MARKET_STREAM_DIR = DATA.data_dir / "market_stream"


def load_recorder_status() -> dict | None:
    """Load market data recorder status, if available."""
    status_path = MARKET_STREAM_DIR / "recorder_status.json"
    if not status_path.exists():
        return None

    import json

    with status_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_recorder_manifests() -> list[dict]:
    """Load all recorder session manifests."""
    manifests_dir = MARKET_STREAM_DIR / "manifests"
    if not manifests_dir.exists():
        return []

    import json

    manifests = []
    for path in sorted(manifests_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            manifests.append(json.load(handle))
    return manifests


def load_quality_reports() -> list[dict]:
    """Load metadata for all quality reports, newest first."""
    from market_data.replay_manifest import list_quality_reports

    reports_dir = MARKET_STREAM_DIR / "quality_reports"
    return list_quality_reports(reports_dir)


def load_quality_report(session_id: str) -> dict | None:
    """Load one quality report JSON by session id."""
    path = MARKET_STREAM_DIR / "quality_reports" / f"{session_id}.json"
    if not path.exists():
        return None

    import json

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_replay_reports() -> list[dict]:
    """Load metadata for all replay reports, newest first."""
    from market_data.replay_report import list_replay_reports

    replays_dir = MARKET_STREAM_DIR / "replays"
    return list_replay_reports(replays_dir)


def load_replay_report(replay_session_id: str) -> dict | None:
    """Load one replay report JSON by replay session id."""
    path = MARKET_STREAM_DIR / "replays" / f"{replay_session_id}.json"
    if not path.exists():
        return None

    import json

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def market_stream_file_sizes() -> dict[str, int]:
    """Return byte sizes of raw and normalized market stream files."""
    sizes = {"raw_bytes": 0, "normalized_bytes": 0}
    stream_dir = MARKET_STREAM_DIR
    if not stream_dir.exists():
        return sizes

    raw_dir = stream_dir / "raw"
    norm_dir = stream_dir / "normalized"
    if raw_dir.exists():
        for f in raw_dir.rglob("*"):
            if f.is_file():
                sizes["raw_bytes"] += f.stat().st_size
    if norm_dir.exists():
        for f in norm_dir.rglob("*"):
            if f.is_file():
                sizes["normalized_bytes"] += f.stat().st_size
    return sizes


def normalize_paper_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize legacy and current paper trade CSV formats."""
    normalized = df.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)

    numeric_cols = (
        "price",
        "quantity",
        "trade_value",
        "cash_balance",
        "realized_pnl",
        "entry_count",
        "fee",
        "slippage",
        "trade_pnl",
        "cumulative_realized_pnl",
    )
    for col in numeric_cols:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    if "session_id" not in normalized.columns:
        normalized["session_id"] = ""
    else:
        normalized["session_id"] = normalized["session_id"].fillna("").astype(str)

    if "mode" not in normalized.columns:
        normalized["mode"] = ""
    else:
        normalized["mode"] = normalized["mode"].fillna("").astype(str)

    if "signal" not in normalized.columns:
        normalized["signal"] = ""
    else:
        normalized["signal"] = normalized["signal"].fillna("").astype(str)

    if "fee" not in normalized.columns:
        normalized["fee"] = 0.0
    if "slippage" not in normalized.columns:
        normalized["slippage"] = 0.0

    if "trade_pnl" not in normalized.columns:
        if "realized_pnl" in normalized.columns:
            normalized["trade_pnl"] = normalized["realized_pnl"].fillna(0.0)
        else:
            normalized["trade_pnl"] = 0.0

    if "cumulative_realized_pnl" not in normalized.columns:
        sell_mask = normalized["action"] == "SELL"
        normalized["cumulative_realized_pnl"] = normalized["trade_pnl"].where(sell_mask, 0.0).cumsum()

    if "realized_pnl" not in normalized.columns:
        normalized["realized_pnl"] = normalized["trade_pnl"]

    return normalized


def available_sessions(trades: pd.DataFrame | None, equity: pd.DataFrame | None) -> list[str]:
    """Return sorted session IDs found in trades or equity logs."""
    ids: set[str] = set()

    if trades is not None and "session_id" in trades.columns:
        ids.update(trades["session_id"].dropna().astype(str).tolist())

    if equity is not None and "session_id" in equity.columns:
        ids.update(equity["session_id"].dropna().astype(str).tolist())

    ids.discard("")
    return sorted(ids)


def filter_trades_by_session(trades: pd.DataFrame, session_id: str | None) -> pd.DataFrame:
    """Filter trades to one session, or return all trades when session_id is None."""
    if session_id is None or session_id == "All sessions":
        return trades.copy()
    return trades[trades["session_id"] == session_id].copy()


def filter_trades_by_date(
    trades: pd.DataFrame,
    start_date: date | None,
    end_date: date | None,
) -> pd.DataFrame:
    """Filter trades to an inclusive UTC date range."""
    filtered = trades.copy()
    if start_date is not None:
        filtered = filtered[filtered["timestamp"].dt.date >= start_date]
    if end_date is not None:
        filtered = filtered[filtered["timestamp"].dt.date <= end_date]
    return filtered


def filter_equity_by_session(equity: pd.DataFrame, session_id: str | None) -> pd.DataFrame:
    """Filter equity rows to one session, or return all rows when session_id is None."""
    if session_id is None or session_id == "All sessions":
        return equity.copy()
    return equity[equity["session_id"] == session_id].copy()


def filter_equity_by_date(
    equity: pd.DataFrame,
    start_date: date | None,
    end_date: date | None,
) -> pd.DataFrame:
    """Filter equity rows to an inclusive UTC date range."""
    filtered = equity.copy()
    if start_date is not None:
        filtered = filtered[filtered["timestamp"].dt.date >= start_date]
    if end_date is not None:
        filtered = filtered[filtered["timestamp"].dt.date <= end_date]
    return filtered


def compute_trade_stats(trades: pd.DataFrame) -> dict:
    """Compute completed-trade statistics from normalized paper trades."""
    if trades.empty:
        return {
            "total_buys": 0,
            "total_sells": 0,
            "completed_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "total_fees_paid": 0.0,
            "realized_pnl": 0.0,
        }

    buys = trades[trades["action"] == "BUY"]
    sells = trades[trades["action"] == "SELL"]
    winning = sells[sells["trade_pnl"] > 0]
    losing = sells[sells["trade_pnl"] < 0]
    closed = len(winning) + len(losing)

    realized_pnl = float(sells["trade_pnl"].sum())
    if "cumulative_realized_pnl" in trades.columns and not trades.empty:
        realized_pnl = float(trades["cumulative_realized_pnl"].iloc[-1])

    return {
        "total_buys": len(buys),
        "total_sells": len(sells),
        "completed_trades": closed,
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate_pct": round((len(winning) / closed) * 100, 2) if closed else 0.0,
        "average_win": round(float(winning["trade_pnl"].mean()), 2) if not winning.empty else 0.0,
        "average_loss": round(float(losing["trade_pnl"].mean()), 2) if not losing.empty else 0.0,
        "total_fees_paid": round(float(trades["fee"].sum()), 2),
        "realized_pnl": round(realized_pnl, 2),
    }


def compute_profit_factor(trades: pd.DataFrame) -> float | None:
    """Return profit factor from sell trade PnL."""
    if trades.empty:
        return None

    sells = trades[trades["action"] == "SELL"]
    if sells.empty:
        return None

    gross_wins = sells.loc[sells["trade_pnl"] > 0, "trade_pnl"].sum()
    gross_losses = abs(sells.loc[sells["trade_pnl"] < 0, "trade_pnl"].sum())

    if gross_losses == 0:
        return None if gross_wins == 0 else float("inf")
    return gross_wins / gross_losses


def compute_cumulative_realized_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """Return trades with a cumulative realized PnL column for charting."""
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "cumulative_realized_pnl"])

    chart = trades.sort_values("timestamp").copy()
    if "cumulative_realized_pnl" in chart.columns:
        return chart[["timestamp", "cumulative_realized_pnl"]]

    chart["cumulative_realized_pnl"] = chart["trade_pnl"].where(
        chart["action"] == "SELL", 0.0
    ).cumsum()
    return chart[["timestamp", "cumulative_realized_pnl"]]


def compute_drawdown_series(equity: pd.DataFrame) -> pd.DataFrame:
    """Return equity timestamps with drawdown percentage over time."""
    if equity.empty:
        return pd.DataFrame(columns=["timestamp", "drawdown_pct"])

    ordered = equity.sort_values("timestamp").copy()
    peak = ordered["total_equity"].cummax()
    drawdown_pct = ((peak - ordered["total_equity"]) / peak.replace(0, pd.NA)) * 100
    drawdown_pct = drawdown_pct.fillna(0.0)

    return pd.DataFrame(
        {
            "timestamp": ordered["timestamp"],
            "drawdown_pct": drawdown_pct,
        }
    )


def pnl_by_coin(trades: pd.DataFrame) -> dict[str, dict]:
    """Build per-coin trade stats from normalized paper trades."""
    if trades.empty or "coin" not in trades.columns:
        return {}

    results: dict[str, dict] = {}

    for coin in sorted(trades["coin"].dropna().unique()):
        coin_trades = trades[trades["coin"] == coin]
        sells = coin_trades[coin_trades["action"] == "SELL"]
        buys = coin_trades[coin_trades["action"] == "BUY"]
        winning = sells[sells["trade_pnl"] > 0]
        losing = sells[sells["trade_pnl"] < 0]
        closed = len(winning) + len(losing)

        results[str(coin)] = {
            "total_buys": len(buys),
            "total_sells": len(sells),
            "completed_trades": closed,
            "realized_pnl": round(float(sells["trade_pnl"].sum()), 2),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate_pct": round((len(winning) / closed) * 100, 2) if closed else 0.0,
            "average_win": round(float(winning["trade_pnl"].mean()), 2) if not winning.empty else 0.0,
            "average_loss": round(float(losing["trade_pnl"].mean()), 2) if not losing.empty else 0.0,
            "fees_paid": round(float(coin_trades["fee"].sum()), 2),
        }

    return results


def equity_summary(equity: pd.DataFrame) -> dict:
    """Summarize ending equity metrics from filtered equity rows."""
    if equity.empty:
        return {
            "starting_equity": PAPER_TRADING.starting_balance_usdc,
            "ending_equity": PAPER_TRADING.starting_balance_usdc,
            "total_return_pct": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "fees_paid": 0.0,
        }

    ordered = equity.sort_values("timestamp")
    starting = float(ordered["total_equity"].iloc[0])
    ending = float(ordered["total_equity"].iloc[-1])
    total_return_pct = ((ending - starting) / starting) * 100 if starting else 0.0

    return {
        "starting_equity": round(starting, 2),
        "ending_equity": round(ending, 2),
        "total_return_pct": round(total_return_pct, 2),
        "realized_pnl": round(float(ordered["realized_pnl"].iloc[-1]), 2),
        "unrealized_pnl": round(float(ordered["unrealized_pnl"].iloc[-1]), 2),
        "fees_paid": round(float(ordered["fees_paid"].iloc[-1]), 2),
    }
