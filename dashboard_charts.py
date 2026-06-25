"""Read-only chart data loaders and paper signal markers for the live dashboard.

Paper-only visualization — no wallets, signing, orders, or execution hooks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from config import MARKET

import dashboard_live

CHART_COINS = tuple(MARKET.watchlist)
RESOLUTION_SECONDS = {"1s": 1, "5s": 5, "10s": 10, "1m": 60}
WINDOW_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}

CHART_RANGE_OPTIONS: dict[str, dict[str, Any]] = {
    "Live 1m": {"seconds": 60, "mode": "live"},
    "Live 5m": {"seconds": 300, "mode": "live"},
    "Live 15m": {"seconds": 900, "mode": "live"},
    "1h": {"seconds": 3600, "mode": "historical"},
    "1d": {"seconds": 86_400, "mode": "historical"},
    "7d": {"seconds": 604_800, "mode": "historical"},
    "30d": {"seconds": 2_592_000, "mode": "historical"},
    "6m": {"seconds": 15_552_000, "mode": "historical"},
    "All local": {"seconds": None, "mode": "all_local"},
}

CHART_RANGE_KEYS = tuple(CHART_RANGE_OPTIONS.keys())
LIVE_MAX_DISPLAY_POINTS = 5_000
HISTORICAL_MAX_DISPLAY_POINTS = 3_000
DEFAULT_CHART_RANGE = "Live 5m"

MARKER_PAPER_BUY = "PAPER_BUY_SIGNAL"
MARKER_PAPER_SELL = "PAPER_SELL_SIGNAL"
MARKER_HOLD = "HOLD"
MARKER_BLOCKED = "BLOCKED"
MARKER_OPEN_POSITION = "OPEN_PAPER_POSITION"
MARKER_PLANNED_EXIT = "PLANNED_PAPER_EXIT"

MARKER_SYMBOLS = {
    MARKER_PAPER_BUY: "triangle-up",
    MARKER_PAPER_SELL: "triangle-down",
    MARKER_HOLD: "circle",
    MARKER_BLOCKED: "x",
    MARKER_OPEN_POSITION: "diamond",
}

MARKER_COLORS = {
    MARKER_PAPER_BUY: "#2ecc71",
    MARKER_PAPER_SELL: "#e74c3c",
    MARKER_HOLD: "#95a5a6",
    MARKER_BLOCKED: "#e67e22",
    MARKER_OPEN_POSITION: "#3498db",
}

NO_FIXED_EXIT_MESSAGE = "Exit: strategy signal or risk rule, no fixed target."


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_timestamp_ms(event: dict[str, Any]) -> int | None:
    ts = event.get("received_timestamp_ms") or event.get("exchange_timestamp_ms")
    if ts is None:
        return None
    try:
        return int(ts)
    except (TypeError, ValueError):
        return None


def _reference_time_s(events: list[dict[str, Any]], *, historical: bool) -> float:
    if events:
        return max(_event_timestamp_ms(event) or 0 for event in events) / 1000.0
    if historical:
        return 0.0
    import time

    return time.time()


def build_price_line_from_events(
    events: list[dict[str, Any]],
    coin: str,
    *,
    resolution_seconds: int = 1,
    window_seconds: int = 300,
    reference_time_s: float | None = None,
) -> dict[str, Any]:
    """Aggregate trade/BBO events into a resolution-bucketed price line for one coin."""
    if reference_time_s is None:
        reference_time_s = _reference_time_s(events, historical=False)

    cutoff_ms = int((reference_time_s - window_seconds) * 1000)
    buckets: dict[int, dict[str, Any]] = {}

    for event in events:
        if event.get("coin") != coin:
            continue
        ts_ms = _event_timestamp_ms(event)
        if ts_ms is None or ts_ms < cutoff_ms:
            continue

        bucket_key = int(ts_ms // 1000 // resolution_seconds) * resolution_seconds
        row = buckets.setdefault(
            bucket_key,
            {"time_ms": bucket_key * 1000, "price": None, "midpoint": None, "bid": None, "ask": None},
        )
        event_type = event.get("_event_type")

        if event_type == "bbo":
            bid = _safe_float(event.get("best_bid_price"))
            ask = _safe_float(event.get("best_ask_price"))
            row["bid"] = bid
            row["ask"] = ask
            if bid is not None and ask is not None:
                row["midpoint"] = (bid + ask) / 2.0
        elif event_type == "trade":
            price = _safe_float(event.get("price"))
            if price is not None:
                row["price"] = price
        elif event_type == "candle":
            close = _safe_float(event.get("close"))
            if close is not None:
                row["candle_close"] = close
        elif event_type in ("allMids", "mid"):
            mid = _safe_float(event.get("mid_price"))
            if mid is not None:
                row["allmids_price"] = mid

    series = sorted(buckets.values(), key=lambda item: item["time_ms"])
    for point in series:
        if point["price"] is None:
            point["price"] = (
                point.get("midpoint")
                or point.get("candle_close")
                or point.get("allmids_price")
            )

    latest = series[-1] if series else {}
    return {
        "coin": coin,
        "series": series,
        "latest_price": latest.get("price"),
        "best_bid": latest.get("bid"),
        "best_ask": latest.get("ask"),
        "spread_bps": _spread_bps(latest.get("bid"), latest.get("ask")),
        "has_data": bool(series),
    }


def _spread_bps(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None or bid <= 0:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return round(((ask - bid) / mid) * 10_000, 2)


def build_candle_series_from_events(
    events: list[dict[str, Any]],
    coin: str,
    *,
    window_seconds: int = 300,
    reference_time_s: float | None = None,
) -> dict[str, Any]:
    """Build OHLC candle rows from normalized candle parquet events."""
    if reference_time_s is None:
        reference_time_s = _reference_time_s(events, historical=False)

    cutoff_ms = int((reference_time_s - window_seconds) * 1000)
    candles: list[dict[str, Any]] = []

    for event in events:
        if event.get("coin") != coin:
            continue
        ts_ms = _event_timestamp_ms(event) or event.get("close_time_ms") or event.get("open_time_ms")
        if ts_ms is None or int(ts_ms) < cutoff_ms:
            continue
        candles.append(
            {
                "time_ms": int(ts_ms),
                "open": _safe_float(event.get("open")),
                "high": _safe_float(event.get("high")),
                "low": _safe_float(event.get("low")),
                "close": _safe_float(event.get("close")),
                "volume": _safe_float(event.get("volume")),
            }
        )

    candles.sort(key=lambda item: item["time_ms"])
    return {"coin": coin, "candles": candles, "has_candles": bool(candles)}


def parse_timeline_timestamp(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    text = str(timestamp).replace(" UTC", "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def _decision_metadata(recommendations: dict | None) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    if not recommendations:
        return meta
    paper = recommendations.get("paper_trader") or {}
    profile_name = paper.get("profile_id") or recommendations.get("strategy_profile") or "default"
    strategy_name = profile_name or "contrarian_v1"
    for decision in recommendations.get("decisions") or []:
        coin = decision.get("coin")
        if coin:
            meta[coin] = {
                "strategy_name": strategy_name,
                "profile_name": profile_name,
                "signal_strength": decision.get("signal_strength"),
            }
    return meta


def format_paper_signal_marker(
    entry: dict[str, Any],
    *,
    strategy_name: str = "—",
    profile_name: str = "—",
    signal_strength: str | None = None,
) -> dict[str, Any]:
    """Format one decision timeline row as a paper signal marker."""
    decision = entry.get("decision", MARKER_HOLD)
    marker_type = decision if decision in MARKER_SYMBOLS else MARKER_HOLD
    return {
        "marker_type": marker_type,
        "label": "paper signal marker",
        "symbol": MARKER_SYMBOLS.get(marker_type, "circle"),
        "color": MARKER_COLORS.get(marker_type, "#95a5a6"),
        "timestamp": entry.get("timestamp"),
        "timestamp_s": parse_timeline_timestamp(entry.get("timestamp")),
        "coin": entry.get("coin"),
        "decision": decision,
        "reason": entry.get("reason"),
        "price": _safe_float(entry.get("price")),
        "signal_strength": signal_strength or entry.get("signal_strength") or "—",
        "spread_bps": entry.get("spread_bps"),
        "bbo_imbalance": entry.get("bbo_imbalance"),
        "strategy_name": strategy_name,
        "profile_name": profile_name,
        "action_taken": entry.get("action_taken", "NONE"),
        "paper_only": True,
        "hover_text": _marker_hover_text(
            entry,
            strategy_name=strategy_name,
            profile_name=profile_name,
            signal_strength=signal_strength,
        ),
    }


def _marker_hover_text(
    entry: dict[str, Any],
    *,
    strategy_name: str,
    profile_name: str,
    signal_strength: str | None,
) -> str:
    lines = [
        "Paper signal marker",
        f"Time: {entry.get('timestamp', '—')}",
        f"Coin: {entry.get('coin', '—')}",
        f"Decision: {entry.get('decision', '—')}",
        f"Reason: {entry.get('reason', '—')}",
        f"Price: {entry.get('price', '—')}",
        f"Signal strength: {signal_strength or entry.get('signal_strength') or '—'}",
        f"Spread bps: {entry.get('spread_bps', '—')}",
        f"BBO imbalance: {entry.get('bbo_imbalance', '—')}",
        f"Strategy: {strategy_name}",
        f"Profile: {profile_name}",
        f"Action taken: {entry.get('action_taken', 'NONE')}",
    ]
    return "<br>".join(lines)


def build_paper_markers(
    timeline: list[dict[str, Any]],
    recommendations: dict | None,
    *,
    coin: str | None = None,
    window_start_s: float,
    reference_time_s: float,
) -> list[dict[str, Any]]:
    """Build paper signal markers filtered to one coin and time window."""
    meta_by_coin = _decision_metadata(recommendations)
    markers: list[dict[str, Any]] = []

    for entry in timeline:
        entry_coin = entry.get("coin")
        if coin and entry_coin != coin:
            continue
        ts_s = parse_timeline_timestamp(entry.get("timestamp"))
        if ts_s is None or ts_s < window_start_s or ts_s > reference_time_s:
            continue
        meta = meta_by_coin.get(entry_coin, {})
        markers.append(
            format_paper_signal_marker(
                entry,
                strategy_name=meta.get("strategy_name", "—"),
                profile_name=meta.get("profile_name", "—"),
                signal_strength=meta.get("signal_strength"),
            )
        )

    return markers


def _resolve_exit_config(recommendations: dict | None):
    if not recommendations:
        return None, "contrarian_v1", "default"
    paper = recommendations.get("paper_trader") or {}
    profile_id = paper.get("profile_id") or recommendations.get("strategy_profile") or ""
    strategy_name = profile_id or "contrarian_v1"
    profile_name = profile_id or "default"
    if not profile_id:
        return None, strategy_name, profile_name
    try:
        from strategy_profiles import get_profile

        profile = get_profile(profile_id)
        return profile.exit_config, profile.strategy_name, profile.profile_id
    except KeyError:
        return None, strategy_name, profile_name


def _planned_price_levels(entry_price: float, exit_config) -> tuple[float | None, float | None]:
    if exit_config is None or entry_price <= 0:
        return None, None
    tp_pct = exit_config.take_profit_pct
    sl_pct = exit_config.stop_loss_pct
    planned_tp = entry_price * (1 + tp_pct / 100.0) if tp_pct is not None else None
    planned_sl = entry_price * (1 - sl_pct / 100.0) if sl_pct is not None else None
    return planned_tp, planned_sl


def build_planned_exit_display(
    position: dict[str, Any],
    exit_config,
    *,
    strategy_name: str = "—",
    profile_name: str = "—",
    holding_time_seconds: float | None = None,
) -> dict[str, Any]:
    """Build read-only planned exit summary for one open paper position."""
    entry = _safe_float(position.get("average_entry")) or 0.0
    current = _safe_float(position.get("current_price"))
    unrealized = _safe_float(position.get("unrealized_pnl"))
    planned_tp, planned_sl = _planned_price_levels(entry, exit_config)

    has_fixed_tp_sl = planned_tp is not None or planned_sl is not None
    exit_summary = NO_FIXED_EXIT_MESSAGE
    signal_exit_note = None

    if exit_config is not None and getattr(exit_config, "signal_exit_enabled", False):
        signal_exit_note = "Signal-based exit enabled when no fixed TP/SL triggers."

    if has_fixed_tp_sl:
        parts = []
        if planned_tp is not None:
            parts.append(f"Planned paper TP at {planned_tp:.4f}")
        if planned_sl is not None:
            parts.append(f"Planned paper SL at {planned_sl:.4f}")
        exit_summary = "; ".join(parts)
    elif signal_exit_note:
        exit_summary = signal_exit_note

    holding_label = "—"
    if holding_time_seconds is not None:
        minutes, seconds = divmod(int(holding_time_seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            holding_label = f"{hours}h {minutes}m {seconds}s"
        elif minutes:
            holding_label = f"{minutes}m {seconds}s"
        else:
            holding_label = f"{seconds}s"

    return {
        "coin": position.get("coin"),
        "paper_entry_price": entry,
        "current_price": current,
        "planned_take_profit": planned_tp,
        "planned_stop_loss": planned_sl,
        "unrealized_paper_pnl": unrealized,
        "holding_time_seconds": holding_time_seconds,
        "holding_time_label": holding_label,
        "exit_summary": exit_summary,
        "has_fixed_tp_sl": has_fixed_tp_sl,
        "strategy_name": strategy_name,
        "profile_name": profile_name,
        "paper_only": True,
        "disclaimer": "Unrealized paper PnL is not guaranteed profit.",
    }


def build_open_position_details(
    recommendations: dict | None,
    paper_trades: Any | None = None,
    *,
    reference_time_s: float | None = None,
) -> list[dict[str, Any]]:
    """Build planned exit displays for all open paper positions."""
    if not recommendations:
        return []

    open_positions = recommendations.get("open_positions") or []
    if not open_positions:
        return []

    exit_config, strategy_name, profile_name = _resolve_exit_config(recommendations)
    holding_by_coin = _holding_times_by_coin(paper_trades, reference_time_s=reference_time_s)

    details: list[dict[str, Any]] = []
    for position in open_positions:
        coin = position.get("coin")
        detail = build_planned_exit_display(
            position,
            exit_config,
            strategy_name=strategy_name,
            profile_name=profile_name,
            holding_time_seconds=holding_by_coin.get(coin),
        )
        details.append(detail)
    return details


def _holding_times_by_coin(
    paper_trades: Any | None,
    *,
    reference_time_s: float | None,
) -> dict[str, float]:
    if paper_trades is None or getattr(paper_trades, "empty", True):
        return {}
    if reference_time_s is None:
        import time

        reference_time_s = time.time()

    holding: dict[str, float] = {}
    df = paper_trades.copy()
    if "timestamp" not in df.columns or "coin" not in df.columns:
        return holding

    side_col = "side" if "side" in df.columns else "action" if "action" in df.columns else None
    if side_col is None:
        return holding

    df = df.sort_values("timestamp")
    for coin in df["coin"].unique():
        coin_rows = df[df["coin"] == coin]
        buys = coin_rows[coin_rows[side_col].astype(str).str.upper().isin(["BUY", "B"])]
        if buys.empty:
            continue
        last_buy = buys.iloc[-1]["timestamp"]
        try:
            opened_s = last_buy.timestamp()
        except AttributeError:
            opened_s = parse_timeline_timestamp(str(last_buy))
        if opened_s is not None:
            holding[str(coin)] = max(0.0, reference_time_s - opened_s)
    return holding


def build_reference_lines(
    position_detail: dict[str, Any] | None,
    *,
    current_price: float | None,
) -> list[dict[str, Any]]:
    """Build horizontal reference lines for paper entry, TP, SL, and current price."""
    lines: list[dict[str, Any]] = []
    if position_detail:
        entry = position_detail.get("paper_entry_price")
        if entry is not None:
            lines.append({"label": "Paper entry", "price": entry, "color": "#3498db", "dash": "solid"})
        tp = position_detail.get("planned_take_profit")
        if tp is not None:
            lines.append({"label": "Planned paper TP", "price": tp, "color": "#2ecc71", "dash": "dash"})
        sl = position_detail.get("planned_stop_loss")
        if sl is not None:
            lines.append({"label": "Planned paper SL", "price": sl, "color": "#e74c3c", "dash": "dash"})
    if current_price is not None:
        lines.append({"label": "Current price", "price": current_price, "color": "#f39c12", "dash": "dot"})
    return lines


def build_open_position_markers(
    position_details: list[dict[str, Any]],
    *,
    coin: str,
) -> list[dict[str, Any]]:
    """Add diamond markers at paper entry for open positions."""
    markers: list[dict[str, Any]] = []
    for detail in position_details:
        if detail.get("coin") != coin:
            continue
        entry = detail.get("paper_entry_price")
        if entry is None:
            continue
        markers.append(
            {
                "marker_type": MARKER_OPEN_POSITION,
                "label": "open paper position",
                "symbol": MARKER_SYMBOLS[MARKER_OPEN_POSITION],
                "color": MARKER_COLORS[MARKER_OPEN_POSITION],
                "coin": coin,
                "price": entry,
                "decision": "OPEN_PAPER_POSITION",
                "reason": detail.get("exit_summary"),
                "hover_text": (
                    f"Open paper position<br>"
                    f"Coin: {coin}<br>"
                    f"Paper entry: {entry}<br>"
                    f"Current: {detail.get('current_price', '—')}<br>"
                    f"Unrealized paper PnL: {detail.get('unrealized_paper_pnl', '—')}<br>"
                    f"Holding: {detail.get('holding_time_label', '—')}<br>"
                    f"{detail.get('exit_summary', '')}"
                ),
                "paper_only": True,
            }
        )
    return markers


RAW_PREVIEW_MODE_LABEL = (
    "Raw preview mode: charts are built in-memory from raw WebSocket JSONL. "
    "Offline normalization is still required for research-grade replay/features."
)

LATEST_FALLBACK_WARNING = (
    "Selected range has no data. Showing latest locally available data instead."
)


def _resolve_chart_session_id(
    session_id: str | None,
    archive_index: dict[str, Any],
) -> str | None:
    if session_id:
        return session_id
    sessions = archive_index.get("sessions") or []
    if sessions:
        return str(sessions[0])
    return dashboard_live._resolve_raw_preview_session_id(None)


def _filter_events_for_range(
    events: list[dict[str, Any]],
    candle_events: list[dict[str, Any]],
    *,
    start_ms: int | None,
    end_ms: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if start_ms is None or end_ms is None:
        return events, candle_events

    filtered_events = [
        event
        for event in events
        if dashboard_live._event_in_range(event, start_ms=int(start_ms), end_ms=int(end_ms))
    ]
    filtered_candles = [
        event
        for event in candle_events
        if dashboard_live._event_in_range(event, start_ms=int(start_ms), end_ms=int(end_ms))
    ]
    return filtered_events, filtered_candles


def _events_have_coin_rows(
    events: list[dict[str, Any]],
    candle_events: list[dict[str, Any]],
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> bool:
    if start_ms is not None and end_ms is not None:
        events, candle_events = _filter_events_for_range(
            events,
            candle_events,
            start_ms=start_ms,
            end_ms=end_ms,
        )
    for event in events + candle_events:
        if event.get("coin") in CHART_COINS:
            return True
    return False


def _latest_event_timestamp_ms(
    events: list[dict[str, Any]],
    candle_events: list[dict[str, Any]],
) -> int | None:
    timestamps: list[int] = []
    for event in events + candle_events:
        ts = _event_timestamp_ms(event)
        if ts is not None:
            timestamps.append(ts)
    return max(timestamps) if timestamps else None


def _determine_chart_empty_reason(
    *,
    chart_data_source: str,
    normalized_available: bool,
    raw_preview_available: bool,
    archive_available: bool,
    range_bounds: dict[str, Any],
    chart_rows_loaded: int,
    fallback_used: bool,
) -> str | None:
    if chart_rows_loaded > 0:
        return None
    if chart_data_source == "error":
        return "Chart data load failed."
    if range_bounds.get("selected_range_empty"):
        return "Selected range is outside local coverage."
    if not normalized_available and not raw_preview_available and not archive_available:
        return "No local market data files found."
    if normalized_available and not raw_preview_available and not archive_available:
        return "Normalized files exist but no rows matched the selected range."
    if raw_preview_available:
        return "Raw preview loaded but no rows matched the selected coin/window."
    if archive_available:
        return "Archive files exist but no rows matched the selected range."
    if fallback_used:
        return "Latest-available fallback did not produce chart rows."
    return "Selected range has no local data."


def format_coin_chart_empty_message(coin: str, chart_data: dict[str, Any]) -> str:
    """Public alias for dashboard empty-state messaging."""
    return dashboard_live.format_coin_chart_empty_message(coin, chart_data)


def downsample_series(
    series: list[dict[str, Any]],
    *,
    max_points: int,
) -> tuple[list[dict[str, Any]], int]:
    """Downsample a time series for display without inventing prices."""
    if len(series) <= max_points or max_points <= 0:
        return series, len(series)
    step = max(1, len(series) // max_points)
    sampled = series[::step]
    if sampled and sampled[-1] is not series[-1]:
        sampled = sampled + [series[-1]]
    return sampled, len(sampled)


def downsample_candles(
    candles: list[dict[str, Any]],
    *,
    max_points: int,
) -> tuple[list[dict[str, Any]], int]:
    return downsample_series(candles, max_points=max_points)


def _max_display_points(*, is_live_range: bool) -> int:
    return LIVE_MAX_DISPLAY_POINTS if is_live_range else HISTORICAL_MAX_DISPLAY_POINTS


def _resolution_for_range(window_seconds: int, *, is_live_range: bool) -> int:
    if is_live_range:
        if window_seconds <= 60:
            return 1
        if window_seconds <= 300:
            return 1
        if window_seconds <= 900:
            return 5
        return 10
    if window_seconds <= 3600:
        return 5
    if window_seconds <= 86_400:
        return 60
    if window_seconds <= 604_800:
        return 300
    return 900


def build_chart_data(
    *,
    session_id: str | None,
    historical: bool,
    window_key: str = "5m",
    range_key: str | None = None,
    resolution_key: str | None = None,
    recommendations: dict | None = None,
    timeline: list[dict[str, Any]] | None = None,
    paper_trades: Any | None = None,
    archive_index: dict[str, Any] | None = None,
    show_latest_fallback: bool = True,
    recorder_running: bool | None = None,
) -> dict[str, Any]:
    """Load chart-ready data for all watchlist coins (read-only)."""
    import dashboard_market_archive as market_archive

    archive_index = archive_index or market_archive.build_market_archive_index()
    resolved_session_id = _resolve_chart_session_id(session_id, archive_index)
    selected_range = range_key or _window_key_to_range(window_key)
    prefer_latest_local = bool(historical or recorder_running is False)
    range_bounds = market_archive.resolve_chart_range_bounds(
        selected_range,
        archive_index=archive_index,
        prefer_latest_local_reference=prefer_latest_local,
    )

    is_live_range = range_bounds["is_live_range"]
    is_all_local = range_bounds["is_all_local"]
    window_seconds = range_bounds["window_seconds"] or WINDOW_SECONDS.get(window_key, 300)
    if window_seconds <= 0:
        window_seconds = WINDOW_SECONDS.get(window_key, 300)

    resolution_seconds = (
        RESOLUTION_SECONDS.get(resolution_key)
        if resolution_key
        else _resolution_for_range(window_seconds, is_live_range=is_live_range)
    )

    use_historical_loader = historical or (not is_live_range)
    start_ms = range_bounds.get("effective_start_ms")
    end_ms = range_bounds.get("effective_end_ms")

    lookback = max(window_seconds + 30, dashboard_live.DEFAULT_LOOKBACK_SECONDS)
    max_events = min(50_000, max(5_000, lookback * 40))
    if not is_live_range:
        max_events = min(100_000, max(10_000, window_seconds * 20))

    events: list[dict[str, Any]] = []
    candle_events: list[dict[str, Any]] = []
    archive_events: list[dict[str, Any]] = []
    archive_candles: list[dict[str, Any]] = []
    normalized_rows = int(archive_index.get("normalized_rows") or 0)
    archive_rows = int(archive_index.get("archive_rows") or 0)
    archive_available = bool((archive_index.get("archive") or {}).get("available"))
    normalized_index_available = bool((archive_index.get("normalized") or {}).get("available"))
    raw_index_available = bool((archive_index.get("raw_preview") or {}).get("available"))

    chart_data_source = "none"
    load_error: str | None = None
    fallback_used = False
    fallback_warning = range_bounds.get("missing_history_warning")

    try:
        if use_historical_loader and start_ms is not None and end_ms is not None:
            events = dashboard_live.load_market_events_for_range(
                session_id=resolved_session_id,
                start_ms=int(start_ms),
                end_ms=int(end_ms),
                max_events=max_events,
            )
            candle_events = dashboard_live.load_candle_events_for_range(
                session_id=resolved_session_id,
                start_ms=int(start_ms),
                end_ms=int(end_ms),
                max_events=max(2_000, min(max_events, 20_000)),
            )
            if archive_available:
                archive_events, archive_candles = dashboard_live.load_archive_market_events_for_range(
                    start_ms=int(start_ms),
                    end_ms=int(end_ms),
                    max_events=max_events,
                )
                if archive_events and not events:
                    events = archive_events
                if archive_candles and not candle_events:
                    candle_events = archive_candles
        else:
            events = dashboard_live.load_recent_market_events(
                session_id=resolved_session_id,
                lookback_seconds=lookback,
                max_events=max_events,
                historical=historical,
            )
            candle_events = dashboard_live.load_recent_candle_events(
                session_id=resolved_session_id,
                lookback_seconds=lookback,
                max_events=max(2_000, lookback),
                historical=historical,
            )
            if start_ms is not None and end_ms is not None:
                events, candle_events = _filter_events_for_range(
                    events,
                    candle_events,
                    start_ms=int(start_ms),
                    end_ms=int(end_ms),
                )

        raw_preview: dict[str, Any] = {
            "available": False,
            "session_id": resolved_session_id,
            "rows": [],
            "market_events": [],
            "candle_events": [],
            "files_read": [],
            "channels_used": [],
            "parse_warnings": [],
            "row_count": 0,
        }

        normalized_data_available = _events_have_coin_rows(
            events,
            candle_events,
            start_ms=int(start_ms) if start_ms is not None else None,
            end_ms=int(end_ms) if end_ms is not None else None,
        )
        if archive_events or archive_candles:
            chart_data_source = "archive" if not normalized_data_available else "normalized"
        elif normalized_data_available:
            chart_data_source = "normalized"

        if not normalized_data_available and resolved_session_id:
            raw_preview = dashboard_live.load_raw_preview_market_data(
                session_id=resolved_session_id,
                coins=CHART_COINS,
                max_events=min(max_events, 20_000),
            )
            if raw_preview.get("available"):
                events = raw_preview.get("market_events") or []
                if not candle_events:
                    candle_events = raw_preview.get("candle_events") or []
                if start_ms is not None and end_ms is not None:
                    events, candle_events = _filter_events_for_range(
                        events,
                        candle_events,
                        start_ms=int(start_ms),
                        end_ms=int(end_ms),
                    )
                chart_data_source = "raw_preview"
                normalized_data_available = False

        if (
            show_latest_fallback
            and not _events_have_coin_rows(
                events,
                candle_events,
                start_ms=int(start_ms) if start_ms is not None else None,
                end_ms=int(end_ms) if end_ms is not None else None,
            )
            and (raw_index_available or normalized_index_available or archive_available)
        ):
            latest_ms = _latest_event_timestamp_ms(events, candle_events)
            if latest_ms is None:
                latest_ms = range_bounds.get("latest_local_timestamp_ms")
            if latest_ms is not None:
                fallback_start = max(
                    int(latest_ms) - (window_seconds * 1000),
                    int(range_bounds.get("earliest_local_timestamp_ms") or 0),
                )
                fallback_events = dashboard_live.load_market_events_for_range(
                    session_id=resolved_session_id,
                    start_ms=fallback_start,
                    end_ms=int(latest_ms),
                    max_events=max_events,
                )
                fallback_candles = dashboard_live.load_candle_events_for_range(
                    session_id=resolved_session_id,
                    start_ms=fallback_start,
                    end_ms=int(latest_ms),
                    max_events=max(2_000, min(max_events, 20_000)),
                )
                if not _events_have_coin_rows(
                    fallback_events,
                    fallback_candles,
                    start_ms=fallback_start,
                    end_ms=int(latest_ms),
                ) and resolved_session_id:
                    raw_preview = dashboard_live.load_raw_preview_market_data(
                        session_id=resolved_session_id,
                        coins=CHART_COINS,
                        max_events=min(max_events, 20_000),
                    )
                    if raw_preview.get("available"):
                        fallback_events = raw_preview.get("market_events") or []
                        fallback_candles = raw_preview.get("candle_events") or []
                        chart_data_source = "raw_preview"

                if _events_have_coin_rows(
                    fallback_events,
                    fallback_candles,
                    start_ms=fallback_start,
                    end_ms=int(latest_ms),
                ):
                    events = fallback_events
                    candle_events = fallback_candles
                    fallback_used = True
                    fallback_warning = LATEST_FALLBACK_WARNING
                    end_ms = int(latest_ms)
                    start_ms = fallback_start
                    range_bounds = {
                        **range_bounds,
                        "effective_start_ms": fallback_start,
                        "effective_end_ms": int(latest_ms),
                    }

        if (
            not _events_have_coin_rows(events, candle_events)
            and (raw_index_available or normalized_index_available or archive_available)
            and chart_data_source == "none"
        ):
            chart_data_source = "local_summary"
    except Exception as exc:
        load_error = str(exc)
        chart_data_source = "error"
        raw_preview = {
            "available": False,
            "session_id": resolved_session_id,
            "rows": [],
            "market_events": [],
            "candle_events": [],
            "files_read": [],
            "channels_used": [],
            "parse_warnings": [],
            "row_count": 0,
        }
        events = []
        candle_events = []
        normalized_data_available = False

    if "raw_preview" not in locals():
        raw_preview = {
            "available": False,
            "session_id": resolved_session_id,
            "rows": [],
            "market_events": [],
            "candle_events": [],
            "files_read": [],
            "channels_used": [],
            "parse_warnings": [],
            "row_count": 0,
        }

    chart_rows_loaded = len(events) + len(candle_events)
    reference_time_s = _reference_time_s(events or candle_events, historical=use_historical_loader)
    if end_ms is not None and use_historical_loader:
        reference_time_s = end_ms / 1000.0
    elif fallback_used and end_ms is not None:
        reference_time_s = end_ms / 1000.0
    window_start_s = reference_time_s - window_seconds
    if start_ms is not None and use_historical_loader:
        window_start_s = start_ms / 1000.0

    timeline = timeline or []
    position_details = build_open_position_details(
        recommendations,
        paper_trades,
        reference_time_s=reference_time_s,
    )
    position_by_coin = {item["coin"]: item for item in position_details}

    max_display_points = _max_display_points(is_live_range=is_live_range)
    chart_rows_displayed = 0

    coins_data: dict[str, Any] = {}
    for coin in CHART_COINS:
        line = build_price_line_from_events(
            events,
            coin,
            resolution_seconds=resolution_seconds,
            window_seconds=window_seconds,
            reference_time_s=reference_time_s,
        )
        candles = build_candle_series_from_events(
            candle_events,
            coin,
            window_seconds=window_seconds,
            reference_time_s=reference_time_s,
        )
        displayed_series, series_count = downsample_series(
            line.get("series") or [],
            max_points=max_display_points,
        )
        line["series"] = displayed_series
        line["series_display_count"] = series_count
        displayed_candles, candle_count = downsample_candles(
            candles.get("candles") or [],
            max_points=max_display_points,
        )
        candles["candles"] = displayed_candles
        candles["candles_display_count"] = candle_count
        chart_rows_displayed += series_count + candle_count
        markers = build_paper_markers(
            timeline,
            recommendations,
            coin=coin,
            window_start_s=window_start_s,
            reference_time_s=reference_time_s,
        )
        markers.extend(build_open_position_markers(position_details, coin=coin))
        ref_lines = build_reference_lines(
            position_by_coin.get(coin),
            current_price=line.get("latest_price"),
        )
        coins_data[coin] = {
            **line,
            **candles,
            "markers": markers,
            "reference_lines": ref_lines,
            "position_detail": position_by_coin.get(coin),
        }

    if chart_data_source == "none" and chart_rows_loaded == 0:
        chart_data_source = "none"
    elif chart_data_source == "none" and chart_rows_loaded > 0:
        chart_data_source = "normalized"

    chart_empty_reason = _determine_chart_empty_reason(
        chart_data_source=chart_data_source,
        normalized_available=normalized_index_available,
        raw_preview_available=bool(raw_preview.get("available")),
        archive_available=archive_available,
        range_bounds=range_bounds,
        chart_rows_loaded=chart_rows_loaded,
        fallback_used=fallback_used,
    )

    data_mode = "historical" if use_historical_loader else "live"
    if is_all_local:
        data_mode_label = market_archive.ALL_LOCAL_LABEL
    elif chart_data_source == "raw_preview":
        data_mode_label = RAW_PREVIEW_MODE_LABEL
    elif chart_data_source == "archive":
        data_mode_label = "Local archive market data (read-only)"
    elif chart_data_source == "local_summary":
        data_mode_label = "Local data coverage summary (no chart rows in selected range)"
    elif use_historical_loader:
        data_mode_label = "Historical / locally collected market data"
    elif session_id:
        data_mode_label = "Live recorder session"
    else:
        data_mode_label = "No live recorder — showing latest available data"

    latest_data_timestamp_ms = range_bounds.get("latest_local_timestamp_ms")
    if events or candle_events:
        event_ts = [
            _event_timestamp_ms(event)
            for event in (events + candle_events)
            if _event_timestamp_ms(event) is not None
        ]
        if event_ts:
            latest_data_timestamp_ms = max(event_ts)

    seconds_since_latest = None
    if latest_data_timestamp_ms is not None and is_live_range and not prefer_latest_local:
        import time

        seconds_since_latest = max(0.0, time.time() - (latest_data_timestamp_ms / 1000.0))

    missing_history_warning = fallback_warning or range_bounds.get("missing_history_warning")

    return {
        "coins": coins_data,
        "window_seconds": window_seconds,
        "resolution_seconds": resolution_seconds,
        "selected_chart_range": selected_range,
        "range_bounds": range_bounds,
        "data_mode": data_mode,
        "data_mode_label": data_mode_label,
        "decision_timeline_available": bool(timeline),
        "open_paper_positions": position_details,
        "reference_time_s": reference_time_s,
        "paper_only": True,
        "market_event_count": len(events),
        "candle_event_count": len(candle_events),
        "chart_rows_loaded": chart_rows_loaded,
        "chart_rows_displayed": chart_rows_displayed,
        "normalized_data_available": bool(events or candle_events) and chart_data_source == "normalized",
        "raw_preview_available": raw_preview.get("available", False),
        "raw_preview_rows": raw_preview.get("row_count", 0),
        "raw_preview_session_id": raw_preview.get("session_id"),
        "raw_preview_files_read": raw_preview.get("files_read", []),
        "raw_preview_channels_used": raw_preview.get("channels_used", []),
        "raw_preview_warnings": raw_preview.get("parse_warnings", []),
        "normalized_rows": normalized_rows,
        "archive_rows": archive_rows,
        "archive_available": archive_available,
        "chart_data_source": chart_data_source,
        "chart_empty_reason": chart_empty_reason,
        "fallback_used": fallback_used,
        "show_latest_fallback": show_latest_fallback,
        "load_error": load_error,
        "missing_history_warning": missing_history_warning,
        "local_coverage_percent": range_bounds.get("local_coverage_percent"),
        "earliest_local_timestamp_ms": range_bounds.get("earliest_local_timestamp_ms"),
        "latest_local_timestamp_ms": range_bounds.get("latest_local_timestamp_ms"),
        "requested_range_start_ms": range_bounds.get("requested_range_start_ms"),
        "requested_range_end_ms": range_bounds.get("requested_range_end_ms"),
        "latest_data_timestamp_ms": latest_data_timestamp_ms,
        "seconds_since_latest_market_update": seconds_since_latest,
        "archive_index": archive_index,
        "is_all_local": is_all_local,
        "is_live_range": is_live_range,
        "resolved_session_id": resolved_session_id,
    }


def _window_key_to_range(window_key: str) -> str:
    mapping = {
        "1m": "Live 1m",
        "5m": "Live 5m",
        "15m": "Live 15m",
        "1h": "1h",
    }
    return mapping.get(window_key, DEFAULT_CHART_RANGE)


def create_price_chart_figure(
    coin: str,
    coin_data: dict[str, Any],
    *,
    chart_mode: str = "line",
    show_markers: bool = True,
    show_risk_lines: bool = True,
    data_mode_label: str = "",
):
    """Build a Plotly figure for one coin (visualization only)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": False}]])

    use_candles = chart_mode == "candles" and coin_data.get("has_candles")
    if use_candles:
        candles = coin_data.get("candles") or []
        fig.add_trace(
            go.Candlestick(
                x=[item["time_ms"] for item in candles],
                open=[item["open"] for item in candles],
                high=[item["high"] for item in candles],
                low=[item["low"] for item in candles],
                close=[item["close"] for item in candles],
                name=f"{coin} candles",
            )
        )
    else:
        series = coin_data.get("series") or []
        times = [item["time_ms"] for item in series]
        prices = [item["price"] for item in series]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=prices,
                mode="lines",
                name=f"{coin} price",
                line={"color": "#ecf0f1", "width": 2},
                connectgaps=True,
            )
        )
        bids = [item.get("bid") for item in series]
        asks = [item.get("ask") for item in series]
        if any(v is not None for v in bids) and any(v is not None for v in asks):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=bids,
                    mode="lines",
                    name="Bid",
                    line={"color": "rgba(46, 204, 113, 0.35)", "width": 1},
                    connectgaps=True,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=asks,
                    mode="lines",
                    name="Ask",
                    line={"color": "rgba(231, 76, 60, 0.35)", "width": 1},
                    fill="tonexty",
                    fillcolor="rgba(52, 152, 219, 0.08)",
                    connectgaps=True,
                )
            )

    if show_risk_lines:
        for line in coin_data.get("reference_lines") or []:
            fig.add_hline(
                y=line["price"],
                line_dash=line.get("dash", "solid"),
                line_color=line.get("color", "#888"),
                annotation_text=line.get("label"),
                annotation_position="top left",
            )

    if show_markers:
        markers = coin_data.get("markers") or []
        by_type: dict[str, list[dict[str, Any]]] = {}
        for marker in markers:
            by_type.setdefault(marker.get("marker_type", MARKER_HOLD), []).append(marker)

        for marker_type, group in by_type.items():
            xs = []
            ys = []
            hovers = []
            for marker in group:
                ts_s = marker.get("timestamp_s")
                price = marker.get("price")
                if ts_s is not None and price is not None:
                    xs.append(ts_s * 1000)
                    ys.append(price)
                    hovers.append(marker.get("hover_text", marker.get("decision", "")))
                elif marker.get("price") is not None:
                    xs.append(coin_data.get("series", [{}])[-1]["time_ms"] if coin_data.get("series") else 0)
                    ys.append(marker["price"])
                    hovers.append(marker.get("hover_text", ""))

            if not xs:
                continue
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    name=f"paper {marker_type}",
                    marker={
                        "symbol": marker.get("symbol", "circle"),
                        "size": 11,
                        "color": marker.get("color", "#95a5a6"),
                        "line": {"width": 1, "color": "#1a1a1a"},
                    },
                    hovertext=hovers,
                    hoverinfo="text",
                )
            )

    latest = coin_data.get("latest_price")
    title = f"{coin} — {latest if latest is not None else '—'}"
    if coin_data.get("spread_bps") is not None:
        title += f"  (spread {coin_data['spread_bps']} bps)"

    fig.update_layout(
        title={"text": title, "x": 0.02, "font": {"size": 18}},
        template="plotly_dark",
        height=420,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        xaxis={"title": "Time (ms)", "showgrid": True, "gridcolor": "#333"},
        yaxis={"title": "Price", "showgrid": True, "gridcolor": "#333"},
        legend={"orientation": "h", "y": 1.08},
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
    )
    if data_mode_label:
        fig.add_annotation(
            text=data_mode_label,
            xref="paper",
            yref="paper",
            x=1,
            y=1.08,
            showarrow=False,
            font={"size": 11, "color": "#aaa"},
            xanchor="right",
        )
    return fig


def chart_controls_are_visualization_only(source: str) -> bool:
    """Return True when chart UI code has no execution/recorder hooks."""
    forbidden = (
        "place_order",
        "execute_order",
        "sign_transaction",
        "private_key",
        "websocket_recorder",
        "subprocess",
        "main.py",
    )
    lowered = source.lower()
    return not any(token in lowered for token in forbidden)
