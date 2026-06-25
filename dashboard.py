"""
Read-only Streamlit dashboard for the paper trading bot.

Displays decision-focused recommendations, session state, price history,
paper trade results, and backtest summaries from local CSV/JSON files.
No wallet, signing, or real exchange access.
"""

from datetime import date, datetime, timezone

import pandas as pd
import streamlit as st

from config import DATA, MARKET, PAPER_TRADING, VERSION
import dashboard_data
import dashboard_live
import dashboard_charts
import dashboard_market_archive
import dashboard_review
import session


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


@st.cache_data(ttl=30)
def load_recorder_status() -> dict | None:
    return dashboard_data.load_recorder_status()


@st.cache_data(ttl=30)
def load_market_stream_sizes() -> dict[str, int]:
    return dashboard_data.market_stream_file_sizes()


@st.cache_data(ttl=30)
def load_quality_reports() -> list[dict]:
    return dashboard_data.load_quality_reports()


@st.cache_data(ttl=30)
def load_quality_report(session_id: str) -> dict | None:
    return dashboard_data.load_quality_report(session_id)


@st.cache_data(ttl=30)
def load_replay_reports() -> list[dict]:
    return dashboard_data.load_replay_reports()


@st.cache_data(ttl=30)
def load_replay_report(replay_session_id: str) -> dict | None:
    return dashboard_data.load_replay_report(replay_session_id)


@st.cache_data(ttl=30)
def load_event_backtest_reports() -> list[dict]:
    return dashboard_data.load_event_backtest_reports()


@st.cache_data(ttl=30)
def load_event_backtest_report(event_backtest_id: str) -> dict | None:
    return dashboard_data.load_event_backtest_report(event_backtest_id)


@st.cache_data(ttl=30)
def load_feature_research_reports() -> list[dict]:
    return dashboard_data.load_feature_research_reports()


@st.cache_data(ttl=30)
def load_feature_research_report(feature_research_id: str) -> dict | None:
    return dashboard_data.load_feature_research_report(feature_research_id)


@st.cache_data(ttl=5)
def load_live_market_dashboard() -> dict:
    return dashboard_data.load_live_market_dashboard_bundle()


def _format_timestamp_ms(timestamp_ms: int | float | None) -> str:
    if timestamp_ms is None:
        return "—"
    try:
        dt = datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (TypeError, ValueError, OSError):
        return str(timestamp_ms)


def _apply_chart_auto_refresh(*, enabled: bool, interval_seconds: int) -> tuple[bool, str | None]:
    """Trigger non-blocking Streamlit reruns when optional dependency is available."""
    if not enabled:
        return False, None
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=max(1, interval_seconds) * 1000, key="live_chart_autorefresh")
        return True, None
    except ImportError:
        return False, dashboard_live.auto_refresh_status_message(
            enabled=True,
            dependency_available=False,
        )


def _render_dashboard_state_diagnostics(
    bundle: dict | None,
    *,
    chart_data: dict | None = None,
) -> None:
    """Collapsed read-only state diagnostics for chart/source debugging."""
    ws = (bundle or {}).get("websocket_status") or {}
    context = (bundle or {}).get("context") or {}
    chart_data = chart_data or {}
    with st.expander("Dashboard State Diagnostics", expanded=False):
        st.write(f"websocket_status: **{ws.get('display_connection_status') or ws.get('connection_state') or '—'}**")
        st.write(
            f"latest_session_id: **"
            f"{dashboard_live.resolve_latest_recorder_session_id(context=context, archive_index=chart_data.get('archive_index')) or '—'}"
            f"**"
        )
        st.write(f"selected_chart_range: **{chart_data.get('selected_chart_range') or '—'}**")
        st.write(f"chart_data_source: **{chart_data.get('chart_data_source', 'none')}**")
        st.write(f"chart_rows_loaded: **{chart_data.get('chart_rows_loaded', 0)}**")
        st.write(f"chart_empty_reason: **{chart_data.get('chart_empty_reason') or '—'}**")
        st.write(f"normalized_available: **{chart_data.get('normalized_data_available', False)}**")
        st.write(f"raw_preview_available: **{chart_data.get('raw_preview_available', False)}**")
        st.write(f"archive_available: **{chart_data.get('archive_available', False)}**")
        st.write(
            f"latest_local_timestamp: **{_format_timestamp_ms(chart_data.get('latest_local_timestamp_ms'))}**"
        )
        st.write(
            f"earliest_local_timestamp: **{_format_timestamp_ms(chart_data.get('earliest_local_timestamp_ms'))}**"
        )
        st.write(f"fallback_used: **{chart_data.get('fallback_used', False)}**")
        review_loaded = st.session_state.get("review_mode_loaded")
        st.write(f"review_mode_loaded: **{review_loaded if review_loaded is not None else '—'}**")


def _render_live_dashboard_debug_expander(
    bundle: dict | None,
    *,
    chart_data: dict | None = None,
    bundle_loaded: bool = False,
    auto_refresh_enabled: bool | None = None,
    refresh_interval_seconds: int | None = None,
) -> None:
    """Read-only debug panel — collapsed by default, schema-safe."""
    try:
        snapshot = dashboard_live.live_dashboard_debug_snapshot(
            bundle,
            chart_data=chart_data,
            bundle_loaded=bundle_loaded,
            auto_refresh_enabled=auto_refresh_enabled,
            refresh_interval_seconds=refresh_interval_seconds,
        )
    except Exception as error:
        with st.expander("Debug (internal)", expanded=False):
            st.error("Debug snapshot failed to build")
            st.exception(error)
        return

    with st.expander("Debug (internal)", expanded=False):
        st.caption("Raw paths, bundle keys, and internal diagnostics — expand only when needed.")
        st.write(f"bundle loaded: **{snapshot.get('bundle_loaded', False)}**")
        st.write(f"auto_refresh_enabled: **{snapshot.get('auto_refresh_enabled')}**")
        st.write(f"refresh_interval_seconds: **{snapshot.get('refresh_interval_seconds')}**")
        st.write(f"selected_chart_range: **{snapshot.get('selected_chart_range') or '—'}**")
        st.write(f"WebSocket status state: **{snapshot.get('websocket_state', '—')}**")
        st.write(f"market rows count: **{snapshot.get('market_rows_count', 0)}**")
        st.write(f"candle rows count: **{snapshot.get('candle_rows_count', 0)}**")
        st.write(f"chart rows loaded: **{snapshot.get('chart_rows_loaded')}**")
        st.write(f"chart rows displayed: **{snapshot.get('chart_rows_displayed')}**")
        st.write(
            f"decision timeline rows count: **{snapshot.get('decision_timeline_rows_count', 0)}**"
        )
        st.write(f"historical fallback: **{snapshot.get('historical_fallback', False)}**")
        st.write(
            f"normalized data available: **{snapshot.get('normalized_data_available', False)}**"
        )
        st.write(f"raw preview available: **{snapshot.get('raw_preview_available', False)}**")
        st.write(f"raw preview rows: **{snapshot.get('raw_preview_rows', 0)}**")
        st.write(f"normalized rows: **{snapshot.get('normalized_rows')}**")
        st.write(f"archive rows: **{snapshot.get('archive_rows')}**")
        st.write(f"raw preview session id: **{snapshot.get('raw_preview_session_id') or '—'}**")
        st.write(f"chart data source: **{snapshot.get('chart_data_source', 'none')}**")
        st.write(
            f"earliest local timestamp: **{_format_timestamp_ms(snapshot.get('earliest_local_timestamp'))}**"
        )
        st.write(
            f"latest local timestamp: **{_format_timestamp_ms(snapshot.get('latest_local_timestamp'))}**"
        )
        st.write(
            f"requested range start: **{_format_timestamp_ms(snapshot.get('requested_range_start'))}**"
        )
        st.write(
            f"requested range end: **{_format_timestamp_ms(snapshot.get('requested_range_end'))}**"
        )
        st.write(f"local coverage percent: **{snapshot.get('local_coverage_percent')}**")
        st.write(
            f"missing history warning: **{snapshot.get('missing_history_warning') or '—'}**"
        )
        raw_files = snapshot.get("raw_preview_files_read") or []
        if raw_files:
            st.write(f"raw preview files read: `{raw_files}`")
        raw_channels = snapshot.get("raw_preview_channels_used") or []
        if raw_channels:
            st.write(f"raw preview channels used: **{raw_channels}**")
        raw_warnings = snapshot.get("raw_preview_warnings") or []
        if raw_warnings:
            st.caption(f"raw preview warnings: {raw_warnings}")
        latest_path = snapshot.get("latest_data_path")
        if latest_path:
            st.write(f"latest data path: `{latest_path}`")
        load_error = snapshot.get("load_error")
        if load_error:
            st.caption(f"load error: {load_error}")
        st.write(f"normalization queue count: **{snapshot.get('normalization_queue_count', 0)}**")
        st.write(
            f"newest unprocessed session id: **{snapshot.get('newest_unprocessed_session_id') or '—'}**"
        )
        st.write(
            f"normalization sessions scanned: **{snapshot.get('normalization_sessions_scanned', 0)}**"
        )
        queue_warnings = snapshot.get("normalization_queue_warnings") or []
        if queue_warnings:
            st.caption(f"normalization queue warnings: {queue_warnings}")

        if bundle:
            with st.expander("Internal bundle keys", expanded=False):
                st.json({key: type(value).__name__ for key, value in bundle.items()})
            with st.expander("Raw JSON diagnostics", expanded=False):
                st.json(
                    {
                        "context": bundle.get("context"),
                        "load_error": bundle.get("load_error"),
                        "websocket_status_keys": list(
                            (bundle.get("websocket_status") or {}).keys()
                        ),
                    }
                )


def _render_live_market_charts(bundle: dict, context: dict) -> dict:
    """Read-only price charts (visualization only, no paper-trade dependency)."""
    st.markdown("#### Live Charts")
    st.caption(dashboard_live.DATA_QUALITY_EXPLANATION)

    ws = bundle.get("websocket_status") or {}
    default_auto_refresh = dashboard_live.default_auto_refresh_enabled(ws)
    refresh_interval = dashboard_live.DEFAULT_AUTO_REFRESH_INTERVAL_SECONDS

    if "live_chart_auto_refresh" not in st.session_state:
        st.session_state["live_chart_auto_refresh"] = default_auto_refresh

    control_cols = st.columns(8)
    coin_options = ["All"] + list(dashboard_charts.CHART_COINS)
    coin_sel = control_cols[0].selectbox("Coin", coin_options, key="live_chart_coin")
    range_options = list(dashboard_charts.CHART_RANGE_KEYS)
    default_range_index = range_options.index(dashboard_charts.DEFAULT_CHART_RANGE)
    range_sel = control_cols[1].selectbox(
        "Chart range",
        range_options,
        index=default_range_index,
        key="live_chart_range",
    )
    resolution_sel = control_cols[2].selectbox(
        "Resolution",
        list(dashboard_charts.RESOLUTION_SECONDS.keys()),
        key="live_chart_resolution",
    )
    chart_mode = control_cols[3].selectbox("Chart mode", ["line", "candles"], key="live_chart_mode")
    show_markers = control_cols[4].checkbox("Paper signal markers", value=False, key="live_chart_markers")
    show_risk_lines = control_cols[5].checkbox("Paper TP/SL lines", value=False, key="live_chart_risk_lines")
    auto_refresh = control_cols[6].checkbox(
        "Auto-refresh live charts",
        value=st.session_state.get("live_chart_auto_refresh", default_auto_refresh),
        key="live_chart_auto_refresh",
    )
    show_latest_fallback = control_cols[7].checkbox(
        "Show latest available data when selected range is empty",
        value=st.session_state.get("live_chart_latest_fallback", True),
        key="live_chart_latest_fallback",
    )

    dependency_available, auto_refresh_message = _apply_chart_auto_refresh(
        enabled=auto_refresh,
        interval_seconds=refresh_interval,
    )

    data_mode = context.get("data_mode")
    historical = data_mode == "historical"
    session_id = context.get("recorder_session_id")
    ws = bundle.get("websocket_status") or {}
    recorder_running = bool(ws.get("running"))
    archive_index = dashboard_market_archive.build_market_archive_index()

    chart_data = dashboard_charts.build_chart_data(
        session_id=session_id,
        historical=historical or context.get("data_mode") == "missing",
        range_key=range_sel,
        resolution_key=resolution_sel,
        recommendations=bundle.get("recommendations") if show_markers else None,
        timeline=bundle.get("decision_timeline") if show_markers else None,
        paper_trades=None,
        archive_index=archive_index,
        show_latest_fallback=show_latest_fallback,
        recorder_running=recorder_running,
    )

    chart_source = chart_data.get("chart_data_source", "none")

    if auto_refresh_message:
        st.info(auto_refresh_message)
    elif auto_refresh and dependency_available:
        st.caption(f"Auto-refresh enabled ({refresh_interval}s). Read-only local data reload only.")

    if data_mode == "missing" and chart_source == "none":
        st.info("No live recorder data found.")
    elif historical and chart_data.get("is_live_range"):
        st.caption("Recorder not running — showing latest locally collected data for the selected range.")

    if chart_data.get("is_all_local"):
        st.info(dashboard_market_archive.ALL_LOCAL_LABEL)

    if chart_data.get("missing_history_warning"):
        st.warning(chart_data.get("missing_history_warning"))

    if (
        data_mode != "missing"
        and not chart_data.get("normalized_data_available")
        and not chart_data.get("raw_preview_available")
        and chart_source == "none"
    ):
        st.info("No market data found for the selected range.")

    if chart_data.get("chart_data_source") == "raw_preview":
        st.warning(dashboard_charts.RAW_PREVIEW_MODE_LABEL)

    mode_label = chart_data.get("data_mode_label", "")
    coins_to_render = (
        list(dashboard_charts.CHART_COINS)
        if coin_sel == "All"
        else [coin_sel]
    )

    for coin in coins_to_render:
        coin_data = (chart_data.get("coins") or {}).get(coin) or {}
        if not coin_data.get("has_data") and not coin_data.get("has_candles"):
            st.info(dashboard_charts.format_coin_chart_empty_message(coin, chart_data))
            continue

        chart_col, side_col = st.columns([3, 1])
        with chart_col:
            fig = dashboard_charts.create_price_chart_figure(
                coin,
                coin_data,
                chart_mode=chart_mode,
                show_markers=show_markers,
                show_risk_lines=show_risk_lines,
                data_mode_label=mode_label,
            )
            st.plotly_chart(fig, use_container_width=True)

        with side_col:
            st.markdown(f"**{coin}**")
            latest = coin_data.get("latest_price")
            st.metric("Latest price", f"{latest:.4f}" if latest is not None else "—")
            spread = coin_data.get("spread_bps")
            if spread is not None:
                st.caption(f"Spread: {spread} bps")

    return chart_data


def _render_header_status_row(
    bundle: dict,
    chart_data: dict | None,
) -> None:
    """Section A: compact header status row."""
    header = dashboard_live.meeting_header_cards(bundle)
    ws = bundle.get("websocket_status") or {}
    ws_status = ws.get("display_connection_status") or header.get("websocket", "—")
    ws_color = (
        "#2ecc71"
        if ws_status == "CONNECTED"
        else "#f39c12"
        if ws_status == "STALE"
        else "#e74c3c"
    )
    bot_color = "#2ecc71" if header.get("paper_bot") == "RUNNING" else "#f39c12"
    chart_source = (chart_data or {}).get("chart_data_source") or "none"
    seconds_since = (chart_data or {}).get("seconds_since_latest_market_update")
    if seconds_since is None:
        hb_age = ws.get("heartbeat_age_seconds")
        if hb_age is not None:
            seconds_since = hb_age

    card_cols = st.columns(5)
    card_cols[0].markdown(
        f"**WebSocket**  \n"
        f"<span style='font-size:1.4rem;font-weight:700;color:{ws_color};'>"
        f"{ws_status}</span>",
        unsafe_allow_html=True,
    )
    card_cols[1].markdown(
        f"**Paper Bot**  \n"
        f"<span style='font-size:1.4rem;font-weight:700;color:{bot_color};'>"
        f"{header.get('paper_bot', '—')}</span>",
        unsafe_allow_html=True,
    )
    card_cols[2].markdown(
        "**Safety**  \n"
        "<span style='font-size:1.4rem;font-weight:700;color:#3498db;'>"
        f"{header.get('safety', 'PAPER ONLY')}</span>",
        unsafe_allow_html=True,
    )
    card_cols[3].metric(
        "Market update age",
        f"{seconds_since:.0f}s" if seconds_since is not None else "—",
    )
    card_cols[4].metric("Chart data source", chart_source)


def _render_recorder_reliability_panel(bundle: dict, context: dict) -> None:
    """Section C: compact recorder reliability panel."""
    st.markdown("#### Recorder Reliability")
    ws = bundle.get("websocket_status") or {}
    session_id = dashboard_live.resolve_latest_recorder_session_id(
        context=context,
    )
    manifest = dashboard_live.resolve_latest_recorder_manifest(
        context=context,
        session_id=session_id,
    )
    checkpoint = (
        dashboard_data.load_recorder_checkpoint(session_id)
        if session_id
        else None
    )
    quality_report = (
        dashboard_data.load_quality_report(session_id)
        if session_id
        else None
    )
    summary = dashboard_live.build_recorder_reliability_summary(
        ws,
        context=context,
        checkpoint=checkpoint,
        manifest=manifest,
        quality_report=quality_report,
    )

    if ws.get("stale"):
        st.warning(
            "Stale WebSocket status — recorder process may have crashed. "
            "Run `python3 websocket_recorder.py --recover-latest`."
        )
    if summary.get("state_mismatch_warning"):
        st.warning(summary["state_mismatch_warning"])

    rcols = st.columns(5)
    rcols[0].metric("Session ID", f"{(summary.get('session_id') or '—')[:12]}…")
    rcols[1].metric("State", summary.get("state", "—"))
    rcols[2].metric("Raw events", summary.get("raw_events", "—"))
    rcols[3].metric("Dropped raw", summary.get("dropped_raw", 0))
    rcols[4].metric("Reconnects", summary.get("reconnects", 0))

    rcols2 = st.columns(5)
    rcols2[0].metric("Gaps", summary.get("gaps", 0))
    rcols2[1].metric("Degraded", "Yes" if summary.get("degraded") else "No")
    rcols2[2].metric("Shutdown reason", summary.get("shutdown_reason") or "—")
    rcols2[3].metric("Manifest", summary.get("manifest_consistency", "—"))
    rcols2[4].metric("Quality", summary.get("quality_label", "—"))

    if summary.get("checkpoint_age"):
        st.caption(f"Latest checkpoint: {summary['checkpoint_age']}")
    if manifest and summary.get("manifest_state"):
        st.caption(f"Latest manifest state: {summary.get('manifest_state')}")


def _render_data_coverage_section(archive_index: dict) -> None:
    """Section D: per-coin data coverage table."""
    st.markdown("#### Data Coverage")
    rows = dashboard_live.build_data_coverage_rows(archive_index)
    display_rows = []
    for row in rows:
        display_rows.append(
            {
                "Coin": row.get("coin"),
                "Earliest local": _format_timestamp_ms(row.get("earliest_local_timestamp")),
                "Latest local": _format_timestamp_ms(row.get("latest_local_timestamp")),
                "Trades": "Yes" if row.get("trades_available") else "No",
                "BBO": "Yes" if row.get("bbo_available") else "No",
                "Candles": "Yes" if row.get("candles_available") else "No",
                "Normalized": "Yes" if row.get("normalized_available") else "No",
                "Raw": "Yes" if row.get("raw_available") else "No",
                "Note": row.get("local_coverage_note", "—"),
            }
        )
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)


def _render_research_readiness_section(
    archive_index: dict,
    *,
    session_id: str | None,
) -> None:
    """Section E: research pipeline readiness labels."""
    st.markdown("#### Research Readiness")
    try:
        replay_count = len(load_replay_reports())
        feature_count = len(load_feature_research_reports())
        backtest_count = len(load_event_backtest_reports())
    except Exception:
        replay_count = feature_count = backtest_count = 0

    readiness = dashboard_live.build_research_readiness(
        archive_index,
        session_id=session_id,
        replay_count=replay_count,
        feature_count=feature_count,
        backtest_count=backtest_count,
    )

    for label, ready in readiness.get("labels", {}).items():
        if ready:
            st.success(f"**{label}**")
        else:
            st.info(f"**{label}** — not ready")

    st.caption(readiness.get("explanation", dashboard_live.DATA_QUALITY_EXPLANATION))
    for key, text in (readiness.get("data_quality_labels") or {}).items():
        st.caption(f"- {text}")

    next_cmd = readiness.get("next_recommended_command")
    if next_cmd:
        st.code(next_cmd, language="bash")


def _normalization_queue_display_rows(rows: list[dict]) -> list[dict]:
    display_rows = []
    for row in rows:
        display_rows.append(
            {
                "Session ID": row.get("session_id"),
                "Start": _format_timestamp_ms(row.get("start_time_ms")),
                "End": _format_timestamp_ms(row.get("end_time_ms")),
                "Duration": row.get("duration_label", "—"),
                "Raw events": row.get("raw_events"),
                "Quality": row.get("quality_label", "unknown"),
                "Shutdown reason": row.get("shutdown_reason", "—"),
                "Normalized": "Yes" if row.get("normalized_exists") else "No",
                "Manifest": row.get("manifest_consistency", "—"),
                "Channels": row.get("recommended_channels", "—"),
                "Recommended command": row.get("recommended_command", "—"),
                "Next step": row.get("next_step", "—"),
            }
        )
    return display_rows


def _render_normalization_queue_section() -> None:
    """Section F: raw-only sessions awaiting offline normalization (display-only commands)."""
    st.markdown("#### Normalization Queue")
    st.caption(
        "Raw-only recorder sessions that can be normalized offline. "
        "Commands are display-only — the dashboard never runs normalization."
    )
    show_processed = st.checkbox(
        "Show processed sessions",
        value=False,
        key="normalization_queue_show_processed",
    )
    queue = dashboard_live.build_normalization_queue(show_processed=show_processed)
    rows = queue.get("rows") or []
    visible_rows = queue.get("visible_rows") or rows[: dashboard_live.DEFAULT_NORMALIZATION_QUEUE_VISIBLE]
    older_rows = queue.get("older_rows") or rows[dashboard_live.DEFAULT_NORMALIZATION_QUEUE_VISIBLE :]

    warnings = queue.get("normalization_queue_warnings") or []
    for warning in warnings[:5]:
        st.caption(f"Queue warning: {warning}")

    if not rows:
        st.info("No raw-usable sessions waiting for offline normalization.")
        return

    st.dataframe(
        pd.DataFrame(_normalization_queue_display_rows(visible_rows)),
        use_container_width=True,
        hide_index=True,
    )

    if older_rows:
        with st.expander(f"Older sessions ({len(older_rows)})", expanded=False):
            st.dataframe(
                pd.DataFrame(_normalization_queue_display_rows(older_rows)),
                use_container_width=True,
                hide_index=True,
            )

    newest = queue.get("newest_unprocessed_session_id")
    if newest and not show_processed:
        st.caption(
            f"Newest unprocessed session: `{newest}`. "
            f"Queue count: {queue.get('normalization_queue_count', 0)}."
        )


def _render_strategy_research_summary() -> None:
    """Section F: compact strategy research summary from event backtests."""
    st.markdown("#### Strategy Research")
    try:
        reports = load_event_backtest_reports()
    except Exception as error:
        st.warning("Event backtest reports unavailable.")
        st.caption(str(error))
        return

    if not reports:
        st.info(
            "No event backtest reports yet. Run "
            "`python3 event_backtest.py --latest-replay --coins BTC ETH SOL "
            "--strategy contrarian_bbo_confirmation_v1` after replay validation."
        )
        return

    latest = reports[0]
    scols = st.columns(5)
    scols[0].metric("Strategy", latest.get("strategy_name", "—"))
    scols[1].metric("Profile", (latest.get("profile_name") or "—")[:20])
    scols[2].metric("Mode", latest.get("run_mode", latest.get("mode", "—")))
    scols[3].metric("Evidence", latest.get("evidence_label", "—"))
    scols[4].metric("Completed trades", latest.get("completed_trades", 0))

    scols2 = st.columns(3)
    scols2[0].metric("Net return", _format_pct(latest.get("net_return_pct")))
    scols2[1].metric("Max drawdown", _format_pct(latest.get("maximum_drawdown_pct")))
    blocked = latest.get("blocked_signals")
    scols2[2].metric("Blocked signals", blocked if blocked is not None else "—")

    blocked_reasons = latest.get("blocked_reasons") or latest.get("top_blocked_reasons") or []
    if blocked_reasons:
        with st.expander("Blocked reasons", expanded=False):
            for reason in blocked_reasons[:10]:
                st.caption(f"- {reason}")

    with st.expander("All event backtest reports", expanded=False):
        summary_rows = [
            {
                "ID": (r.get("event_backtest_id") or "—")[:8],
                "Strategy": r.get("strategy_name"),
                "Profile": r.get("profile_name"),
                "Trades": r.get("completed_trades"),
                "Net return (%)": r.get("net_return_pct"),
                "Evidence": r.get("evidence_label"),
            }
            for r in reports[:10]
        ]
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


def _render_paper_portfolio_section(bundle: dict) -> None:
    """Section G: paper portfolio/risk with detailed tables behind expanders."""
    st.markdown("#### Paper Portfolio / Risk")
    portfolio = bundle.get("paper_portfolio") or {}
    if not portfolio:
        st.info("No paper portfolio data found. Run `python3 main.py` to populate paper metrics.")
        return

    pcols1 = st.columns(4)
    pcols1[0].metric("Paper equity", _format_money(portfolio.get("paper_equity")))
    pcols1[1].metric("Open positions", portfolio.get("open_paper_positions", 0))
    pcols1[2].metric("Exposure", _format_money(portfolio.get("exposure")))
    pcols1[3].metric("Capital readiness", portfolio.get("capital_readiness_label", "—"))

    pcols2 = st.columns(4)
    pcols2[0].metric("Fees", _format_money(portfolio.get("fees")))
    pcols2[1].metric("Slippage", _format_money(portfolio.get("slippage")))
    pcols2[2].metric("Max drawdown", _format_pct(portfolio.get("max_drawdown_pct")))
    pcols2[3].metric("Session trades", portfolio.get("session_trade_count", 0))

    blocked_reasons = portfolio.get("blocked_reasons") or []
    if blocked_reasons:
        with st.expander("Blocked reasons", expanded=False):
            for reason in blocked_reasons[:10]:
                st.caption(f"- {reason}")

    with st.expander("Per-second market data (raw table)", expanded=False):
        per_second = bundle.get("per_second_market") or []
        has_market_values = any(
            row.get("latest_price") is not None or row.get("best_bid") is not None
            for row in per_second
        )
        if per_second and has_market_values:
            ps_rows = [
                {
                    "Coin": row.get("coin"),
                    "Latest price": row.get("latest_price"),
                    "Best bid": row.get("best_bid"),
                    "Best ask": row.get("best_ask"),
                    "Spread bps": row.get("spread_bps"),
                    "Trades/s": row.get("trades_per_second"),
                    "Last update": row.get("last_update_time"),
                }
                for row in per_second
            ]
            st.dataframe(pd.DataFrame(ps_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No normalized market data found yet.")

    with st.expander("Live bot decisions (paper-only)", expanded=False):
        decisions = bundle.get("live_bot_decisions") or []
        if decisions:
            dec_rows = [
                {
                    "Coin": item.get("coin"),
                    "Decision": item.get("decision"),
                    "Reason": item.get("reason"),
                    "Signal strength": item.get("signal_strength"),
                }
                for item in decisions
            ]
            st.dataframe(pd.DataFrame(dec_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Run `python3 main.py` to populate paper-only bot decisions.")

    with st.expander("Decision timeline (paper-only)", expanded=False):
        timeline = bundle.get("decision_timeline") or []
        if timeline:
            tl_rows = [
                {
                    "Timestamp": item.get("timestamp"),
                    "Coin": item.get("coin"),
                    "Decision": item.get("decision"),
                    "Reason": item.get("reason"),
                    "Price": item.get("price"),
                }
                for item in timeline
            ]
            st.dataframe(pd.DataFrame(tl_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No paper decision timeline found.")

    with st.expander("Feature signals (research indicators)", expanded=False):
        features = bundle.get("feature_signals") or []
        if features:
            feat_rows = [
                {
                    "Coin": item.get("coin"),
                    "BBO imbalance": item.get("bbo_imbalance"),
                    "Return 5s (%)": item.get("return_5s"),
                    "Spread bps": item.get("spread_bps"),
                }
                for item in features
            ]
            st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No normalized market data found yet.")


def _render_live_market_dashboard() -> None:
    st.subheader("Live Market Dashboard")
    st.caption("Read-only public market data. Paper-only decisions. No orders or wallet actions.")

    bundle: dict | None = None
    chart_data: dict | None = None
    bundle_loaded = False
    archive_index: dict = {}

    try:
        bundle = load_live_market_dashboard()
        bundle_loaded = True
        load_error = bundle.get("load_error")
        if load_error:
            st.warning(f"Live dashboard partial load: {load_error}")

        context = bundle.get("context") or {}
        session_id = context.get("recorder_session_id")

        try:
            archive_index = dashboard_market_archive.build_market_archive_index()
        except Exception as error:
            st.warning("Data coverage index unavailable.")
            st.caption(str(error))
            archive_index = {}

        st.divider()

        try:
            chart_data = _render_live_market_charts(bundle, context)
        except Exception as error:
            st.error("Live charts failed to load")
            st.exception(error)
            chart_data = chart_data or {}

        try:
            _render_header_status_row(bundle, chart_data)
        except Exception as error:
            st.warning("Header status row unavailable.")
            st.caption(str(error))

        st.divider()

        try:
            _render_recorder_reliability_panel(bundle, context)
        except Exception as error:
            st.warning("Recorder reliability panel unavailable.")
            st.caption(str(error))

        st.divider()

        try:
            _render_data_coverage_section(archive_index)
        except Exception as error:
            st.warning("Data coverage section unavailable.")
            st.caption(str(error))

        st.divider()

        try:
            _render_research_readiness_section(archive_index, session_id=session_id)
        except Exception as error:
            st.warning("Research readiness section unavailable.")
            st.caption(str(error))

        st.divider()

        try:
            _render_normalization_queue_section()
        except Exception as error:
            st.warning("Normalization queue section unavailable.")
            st.caption(str(error))

        st.divider()

        try:
            _render_strategy_research_summary()
        except Exception as error:
            st.warning("Strategy research section unavailable.")
            st.caption(str(error))

        st.divider()

        try:
            _render_paper_portfolio_section(bundle)
        except Exception as error:
            st.warning("Paper portfolio section unavailable.")
            st.caption(str(error))

    except Exception as error:
        st.error("Live Market Dashboard failed to load")
        st.exception(error)
    finally:
        auto_refresh_enabled = st.session_state.get("live_chart_auto_refresh")
        _render_dashboard_state_diagnostics(bundle, chart_data=chart_data)
        _render_live_dashboard_debug_expander(
            bundle,
            chart_data=chart_data,
            bundle_loaded=bundle_loaded,
            auto_refresh_enabled=auto_refresh_enabled,
            refresh_interval_seconds=dashboard_live.DEFAULT_AUTO_REFRESH_INTERVAL_SECONDS,
        )


def _render_market_data_recorder() -> None:
    st.subheader("Market Data Recorder")
    st.caption(
        "Read-only public market data. No orders or wallet actions."
    )

    status = load_recorder_status()
    sizes = load_market_stream_sizes()

    if status is None:
        st.info(
            "No recorder session found. Run "
            "`python3 websocket_recorder.py --long-run --coins BTC ETH SOL` to start collecting."
        )
        return

    running = status.get("running", False)
    state = status.get("state", "STOPPED")
    conn_state = status.get("connection_state", "—")

    if state == "STALE_STATUS":
        st.warning(
            "Stale status — recorder process may have crashed. "
            "Run `python3 websocket_recorder.py --recover-latest`"
        )

    conn_label = "Connected" if conn_state == "CONNECTED" and running else (
        "Disconnected" if not running else conn_state
    )

    cols = st.columns(4)
    cols[0].metric("Status", conn_label if running else state)
    cols[1].metric("PID alive", "Yes" if status.get("pid_alive") else "No")
    cols[2].metric("Reconnects", status.get("reconnect_count", 0))
    cols[3].metric("Msgs/sec", f"{status.get('messages_per_second', 0):.2f}")

    hb_age = status.get("heartbeat_age_seconds")
    cols2 = st.columns(4)
    cols2[0].metric("Heartbeat age", f"{hb_age:.0f}s" if hb_age is not None else "—")
    cols2[1].metric("Long-run", "Yes" if status.get("long_run") else "No")
    cols2[2].metric("Live norm", "On" if status.get("live_normalization") else "Off")
    cols2[3].metric("Dropped norm", status.get("normalized_dropped_count", 0))

    cols3 = st.columns(4)
    cols3[0].metric("Malformed", status.get("malformed_count", 0))
    cols3[1].metric("Duplicates", status.get("duplicate_count", 0))
    cols3[2].metric("Gaps", status.get("gap_count", 0))
    cols3[3].metric(
        "Active Subs",
        f"{status.get('active_subscriptions', 0)}/{status.get('requested_subscriptions', '—')}",
    )

    session_id = status.get("recorder_session_id", "—")
    raw_mb = sizes.get("raw_bytes", 0) / (1024 * 1024)
    norm_mb = sizes.get("normalized_bytes", 0) / (1024 * 1024)
    st.caption(
        f"Session: `{session_id}` | "
        f"Coins: {', '.join(status.get('coins', []))} | "
        f"Last event: {status.get('last_message_at_ms', '—')} | "
        f"Raw: {raw_mb:.2f} MB | Normalized: {norm_mb:.2f} MB | "
        f"Degraded: {status.get('degraded', False)}"
    )

    if session_id and session_id != "—":
        checkpoint = dashboard_data.load_recorder_checkpoint(session_id)
        if checkpoint:
            st.caption(
                f"Latest checkpoint: {checkpoint.get('generated_at', '—')} | "
                f"Duration: {checkpoint.get('session_duration_seconds', 0):.0f}s"
            )
        if dashboard_data.load_recovery_report(session_id):
            recovery = dashboard_data.load_recovery_report(session_id)
            report_paths = (recovery or {}).get("report_paths", {})
            path_hint = report_paths.get("json") or f"data/market_stream/recovery_reports/{session_id}.json"
            st.caption(f"Recovery report: `{path_hint}`")
        norm_report = dashboard_data.load_normalization_report(session_id)
        if norm_report:
            st.caption(
                f"Normalization: {norm_report.get('events_written', 0)} events written"
            )

    channel_counts = status.get("channel_message_counts", {})
    if channel_counts:
        st.markdown("**Channel counts:**")
        channel_df = pd.DataFrame(
            [{"channel": k, "count": v} for k, v in sorted(channel_counts.items())]
        )
        st.dataframe(channel_df, use_container_width=True, hide_index=True)

    _render_quality_report_section()


def _render_quality_report_section() -> None:
    reports = load_quality_reports()
    if not reports:
        st.caption(
            "No quality reports yet. Run "
            "`python3 market_data_quality_report.py --latest` after a recorder session."
        )
        return

    st.markdown("**Data quality report**")
    report_options = {
        f"{r.get('session_id', 'unknown')[:8]}… ({r.get('quality_label', '—')})": r.get("session_id")
        for r in reports
        if r.get("session_id")
    }
    labels = list(report_options.keys())
    if not labels:
        st.warning("Quality reports found but missing session IDs.")
        return
    selected_label = st.selectbox("Latest quality reports", labels, index=0)
    session_id = report_options[selected_label]
    report = load_quality_report(session_id)
    if report is None:
        st.warning("Quality report file not found.")
        return

    qcols = st.columns(4)
    qcols[0].metric("Quality label", report.get("quality_label", "—"))
    qcols[1].metric("Warnings", report.get("warning_count", 0))
    gap_summary = report.get("gap_summary") or {}
    qcols[2].metric("Gaps", gap_summary.get("total_gaps", 0))
    qcols[3].metric("Generated", (report.get("generated_at") or "—")[:19])

    channel_labels = report.get("channel_quality_labels") or {}
    if channel_labels:
        st.markdown("**Channel quality labels:**")
        ch_label_df = pd.DataFrame(
            [{"channel": k, "quality_label": v} for k, v in sorted(channel_labels.items())]
        )
        st.dataframe(ch_label_df, use_container_width=True, hide_index=True)

    latency = report.get("latency_summaries") or {}
    if latency:
        st.markdown("**Latency by channel (ms):**")
        lat_rows = []
        for ch, stats in sorted(latency.items()):
            if not stats.get("latency_available", True):
                lat_rows.append(
                    {
                        "channel": ch,
                        "count": "—",
                        "median": "—",
                        "p95": "—",
                        "note": stats.get("note", "unavailable"),
                    }
                )
            else:
                lat_rows.append(
                    {
                        "channel": ch,
                        "count": stats.get("count", 0),
                        "median": stats.get("median_ms"),
                        "p95": stats.get("p95_ms"),
                        "note": stats.get("note") or (
                            "informational" if stats.get("informational_only") else ""
                        ),
                    }
                )
        st.dataframe(pd.DataFrame(lat_rows), use_container_width=True, hide_index=True)

    usability = report.get("usability_flags") or {}
    if usability:
        st.markdown("**Usability flags:**")
        flag_rows = [
            {
                "flag": name,
                "usable": flag.get("usable"),
                "reason": flag.get("reason"),
            }
            for name, flag in sorted(usability.items())
        ]
        st.dataframe(pd.DataFrame(flag_rows), use_container_width=True, hide_index=True)

    json_path = f"data/market_stream/quality_reports/{session_id}.json"
    md_path = f"data/market_stream/quality_reports/{session_id}.md"
    st.caption(f"Reports: `{json_path}` | `{md_path}`")

    _render_replay_reports_section()


def _render_replay_reports_section() -> None:
    reports = load_replay_reports()
    if not reports:
        st.caption(
            "No replay reports yet. Run "
            "`python3 market_data_replay.py --session-id <ID> --dry-run` after a recorder session."
        )
        return

    st.markdown("**Replay reports**")
    report_options = {
        f"{r['replay_session_id'][:8]}… ({r.get('total_events', 0)} events)": r["replay_session_id"]
        for r in reports
    }
    labels = list(report_options.keys())
    selected_label = st.selectbox("Latest replay reports", labels, index=0, key="replay_report_selector")
    replay_id = report_options[selected_label]
    report = load_replay_report(replay_id)
    if report is None:
        st.warning("Replay report file not found.")
        return

    rcols = st.columns(4)
    rcols[0].metric("Replay session", f"{replay_id[:12]}…")
    rcols[1].metric("Source session", f"{(report.get('source_recorder_session_id') or '—')[:12]}…")
    rcols[2].metric("Total events", report.get("total_events", 0))
    rcols[3].metric("Dry run", "Yes" if report.get("dry_run") else "No")

    rcols2 = st.columns(4)
    rcols2[0].metric("Quality label", report.get("quality_label", "—"))
    ordering_warnings = report.get("ordering_warnings") or []
    inherited = report.get("data_quality_warnings_inherited") or []
    rcols2[1].metric("Warnings", len(ordering_warnings) + len(inherited))
    rcols2[2].metric("Speed", report.get("speed", 1.0))
    rcols2[3].metric("Stream hash", f"{(report.get('stream_hash') or '—')[:12]}…")

    by_channel = report.get("event_counts_by_channel") or {}
    if by_channel:
        st.markdown("**Counts by channel:**")
        ch_df = pd.DataFrame(
            [{"channel": k, "count": v} for k, v in sorted(by_channel.items())]
        )
        st.dataframe(ch_df, use_container_width=True, hide_index=True)

    by_coin = report.get("event_counts_by_coin") or {}
    if by_coin:
        st.markdown("**Counts by coin:**")
        coin_df = pd.DataFrame(
            [{"coin": k, "count": v} for k, v in sorted(by_coin.items())]
        )
        st.dataframe(coin_df, use_container_width=True, hide_index=True)

    json_path = report.get("json_path") or f"data/market_stream/replays/{replay_id}.json"
    md_path = report.get("markdown_path") or f"data/market_stream/replays/{replay_id}.md"
    st.caption(f"Reports: `{json_path}` | `{md_path}`")

    _render_event_backtests_section()
    _render_feature_research_section()


def _render_event_backtests_section() -> None:
    reports = load_event_backtest_reports()
    if not reports:
        st.caption(
            "No event backtest reports yet. Run "
            "`python3 event_backtest.py --latest-replay --coins BTC ETH SOL "
            "--channels trades bbo --strategy contrarian_bbo_medium` "
            "(or `--strategy contrarian_bbo_confirmation_v1` for default medium profile)."
        )
        return

    st.markdown("**Event Backtests**")
    report_options = {
        f"{r['event_backtest_id'][:8]}… ({r.get('completed_trades', 0)} trades)": r[
            "event_backtest_id"
        ]
        for r in reports
    }
    labels = list(report_options.keys())
    selected_label = st.selectbox(
        "Latest event backtests", labels, index=0, key="event_backtest_selector"
    )
    backtest_id = report_options[selected_label]
    report = load_event_backtest_report(backtest_id)
    if report is None:
        st.warning("Event backtest report file not found.")
        return

    ecols = st.columns(4)
    ecols[0].metric("Strategy", report.get("strategy_name", "—"))
    ecols[1].metric("Profile", (report.get("profile_name") or "—")[:20])
    ecols[2].metric("Replay session", f"{(report.get('replay_session_id') or '—')[:12]}…")
    ecols[3].metric("Recorder session", f"{(report.get('source_recorder_session_id') or '—')[:12]}…")

    ecols2 = st.columns(4)
    ecols2[0].metric("Starting equity", _format_money(report.get("starting_equity")))
    ecols2[1].metric("Ending equity", _format_money(report.get("ending_equity")))
    ecols2[2].metric("Net return", _format_pct(report.get("net_return_pct")))
    ecols2[3].metric("Completed trades", report.get("completed_trades", 0))

    ecols3 = st.columns(4)
    ecols3[0].metric("Fees", _format_money(report.get("total_fees")))
    ecols3[1].metric("Slippage", _format_money(report.get("total_slippage_cost")))
    ecols3[2].metric("Max drawdown", _format_pct(report.get("maximum_drawdown_pct")))
    pf = report.get("profit_factor")
    ecols3[3].metric("Profit factor", f"{pf:.2f}" if pf is not None else "—")

    ecols4 = st.columns(3)
    ecols4[0].metric("BBO availability", _format_pct(report.get("bbo_availability_pct")))
    ecols4[1].metric("Evidence label", report.get("evidence_label", "—"))
    warnings = report.get("warnings") or []
    ecols4[2].metric("Warnings", len(warnings))

    if warnings:
        st.markdown("**Warnings:**")
        for warning in warnings[:10]:
            st.caption(f"- {warning}")

    strategy_name = report.get("strategy_name", "")
    if strategy_name == "contrarian_bbo_confirmation_v1":
        st.markdown("**BBO confirmation research strategy**")
        bbo_cols = st.columns(3)
        bbo_cols[0].caption(
            "Profiles: `contrarian_bbo_light`, `contrarian_bbo_medium`, "
            "`contrarian_bbo_strong`, `contrarian_bbo_rebound`"
        )
        blocked_count = report.get("blocked_signals", 0)
        blocked_reasons = report.get("blocked_reasons") or {}
        bbo_cols[1].caption(f"BBO blocked signals: {blocked_count}")
        if blocked_reasons:
            top_reason = max(blocked_reasons.items(), key=lambda x: x[1])
            bbo_cols[2].caption(f"Top blocked reason: {top_reason[0]} ({top_reason[1]})")
        adapter_notes = report.get("adapter_notes") or []
        for note in adapter_notes[:3]:
            st.caption(f"- {note}")

    json_path = report.get("json_path") or f"data/event_backtests/{backtest_id}.json"
    md_path = report.get("markdown_path") or f"data/event_backtests/{backtest_id}.md"
    st.caption(f"Reports: `{json_path}` | `{md_path}`")


def _render_feature_research_section() -> None:
    reports = load_feature_research_reports()
    if not reports:
        st.caption(
            "No feature research reports yet. Run "
            "`python3 feature_research.py --replay-session-id <ID> --coins BTC ETH SOL "
            "--channels trades bbo --horizons-seconds 10 30 60 300`."
        )
        return

    st.markdown("**Feature Research**")
    report_options = {
        f"{r['feature_research_id'][:8]}… ({r.get('row_count', 0)} rows)": r[
            "feature_research_id"
        ]
        for r in reports
    }
    labels = list(report_options.keys())
    selected_label = st.selectbox(
        "Latest feature research", labels, index=0, key="feature_research_selector"
    )
    research_id = report_options[selected_label]
    report = load_feature_research_report(research_id)
    if report is None:
        st.warning("Feature research report file not found.")
        return

    fcols = st.columns(4)
    fcols[0].metric("Replay session", f"{(report.get('replay_session_id') or '—')[:12]}…")
    fcols[1].metric("Recorder session", f"{(report.get('source_recorder_session_id') or '—')[:12]}…")
    fcols[2].metric("Row count", report.get("row_count", 0))
    fcols[3].metric("Horizons", len(report.get("horizons") or []))

    fcols2 = st.columns(4)
    missing = report.get("missing_data_summary") or {}
    total_rows = missing.get("total_rows") or report.get("row_count") or 0
    feature_missing = missing.get("features") or {}
    missing_rates = [info.get("missing_rate", 0) for info in feature_missing.values()]
    avg_missing = (sum(missing_rates) / len(missing_rates) * 100) if missing_rates else 0.0
    leakage = report.get("leakage_check") or {}
    possible = report.get("top_possible_signals") or []
    rejected = report.get("rejected_signals") or []

    fcols2[0].metric("Missing-data rate", f"{avg_missing:.1f}%")
    fcols2[1].metric("Leakage check", "PASS" if leakage.get("passed") else "FAIL")
    fcols2[2].metric("Possible signals", len(possible))
    fcols2[3].metric("Rejected signals", len(rejected))

    hash_val = report.get("deterministic_dataset_hash") or "—"
    st.caption(f"Deterministic hash: `{hash_val[:16]}…`" if len(hash_val) > 16 else f"Deterministic hash: `{hash_val}`")

    if possible:
        st.markdown("**Top possible signals:**")
        for sig in possible[:5]:
            st.caption(
                f"- {sig.get('feature')} @ {sig.get('horizon_seconds')}s — "
                f"corr={sig.get('correlation')}, {sig.get('reason')}"
            )

    if rejected:
        st.markdown("**Rejected / unstable signals:**")
        for sig in rejected[:5]:
            st.caption(
                f"- {sig.get('feature')} @ {sig.get('horizon_seconds')}s — "
                f"{sig.get('candidate_signal')}"
            )

    json_path = report.get("json_path") or f"data/feature_research/{research_id}.json"
    md_path = report.get("markdown_path") or f"data/feature_research/{research_id}.md"
    st.caption(f"Reports: `{json_path}` | `{md_path}`")


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


def _render_profile_banner(snapshot: dict | None) -> None:
    if snapshot is None:
        return

    paper = snapshot.get("paper_trader") or {}
    profile_id = paper.get("profile_id") or snapshot.get("strategy_profile", "")

    if profile_id != session.PAPER_TEST_PROFILE_SIZE_300:
        return

    st.info(
        "**PAPER TEST PROFILE:** `contrarian_v1_size_300_test`  \n"
        f"Trade size: ${paper.get('trade_size_usdc', 0):,.0f}  \n"
        f"Per-coin cap: ${paper.get('max_allocation_per_coin_usdc', 0):,.0f}  \n"
        f"Total exposure cap: ${paper.get('max_total_open_exposure_usdc', 0):,.0f}  \n"
        "Leverage: disabled"
    )


def _render_session_summary(snapshot: dict | None) -> None:
    st.subheader("Session Summary")
    _render_profile_banner(snapshot)

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
    coin = decision.get("coin", "—")
    recommendation = decision.get("recommendation", "—")
    color = _recommendation_color(recommendation)

    st.markdown(
        f"### {coin}  \n"
        f"<span style='color:{color}; font-size: 1.35em; font-weight: 700;'>"
        f"{recommendation}</span>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"**Signal:** {decision.get('signal', '—')} "
        f"({decision.get('signal_strength', '—')})"
    )
    st.markdown(f"**Risk:** {decision.get('risk_status', '—')}")
    st.markdown(f"**Reason:** {decision.get('primary_reason', '—')}")
    st.markdown(f"**Supporting:** {_supporting_reason(decision)}")
    st.markdown(
        f"**Confidence:** {decision.get('confidence', '—')} "
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
    decisions_by_coin = {
        item.get("coin"): item for item in decisions if item.get("coin")
    }

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


def _format_costs_pct(value: float | str | None) -> str:
    if value is None:
        return "—"
    if value == "N/A":
        return "N/A"
    return f"{value:.2f}%"


def _exit_strategy_profiles(profile_results: dict) -> list[str]:
    """Return exit experiment profiles present in an experiment."""
    known = (
        "contrarian_v1_entry_medium",
        "contrarian_v1_medium_signal_exit",
        "contrarian_v1_medium_tp25_sl25",
        "contrarian_v1_medium_tp35_sl25",
        "contrarian_v1_medium_tp50_sl30",
        "contrarian_v1_medium_tp35_sl25_maxhold",
        "hold_baseline",
    )
    return [profile_id for profile_id in known if profile_id in profile_results]


def _exit_summary_row(profile_id: str, payload: dict) -> dict:
    profile = payload.get("profile", {})
    thresholds = profile.get("thresholds", {})
    exit_config = profile.get("exit_config") or {}
    oos = payload.get("out_of_sample", {})
    dev = payload.get("development", {})
    comparison_label = payload.get("exit_comparison_label", "—")
    return {
        "Profile": profile_id,
        "DOWN (%)": thresholds.get("down_pct"),
        "STRONG_DOWN (%)": thresholds.get("strong_down_pct"),
        "Take-profit (%)": exit_config.get("take_profit_pct"),
        "Stop-loss (%)": exit_config.get("stop_loss_pct"),
        "Max hold (s)": exit_config.get("max_hold_seconds"),
        "Exit mode": exit_config.get("exit_mode", "legacy_signal"),
        "Dev return (%)": dev.get("net_return_pct"),
        "OOS return (%)": oos.get("net_return_pct"),
        "Completed trades": oos.get("completed_trades"),
        "TP exits": oos.get("take_profit_exits"),
        "SL exits": oos.get("stop_loss_exits"),
        "Avg net return/trade": oos.get("average_net_return_per_trade"),
        "Avg holding (s)": oos.get("average_holding_seconds"),
        "Profit factor": oos.get("profit_factor"),
        "Drawdown (%)": oos.get("maximum_drawdown_pct"),
        "Fees": oos.get("fees_paid"),
        "Slippage": oos.get("slippage_cost"),
        "Evidence": payload.get("evidence_assessment", "—"),
        "Comparison label": comparison_label,
    }


def _trend_filter_profiles(profile_results: dict) -> list[str]:
    """Return trend-filter experiment profiles present in an experiment."""
    known = (
        "contrarian_v1_entry_medium",
        "contrarian_v1_medium_no_trend_filter",
        "contrarian_v1_medium_trend30",
        "contrarian_v1_medium_trend50",
        "contrarian_v1_medium_worsening",
        "contrarian_v1_medium_combined",
        "hold_baseline",
    )
    return [profile_id for profile_id in known if profile_id in profile_results]


def _trend_filter_summary_row(profile_id: str, payload: dict) -> dict:
    profile = payload.get("profile", {})
    trend_filter = profile.get("trend_filter") or {}
    oos = payload.get("out_of_sample", {})
    dev = payload.get("development", {})
    combined = payload.get("combined", {})
    comparison_label = payload.get("trend_comparison_label", "—")
    warnings = payload.get("overfitting_warnings", [])
    return {
        "Profile": profile_id,
        "Filter type": trend_filter.get("filter_type", "NONE"),
        "Filter threshold (%)": trend_filter.get("five_minute_block_threshold_pct"),
        "Worsening rule": trend_filter.get("worsening_momentum_rule", "—")[:80] + "…"
        if trend_filter.get("worsening_momentum_rule")
        else "—",
        "Entry signals": oos.get("total_entry_signals"),
        "Entries executed": oos.get("entries_executed"),
        "Entries filtered": int(oos.get("entries_blocked_by_trend_filter", 0))
        + int(oos.get("entries_blocked_insufficient_history", 0)),
        "Filter (%)": oos.get("entry_filter_percentage"),
        "Dev return (%)": dev.get("net_return_pct"),
        "OOS return (%)": oos.get("net_return_pct"),
        "Combined return (%)": combined.get("net_return_pct"),
        "Completed trades": oos.get("completed_trades"),
        "Avg net return/trade": oos.get("average_net_return_per_trade"),
        "Profit factor": oos.get("profit_factor"),
        "Drawdown (%)": oos.get("maximum_drawdown_pct"),
        "Fees": oos.get("fees_paid"),
        "Slippage": oos.get("slippage_cost"),
        "Evidence": payload.get("evidence_assessment", "—"),
        "Comparison label": comparison_label,
        "Warnings": "; ".join(warnings) if warnings else "—",
    }


def _contrarian_threshold_profiles(profile_results: dict) -> list[str]:
    """Return contrarian entry-threshold profiles present in an experiment."""
    known = (
        "contrarian_v1_test",
        "contrarian_v1_entry_medium",
        "contrarian_v1_entry_selective",
    )
    return [profile_id for profile_id in known if profile_id in profile_results]


def _threshold_summary_row(profile_id: str, payload: dict) -> dict:
    profile = payload.get("profile", {})
    thresholds = profile.get("thresholds", {})
    oos = payload.get("out_of_sample", {})
    dev = payload.get("development", {})
    combined = payload.get("combined", {})
    return {
        "Profile": profile_id,
        "DOWN (%)": thresholds.get("down_pct"),
        "STRONG_DOWN (%)": thresholds.get("strong_down_pct"),
        "UP (%)": thresholds.get("up_pct"),
        "STRONG_UP (%)": thresholds.get("strong_up_pct"),
        "Dev return (%)": dev.get("net_return_pct"),
        "OOS return (%)": oos.get("net_return_pct"),
        "Combined return (%)": combined.get("net_return_pct"),
        "OOS completed trades": oos.get("completed_trades"),
        "OOS fees": oos.get("fees_paid"),
        "OOS slippage": oos.get("slippage_cost"),
        "OOS profit factor": oos.get("profit_factor"),
        "OOS max drawdown (%)": oos.get("maximum_drawdown_pct"),
        "Evidence": payload.get("evidence_assessment", "—"),
    }


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

    exit_profiles = _exit_strategy_profiles(profile_results)
    exit_comparison = experiment.get("exit_comparison")
    if exit_profiles:
        st.markdown("**Exit Strategy Comparison**")
        st.caption(
            "Exit targets can change trade duration and risk. "
            "Better historical results do not guarantee future performance."
        )
        if exit_comparison:
            for variant_id, summary in exit_comparison.get("comparisons", {}).items():
                if variant_id in profile_results:
                    profile_results[variant_id]["exit_comparison_label"] = summary.get(
                        "label", "—"
                    )
        exit_rows = [
            _exit_summary_row(profile_id, profile_results[profile_id])
            for profile_id in exit_profiles
        ]
        st.dataframe(pd.DataFrame(exit_rows), width="stretch", hide_index=True)

        if exit_comparison:
            st.caption(
                f"Baseline: {exit_comparison.get('baseline_profile')} — "
                "labels are descriptive only."
            )
            exit_comparison_rows = []
            for variant_id, summary in exit_comparison.get("comparisons", {}).items():
                exit_comparison_rows.append(
                    {
                        "Variant": variant_id,
                        "Label": summary.get("label"),
                        "OOS return diff (%)": summary.get(
                            "oos_return_difference_vs_baseline"
                        ),
                        "Trade diff": summary.get(
                            "completed_trade_difference_vs_baseline"
                        ),
                        "Avg net/trade diff": summary.get(
                            "average_net_return_per_trade_difference_vs_baseline"
                        ),
                        "Profit factor diff": summary.get(
                            "profit_factor_difference_vs_baseline"
                        ),
                        "Drawdown diff (%)": summary.get(
                            "maximum_drawdown_difference_vs_baseline"
                        ),
                        "Cost diff": summary.get("total_cost_difference_vs_baseline"),
                        "Holding time diff (s)": summary.get(
                            "average_holding_time_difference_vs_baseline"
                        ),
                        "SL exit diff": summary.get(
                            "stop_loss_exit_difference_vs_baseline"
                        ),
                        "TP exit diff": summary.get(
                            "take_profit_exit_difference_vs_baseline"
                        ),
                    }
                )
            st.dataframe(
                pd.DataFrame(exit_comparison_rows),
                width="stretch",
                hide_index=True,
            )

    trend_profiles = _trend_filter_profiles(profile_results)
    trend_comparison = experiment.get("trend_comparison")
    if trend_profiles:
        st.markdown("**Trend Filter Comparison**")
        st.caption(
            "Trend filters may avoid some falling markets, but they can also block "
            "valid rebounds."
        )
        if trend_comparison:
            for variant_id, summary in trend_comparison.get("comparisons", {}).items():
                if variant_id in profile_results:
                    profile_results[variant_id]["trend_comparison_label"] = summary.get(
                        "label", "—"
                    )
        trend_rows = [
            _trend_filter_summary_row(profile_id, profile_results[profile_id])
            for profile_id in trend_profiles
        ]
        st.dataframe(pd.DataFrame(trend_rows), width="stretch", hide_index=True)

        if trend_comparison:
            st.caption(
                f"Baseline: {trend_comparison.get('baseline_profile')} — "
                "labels are descriptive only."
            )
            trend_comparison_rows = []
            for variant_id, summary in trend_comparison.get("comparisons", {}).items():
                trend_comparison_rows.append(
                    {
                        "Variant": variant_id,
                        "Label": summary.get("label"),
                        "OOS return diff (%)": summary.get(
                            "oos_return_difference_vs_baseline"
                        ),
                        "Trade diff": summary.get(
                            "completed_trade_difference_vs_baseline"
                        ),
                        "Blocked entries diff": summary.get(
                            "blocked_entry_count_vs_baseline"
                        ),
                        "Filter pct diff": summary.get("filter_percentage_vs_baseline"),
                        "Avg net/trade diff": summary.get(
                            "average_net_return_per_trade_difference_vs_baseline"
                        ),
                        "Profit factor diff": summary.get(
                            "profit_factor_difference_vs_baseline"
                        ),
                        "Drawdown diff (%)": summary.get(
                            "maximum_drawdown_difference_vs_baseline"
                        ),
                        "Fee diff": summary.get("fee_difference_vs_baseline"),
                        "Slippage diff": summary.get("slippage_difference_vs_baseline"),
                    }
                )
            st.dataframe(
                pd.DataFrame(trend_comparison_rows),
                width="stretch",
                hide_index=True,
            )

    contrarian_profiles = _contrarian_threshold_profiles(profile_results)
    if contrarian_profiles:
        st.markdown("**Contrarian entry thresholds (side by side)**")
        st.caption(
            "Wider thresholds may reduce trade frequency, but fewer trades do not "
            "automatically mean better performance."
        )
        threshold_rows = [
            _threshold_summary_row(profile_id, profile_results[profile_id])
            for profile_id in contrarian_profiles
        ]
        st.dataframe(pd.DataFrame(threshold_rows), width="stretch", hide_index=True)

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
                "OOS costs % of gross": oos.get("costs_pct_of_gross_profit"),
            }
        )

    st.markdown("**Profile comparison**")
    st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)

    threshold_comparison = experiment.get("threshold_comparison")
    if threshold_comparison:
        st.markdown("**Threshold comparison vs contrarian_v1_test**")
        st.caption(
            "Labels are descriptive only — no automatic winner is declared. "
            "See label rules in the experiment JSON."
        )
        comparison_summary_rows = []
        for variant_id, summary in threshold_comparison.get("comparisons", {}).items():
            comparison_summary_rows.append(
                {
                    "Variant": variant_id,
                    "Label": summary.get("label"),
                    "Trade reduction": summary.get("trade_count_reduction_vs_baseline"),
                    "Fee reduction": summary.get("fee_reduction_vs_baseline"),
                    "Slippage reduction": summary.get("slippage_reduction_vs_baseline"),
                    "OOS return diff (%)": summary.get("oos_return_difference_vs_baseline"),
                    "Profit factor diff": summary.get("profit_factor_difference_vs_baseline"),
                    "Max drawdown diff (%)": summary.get(
                        "maximum_drawdown_difference_vs_baseline"
                    ),
                }
            )
        st.dataframe(
            pd.DataFrame(comparison_summary_rows),
            width="stretch",
            hide_index=True,
        )

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

            detail_cols = st.columns(5)
            detail_cols[0].metric("Max drawdown", _format_pct(metrics.get("maximum_drawdown_pct")))
            detail_cols[1].metric(
                "Profit factor",
                _format_profit_factor(metrics.get("profit_factor")),
            )
            detail_cols[2].metric(
                "Costs % of gross",
                _format_costs_pct(metrics.get("costs_pct_of_gross_profit")),
            )
            detail_cols[3].metric(
                "Longest losing seq.",
                str(metrics.get("longest_losing_sequence", 0)),
            )
            detail_cols[4].metric(
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

    test_mode = results.get("test_mode") or {}
    production_mode = results.get("production_mode") or {}
    test_block = test_mode.get("after_forced_close") or {}
    production_block = production_mode.get("after_forced_close") or {}

    if not test_block or not production_block:
        st.warning("Backtest results file is missing expected mode blocks.")
        return

    test_dd = test_block.get("maximum_drawdown_pct")
    prod_dd = production_block.get("maximum_drawdown_pct")
    max_drawdown = max(
        test_dd if test_dd is not None else 0.0,
        prod_dd if prod_dd is not None else 0.0,
    )

    summary_cols = st.columns(5)
    summary_cols[0].metric(
        "Test-Mode Return",
        _format_pct(test_block.get("total_return_pct")),
    )
    summary_cols[1].metric(
        "Production Return",
        _format_pct(production_block.get("total_return_pct")),
    )
    summary_cols[2].metric("Maximum Drawdown", _format_pct(max_drawdown))
    summary_cols[3].metric(
        "Fees Paid (Test)",
        _format_money(test_block.get("total_fees_paid")),
        help="Test-mode simulation fees. See details for production.",
    )
    summary_cols[4].metric(
        "Completed Trades (Test)",
        str(test_block.get("completed_trades", 0)),
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

    with tab_dashboard:
        _render_dashboard()

    with tab_review:
        dashboard_review.render_review_mode(
            load_experiment=load_experiment,
            list_experiments_fn=list_experiments,
        )


def _render_dashboard() -> None:
    _render_live_market_dashboard()
    st.divider()

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
    _render_market_data_recorder()

    st.divider()
    prices = load_price_log()
    _render_price_section(prices)

    st.divider()
    _render_experiment_analysis()

    st.divider()
    _render_backtest_results()


if __name__ == "__main__":
    main()
