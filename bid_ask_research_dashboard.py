"""
Hyperliquid Bid/Ask ML Research Lab — read-only visual research dashboard.

Paper-only, research-only. Displays pipeline progress, data quality, walk-forward
results, candidate trade-offs, and paper decisions from local JSON/CSV artifacts.
No execution, no AWS, no writes, no subprocesses.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT
from bid_ask_validator import bid_ask_csv_path

try:
    import altair as alt

    _HAS_ALTAIR = True
except ImportError:
    _HAS_ALTAIR = False

# ---------------------------------------------------------------------------
# Safety constants — always enforced in the UI
# ---------------------------------------------------------------------------
REAL_EXECUTION_READY = False
DATA_ROOT = Path("data/bid_ask_csv")
PHASE3_AUDIT_PATH = Path("reports/bid_ask_research_scheduler/phase3_audit_latest.json")
WALK_FORWARD_DIR = Path("reports/bid_ask_walk_forward")
CANDIDATE_VALIDATION_DIR = Path("reports/bid_ask_candidate_validation")
RESEARCH_COINS = ("BTC", "ETH")
HISTORICAL_SOURCE = "historical"
PHASE3_TARGET_HOURS = 14 * 24

CANDIDATE_B_STRICT_REPORT = WALK_FORWARD_DIR / "walk_forward_BTC_20260706T115051.json"
CANDIDATE_B_RELAXED_25_REPORT = WALK_FORWARD_DIR / "walk_forward_BTC_20260706T142556.json"
CANDIDATE_B_RELAXED_62_REPORT = WALK_FORWARD_DIR / "walk_forward_BTC_20260706T120135.json"

CANDIDATE_B_FALLBACKS: tuple[dict[str, Any], ...] = (
    {"config": "Strict Candidate B", "trades": 5, "cumulative_net_bps": 3.84},
    {"config": "Relaxed 25-trade", "trades": 25, "cumulative_net_bps": -101.82},
    {"config": "Relaxed 62-trade", "trades": 62, "cumulative_net_bps": -267.95},
)

MILESTONES: tuple[tuple[str, int, str], ...] = (
    ("Data Pipeline Works", 80, "Historical 1s CSVs advancing for BTC and ETH."),
    ("Paper Backtest Engine", 85, "Walk-forward evaluator producing paper trade reports."),
    ("First Candidate Found", 40, "Candidate B is positive (+3.84 bps) but only 5 trades."),
    ("Candidate Scaling", 20, "Relaxing filters adds trades but destroys post-cost returns."),
    ("Robustness", 15, "Limited windows; concentrated time buckets flagged."),
    ("Out-of-Sample Paper Test", 0, "Not started — waiting for a frozen candidate."),
    ("Paper Operations", 70, "Paper decision viewer and validation loop active."),
    ("Market Ready", 0, "Real execution disabled — not market ready."),
)

ARCHITECTURE_DIAGRAM = """
**Hyperliquid S3 / WebSocket**
→ **Raw L2 Data**
→ **Bid/Ask Importer**
→ **1s CSV**
→ **15s / 1m Resampler**
→ **Validator**
→ **Feature Builder**
→ **Walk-Forward Evaluator**
→ **Candidate Validator**
→ **Paper Decision Dashboard** ← *you are here*
"""

NEXT_RESEARCH_PLAN = [
    "Finish BTC/ETH 14-day sprint",
    "Inspect losing trades",
    "Test horizons 10 / 15 / 20 / 30 / 40",
    "Compare BTC vs ETH",
    "Build candidate grid",
    "Freeze promising config",
    "Out-of-sample test",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _parse_report_stamp(path: Path) -> str:
    stem = path.stem
    if stem.startswith("walk_forward_"):
        return stem.rsplit("_", 1)[-1]
    if "_validation_" in stem:
        return stem.rsplit("_validation_", 1)[-1]
    return stem


def _fmt_ts_ms(ts_ms: int | None) -> str:
    if ts_ms is None:
        return "—"
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open(encoding="utf-8") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except OSError:
        return 0


def _latest_validation_report(coin: str, interval: str) -> dict[str, Any] | None:
    reports_dir = DATA_ROOT / "reports"
    if not reports_dir.is_dir():
        return None
    pattern = f"{coin}_{interval}_validation_*.json"
    candidates = sorted(reports_dir.glob(pattern), key=_parse_report_stamp, reverse=True)
    for path in candidates:
        payload = _load_json(path)
        if payload is not None:
            payload["_report_path"] = str(path)
            return payload
    return None


def _latest_repair_report(coin: str, interval: str) -> dict[str, Any] | None:
    reports_dir = DATA_ROOT / "reports"
    if not reports_dir.is_dir():
        return None
    pattern = f"{coin}_{interval}_repair_*.json"
    candidates = sorted(reports_dir.glob(pattern), key=_parse_report_stamp, reverse=True)
    for path in candidates:
        payload = _load_json(path)
        if payload is not None:
            payload["_report_path"] = str(path)
            return payload
    return None


def _latest_candidate_validation() -> dict[str, Any] | None:
    if not CANDIDATE_VALIDATION_DIR.is_dir():
        return None
    candidates = sorted(
        CANDIDATE_VALIDATION_DIR.glob("candidate_validation_*.json"),
        key=_parse_report_stamp,
        reverse=True,
    )
    for path in candidates:
        payload = _load_json(path)
        if payload is not None:
            payload["_report_path"] = str(path)
            return payload
    return None


def _latest_walk_forward_for_coin(coin: str) -> dict[str, Any] | None:
    if not WALK_FORWARD_DIR.is_dir():
        return None
    pattern = f"walk_forward_{coin}_*.json"
    candidates = sorted(WALK_FORWARD_DIR.glob(pattern), key=_parse_report_stamp, reverse=True)
    for path in candidates:
        payload = _load_json(path)
        if payload is not None:
            payload["_report_path"] = str(path)
            payload["_report_name"] = path.name
            return payload
    return None


def _walk_forward_summaries(limit: int = 12) -> list[dict[str, Any]]:
    if not WALK_FORWARD_DIR.is_dir():
        return []
    reports = sorted(WALK_FORWARD_DIR.glob("walk_forward_*.json"), key=_parse_report_stamp, reverse=True)[:limit]
    rows: list[dict[str, Any]] = []
    for path in reports:
        payload = _load_json(path)
        if payload is None:
            continue
        coins = payload.get("coins") or []
        coin = coins[0] if coins else "—"
        rows.append(
            {
                "stamp": _parse_report_stamp(path),
                "coin": coin,
                "interval": payload.get("interval"),
                "horizon": payload.get("horizon"),
                "windows": payload.get("completed_windows"),
                "model_accuracy": payload.get("model_accuracy_mean"),
                "baseline_accuracy": payload.get("baseline_accuracy_mean"),
                "accuracy_edge": payload.get("accuracy_edge_vs_baseline"),
                "cumulative_net_bps": payload.get("cumulative_net_return_bps"),
                "accepted_trades": payload.get("filtered_paper_trades") or payload.get("accepted_trades_count"),
                "positive_after_costs": payload.get("positive_after_costs"),
            }
        )
    return rows


def _walk_forward_summary_from_path(path: Path) -> dict[str, Any] | None:
    payload = _load_json(path)
    if payload is None:
        return None
    return {
        "exists": True,
        "coin": (payload.get("coins") or ["—"])[0],
        "windows": payload.get("completed_windows"),
        "model_accuracy": payload.get("model_accuracy_mean"),
        "baseline_accuracy": payload.get("baseline_accuracy_mean"),
        "accuracy_edge": payload.get("accuracy_edge_vs_baseline"),
        "cumulative_net_bps": payload.get("cumulative_net_return_bps"),
        "accepted_trades": payload.get("filtered_paper_trades"),
        "positive_after_costs": payload.get("positive_after_costs"),
        "min_confidence": payload.get("min_confidence"),
        "min_edge_bps": payload.get("min_edge_bps"),
    }


def _latest_accepted_trades_csv() -> Path | None:
    candidate = _latest_candidate_validation()
    if candidate:
        csv_path = candidate.get("accepted_trades_csv_path")
        if csv_path and Path(csv_path).is_file():
            return Path(csv_path)
    if not WALK_FORWARD_DIR.is_dir():
        return None
    csv_files = sorted(WALK_FORWARD_DIR.glob("accepted_trades_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csv_files[0] if csv_files else None


def _load_paper_decisions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    display_cols = [
        column
        for column in (
            "timestamp_iso",
            "coin",
            "predicted_class",
            "confidence",
            "future_return_bps",
            "cost_bps",
            "net_return_bps",
        )
        if column in df.columns
    ]
    if "timestamp_iso" not in display_cols and "timestamp_ms" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        display_cols = ["timestamp"] + [c for c in display_cols if c != "timestamp_iso"]
    out = df[display_cols].copy() if display_cols else df.copy()
    out["decision"] = "PAPER_ACCEPTED"
    rename = {"timestamp_iso": "timestamp"}
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    return out


def _coin_coverage(coin: str, phase3: dict[str, Any] | None) -> dict[str, Any]:
    imported = (phase3 or {}).get("imported_historical_1s") or {}
    coin_data = imported.get(coin) or {}
    resampled = ((phase3 or {}).get("resampled_coverage") or {}).get(coin) or {}
    rows_1s = coin_data.get("row_count") or _count_csv_rows(
        bid_ask_csv_path(coin, "1s", DATA_ROOT, source=HISTORICAL_SOURCE)
    )
    rows_15s = (resampled.get("15s") or {}).get("row_count") or _count_csv_rows(
        bid_ask_csv_path(coin, "15s", DATA_ROOT, source=HISTORICAL_SOURCE)
    )
    return {
        "rows_1s": rows_1s,
        "rows_15s": rows_15s,
        "first_ms": coin_data.get("first_timestamp_ms"),
        "last_ms": coin_data.get("last_timestamp_ms"),
        "last_ts": coin_data.get("last_timestamp") or _fmt_ts_ms(coin_data.get("last_timestamp_ms")),
        "coverage_hours": coin_data.get("coverage_hours"),
    }


def _data_quality_row(coin: str, interval: str) -> dict[str, Any]:
    validation = _latest_validation_report(coin, interval)
    repair = _latest_repair_report(coin, interval) if interval == "1s" else None
    path = bid_ask_csv_path(coin, interval, DATA_ROOT, source=HISTORICAL_SOURCE)
    row_count = validation.get("row_count") if validation else _count_csv_rows(path)
    monotonic = repair.get("monotonic_after") if repair else None
    if monotonic is None and validation:
        warnings = validation.get("warnings") or []
        monotonic = not any("non-monotonic" in str(w).lower() for w in warnings)
    warnings = (validation or {}).get("warnings") or []
    missing_warning = next((w for w in warnings if "missing" in str(w).lower() or "incomplete" in str(w).lower()), None)
    return {
        "coin": coin,
        "interval": interval,
        "rows": row_count,
        "validation": validation.get("passed") if validation else None,
        "duplicates": validation.get("duplicate_timestamp_count") if validation else None,
        "monotonic": monotonic,
        "missing_warning": missing_warning or "—",
    }


def _collection_rows_df(phase3: dict[str, Any] | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for coin in RESEARCH_COINS:
        cov = _coin_coverage(coin, phase3)
        for interval, count in (("1s", cov["rows_1s"]), ("15s", cov["rows_15s"])):
            rows.append({"coin": coin, "interval": interval, "label": f"{coin} {interval}", "rows": count})
    return pd.DataFrame(rows)


def _coverage_timeline_df(phase3: dict[str, Any] | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for coin in RESEARCH_COINS:
        cov = _coin_coverage(coin, phase3)
        if cov["first_ms"] and cov["last_ms"]:
            rows.append(
                {
                    "coin": coin,
                    "start": pd.to_datetime(cov["first_ms"], unit="ms", utc=True),
                    "end": pd.to_datetime(cov["last_ms"], unit="ms", utc=True),
                    "hours": cov["coverage_hours"] or 0,
                }
            )
    return pd.DataFrame(rows)


def _phase3_imported_hours(phase3: dict[str, Any] | None) -> float:
    if not phase3:
        return 0.0
    imported = phase3.get("imported_historical_1s") or {}
    hours = [float((imported.get(coin) or {}).get("coverage_hours") or 0) for coin in RESEARCH_COINS]
    hours = [h for h in hours if h > 0]
    return sum(hours) / len(hours) if hours else 0.0


def _candidate_b_rows() -> list[dict[str, Any]]:
    scenarios = [
        ("Strict Candidate B", CANDIDATE_B_STRICT_REPORT, CANDIDATE_B_FALLBACKS[0]),
        ("Relaxed 25-trade", CANDIDATE_B_RELAXED_25_REPORT, CANDIDATE_B_FALLBACKS[1]),
        ("Relaxed 62-trade", CANDIDATE_B_RELAXED_62_REPORT, CANDIDATE_B_FALLBACKS[2]),
    ]
    rows: list[dict[str, Any]] = []
    for label, path, fallback in scenarios:
        summary = _walk_forward_summary_from_path(path)
        if summary:
            rows.append(
                {
                    "config": label,
                    "trades": summary.get("accepted_trades") or fallback["trades"],
                    "cumulative_net_bps": summary.get("cumulative_net_bps") or fallback["cumulative_net_bps"],
                    "source": "report",
                }
            )
        else:
            rows.append(
                {
                    "config": label,
                    "trades": fallback["trades"],
                    "cumulative_net_bps": fallback["cumulative_net_bps"],
                    "source": "fallback",
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def _render_bar_chart(df: pd.DataFrame, x: str, y: str, title: str, *, color: str | None = None) -> None:
    import streamlit as st

    if df.empty:
        st.info("Not available yet")
        return
    if _HAS_ALTAIR:
        enc: dict[str, Any] = {"x": alt.X(f"{x}:N", title=x), "y": alt.Y(f"{y}:Q", title=y)}
        if color and color in df.columns:
            enc["color"] = alt.Color(f"{color}:N", legend=alt.Legend(title=color))
        chart = alt.Chart(df).mark_bar().encode(**enc).properties(title=title, height=280)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption(title)
        chart_df = df.set_index(x)[[y]] if color is None else df
        st.bar_chart(chart_df if color is None else df.set_index(x)[y])


def _render_grouped_accuracy_chart(df: pd.DataFrame, title: str) -> None:
    import streamlit as st

    if df.empty:
        st.info("Not available yet")
        return
    melted = df.melt(id_vars=["coin"], value_vars=["model_accuracy", "baseline_accuracy"], var_name="metric", value_name="accuracy")
    melted["metric"] = melted["metric"].map({"model_accuracy": "Model", "baseline_accuracy": "Baseline"})
    if _HAS_ALTAIR:
        chart = (
            alt.Chart(melted)
            .mark_bar()
            .encode(
                x=alt.X("coin:N", title="Coin"),
                y=alt.Y("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("metric:N", legend=alt.Legend(title="Metric")),
                xOffset="metric:N",
            )
            .properties(title=title, height=280)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption(title)
        st.bar_chart(df.set_index("coin")[["model_accuracy", "baseline_accuracy"]])


def _render_line_chart(df: pd.DataFrame, x: str, y: str, title: str, *, series: str | None = None) -> None:
    import streamlit as st

    if df.empty or len(df) < 2:
        st.info("Not enough walk-forward runs for a trend chart yet.")
        return
    if _HAS_ALTAIR:
        enc: dict[str, Any] = {
            "x": alt.X(f"{x}:O", title="Run"),
            "y": alt.Y(f"{y}:Q", title=y),
        }
        if series and series in df.columns:
            enc["color"] = alt.Color(f"{series}:N", legend=alt.Legend(title=series))
        chart = alt.Chart(df).mark_line(point=True).encode(**enc).properties(title=title, height=280)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption(title)
        st.line_chart(df.set_index(x)[y])


def _render_timeline_chart(df: pd.DataFrame, title: str) -> None:
    import streamlit as st

    if df.empty:
        st.info("Coverage timeline not available yet.")
        return
    if _HAS_ALTAIR:
        chart = (
            alt.Chart(df)
            .mark_bar(size=18)
            .encode(
                y=alt.Y("coin:N", title="Coin"),
                x=alt.X("start:T", title="Coverage window"),
                x2="end:T",
                tooltip=["coin", "start", "end", "hours"],
            )
            .properties(title=title, height=160)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption(title)
        st.bar_chart(df.set_index("coin")[["hours"]])


def _render_histogram(series: pd.Series, title: str) -> None:
    import streamlit as st

    if series.empty:
        st.info("Not available yet")
        return
    counts, bins = pd.cut(series, bins=min(12, max(4, len(series) // 2)), retbins=True)
    hist = counts.value_counts().sort_index()
    hist_df = pd.DataFrame({"bin": hist.index.astype(str), "count": hist.values})
    if _HAS_ALTAIR:
        chart = (
            alt.Chart(hist_df)
            .mark_bar()
            .encode(x=alt.X("bin:N", title="net_return_bps bucket"), y=alt.Y("count:Q", title="Trades"))
            .properties(title=title, height=240)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption(title)
        st.bar_chart(hist_df.set_index("bin"))


def _render_scatter_chart(df: pd.DataFrame, x: str, y: str, title: str, label: str) -> None:
    import streamlit as st

    if df.empty:
        st.info("Not available yet")
        return
    if _HAS_ALTAIR:
        chart = (
            alt.Chart(df)
            .mark_circle(size=120)
            .encode(
                x=alt.X(f"{x}:Q", title=x),
                y=alt.Y(f"{y}:Q", title=y),
                color=alt.Color("config:N"),
                tooltip=["config", x, y],
            )
            .properties(title=title, height=280)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption(title)
        st.scatter_chart(df, x=x, y=y, color=label)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------
def _render_header() -> None:
    import streamlit as st

    st.markdown("# Hyperliquid Bid/Ask ML Research Lab")
    st.markdown(
        "*Paper-only Web3 market data research system for bid/ask microstructure modeling.*"
    )
    st.error(
        "**Paper-only** | **Real execution disabled** | **Not market ready** | "
        "Simulated returns are **not** profitability proof."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("paper_only", str(PAPER_ONLY).lower())
    c2.metric("execution_enabled", str(EXECUTION_ENABLED).lower())
    c3.metric("real_execution_ready", str(REAL_EXECUTION_READY).lower())
    c4.metric("market_ready", "false")
    st.caption(SAFETY_STATEMENT)


def _render_kpi_cards(phase3: dict[str, Any] | None, candidate: dict[str, Any] | None) -> None:
    import streamlit as st

    btc = _coin_coverage("BTC", phase3)
    eth = _coin_coverage("ETH", phase3)
    downloads_total = (phase3 or {}).get("downloads", {}).get("total_files", "—")
    cand_trades = (candidate or {}).get("filtered_paper_trades")
    if cand_trades is None:
        strict = _walk_forward_summary_from_path(CANDIDATE_B_STRICT_REPORT)
        cand_trades = (strict or {}).get("accepted_trades")
    cand_bps = (candidate or {}).get("cumulative_net_return_bps")
    if cand_bps is None:
        strict = _walk_forward_summary_from_path(CANDIDATE_B_STRICT_REPORT)
        cand_bps = (strict or {}).get("cumulative_net_bps")
    status = "not market ready"
    if cand_trades is not None and cand_bps is not None:
        status = f"{cand_trades} trades, {cand_bps:+.2f} bps"

    cols = st.columns(4)
    cols[0].metric("BTC 1s rows", f"{btc['rows_1s']:,}")
    cols[1].metric("ETH 1s rows", f"{eth['rows_1s']:,}")
    cols[2].metric("BTC 15s rows", f"{btc['rows_15s']:,}")
    cols[3].metric("ETH 15s rows", f"{eth['rows_15s']:,}")

    cols2 = st.columns(4)
    cols2[0].metric("Latest BTC timestamp", btc["last_ts"])
    cols2[1].metric("Latest ETH timestamp", eth["last_ts"])
    cols2[2].metric("Backfill downloads total", downloads_total)
    cols2[3].metric("Candidate B status", status)


def _render_data_collection(phase3: dict[str, Any] | None) -> None:
    import streamlit as st

    st.markdown("## Data Collection Progress")

    rows_df = _collection_rows_df(phase3)
    timeline_df = _coverage_timeline_df(phase3)
    imported_hours = _phase3_imported_hours(phase3)
    progress = min(imported_hours / PHASE3_TARGET_HOURS, 1.0) if PHASE3_TARGET_HOURS else 0.0

    left, right = st.columns(2)
    with left:
        _render_bar_chart(rows_df, "label", "rows", "Rows by coin and interval")
    with right:
        _render_timeline_chart(timeline_df, "BTC / ETH coverage window (1s historical)")

    st.markdown("**Phase 3 — 14-day sprint**")
    st.progress(progress, text=f"{imported_hours:.1f}h imported / {PHASE3_TARGET_HOURS}h target ({progress:.0%})")

    cfg = (phase3 or {}).get("phase_config") or {}
    downloads = (phase3 or {}).get("downloads") or {}
    m1, m2 = st.columns(2)
    m1.metric("Downloads total files", downloads.get("total_files", "Not available yet"))
    m2.metric("Effective daily file capacity", cfg.get("effective_daily_file_capacity", "Not available yet"))

    if phase3 is None:
        st.warning("Phase 3 audit report not available yet — showing CSV-based estimates where possible.")


def _render_data_quality() -> None:
    import streamlit as st

    st.markdown("## Data Quality")
    st.markdown(
        "Data quality matters because ML results are meaningless if timestamps are duplicated, "
        "unsorted, or polluted by mock rows."
    )

    rows = [_data_quality_row(coin, interval) for coin in RESEARCH_COINS for interval in ("1s", "15s")]
    if not rows:
        st.info("Not available yet")
        return

    badge_cols = st.columns(len(rows))
    for col, row in zip(badge_cols, rows):
        passed = row["validation"]
        if passed is True:
            badge = "✅ PASS"
        elif passed is False:
            badge = "❌ FAIL"
        else:
            badge = "⏳ N/A"
        col.markdown(f"**{row['coin']} {row['interval']}**  \n{badge}")

    table = pd.DataFrame(rows)
    display = table.copy()
    display["validation"] = display["validation"].map(
        lambda v: "PASS" if v is True else ("FAIL" if v is False else "—")
    )
    display["monotonic"] = display["monotonic"].map(
        lambda v: "yes" if v is True else ("no" if v is False else "—")
    )
    display = display.rename(
        columns={
            "validation": "validation_status",
            "duplicates": "duplicate_timestamps",
            "missing_warning": "missing_incomplete_bars",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_walk_forward() -> None:
    import streamlit as st

    st.markdown("## Model / Walk-Forward Testing")
    st.caption("Research-only simulated performance — not live trading.")

    latest_by_coin: list[dict[str, Any]] = []
    for coin in RESEARCH_COINS:
        report = _latest_walk_forward_for_coin(coin)
        if report:
            latest_by_coin.append(
                {
                    "coin": coin,
                    "model_accuracy": report.get("model_accuracy_mean"),
                    "baseline_accuracy": report.get("baseline_accuracy_mean"),
                    "accuracy_edge": report.get("accuracy_edge_vs_baseline"),
                    "windows": report.get("completed_windows"),
                    "cumulative_net_bps": report.get("cumulative_net_return_bps"),
                    "accepted_trades": report.get("filtered_paper_trades") or report.get("accepted_trades_count"),
                    "positive_after_costs": report.get("positive_after_costs"),
                }
            )

    if not latest_by_coin:
        st.info("No walk-forward reports found yet.")
        return

    accuracy_df = pd.DataFrame(latest_by_coin)
    _render_grouped_accuracy_chart(accuracy_df, "Latest model accuracy vs baseline")

    summaries = _walk_forward_summaries(limit=10)
    if summaries:
        runs_df = pd.DataFrame(summaries).iloc[::-1].reset_index(drop=True)
        runs_df["run_label"] = runs_df.apply(
            lambda r: f"{r['coin']} H{r['horizon']}" if r.get("horizon") else str(r["coin"]),
            axis=1,
        )
        left, right = st.columns(2)
        with left:
            _render_bar_chart(
                runs_df,
                "run_label",
                "cumulative_net_bps",
                "Cumulative net return (bps) — recent runs",
                color="coin",
            )
        with right:
            _render_line_chart(runs_df, "stamp", "cumulative_net_bps", "Cumulative net return trend", series="coin")

    mcols = st.columns(4)
    primary = latest_by_coin[0]
    mcols[0].metric("Windows completed", primary.get("windows", "—"))
    edge = primary.get("accuracy_edge")
    mcols[1].metric("Accuracy edge", f"{edge:.3f}" if edge is not None else "—")
    mcols[2].metric(
        "Positive after costs",
        "yes" if primary.get("positive_after_costs") else "no",
    )
    mcols[3].metric("Accepted trades", primary.get("accepted_trades", "—"))


def _render_candidate_b() -> None:
    import streamlit as st

    st.markdown("## Candidate B Trade-Off")
    st.caption("Strict vs relaxed filter experiments on BTC DOWN microstructure.")

    rows = _candidate_b_rows()
    df = pd.DataFrame(rows)

    left, right = st.columns(2)
    with left:
        _render_bar_chart(df, "config", "trades", "Trades by config")
    with right:
        _render_bar_chart(df, "config", "cumulative_net_bps", "Cumulative net return (bps) by config")

    _render_scatter_chart(df, "trades", "cumulative_net_bps", "Trades vs cumulative net return", "config")

    display = df[["config", "trades", "cumulative_net_bps"]].copy()
    display["cumulative_net_bps"] = display["cumulative_net_bps"].map(lambda v: f"{v:+.2f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.info(
        "**Main takeaway:** Relaxing filters increases trade count but destroys performance. "
        "The current blocker is finding a candidate that scales."
    )


def _render_paper_decisions() -> None:
    import streamlit as st

    st.markdown("## Live Paper Decision Viewer")
    st.warning("These are **paper decisions only**. No real orders were placed.")

    csv_path = _latest_accepted_trades_csv()
    if csv_path is None:
        st.info("No accepted trades CSV found yet.")
        return

    try:
        decisions = _load_paper_decisions(csv_path)
    except (OSError, pd.errors.EmptyDataError, ValueError) as exc:
        st.error(f"Could not read paper decisions: {exc}")
        return

    if "net_return_bps" in decisions.columns:
        _render_histogram(decisions["net_return_bps"].dropna(), "Paper trade net_return_bps distribution")

    st.markdown("**Latest accepted paper trades**")
    st.dataframe(decisions, use_container_width=True, hide_index=True)

    if "net_return_bps" in decisions.columns and not decisions.empty:
        sorted_df = decisions.sort_values("net_return_bps")
        with st.expander("Worst 10 paper trades"):
            st.dataframe(sorted_df.head(10), use_container_width=True, hide_index=True)
        with st.expander("Best 10 paper trades"):
            st.dataframe(sorted_df.tail(10).iloc[::-1], use_container_width=True, hide_index=True)


def _render_milestones() -> None:
    import streamlit as st

    st.markdown("## Milestone Journey")
    for index, (name, pct, status) in enumerate(MILESTONES, start=1):
        st.markdown(f"**Milestone {index}: {name}** — {pct}%")
        st.caption(status)
        st.progress(pct / 100)


def _render_architecture() -> None:
    import streamlit as st

    st.markdown("## Architecture")
    st.markdown(ARCHITECTURE_DIAGRAM)


def _render_next_plan() -> None:
    import streamlit as st

    st.markdown("## Next Research Plan")
    for item in NEXT_RESEARCH_PLAN:
        st.checkbox(item, value=False, disabled=True)


def _render_debug(phase3: dict[str, Any] | None) -> None:
    import streamlit as st

    with st.expander("Debug / raw report details"):
        st.markdown("Read-only artifact references for troubleshooting.")
        if phase3:
            st.json(
                {
                    "phase3_audit": str(PHASE3_AUDIT_PATH),
                    "downloads_total": (phase3.get("downloads") or {}).get("total_files"),
                    "next_action": phase3.get("next_recommended_action"),
                }
            )
        candidate = _latest_candidate_validation()
        if candidate:
            st.text(f"Latest candidate validation: {candidate.get('_report_path', '—')}")
        for coin in RESEARCH_COINS:
            wf = _latest_walk_forward_for_coin(coin)
            if wf:
                st.text(f"Latest walk-forward {coin}: {wf.get('_report_path', '—')}")
        csv_path = _latest_accepted_trades_csv()
        if csv_path:
            st.text(f"Latest accepted trades CSV: {csv_path}")
        for coin in RESEARCH_COINS:
            for interval in ("1s", "15s"):
                validation = _latest_validation_report(coin, interval)
                if validation:
                    st.text(f"Validation {coin} {interval}: {validation.get('_report_path', '—')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Bid/Ask ML Research Lab",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    phase3 = _load_json(PHASE3_AUDIT_PATH)
    candidate = _latest_candidate_validation()

    _render_header()
    st.divider()
    _render_kpi_cards(phase3, candidate)
    st.divider()
    _render_data_collection(phase3)
    st.divider()
    _render_data_quality()
    st.divider()
    _render_walk_forward()
    st.divider()
    _render_candidate_b()
    st.divider()
    _render_paper_decisions()
    st.divider()
    _render_milestones()
    st.divider()
    _render_architecture()
    st.divider()
    _render_next_plan()
    st.divider()
    _render_debug(phase3)

    st.caption(
        "Read-only dashboard — no file writes, no subprocesses, no AWS downloads, "
        "no orders, no wallets, no keys."
    )


if __name__ == "__main__":
    main()
