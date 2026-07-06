"""
Hyperliquid Bid/Ask ML Research Dashboard — read-only, paper-only, research-only.

Displays pipeline status, data quality, walk-forward results, candidate research,
and paper decisions from local JSON/CSV artifacts. No execution, no AWS, no writes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT
from bid_ask_validator import bid_ask_csv_path

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

CANDIDATE_B_STRICT_REPORT = WALK_FORWARD_DIR / "walk_forward_BTC_20260706T115051.json"
CANDIDATE_B_RELAXED_25_REPORT = WALK_FORWARD_DIR / "walk_forward_BTC_20260706T142556.json"
CANDIDATE_B_RELAXED_62_REPORT = WALK_FORWARD_DIR / "walk_forward_BTC_20260706T120135.json"

MILESTONES: tuple[tuple[str, int], ...] = (
    ("Data Pipeline Works", 80),
    ("Paper Backtest Engine Works", 85),
    ("First Candidate Found", 40),
    ("Candidate Scaling", 20),
    ("Robustness", 15),
    ("Out-of-Sample Paper Test", 0),
    ("Paper Operations", 70),
    ("Market Ready", 0),
)

ARCHITECTURE_DIAGRAM = """
```
Hyperliquid S3 / WebSocket
        │
        ▼
   Raw L2 data
        │
        ▼
Bid/Ask importer / extractor
        │
        ▼
     1s CSV
        │
        ▼
  15s / 1m resampler
        │
        ▼
    Validator
        │
        ▼
  Feature builder
        │
        ▼
  Model training
        │
        ▼
Walk-forward evaluator
        │
        ▼
Candidate validator
        │
        ▼
Paper decision dashboard  ← you are here
```
"""

NEXT_RESEARCH_PLAN = [
    "Finish BTC/ETH 14-day data sprint",
    "Inspect losing trades from relaxed Candidate B",
    "Test horizons 10 / 15 / 20 / 30 / 40",
    "Compare BTC vs ETH",
    "Build candidate grid",
    "Freeze promising candidate",
    "Test out-of-sample",
]


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
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_num(value: Any, *, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            return f"{value:.{digits}f}{suffix}"
        return f"{value}{suffix}"
    return str(value)


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
    candidates = sorted(
        reports_dir.glob(pattern),
        key=_parse_report_stamp,
        reverse=True,
    )
    for path in candidates:
        payload = _load_json(path)
        if payload is not None:
            payload["report_path"] = str(path)
            return payload
    return None


def _latest_repair_report(coin: str, interval: str) -> dict[str, Any] | None:
    reports_dir = DATA_ROOT / "reports"
    if not reports_dir.is_dir():
        return None
    pattern = f"{coin}_{interval}_repair_*.json"
    candidates = sorted(
        reports_dir.glob(pattern),
        key=_parse_report_stamp,
        reverse=True,
    )
    for path in candidates:
        payload = _load_json(path)
        if payload is not None:
            payload["report_path"] = str(path)
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
            payload["report_path"] = str(path)
            return payload
    return None


def _walk_forward_summaries(limit: int = 8) -> list[dict[str, Any]]:
    if not WALK_FORWARD_DIR.is_dir():
        return []
    reports = sorted(
        WALK_FORWARD_DIR.glob("walk_forward_*.json"),
        key=_parse_report_stamp,
        reverse=True,
    )[:limit]
    rows: list[dict[str, Any]] = []
    for path in reports:
        payload = _load_json(path)
        if payload is None:
            continue
        coins = payload.get("coins") or []
        coin = coins[0] if coins else "—"
        rows.append(
            {
                "report": path.name,
                "coin": coin,
                "interval": payload.get("interval"),
                "horizon": payload.get("horizon"),
                "windows": payload.get("completed_windows"),
                "model_accuracy": payload.get("model_accuracy_mean"),
                "baseline_accuracy": payload.get("baseline_accuracy_mean"),
                "accuracy_edge": payload.get("accuracy_edge_vs_baseline"),
                "cumulative_net_bps": payload.get("cumulative_net_return_bps"),
                "average_net_bps": payload.get("average_net_return_bps"),
                "win_rate": payload.get("win_rate_after_costs"),
                "accepted_trades": payload.get("filtered_paper_trades")
                or payload.get("accepted_trades_count"),
                "positive_after_costs": payload.get("positive_after_costs"),
                "beats_baseline": payload.get("beats_baseline_accuracy"),
                "min_confidence": payload.get("min_confidence"),
                "min_edge_bps": payload.get("min_edge_bps"),
            }
        )
    return rows


def _walk_forward_summary_from_path(path: Path) -> dict[str, Any] | None:
    payload = _load_json(path)
    if payload is None:
        return None
    return {
        "report": path.name,
        "exists": True,
        "coin": (payload.get("coins") or ["—"])[0],
        "interval": payload.get("interval"),
        "horizon": payload.get("horizon"),
        "windows": payload.get("completed_windows"),
        "model_accuracy": payload.get("model_accuracy_mean"),
        "baseline_accuracy": payload.get("baseline_accuracy_mean"),
        "accuracy_edge": payload.get("accuracy_edge_vs_baseline"),
        "cumulative_net_bps": payload.get("cumulative_net_return_bps"),
        "average_net_bps": payload.get("average_net_return_bps"),
        "win_rate": payload.get("win_rate_after_costs"),
        "accepted_trades": payload.get("filtered_paper_trades"),
        "positive_after_costs": payload.get("positive_after_costs"),
        "beats_baseline": payload.get("beats_baseline_accuracy"),
        "min_confidence": payload.get("min_confidence"),
        "min_edge_bps": payload.get("min_edge_bps"),
        "trade_classes": (
            payload.get("per_coin", {})
            .get("BTC", {})
            .get("trade_filter_summary", {})
            .get("trade_classes")
        ),
        "accepted_trades_csv": payload.get("accepted_trades_csv_path"),
    }


def _latest_accepted_trades_csv() -> Path | None:
    candidate = _latest_candidate_validation()
    if candidate:
        csv_path = candidate.get("accepted_trades_csv_path")
        if csv_path and Path(csv_path).is_file():
            return Path(csv_path)
    if not WALK_FORWARD_DIR.is_dir():
        return None
    csv_files = sorted(
        WALK_FORWARD_DIR.glob("accepted_trades_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
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


def _data_quality_row(coin: str, interval: str) -> dict[str, Any]:
    path = bid_ask_csv_path(coin, interval, DATA_ROOT, source=HISTORICAL_SOURCE)
    validation = _latest_validation_report(coin, interval)
    repair = _latest_repair_report(coin, interval) if interval == "1s" else None
    row_count = validation.get("row_count") if validation else _count_csv_rows(path)
    coverage_end = validation.get("coverage_end") if validation else None
    monotonic = repair.get("monotonic_after") if repair else None
    if monotonic is None and validation:
        warnings = validation.get("warnings") or []
        monotonic = not any("non-monotonic" in str(w).lower() for w in warnings)
    return {
        "coin": coin,
        "interval": interval,
        "rows": row_count,
        "latest_timestamp": _fmt_ts_ms(coverage_end),
        "validation_passed": validation.get("passed") if validation else None,
        "duplicate_timestamps": validation.get("duplicate_timestamp_count") if validation else None,
        "monotonic": monotonic,
        "path": str(path),
        "validation_report": validation.get("report_path") if validation else None,
    }


def _overview_cards(phase3: dict[str, Any] | None, candidate: dict[str, Any] | None) -> dict[str, str]:
    imported = (phase3 or {}).get("imported_historical_1s") or {}
    btc_cov = imported.get("BTC", {}).get("coverage_hours")
    candidate_trades = (candidate or {}).get("filtered_paper_trades", "—")
    candidate_status = (candidate or {}).get("validation_status", {}).get("not_market_ready", True)
    return {
        "Data Pipeline": "Historical 1s CSVs repaired & validated (BTC/ETH)",
        "Backfill": f"Phase 3 sprint active — ~{btc_cov or '?'}h coverage per coin",
        "Validation": "Latest validation reports passing (with expected gaps)",
        "Model Training": "RandomForest microstructure features on 15s bars",
        "Candidate Status": f"Candidate B: {candidate_trades} paper trades — not market ready",
        "Market Readiness": "0% — real execution disabled, research only",
    }


def _render_safety_banner() -> None:
    import streamlit as st

    st.title("Hyperliquid Bid/Ask ML Research Dashboard")
    st.error(
        "**Paper-only / research-only** — Real execution is **disabled**. "
        "This system is **not market ready**. Simulated returns do **not** prove profitability."
    )
    cols = st.columns(3)
    cols[0].metric("paper_only", str(PAPER_ONLY).lower())
    cols[1].metric("execution_enabled", str(EXECUTION_ENABLED).lower())
    cols[2].metric("real_execution_ready", str(REAL_EXECUTION_READY).lower())
    st.caption(SAFETY_STATEMENT)


def _render_overview(cards: dict[str, str]) -> None:
    import streamlit as st

    st.header("Project Overview")
    cols = st.columns(3)
    items = list(cards.items())
    for index, (title, body) in enumerate(items):
        with cols[index % 3]:
            st.markdown(f"**{title}**")
            st.write(body)


def _render_backfill_monitor(phase3: dict[str, Any] | None) -> None:
    import streamlit as st

    st.header("Historical Backfill Monitor")
    if phase3 is None:
        st.info("No phase3 audit report found at reports/bid_ask_research_scheduler/phase3_audit_latest.json")
        return

    cfg = phase3.get("phase_config") or {}
    imported = phase3.get("imported_historical_1s") or {}
    downloads = phase3.get("downloads") or {}

    c1, c2, c3 = st.columns(3)
    c1.metric("Phase", cfg.get("phase_backfill_name", "—"))
    c2.metric("Focused coins", ", ".join(cfg.get("focused_coins") or []))
    c3.metric("Target days", cfg.get("target_days", "—"))

    c4, c5, c6 = st.columns(3)
    c4.metric("Daily file limit", cfg.get("daily_file_limit", "—"))
    c5.metric("Daily chunk limit", cfg.get("daily_chunk_limit", "—"))
    c6.metric("Effective daily capacity", cfg.get("effective_daily_file_capacity", "—"))

    st.metric("Downloads total files", downloads.get("total_files", "—"))

    ts_cols = st.columns(2)
    btc_ts = imported.get("BTC", {}).get("last_timestamp", "—")
    eth_ts = imported.get("ETH", {}).get("last_timestamp", "—")
    ts_cols[0].metric("BTC latest timestamp", btc_ts)
    ts_cols[1].metric("ETH latest timestamp", eth_ts)

    action = phase3.get("next_recommended_action")
    if action:
        st.warning(f"**Next recommended action:** {action}")

    warnings = phase3.get("warnings") or []
    if warnings:
        with st.expander("Audit warnings"):
            for warning in warnings:
                st.write(f"- {warning}")


def _render_data_quality() -> None:
    import streamlit as st

    st.header("Data Quality Panel")
    st.caption("Historical BTC/ETH CSVs and latest validation reports (read-only).")

    rows = []
    for coin in RESEARCH_COINS:
        for interval in ("1s", "15s"):
            rows.append(_data_quality_row(coin, interval))

    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No data quality rows available.")
        return

    display = df.copy()
    display["validation_passed"] = display["validation_passed"].map(
        lambda v: "PASS" if v is True else ("FAIL" if v is False else "—")
    )
    display["monotonic"] = display["monotonic"].map(
        lambda v: "yes" if v is True else ("no" if v is False else "—")
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_walk_forward() -> None:
    import streamlit as st

    st.header("Model / Walk-Forward Testing")
    st.caption("Latest walk-forward JSON summaries. Research-only simulated performance.")

    summaries = _walk_forward_summaries(limit=10)
    if not summaries:
        st.info("No walk-forward reports found.")
        return

    df = pd.DataFrame(summaries)
    format_cols = {
        "model_accuracy": "{:.3f}",
        "baseline_accuracy": "{:.3f}",
        "accuracy_edge": "{:.3f}",
        "cumulative_net_bps": "{:.2f}",
        "average_net_bps": "{:.2f}",
        "win_rate": "{:.1%}",
    }
    st.dataframe(
        df.style.format(format_cols, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )


def _candidate_row(label: str, path: Path, description: str) -> dict[str, Any]:
    summary = _walk_forward_summary_from_path(path)
    if summary is None:
        return {"scenario": label, "description": description, "report": path.name, "exists": False}
    summary["scenario"] = label
    summary["description"] = description
    return summary


def _render_candidate_b() -> None:
    import streamlit as st

    st.header("Candidate B Panel")
    st.caption("BTC DOWN microstructure H20 — strict vs relaxed filter experiments.")

    candidate_val = _latest_candidate_validation()
    if candidate_val:
        st.markdown(
            f"Latest candidate validation: **{candidate_val.get('candidate_name', '—')}** "
            f"({candidate_val.get('run_timestamp', '—')})"
        )

    scenarios = [
        _candidate_row(
            "Current (5 trades)",
            CANDIDATE_B_STRICT_REPORT,
            "Strict Candidate B: min_confidence=0.65, min_edge=2bps, DOWN only",
        ),
        _candidate_row(
            "Relaxed (25 trades)",
            CANDIDATE_B_RELAXED_25_REPORT,
            "Relaxed: min_confidence=0.60, min_edge=1bps, DOWN only",
        ),
        _candidate_row(
            "Relaxed (62 trades)",
            CANDIDATE_B_RELAXED_62_REPORT,
            "More relaxed: min_confidence=0.60, min_edge=1bps, DOWN+UP classes",
        ),
    ]

    table_rows = []
    for scenario in scenarios:
        if not scenario.get("exists", True) and "accepted_trades" not in scenario:
            table_rows.append({"scenario": scenario["scenario"], "status": "report missing"})
            continue
        table_rows.append(
            {
                "scenario": scenario.get("scenario"),
                "trades": scenario.get("accepted_trades"),
                "cumulative_net_bps": scenario.get("cumulative_net_bps"),
                "average_net_bps": scenario.get("average_net_bps"),
                "win_rate": scenario.get("win_rate"),
                "positive_after_costs": scenario.get("positive_after_costs"),
                "beats_baseline": scenario.get("beats_baseline"),
                "filters": (
                    f"conf≥{scenario.get('min_confidence')} "
                    f"edge≥{scenario.get('min_edge_bps')}bps "
                    f"{scenario.get('trade_classes') or 'DOWN'}"
                ),
            }
        )

    df = pd.DataFrame(table_rows)
    st.dataframe(
        df.style.format(
            {
                "cumulative_net_bps": "{:.2f}",
                "average_net_bps": "{:.2f}",
                "win_rate": "{:.1%}",
            },
            na_rep="—",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Why Candidate B is not market ready**")
    st.markdown(
        "- Only **5 trades** pass strict filters — far below the ~20+ needed for confidence\n"
        "- Relaxing filters to **25 or 62 trades** flips cumulative net return **negative** after costs\n"
        "- Stability warnings: concentrated buckets, sparse time windows\n"
        "- Validation flags: `insufficient_trades`, `unstable_concentrated`, `not_market_ready`"
    )
    st.warning(
        "**Current blocker:** increasing trade count destroys post-cost performance. "
        "Need a candidate that scales trades while staying positive after realistic costs."
    )


def _render_paper_decisions() -> None:
    import streamlit as st

    st.header("Live Paper Decision Viewer")
    csv_path = _latest_accepted_trades_csv()
    if csv_path is None:
        st.info("No accepted_trades CSV found.")
        return

    st.caption(f"Source: `{csv_path}`")
    try:
        decisions = _load_paper_decisions(csv_path)
    except (OSError, pd.errors.EmptyDataError, ValueError) as exc:
        st.error(f"Could not read paper decisions: {exc}")
        return

    st.dataframe(decisions, use_container_width=True, hide_index=True)
    st.info(
        "These are **paper decisions only**. Labels show what the research filter would have accepted. "
        "**No real orders were placed.** Real execution is disabled."
    )


def _render_milestones() -> None:
    import streamlit as st

    st.header("Milestone Journey")
    labels = [
        "Milestone 1: Data Pipeline Works",
        "Milestone 2: Paper Backtest Engine Works",
        "Milestone 3: First Candidate Found",
        "Milestone 4: Candidate Scaling",
        "Milestone 5: Robustness",
        "Milestone 6: Out-of-Sample Paper Test",
        "Milestone 7: Paper Operations",
        "Milestone 8: Market Ready",
    ]
    for label, (name, pct) in zip(labels, MILESTONES):
        st.markdown(f"**{label}** — {name}: {pct}%")
        st.progress(pct / 100)


def _render_architecture() -> None:
    import streamlit as st

    st.header("Architecture Diagram")
    st.markdown(ARCHITECTURE_DIAGRAM)


def _render_next_plan() -> None:
    import streamlit as st

    st.header("Next Research Plan")
    for item in NEXT_RESEARCH_PLAN:
        st.markdown(f"- {item}")


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Bid/Ask ML Research",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    phase3 = _load_json(PHASE3_AUDIT_PATH)
    candidate = _latest_candidate_validation()
    cards = _overview_cards(phase3, candidate)

    _render_safety_banner()
    st.divider()
    _render_overview(cards)
    st.divider()
    _render_backfill_monitor(phase3)
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
    st.caption(
        "Read-only dashboard — no file writes, no subprocesses, no AWS downloads, "
        "no orders, no wallets, no keys."
    )


if __name__ == "__main__":
    main()
