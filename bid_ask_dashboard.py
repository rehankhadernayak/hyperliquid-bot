"""
Bid/Ask Research Dashboard — read-only, research-only, paper-only.

Displays local bid/ask CSV files, validation reports, training reports, and
suggested pipeline commands. No wallets, signing, orders, or network access.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bid_ask_schema import SAFETY_STATEMENT
from bid_ask_validator import bid_ask_csv_path

DEFAULT_DATA_ROOT = Path("data/bid_ask_csv")
DEFAULT_TRAINING_REPORTS_DIR = Path("reports/bid_ask_training")
SUPPORTED_COINS = ("BTC", "ETH", "SOL")
SUPPORTED_INTERVALS = ("1s", "15s", "1m")
CSV_PREVIEW_ROWS = 20

RESEARCH_DISCLAIMER = "Research-only results. This does not prove profitability."

DATA_COVERAGE_STATEMENT = (
    "Using locally collected data only. This is not full exchange all-time history."
)


def _reports_dir(data_root: Path) -> Path:
    return data_root / "reports"


def _parse_report_stamp(path: Path) -> str:
    """Extract sortable timestamp token from a pipeline report filename."""
    stem = path.stem
    if "_validation_" in stem:
        return stem.rsplit("_validation_", 1)[-1]
    if "_resample_" in stem:
        return stem.rsplit("_resample_", 1)[-1]
    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[1]:
        return parts[1]
    return stem


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open(encoding="utf-8") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except OSError:
        return 0


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _csv_coverage(path: Path, interval: str) -> tuple[int | None, int | None]:
    if not path.is_file():
        return None, None
    try:
        df = pd.read_csv(path, usecols=[0])
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None, None
    if df.empty:
        return None, None
    ts_col = "timestamp_ms" if interval == "1s" else "start_timestamp_ms"
    try:
        full = pd.read_csv(path, usecols=[ts_col])
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None, None
    if full.empty or ts_col not in full.columns:
        return None, None
    timestamps = full[ts_col].astype("int64")
    return int(timestamps.min()), int(timestamps.max())


def discover_bid_ask_files(
    data_root: Path = DEFAULT_DATA_ROOT,
    training_reports_dir: Path = DEFAULT_TRAINING_REPORTS_DIR,
) -> dict[str, Any]:
    """Discover local bid/ask CSV files and JSON reports."""
    csv_1s = sorted((data_root / "1s").glob("*_1s.csv")) if (data_root / "1s").is_dir() else []
    csv_15s = sorted((data_root / "15s").glob("*_15s.csv")) if (data_root / "15s").is_dir() else []
    csv_1m = sorted((data_root / "1m").glob("*_1m.csv")) if (data_root / "1m").is_dir() else []

    reports_path = _reports_dir(data_root)
    validation_reports: list[Path] = []
    extraction_reports: list[Path] = []
    if reports_path.is_dir():
        for path in sorted(reports_path.glob("*.json")):
            name = path.name
            if "_validation_" in name:
                validation_reports.append(path)
            elif "_resample_" not in name:
                extraction_reports.append(path)

    training_reports: list[Path] = []
    if training_reports_dir.is_dir():
        training_reports = sorted(training_reports_dir.glob("*.json"))

    locally_collected = False
    for path in extraction_reports:
        payload = _load_json(path)
        if payload and payload.get("locally_collected_data_only") is True:
            locally_collected = True
            break

    return {
        "csv_1s": csv_1s,
        "csv_15s": csv_15s,
        "csv_1m": csv_1m,
        "validation_reports": validation_reports,
        "extraction_reports": extraction_reports,
        "training_reports": training_reports,
        "locally_collected_data_only": locally_collected,
        "counts": {
            "csv_1s": len(csv_1s),
            "csv_15s": len(csv_15s),
            "csv_1m": len(csv_1m),
            "validation_reports": len(validation_reports),
            "training_reports": len(training_reports),
        },
    }


def load_csv_preview(
    coin: str,
    interval: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    n: int = CSV_PREVIEW_ROWS,
) -> dict[str, Any]:
    """Load the first n rows of a bid/ask CSV for preview."""
    path = bid_ask_csv_path(coin, interval, data_root)
    if not path.is_file():
        return {
            "coin": coin,
            "interval": interval,
            "path": str(path),
            "found": False,
            "preview": pd.DataFrame(),
            "total_rows": 0,
            "error": None,
        }
    try:
        preview = pd.read_csv(path, nrows=n)
        total_rows = _count_csv_rows(path)
        return {
            "coin": coin,
            "interval": interval,
            "path": str(path),
            "found": True,
            "preview": preview,
            "total_rows": total_rows,
            "error": None,
        }
    except (OSError, pd.errors.EmptyDataError, ValueError) as exc:
        return {
            "coin": coin,
            "interval": interval,
            "path": str(path),
            "found": False,
            "preview": pd.DataFrame(),
            "total_rows": 0,
            "error": str(exc),
        }


def load_latest_validation_report(
    coin: str,
    interval: str,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any] | None:
    """Return the newest validation report for a coin/interval pair."""
    reports_path = _reports_dir(data_root)
    if not reports_path.is_dir():
        return None

    pattern = f"{coin}_{interval}_validation_*.json"
    candidates = sorted(
        reports_path.glob(pattern),
        key=lambda path: _parse_report_stamp(path),
        reverse=True,
    )
    for path in candidates:
        payload = _load_json(path)
        if payload is not None:
            payload.setdefault("report_path", str(path))
            return payload
    return None


def load_latest_training_report(
    coin: str,
    training_reports_dir: Path = DEFAULT_TRAINING_REPORTS_DIR,
) -> dict[str, Any] | None:
    """Return the newest training report for a coin."""
    if not training_reports_dir.is_dir():
        return None

    reports: list[tuple[str, dict[str, Any]]] = []
    for path in training_reports_dir.glob("*.json"):
        payload = _load_json(path)
        if payload is None:
            continue
        if payload.get("coin", "").upper() != coin.upper():
            continue
        generated_at = payload.get("generated_at") or path.stem
        reports.append((generated_at, payload))

    if not reports:
        return None

    reports.sort(key=lambda item: item[0], reverse=True)
    latest = reports[0][1]
    latest.setdefault("report_path", str(training_reports_dir / f"{latest.get('report_id', 'unknown')}.json"))
    return latest


def build_coverage_table(
    data_root: Path = DEFAULT_DATA_ROOT,
    coins: tuple[str, ...] = SUPPORTED_COINS,
) -> pd.DataFrame:
    """Build per-coin coverage summary for the dashboard table."""
    rows: list[dict[str, Any]] = []
    for coin in coins:
        row: dict[str, Any] = {"coin": coin}
        coverage_starts: list[int] = []
        coverage_ends: list[int] = []

        for interval in SUPPORTED_INTERVALS:
            path = bid_ask_csv_path(coin, interval, data_root)
            has_csv = path.is_file()
            row[f"has_{interval}_csv"] = has_csv
            row[f"row_count_{interval}"] = _count_csv_rows(path) if has_csv else 0

            if has_csv:
                start, end = _csv_coverage(path, interval)
                if start is not None:
                    coverage_starts.append(start)
                if end is not None:
                    coverage_ends.append(end)

            validation = load_latest_validation_report(coin, interval, data_root)
            row[f"validation_passed_{interval}"] = (
                validation.get("passed") if validation else None
            )

        validation_1s = load_latest_validation_report(coin, "1s", data_root)
        row["missing_seconds_1s"] = (
            validation_1s.get("missing_periods_count") if validation_1s else None
        )

        row["coverage_start"] = min(coverage_starts) if coverage_starts else None
        row["coverage_end"] = max(coverage_ends) if coverage_ends else None
        rows.append(row)

    return pd.DataFrame(rows)


def build_chart_dataframe(
    coin: str,
    interval: str,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> pd.DataFrame:
    """Return columns needed for bid/ask and spread charts."""
    path = bid_ask_csv_path(coin, interval, data_root)
    if not path.is_file():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if interval == "1s":
        columns = [
            "timestamp_ms",
            "best_bid",
            "best_ask",
            "mid_price",
            "spread",
            "spread_bps",
        ]
    else:
        columns = [
            "start_timestamp_ms",
            "avg_bid",
            "avg_ask",
            "avg_mid",
            "close_mid",
            "avg_spread",
            "avg_spread_bps",
        ]

    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.DataFrame()
    return df[available].copy()


def build_suggested_commands(coin: str = "BTC") -> dict[str, str]:
    """Return copyable pipeline commands (display only — never executed)."""
    return {
        "extract": (
            f"python3 bid_ask_data_pipeline.py extract "
            f"--coins {coin} --source local --interval 1s"
        ),
        "validate": (
            f"python3 bid_ask_data_pipeline.py validate "
            f"--coins {coin} --interval 1s"
        ),
        "resample": (
            f"python3 bid_ask_data_pipeline.py resample "
            f"--coins {coin} --from-interval 1s --to-interval all"
        ),
        "train": (
            f"python3 bid_ask_data_pipeline.py train "
            f"--coin {coin} --interval 15s --horizon 1"
        ),
        "run-all": (
            "python3 bid_ask_data_pipeline.py run-all "
            "--coins BTC ETH SOL --train-interval 15s --horizon 1"
        ),
    }


def _format_timestamp_ms(value: int | None) -> str:
    if value is None:
        return "—"
    dt = datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _format_bool(value: bool | None) -> str:
    if value is None:
        return "—"
    return "Yes" if value else "No"


def _render_pipeline_status(
    discovery: dict[str, Any],
) -> None:
    import streamlit as st

    st.subheader("1. Pipeline Status")
    counts = discovery["counts"]
    cols = st.columns(6)
    cols[0].metric("1s CSV files", counts["csv_1s"])
    cols[1].metric("15s CSV files", counts["csv_15s"])
    cols[2].metric("1m CSV files", counts["csv_1m"])
    cols[3].metric("Validation reports", counts["validation_reports"])
    cols[4].metric("Training reports", counts["training_reports"])
    locally = "Yes" if discovery["locally_collected_data_only"] else "Unknown"
    cols[5].metric("Locally collected data only", locally)
    st.caption(DATA_COVERAGE_STATEMENT)


def _render_coverage_table(data_root: Path) -> None:
    import streamlit as st

    st.subheader("2. Coin Coverage Table")
    coverage = build_coverage_table(data_root=data_root)
    if coverage.empty:
        st.info("No coin coverage data found.")
        return

    display = coverage.copy()
    display["coverage_start"] = display["coverage_start"].apply(_format_timestamp_ms)
    display["coverage_end"] = display["coverage_end"].apply(_format_timestamp_ms)
    for interval in SUPPORTED_INTERVALS:
        col = f"validation_passed_{interval}"
        display[col] = display[col].apply(_format_bool)
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_csv_preview(data_root: Path, coin: str, interval: str) -> None:
    import streamlit as st

    st.subheader("3. CSV Preview")
    preview = load_csv_preview(coin, interval, data_root)
    st.text(f"File: {preview['path']}")
    if not preview["found"]:
        st.warning("CSV file not found for this coin/interval.")
        if preview.get("error"):
            st.caption(f"Error: {preview['error']}")
        return

    st.caption(f"Total rows: {preview['total_rows']:,}")
    st.dataframe(preview["preview"], use_container_width=True, hide_index=True)

    path = Path(preview["path"])
    if path.is_file():
        try:
            csv_bytes = path.read_bytes()
            st.download_button(
                label="Download CSV",
                data=csv_bytes,
                file_name=path.name,
                mime="text/csv",
            )
        except OSError:
            st.caption("Download unavailable (file read error).")


def _render_bid_ask_chart(data_root: Path, coin: str, interval: str) -> None:
    import streamlit as st

    st.subheader("4. Bid / Ask / Mid Chart")
    chart_df = build_chart_dataframe(coin, interval, data_root)
    if chart_df.empty:
        st.info("No chart data available for this coin/interval.")
        return

    if interval == "1s":
        x_col = "timestamp_ms"
        y_cols = ["best_bid", "best_ask", "mid_price"]
    else:
        x_col = "start_timestamp_ms"
        y_cols = [column for column in ("avg_bid", "avg_ask", "avg_mid", "close_mid") if column in chart_df.columns]

    plot_df = chart_df.set_index(x_col)[y_cols]
    st.line_chart(plot_df, use_container_width=True)


def _render_spread_chart(data_root: Path, coin: str, interval: str) -> None:
    import streamlit as st

    st.subheader("5. Spread Chart")
    chart_df = build_chart_dataframe(coin, interval, data_root)
    if chart_df.empty:
        st.info("No spread data available for this coin/interval.")
        return

    if interval == "1s":
        x_col = "timestamp_ms"
        spread_cols = ["spread", "spread_bps"]
    else:
        x_col = "start_timestamp_ms"
        spread_cols = ["avg_spread", "avg_spread_bps"]

    available = [column for column in spread_cols if column in chart_df.columns]
    if not available:
        st.info("Spread columns not found in CSV.")
        return

    plot_df = chart_df.set_index(x_col)[available]
    st.line_chart(plot_df, use_container_width=True)


def _render_validation_summary(data_root: Path, coin: str, interval: str) -> None:
    import streamlit as st

    st.subheader("6. Validation Report Summary")
    report = load_latest_validation_report(coin, interval, data_root)
    if report is None:
        st.info("No validation report found for this coin/interval.")
        return

    cols = st.columns(3)
    cols[0].metric("Passed", _format_bool(report.get("passed")))
    cols[1].metric("Row count", report.get("row_count", "—"))
    cols[2].metric("Missing periods", report.get("missing_periods_count", "—"))

    cols2 = st.columns(2)
    cols2[0].metric("Duplicate timestamps", report.get("duplicate_timestamp_count", "—"))
    cols2[1].metric("Invalid bid/ask count", report.get("invalid_bid_ask_count", "—"))

    warnings = report.get("warnings") or []
    if warnings:
        st.markdown("**Warnings**")
        for warning in warnings:
            st.write(f"- {warning}")
    else:
        st.caption("No warnings.")

    if report.get("report_path"):
        st.caption(f"Report: {report['report_path']}")


def _render_training_summary(
    coin: str,
    training_reports_dir: Path,
) -> None:
    import streamlit as st

    st.subheader("7. Training Report Summary")
    st.warning(RESEARCH_DISCLAIMER)

    report = load_latest_training_report(coin, training_reports_dir)
    if report is None:
        st.info("No training report found for this coin.")
        return

    evaluation = report.get("evaluation") or {}
    label_counts = evaluation.get("label_counts") or {}
    test_counts = label_counts.get("test") or {}

    cols = st.columns(4)
    cols[0].metric("Model used", report.get("model_used", "—"))
    cols[1].metric("Interval", report.get("interval", "—"))
    cols[2].metric("Horizon", report.get("horizon", "—"))
    cols[3].metric("Accuracy", evaluation.get("accuracy", "—"))

    cols2 = st.columns(3)
    cols2[0].metric("Train rows", evaluation.get("train_rows", "—"))
    cols2[1].metric("Test rows", evaluation.get("test_rows", "—"))
    cols2[2].metric("Baseline accuracy", evaluation.get("baseline_accuracy", "—"))

    st.markdown("**Test label counts**")
    label_cols = st.columns(3)
    label_cols[0].metric("UP", test_counts.get("UP", 0))
    label_cols[1].metric("DOWN", test_counts.get("DOWN", 0))
    label_cols[2].metric("FLAT", test_counts.get("FLAT", 0))

    warnings = evaluation.get("warnings") or []
    if warnings:
        st.markdown("**Warnings**")
        for warning in warnings:
            st.write(f"- {warning}")

    if report.get("report_path"):
        st.caption(f"Report: {report['report_path']}")


def _render_next_commands(coin: str) -> None:
    import streamlit as st

    st.subheader("8. Next Commands")
    st.caption("Suggested commands for the terminal. The dashboard does not execute these.")
    commands = build_suggested_commands(coin)
    for name, command in commands.items():
        st.markdown(f"**{name}**")
        st.code(command, language="bash")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Bid/Ask Research Dashboard", layout="wide")
    st.title("Bid/Ask Research Dashboard")
    st.caption("Read-only · Research-only · Paper-only · Local files only")
    st.info(RESEARCH_DISCLAIMER)
    st.caption(SAFETY_STATEMENT)

    data_root = DEFAULT_DATA_ROOT
    training_reports_dir = DEFAULT_TRAINING_REPORTS_DIR

    discovery = discover_bid_ask_files(data_root, training_reports_dir)
    _render_pipeline_status(discovery)

    st.divider()
    _render_coverage_table(data_root)

    st.divider()
    selector_cols = st.columns(2)
    with selector_cols[0]:
        coin = st.selectbox("Coin", SUPPORTED_COINS, key="bid_ask_coin")
    with selector_cols[1]:
        interval = st.selectbox("Interval", SUPPORTED_INTERVALS, key="bid_ask_interval")

    _render_csv_preview(data_root, coin, interval)

    st.divider()
    _render_bid_ask_chart(data_root, coin, interval)

    st.divider()
    _render_spread_chart(data_root, coin, interval)

    st.divider()
    _render_validation_summary(data_root, coin, interval)

    st.divider()
    _render_training_summary(coin, training_reports_dir)

    st.divider()
    _render_next_commands(coin)


if __name__ == "__main__":
    main()
