"""Standalone analyzer for accepted paper trades CSVs (research-only)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT

DEFAULT_COIN = "BTC"
DEFAULT_REPORTS_DIR = Path("reports/bid_ask_walk_forward")
DEFAULT_OUTPUT_DIR = Path("reports/accepted_trades_analysis")

REQUIRED_COLUMNS = (
    "coin",
    "timestamp_iso",
    "predicted_class",
    "net_return_bps",
)

OPTIONAL_COLUMNS = (
    "window_index",
    "row_index",
    "actual_label",
    "confidence",
    "future_return_bps",
    "gross_return_bps",
    "cost_bps",
    "spread_bps",
)

LABELS = ("UP", "DOWN", "FLAT")
SPREAD_BUCKETS = ("tight", "normal", "wide", "too_wide")
CONFIDENCE_BUCKETS = ("low", "medium", "high")
COST_BUCKETS = ("low", "medium", "high")

MIN_BUCKET_TRADES = 3
LOSS_CONCENTRATION_WARNING_FRACTION = 0.50
POSITIVE_CONCENTRATION_WARNING_FRACTION = 0.70
MIN_TRADES_FOR_FILTER_SUGGESTION = 3

SAFETY_NOTE = (
    "Paper-only accepted-trades analysis. Reads local CSV artifacts only. "
    "No orders, execution, wallets, AWS, network, or shell commands."
)

RESEARCH_ONLY_DISCLAIMER = (
    "Research-only analysis of historical accepted paper trades. "
    "Does not prove profitability. Relaxed Candidate B losses often reflect "
    "filter dilution, cost drag, and concentrated losing regimes — not live PnL."
)


def spread_bucket(spread_bps: float) -> str:
    """Classify spread into research review buckets."""
    if spread_bps <= 2:
        return "tight"
    if spread_bps <= 10:
        return "normal"
    if spread_bps <= 25:
        return "wide"
    return "too_wide"


def confidence_bucket(confidence: float) -> str:
    """Classify model confidence into review buckets."""
    if confidence < 0.80:
        return "low"
    if confidence < 0.95:
        return "medium"
    return "high"


def cost_bucket(cost_bps: float) -> str:
    """Classify execution cost into review buckets."""
    if cost_bps <= 5:
        return "low"
    if cost_bps <= 8:
        return "medium"
    return "high"


def _parse_timestamp_iso(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def discover_latest_accepted_trades_csv(
    coin: str,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> Path | None:
    """Return the newest single-coin accepted_trades CSV for a coin."""
    if not reports_dir.is_dir():
        return None

    coin_upper = coin.upper()
    prefix = f"accepted_trades_{coin_upper}_"
    matches: list[Path] = []
    for path in reports_dir.glob(f"{prefix}*.csv"):
        suffix = path.name[len(prefix) :]
        if suffix and suffix[0].isdigit():
            matches.append(path)

    if not matches:
        return None
    return sorted(matches, key=lambda path: path.name)[-1]


def resolve_accepted_trades_csv(
    *,
    coin: str,
    csv_path: Path | None,
    reports_dir: Path,
    use_latest: bool,
) -> Path:
    """Resolve explicit CSV path or latest accepted trades file."""
    if csv_path is not None:
        return csv_path
    if not use_latest:
        raise ValueError("Provide --csv or enable --latest")
    latest = discover_latest_accepted_trades_csv(coin, reports_dir=reports_dir)
    if latest is None:
        raise FileNotFoundError(
            f"No accepted trades CSV found for {coin.upper()} in {reports_dir}"
        )
    return latest


def load_accepted_trades_csv(path: Path) -> pd.DataFrame:
    """Load an accepted trades CSV safely."""
    if not path.is_file():
        raise FileNotFoundError(f"Accepted trades CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df["net_return_bps"] = pd.to_numeric(df["net_return_bps"], errors="coerce")
    if df["net_return_bps"].isna().any():
        raise ValueError(f"Invalid net_return_bps values in {path}")

    return df


def _empty_bucket_metrics() -> dict[str, Any]:
    return {
        "trades": 0,
        "cumulative_net_return_bps": 0.0,
        "average_net_return_bps": 0.0,
        "win_rate": 0.0,
    }


def aggregate_net_return_bucket(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute net-return metrics for one bucket of accepted trades."""
    if not records:
        return _empty_bucket_metrics()

    net_returns = [float(record["net_return_bps"]) for record in records]
    trades = len(net_returns)
    cumulative = float(sum(net_returns))
    wins = sum(1 for value in net_returns if value > 0)
    return {
        "trades": trades,
        "cumulative_net_return_bps": cumulative,
        "average_net_return_bps": cumulative / trades,
        "win_rate": wins / trades,
    }


def _group_by_column(
    df: pd.DataFrame,
    *,
    column: str,
    value_fn: Any | None = None,
) -> dict[str, dict[str, Any]]:
    if column not in df.columns:
        return {}

    working = df.copy()
    if value_fn is not None:
        working["_bucket"] = working[column].apply(value_fn)
        group_column = "_bucket"
    else:
        group_column = column

    grouped: dict[str, dict[str, Any]] = {}
    for key, group in working.groupby(group_column, dropna=False):
        label = str(key)
        records = group.to_dict(orient="records")
        grouped[label] = aggregate_net_return_bucket(records)
    return dict(sorted(grouped.items()))


def _net_return_by_hour(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if "timestamp_iso" not in df.columns:
        return {}

    working = df.copy()
    hours: list[str | None] = []
    for value in working["timestamp_iso"]:
        parsed = _parse_timestamp_iso(value)
        hours.append(f"{parsed.hour:02d}" if parsed is not None else None)
    working["_hour"] = hours
    working = working[working["_hour"].notna()]
    if working.empty:
        return {}

    grouped: dict[str, dict[str, Any]] = {}
    for hour, group in working.groupby("_hour"):
        grouped[str(hour)] = aggregate_net_return_bucket(group.to_dict(orient="records"))
    return dict(sorted(grouped.items()))


def _trade_summary_row(row: pd.Series) -> dict[str, Any]:
    summary = {
        "timestamp_iso": str(row.get("timestamp_iso", "")),
        "predicted_class": str(row.get("predicted_class", "")),
        "net_return_bps": float(row["net_return_bps"]),
    }
    for column in ("coin", "window_index", "row_index", "confidence", "spread_bps", "cost_bps"):
        if column in row.index and pd.notna(row[column]):
            summary[column] = row[column]
    return summary


def compute_loss_concentration(net_returns: list[float], *, worst_n: int = 5) -> dict[str, Any]:
    """Worst N trades as a fraction of total losses."""
    losses = [value for value in net_returns if value < 0]
    total_loss = float(sum(losses))
    if not losses:
        return {
            "worst_trade_count": 0,
            "worst_trades_net_bps": 0.0,
            "total_loss_bps": 0.0,
            "worst_trades_pct_of_total_loss": 0.0,
            "concentration_warning": False,
        }

    worst = sorted(losses)[: max(1, min(worst_n, len(losses)))]
    worst_sum = float(sum(worst))
    pct = abs(worst_sum / total_loss) if total_loss < 0 else 0.0
    return {
        "worst_trade_count": len(worst),
        "worst_trades_net_bps": worst_sum,
        "total_loss_bps": total_loss,
        "worst_trades_pct_of_total_loss": pct,
        "concentration_warning": pct >= LOSS_CONCENTRATION_WARNING_FRACTION,
    }


def compute_positive_concentration(
    bucket_groups: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Best bucket as a fraction of total positive net return."""
    candidates: list[tuple[str, str, float]] = []
    for group_name, group in bucket_groups.items():
        for bucket_key, metrics in group.items():
            cumulative = float(metrics.get("cumulative_net_return_bps", 0.0))
            trades = int(metrics.get("trades", 0))
            if trades > 0 and cumulative > 0:
                candidates.append((group_name, bucket_key, cumulative))

    if not candidates:
        return {
            "best_bucket_group": None,
            "best_bucket_key": None,
            "best_bucket_net_bps": 0.0,
            "total_positive_net_bps": 0.0,
            "best_bucket_pct_of_positive_net": 0.0,
            "concentration_warning": False,
        }

    total_positive = float(sum(item[2] for item in candidates))
    group_name, bucket_key, best_net = max(candidates, key=lambda item: item[2])
    pct = best_net / total_positive if total_positive > 0 else 0.0
    return {
        "best_bucket_group": group_name,
        "best_bucket_key": bucket_key,
        "best_bucket_net_bps": best_net,
        "total_positive_net_bps": total_positive,
        "best_bucket_pct_of_positive_net": pct,
        "concentration_warning": pct >= POSITIVE_CONCENTRATION_WARNING_FRACTION,
    }


def identify_suspicious_regimes(
    *,
    net_return_by_hour: dict[str, dict[str, Any]],
    net_return_by_predicted_class: dict[str, dict[str, Any]],
    net_return_by_confidence_bucket: dict[str, dict[str, Any]],
    net_return_by_spread_bucket: dict[str, dict[str, Any]],
    net_return_by_cost_bucket: dict[str, dict[str, Any]],
    loss_concentration: dict[str, Any],
    positive_concentration: dict[str, Any],
    cumulative_net_return_bps: float,
) -> list[dict[str, Any]]:
    """Flag regimes that may explain relaxed Candidate B underperformance."""
    regimes: list[dict[str, Any]] = []

    if cumulative_net_return_bps < 0:
        regimes.append(
            {
                "regime_type": "negative_overall",
                "key": "all_trades",
                "detail": (
                    f"Cumulative net return is negative ({cumulative_net_return_bps:.2f} bps)."
                ),
                "cumulative_net_return_bps": cumulative_net_return_bps,
            }
        )

    def _append_negative_buckets(
        group_name: str,
        group: dict[str, dict[str, Any]],
        *,
        prefix: str,
    ) -> None:
        for bucket_key, metrics in group.items():
            trades = int(metrics.get("trades", 0))
            cumulative = float(metrics.get("cumulative_net_return_bps", 0.0))
            if trades >= MIN_BUCKET_TRADES and cumulative < 0:
                regimes.append(
                    {
                        "regime_type": group_name,
                        "key": bucket_key,
                        "detail": (
                            f"{prefix}{bucket_key}: {trades} trades, "
                            f"{cumulative:.2f} bps cumulative net"
                        ),
                        "trades": trades,
                        "cumulative_net_return_bps": cumulative,
                    }
                )

    _append_negative_buckets("negative_hour", net_return_by_hour, prefix="hour ")
    _append_negative_buckets(
        "negative_predicted_class",
        net_return_by_predicted_class,
        prefix="class ",
    )
    _append_negative_buckets(
        "negative_confidence_bucket",
        net_return_by_confidence_bucket,
        prefix="confidence ",
    )
    _append_negative_buckets(
        "negative_spread_bucket",
        net_return_by_spread_bucket,
        prefix="spread ",
    )
    _append_negative_buckets(
        "negative_cost_bucket",
        net_return_by_cost_bucket,
        prefix="cost ",
    )

    if loss_concentration.get("concentration_warning"):
        pct = float(loss_concentration.get("worst_trades_pct_of_total_loss", 0.0))
        regimes.append(
            {
                "regime_type": "loss_concentration",
                "key": "worst_5_trades",
                "detail": (
                    f"Worst 5 trades account for {pct:.1%} of total losses."
                ),
                "worst_trades_pct_of_total_loss": pct,
            }
        )

    if positive_concentration.get("concentration_warning"):
        pct = float(positive_concentration.get("best_bucket_pct_of_positive_net", 0.0))
        group = positive_concentration.get("best_bucket_group")
        key = positive_concentration.get("best_bucket_key")
        regimes.append(
            {
                "regime_type": "positive_concentration",
                "key": f"{group}:{key}",
                "detail": (
                    f"Best bucket ({group}/{key}) holds {pct:.1%} "
                    "of positive net return."
                ),
                "best_bucket_pct_of_positive_net": pct,
            }
        )

    return regimes


def recommend_next_filters(
    *,
    net_return_by_hour: dict[str, dict[str, Any]],
    net_return_by_predicted_class: dict[str, dict[str, Any]],
    net_return_by_confidence_bucket: dict[str, dict[str, Any]],
    net_return_by_spread_bucket: dict[str, dict[str, Any]],
    net_return_by_cost_bucket: dict[str, dict[str, Any]],
    suspicious_regimes: list[dict[str, Any]],
    cumulative_net_return_bps: float,
) -> list[dict[str, Any]]:
    """Suggest paper-only filters to test next; does not execute trades."""
    filters: list[dict[str, Any]] = []

    if cumulative_net_return_bps < 0:
        filters.append(
            {
                "filter_type": "tighten_trade_gate",
                "action": "Reduce accepted trade count before re-running relaxed Candidate B",
                "reason": (
                    f"Accepted trades are net negative ({cumulative_net_return_bps:.2f} bps)."
                ),
                "paper_only": True,
            }
        )

    for hour, metrics in net_return_by_hour.items():
        trades = int(metrics.get("trades", 0))
        cumulative = float(metrics.get("cumulative_net_return_bps", 0.0))
        if trades >= MIN_TRADES_FOR_FILTER_SUGGESTION and cumulative < 0:
            filters.append(
                {
                    "filter_type": "block_hour",
                    "hour_utc": hour,
                    "action": f"Test blocking UTC hour {hour}",
                    "reason": (
                        f"Hour {hour} lost {cumulative:.2f} bps across {trades} trades"
                    ),
                    "paper_only": True,
                }
            )

    for predicted_class, metrics in net_return_by_predicted_class.items():
        trades = int(metrics.get("trades", 0))
        cumulative = float(metrics.get("cumulative_net_return_bps", 0.0))
        if trades >= MIN_TRADES_FOR_FILTER_SUGGESTION and cumulative < 0:
            filters.append(
                {
                    "filter_type": "block_predicted_class",
                    "predicted_class": predicted_class,
                    "action": f"Test excluding predicted_class={predicted_class!r}",
                    "reason": (
                        f"Class {predicted_class} lost {cumulative:.2f} bps "
                        f"across {trades} trades"
                    ),
                    "paper_only": True,
                }
            )

    for bucket_name in CONFIDENCE_BUCKETS:
        metrics = net_return_by_confidence_bucket.get(bucket_name)
        if not metrics:
            continue
        trades = int(metrics.get("trades", 0))
        cumulative = float(metrics.get("cumulative_net_return_bps", 0.0))
        if trades >= MIN_TRADES_FOR_FILTER_SUGGESTION and cumulative < 0:
            threshold = 0.80 if bucket_name == "low" else 0.95
            filters.append(
                {
                    "filter_type": "raise_min_confidence",
                    "confidence_threshold": threshold,
                    "action": f"Test min_confidence >= {threshold:.2f}",
                    "reason": (
                        f"Confidence bucket {bucket_name} lost {cumulative:.2f} bps "
                        f"across {trades} trades"
                    ),
                    "paper_only": True,
                }
            )

    for bucket_name in ("wide", "too_wide"):
        metrics = net_return_by_spread_bucket.get(bucket_name)
        if not metrics:
            continue
        trades = int(metrics.get("trades", 0))
        cumulative = float(metrics.get("cumulative_net_return_bps", 0.0))
        if trades >= MIN_TRADES_FOR_FILTER_SUGGESTION and cumulative < 0:
            threshold = 10 if bucket_name == "wide" else 25
            filters.append(
                {
                    "filter_type": "tighten_max_spread_bps",
                    "max_spread_bps": threshold,
                    "action": f"Test max_spread_bps <= {threshold}",
                    "reason": (
                        f"Spread bucket {bucket_name} lost {cumulative:.2f} bps "
                        f"across {trades} trades"
                    ),
                    "paper_only": True,
                }
            )

    for bucket_name in ("medium", "high"):
        metrics = net_return_by_cost_bucket.get(bucket_name)
        if not metrics:
            continue
        trades = int(metrics.get("trades", 0))
        cumulative = float(metrics.get("cumulative_net_return_bps", 0.0))
        if trades >= MIN_TRADES_FOR_FILTER_SUGGESTION and cumulative < 0:
            threshold = 5 if bucket_name == "medium" else 8
            filters.append(
                {
                    "filter_type": "tighten_max_cost_bps",
                    "max_cost_bps": threshold,
                    "action": f"Test blocking trades with cost_bps > {threshold}",
                    "reason": (
                        f"Cost bucket {bucket_name} lost {cumulative:.2f} bps "
                        f"across {trades} trades"
                    ),
                    "paper_only": True,
                }
            )

    if len(suspicious_regimes) >= 3:
        filters.append(
            {
                "filter_type": "regime_review",
                "action": "Review multiple suspicious regimes before scaling trade count",
                "reason": (
                    f"{len(suspicious_regimes)} suspicious regimes detected; "
                    "relaxed filters likely admit low-edge trades."
                ),
                "paper_only": True,
            }
        )

    return filters


def analyze_accepted_trades(df: pd.DataFrame) -> dict[str, Any]:
    """Compute accepted-trades diagnostics from a loaded dataframe."""
    net_returns = df["net_return_bps"].astype(float).tolist()
    trade_count = len(net_returns)
    cumulative_net_return_bps = float(sum(net_returns))
    average_net_return_bps = cumulative_net_return_bps / trade_count if trade_count else 0.0
    wins = sum(1 for value in net_returns if value > 0)
    win_rate = wins / trade_count if trade_count else 0.0

    sorted_df = df.sort_values("net_return_bps")
    worst_trades = [_trade_summary_row(row) for _, row in sorted_df.head(5).iterrows()]
    best_trades = [
        _trade_summary_row(row)
        for _, row in sorted_df.tail(5).iloc[::-1].iterrows()
    ]

    net_return_by_hour = _net_return_by_hour(df)
    net_return_by_predicted_class = _group_by_column(df, column="predicted_class")

    net_return_by_confidence_bucket: dict[str, dict[str, Any]] = {}
    if "confidence" in df.columns:
        confidence_series = pd.to_numeric(df["confidence"], errors="coerce")
        bucketed = confidence_series.apply(
            lambda value: confidence_bucket(float(value)) if pd.notna(value) else None
        )
        working = df.copy()
        working["_confidence_bucket"] = bucketed
        working = working[working["_confidence_bucket"].notna()]
        for bucket in CONFIDENCE_BUCKETS:
            group = working[working["_confidence_bucket"] == bucket]
            net_return_by_confidence_bucket[bucket] = aggregate_net_return_bucket(
                group.to_dict(orient="records")
            )

    net_return_by_spread_bucket: dict[str, dict[str, Any]] = {}
    if "spread_bps" in df.columns:
        spread_series = pd.to_numeric(df["spread_bps"], errors="coerce")
        bucketed = spread_series.apply(
            lambda value: spread_bucket(float(value)) if pd.notna(value) else None
        )
        working = df.copy()
        working["_spread_bucket"] = bucketed
        working = working[working["_spread_bucket"].notna()]
        for bucket in SPREAD_BUCKETS:
            group = working[working["_spread_bucket"] == bucket]
            net_return_by_spread_bucket[bucket] = aggregate_net_return_bucket(
                group.to_dict(orient="records")
            )

    net_return_by_cost_bucket: dict[str, dict[str, Any]] = {}
    if "cost_bps" in df.columns:
        cost_series = pd.to_numeric(df["cost_bps"], errors="coerce")
        bucketed = cost_series.apply(
            lambda value: cost_bucket(float(value)) if pd.notna(value) else None
        )
        working = df.copy()
        working["_cost_bucket"] = bucketed
        working = working[working["_cost_bucket"].notna()]
        for bucket in COST_BUCKETS:
            group = working[working["_cost_bucket"] == bucket]
            net_return_by_cost_bucket[bucket] = aggregate_net_return_bucket(
                group.to_dict(orient="records")
            )

    loss_concentration = compute_loss_concentration(net_returns)
    bucket_groups = {
        "by_hour": net_return_by_hour,
        "by_predicted_class": net_return_by_predicted_class,
        "by_confidence_bucket": net_return_by_confidence_bucket,
        "by_spread_bucket": net_return_by_spread_bucket,
        "by_cost_bucket": net_return_by_cost_bucket,
    }
    positive_concentration = compute_positive_concentration(bucket_groups)
    suspicious_regimes = identify_suspicious_regimes(
        net_return_by_hour=net_return_by_hour,
        net_return_by_predicted_class=net_return_by_predicted_class,
        net_return_by_confidence_bucket=net_return_by_confidence_bucket,
        net_return_by_spread_bucket=net_return_by_spread_bucket,
        net_return_by_cost_bucket=net_return_by_cost_bucket,
        loss_concentration=loss_concentration,
        positive_concentration=positive_concentration,
        cumulative_net_return_bps=cumulative_net_return_bps,
    )
    recommended_next_filters = recommend_next_filters(
        net_return_by_hour=net_return_by_hour,
        net_return_by_predicted_class=net_return_by_predicted_class,
        net_return_by_confidence_bucket=net_return_by_confidence_bucket,
        net_return_by_spread_bucket=net_return_by_spread_bucket,
        net_return_by_cost_bucket=net_return_by_cost_bucket,
        suspicious_regimes=suspicious_regimes,
        cumulative_net_return_bps=cumulative_net_return_bps,
    )

    return {
        "trade_count": trade_count,
        "cumulative_net_return_bps": cumulative_net_return_bps,
        "average_net_return_bps": average_net_return_bps,
        "win_rate": win_rate,
        "best_trades": best_trades,
        "worst_trades": worst_trades,
        "net_return_by_hour": net_return_by_hour,
        "net_return_by_predicted_class": net_return_by_predicted_class,
        "net_return_by_confidence_bucket": net_return_by_confidence_bucket,
        "net_return_by_spread_bucket": net_return_by_spread_bucket,
        "net_return_by_cost_bucket": net_return_by_cost_bucket,
        "loss_concentration": loss_concentration,
        "positive_concentration": positive_concentration,
        "suspicious_regimes": suspicious_regimes,
        "recommended_next_filters": recommended_next_filters,
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "safety_note": SAFETY_NOTE,
    }


def _analysis_stamp(analysis: dict[str, Any]) -> str:
    generated_at = analysis.get("generated_at")
    if isinstance(generated_at, str) and generated_at:
        parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        return parsed.strftime("%Y%m%dT%H%M%SZ")
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_analysis_json(
    analysis: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    json_name: str | None = None,
) -> Path:
    """Write analysis JSON under reports/accepted_trades_analysis/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    coin = str(analysis.get("coin", "UNKNOWN")).upper()
    stamp = _analysis_stamp(analysis)
    filename = json_name or f"accepted_trades_analysis_{coin}_{stamp}.json"
    output_path = output_dir / filename
    output_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    analysis["json_report_path"] = str(output_path)
    return output_path


def _format_bps(value: float) -> str:
    return f"{value:.2f} bps"


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _bucket_table(
    title: str,
    buckets: dict[str, dict[str, Any]],
    *,
    ordered_keys: tuple[str, ...] | None = None,
) -> list[str]:
    lines = [f"## {title}", "", "| Bucket | Trades | Cumulative net | Avg net | Win rate |", "| --- | ---: | ---: | ---: | ---: |"]
    keys = list(ordered_keys) if ordered_keys else sorted(buckets.keys())
    for key in keys:
        metrics = buckets.get(key) or _empty_bucket_metrics()
        lines.append(
            f"| {key} | {metrics['trades']} | "
            f"{_format_bps(float(metrics['cumulative_net_return_bps']))} | "
            f"{_format_bps(float(metrics['average_net_return_bps']))} | "
            f"{_format_pct(float(metrics['win_rate']))} |"
        )
    lines.append("")
    return lines


def write_analysis_markdown(
    analysis: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    markdown_name: str | None = None,
) -> Path:
    """Write analysis Markdown under reports/accepted_trades_analysis/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    coin = str(analysis.get("coin", "UNKNOWN")).upper()
    stamp = _analysis_stamp(analysis)
    filename = markdown_name or f"accepted_trades_analysis_{coin}_{stamp}.md"
    output_path = output_dir / filename

    loss = analysis.get("loss_concentration", {})
    positive = analysis.get("positive_concentration", {})

    lines = [
        "# Accepted Paper Trades Analysis",
        "",
        f"Generated at: {analysis.get('generated_at', 'n/a')}",
        f"Source CSV: {analysis.get('source_csv_path', 'n/a')}",
        f"Coin: {coin}",
        "",
        "## Summary",
        "",
        f"- trade_count: {analysis.get('trade_count', 0)}",
        f"- cumulative_net_return_bps: {_format_bps(float(analysis.get('cumulative_net_return_bps', 0.0)))}",
        f"- average_net_return_bps: {_format_bps(float(analysis.get('average_net_return_bps', 0.0)))}",
        f"- win_rate: {_format_pct(float(analysis.get('win_rate', 0.0)))}",
        f"- paper_only: {str(analysis.get('paper_only', True)).lower()}",
        f"- execution_enabled: {str(analysis.get('execution_enabled', False)).lower()}",
        "",
        "## Why relaxed Candidate B may lose",
        "",
    ]

    cumulative = float(analysis.get("cumulative_net_return_bps", 0.0))
    if cumulative >= 0:
        lines.append(
            "- This sample is not net negative; relaxed-filter losses may appear "
            "only when trade count scales beyond strict Candidate B."
        )
    else:
        lines.append(
            f"- Net negative after costs ({_format_bps(cumulative)}), often from "
            "admitting more marginal trades when filters relax."
        )
        if loss.get("concentration_warning"):
            lines.append(
                f"- Losses are concentrated: worst trades are "
                f"{_format_pct(float(loss.get('worst_trades_pct_of_total_loss', 0.0)))} "
                "of total losses."
            )
        if positive.get("concentration_warning"):
            lines.append(
                f"- Gains are concentrated in {positive.get('best_bucket_group')}/"
                f"{positive.get('best_bucket_key')} "
                f"({_format_pct(float(positive.get('best_bucket_pct_of_positive_net', 0.0)))} "
                "of positive net)."
            )

    lines.extend(["", "## Best trades", ""])
    best_trades = analysis.get("best_trades", [])
    if not best_trades:
        lines.append("No trades.")
    else:
        for index, trade in enumerate(best_trades, start=1):
            lines.append(
                f"{index}. {trade.get('timestamp_iso', 'n/a')} "
                f"{trade.get('predicted_class', 'n/a')} "
                f"net={_format_bps(float(trade.get('net_return_bps', 0.0)))}"
            )

    lines.extend(["", "## Worst trades", ""])
    worst_trades = analysis.get("worst_trades", [])
    if not worst_trades:
        lines.append("No trades.")
    else:
        for index, trade in enumerate(worst_trades, start=1):
            lines.append(
                f"{index}. {trade.get('timestamp_iso', 'n/a')} "
                f"{trade.get('predicted_class', 'n/a')} "
                f"net={_format_bps(float(trade.get('net_return_bps', 0.0)))}"
            )

    lines.extend(
        _bucket_table("Net return by hour (UTC)", analysis.get("net_return_by_hour", {}))
    )
    lines.extend(
        _bucket_table(
            "Net return by predicted class",
            analysis.get("net_return_by_predicted_class", {}),
            ordered_keys=LABELS,
        )
    )

    confidence_buckets = analysis.get("net_return_by_confidence_bucket", {})
    if confidence_buckets:
        lines.extend(
            _bucket_table(
                "Net return by confidence bucket",
                confidence_buckets,
                ordered_keys=CONFIDENCE_BUCKETS,
            )
        )

    spread_buckets = analysis.get("net_return_by_spread_bucket", {})
    if spread_buckets:
        lines.extend(
            _bucket_table(
                "Net return by spread bucket",
                spread_buckets,
                ordered_keys=SPREAD_BUCKETS,
            )
        )

    cost_buckets = analysis.get("net_return_by_cost_bucket", {})
    if cost_buckets:
        lines.extend(
            _bucket_table(
                "Net return by cost bucket",
                cost_buckets,
                ordered_keys=COST_BUCKETS,
            )
        )

    lines.extend(
        [
            "## Loss concentration",
            "",
            f"- worst_trades_pct_of_total_loss: "
            f"{_format_pct(float(loss.get('worst_trades_pct_of_total_loss', 0.0)))}",
            f"- total_loss_bps: {_format_bps(float(loss.get('total_loss_bps', 0.0)))}",
            "",
            "## Positive concentration",
            "",
            f"- best_bucket: {positive.get('best_bucket_group')}/{positive.get('best_bucket_key')}",
            f"- best_bucket_pct_of_positive_net: "
            f"{_format_pct(float(positive.get('best_bucket_pct_of_positive_net', 0.0)))}",
            "",
            "## Suspicious regimes",
            "",
        ]
    )

    regimes = analysis.get("suspicious_regimes", [])
    if not regimes:
        lines.append("No suspicious regimes flagged.")
    else:
        for index, regime in enumerate(regimes, start=1):
            lines.append(f"{index}. **{regime.get('regime_type')}** — {regime.get('detail')}")

    lines.extend(["", "## Recommended next filters to test", ""])
    filters = analysis.get("recommended_next_filters", [])
    if not filters:
        lines.append("No filter suggestions.")
    else:
        for index, rule in enumerate(filters, start=1):
            lines.append(f"{index}. **{rule.get('filter_type')}**: {rule.get('action')}")
            lines.append(f"   - Reason: {rule.get('reason')}")

    lines.extend(
        [
            "",
            "## Research-only disclaimer",
            "",
            RESEARCH_ONLY_DISCLAIMER,
            "",
            SAFETY_STATEMENT,
            "",
            SAFETY_NOTE,
            "",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    analysis["markdown_report_path"] = str(output_path)
    return output_path


def run_analysis(
    csv_path: Path,
    *,
    coin: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    json_name: str | None = None,
    markdown_name: str | None = None,
) -> dict[str, Any]:
    """Load, analyze, and write reports for one accepted trades CSV."""
    df = load_accepted_trades_csv(csv_path)
    analysis = analyze_accepted_trades(df)
    analysis["generated_at"] = datetime.now(timezone.utc).isoformat()
    analysis["source_csv_path"] = str(csv_path)
    analysis["coin"] = coin.upper()
    write_analysis_json(analysis, output_dir=output_dir, json_name=json_name)
    write_analysis_markdown(analysis, output_dir=output_dir, markdown_name=markdown_name)
    return analysis


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze accepted paper trades CSVs "
            "(research-only, no execution)."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze one accepted trades CSV and write JSON/Markdown reports",
    )
    analyze_parser.add_argument(
        "--coin",
        default=DEFAULT_COIN,
        help=f"Coin label for latest CSV discovery (default: {DEFAULT_COIN})",
    )
    analyze_parser.add_argument(
        "--csv",
        type=Path,
        help="Explicit path to an accepted trades CSV",
    )
    analyze_parser.add_argument(
        "--latest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use latest accepted_trades CSV when --csv is omitted (default: true)",
    )
    analyze_parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help=f"Directory to search for accepted trades CSVs (default: {DEFAULT_REPORTS_DIR})",
    )
    analyze_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for analysis reports (default: {DEFAULT_OUTPUT_DIR})",
    )
    analyze_parser.add_argument(
        "--json-name",
        help="Optional JSON output filename",
    )
    analyze_parser.add_argument(
        "--markdown-name",
        help="Optional Markdown output filename",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command != "analyze":
        parser.print_help()
        return 1

    try:
        csv_path = resolve_accepted_trades_csv(
            coin=args.coin,
            csv_path=args.csv,
            reports_dir=args.reports_dir,
            use_latest=args.latest,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 1

    analysis = run_analysis(
        csv_path,
        coin=args.coin,
        output_dir=args.output_dir,
        json_name=args.json_name,
        markdown_name=args.markdown_name,
    )

    print(f"source_csv_path={analysis['source_csv_path']}")
    print(f"json_report_path={analysis['json_report_path']}")
    print(f"markdown_report_path={analysis['markdown_report_path']}")
    print(f"trade_count={analysis['trade_count']}")
    print(f"cumulative_net_return_bps={analysis['cumulative_net_return_bps']:.2f}")
    print(f"average_net_return_bps={analysis['average_net_return_bps']:.2f}")
    print(f"win_rate={analysis['win_rate']:.4f}")
    for regime in analysis.get("suspicious_regimes", [])[:3]:
        print(f"suspicious_regime: {regime.get('detail')}")
    print(SAFETY_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
