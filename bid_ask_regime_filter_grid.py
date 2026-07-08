"""Standalone post-signal regime filter grid for accepted paper trades (research-only).

Applies exploratory post-hoc filters to accepted paper trades CSVs and ranks
filtered variants for hypothesis generation. Does not retrain models and does
not claim market readiness.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bid_ask_accepted_trades_analyzer import (
    compute_loss_concentration,
    discover_latest_accepted_trades_csv,
    load_accepted_trades_csv,
    resolve_accepted_trades_csv,
    spread_bucket,
)
from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT

DEFAULT_COIN = "BTC"
DEFAULT_REPORTS_DIR = Path("reports/bid_ask_walk_forward")
DEFAULT_OUTPUT_DIR = Path("reports/regime_filter_grid")

DEFAULT_EXCLUDE_HOURS_LIST = ("", "3", "5", "22", "3,5", "3,5,22")
DEFAULT_MAX_COST_BPS_LIST = ("", "5", "6", "7")
DEFAULT_SPREAD_BUCKET_LIST = ("", "tight", "normal")

DEFAULT_MIN_TRADES = 5
DEFAULT_TOP_N = 20

DIAGNOSTIC_SCORE_POSITIVE_AFTER_COSTS_BONUS = 1_000.0
DIAGNOSTIC_SCORE_TRADE_COUNT_WEIGHT = 10.0
DIAGNOSTIC_SCORE_AVERAGE_NET_RETURN_WEIGHT = 100.0
DIAGNOSTIC_SCORE_REMOVED_LOSS_BONUS_WEIGHT = 2.0
DIAGNOSTIC_SCORE_REMOVED_GAIN_PENALTY_WEIGHT = 3.0
DIAGNOSTIC_SCORE_TOO_FEW_TRADES_PENALTY = 500.0
DIAGNOSTIC_SCORE_TOO_MANY_REMOVED_PENALTY = 250.0
TOO_MANY_REMOVED_FRACTION = 0.50

SAFETY_NOTE = (
    "Regime filter grid is paper-only and research-only. Reads local accepted "
    "trades CSVs only. No orders, execution, wallets, AWS, network, or shell "
    "commands."
)

EXPLORATORY_DISCLAIMER = (
    "These are post-hoc exploratory filters. They are useful for finding "
    "hypotheses, not proof."
)

REQUIRED_COLUMNS = (
    "coin",
    "timestamp_iso",
    "predicted_class",
    "net_return_bps",
)


@dataclass(frozen=True)
class RegimeFilterSpec:
    exclude_hours: frozenset[str]
    max_cost_bps: float | None
    spread_bucket: str | None

    @property
    def filter_id(self) -> str:
        parts: list[str] = []
        if self.exclude_hours:
            hours = ",".join(sorted(self.exclude_hours))
            parts.append(f"exclude_hours={hours}")
        else:
            parts.append("exclude_hours=none")
        if self.max_cost_bps is not None:
            parts.append(f"max_cost_bps={self.max_cost_bps:g}")
        else:
            parts.append("max_cost_bps=none")
        if self.spread_bucket:
            parts.append(f"spread_bucket={self.spread_bucket}")
        else:
            parts.append("spread_bucket=none")
        return "|".join(parts)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _report_stamp(run_timestamp: str) -> str:
    if run_timestamp:
        return run_timestamp.replace(":", "").replace("+", "").replace("-", "")[:15]
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_semicolon_list(raw: str, *, default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse semicolon-separated grid dimension values."""
    text = str(raw).strip()
    if not text:
        return default
    return tuple(item.strip() for item in text.split(";"))


def parse_exclude_hours(raw: str) -> frozenset[str]:
    """Parse comma-separated UTC hour tokens like '3,5,22' into zero-padded hours."""
    text = str(raw).strip()
    if not text:
        return frozenset()
    hours: set[str] = set()
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        hour = int(token)
        if hour < 0 or hour > 23:
            raise ValueError(f"Invalid hour token: {token}")
        hours.add(f"{hour:02d}")
    return frozenset(hours)


def parse_optional_float(raw: str) -> float | None:
    text = str(raw).strip()
    if not text:
        return None
    return float(text)


def parse_optional_spread_bucket(raw: str) -> str | None:
    text = str(raw).strip().lower()
    if not text:
        return None
    if text not in {"tight", "normal", "wide", "too_wide"}:
        raise ValueError(f"Unsupported spread bucket: {raw}")
    return text


def build_filter_combinations(
    *,
    exclude_hours_list: tuple[str, ...] = DEFAULT_EXCLUDE_HOURS_LIST,
    max_cost_bps_list: tuple[str, ...] = DEFAULT_MAX_COST_BPS_LIST,
    spread_bucket_list: tuple[str, ...] = DEFAULT_SPREAD_BUCKET_LIST,
) -> list[RegimeFilterSpec]:
    """Build the cartesian product of exploratory regime filter specs."""
    specs: list[RegimeFilterSpec] = []
    for exclude_raw, max_cost_raw, spread_raw in itertools.product(
        exclude_hours_list,
        max_cost_bps_list,
        spread_bucket_list,
    ):
        specs.append(
            RegimeFilterSpec(
                exclude_hours=parse_exclude_hours(exclude_raw),
                max_cost_bps=parse_optional_float(max_cost_raw),
                spread_bucket=parse_optional_spread_bucket(spread_raw),
            )
        )
    return specs


def _parse_timestamp_hour(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        return f"{parsed.hour:02d}"
    except ValueError:
        return None


def enrich_accepted_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived hour and spread_bucket columns when possible."""
    working = df.copy()
    working["_hour"] = working["timestamp_iso"].map(_parse_timestamp_hour)

    if "spread_bucket" in working.columns:
        working["_spread_bucket"] = working["spread_bucket"].astype(str).str.lower()
    elif "spread_bps" in working.columns:
        spread_values = pd.to_numeric(working["spread_bps"], errors="coerce")
        working["_spread_bucket"] = spread_values.map(
            lambda value: spread_bucket(float(value)) if pd.notna(value) else None
        )
    else:
        working["_spread_bucket"] = None

    if "cost_bps" in working.columns:
        working["_cost_bps"] = pd.to_numeric(working["cost_bps"], errors="coerce")
    else:
        working["_cost_bps"] = pd.NA

    return working


def apply_regime_filter(df: pd.DataFrame, spec: RegimeFilterSpec) -> pd.DataFrame:
    """Return rows that pass the exploratory regime filter."""
    mask = pd.Series(True, index=df.index)

    if spec.exclude_hours:
        hour_mask = df["_hour"].notna() & ~df["_hour"].isin(spec.exclude_hours)
        mask &= hour_mask

    if spec.max_cost_bps is not None:
        if "_cost_bps" not in df.columns or df["_cost_bps"].isna().all():
            return df.iloc[0:0].copy()
        mask &= df["_cost_bps"].notna() & (df["_cost_bps"] <= spec.max_cost_bps)

    if spec.spread_bucket is not None:
        if df["_spread_bucket"].isna().all():
            return df.iloc[0:0].copy()
        mask &= df["_spread_bucket"].notna() & (df["_spread_bucket"] == spec.spread_bucket)

    return df[mask].copy()


def _trade_metrics(net_returns: list[float]) -> dict[str, float | int]:
    if not net_returns:
        return {
            "trades": 0,
            "cumulative_net_return_bps": 0.0,
            "average_net_return_bps": 0.0,
            "win_rate": 0.0,
        }
    cumulative = float(sum(net_returns))
    trades = len(net_returns)
    wins = sum(1 for value in net_returns if value > 0)
    return {
        "trades": trades,
        "cumulative_net_return_bps": cumulative,
        "average_net_return_bps": cumulative / trades,
        "win_rate": wins / trades,
    }


def _removed_return_sums(before_df: pd.DataFrame, after_df: pd.DataFrame) -> tuple[float, float]:
    removed_indices = set(before_df.index) - set(after_df.index)
    removed_returns = [
        float(before_df.loc[index, "net_return_bps"]) for index in removed_indices
    ]
    removed_loss_bps = float(sum(value for value in removed_returns if value < 0))
    removed_gain_bps = float(sum(value for value in removed_returns if value > 0))
    return removed_loss_bps, removed_gain_bps


def evaluate_filter_variant(
    df: pd.DataFrame,
    spec: RegimeFilterSpec,
    *,
    min_trades: int = DEFAULT_MIN_TRADES,
) -> dict[str, Any]:
    """Evaluate one exploratory filter combination on accepted trades."""
    before_returns = [float(value) for value in df["net_return_bps"].tolist()]
    filtered = apply_regime_filter(df, spec)
    after_returns = [float(value) for value in filtered["net_return_bps"].tolist()]

    before_metrics = _trade_metrics(before_returns)
    after_metrics = _trade_metrics(after_returns)
    removed_loss_bps, removed_gain_bps = _removed_return_sums(df, filtered)
    concentration = compute_loss_concentration(after_returns)

    trades_before = int(before_metrics["trades"])
    trades_after = int(after_metrics["trades"])
    trades_removed = trades_before - trades_after
    positive_after_costs = float(after_metrics["cumulative_net_return_bps"]) > 0
    meets_min_trades = trades_after >= min_trades
    removed_fraction = trades_removed / trades_before if trades_before else 0.0
    too_many_removed = removed_fraction > TOO_MANY_REMOVED_FRACTION

    score_details = compute_diagnostic_score(
        trades_before=trades_before,
        trades_after=trades_after,
        cumulative_net_return_bps_after=float(after_metrics["cumulative_net_return_bps"]),
        average_net_return_bps_after=float(after_metrics["average_net_return_bps"]),
        removed_loss_bps=removed_loss_bps,
        removed_gain_bps=removed_gain_bps,
        min_trades=min_trades,
    )

    return {
        "filter_id": spec.filter_id,
        "exclude_hours": sorted(spec.exclude_hours),
        "max_cost_bps": spec.max_cost_bps,
        "spread_bucket": spec.spread_bucket,
        "trades_before": trades_before,
        "trades_after": trades_after,
        "trades_removed": trades_removed,
        "cumulative_net_return_bps_before": float(before_metrics["cumulative_net_return_bps"]),
        "cumulative_net_return_bps_after": float(after_metrics["cumulative_net_return_bps"]),
        "average_net_return_bps_after": float(after_metrics["average_net_return_bps"]),
        "win_rate_after": float(after_metrics["win_rate"]),
        "removed_loss_bps": removed_loss_bps,
        "removed_gain_bps": removed_gain_bps,
        "removed_fraction": removed_fraction,
        "positive_after_costs": positive_after_costs,
        "meets_min_trades": meets_min_trades,
        "too_many_removed": too_many_removed,
        "concentration_warning": bool(concentration.get("concentration_warning")),
        "exploratory_overfit_warning": True,
        **score_details,
    }


def compute_diagnostic_score(
    *,
    trades_before: int,
    trades_after: int,
    cumulative_net_return_bps_after: float,
    average_net_return_bps_after: float,
    removed_loss_bps: float,
    removed_gain_bps: float,
    min_trades: int = DEFAULT_MIN_TRADES,
) -> dict[str, Any]:
    """Rank exploratory filter variants for paper diagnostics."""
    positive_bonus = (
        DIAGNOSTIC_SCORE_POSITIVE_AFTER_COSTS_BONUS
        if cumulative_net_return_bps_after > 0
        else 0.0
    )
    trade_bonus = trades_after * DIAGNOSTIC_SCORE_TRADE_COUNT_WEIGHT
    average_return_bonus = (
        max(average_net_return_bps_after, 0.0) * DIAGNOSTIC_SCORE_AVERAGE_NET_RETURN_WEIGHT
    )
    removed_loss_bonus = abs(min(removed_loss_bps, 0.0)) * DIAGNOSTIC_SCORE_REMOVED_LOSS_BONUS_WEIGHT
    removed_gain_penalty = removed_gain_bps * DIAGNOSTIC_SCORE_REMOVED_GAIN_PENALTY_WEIGHT
    too_few_trades = trades_after < min_trades
    too_few_trades_penalty = (
        DIAGNOSTIC_SCORE_TOO_FEW_TRADES_PENALTY if too_few_trades else 0.0
    )
    removed_fraction = (trades_before - trades_after) / trades_before if trades_before else 0.0
    too_many_removed = removed_fraction > TOO_MANY_REMOVED_FRACTION
    too_many_removed_penalty = (
        DIAGNOSTIC_SCORE_TOO_MANY_REMOVED_PENALTY if too_many_removed else 0.0
    )

    diagnostic_score = (
        positive_bonus
        + trade_bonus
        + average_return_bonus
        + removed_loss_bonus
        - removed_gain_penalty
        - too_few_trades_penalty
        - too_many_removed_penalty
    )

    return {
        "diagnostic_score": diagnostic_score,
        "positive_after_costs_bonus": positive_bonus,
        "trade_count_bonus": trade_bonus,
        "average_net_return_bonus": average_return_bonus,
        "removed_loss_bonus": removed_loss_bonus,
        "removed_gain_penalty": removed_gain_penalty,
        "too_few_trades_penalty": too_few_trades_penalty,
        "too_many_removed_penalty": too_many_removed_penalty,
        "too_few_trades": too_few_trades,
        "real_execution_ready": False,
    }


def rank_filter_variants(
    variants: list[dict[str, Any]],
    *,
    min_trades: int = DEFAULT_MIN_TRADES,
) -> list[dict[str, Any]]:
    """Rank variants by diagnostic score, keeping only min-trades survivors in order."""
    eligible = [variant for variant in variants if int(variant.get("trades_after", 0)) >= min_trades]
    ranked = sorted(
        eligible,
        key=lambda row: (
            float(row.get("diagnostic_score", 0.0)),
            int(row.get("trades_after", 0)),
            float(row.get("cumulative_net_return_bps_after", 0.0)),
        ),
        reverse=True,
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def evaluate_regime_filter_grid(
    df: pd.DataFrame,
    *,
    exclude_hours_list: tuple[str, ...] = DEFAULT_EXCLUDE_HOURS_LIST,
    max_cost_bps_list: tuple[str, ...] = DEFAULT_MAX_COST_BPS_LIST,
    spread_bucket_list: tuple[str, ...] = DEFAULT_SPREAD_BUCKET_LIST,
    min_trades: int = DEFAULT_MIN_TRADES,
) -> list[dict[str, Any]]:
    """Evaluate all exploratory filter combinations."""
    enriched = enrich_accepted_trades(df)
    specs = build_filter_combinations(
        exclude_hours_list=exclude_hours_list,
        max_cost_bps_list=max_cost_bps_list,
        spread_bucket_list=spread_bucket_list,
    )
    return [
        evaluate_filter_variant(enriched, spec, min_trades=min_trades)
        for spec in specs
    ]


def build_regime_filter_report(
    *,
    coin: str,
    source_csv_path: Path,
    variants: list[dict[str, Any]],
    min_trades: int,
    top_n: int,
    run_timestamp: str | None = None,
) -> dict[str, Any]:
    """Build the full regime filter grid report payload."""
    run_timestamp = run_timestamp or _utc_now_iso()
    ranked = rank_filter_variants(variants, min_trades=min_trades)
    baseline = next(
        (
            variant
            for variant in variants
            if not variant.get("exclude_hours")
            and variant.get("max_cost_bps") is None
            and not variant.get("spread_bucket")
        ),
        None,
    )
    return {
        "generated_at": run_timestamp,
        "run_timestamp": run_timestamp,
        "coin": coin.upper(),
        "source_csv_path": str(source_csv_path),
        "variant_count": len(variants),
        "ranked_count": len(ranked),
        "min_trades": min_trades,
        "top_n": ranked[:top_n],
        "ranked": ranked,
        "variants": variants,
        "baseline_variant": baseline,
        "best_variant": ranked[0] if ranked else None,
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "real_execution_ready": False,
        "exploratory_overfit_warning": True,
        "safety_note": SAFETY_NOTE,
        "exploratory_disclaimer": EXPLORATORY_DISCLAIMER,
        "safety_statement": SAFETY_STATEMENT,
    }


def write_regime_filter_csv(
    report: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write ranked regime filter variants CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    coin = str(report.get("coin", "UNKNOWN")).upper()
    stamp = _report_stamp(str(report.get("run_timestamp", "")))
    output_path = output_dir / f"regime_filter_grid_{coin}_{stamp}.csv"

    fieldnames = [
        "rank",
        "filter_id",
        "exclude_hours",
        "max_cost_bps",
        "spread_bucket",
        "diagnostic_score",
        "trades_before",
        "trades_after",
        "trades_removed",
        "cumulative_net_return_bps_before",
        "cumulative_net_return_bps_after",
        "average_net_return_bps_after",
        "win_rate_after",
        "removed_loss_bps",
        "removed_gain_bps",
        "positive_after_costs",
        "concentration_warning",
        "exploratory_overfit_warning",
        "meets_min_trades",
        "real_execution_ready",
    ]

    ranked = report.get("ranked") or []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked:
            writer.writerow(
                {
                    "rank": row.get("rank"),
                    "filter_id": row.get("filter_id"),
                    "exclude_hours": ",".join(row.get("exclude_hours") or []),
                    "max_cost_bps": row.get("max_cost_bps"),
                    "spread_bucket": row.get("spread_bucket"),
                    "diagnostic_score": row.get("diagnostic_score"),
                    "trades_before": row.get("trades_before"),
                    "trades_after": row.get("trades_after"),
                    "trades_removed": row.get("trades_removed"),
                    "cumulative_net_return_bps_before": row.get("cumulative_net_return_bps_before"),
                    "cumulative_net_return_bps_after": row.get("cumulative_net_return_bps_after"),
                    "average_net_return_bps_after": row.get("average_net_return_bps_after"),
                    "win_rate_after": row.get("win_rate_after"),
                    "removed_loss_bps": row.get("removed_loss_bps"),
                    "removed_gain_bps": row.get("removed_gain_bps"),
                    "positive_after_costs": row.get("positive_after_costs"),
                    "concentration_warning": row.get("concentration_warning"),
                    "exploratory_overfit_warning": row.get("exploratory_overfit_warning"),
                    "meets_min_trades": row.get("meets_min_trades"),
                    "real_execution_ready": row.get("real_execution_ready"),
                }
            )

    report["csv_report_path"] = str(output_path)
    return output_path


def write_regime_filter_json(
    report: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write regime filter grid JSON report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    coin = str(report.get("coin", "UNKNOWN")).upper()
    stamp = _report_stamp(str(report.get("run_timestamp", "")))
    output_path = output_dir / f"regime_filter_grid_{coin}_{stamp}.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["json_report_path"] = str(output_path)
    return output_path


def write_regime_filter_markdown(
    report: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write regime filter grid Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    coin = str(report.get("coin", "UNKNOWN")).upper()
    stamp = _report_stamp(str(report.get("run_timestamp", "")))
    output_path = output_dir / f"regime_filter_grid_{coin}_{stamp}.md"

    top_n = report.get("top_n") or []
    baseline = report.get("baseline_variant") or {}
    best = report.get("best_variant")

    lines = [
        "# Regime Filter Grid Report",
        "",
        f"Generated at: {report.get('generated_at', 'n/a')}",
        f"Source CSV: {report.get('source_csv_path', 'n/a')}",
        f"Coin: {coin}",
        f"Variants evaluated: {report.get('variant_count', 0)}",
        f"Ranked variants (min_trades={report.get('min_trades', DEFAULT_MIN_TRADES)}): "
        f"{report.get('ranked_count', 0)}",
        "",
        "## Safety",
        "",
        f"- paper_only: {str(report.get('paper_only', True)).lower()}",
        f"- execution_enabled: {str(report.get('execution_enabled', False)).lower()}",
        f"- real_execution_ready: {str(report.get('real_execution_ready', False)).lower()}",
        f"- exploratory_overfit_warning: true",
        "",
        report.get("safety_note", SAFETY_NOTE),
        "",
        "## Exploratory Disclaimer",
        "",
        report.get("exploratory_disclaimer", EXPLORATORY_DISCLAIMER),
        "",
        "## Baseline (No Filters)",
        "",
        f"- trades: {baseline.get('trades_after', 0)}",
        f"- cumulative_net_return_bps: {float(baseline.get('cumulative_net_return_bps_after', 0.0)):.2f}",
        f"- average_net_return_bps: {float(baseline.get('average_net_return_bps_after', 0.0)):.2f}",
        "",
        "## Top Filter Variants",
        "",
        "| Rank | Filter | Score | Trades | Cum Net (bps) | Avg Net (bps) | Positive | Overfit Warn |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    for row in top_n:
        lines.append(
            f"| {row.get('rank', '-')} | {row.get('filter_id', 'n/a')} | "
            f"{float(row.get('diagnostic_score', 0.0)):.2f} | "
            f"{row.get('trades_after', 0)} | "
            f"{float(row.get('cumulative_net_return_bps_after', 0.0)):.2f} | "
            f"{float(row.get('average_net_return_bps_after', 0.0)):.2f} | "
            f"{str(row.get('positive_after_costs', False)).lower()} | true |"
        )

    if not top_n:
        lines.append("| - | - | - | - | - | - | - | - |")

    lines.extend(["", "## Best Variant", ""])
    if best is None:
        lines.append("No variants met min_trades.")
    else:
        lines.extend(
            [
                f"- filter_id: {best.get('filter_id')}",
                f"- diagnostic_score: {float(best.get('diagnostic_score', 0.0)):.2f}",
                f"- trades_after: {best.get('trades_after', 0)}",
                f"- cumulative_net_return_bps_after: "
                f"{float(best.get('cumulative_net_return_bps_after', 0.0)):.2f}",
                f"- removed_loss_bps: {float(best.get('removed_loss_bps', 0.0)):.2f}",
                f"- removed_gain_bps: {float(best.get('removed_gain_bps', 0.0)):.2f}",
                f"- exploratory_overfit_warning: true",
                f"- real_execution_ready: false",
            ]
        )

    lines.extend(
        [
            "",
            "## Research Interpretation",
            "",
            EXPLORATORY_DISCLAIMER,
            "",
            "Post-hoc regime filters can look strong on the same sample that suggested "
            "them. Treat ranked variants as hypotheses to test on fresh walk-forward "
            "windows, not as validated trading rules.",
            "",
            report.get("safety_statement", SAFETY_STATEMENT),
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["markdown_report_path"] = str(output_path)
    return output_path


def run_regime_filter_grid(
    *,
    coin: str = DEFAULT_COIN,
    csv_path: Path | None = None,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    use_latest: bool = True,
    exclude_hours_list: tuple[str, ...] = DEFAULT_EXCLUDE_HOURS_LIST,
    max_cost_bps_list: tuple[str, ...] = DEFAULT_MAX_COST_BPS_LIST,
    spread_bucket_list: tuple[str, ...] = DEFAULT_SPREAD_BUCKET_LIST,
    min_trades: int = DEFAULT_MIN_TRADES,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, Any]:
    """Load accepted trades and run the exploratory regime filter grid."""
    resolved_csv = resolve_accepted_trades_csv(
        coin=coin,
        csv_path=csv_path,
        reports_dir=reports_dir,
        use_latest=use_latest,
    )
    df = load_accepted_trades_csv(resolved_csv)
    variants = evaluate_regime_filter_grid(
        df,
        exclude_hours_list=exclude_hours_list,
        max_cost_bps_list=max_cost_bps_list,
        spread_bucket_list=spread_bucket_list,
        min_trades=min_trades,
    )
    report = build_regime_filter_report(
        coin=coin,
        source_csv_path=resolved_csv,
        variants=variants,
        min_trades=min_trades,
        top_n=top_n,
    )
    write_regime_filter_json(report, output_dir=output_dir)
    write_regime_filter_csv(report, output_dir=output_dir)
    write_regime_filter_markdown(report, output_dir=output_dir)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone post-signal regime filter grid for accepted paper trades "
            "(research-only, no real execution)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run exploratory regime filter grid on accepted trades CSV",
    )
    run_parser.add_argument("--coin", default=DEFAULT_COIN, help="Coin symbol (default: BTC)")
    run_parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Explicit accepted trades CSV path",
    )
    run_parser.add_argument(
        "--latest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use latest accepted trades CSV when --csv is omitted (default: true)",
    )
    run_parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory containing accepted_trades CSV artifacts",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for regime filter grid reports",
    )
    run_parser.add_argument(
        "--exclude-hours-list",
        default="",
        help=(
            "Semicolon-separated exclude-hour specs (comma-separated hours within each "
            f"spec). Default: {';'.join(DEFAULT_EXCLUDE_HOURS_LIST)}"
        ),
    )
    run_parser.add_argument(
        "--max-cost-bps-list",
        default="",
        help=(
            "Semicolon-separated max cost bps thresholds. "
            f"Default: {';'.join(DEFAULT_MAX_COST_BPS_LIST)}"
        ),
    )
    run_parser.add_argument(
        "--spread-bucket-list",
        default="",
        help=(
            "Semicolon-separated spread bucket keep filters. "
            f"Default: {';'.join(DEFAULT_SPREAD_BUCKET_LIST)}"
        ),
    )
    run_parser.add_argument(
        "--min-trades",
        type=int,
        default=DEFAULT_MIN_TRADES,
        help="Minimum trades required for ranked variants",
    )
    run_parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of top ranked variants to include in summary output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.error(f"Unsupported command: {args.command}")

    exclude_hours_list = parse_semicolon_list(
        args.exclude_hours_list,
        default=DEFAULT_EXCLUDE_HOURS_LIST,
    )
    max_cost_bps_list = parse_semicolon_list(
        args.max_cost_bps_list,
        default=DEFAULT_MAX_COST_BPS_LIST,
    )
    spread_bucket_list = parse_semicolon_list(
        args.spread_bucket_list,
        default=DEFAULT_SPREAD_BUCKET_LIST,
    )

    report = run_regime_filter_grid(
        coin=args.coin,
        csv_path=args.csv,
        reports_dir=args.reports_dir,
        output_dir=args.output_dir,
        use_latest=args.latest,
        exclude_hours_list=exclude_hours_list,
        max_cost_bps_list=max_cost_bps_list,
        spread_bucket_list=spread_bucket_list,
        min_trades=args.min_trades,
        top_n=args.top_n,
    )

    baseline = report.get("baseline_variant") or {}
    best = report.get("best_variant")
    print(f"coin={report.get('coin')}")
    print(f"source_csv_path={report.get('source_csv_path')}")
    print(f"variant_count={report.get('variant_count')}")
    print(f"ranked_count={report.get('ranked_count')}")
    print(
        "baseline_trades="
        f"{baseline.get('trades_after', 0)} "
        f"cumulative_net_return_bps="
        f"{float(baseline.get('cumulative_net_return_bps_after', 0.0)):.2f}"
    )
    if best:
        print(
            f"best_filter_id={best.get('filter_id')} "
            f"diagnostic_score={float(best.get('diagnostic_score', 0.0)):.2f} "
            f"trades_after={best.get('trades_after', 0)}"
        )
    else:
        print("best_filter_id=none")
    print(f"json_report_path={report.get('json_report_path')}")
    print(f"csv_report_path={report.get('csv_report_path')}")
    print(f"markdown_report_path={report.get('markdown_report_path')}")
    print(SAFETY_NOTE)
    print(EXPLORATORY_DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
