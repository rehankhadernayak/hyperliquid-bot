"""
Offline strategy experiment runner.

Replays historical prices for multiple immutable strategy profiles over identical
timestamps and coins. Never calls the network or modifies live data files.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backtester import load_price_ticks, parse_timestamp, prices_to_mids
from config import DATA, DEFAULT_EXPERIMENT, EVIDENCE, VERSION
from experiment_evidence import assess_evidence
from experiment_metrics import compute_period_metrics
from experiment_overfitting import collect_overfitting_warnings
from experiment_threshold_comparison import build_threshold_comparison
from experiment_split import (
    ChronologicalSplit,
    split_timestamps_chronologically,
    walk_forward_folds,
)
from experiment_trader import ExperimentPaperTrader
from market_analyzer import PriceHistory, update_start_prices
from market_analyzer import record_price_history, rolling_percent_change
from strategies import get_strategy
from strategy_profiles import StrategyProfile, get_profile
from profile_validation import validate_profile_fields
from strategy_types import StrategyContext


def hash_input_file(path: Path) -> str:
    """Return SHA-256 hex digest of the price log contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _silent_skip(**_kwargs) -> None:
    return None


def _evaluate_tradable_signal(
    strategy,
    profile: StrategyProfile,
    coin: str,
    current_price: float,
    price_history: PriceHistory,
    timestamp: str,
    now: float,
    trader: ExperimentPaperTrader,
) -> str | None:
    """Return a tradable signal or None when history is insufficient."""
    history = price_history.get(coin, [])
    change_since_start = rolling_percent_change(
        history,
        current_price,
        now,
        profile.lookback_seconds,
    )
    if change_since_start is None:
        return None

    context = StrategyContext(
        coin=coin,
        current_price=current_price,
        price_history=price_history,
        timestamp=timestamp,
        now=now,
        trader=trader,
        thresholds=profile.thresholds,
        lookback_seconds=profile.lookback_seconds,
    )
    result = strategy.evaluate(context)
    if result.signal in ("N/A", "FLAT"):
        return None
    return result.signal


def replay_profile(
    timestamps: list[str],
    prices_by_timestamp: dict[str, dict[str, float]],
    profile: StrategyProfile,
    *,
    close_at_end: bool | None = None,
) -> tuple[ExperimentPaperTrader, list[float], dict[str, float], str]:
    """
    Replay one profile over the given timestamps with an isolated trader.

    Price history is rebuilt from scratch — no data from outside the timestamp
    list is used (prevents look-ahead when evaluating holdout periods).
    """
    strategy = get_strategy(profile.strategy_name)
    limits = profile.trading_limits()
    trader = ExperimentPaperTrader(profile=profile, limits=limits)

    start_prices: dict[str, float] = {}
    price_history: PriceHistory = {}
    equity_curve: list[float] = [limits.starting_balance_usdc]
    current_prices: dict[str, float] = {}
    last_timestamp = timestamps[-1] if timestamps else ""

    for timestamp in timestamps:
        coin_prices = prices_by_timestamp[timestamp]
        mids = prices_to_mids(coin_prices)
        now = parse_timestamp(timestamp)

        update_start_prices(start_prices, mids)

        current_prices = {}
        for coin, price in coin_prices.items():
            current_prices[coin] = price
            record_price_history(price_history, coin, price, now)

        trader.update_session_risk(current_prices)

        for coin, price in coin_prices.items():
            if coin not in start_prices:
                continue

            signal = _evaluate_tradable_signal(
                strategy,
                profile,
                coin,
                price,
                price_history,
                timestamp,
                now,
                trader,
            )
            if signal is None:
                continue
            trader.process_signal(coin, signal, price, timestamp, now)

        trader.record_exposure_tick(current_prices)
        equity_curve.append(trader.total_equity(current_prices))
        last_timestamp = timestamp

    should_close = (
        close_at_end
        if close_at_end is not None
        else profile.close_positions_at_end
    )
    if should_close and timestamps:
        trader.force_close_all_positions(current_prices, last_timestamp)
        equity_curve.append(trader.total_equity(current_prices))

    return trader, equity_curve, current_prices, last_timestamp


def run_profile_period(
    timestamps: list[str],
    prices_by_timestamp: dict[str, dict[str, float]],
    profile: StrategyProfile,
    period_label: str,
) -> dict[str, Any]:
    """Run one profile over one chronological period."""
    if not timestamps:
        limits = profile.trading_limits()
        return {
            "period_label": period_label,
            "starting_equity": limits.starting_balance_usdc,
            "ending_equity": limits.starting_balance_usdc,
            "net_return_pct": 0.0,
            "realized_pnl": 0.0,
            "fees_paid": 0.0,
            "slippage_cost": 0.0,
            "gross_profit": 0.0,
            "costs_pct_of_gross_profit": "N/A",
            "total_buys": 0,
            "total_sells": 0,
            "completed_trades": 0,
            "winning_closed_trades": 0,
            "losing_closed_trades": 0,
            "win_rate_pct": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "profit_factor": None,
            "maximum_drawdown_pct": 0.0,
            "exposure_time_pct": 0.0,
            "longest_losing_sequence": 0,
            "average_holding_seconds": 0.0,
            "return_per_completed_trade": 0.0,
            "results_by_coin": {},
            "equity_curve": [limits.starting_balance_usdc],
            "trade_log": [],
            "paused_during_period": False,
        }

    trader, equity_curve, current_prices, _ = replay_profile(
        timestamps,
        prices_by_timestamp,
        profile,
    )
    return compute_period_metrics(
        trader,
        equity_curve,
        current_prices,
        period_label=period_label,
        limits=profile.trading_limits(),
    )


def run_experiment(
    profile_ids: list[str],
    *,
    csv_path: Path = DATA.price_log_csv,
    development_pct: float = DEFAULT_EXPERIMENT.development_pct,
    walk_forward: bool = False,
) -> dict[str, Any]:
    """
    Run multiple profiles on identical historical data and return full results.

    Never modifies live paper-trading CSV files.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Price log not found: {csv_path}")

    profiles = [get_profile(profile_id) for profile_id in profile_ids]
    for profile in profiles:
        validate_profile_fields(profile)
    timestamps, prices_by_timestamp = load_price_ticks(csv_path)
    if not timestamps:
        raise ValueError(f"No price rows found in {csv_path}")

    split = split_timestamps_chronologically(
        timestamps,
        development_pct=development_pct,
    )
    split_identical = len(split.out_of_sample) == 0 or split.development == split.out_of_sample

    experiment_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    input_hash = hash_input_file(csv_path)
    row_count = sum(len(coins) for coins in prices_by_timestamp.values())

    global_warnings: list[str] = []
    if split_identical:
        global_warnings.append(
            "Development and out-of-sample periods overlap or are identical."
        )

    profile_results: dict[str, Any] = {}

    for profile in profiles:
        frozen_profile = copy.deepcopy(profile)
        development_metrics = run_profile_period(
            list(split.development),
            prices_by_timestamp,
            frozen_profile,
            "development",
        )
        out_of_sample_metrics = run_profile_period(
            list(split.out_of_sample),
            prices_by_timestamp,
            frozen_profile,
            "out_of_sample",
        )
        combined_metrics = run_profile_period(
            list(split.combined),
            prices_by_timestamp,
            frozen_profile,
            "combined",
        )

        evidence = assess_evidence(development_metrics, out_of_sample_metrics)
        warnings = collect_overfitting_warnings(
            development=development_metrics,
            out_of_sample=out_of_sample_metrics,
            combined=combined_metrics,
            profile_count=len(profiles),
            split_identical=split_identical,
            used_future_data=False,
        )

        walk_forward_results: list[dict[str, Any]] = []
        if walk_forward:
            folds = walk_forward_folds(
                timestamps,
                min_development_timestamps=DEFAULT_EXPERIMENT.walk_forward_min_development_timestamps,
                evaluation_window=DEFAULT_EXPERIMENT.walk_forward_evaluation_window,
            )
            for fold in folds:
                fold_metrics = run_profile_period(
                    list(fold.evaluation),
                    prices_by_timestamp,
                    frozen_profile,
                    f"walk_forward_fold_{fold.fold_index}",
                )
                walk_forward_results.append(
                    {
                        "fold_index": fold.fold_index,
                        "development_timestamps": len(fold.development),
                        "evaluation_timestamps": len(fold.evaluation),
                        "metrics": fold_metrics,
                    }
                )

        profile_results[profile.profile_id] = {
            "profile": frozen_profile.to_dict(),
            "development": development_metrics,
            "out_of_sample": out_of_sample_metrics,
            "combined": combined_metrics,
            "evidence_assessment": evidence,
            "overfitting_warnings": warnings,
            "walk_forward": walk_forward_results,
        }

    threshold_comparison = build_threshold_comparison(profile_results)

    return {
        "experiment_id": experiment_id,
        "version": VERSION,
        "created_at": created_at,
        "configuration": {
            "profiles": profile_ids,
            "development_pct": development_pct,
            "walk_forward": walk_forward,
            "evidence_rules": asdict(EVIDENCE),
            "source_file": str(csv_path),
        },
        "input_data_hash": input_hash,
        "date_range": {
            "start": timestamps[0],
            "end": timestamps[-1],
        },
        "row_count": row_count,
        "timestamp_count": len(timestamps),
        "split": {
            "development_pct": development_pct,
            "development_timestamps": len(split.development),
            "out_of_sample_timestamps": len(split.out_of_sample),
            "development_start": split.development[0] if split.development else None,
            "development_end": split.development[-1] if split.development else None,
            "out_of_sample_start": split.out_of_sample[0] if split.out_of_sample else None,
            "out_of_sample_end": split.out_of_sample[-1] if split.out_of_sample else None,
        },
        "warnings": global_warnings,
        "profile_results": profile_results,
        "threshold_comparison": threshold_comparison,
    }


def save_experiment(results: dict[str, Any], output_dir: Path = DATA.experiments_dir) -> Path:
    """Write experiment JSON under data/experiments/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{results['experiment_id']}.json"

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)

    return output_path


def _json_default(value: object) -> object:
    if value == float("inf"):
        return "inf"
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline strategy profile experiments (paper trading only)."
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        required=True,
        help="Strategy profile ids to compare (e.g. contrarian_v1_test hold_baseline)",
    )
    parser.add_argument(
        "--price-log",
        type=Path,
        default=DATA.price_log_csv,
        help="Historical price CSV (default: data/price_log.csv)",
    )
    parser.add_argument(
        "--development-pct",
        type=float,
        default=DEFAULT_EXPERIMENT.development_pct,
        help="Fraction of timestamps used for development (default: 0.70)",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Also run walk-forward evaluation folds",
    )
    args = parser.parse_args()

    print("Starting offline strategy experiment (read-only, no network)...")
    print(f"  Version: {VERSION}")
    print(f"  Profiles: {', '.join(args.profiles)}")
    print(f"  Price log: {args.price_log}")

    try:
        results = run_experiment(
            args.profiles,
            csv_path=args.price_log,
            development_pct=args.development_pct,
            walk_forward=args.walk_forward,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"\nExperiment failed: {exc}")
        sys.exit(1)

    output_path = save_experiment(results)
    print(f"\nExperiment complete: {results['experiment_id']}")
    print(f"Results saved to {output_path}")
    print(f"Timestamps: {results['timestamp_count']:,} | Rows: {results['row_count']:,}")

    for profile_id, payload in results["profile_results"].items():
        dev = payload["development"]
        oos = payload["out_of_sample"]
        print(
            f"\n  {profile_id}: evidence={payload['evidence_assessment']} | "
            f"dev return={dev['net_return_pct']:+.2f}% | "
            f"OOS return={oos['net_return_pct']:+.2f}% | "
            f"completed (OOS)={oos['completed_trades']}"
        )
        for warning in payload["overfitting_warnings"]:
            print(f"    warning: {warning}")

    comparison = results.get("threshold_comparison")
    if comparison:
        print("\nThreshold comparison vs contrarian_v1_test:")
        for variant_id, summary in comparison.get("comparisons", {}).items():
            print(
                f"  {variant_id}: {summary['label']} | "
                f"trade reduction={summary['trade_count_reduction_vs_baseline']} | "
                f"OOS return diff={summary['oos_return_difference_vs_baseline']:+.2f}% | "
                f"cost reduction=${summary['total_cost_reduction_vs_baseline']:.2f}"
            )

    print(
        "\nHistorical and paper results do not guarantee future performance."
    )


if __name__ == "__main__":
    main()
