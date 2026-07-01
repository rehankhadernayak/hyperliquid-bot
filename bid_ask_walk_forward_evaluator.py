"""Walk-forward paper evaluation of bid/ask models with realistic cost simulation.

Research-only and paper-only. No orders, wallets, or network calls.
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bid_ask_model_io import build_artifact_payload, save_model_artifact
from bid_ask_resampler import normalize_source, resolve_bid_ask_data_dir, resampled_csv_path
from bid_ask_schema import EXECUTION_ENABLED, PAPER_ONLY, SAFETY_STATEMENT
from bid_ask_train import (
    FEATURE_COLUMNS,
    LABELS,
    MajorityClassBaseline,
    RuleBaseline,
    _accuracy,
    _feature_matrix,
    _predict_sklearn,
    _rows_ready_for_model,
    create_features,
    create_labels,
    load_training_data,
    train_baseline_model,
    train_optional_sklearn_model,
)

SAFETY_NOTE = (
    "Walk-forward evaluation is research-only and paper-only. "
    "Simulated returns do not prove profitability. No real orders are placed."
)

DEFAULT_DATA_ROOT = Path("data/bid_ask_csv")
DEFAULT_REPORTS_DIR = Path("reports/bid_ask_walk_forward")
DEFAULT_MODELS_DIR = Path("models/bid_ask_walk_forward")

SUPPORTED_INTERVALS = frozenset({"15s", "1m"})
SUPPORTED_SOURCES = frozenset({"local", "historical", "combined"})

DEFAULT_TRAIN_BARS = 500
DEFAULT_TEST_BARS = 100
DEFAULT_STEP_BARS = 100
DEFAULT_FEE_BPS = 2.0
DEFAULT_SLIPPAGE_BPS = 1.0
DEFAULT_SPREAD_BPS = 2.0
DEFAULT_MIN_CONFIDENCE = 0.4
DEFAULT_THRESHOLD_BPS = 1.0
DEFAULT_HORIZON = 5

MIN_ROWS_WARNING = 50
MIN_WINDOWS_READY = 2
MIN_TRADES_READY = 10
MIN_CLASS_FRACTION = 0.10
MAX_CLASS_FRACTION = 0.70
MISSING_DATA_FRACTION_WARNING = 0.05


def resolve_coins(*, coin: str | None, coins: list[str] | None) -> list[str]:
    """Merge --coin and --coins into a deduplicated uppercase list."""
    resolved: list[str] = []
    if coins:
        resolved.extend(str(value).upper() for value in coins)
    if coin:
        resolved.append(str(coin).upper())
    if not resolved:
        resolved.append("BTC")
    seen: set[str] = set()
    ordered: list[str] = []
    for value in resolved:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def generate_walk_forward_windows(
    n_rows: int,
    *,
    train_bars: int,
    test_bars: int,
    step_bars: int,
) -> list[tuple[int, int, int, int]]:
    """Return chronological (train_start, train_end, test_start, test_end) index ranges."""
    if train_bars < 1 or test_bars < 1 or step_bars < 1:
        raise ValueError("train_bars, test_bars, and step_bars must be >= 1")

    windows: list[tuple[int, int, int, int]] = []
    start = 0
    while True:
        train_start = start
        train_end = start + train_bars
        test_start = train_end
        test_end = test_start + test_bars
        if test_end > n_rows:
            break
        windows.append((train_start, train_end, test_start, test_end))
        start += step_bars
    return windows


def _spread_cost_bps(row: pd.Series, default_spread_bps: float) -> float:
    if "avg_spread_bps" in row.index and pd.notna(row["avg_spread_bps"]):
        return float(row["avg_spread_bps"])
    if "spread_bps" in row.index and pd.notna(row["spread_bps"]):
        return float(row["spread_bps"])
    return float(default_spread_bps)


def _estimate_cost_bps(
    row: pd.Series,
    *,
    fee_bps: float,
    slippage_bps: float,
    default_spread_bps: float,
) -> float:
    spread_cost = _spread_cost_bps(row, default_spread_bps)
    return fee_bps + slippage_bps + spread_cost


def _prediction_confidence(
    *,
    sklearn_result: dict[str, Any],
    model_predictions: list[str],
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    model_used: str,
) -> list[float]:
    model = sklearn_result.get("model")
    scaler = sklearn_result.get("scaler")
    if model is not None and scaler is not None and hasattr(model, "predict_proba"):
        X_test = _feature_matrix(test_df)
        if len(X_test) == 0:
            return []
        confidences: list[float] = []
        for proba in model.predict_proba(scaler.transform(X_test)):
            confidences.append(float(max(proba)))
        return confidences

    if model_used == MajorityClassBaseline.model_type:
        labels = train_df["label"].tolist()
        if not labels:
            return [0.0] * len(model_predictions)
        majority_fraction = labels.count(
            str(pd.Series(labels).value_counts().idxmax())
        ) / len(labels)
        return [majority_fraction] * len(model_predictions)

    if model_used == RuleBaseline.model_type:
        returns = test_df["mid_return_1"].astype(float).abs()
        max_return = float(returns.max()) if len(returns) else 1.0
        if max_return <= 0:
            return [0.5] * len(model_predictions)
        return [min(1.0, float(value) / max_return) for value in returns]

    return [1.0] * len(model_predictions)


def simulate_paper_signal_returns(
    test_df: pd.DataFrame,
    predictions: list[str],
    confidences: list[float],
    *,
    fee_bps: float,
    slippage_bps: float,
    default_spread_bps: float,
    min_confidence: float,
) -> dict[str, Any]:
    """Simulate paper-only cost-adjusted returns for model signals."""
    gross_returns: list[float] = []
    net_returns: list[float] = []
    trade_returns: list[float] = []
    signal_count = 0
    trade_count = 0

    for index, (_, row) in enumerate(test_df.iterrows()):
        prediction = predictions[index] if index < len(predictions) else "FLAT"
        confidence = float(confidences[index]) if index < len(confidences) else 0.0
        future_return = float(row["future_return_bps"])
        cost_bps = _estimate_cost_bps(
            row,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            default_spread_bps=default_spread_bps,
        )

        if prediction not in ("UP", "DOWN") or confidence < min_confidence:
            gross_returns.append(0.0)
            net_returns.append(0.0)
            continue

        signal_count += 1
        if prediction == "UP":
            gross = future_return
        else:
            gross = -future_return

        net = gross - cost_bps
        gross_returns.append(gross)
        net_returns.append(net)
        trade_count += 1
        trade_returns.append(net)

    return {
        "gross_returns_bps": gross_returns,
        "net_returns_bps": net_returns,
        "trade_returns_bps": trade_returns,
        "total_paper_signals": signal_count,
        "total_paper_trades": trade_count,
    }


def max_drawdown_bps(net_returns: list[float]) -> float:
    """Peak-to-trough drawdown on cumulative net returns."""
    if not net_returns:
        return 0.0
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in net_returns:
        cumulative += value
        if cumulative > peak:
            peak = cumulative
        drawdown = peak - cumulative
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def longest_losing_streak(net_returns: list[float]) -> int:
    """Longest run of consecutive negative net returns."""
    max_streak = 0
    current = 0
    for value in net_returns:
        if value < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _no_trade_predictions(count: int) -> list[str]:
    return ["FLAT"] * count


def _label_imbalance_warnings(labels: list[str], split_name: str) -> list[str]:
    warnings: list[str] = []
    total = len(labels)
    if total == 0:
        return warnings
    counts = pd.Series(labels).value_counts()
    for label in LABELS:
        fraction = counts.get(label, 0) / total
        if fraction < MIN_CLASS_FRACTION:
            warnings.append(
                f"class_imbalance: {split_name} {label} is {fraction:.1%} "
                f"(under {MIN_CLASS_FRACTION:.0%})"
            )
        if fraction > MAX_CLASS_FRACTION:
            warnings.append(
                f"class_imbalance: {split_name} {label} is {fraction:.1%} "
                f"(over {MAX_CLASS_FRACTION:.0%})"
            )
    return warnings


def _missing_data_warning(df: pd.DataFrame) -> str | None:
    check_columns = [
        column
        for column in ("avg_spread_bps", "spread_bps", "close_mid", "avg_bbo_imbalance")
        if column in df.columns
    ]
    if not check_columns or df.empty:
        return None
    missing_fraction = float(df[check_columns].isna().any(axis=1).mean())
    if missing_fraction >= MISSING_DATA_FRACTION_WARNING:
        return (
            f"high_missing_data: {missing_fraction:.1%} of rows have missing "
            f"values in {check_columns}"
        )
    return None


def evaluate_walk_forward_window(
    model_df: pd.DataFrame,
    window: tuple[int, int, int, int],
    *,
    window_index: int,
    fee_bps: float,
    slippage_bps: float,
    default_spread_bps: float,
    min_confidence: float,
    save_window_models: bool,
    models_dir: Path,
    coin: str,
    interval: str,
    horizon: int,
    threshold_bps: float,
    source: str,
) -> dict[str, Any]:
    """Train on one chronological window and evaluate on the following test slice."""
    train_start, train_end, test_start, test_end = window
    train_df = model_df.iloc[train_start:train_end].copy().reset_index(drop=True)
    test_df = model_df.iloc[test_start:test_end].copy().reset_index(drop=True)

    if len(train_df) < 5 or len(test_df) < 1:
        return {
            "window_index": window_index,
            "status": "skipped",
            "reason": "too_few_rows",
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        }

    baseline = train_baseline_model(train_df)
    sklearn_result = train_optional_sklearn_model(train_df)

    majority_model: MajorityClassBaseline = baseline["majority_model"]
    rule_model: RuleBaseline = baseline["rule_model"]
    baseline_predictions = majority_model.predict(test_df)
    no_trade_predictions = _no_trade_predictions(len(test_df))
    momentum_predictions = rule_model.predict(_feature_matrix(test_df))
    sklearn_predictions = _predict_sklearn(sklearn_result, test_df)

    if sklearn_predictions is not None:
        model_predictions = sklearn_predictions
        model_used = sklearn_result.get("model_type") or "sklearn"
    else:
        model_predictions = baseline_predictions
        model_used = majority_model.model_type

    y_true = test_df["label"].tolist()
    confidences = _prediction_confidence(
        sklearn_result=sklearn_result,
        model_predictions=model_predictions,
        test_df=test_df,
        train_df=train_df,
        model_used=model_used,
    )
    paper_simulation = simulate_paper_signal_returns(
        test_df,
        model_predictions,
        confidences,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        default_spread_bps=default_spread_bps,
        min_confidence=min_confidence,
    )
    trade_returns = paper_simulation["trade_returns_bps"]
    wins = sum(1 for value in trade_returns if value > 0)
    win_rate = wins / len(trade_returns) if trade_returns else 0.0

    model_artifact_path: str | None = None
    if save_window_models and sklearn_result.get("model") is not None:
        artifact_id = f"wf_{window_index}_{uuid.uuid4().hex[:8]}"
        payload = build_artifact_payload(
            model=sklearn_result["model"],
            scaler=sklearn_result.get("scaler"),
            feature_names=FEATURE_COLUMNS,
            coin=coin,
            interval=interval,
            horizon=horizon,
            threshold_bps=threshold_bps,
            train_rows=len(train_df),
            test_rows=len(test_df),
            baseline_accuracy=_accuracy(y_true, baseline_predictions),
            model_accuracy=_accuracy(y_true, model_predictions),
            model_type=str(sklearn_result.get("model_type") or model_used),
        )
        payload["source"] = normalize_source(source)
        payload["walk_forward_window_index"] = window_index
        saved_path = save_model_artifact(payload, models_dir=models_dir, artifact_id=artifact_id)
        model_artifact_path = str(saved_path)

    return {
        "window_index": window_index,
        "status": "completed",
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "model_used": model_used,
        "model_accuracy": _accuracy(y_true, model_predictions),
        "baseline_accuracy": _accuracy(y_true, baseline_predictions),
        "no_trade_baseline_accuracy": _accuracy(y_true, no_trade_predictions),
        "momentum_baseline_accuracy": _accuracy(y_true, momentum_predictions),
        "prediction_counts": {
            label: model_predictions.count(label) for label in LABELS
        },
        "baseline_prediction_counts": {
            label: baseline_predictions.count(label) for label in LABELS
        },
        "total_paper_signals": paper_simulation["total_paper_signals"],
        "total_paper_trades": paper_simulation["total_paper_trades"],
        "average_gross_return_bps": (
            float(np.mean(paper_simulation["gross_returns_bps"]))
            if paper_simulation["gross_returns_bps"]
            else 0.0
        ),
        "average_net_return_bps": (
            float(np.mean(trade_returns)) if trade_returns else 0.0
        ),
        "cumulative_net_return_bps": float(sum(trade_returns)),
        "win_rate_after_costs": win_rate,
        "longest_losing_streak": longest_losing_streak(trade_returns),
        "max_drawdown_bps": max_drawdown_bps(trade_returns),
        "trade_returns_bps": trade_returns,
        "model_artifact_path": model_artifact_path,
        "warnings": _label_imbalance_warnings(y_true, "test"),
    }


def evaluate_coin_walk_forward(
    *,
    coin: str,
    interval: str,
    source: str,
    horizon: int,
    threshold_bps: float,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    fee_bps: float,
    slippage_bps: float,
    default_spread_bps: float,
    min_confidence: float,
    data_root: Path,
    save_window_models: bool,
    models_dir: Path,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Run walk-forward evaluation for one coin."""
    normalized_source = normalize_source(source)
    warnings: list[str] = [SAFETY_NOTE]

    try:
        raw_df = load_training_data(coin, interval, data_root, source=normalized_source)
    except FileNotFoundError:
        return {
            "coin": coin,
            "status": "skipped",
            "reason": "csv_not_found",
            "warnings": warnings + [f"too_few_rows: CSV not found for {coin}"],
            "windows": [],
        }

    if max_rows is not None and max_rows > 0:
        raw_df = raw_df.iloc[:max_rows].copy()

    missing_warning = _missing_data_warning(raw_df)
    if missing_warning:
        warnings.append(missing_warning)

    featured_df = create_features(raw_df)
    labeled_df = create_labels(featured_df, horizon=horizon, threshold_bps=threshold_bps)
    model_df = _rows_ready_for_model(labeled_df)

    if len(model_df) < MIN_ROWS_WARNING:
        warnings.append(f"too_few_rows: only {len(model_df)} model-ready rows for {coin}")

    windows = generate_walk_forward_windows(
        len(model_df),
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=step_bars,
    )

    window_results: list[dict[str, Any]] = []
    for window_index, window in enumerate(windows):
        result = evaluate_walk_forward_window(
            model_df,
            window,
            window_index=window_index,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            default_spread_bps=default_spread_bps,
            min_confidence=min_confidence,
            save_window_models=save_window_models,
            models_dir=models_dir,
            coin=coin,
            interval=interval,
            horizon=horizon,
            threshold_bps=threshold_bps,
            source=normalized_source,
        )
        window_results.append(result)

    completed = [item for item in window_results if item.get("status") == "completed"]
    skipped = [item for item in window_results if item.get("status") != "completed"]

    model_accuracies = [float(item["model_accuracy"]) for item in completed]
    baseline_accuracies = [float(item["baseline_accuracy"]) for item in completed]
    model_accuracy_mean = float(np.mean(model_accuracies)) if model_accuracies else 0.0
    baseline_accuracy_mean = float(np.mean(baseline_accuracies)) if baseline_accuracies else 0.0

    total_paper_signals = sum(int(item.get("total_paper_signals", 0)) for item in completed)
    total_paper_trades = sum(int(item.get("total_paper_trades", 0)) for item in completed)

    per_trade_returns: list[float] = []
    for window in completed:
        per_trade_returns.extend(float(value) for value in window.get("trade_returns_bps", []))

    cumulative_net_return_bps = float(sum(item.get("cumulative_net_return_bps", 0.0) for item in completed))
    average_gross_return_bps = (
        float(np.mean([item.get("average_gross_return_bps", 0.0) for item in completed]))
        if completed
        else 0.0
    )
    average_net_return_bps = (
        float(np.mean(per_trade_returns)) if per_trade_returns else 0.0
    )
    win_rate_after_costs = (
        sum(1 for value in per_trade_returns if value > 0) / len(per_trade_returns)
        if per_trade_returns
        else 0.0
    )

    if len(windows) < MIN_WINDOWS_READY:
        warnings.append(
            f"too_few_windows: only {len(windows)} possible windows "
            f"(need {MIN_WINDOWS_READY})"
        )
    if model_accuracy_mean <= baseline_accuracy_mean and completed:
        warnings.append("model_not_better_than_baseline")
    if cumulative_net_return_bps <= 0 and total_paper_trades > 0:
        warnings.append("negative_after_costs")

    train_labels = model_df["label"].tolist() if not model_df.empty else []
    warnings.extend(_label_imbalance_warnings(train_labels, "all"))

    beats_baseline_accuracy = model_accuracy_mean > baseline_accuracy_mean
    positive_after_costs = cumulative_net_return_bps > 0
    enough_windows = len(completed) >= MIN_WINDOWS_READY
    enough_trades = total_paper_trades >= MIN_TRADES_READY
    ready_for_more_research = (
        beats_baseline_accuracy
        and positive_after_costs
        and enough_windows
        and enough_trades
    )

    source_path = str(resampled_csv_path(coin, interval, data_root, source=normalized_source))

    return {
        "coin": coin,
        "status": "completed" if completed else "skipped",
        "source_path": source_path,
        "model_ready_rows": len(model_df),
        "total_windows": len(windows),
        "completed_windows": len(completed),
        "skipped_windows": len(skipped),
        "model_accuracy_mean": model_accuracy_mean,
        "baseline_accuracy_mean": baseline_accuracy_mean,
        "accuracy_edge_vs_baseline": model_accuracy_mean - baseline_accuracy_mean,
        "total_paper_signals": total_paper_signals,
        "total_paper_trades": total_paper_trades,
        "average_gross_return_bps": average_gross_return_bps,
        "average_net_return_bps": average_net_return_bps,
        "cumulative_net_return_bps": cumulative_net_return_bps,
        "max_drawdown_bps": max_drawdown_bps(per_trade_returns),
        "win_rate_after_costs": win_rate_after_costs,
        "longest_losing_streak": longest_losing_streak(per_trade_returns),
        "readiness": {
            "beats_baseline_accuracy": beats_baseline_accuracy,
            "positive_after_costs": positive_after_costs,
            "enough_windows": enough_windows,
            "enough_trades": enough_trades,
            "ready_for_more_research": ready_for_more_research,
            "real_execution_ready": False,
        },
        "windows": window_results,
        "warnings": warnings,
    }


def build_walk_forward_report(
    *,
    coins: list[str],
    interval: str,
    source: str,
    horizon: int,
    threshold_bps: float,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    fee_bps: float,
    slippage_bps: float,
    default_spread_bps: float,
    min_confidence: float,
    per_coin_results: list[dict[str, Any]],
    data_root: Path,
) -> dict[str, Any]:
    """Aggregate per-coin walk-forward results into one combined report."""
    normalized_source = normalize_source(source)
    data_dir = resolve_bid_ask_data_dir(interval, normalized_source, data_root)

    completed_coins = [item for item in per_coin_results if item.get("completed_windows", 0) > 0]
    warnings: list[str] = [SAFETY_NOTE]

    total_windows = sum(int(item.get("total_windows", 0)) for item in per_coin_results)
    completed_windows = sum(int(item.get("completed_windows", 0)) for item in per_coin_results)
    skipped_windows = sum(int(item.get("skipped_windows", 0)) for item in per_coin_results)
    total_paper_signals = sum(int(item.get("total_paper_signals", 0)) for item in per_coin_results)
    total_paper_trades = sum(int(item.get("total_paper_trades", 0)) for item in per_coin_results)

    model_accuracy_values = [
        float(item["model_accuracy_mean"])
        for item in completed_coins
        if item.get("model_accuracy_mean") is not None
    ]
    baseline_accuracy_values = [
        float(item["baseline_accuracy_mean"])
        for item in completed_coins
        if item.get("baseline_accuracy_mean") is not None
    ]
    model_accuracy_mean = float(np.mean(model_accuracy_values)) if model_accuracy_values else 0.0
    baseline_accuracy_mean = (
        float(np.mean(baseline_accuracy_values)) if baseline_accuracy_values else 0.0
    )

    cumulative_net_return_bps = float(
        sum(float(item.get("cumulative_net_return_bps", 0.0)) for item in completed_coins)
    )
    gross_values = [
        float(item.get("average_gross_return_bps", 0.0))
        for item in completed_coins
        if item.get("total_paper_trades", 0) > 0
    ]
    net_values = [
        float(item.get("average_net_return_bps", 0.0))
        for item in completed_coins
        if item.get("total_paper_trades", 0) > 0
    ]
    drawdown_values = [
        float(item.get("max_drawdown_bps", 0.0)) for item in completed_coins
    ]
    win_rates = [
        float(item.get("win_rate_after_costs", 0.0))
        for item in completed_coins
        if item.get("total_paper_trades", 0) > 0
    ]
    losing_streaks = [
        int(item.get("longest_losing_streak", 0)) for item in completed_coins
    ]

    for item in per_coin_results:
        warnings.extend(item.get("warnings", []))

    if total_windows < MIN_WINDOWS_READY:
        warnings.append(
            f"too_few_windows: aggregate total_windows={total_windows} "
            f"(need {MIN_WINDOWS_READY})"
        )
    if model_accuracy_mean <= baseline_accuracy_mean and completed_coins:
        warnings.append("model_not_better_than_baseline")
    if cumulative_net_return_bps <= 0 and total_paper_trades > 0:
        warnings.append("negative_after_costs")

    beats_baseline_accuracy = model_accuracy_mean > baseline_accuracy_mean
    positive_after_costs = cumulative_net_return_bps > 0
    enough_windows = completed_windows >= MIN_WINDOWS_READY
    enough_trades = total_paper_trades >= MIN_TRADES_READY
    ready_for_more_research = (
        beats_baseline_accuracy
        and positive_after_costs
        and enough_windows
        and enough_trades
    )

    per_coin_summary = {
        item["coin"]: {
            "status": item.get("status"),
            "completed_windows": item.get("completed_windows", 0),
            "model_accuracy_mean": item.get("model_accuracy_mean", 0.0),
            "baseline_accuracy_mean": item.get("baseline_accuracy_mean", 0.0),
            "accuracy_edge_vs_baseline": item.get("accuracy_edge_vs_baseline", 0.0),
            "total_paper_trades": item.get("total_paper_trades", 0),
            "cumulative_net_return_bps": item.get("cumulative_net_return_bps", 0.0),
            "readiness": item.get("readiness", {}),
        }
        for item in per_coin_results
    }

    return {
        "report_id": uuid.uuid4().hex,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coins": coins,
        "interval": interval,
        "source": normalized_source,
        "data_dir": str(data_dir),
        "data_root": str(data_root),
        "horizon": horizon,
        "threshold_bps": threshold_bps,
        "train_bars": train_bars,
        "test_bars": test_bars,
        "step_bars": step_bars,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "default_spread_bps": default_spread_bps,
        "min_confidence": min_confidence,
        "total_windows": total_windows,
        "completed_windows": completed_windows,
        "skipped_windows": skipped_windows,
        "model_accuracy_mean": model_accuracy_mean,
        "baseline_accuracy_mean": baseline_accuracy_mean,
        "accuracy_edge_vs_baseline": model_accuracy_mean - baseline_accuracy_mean,
        "total_paper_signals": total_paper_signals,
        "total_paper_trades": total_paper_trades,
        "average_gross_return_bps": float(np.mean(gross_values)) if gross_values else 0.0,
        "average_net_return_bps": float(np.mean(net_values)) if net_values else 0.0,
        "cumulative_net_return_bps": cumulative_net_return_bps,
        "max_drawdown_bps": float(max(drawdown_values)) if drawdown_values else 0.0,
        "win_rate_after_costs": float(np.mean(win_rates)) if win_rates else 0.0,
        "longest_losing_streak": max(losing_streaks) if losing_streaks else 0,
        "per_coin": per_coin_summary,
        "per_coin_details": per_coin_results,
        "readiness": {
            "beats_baseline_accuracy": beats_baseline_accuracy,
            "positive_after_costs": positive_after_costs,
            "enough_windows": enough_windows,
            "enough_trades": enough_trades,
            "ready_for_more_research": ready_for_more_research,
            "real_execution_ready": False,
        },
        "warnings": list(dict.fromkeys(warnings)),
        "paper_only": PAPER_ONLY,
        "execution_enabled": EXECUTION_ENABLED,
        "real_execution_ready": False,
        "safety_statement": SAFETY_STATEMENT,
        "safety_note": SAFETY_NOTE,
    }


def _report_stamp(report: dict[str, Any]) -> str:
    generated_at = str(report.get("generated_at", ""))
    if generated_at:
        return generated_at.replace(":", "").replace("+", "").replace("-", "")[:15]
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_walk_forward_json(
    report: dict[str, Any],
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> Path:
    """Write timestamped JSON walk-forward report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = _report_stamp(report)
    coins_label = "_".join(report.get("coins", ["UNKNOWN"]))
    output_path = reports_dir / f"walk_forward_{coins_label}_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    report["json_report_path"] = str(output_path)
    return output_path


def write_walk_forward_markdown(
    report: dict[str, Any],
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> Path:
    """Write timestamped Markdown walk-forward report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = _report_stamp(report)
    coins_label = "_".join(report.get("coins", ["UNKNOWN"]))
    output_path = reports_dir / f"walk_forward_{coins_label}_{stamp}.md"

    readiness = report.get("readiness", {})
    lines = [
        "# Walk-Forward Paper Evaluation",
        "",
        f"Generated at: {report.get('generated_at', 'n/a')}",
        f"Coins: {', '.join(report.get('coins', []))}",
        f"Interval: {report.get('interval')} | Source: {report.get('source')}",
        f"Horizon: {report.get('horizon')} bars | Threshold: {report.get('threshold_bps')} bps",
        "",
        "## Safety",
        "",
        f"- paper_only: {str(report.get('paper_only', True)).lower()}",
        f"- execution_enabled: {str(report.get('execution_enabled', False)).lower()}",
        f"- real_execution_ready: {str(report.get('real_execution_ready', False)).lower()}",
        "",
        "## Aggregate metrics",
        "",
        f"- total_windows: {report.get('total_windows', 0)}",
        f"- completed_windows: {report.get('completed_windows', 0)}",
        f"- skipped_windows: {report.get('skipped_windows', 0)}",
        f"- model_accuracy_mean: {report.get('model_accuracy_mean', 0.0):.4f}",
        f"- baseline_accuracy_mean: {report.get('baseline_accuracy_mean', 0.0):.4f}",
        f"- accuracy_edge_vs_baseline: {report.get('accuracy_edge_vs_baseline', 0.0):.4f}",
        f"- total_paper_signals: {report.get('total_paper_signals', 0)}",
        f"- total_paper_trades: {report.get('total_paper_trades', 0)}",
        f"- average_gross_return_bps: {report.get('average_gross_return_bps', 0.0):.4f}",
        f"- average_net_return_bps: {report.get('average_net_return_bps', 0.0):.4f}",
        f"- cumulative_net_return_bps: {report.get('cumulative_net_return_bps', 0.0):.4f}",
        f"- max_drawdown_bps: {report.get('max_drawdown_bps', 0.0):.4f}",
        f"- win_rate_after_costs: {report.get('win_rate_after_costs', 0.0):.4f}",
        f"- longest_losing_streak: {report.get('longest_losing_streak', 0)}",
        "",
        "## Readiness",
        "",
        f"- beats_baseline_accuracy: {readiness.get('beats_baseline_accuracy', False)}",
        f"- positive_after_costs: {readiness.get('positive_after_costs', False)}",
        f"- enough_windows: {readiness.get('enough_windows', False)}",
        f"- enough_trades: {readiness.get('enough_trades', False)}",
        f"- ready_for_more_research: {readiness.get('ready_for_more_research', False)}",
        f"- real_execution_ready: {readiness.get('real_execution_ready', False)}",
        "",
        "## Model vs baseline after costs",
        "",
    ]

    if readiness.get("beats_baseline_accuracy") and readiness.get("positive_after_costs"):
        lines.append(
            "Model accuracy beats baseline and cumulative net returns are positive "
            "after simulated costs (paper-only)."
        )
    elif readiness.get("beats_baseline_accuracy"):
        lines.append(
            "Model accuracy beats baseline, but cumulative net returns are not "
            "positive after simulated costs."
        )
    else:
        lines.append(
            "Model does not clearly beat baseline on accuracy and/or net returns "
            "after simulated costs."
        )

    lines.extend(
        [
            "",
            "## Per-coin summary",
            "",
            "| Coin | Windows | Model acc | Baseline acc | Edge | Trades | Net bps |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for coin, summary in sorted(report.get("per_coin", {}).items()):
        lines.append(
            f"| {coin} | {summary.get('completed_windows', 0)} | "
            f"{summary.get('model_accuracy_mean', 0.0):.4f} | "
            f"{summary.get('baseline_accuracy_mean', 0.0):.4f} | "
            f"{summary.get('accuracy_edge_vs_baseline', 0.0):.4f} | "
            f"{summary.get('total_paper_trades', 0)} | "
            f"{summary.get('cumulative_net_return_bps', 0.0):.4f} |"
        )

    lines.extend(["", "## Warnings", ""])
    warning_items = report.get("warnings", [])
    if not warning_items:
        lines.append("No warnings.")
    else:
        for warning in warning_items:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Research-only disclaimer",
            "",
            report.get("safety_note", SAFETY_NOTE),
            "",
            report.get("safety_statement", SAFETY_STATEMENT),
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["markdown_report_path"] = str(output_path)
    return output_path


def run_walk_forward_evaluation(
    *,
    coins: list[str],
    interval: str,
    source: str = "local",
    horizon: int = DEFAULT_HORIZON,
    threshold_bps: float = DEFAULT_THRESHOLD_BPS,
    train_bars: int = DEFAULT_TRAIN_BARS,
    test_bars: int = DEFAULT_TEST_BARS,
    step_bars: int = DEFAULT_STEP_BARS,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    default_spread_bps: float = DEFAULT_SPREAD_BPS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    data_root: Path = DEFAULT_DATA_ROOT,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    save_window_models: bool = False,
    max_rows: int | None = None,
    save_reports: bool = True,
) -> dict[str, Any]:
    """Run walk-forward evaluation for one or more coins and write reports."""
    per_coin_results = [
        evaluate_coin_walk_forward(
            coin=coin,
            interval=interval,
            source=source,
            horizon=horizon,
            threshold_bps=threshold_bps,
            train_bars=train_bars,
            test_bars=test_bars,
            step_bars=step_bars,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            default_spread_bps=default_spread_bps,
            min_confidence=min_confidence,
            data_root=data_root,
            save_window_models=save_window_models,
            models_dir=models_dir,
            max_rows=max_rows,
        )
        for coin in coins
    ]

    report = build_walk_forward_report(
        coins=coins,
        interval=interval,
        source=source,
        horizon=horizon,
        threshold_bps=threshold_bps,
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=step_bars,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        default_spread_bps=default_spread_bps,
        min_confidence=min_confidence,
        per_coin_results=per_coin_results,
        data_root=data_root,
    )

    if save_reports:
        write_walk_forward_json(report, reports_dir=reports_dir)
        write_walk_forward_markdown(report, reports_dir=reports_dir)

    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Walk-forward paper evaluation of bid/ask models with cost simulation "
            "(research-only, paper-only)."
        )
    )
    parser.add_argument("--coin", help="Single coin symbol, e.g. BTC")
    parser.add_argument(
        "--coins",
        nargs="+",
        help="Multiple coin symbols, e.g. BTC ETH SOL",
    )
    parser.add_argument(
        "--interval",
        default="15s",
        choices=sorted(SUPPORTED_INTERVALS),
        help="Resampled interval (default: 15s)",
    )
    parser.add_argument(
        "--source",
        default="local",
        choices=sorted(SUPPORTED_SOURCES),
        help="Data source: local, historical, or combined (default: local)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"Label horizon in bars (default: {DEFAULT_HORIZON})",
    )
    parser.add_argument(
        "--threshold-bps",
        type=float,
        default=DEFAULT_THRESHOLD_BPS,
        help=f"UP/DOWN label threshold in bps (default: {DEFAULT_THRESHOLD_BPS})",
    )
    parser.add_argument(
        "--train-bars",
        type=int,
        default=DEFAULT_TRAIN_BARS,
        help=f"Training window size in bars (default: {DEFAULT_TRAIN_BARS})",
    )
    parser.add_argument(
        "--test-bars",
        type=int,
        default=DEFAULT_TEST_BARS,
        help=f"Test window size in bars (default: {DEFAULT_TEST_BARS})",
    )
    parser.add_argument(
        "--step-bars",
        type=int,
        default=DEFAULT_STEP_BARS,
        help=f"Step size between windows in bars (default: {DEFAULT_STEP_BARS})",
    )
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=DEFAULT_FEE_BPS,
        help=f"Simulated fee in bps (default: {DEFAULT_FEE_BPS})",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=DEFAULT_SLIPPAGE_BPS,
        help=f"Simulated slippage in bps (default: {DEFAULT_SLIPPAGE_BPS})",
    )
    parser.add_argument(
        "--default-spread-bps",
        type=float,
        default=DEFAULT_SPREAD_BPS,
        help=f"Fallback spread cost in bps (default: {DEFAULT_SPREAD_BPS})",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=f"Minimum model confidence for paper signals (default: {DEFAULT_MIN_CONFIDENCE})",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing bid/ask CSV files",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory for walk-forward JSON and Markdown reports",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory for optional per-window model artifacts",
    )
    parser.add_argument(
        "--save-window-models",
        action="store_true",
        help="Save sklearn model artifact for each completed window",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows loaded per coin (research/debug)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    coins = resolve_coins(coin=args.coin, coins=args.coins)

    report = run_walk_forward_evaluation(
        coins=coins,
        interval=args.interval,
        source=args.source,
        horizon=args.horizon,
        threshold_bps=args.threshold_bps,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        default_spread_bps=args.default_spread_bps,
        min_confidence=args.min_confidence,
        data_root=args.data_root,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        save_window_models=args.save_window_models,
        max_rows=args.max_rows,
    )

    print(
        f"coins={','.join(coins)} interval={args.interval} source={report['source']} "
        f"horizon={args.horizon}"
    )
    print(
        f"windows={report['completed_windows']}/{report['total_windows']} "
        f"model_accuracy_mean={report['model_accuracy_mean']:.4f} "
        f"baseline_accuracy_mean={report['baseline_accuracy_mean']:.4f}"
    )
    print(
        f"cumulative_net_return_bps={report['cumulative_net_return_bps']:.4f} "
        f"beats_baseline={report['readiness']['beats_baseline_accuracy']} "
        f"positive_after_costs={report['readiness']['positive_after_costs']}"
    )
    if report.get("json_report_path"):
        print(f"json_report_path={report['json_report_path']}")
    if report.get("markdown_report_path"):
        print(f"markdown_report_path={report['markdown_report_path']}")
    for warning in report.get("warnings", []):
        if warning != SAFETY_NOTE:
            print(f"warning: {warning}")
    print(SAFETY_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
