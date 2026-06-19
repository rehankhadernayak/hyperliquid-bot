"""Analysis of feature predictive value."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from feature_research.feature_config import FeatureResearchConfig
from feature_research.feature_models import (
    FEATURE_COLUMN_NAMES,
    FeatureLabel,
    FeatureRow,
    SignalCandidate,
)


@dataclass
class FeatureAnalysisResult:
    feature_name: str
    horizon_seconds: int
    row_count: int
    missing_count: int
    correlation: float | None
    information_coefficient: float | None
    monotonicity_score: float | None
    bucket_analysis: list[dict[str, Any]] = field(default_factory=list)
    stability_by_coin: dict[str, float | None] = field(default_factory=dict)
    stability_by_time: dict[str, float | None] = field(default_factory=dict)
    candidate_signal: str = SignalCandidate.NO_SIGNAL.value
    candidate_reason: str = ""


def _safe_correlation(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    for rank, (idx, _) in enumerate(indexed, start=1):
        ranks[idx] = float(rank)
    return ranks


def _information_coefficient(feature_vals: list[float], label_vals: list[float]) -> float | None:
    if len(feature_vals) < 5:
        return None
    return _safe_correlation(_rank_values(feature_vals), _rank_values(label_vals))


def _quantile_buckets(
    values: list[tuple[float, float, bool | None, float | None, float | None]],
    num_buckets: int,
) -> tuple[list[dict[str, Any]], int]:
    """Bucket (feature, future_return, up_after_costs, max_fav, max_adv) tuples."""
    if not values:
        return [], num_buckets

    actual_buckets = num_buckets
    if len(values) < num_buckets * 5:
        actual_buckets = max(2, len(values) // 5)
        if actual_buckets < 2:
            return [], num_buckets

    sorted_vals = sorted(values, key=lambda x: x[0])
    bucket_size = len(sorted_vals) / actual_buckets
    buckets: list[dict[str, Any]] = []

    labels_pct = ["bottom 20%", "20-40%", "40-60%", "60-80%", "top 20%"]
    for i in range(actual_buckets):
        start = int(i * bucket_size)
        end = int((i + 1) * bucket_size) if i < actual_buckets - 1 else len(sorted_vals)
        chunk = sorted_vals[start:end]
        if not chunk:
            continue

        returns = [c[1] for c in chunk]
        wins = [c[2] for c in chunk if c[2] is not None]
        favs = [c[3] for c in chunk if c[3] is not None]
        advs = [c[4] for c in chunk if c[4] is not None]

        label = labels_pct[i] if i < len(labels_pct) else f"bucket {i + 1}"
        buckets.append(
            {
                "bucket": label,
                "row_count": len(chunk),
                "feature_min": chunk[0][0],
                "feature_max": chunk[-1][0],
                "avg_future_return": sum(returns) / len(returns) if returns else None,
                "win_rate_after_costs": sum(1 for w in wins if w) / len(wins) if wins else None,
                "avg_favorable_move": sum(favs) / len(favs) if favs else None,
                "avg_adverse_move": sum(advs) / len(advs) if advs else None,
            }
        )

    return buckets, actual_buckets


def _monotonicity_score(buckets: list[dict[str, Any]]) -> float | None:
    if len(buckets) < 2:
        return None
    returns = [b["avg_future_return"] for b in buckets if b.get("avg_future_return") is not None]
    if len(returns) < 2:
        return None
    increasing = sum(1 for i in range(1, len(returns)) if returns[i] >= returns[i - 1])
    decreasing = sum(1 for i in range(1, len(returns)) if returns[i] <= returns[i - 1])
    return max(increasing, decreasing) / (len(returns) - 1)


def _assign_candidate_signal(
    *,
    row_count: int,
    missing_count: int,
    correlation: float | None,
    ic: float | None,
    monotonicity: float | None,
    buckets: list[dict[str, Any]],
    stability_by_coin: dict[str, float | None],
    horizons_consistent: int,
    config: FeatureResearchConfig,
) -> tuple[str, str]:
    missing_rate = missing_count / row_count if row_count else 1.0

    if row_count < config.min_rows_for_signal:
        return SignalCandidate.NO_SIGNAL.value, "insufficient rows"

    if missing_rate > config.max_missing_rate:
        return SignalCandidate.REJECT.value, f"missing rate {missing_rate:.1%} exceeds threshold"

    if correlation is None and ic is None:
        return SignalCandidate.NO_SIGNAL.value, "no measurable correlation"

    abs_corr = abs(correlation) if correlation is not None else 0.0
    abs_ic = abs(ic) if ic is not None else 0.0

    if abs_corr < 0.02 and abs_ic < 0.02:
        return SignalCandidate.NO_SIGNAL.value, "correlation too weak"

    coin_corrs = [abs(v) for v in stability_by_coin.values() if v is not None]
    if coin_corrs and max(coin_corrs) > 3 * (sum(coin_corrs) / len(coin_corrs)):
        return SignalCandidate.UNSTABLE_SIGNAL.value, "dominated by one coin"

    if monotonicity is not None and monotonicity < 0.4:
        return SignalCandidate.UNSTABLE_SIGNAL.value, "non-monotonic bucket performance"

    if abs_corr >= 0.05 and abs_ic >= 0.03 and horizons_consistent >= 2:
        if row_count >= config.min_rows_for_signal * 2:
            return (
                SignalCandidate.POSSIBLE_SIGNAL_NEEDS_MORE_DATA.value,
                "consistent direction across horizons, above noise threshold",
            )
        return SignalCandidate.WEAK_SIGNAL.value, "promising but needs more data"

    if abs_corr >= 0.03 or abs_ic >= 0.02:
        return SignalCandidate.WEAK_SIGNAL.value, "weak predictive hint"

    return SignalCandidate.NO_SIGNAL.value, "below signal threshold"


class FeatureAnalyzer:
    """Simple explainable analysis of feature predictive value."""

    def __init__(self, config: FeatureResearchConfig) -> None:
        self._config = config

    def analyze(
        self,
        rows: tuple[FeatureRow, ...],
        labels: list[FeatureLabel],
    ) -> list[FeatureAnalysisResult]:
        label_map = {label.feature_row_id: label.labels for label in labels}
        results: list[FeatureAnalysisResult] = []

        horizon_consistency: dict[str, dict[int, float | None]] = {}

        for feature_name in FEATURE_COLUMN_NAMES:
            for horizon in self._config.horizons_seconds:
                result = self._analyze_feature_horizon(
                    feature_name, horizon, rows, label_map
                )
                results.append(result)
                if feature_name not in horizon_consistency:
                    horizon_consistency[feature_name] = {}
                horizon_consistency[feature_name][horizon] = result.correlation

        for result in results:
            consistent = sum(
                1
                for h, corr in horizon_consistency.get(result.feature_name, {}).items()
                if corr is not None
                and result.correlation is not None
                and (corr > 0) == (result.correlation > 0)
            )
            candidate, reason = _assign_candidate_signal(
                row_count=result.row_count,
                missing_count=result.missing_count,
                correlation=result.correlation,
                ic=result.information_coefficient,
                monotonicity=result.monotonicity_score,
                buckets=result.bucket_analysis,
                stability_by_coin=result.stability_by_coin,
                horizons_consistent=consistent,
                config=self._config,
            )
            result.candidate_signal = candidate
            result.candidate_reason = reason

        return results

    def _analyze_feature_horizon(
        self,
        feature_name: str,
        horizon: int,
        rows: tuple[FeatureRow, ...],
        label_map: dict[str, dict[str, Any]],
    ) -> FeatureAnalysisResult:
        ret_key = f"future_return_{horizon}s_pct"
        up_key = f"future_up_after_costs_{horizon}s"

        pairs: list[tuple[float, float, bool | None, float | None, float | None]] = []
        missing_count = 0

        for row in rows:
            feat_val = row.features.get(feature_name)
            labels = label_map.get(row.feature_row_id, {})
            future_ret = labels.get(ret_key)
            up_after = labels.get(up_key)
            max_fav = labels.get("max_favorable_move_pct")
            max_adv = labels.get("max_adverse_move_pct")

            if feat_val is None or future_ret is None:
                missing_count += 1
                continue
            if not isinstance(feat_val, (int, float)):
                missing_count += 1
                continue
            pairs.append((float(feat_val), float(future_ret), up_after, max_fav, max_adv))

        row_count = len(rows)
        feature_vals = [p[0] for p in pairs]
        label_vals = [p[1] for p in pairs]

        correlation = _safe_correlation(feature_vals, label_vals)
        ic = _information_coefficient(feature_vals, label_vals)
        buckets, _ = _quantile_buckets(pairs, self._config.quantile_buckets)
        monotonicity = _monotonicity_score(buckets)

        stability_by_coin: dict[str, float | None] = {}
        for coin in self._config.coins:
            coin_pairs = [
                (row.features.get(feature_name), label_map.get(row.feature_row_id, {}).get(ret_key))
                for row in rows
                if row.coin == coin
            ]
            valid = [
                (float(f), float(l))
                for f, l in coin_pairs
                if f is not None and l is not None and isinstance(f, (int, float))
            ]
            if len(valid) >= 5:
                stability_by_coin[coin] = _safe_correlation(
                    [v[0] for v in valid], [v[1] for v in valid]
                )
            else:
                stability_by_coin[coin] = None

        stability_by_time = self._time_window_stability(
            rows, label_map, feature_name, ret_key
        )

        return FeatureAnalysisResult(
            feature_name=feature_name,
            horizon_seconds=horizon,
            row_count=row_count,
            missing_count=missing_count,
            correlation=correlation,
            information_coefficient=ic,
            monotonicity_score=monotonicity,
            bucket_analysis=buckets,
            stability_by_coin=stability_by_coin,
            stability_by_time=stability_by_time,
        )

    @staticmethod
    def _time_window_stability(
        rows: tuple[FeatureRow, ...],
        label_map: dict[str, dict[str, Any]],
        feature_name: str,
        ret_key: str,
    ) -> dict[str, float | None]:
        if not rows:
            return {}
        times = [row.event_timestamp_ms for row in rows]
        min_t, max_t = min(times), max(times)
        mid = (min_t + max_t) / 2

        windows: dict[str, list[tuple[float, float]]] = {"first_half": [], "second_half": []}
        for row in rows:
            feat = row.features.get(feature_name)
            label = label_map.get(row.feature_row_id, {}).get(ret_key)
            if feat is None or label is None or not isinstance(feat, (int, float)):
                continue
            key = "first_half" if row.event_timestamp_ms <= mid else "second_half"
            windows[key].append((float(feat), float(label)))

        return {
            w: _safe_correlation([p[0] for p in pairs], [p[1] for p in pairs])
            if len(pairs) >= 5
            else None
            for w, pairs in windows.items()
        }


def summarize_missing_data(
    rows: tuple[FeatureRow, ...],
) -> dict[str, Any]:
    if not rows:
        return {"total_rows": 0, "features": {}}

    feature_missing: dict[str, int] = {}
    for row in rows:
        for name, is_missing in row.missing_flags.items():
            if is_missing:
                feature_missing[name] = feature_missing.get(name, 0) + 1

    total = len(rows)
    return {
        "total_rows": total,
        "features": {
            name: {"missing_count": count, "missing_rate": count / total}
            for name, count in sorted(feature_missing.items())
        },
    }


def top_signals(results: list[FeatureAnalysisResult]) -> dict[str, list[dict[str, Any]]]:
    possible = []
    rejected = []
    for r in results:
        entry = {
            "feature": r.feature_name,
            "horizon_seconds": r.horizon_seconds,
            "correlation": r.correlation,
            "information_coefficient": r.information_coefficient,
            "candidate_signal": r.candidate_signal,
            "reason": r.candidate_reason,
        }
        if r.candidate_signal == SignalCandidate.POSSIBLE_SIGNAL_NEEDS_MORE_DATA.value:
            possible.append(entry)
        elif r.candidate_signal in (
            SignalCandidate.REJECT.value,
            SignalCandidate.UNSTABLE_SIGNAL.value,
        ):
            rejected.append(entry)

    possible.sort(key=lambda x: abs(x.get("correlation") or 0), reverse=True)
    rejected.sort(key=lambda x: x.get("feature", ""))
    return {"possible": possible[:20], "rejected": rejected[:20]}
