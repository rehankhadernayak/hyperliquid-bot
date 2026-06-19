"""Build future labels after feature rows are frozen."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from feature_research.feature_config import FeatureResearchConfig
from feature_research.feature_models import FeatureLabel, FeatureRow, FutureDirection
from market_data.replay_events import ReplayEvent


@dataclass
class _PricePoint:
    timestamp_ms: int
    price: float


@dataclass
class LabelBuildStats:
    rows_labeled: int = 0
    insufficient_future: int = 0


class LabelBuilder:
    """Attaches future labels using strictly later events."""

    def __init__(self, config: FeatureResearchConfig) -> None:
        self._config = config
        self._price_timeline: dict[str, list[_PricePoint]] = {
            coin: [] for coin in config.coins
        }
        self._stats = LabelBuildStats()

    @property
    def stats(self) -> LabelBuildStats:
        return self._stats

    def ingest_event(self, event: ReplayEvent) -> None:
        """Build price timeline from all events (for labeling only)."""
        coin = event.coin
        if not coin or coin not in self._config.coins:
            return

        price: float | None = None
        if event.original_event_type == "trade":
            price = event.event.price
        elif event.original_event_type == "bbo":
            bbo = event.event
            price = (bbo.best_bid_price + bbo.best_ask_price) / 2.0

        if price is not None:
            self._price_timeline[coin].append(
                _PricePoint(timestamp_ms=event.primary_time_ms, price=price)
            )

    def build_labels(self, rows: tuple[FeatureRow, ...]) -> list[FeatureLabel]:
        """Label frozen feature rows using future price timeline."""
        for coin in self._price_timeline:
            self._price_timeline[coin].sort(key=lambda p: p.timestamp_ms)

        labels: list[FeatureLabel] = []
        for row in rows:
            label = self._label_row(row)
            labels.append(label)
            self._stats.rows_labeled += 1
        return labels

    def _label_row(self, row: FeatureRow) -> FeatureLabel:
        coin = row.coin
        timeline = self._price_timeline.get(coin, [])
        entry_price = row.latest_trade_price
        entry_ms = row.event_timestamp_ms
        cost_pct = self._config.round_trip_cost_pct

        labels: dict[str, Any] = {}
        max_horizon = max(self._config.horizons_seconds)

        future_points = [p for p in timeline if p.timestamp_ms > entry_ms]
        if not future_points:
            self._stats.insufficient_future += 1
            for horizon in self._config.horizons_seconds:
                labels[f"future_return_{horizon}s_pct"] = None
                labels[f"future_up_after_costs_{horizon}s"] = None
                labels[f"future_down_after_costs_{horizon}s"] = None
                labels[f"future_abs_move_after_costs_{horizon}s"] = None
                labels[f"future_direction_{horizon}s"] = FutureDirection.INSUFFICIENT_FUTURE_DATA.value
            labels["max_favorable_move_pct"] = None
            labels["max_adverse_move_pct"] = None
            labels["time_to_max_favorable_ms"] = None
            labels["time_to_max_adverse_ms"] = None
            return FeatureLabel(feature_row_id=row.feature_row_id, labels=labels)

        max_favorable = 0.0
        max_adverse = 0.0
        time_favorable: int | None = None
        time_adverse: int | None = None

        horizon_end_ms = entry_ms + max_horizon * 1000
        scan_points = [p for p in future_points if p.timestamp_ms <= horizon_end_ms]

        for point in scan_points:
            move_pct = ((point.price - entry_price) / entry_price) * 100.0 if entry_price else 0.0
            if move_pct > max_favorable:
                max_favorable = move_pct
                time_favorable = point.timestamp_ms - entry_ms
            if move_pct < max_adverse:
                max_adverse = move_pct
                time_adverse = point.timestamp_ms - entry_ms

        labels["max_favorable_move_pct"] = max_favorable
        labels["max_adverse_move_pct"] = max_adverse
        labels["time_to_max_favorable_ms"] = time_favorable
        labels["time_to_max_adverse_ms"] = time_adverse

        for horizon in self._config.horizons_seconds:
            target_ms = entry_ms + horizon * 1000
            future_price = self._price_at_or_after(future_points, target_ms)

            ret_key = f"future_return_{horizon}s_pct"
            up_key = f"future_up_after_costs_{horizon}s"
            down_key = f"future_down_after_costs_{horizon}s"
            abs_key = f"future_abs_move_after_costs_{horizon}s"
            dir_key = f"future_direction_{horizon}s"

            if future_price is None:
                labels[ret_key] = None
                labels[up_key] = None
                labels[down_key] = None
                labels[abs_key] = None
                labels[dir_key] = FutureDirection.INSUFFICIENT_FUTURE_DATA.value
                continue

            future_return = ((future_price - entry_price) / entry_price) * 100.0 if entry_price else 0.0
            labels[ret_key] = future_return

            net_up = future_return - cost_pct
            net_down = -future_return - cost_pct
            labels[up_key] = net_up > 0
            labels[down_key] = net_down > 0
            labels[abs_key] = abs(future_return) - cost_pct

            if abs(future_return) <= cost_pct:
                labels[dir_key] = FutureDirection.FLAT.value
            elif future_return > cost_pct:
                labels[dir_key] = FutureDirection.UP.value
            elif future_return < -cost_pct:
                labels[dir_key] = FutureDirection.DOWN.value
            else:
                labels[dir_key] = FutureDirection.FLAT.value

        return FeatureLabel(feature_row_id=row.feature_row_id, labels=labels)

    @staticmethod
    def _price_at_or_after(points: list[_PricePoint], target_ms: int) -> float | None:
        for point in points:
            if point.timestamp_ms >= target_ms:
                return point.price
        return None
