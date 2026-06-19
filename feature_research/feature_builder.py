"""Build feature rows from replay events (current/past data only)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from event_backtester.market_state import MarketStateManager
from feature_research.feature_config import FeatureResearchConfig
from feature_research.feature_models import (
    FEATURE_COLUMN_NAMES,
    FeatureRow,
    make_feature_row_id,
)
from market_analyzer import (
    PriceHistory,
    price_at_or_before,
    rolling_percent_change,
)
from market_data.replay_events import ReplayEvent


@dataclass
class FeatureBuildStats:
    events_processed: int = 0
    trade_events: int = 0
    bbo_events: int = 0
    rows_created: int = 0
    skipped_no_bbo: int = 0
    skipped_stale_bbo: int = 0
    skipped_no_price_history: int = 0
    skipped_wrong_coin: int = 0


@dataclass
class _FeatureState:
    """Extended per-coin state for feature calculation."""

    spread_bps_history: list[tuple[float, float]] = field(default_factory=list)
    bbo_imbalance_history: list[tuple[float, float]] = field(default_factory=list)
    trade_flow_history: list[dict[str, Any]] = field(default_factory=list)
    tick_directions: list[int] = field(default_factory=list)
    last_trade_price: float | None = None
    max_trade_flow: int = 500
    max_spread_history: int = 200


class FeatureBuilder:
    """Creates immutable feature rows at trade decision points."""

    def __init__(self, config: FeatureResearchConfig) -> None:
        self._config = config
        self._market_state = MarketStateManager(config.coins)
        self._feature_state: dict[str, _FeatureState] = {
            coin: _FeatureState() for coin in config.coins
        }
        self._rows: list[FeatureRow] = []
        self._stats = FeatureBuildStats()
        self._cross_coin_returns: dict[str, float | None] = {
            coin: None for coin in config.coins
        }

    @property
    def stats(self) -> FeatureBuildStats:
        return self._stats

    @property
    def rows(self) -> list[FeatureRow]:
        return list(self._rows)

    def process_event(self, event: ReplayEvent) -> FeatureRow | None:
        """Process one replay event. Returns new row if created."""
        self._stats.events_processed += 1
        coin = event.coin

        if event.original_event_type == "bbo":
            self._stats.bbo_events += 1
            self._market_state.update_from_event(event)
            return None

        if event.original_event_type != "trade":
            return None

        self._stats.trade_events += 1
        if not coin or coin not in self._config.coins:
            self._stats.skipped_wrong_coin += 1
            return None

        self._market_state.update_from_event(event)
        self._update_trade_flow(coin, event)

        if not self._market_state.bbo_available(coin, event.primary_time_ms, self._config.bbo_stale_ms):
            reason = self._market_state.bbo_stale_reason(
                coin, event.primary_time_ms, self._config.bbo_stale_ms
            )
            if reason:
                if self._market_state.get_state(coin) and self._market_state.get_state(coin).bbo_update_timestamp_ms:
                    self._stats.skipped_stale_bbo += 1
                else:
                    self._stats.skipped_no_bbo += 1
            return None

        state = self._market_state.get_state(coin)
        if state is None or state.latest_trade_price is None:
            self._stats.skipped_no_price_history += 1
            return None

        history = self._market_state.price_history.get(coin, [])
        if len(history) < 2:
            self._stats.skipped_no_price_history += 1
            return None

        self._update_cross_coin_returns(event.primary_time_ms)
        features, missing_flags = self._compute_features(coin, event, state)

        row = FeatureRow(
            feature_row_id=make_feature_row_id(
                event.replay_session_id, coin, event.replay_sequence
            ),
            replay_session_id=event.replay_session_id,
            source_recorder_session_id=event.source_recorder_session_id,
            coin=coin,
            event_timestamp_ms=event.primary_time_ms,
            received_timestamp_ms=event.received_timestamp_ms,
            latest_trade_price=state.latest_trade_price,
            latest_midpoint=state.midpoint,
            best_bid=state.best_bid_price,
            best_ask=state.best_ask_price,
            spread=state.spread,
            spread_bps=state.spread_bps,
            recent_trade_count=len(state.recent_trades),
            recent_volume=sum(t.get("size", 0) for t in state.recent_trades),
            current_event_type=event.original_event_type,
            source_replay_sequence=event.replay_sequence,
            feature_config_hash=self._config.config_hash(),
            features=features,
            missing_flags=missing_flags,
        )
        self._rows.append(row)
        self._stats.rows_created += 1
        return row

    def _update_trade_flow(self, coin: str, event: ReplayEvent) -> None:
        fs = self._feature_state[coin]
        trade = event.event
        now = event.primary_time_ms / 1000.0
        side = getattr(trade, "side", "")
        is_buy = side.upper() in ("B", "BUY")
        fs.trade_flow_history.append(
            {
                "timestamp_s": now,
                "side": "buy" if is_buy else "sell",
                "size": trade.size,
                "price": trade.price,
            }
        )
        if len(fs.trade_flow_history) > fs.max_trade_flow:
            fs.trade_flow_history = fs.trade_flow_history[-fs.max_trade_flow :]

        if fs.last_trade_price is not None:
            if trade.price > fs.last_trade_price:
                fs.tick_directions.append(1)
            elif trade.price < fs.last_trade_price:
                fs.tick_directions.append(-1)
            else:
                fs.tick_directions.append(0)
        fs.last_trade_price = trade.price

        state = self._market_state.get_state(coin)
        if state and state.spread_bps is not None:
            fs.spread_bps_history.append((now, state.spread_bps))
            if len(fs.spread_bps_history) > fs.max_spread_history:
                fs.spread_bps_history = fs.spread_bps_history[-fs.max_spread_history :]

        if state and state.best_bid_size and state.best_ask_size:
            total = state.best_bid_size + state.best_ask_size
            imbalance = (state.best_bid_size - state.best_ask_size) / total if total else 0.0
            fs.bbo_imbalance_history.append((now, imbalance))
            if len(fs.bbo_imbalance_history) > fs.max_spread_history:
                fs.bbo_imbalance_history = fs.bbo_imbalance_history[-fs.max_spread_history :]

    def _update_cross_coin_returns(self, event_time_ms: int) -> None:
        now = event_time_ms / 1000.0
        for coin in self._config.coins:
            history = self._market_state.price_history.get(coin, [])
            if not history:
                self._cross_coin_returns[coin] = None
                continue
            ret = rolling_percent_change(history, history[-1][1], now, 30)
            self._cross_coin_returns[coin] = ret

    def _compute_features(
        self, coin: str, event: ReplayEvent, state: Any
    ) -> tuple[dict[str, Any], dict[str, bool]]:
        now = event.primary_time_ms / 1000.0
        price = state.latest_trade_price
        history = self._market_state.price_history.get(coin, [])
        mid_history = self._market_state.mid_history.get(coin, [])
        fs = self._feature_state[coin]

        features: dict[str, Any] = {}
        missing: dict[str, bool] = {}

        def set_feature(name: str, value: Any) -> None:
            features[name] = value
            missing[name] = value is None

        for window in (5, 10, 30, 60):
            set_feature(
                f"return_{window}s_pct",
                rolling_percent_change(history, price, now, window),
            )

        for window in (10, 30, 60):
            set_feature(f"volatility_{window}s", self._volatility(history, now, window))

        set_feature("spread", state.spread)
        set_feature("spread_bps", state.spread_bps)
        set_feature(
            "spread_bps_percentile_recent",
            self._spread_percentile(fs.spread_bps_history, state.spread_bps, now),
        )
        set_feature("bbo_available", True)
        bbo_age = (
            event.primary_time_ms - state.bbo_update_timestamp_ms
            if state.bbo_update_timestamp_ms
            else None
        )
        set_feature("bbo_age_ms", bbo_age)
        set_feature("bid_size", state.best_bid_size)
        set_feature("ask_size", state.best_ask_size)

        total_size = (state.best_bid_size or 0) + (state.best_ask_size or 0)
        imbalance = (
            ((state.best_bid_size or 0) - (state.best_ask_size or 0)) / total_size
            if total_size
            else None
        )
        set_feature("bbo_imbalance", imbalance)
        set_feature(
            "bbo_imbalance_change_10s",
            self._imbalance_change(fs.bbo_imbalance_history, now, 10),
        )
        set_feature(
            "bbo_imbalance_change_30s",
            self._imbalance_change(fs.bbo_imbalance_history, now, 30),
        )

        flow_10 = self._trade_flow_window(fs.trade_flow_history, now, 10)
        flow_30 = self._trade_flow_window(fs.trade_flow_history, now, 30)
        set_feature("buy_trade_count_10s", flow_10["buy_count"])
        set_feature("sell_trade_count_10s", flow_10["sell_count"])
        set_feature("buy_volume_10s", flow_10["buy_volume"])
        set_feature("sell_volume_10s", flow_10["sell_volume"])
        set_feature("trade_imbalance_10s", flow_10["imbalance"])
        set_feature("trade_imbalance_30s", flow_30["imbalance"])
        set_feature("volume_burst_30s", flow_30["total_volume"])

        set_feature(
            "price_acceleration_10s",
            self._acceleration(history, now, 10),
        )
        set_feature(
            "price_acceleration_30s",
            self._acceleration(history, now, 30),
        )
        set_feature("consecutive_up_ticks", self._consecutive_ticks(fs.tick_directions, 1))
        set_feature("consecutive_down_ticks", self._consecutive_ticks(fs.tick_directions, -1))

        for cross_coin in ("BTC", "ETH", "SOL"):
            key = f"{cross_coin.lower()}_return_30s"
            set_feature(key, self._cross_coin_returns.get(cross_coin))

        coin_ret = self._cross_coin_returns.get(coin)
        btc_ret = self._cross_coin_returns.get("BTC")
        eth_ret = self._cross_coin_returns.get("ETH")
        if coin_ret is not None and btc_ret is not None:
            set_feature("coin_minus_btc_return_30s", coin_ret - btc_ret)
        else:
            set_feature("coin_minus_btc_return_30s", None)
        if coin_ret is not None and eth_ret is not None:
            set_feature("coin_minus_eth_return_30s", coin_ret - eth_ret)
        else:
            set_feature("coin_minus_eth_return_30s", None)

        for name in FEATURE_COLUMN_NAMES:
            if name not in features:
                set_feature(name, None)

        return features, missing

    @staticmethod
    def _volatility(history: list[tuple[float, float]], now: float, window: int) -> float | None:
        cutoff = now - window
        prices = [p for t, p in history if t >= cutoff]
        if len(prices) < 3:
            return None
        returns = [
            (prices[i] - prices[i - 1]) / prices[i - 1] * 100
            for i in range(1, len(prices))
            if prices[i - 1] != 0
        ]
        if len(returns) < 2:
            return None
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance)

    @staticmethod
    def _spread_percentile(
        history: list[tuple[float, float]], current: float | None, now: float
    ) -> float | None:
        if current is None or not history:
            return None
        recent = [v for t, v in history if t >= now - 60]
        if len(recent) < 5:
            return None
        sorted_vals = sorted(recent)
        rank = sum(1 for v in sorted_vals if v <= current)
        return (rank / len(sorted_vals)) * 100.0

    @staticmethod
    def _imbalance_change(
        history: list[tuple[float, float]], now: float, window: int
    ) -> float | None:
        if not history:
            return None
        current = history[-1][1]
        past = price_at_or_before(history, now - window)
        if past is None:
            return None
        return current - past[1]

    @staticmethod
    def _trade_flow_window(
        history: list[dict[str, Any]], now: float, window: int
    ) -> dict[str, Any]:
        cutoff = now - window
        recent = [t for t in history if t["timestamp_s"] >= cutoff]
        buy_count = sum(1 for t in recent if t["side"] == "buy")
        sell_count = sum(1 for t in recent if t["side"] == "sell")
        buy_vol = sum(t["size"] for t in recent if t["side"] == "buy")
        sell_vol = sum(t["size"] for t in recent if t["side"] == "sell")
        total = buy_vol + sell_vol
        imbalance = (buy_vol - sell_vol) / total if total else 0.0
        return {
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "imbalance": imbalance,
            "total_volume": total,
        }

    @staticmethod
    def _acceleration(history: list[tuple[float, float]], now: float, window: int) -> float | None:
        half = window // 2
        if half < 1:
            return None
        ret_first = rolling_percent_change(history, history[-1][1], now, window)
        ret_second = rolling_percent_change(history, history[-1][1], now, half)
        if ret_first is None or ret_second is None:
            return None
        return ret_second - (ret_first - ret_second)

    @staticmethod
    def _consecutive_ticks(directions: list[int], target: int) -> int:
        count = 0
        for d in reversed(directions):
            if d == target:
                count += 1
            elif d != 0:
                break
        return count

    def freeze_rows(self) -> tuple[FeatureRow, ...]:
        """Return immutable tuple of feature rows."""
        return tuple(self._rows)
