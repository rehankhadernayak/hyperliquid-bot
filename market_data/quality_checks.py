"""Offline quality checks for recorder session data."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from market_data.gap_detection import GapDetector
from market_data.quality_models import (
    BACKWARD_TIMESTAMP_CRITICAL_COUNT,
    BACKWARD_TIMESTAMP_DEGRADED_COUNT,
    BBO_STALE_MS,
    CANDLE_INTERVAL_MS,
    CHANNEL_STALENESS_MS,
    CHANNEL_TO_EVENT_TYPE,
    DUPLICATE_TIMESTAMP_RATE_WARNING,
    EXCHANGE_TIME_GAP_MS,
    IssueSeverity,
    IntegrityWarning,
    L2_STALE_MS,
    LATENCY_P95_WARNING_MS,
    LATENCY_P99_WARNING_MS,
    MIN_RAW_EVENTS_RESEARCH,
    MIN_SESSION_DURATION_RESEARCH_SEC,
    MALFORMED_RATE_DEGRADED,
    MALFORMED_RATE_UNUSABLE,
    MAX_RECONNECT_COUNT_GOOD,
    MAX_RECONNECT_COUNT_USABLE,
    MISSING_NORMALIZED_FRACTION_DEGRADED,
    NEGATIVE_LATENCY_WARNING_COUNT,
    NORMALIZED_COUNT_TOLERANCE,
    NORMALIZED_TO_RAW_EXPANSION_MAX,
    RAW_COUNT_TOLERANCE,
    RECEIVE_TIME_GAP_MS,
    TRADE_PRICE_JUMP_WARNING,
    WATCH_CHANNELS,
    WATCH_COINS,
    ChannelSummary,
    CoinSummary,
    GapSummaryEntry,
    LatencySummary,
    QualityLabel,
)
from market_data.replay_manifest import ManifestFileEntry, SessionManifest


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    idx = int(math.ceil(p * len(sorted_vals))) - 1
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]


def _extract_coin(record: dict[str, Any]) -> str | None:
    channel = record.get("channel")
    subscription = record.get("subscription") or {}
    if isinstance(subscription, dict) and subscription.get("coin"):
        return str(subscription["coin"])

    payload = record.get("raw_payload") or {}
    data = payload.get("data")
    if isinstance(data, dict) and data.get("coin"):
        return str(data["coin"])
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return str(data[0].get("coin", "")) or None
    if channel == "allMids":
        return None
    return None


def _iter_raw_records(
    raw_files: list[ManifestFileEntry],
    session_id: str,
) -> Iterator[tuple[dict[str, Any] | None, str | None, ManifestFileEntry]]:
    """Yield (record, parse_error, file_entry) from raw JSONL files."""
    for file_entry in raw_files:
        if not file_entry.exists or not file_entry.readable:
            continue
        try:
            with file_entry.path.open(encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        yield None, f"{file_entry.path}:{line_no}: {exc}", file_entry
                        continue
                    if record.get("recorder_session_id") != session_id:
                        continue
                    yield record, None, file_entry
        except OSError as exc:
            yield None, f"{file_entry.path}: {exc}", file_entry


@dataclass
class InspectionState:
    channel_summaries: dict[str, ChannelSummary] = field(default_factory=dict)
    coin_summaries: dict[str, CoinSummary] = field(default_factory=dict)
    latency_by_channel: dict[str, LatencySummary] = field(default_factory=dict)
    warnings: list[IntegrityWarning] = field(default_factory=list)
    gaps: list[GapSummaryEntry] = field(default_factory=list)
    raw_inspected_count: int = 0
    raw_malformed_count: int = 0
    raw_unreadable_files: int = 0
    normalized_inspected_count: int = 0
    normalized_missing_files: int = 0
    bbo_spread_bps: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    trade_keys_seen: set[str] = field(default_factory=set)
    duplicate_trade_keys: int = 0
    recent_mid_by_coin: dict[str, float] = field(default_factory=dict)
    candle_opens: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    local_ts_last_by_file: dict[str, int] = field(default_factory=dict)
    exchange_ts_last: dict[str, int] = field(default_factory=dict)
    exchange_ts_duplicates: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    backward_exchange_count: int = 0
    local_backward_count: int = 0
    out_of_range_count: int = 0
    file_rotation_issues: int = 0
    last_sequence: int | None = None
    gap_detector: GapDetector = field(default_factory=lambda: GapDetector(CANDLE_INTERVAL_MS))

    def channel(self, name: str) -> ChannelSummary:
        if name not in self.channel_summaries:
            self.channel_summaries[name] = ChannelSummary(channel=name)
        return self.channel_summaries[name]

    def coin(self, name: str) -> CoinSummary:
        if name not in self.coin_summaries:
            self.coin_summaries[name] = CoinSummary(coin=name)
        return self.coin_summaries[name]

    def latency(self, name: str) -> LatencySummary:
        if name not in self.latency_by_channel:
            self.latency_by_channel[name] = LatencySummary(channel=name)
        return self.latency_by_channel[name]

    def add_warning(
        self,
        severity: IssueSeverity,
        category: str,
        message: str,
        channel: str | None = None,
        coin: str | None = None,
        **details: Any,
    ) -> None:
        self.warnings.append(
            IntegrityWarning(
                severity=severity.value,
                category=category,
                message=message,
                channel=channel,
                coin=coin,
                details=details,
            )
        )


def inspect_session(manifest: SessionManifest) -> InspectionState:
    """Run all quality checks against manifest-listed files."""
    state = InspectionState()

    if not manifest.exists:
        state.add_warning(
            IssueSeverity.CRITICAL,
            "manifest",
            "Session manifest is missing or unreadable",
        )
        return state

    session_id = manifest.session_id
    start_ms = manifest.start_time_ms
    end_ms = manifest.end_time_ms

    for ch in WATCH_CHANNELS:
        state.channel(ch)
    for coin in manifest.coins or WATCH_COINS:
        state.coin(coin)

    state.latency("allMids").latency_available = False
    state.latency("allMids").note = "allMids messages do not carry exchange send timestamps"
    state.latency("candle").informational_only = True
    state.latency("candle").note = (
        "Candle timestamps represent candle open/close times, not message-send latency"
    )
    state.latency("activeAssetCtx").informational_only = True
    state.latency("activeAssetCtx").note = (
        "activeAssetCtx timestamps are not comparable to trade event timestamps"
    )

    missing_raw = [f for f in manifest.raw_files if not f.exists]
    if missing_raw:
        for f in missing_raw:
            state.add_warning(
                IssueSeverity.CRITICAL,
                "missing_file",
                f"Raw file listed in manifest is missing: {f.path}",
            )
    unreadable_raw = [f for f in manifest.raw_files if f.exists and not f.readable]
    state.raw_unreadable_files = len(unreadable_raw)
    for f in unreadable_raw:
        state.add_warning(
            IssueSeverity.CRITICAL,
            "unreadable_file",
            f"Raw file unreadable: {f.path}",
        )

    missing_norm = [f for f in manifest.normalized_files if not f.exists]
    state.normalized_missing_files = len(missing_norm)
    if missing_norm:
        fraction = len(missing_norm) / max(len(manifest.normalized_files), 1)
        severity = (
            IssueSeverity.DEGRADED
            if fraction >= MISSING_NORMALIZED_FRACTION_DEGRADED
            else IssueSeverity.WARNING
        )
        state.add_warning(
            severity,
            "missing_file",
            f"{len(missing_norm)} normalized file(s) missing from manifest list",
            missing_fraction=round(fraction, 4),
        )

    for file_entry in manifest.normalized_files:
        if file_entry.exists and file_entry.row_count is not None:
            state.normalized_inspected_count += file_entry.row_count
            _accumulate_normalized_channel_count(state, file_entry)
        elif file_entry.exists:
            state.add_warning(
                IssueSeverity.WARNING,
                "normalized",
                f"Could not read normalized file: {file_entry.path}",
            )

    _inspect_raw_records(state, manifest, session_id, start_ms, end_ms)
    _merge_manifest_gaps(state, manifest)
    _finalize_candle_continuity_warnings(state)
    _finalize_summaries(state, manifest)
    _assign_channel_labels(state, manifest)
    return state


def _accumulate_normalized_channel_count(state: InspectionState, file_entry: ManifestFileEntry) -> None:
    path_str = str(file_entry.path).replace("\\", "/")
    for channel, event_type in CHANNEL_TO_EVENT_TYPE.items():
        if f"event_type={event_type}/" in path_str:
            state.channel(channel).normalized_event_count += file_entry.row_count or 0
            break


def _inspect_raw_records(
    state: InspectionState,
    manifest: SessionManifest,
    session_id: str,
    start_ms: int | None,
    end_ms: int | None,
) -> None:
    latency_samples: dict[str, list[float]] = defaultdict(list)
    bbo_last_recv: dict[str, int] = {}
    l2_last_recv: dict[str, int] = {}
    bbo_stale_counts: dict[str, int] = defaultdict(int)
    l2_stale_counts: dict[str, int] = defaultdict(int)

    for record, parse_error, file_entry in _iter_raw_records(manifest.raw_files, session_id):
        if parse_error:
            state.raw_malformed_count += 1
            state.add_warning(
                IssueSeverity.CRITICAL,
                "malformed_raw",
                parse_error,
            )
            continue
        if record is None:
            continue

        state.raw_inspected_count += 1
        channel = str(record.get("channel", "unknown"))
        coin = _extract_coin(record)
        recv_ms = int(record.get("received_timestamp_ms") or 0)
        exch_ms = record.get("exchange_timestamp_ms")
        sequence = record.get("sequence")

        ch_sum = state.channel(channel)
        ch_sum.raw_event_count += 1
        if ch_sum.first_received_timestamp_ms is None or recv_ms < ch_sum.first_received_timestamp_ms:
            ch_sum.first_received_timestamp_ms = recv_ms
        if ch_sum.last_received_timestamp_ms is None or recv_ms > ch_sum.last_received_timestamp_ms:
            ch_sum.last_received_timestamp_ms = recv_ms

        if coin and coin in state.coin_summaries:
            c_sum = state.coin(coin)
            c_sum.raw_event_count += 1
            if c_sum.first_received_timestamp_ms is None or recv_ms < c_sum.first_received_timestamp_ms:
                c_sum.first_received_timestamp_ms = recv_ms
            if c_sum.last_received_timestamp_ms is None or recv_ms > c_sum.last_received_timestamp_ms:
                c_sum.last_received_timestamp_ms = recv_ms

        file_key = str(file_entry.path)
        prev_local = state.local_ts_last_by_file.get(file_key)
        if prev_local is not None and recv_ms < prev_local:
            state.local_backward_count += 1
            state.add_warning(
                IssueSeverity.WARNING,
                "timestamp_order",
                f"Local received timestamp went backward in {file_key}",
                channel=channel,
                coin=coin,
            )
        state.local_ts_last_by_file[file_key] = recv_ms

        if sequence is not None and state.last_sequence is not None:
            if int(sequence) < state.last_sequence:
                state.file_rotation_issues += 1
        if sequence is not None:
            state.last_sequence = int(sequence)

        if start_ms and recv_ms < start_ms:
            state.out_of_range_count += 1
        if end_ms and recv_ms > end_ms:
            state.out_of_range_count += 1

        if exch_ms is not None:
            exch_ms = int(exch_ms)
            ts_key = f"{channel}:{coin or 'global'}"
            prev_exch = state.exchange_ts_last.get(ts_key)
            if prev_exch is not None:
                if exch_ms < prev_exch:
                    state.backward_exchange_count += 1
                    state.gap_detector.record_exchange_timestamp(channel, coin, exch_ms, recv_ms)
                elif exch_ms == prev_exch:
                    state.exchange_ts_duplicates[ts_key] += 1
                gap_ms = exch_ms - prev_exch
                if gap_ms > EXCHANGE_TIME_GAP_MS:
                    state.add_warning(
                        IssueSeverity.WARNING,
                        "exchange_gap",
                        f"Large exchange timestamp gap ({gap_ms} ms) on {channel}",
                        channel=channel,
                        coin=coin,
                        gap_ms=gap_ms,
                    )
            state.exchange_ts_last[ts_key] = exch_ms

            if prev_local is not None:
                recv_gap = recv_ms - prev_local
                if recv_gap > RECEIVE_TIME_GAP_MS:
                    state.gaps.append(
                        GapSummaryEntry(
                            gap_type="RECEIVE_TIME_GAP",
                            coin=coin,
                            channel=channel,
                            start_ms=prev_local,
                            end_ms=recv_ms,
                            duration_ms=recv_gap,
                            certainty="INFERRED",
                            severity=IssueSeverity.WARNING.value,
                        )
                    )

        if channel in ("trades", "bbo", "l2Book") and exch_ms is not None:
            lat = recv_ms - int(exch_ms)
            latency_samples[channel].append(float(lat))
            lat_sum = state.latency(channel)
            lat_sum.count += 1
            if lat < 0:
                lat_sum.negative_latency_count += 1
        elif channel == "candle" and exch_ms is not None:
            lat_sum = state.latency("candle")
            lat_sum.count += 1
            latency_samples["candle"].append(float(recv_ms - int(exch_ms)))
        elif channel == "activeAssetCtx":
            state.latency("activeAssetCtx").missing_timestamp_count += 1
        elif channel == "allMids":
            state.latency("allMids").missing_timestamp_count += 1

        if channel == "bbo":
            _check_bbo_raw(state, record, coin, bbo_last_recv, bbo_stale_counts)
        elif channel == "l2Book":
            _check_l2_raw(state, record, coin, l2_last_recv, l2_stale_counts)
        elif channel == "trades":
            _check_trades_raw(state, record, coin)
        elif channel == "candle":
            _check_candle_raw(state, record, coin, end_ms)
        elif channel == "allMids":
            _check_all_mids_raw(state, record)

    for channel, samples in latency_samples.items():
        lat_sum = state.latency(channel)
        if not samples:
            continue
        lat_sum.minimum_ms = min(samples)
        lat_sum.maximum_ms = max(samples)
        lat_sum.median_ms = _percentile(samples, 0.5)
        lat_sum.p90_ms = _percentile(samples, 0.90)
        lat_sum.p95_ms = _percentile(samples, 0.95)
        lat_sum.p99_ms = _percentile(samples, 0.99)
        if lat_sum.negative_latency_count >= NEGATIVE_LATENCY_WARNING_COUNT:
            state.add_warning(
                IssueSeverity.WARNING,
                "clock_quality",
                f"{lat_sum.negative_latency_count} negative latency sample(s) on {channel}",
                channel=channel,
            )
        if lat_sum.informational_only:
            continue
        if lat_sum.p95_ms and lat_sum.p95_ms > LATENCY_P95_WARNING_MS:
            state.add_warning(
                IssueSeverity.WARNING,
                "latency",
                f"High p95 latency on {channel}: {lat_sum.p95_ms:.0f} ms",
                channel=channel,
            )
        if lat_sum.p99_ms and lat_sum.p99_ms > LATENCY_P99_WARNING_MS:
            state.add_warning(
                IssueSeverity.WARNING,
                "latency",
                f"High p99 latency on {channel}: {lat_sum.p99_ms:.0f} ms",
                channel=channel,
            )

    for coin, count in bbo_stale_counts.items():
        if count:
            state.channel("bbo").stale_period_count += count
            state.coin(coin).stale_period_count += count
    for coin, count in l2_stale_counts.items():
        if count:
            state.channel("l2Book").stale_period_count += count
            state.coin(coin).stale_period_count += count


def _check_bbo_raw(
    state: InspectionState,
    record: dict[str, Any],
    coin: str | None,
    bbo_last_recv: dict[str, int],
    stale_counts: dict[str, int],
) -> None:
    data = (record.get("raw_payload") or {}).get("data") or {}
    bbo = data.get("bbo") or []
    if len(bbo) < 2:
        state.add_warning(
            IssueSeverity.DEGRADED,
            "bbo_sanity",
            "BBO message missing bid/ask levels",
            channel="bbo",
            coin=coin,
        )
        return

    bid = bbo[0] or {}
    ask = bbo[1] or {}
    try:
        bid_px = float(bid.get("px", 0))
        ask_px = float(ask.get("px", 0))
        bid_sz = float(bid.get("sz", 0))
        ask_sz = float(ask.get("sz", 0))
    except (TypeError, ValueError):
        state.channel("bbo").malformed_count += 1
        return

    if bid_px <= 0 or ask_px <= 0:
        state.add_warning(
            IssueSeverity.CRITICAL,
            "bbo_sanity",
            "Non-positive BBO price",
            channel="bbo",
            coin=coin,
        )
    if bid_sz < 0 or ask_sz < 0:
        state.add_warning(
            IssueSeverity.DEGRADED,
            "bbo_sanity",
            "Negative BBO size",
            channel="bbo",
            coin=coin,
        )

    spread = ask_px - bid_px
    spread_bps = (spread / bid_px * 10000) if bid_px > 0 else 0
    if spread < 0 or bid_px > ask_px:
        state.add_warning(
            IssueSeverity.CRITICAL,
            "bbo_sanity",
            f"Crossed BBO market: bid={bid_px} ask={ask_px}",
            channel="bbo",
            coin=coin,
        )
    elif spread_bps >= 0 and coin:
        state.bbo_spread_bps[coin].append(spread_bps)

    recv_ms = int(record.get("received_timestamp_ms") or 0)
    if coin:
        prev = bbo_last_recv.get(coin)
        if prev is not None and recv_ms - prev > BBO_STALE_MS:
            stale_counts[coin] += 1
        bbo_last_recv[coin] = recv_ms
        state.recent_mid_by_coin[coin] = (bid_px + ask_px) / 2


def _check_l2_raw(
    state: InspectionState,
    record: dict[str, Any],
    coin: str | None,
    l2_last_recv: dict[str, int],
    stale_counts: dict[str, int],
) -> None:
    data = (record.get("raw_payload") or {}).get("data") or {}
    levels = data.get("levels") or [[], []]
    bids = levels[0] if len(levels) > 0 else []
    asks = levels[1] if len(levels) > 1 else []

    bid_prices: list[float] = []
    ask_prices: list[float] = []
    for lvl in bids:
        try:
            px = float(lvl.get("px", 0))
            sz = float(lvl.get("sz", 0))
            if px <= 0:
                state.add_warning(
                    IssueSeverity.DEGRADED,
                    "l2_sanity",
                    "Non-positive bid price in L2 book",
                    channel="l2Book",
                    coin=coin,
                )
            if sz < 0:
                state.add_warning(
                    IssueSeverity.DEGRADED,
                    "l2_sanity",
                    "Negative bid size in L2 book",
                    channel="l2Book",
                    coin=coin,
                )
            bid_prices.append(px)
        except (TypeError, ValueError, AttributeError):
            state.channel("l2Book").malformed_count += 1

    for lvl in asks:
        try:
            px = float(lvl.get("px", 0))
            sz = float(lvl.get("sz", 0))
            if px <= 0:
                state.add_warning(
                    IssueSeverity.DEGRADED,
                    "l2_sanity",
                    "Non-positive ask price in L2 book",
                    channel="l2Book",
                    coin=coin,
                )
            if sz < 0:
                state.add_warning(
                    IssueSeverity.DEGRADED,
                    "l2_sanity",
                    "Negative ask size in L2 book",
                    channel="l2Book",
                    coin=coin,
                )
            ask_prices.append(px)
        except (TypeError, ValueError, AttributeError):
            state.channel("l2Book").malformed_count += 1

    if len(bid_prices) != len(set(bid_prices)):
        state.add_warning(
            IssueSeverity.WARNING,
            "l2_sanity",
            "Duplicate bid price levels in L2 book",
            channel="l2Book",
            coin=coin,
        )
    if len(ask_prices) != len(set(ask_prices)):
        state.add_warning(
            IssueSeverity.WARNING,
            "l2_sanity",
            "Duplicate ask price levels in L2 book",
            channel="l2Book",
            coin=coin,
        )

    if bid_prices and bid_prices != sorted(bid_prices, reverse=True):
        state.add_warning(
            IssueSeverity.WARNING,
            "l2_sanity",
            "Bid levels not sorted descending",
            channel="l2Book",
            coin=coin,
        )
    if ask_prices and ask_prices != sorted(ask_prices):
        state.add_warning(
            IssueSeverity.WARNING,
            "l2_sanity",
            "Ask levels not sorted ascending",
            channel="l2Book",
            coin=coin,
        )

    if bid_prices and ask_prices and max(bid_prices) > min(ask_prices):
        state.add_warning(
            IssueSeverity.CRITICAL,
            "l2_sanity",
            "Crossed L2 book: best bid exceeds best ask",
            channel="l2Book",
            coin=coin,
        )

    recv_ms = int(record.get("received_timestamp_ms") or 0)
    if coin:
        prev = l2_last_recv.get(coin)
        if prev is not None and recv_ms - prev > L2_STALE_MS:
            stale_counts[coin] += 1
        l2_last_recv[coin] = recv_ms


def _check_trades_raw(state: InspectionState, record: dict[str, Any], coin: str | None) -> None:
    data = (record.get("raw_payload") or {}).get("data") or []
    if not isinstance(data, list):
        state.channel("trades").malformed_count += 1
        return

    for trade in data:
        try:
            px = float(trade.get("px", 0))
            sz = float(trade.get("sz", 0))
            side = trade.get("side")
            tid = trade.get("tid")
            tx_hash = trade.get("hash")
            trade_coin = str(trade.get("coin", coin or ""))
        except (TypeError, ValueError):
            state.channel("trades").malformed_count += 1
            continue

        if px <= 0 or sz <= 0:
            state.add_warning(
                IssueSeverity.DEGRADED,
                "trade_sanity",
                f"Invalid trade price/size: px={px} sz={sz}",
                channel="trades",
                coin=trade_coin,
            )
        if side not in ("A", "B"):
            state.add_warning(
                IssueSeverity.WARNING,
                "trade_sanity",
                f"Unexpected trade side: {side!r}",
                channel="trades",
                coin=trade_coin,
            )
        if tid is None:
            state.add_warning(
                IssueSeverity.WARNING,
                "trade_sanity",
                "Trade missing tid",
                channel="trades",
                coin=trade_coin,
            )
        else:
            key = f"{trade_coin}:{trade.get('time')}:{tid}"
            if key in state.trade_keys_seen:
                state.duplicate_trade_keys += 1
                state.channel("trades").duplicate_count += 1
            state.trade_keys_seen.add(key)

        if not tx_hash:
            state.add_warning(
                IssueSeverity.INFO,
                "trade_sanity",
                "Trade missing transaction hash",
                channel="trades",
                coin=trade_coin,
            )

        mid = state.recent_mid_by_coin.get(trade_coin)
        if mid and mid > 0:
            jump = abs(px - mid) / mid
            if jump > TRADE_PRICE_JUMP_WARNING:
                state.add_warning(
                    IssueSeverity.INFO,
                    "trade_outlier",
                    f"Trade price {px} deviates {jump:.1%} from recent mid {mid:.2f}",
                    channel="trades",
                    coin=trade_coin,
                    price=px,
                    mid=mid,
                )


def _check_candle_raw(
    state: InspectionState,
    record: dict[str, Any],
    coin: str | None,
    session_end_ms: int | None,
) -> None:
    data = (record.get("raw_payload") or {}).get("data") or {}
    try:
        open_time = int(data.get("t", 0))
        close_time = int(data.get("T", 0))
        o = float(data.get("o", 0))
        h = float(data.get("h", 0))
        l = float(data.get("l", 0))
        c = float(data.get("c", 0))
        v = float(data.get("v", 0))
        n = data.get("n")
        trade_count = int(n) if n is not None else None
    except (TypeError, ValueError):
        state.channel("candle").malformed_count += 1
        return

    if h < max(o, c, l) or l > min(o, c, h):
        state.add_warning(
            IssueSeverity.DEGRADED,
            "candle_ohlc",
            f"Invalid OHLC: o={o} h={h} l={l} c={c}",
            channel="candle",
            coin=coin,
        )
    if v < 0:
        state.add_warning(
            IssueSeverity.DEGRADED,
            "candle_ohlc",
            f"Negative candle volume: {v}",
            channel="candle",
            coin=coin,
        )
    if trade_count is not None and trade_count < 0:
        state.add_warning(
            IssueSeverity.DEGRADED,
            "candle_ohlc",
            f"Negative trade_count: {trade_count}",
            channel="candle",
            coin=coin,
        )

    candle_coin = coin or str(data.get("s", ""))
    key = f"{candle_coin}:1m"
    opens = state.candle_opens[key]
    if open_time in opens:
        ch_sum = state.channel("candle")
        ch_sum.duplicate_count += 1
    elif opens and open_time < opens[-1]:
        state.add_warning(
            IssueSeverity.WARNING,
            "candle_continuity",
            f"Out-of-order candle open {open_time} after {opens[-1]}",
            channel="candle",
            coin=candle_coin,
        )
    elif opens:
        expected = opens[-1] + CANDLE_INTERVAL_MS
        if open_time > expected:
            partial_current_candle = (
                session_end_ms is not None
                and expected <= session_end_ms < expected + CANDLE_INTERVAL_MS
            )
            if not partial_current_candle:
                state.gaps.append(
                    GapSummaryEntry(
                        gap_type="MISSING_CANDLE_PERIOD",
                        coin=candle_coin,
                        channel="candle",
                        start_ms=expected,
                        end_ms=open_time,
                        duration_ms=open_time - expected,
                        certainty="INFERRED",
                        severity=IssueSeverity.WARNING.value,
                    )
                )
                state.channel("candle").gap_count += 1
    opens.append(open_time)
    state.gap_detector.record_candle(candle_coin or "unknown", open_time)


def _finalize_candle_continuity_warnings(state: InspectionState) -> None:
    """Summarize candle duplicate open times (expected for streaming bar updates)."""
    for ch_sum in state.channel_summaries.values():
        if ch_sum.channel == "candle" and ch_sum.duplicate_count > 0:
            state.add_warning(
                IssueSeverity.INFO,
                "candle_continuity",
                (
                    f"{ch_sum.duplicate_count} candle message(s) reused existing open times "
                    "(normal for streaming partial bar updates)"
                ),
                channel="candle",
            )


def _check_all_mids_raw(state: InspectionState, record: dict[str, Any]) -> None:
    data = (record.get("raw_payload") or {}).get("data") or {}
    mids = data.get("mids") or {}
    if not isinstance(mids, dict) or not mids:
        state.channel("allMids").malformed_count += 1


def _merge_manifest_gaps(state: InspectionState, manifest: SessionManifest) -> None:
    gap_stats = manifest.gap_statistics
    for gap in gap_stats.get("gaps") or []:
        gap_type = gap.get("gap_type", "UNKNOWN")
        severity = IssueSeverity.INFO.value
        if gap_type == "BACKWARD_EXCHANGE_TIMESTAMP":
            severity = IssueSeverity.WARNING.value
        elif gap_type == "RECONNECT_WINDOW":
            severity = IssueSeverity.DEGRADED.value
        elif gap_type == "CHANNEL_STALENESS":
            severity = IssueSeverity.DEGRADED.value
        elif gap_type == "MISSING_CANDLE_PERIOD":
            severity = IssueSeverity.WARNING.value

        entry = GapSummaryEntry(
            gap_type=gap_type,
            coin=gap.get("coin"),
            channel=str(gap.get("channel", "unknown")),
            start_ms=int(gap.get("started_at_ms") or 0),
            end_ms=int(gap.get("ended_at_ms") or 0),
            duration_ms=int(gap.get("duration_ms") or 0),
            certainty=str(gap.get("certainty", "INFERRED")),
            severity=severity,
        )
        if not any(
            g.gap_type == entry.gap_type
            and g.start_ms == entry.start_ms
            and g.channel == entry.channel
            for g in state.gaps
        ):
            state.gaps.append(entry)


def _finalize_summaries(state: InspectionState, manifest: SessionManifest) -> None:
    manifest_data = manifest.manifest_data or {}
    dup_manifest = int(manifest_data.get("duplicate_count") or 0)
    malformed_manifest = int(manifest_data.get("malformed_count") or 0)

    for ch_sum in state.channel_summaries.values():
        if ch_sum.first_received_timestamp_ms and ch_sum.last_received_timestamp_ms:
            ch_sum.observed_duration_ms = (
                ch_sum.last_received_timestamp_ms - ch_sum.first_received_timestamp_ms
            )
            if ch_sum.observed_duration_ms > 0:
                ch_sum.messages_per_second = round(
                    ch_sum.raw_event_count / (ch_sum.observed_duration_ms / 1000),
                    4,
                )

    for c_sum in state.coin_summaries.values():
        if c_sum.first_received_timestamp_ms and c_sum.last_received_timestamp_ms:
            c_sum.observed_duration_ms = (
                c_sum.last_received_timestamp_ms - c_sum.first_received_timestamp_ms
            )
            if c_sum.observed_duration_ms > 0:
                c_sum.messages_per_second = round(
                    c_sum.raw_event_count / (c_sum.observed_duration_ms / 1000),
                    4,
                )

    raw_manifest = manifest.raw_row_count_manifest
    if raw_manifest > 0:
        diff = abs(state.raw_inspected_count - raw_manifest) / raw_manifest
        if diff > RAW_COUNT_TOLERANCE:
            state.add_warning(
                IssueSeverity.DEGRADED if diff > RAW_COUNT_TOLERANCE * 2 else IssueSeverity.WARNING,
                "count_consistency",
                f"Raw count mismatch: manifest={raw_manifest} inspected={state.raw_inspected_count}",
                tolerance=RAW_COUNT_TOLERANCE,
            )

    norm_manifest = manifest.normalized_row_count_manifest
    if norm_manifest > 0 and state.normalized_inspected_count > 0:
        diff = abs(state.normalized_inspected_count - norm_manifest) / norm_manifest
        if diff > NORMALIZED_COUNT_TOLERANCE:
            state.add_warning(
                IssueSeverity.DEGRADED if diff > NORMALIZED_COUNT_TOLERANCE * 2 else IssueSeverity.WARNING,
                "count_consistency",
                (
                    f"Normalized count mismatch: manifest={norm_manifest} "
                    f"inspected={state.normalized_inspected_count}"
                ),
                tolerance=NORMALIZED_COUNT_TOLERANCE,
            )

    if state.raw_inspected_count > 0 and state.normalized_inspected_count > 0:
        ratio = state.normalized_inspected_count / state.raw_inspected_count
        if ratio > NORMALIZED_TO_RAW_EXPANSION_MAX:
            state.add_warning(
                IssueSeverity.INFO,
                "count_consistency",
                f"High normalized-to-raw expansion ratio: {ratio:.2f}x (allMids expands to many rows)",
                ratio=round(ratio, 2),
            )

    if malformed_manifest or state.raw_malformed_count:
        total = max(state.raw_inspected_count + state.raw_malformed_count, 1)
        rate = (malformed_manifest + state.raw_malformed_count) / total
        if rate >= MALFORMED_RATE_UNUSABLE:
            state.add_warning(
                IssueSeverity.CRITICAL,
                "malformed",
                f"Malformed rate {rate:.2%} exceeds unusable threshold",
            )
        elif rate >= MALFORMED_RATE_DEGRADED:
            state.add_warning(
                IssueSeverity.DEGRADED,
                "malformed",
                f"Malformed rate {rate:.2%} exceeds degraded threshold",
            )

    if dup_manifest:
        state.add_warning(
            IssueSeverity.INFO,
            "duplicates",
            f"Manifest reports {dup_manifest} duplicate(s) excluded from normalized output",
        )

    if state.backward_exchange_count >= BACKWARD_TIMESTAMP_CRITICAL_COUNT:
        state.add_warning(
            IssueSeverity.CRITICAL,
            "timestamp_order",
            f"{state.backward_exchange_count} backward exchange timestamps detected",
        )
    elif state.backward_exchange_count >= BACKWARD_TIMESTAMP_DEGRADED_COUNT:
        state.add_warning(
            IssueSeverity.DEGRADED,
            "timestamp_order",
            f"{state.backward_exchange_count} backward exchange timestamps detected",
        )
    elif state.backward_exchange_count:
        state.add_warning(
            IssueSeverity.WARNING,
            "timestamp_order",
            f"{state.backward_exchange_count} backward exchange timestamp(s) (common in batched trades)",
        )

    reconnects = int(manifest_data.get("reconnect_count") or 0)
    if reconnects > MAX_RECONNECT_COUNT_USABLE:
        state.add_warning(
            IssueSeverity.DEGRADED,
            "reconnect",
            f"High reconnect count: {reconnects}",
        )
    elif reconnects > MAX_RECONNECT_COUNT_GOOD:
        state.add_warning(
            IssueSeverity.WARNING,
            "reconnect",
            f"Reconnect count {reconnects} above good threshold ({MAX_RECONNECT_COUNT_GOOD})",
        )

    for ts_key, dup_count in state.exchange_ts_duplicates.items():
        channel = ts_key.split(":")[0]
        ch_sum = state.channel(channel)
        ch_sum.duplicate_count += dup_count
        ch_events = max(ch_sum.raw_event_count, 1)
        if dup_count / ch_events > DUPLICATE_TIMESTAMP_RATE_WARNING:
            state.add_warning(
                IssueSeverity.INFO,
                "duplicate_timestamp",
                f"High duplicate exchange timestamp rate on {channel}",
                channel=channel,
            )


def _assign_channel_labels(state: InspectionState, manifest: SessionManifest) -> None:
    critical_channels = {
        w.channel for w in state.warnings if w.severity == IssueSeverity.CRITICAL.value and w.channel
    }
    degraded_channels = {
        w.channel for w in state.warnings if w.severity == IssueSeverity.DEGRADED.value and w.channel
    }
    warning_channels = {
        w.channel for w in state.warnings if w.severity == IssueSeverity.WARNING.value and w.channel
    }

    for ch_name, ch_sum in state.channel_summaries.items():
        if ch_name in critical_channels or ch_sum.raw_event_count == 0:
            if ch_name in ("trades", "bbo", "allMids") and ch_sum.raw_event_count == 0:
                ch_sum.quality_label = QualityLabel.UNUSABLE.value
            elif ch_sum.raw_event_count == 0:
                ch_sum.quality_label = QualityLabel.DEGRADED.value
            else:
                ch_sum.quality_label = QualityLabel.DEGRADED.value
        elif ch_name in degraded_channels or ch_sum.malformed_count > 0:
            ch_sum.quality_label = QualityLabel.DEGRADED.value
        elif ch_name in warning_channels or ch_sum.gap_count > 0 or ch_sum.stale_period_count > 0:
            ch_sum.quality_label = QualityLabel.USABLE_WITH_WARNINGS.value
        elif ch_sum.raw_event_count > 0:
            ch_sum.quality_label = QualityLabel.GOOD_FOR_RESEARCH.value
        else:
            ch_sum.quality_label = QualityLabel.UNUSABLE.value

    for coin_name, c_sum in state.coin_summaries.items():
        coin_warnings = [w for w in state.warnings if w.coin == coin_name]
        if c_sum.raw_event_count == 0:
            c_sum.quality_label = QualityLabel.UNUSABLE.value
        elif any(w.severity == IssueSeverity.CRITICAL.value for w in coin_warnings):
            c_sum.quality_label = QualityLabel.DEGRADED.value
        elif any(w.severity == IssueSeverity.DEGRADED.value for w in coin_warnings):
            c_sum.quality_label = QualityLabel.DEGRADED.value
        elif coin_warnings:
            c_sum.quality_label = QualityLabel.USABLE_WITH_WARNINGS.value
        else:
            c_sum.quality_label = QualityLabel.GOOD_FOR_RESEARCH.value


def assign_overall_label(
    state: InspectionState,
    manifest: SessionManifest,
) -> QualityLabel:
    """Assign overall session quality label from inspection state."""
    if not manifest.exists:
        return QualityLabel.UNUSABLE

    if state.raw_unreadable_files and not state.raw_inspected_count:
        return QualityLabel.UNUSABLE

    if not state.raw_inspected_count:
        return QualityLabel.UNUSABLE

    critical = [w for w in state.warnings if w.severity == IssueSeverity.CRITICAL.value]
    degraded = [w for w in state.warnings if w.severity == IssueSeverity.DEGRADED.value]
    warnings = [w for w in state.warnings if w.severity == IssueSeverity.WARNING.value]

    manifest_data = manifest.manifest_data or {}
    total_raw = max(state.raw_inspected_count, 1)
    malformed_rate = (state.raw_malformed_count + int(manifest_data.get("malformed_count") or 0)) / total_raw

    if malformed_rate >= MALFORMED_RATE_UNUSABLE:
        return QualityLabel.UNUSABLE
    if critical and any(w.category in ("missing_file", "malformed_raw", "unreadable_file") for w in critical):
        return QualityLabel.UNUSABLE
    if state.raw_inspected_count < MIN_RAW_EVENTS_RESEARCH and not critical:
        pass  # handled via warnings, not unusable

    missing_norm_frac = state.normalized_missing_files / max(len(manifest.normalized_files), 1)
    reconnects = int(manifest_data.get("reconnect_count") or 0)

    if critical or malformed_rate >= MALFORMED_RATE_DEGRADED:
        return QualityLabel.DEGRADED
    if missing_norm_frac >= MISSING_NORMALIZED_FRACTION_DEGRADED:
        return QualityLabel.DEGRADED
    if reconnects > MAX_RECONNECT_COUNT_USABLE:
        return QualityLabel.DEGRADED
    if state.backward_exchange_count >= BACKWARD_TIMESTAMP_DEGRADED_COUNT:
        return QualityLabel.DEGRADED
    if degraded:
        return QualityLabel.DEGRADED
    if warnings or state.gaps or reconnects > MAX_RECONNECT_COUNT_GOOD:
        return QualityLabel.USABLE_WITH_WARNINGS
    return QualityLabel.GOOD_FOR_RESEARCH


def build_usability_flags(
    state: InspectionState,
    manifest: SessionManifest,
    overall: QualityLabel,
) -> dict[str, dict[str, bool | str]]:
    """Build explicit usability flags with reasons."""
    manifest_data = manifest.manifest_data or {}
    duration_sec = 0.0
    if manifest.start_time_ms and manifest.end_time_ms:
        duration_sec = (manifest.end_time_ms - manifest.start_time_ms) / 1000

    trades_ok = state.channel_summaries.get("trades", ChannelSummary(channel="trades")).raw_event_count > 0
    bbo_ok = state.channel_summaries.get("bbo", ChannelSummary(channel="bbo")).raw_event_count > 0
    l2_ok = state.channel_summaries.get("l2Book", ChannelSummary(channel="l2Book")).raw_event_count > 0
    candle_ok = state.channel_summaries.get("candle", ChannelSummary(channel="candle")).raw_event_count > 0

    research_ok = overall in (QualityLabel.GOOD_FOR_RESEARCH, QualityLabel.USABLE_WITH_WARNINGS)
    research_reason = (
        "Session has readable raw market data suitable for exploratory research"
        if research_ok
        else f"Overall quality label is {overall.value}; address warnings before research use"
    )

    return {
        "usable_for_latency_research": {
            "usable": research_ok and any(
                state.latency(ch).count > 0 for ch in ("trades", "bbo", "l2Book")
            ),
            "reason": (
                "Trades, BBO, and L2Book provide exchange timestamps for latency analysis; "
                "exclude candle and activeAssetCtx from execution-latency summaries"
                if research_ok
                else "Insufficient quality or missing timestamped channels for latency research"
            ),
        },
        "usable_for_trade_replay": {
            "usable": trades_ok and overall != QualityLabel.UNUSABLE,
            "reason": (
                "Trade channel raw data preserved with IDs and hashes"
                if trades_ok
                else "No usable trade data in session"
            ),
        },
        "usable_for_bbo_replay": {
            "usable": bbo_ok and overall != QualityLabel.UNUSABLE,
            "reason": (
                "BBO channel has continuous quote updates"
                if bbo_ok
                else "No usable BBO data in session"
            ),
        },
        "usable_for_l2_replay": {
            "usable": l2_ok and overall != QualityLabel.UNUSABLE,
            "reason": (
                "L2 book snapshots available (lower frequency than BBO)"
                if l2_ok
                else "No usable L2 book data in session"
            ),
        },
        "usable_for_candle_replay": {
            "usable": candle_ok and overall != QualityLabel.UNUSABLE,
            "reason": (
                "Candle channel present; verify continuity warnings before bar replay"
                if candle_ok
                else "No usable candle data in session"
            ),
        },
        "usable_for_feature_engineering": {
            "usable": research_ok and state.raw_inspected_count >= MIN_RAW_EVENTS_RESEARCH,
            "reason": (
                "Multi-channel raw data available for offline feature extraction"
                if research_ok and state.raw_inspected_count >= MIN_RAW_EVENTS_RESEARCH
                else "Insufficient event volume or quality for feature engineering"
            ),
        },
        "usable_for_backtesting": {
            "usable": False,
            "reason": (
                "Backtesting replay engine is not implemented yet; data may be usable later "
                "after replay validation"
            ),
        },
    }


def build_recommendations(
    state: InspectionState,
    manifest: SessionManifest,
    overall: QualityLabel,
) -> list[str]:
    """Generate practical next-step recommendations."""
    recs: list[str] = []
    manifest_data = manifest.manifest_data or {}

    duration_sec = 0.0
    if manifest.start_time_ms and manifest.end_time_ms:
        duration_sec = (manifest.end_time_ms - manifest.start_time_ms) / 1000

    if duration_sec < MIN_SESSION_DURATION_RESEARCH_SEC:
        recs.append(
            f"Run a longer recorder session (current ~{duration_sec:.0f}s; "
            f"recommend >= {MIN_SESSION_DURATION_RESEARCH_SEC}s for strategy evaluation context)"
        )

    if state.normalized_missing_files:
        recs.append("Investigate missing normalized files; raw data remains the source of truth")

    if len(manifest.normalized_files) > 500:
        recs.append("Consider consolidating many small Parquet files for faster offline reads")

    if state.backward_exchange_count:
        recs.append(
            "Investigate backward exchange timestamps in trades channel "
            "(often benign for batched trade messages)"
        )

    recs.append("Exclude candle latency from execution-latency summaries")

    if state.channel_summaries.get("trades") and state.channel_summaries.get("bbo"):
        recs.append("Use trades and BBO for replay; treat activeAssetCtx as contextual data only")

    if overall == QualityLabel.GOOD_FOR_RESEARCH:
        recs.append("Session quality is good for research; proceed with offline analysis on raw JSONL")
    elif overall == QualityLabel.USABLE_WITH_WARNINGS:
        recs.append("Review warnings before relying on this session for research conclusions")
    elif overall == QualityLabel.DEGRADED:
        recs.append("Re-record session or exclude degraded channels from analysis")
    else:
        recs.append("Do not use this session for research; re-run the recorder")

    reconnects = int(manifest_data.get("reconnect_count") or 0)
    if reconnects:
        recs.append(f"Session had {reconnects} reconnect(s); check gap records around reconnect windows")

    recs.append(
        "Quality reports assess data integrity only — they do not prove strategy profitability"
    )
    return recs
