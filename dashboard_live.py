"""Read-only live market dashboard data loaders and aggregations.

Aggregates WebSocket recorder output into meeting-friendly summaries.
Paper-only — no wallets, signing, or exchange execution.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import DATA, MARKET, PAPER_TRADING, VERSION
from market_analyzer import PriceHistory, price_at_or_before, rolling_percent_change

import dashboard_data

DECISION_HOLD = "HOLD"
DECISION_PAPER_BUY = "PAPER_BUY_SIGNAL"
DECISION_PAPER_SELL = "PAPER_SELL_SIGNAL"
DECISION_BLOCKED = "BLOCKED"

RECOMMENDATION_TO_DECISION = {
    "PAPER BUY": DECISION_PAPER_BUY,
    "PAPER SELL": DECISION_PAPER_SELL,
    "HOLD": DECISION_HOLD,
    "BLOCKED": DECISION_BLOCKED,
}

DEFAULT_LOOKBACK_SECONDS = 120
DEFAULT_MAX_EVENTS = 25_000
DECISION_TIMELINE_LIMIT = 100
SPREAD_FILTER_BPS = 25.0

_PARQUET_AVAILABLE = False
try:
    import pyarrow.parquet as pq

    _PARQUET_AVAILABLE = True
except ImportError:
    pq = None  # type: ignore[assignment]


def decision_timeline_path() -> Path:
    return DATA.data_dir / "decision_timeline.jsonl"


def map_recommendation_to_decision(recommendation: str) -> str:
    return RECOMMENDATION_TO_DECISION.get(recommendation, DECISION_HOLD)


def format_last_message_time(last_message_at_ms: int | float | None) -> str:
    if last_message_at_ms is None:
        return "—"
    try:
        dt = datetime.fromtimestamp(int(last_message_at_ms) / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (TypeError, ValueError, OSError):
        return str(last_message_at_ms)


def load_live_dashboard_context() -> dict[str, Any]:
    """Resolve live vs historical recorder context without raising on missing files."""
    status = dashboard_data.load_recorder_status()
    manifests = sorted(
        dashboard_data.load_recorder_manifests(),
        key=lambda item: item.get("generated_at") or item.get("started_at") or "",
        reverse=True,
    )

    if status and status.get("running"):
        session_id = status.get("recorder_session_id")
        return {
            "data_mode": "live",
            "data_mode_label": "Live recorder session",
            "recorder_session_id": session_id,
            "status": status,
            "manifest": _manifest_for_session(manifests, session_id),
            "stale": status.get("state") == "STALE_STATUS",
        }

    if status and status.get("recorder_session_id"):
        session_id = status.get("recorder_session_id")
        return {
            "data_mode": "historical",
            "data_mode_label": "Historical / last recorder session (recorder not running)",
            "recorder_session_id": session_id,
            "status": status,
            "manifest": _manifest_for_session(manifests, session_id),
            "stale": status.get("state") == "STALE_STATUS",
        }

    if manifests:
        manifest = manifests[0]
        session_id = manifest.get("recorder_session_id")
        return {
            "data_mode": "historical",
            "data_mode_label": "Historical / replayed from latest recorder manifest",
            "recorder_session_id": session_id,
            "status": status,
            "manifest": manifest,
            "stale": False,
        }

    return {
        "data_mode": "missing",
        "data_mode_label": "No recorder data available",
        "recorder_session_id": None,
        "status": status,
        "manifest": None,
        "stale": False,
    }


def _manifest_for_session(manifests: list[dict], session_id: str | None) -> dict | None:
    if not session_id:
        return None
    for manifest in manifests:
        if manifest.get("recorder_session_id") == session_id:
            return manifest
    return None


def display_connection_status(context: dict[str, Any], status: dict[str, Any]) -> str:
    """Map recorder status to meeting-friendly connected / stale / disconnected."""
    stale = bool(context.get("stale") or status.get("state") == "STALE_STATUS")
    if stale:
        return "STALE"
    if status.get("running") and status.get("connection_state") == "CONNECTED":
        return "CONNECTED"
    return "DISCONNECTED"


def build_websocket_status_panel(context: dict[str, Any]) -> dict[str, Any]:
    """Build read-only WebSocket status fields for the dashboard."""
    status = context.get("status") or {}
    manifest = context.get("manifest") or {}

    latency = manifest.get("latency_statistics") or {}
    if not latency:
        latency = {
            "p50_ms": status.get("latency_p50_ms"),
            "p95_ms": status.get("latency_p95_ms"),
            "p99_ms": status.get("latency_p99_ms"),
        }

    stale = bool(context.get("stale") or status.get("state") == "STALE_STATUS")

    return {
        "connection_state": status.get("connection_state", "DISCONNECTED"),
        "display_connection_status": display_connection_status(context, status),
        "recorder_session_id": context.get("recorder_session_id") or "—",
        "heartbeat_age_seconds": status.get("heartbeat_age_seconds"),
        "uptime_seconds": status.get("uptime_seconds"),
        "reconnect_count": status.get("reconnect_count", manifest.get("reconnect_count", 0)),
        "active_subscriptions": status.get("active_subscriptions", 0),
        "requested_subscriptions": status.get("requested_subscriptions", "—"),
        "messages_per_second": status.get("messages_per_second", 0.0),
        "last_message_at_ms": status.get("last_message_at_ms"),
        "last_message_time": format_last_message_time(status.get("last_message_at_ms")),
        "raw_queue_size": status.get("raw_queue_size", 0),
        "normalized_queue_size": status.get("normalized_queue_size", 0),
        "latency_p50_ms": latency.get("p50_ms", status.get("latency_p50_ms")),
        "latency_p95_ms": latency.get("p95_ms", status.get("latency_p95_ms")),
        "latency_p99_ms": latency.get("p99_ms", status.get("latency_p99_ms")),
        "malformed_count": status.get("malformed_count", manifest.get("malformed_count", 0)),
        "duplicate_count": status.get("duplicate_count", manifest.get("duplicate_count", 0)),
        "gap_count": status.get("gap_count", manifest.get("gap_statistics", {}).get("total_gaps", 0)),
        "long_run": bool(status.get("long_run", manifest.get("long_run", False))),
        "running": bool(status.get("running", False)),
        "state": status.get("state", "STOPPED"),
        "stale": stale,
        "data_mode": context.get("data_mode", "missing"),
        "data_mode_label": context.get("data_mode_label", ""),
    }


def _recent_parquet_files(
    stream_dir: Path,
    event_type: str,
    coin: str,
    *,
    max_files: int = 8,
    max_dates: int = 2,
) -> list[Path]:
    """Return newest parquet files for one coin/event type without full-tree scans."""
    type_dir = stream_dir / f"event_type={event_type}"
    if not type_dir.exists():
        return []

    collected: list[Path] = []
    date_dirs = sorted(type_dir.glob("date=*"), reverse=True)
    for date_dir in date_dirs[:max_dates]:
        coin_dir = date_dir / f"coin={coin}"
        if not coin_dir.exists():
            continue
        files = sorted(coin_dir.glob("*.parquet"), key=lambda path: path.stat().st_mtime, reverse=True)
        collected.extend(files)
        if len(collected) >= max_files:
            break

    return sorted(collected, key=lambda path: path.stat().st_mtime, reverse=True)[:max_files]


def _parquet_files_for_range(
    stream_dir: Path,
    event_type: str,
    coin: str,
    *,
    start_ms: int,
    end_ms: int,
    max_files: int = 64,
) -> list[Path]:
    """Return parquet files likely overlapping a timestamp range (bounded scan)."""
    type_dir = stream_dir / f"event_type={event_type}"
    if not type_dir.exists():
        return []

    start_date = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).date()
    end_date = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).date()

    collected: list[Path] = []
    try:
        date_dirs = sorted(type_dir.glob("date=*"), reverse=True)
    except OSError:
        return []

    for date_dir in date_dirs:
        if len(collected) >= max_files:
            break
        parsed = _parse_partition_date_dir(date_dir.name)
        if parsed is not None and (parsed < start_date or parsed > end_date):
            continue
        coin_dir = date_dir / f"coin={coin}"
        if not coin_dir.exists():
            continue
        try:
            files = sorted(coin_dir.glob("*.parquet"), key=lambda path: path.stat().st_mtime, reverse=True)
        except OSError:
            continue
        collected.extend(files)
        if len(collected) >= max_files:
            break

    return sorted(collected, key=lambda path: path.stat().st_mtime, reverse=True)[:max_files]


def _parse_partition_date_dir(name: str):
    if not name.startswith("date="):
        return None
    text = name.split("=", 1)[1]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def _event_in_range(row: dict[str, Any], *, start_ms: int, end_ms: int) -> bool:
    ts = row.get("received_timestamp_ms") or row.get("exchange_timestamp_ms")
    if ts is None:
        ts = row.get("close_time_ms") or row.get("open_time_ms")
    if ts is None:
        return False
    ts_int = int(ts)
    return start_ms <= ts_int <= end_ms


def load_market_events_for_range(
    *,
    session_id: str | None,
    coins: tuple[str, ...] = MARKET.watchlist,
    start_ms: int,
    end_ms: int,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> list[dict[str, Any]]:
    """Load trade and BBO events from normalized parquet within a time range."""
    if not _PARQUET_AVAILABLE:
        return []

    stream_dir = dashboard_data.MARKET_STREAM_DIR / "normalized"
    if not stream_dir.exists():
        return []

    events: list[dict[str, Any]] = []
    for event_type in ("trade", "bbo"):
        for coin in coins:
            if len(events) >= max_events:
                break
            for file_path in _parquet_files_for_range(
                stream_dir,
                event_type,
                coin,
                start_ms=start_ms,
                end_ms=end_ms,
            ):
                if len(events) >= max_events:
                    break
                try:
                    table = pq.ParquetFile(file_path).read()
                except Exception:
                    continue
                rows = table.to_pylist()
                for row in reversed(rows):
                    if not _event_in_range(row, start_ms=start_ms, end_ms=end_ms):
                        continue
                    row["_event_type"] = event_type
                    events.append(row)
                    if len(events) >= max_events:
                        break

    events.sort(key=lambda row: int(row.get("received_timestamp_ms") or row.get("exchange_timestamp_ms") or 0))
    return events[-max_events:]


def load_candle_events_for_range(
    *,
    session_id: str | None,
    coins: tuple[str, ...] = MARKET.watchlist,
    start_ms: int,
    end_ms: int,
    max_events: int = 5_000,
) -> list[dict[str, Any]]:
    """Load candle events from normalized parquet within a time range."""
    if not _PARQUET_AVAILABLE:
        return []

    stream_dir = dashboard_data.MARKET_STREAM_DIR / "normalized"
    if not stream_dir.exists():
        return []

    events: list[dict[str, Any]] = []
    for coin in coins:
        if len(events) >= max_events:
            break
        for file_path in _parquet_files_for_range(
            stream_dir,
            "candle",
            coin,
            start_ms=start_ms,
            end_ms=end_ms,
        ):
            if len(events) >= max_events:
                break
            try:
                table = pq.ParquetFile(file_path).read()
            except Exception:
                continue
            rows = table.to_pylist()
            for row in reversed(rows):
                if not _event_in_range(row, start_ms=start_ms, end_ms=end_ms):
                    continue
                row["_event_type"] = "candle"
                events.append(row)
                if len(events) >= max_events:
                    break

    events.sort(key=lambda row: int(row.get("close_time_ms") or row.get("received_timestamp_ms") or 0))
    return events[-max_events:]


def load_archive_market_events_for_range(
    *,
    start_ms: int,
    end_ms: int,
    coins: tuple[str, ...] = MARKET.watchlist,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load market events from optional local archive parquet (read-only)."""
    if not _PARQUET_AVAILABLE:
        return [], []

    archive_dir = dashboard_data.MARKET_STREAM_DIR / "archive"
    if not archive_dir.exists():
        return [], []

    market_events: list[dict[str, Any]] = []
    candle_events: list[dict[str, Any]] = []
    coin_set = set(coins)

    try:
        parquet_files = sorted(
            archive_dir.rglob("*.parquet"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return [], []

    for file_path in parquet_files[:32]:
        if len(market_events) + len(candle_events) >= max_events:
            break
        try:
            table = pq.ParquetFile(file_path).read()
        except Exception:
            continue
        rows = table.to_pylist()
        for row in reversed(rows):
            if not _event_in_range(row, start_ms=start_ms, end_ms=end_ms):
                continue
            coin = row.get("coin")
            if coin is not None and coin not in coin_set:
                continue
            event_type = row.get("event_type") or row.get("_event_type")
            if event_type == "candle" or row.get("close") is not None:
                row["_event_type"] = "candle"
                candle_events.append(row)
            else:
                row["_event_type"] = event_type or "trade"
                market_events.append(row)
            if len(market_events) + len(candle_events) >= max_events:
                break

    market_events.sort(
        key=lambda row: int(row.get("received_timestamp_ms") or row.get("exchange_timestamp_ms") or 0)
    )
    candle_events.sort(key=lambda row: int(row.get("close_time_ms") or row.get("received_timestamp_ms") or 0))
    return market_events[-max_events:], candle_events[-max_events:]


DEFAULT_AUTO_REFRESH_INTERVAL_SECONDS = 1


def default_auto_refresh_enabled(websocket_status: dict[str, Any] | None) -> bool:
    """Default auto-refresh on when recorder is running and connected."""
    ws = websocket_status or {}
    display = str(ws.get("display_connection_status") or "").upper()
    if display in {"CONNECTED", "RUNNING"}:
        return True
    return bool(ws.get("running")) and ws.get("connection_state") == "CONNECTED"


def auto_refresh_status_message(
    *,
    enabled: bool,
    dependency_available: bool,
) -> str | None:
    if not enabled:
        return None
    if dependency_available:
        return None
    return (
        "Auto-refresh requires the optional `streamlit-autorefresh` package. "
        "Install with: pip install streamlit-autorefresh"
    )


def load_recent_market_events(
    *,
    session_id: str | None,
    coins: tuple[str, ...] = MARKET.watchlist,
    lookback_seconds: int = DEFAULT_LOOKBACK_SECONDS,
    max_events: int = DEFAULT_MAX_EVENTS,
    historical: bool = False,
) -> list[dict[str, Any]]:
    """Load recent trade and BBO events from normalized parquet (read-only)."""
    if not _PARQUET_AVAILABLE:
        return []

    stream_dir = dashboard_data.MARKET_STREAM_DIR / "normalized"
    if not stream_dir.exists():
        return []

    cutoff_ms = None if historical else int(time.time() * 1000) - (lookback_seconds * 1000)
    events: list[dict[str, Any]] = []

    for event_type in ("trade", "bbo"):
        for coin in coins:
            if len(events) >= max_events:
                break
            for file_path in _recent_parquet_files(stream_dir, event_type, coin):
                if len(events) >= max_events:
                    break
                try:
                    table = pq.ParquetFile(file_path).read()
                except Exception:
                    continue
                rows = table.to_pylist()
                for row in reversed(rows):
                    ts = row.get("received_timestamp_ms") or row.get("exchange_timestamp_ms")
                    if ts is None:
                        continue
                    if cutoff_ms is not None and int(ts) < cutoff_ms:
                        continue
                    row["_event_type"] = event_type
                    events.append(row)
                    if len(events) >= max_events:
                        break

    events.sort(key=lambda row: int(row.get("received_timestamp_ms") or row.get("exchange_timestamp_ms") or 0))
    return events[-max_events:]


def load_recent_candle_events(
    *,
    session_id: str | None,
    coins: tuple[str, ...] = MARKET.watchlist,
    lookback_seconds: int = DEFAULT_LOOKBACK_SECONDS,
    max_events: int = 5_000,
    historical: bool = False,
) -> list[dict[str, Any]]:
    """Load recent candle events from normalized parquet (read-only)."""
    if not _PARQUET_AVAILABLE:
        return []

    stream_dir = dashboard_data.MARKET_STREAM_DIR / "normalized"
    if not stream_dir.exists():
        return []

    cutoff_ms = None if historical else int(time.time() * 1000) - (lookback_seconds * 1000)
    events: list[dict[str, Any]] = []

    for coin in coins:
        if len(events) >= max_events:
            break
        for file_path in _recent_parquet_files(stream_dir, "candle", coin):
            if len(events) >= max_events:
                break
            try:
                table = pq.ParquetFile(file_path).read()
            except Exception:
                continue
            rows = table.to_pylist()
            for row in reversed(rows):
                ts = row.get("close_time_ms") or row.get("received_timestamp_ms")
                if ts is None:
                    continue
                if cutoff_ms is not None and int(ts) < cutoff_ms:
                    continue
                row["_event_type"] = "candle"
                events.append(row)
                if len(events) >= max_events:
                    break

    events.sort(key=lambda row: int(row.get("close_time_ms") or row.get("received_timestamp_ms") or 0))
    return events[-max_events:]


RAW_PREVIEW_CHANNELS = frozenset({"trades", "bbo", "candle", "allMids"})
RAW_PREVIEW_TAIL_BYTES = 512_000
RAW_PREVIEW_MAX_FILES = 8
RAW_PREVIEW_TAIL_LINES = 5_000


def _resolve_raw_preview_session_id(session_id: str | None) -> str | None:
    if session_id:
        return session_id
    from market_data.recovery import find_latest_session_id

    return find_latest_session_id(dashboard_data.MARKET_STREAM_DIR)


def _read_jsonl_tail(path: Path, *, max_lines: int, tail_bytes: int) -> list[str]:
    """Read the trailing portion of a JSONL file without loading the whole file."""
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size <= 0:
        return []

    read_size = min(size, tail_bytes)
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, size - read_size))
            chunk = handle.read(read_size)
    except OSError:
        return []

    text = chunk.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if size > read_size and lines:
        lines = lines[1:]
    return lines[-max_lines:]


def _recent_raw_jsonl_files(raw_dir: Path, *, max_files: int) -> list[Path]:
    if not raw_dir.exists():
        return []
    try:
        candidates = [
            path
            for path in raw_dir.rglob("*.jsonl")
            if path.is_file()
        ]
    except OSError:
        return []
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[:max_files]


def _preview_timestamp_label(timestamp_ms: int) -> str:
    try:
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (TypeError, ValueError, OSError):
        return str(timestamp_ms)


def _event_to_preview_row(event: Any, source_channel: str) -> dict[str, Any] | None:
    from market_data.event_models import BboEvent, CandleEvent, MidPriceEvent, TradeEvent

    ts_ms = int(getattr(event, "received_timestamp_ms", 0) or 0)
    row: dict[str, Any] = {
        "timestamp_ms": ts_ms,
        "timestamp": _preview_timestamp_label(ts_ms),
        "source_channel": source_channel,
    }

    if isinstance(event, TradeEvent):
        row.update(
            {
                "coin": event.coin,
                "price": event.price,
                "volume": event.size,
            }
        )
        return row

    if isinstance(event, BboEvent):
        midpoint = (event.best_bid_price + event.best_ask_price) / 2.0
        row.update(
            {
                "coin": event.coin,
                "price": midpoint,
                "best_bid": event.best_bid_price,
                "best_ask": event.best_ask_price,
                "spread": event.spread,
                "spread_bps": event.spread_bps,
            }
        )
        return row

    if isinstance(event, CandleEvent):
        row.update(
            {
                "coin": event.coin,
                "price": event.close,
                "open": event.open,
                "high": event.high,
                "low": event.low,
                "close": event.close,
                "volume": event.volume,
            }
        )
        return row

    if isinstance(event, MidPriceEvent):
        row.update(
            {
                "coin": event.coin,
                "price": event.mid_price,
            }
        )
        return row

    return None


def _event_to_chart_dict(event: Any, source_channel: str) -> dict[str, Any] | None:
    from dataclasses import asdict

    from market_data.event_models import BboEvent, CandleEvent, MidPriceEvent, TradeEvent

    if isinstance(event, TradeEvent):
        payload = asdict(event)
        payload["_event_type"] = "trade"
        payload["_source_channel"] = source_channel
        return payload

    if isinstance(event, BboEvent):
        payload = asdict(event)
        payload["_event_type"] = "bbo"
        payload["_source_channel"] = source_channel
        return payload

    if isinstance(event, CandleEvent):
        payload = asdict(event)
        payload["_event_type"] = "candle"
        payload["_source_channel"] = source_channel
        return payload

    if isinstance(event, MidPriceEvent):
        payload = asdict(event)
        payload["_event_type"] = "allMids"
        payload["mid_price"] = event.mid_price
        payload["_source_channel"] = source_channel
        return payload

    return None


def load_raw_preview_market_data(
    session_id: str | None = None,
    coins: tuple[str, ...] = MARKET.watchlist,
    max_events: int = 20_000,
) -> dict[str, Any]:
    """Build in-memory chart preview rows from recent raw JSONL (read-only, no writes)."""
    from market_data.normalizer import normalize_message

    resolved_session = _resolve_raw_preview_session_id(session_id)
    coin_set = set(coins)
    empty = {
        "available": False,
        "session_id": resolved_session,
        "rows": [],
        "market_events": [],
        "candle_events": [],
        "files_read": [],
        "channels_used": [],
        "parse_warnings": [],
        "row_count": 0,
    }
    if not resolved_session:
        return empty

    raw_dir = dashboard_data.MARKET_STREAM_DIR / "raw"
    files = _recent_raw_jsonl_files(raw_dir, max_files=RAW_PREVIEW_MAX_FILES)
    if not files:
        return empty

    rows: list[dict[str, Any]] = []
    market_events: list[dict[str, Any]] = []
    candle_events: list[dict[str, Any]] = []
    files_read: list[str] = []
    channels_used: set[str] = set()
    parse_warnings: list[str] = []
    events_collected = 0

    per_file_lines = max(200, max_events // max(1, len(files)))

    for file_path in files:
        if events_collected >= max_events:
            break

        tail_lines = _read_jsonl_tail(
            file_path,
            max_lines=min(RAW_PREVIEW_TAIL_LINES, per_file_lines),
            tail_bytes=RAW_PREVIEW_TAIL_BYTES,
        )
        if not tail_lines:
            continue

        file_had_session_rows = False
        for line in tail_lines:
            if events_collected >= max_events:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                parse_warnings.append(f"{file_path.name}: {exc}")
                continue

            if record.get("recorder_session_id") != resolved_session:
                continue

            channel = str(record.get("channel") or "")
            if channel not in RAW_PREVIEW_CHANNELS:
                continue

            payload = record.get("raw_payload") or {}
            data = payload.get("data", payload)
            received_ms = int(record.get("received_timestamp_ms") or 0)
            if received_ms <= 0:
                continue

            try:
                normalized = normalize_message(channel, data, received_ms)
            except Exception as exc:
                parse_warnings.append(f"{file_path.name} ({channel}): {exc}")
                continue

            for event in normalized:
                coin = getattr(event, "coin", None)
                if coin is not None and coin not in coin_set:
                    continue

                preview_row = _event_to_preview_row(event, channel)
                chart_row = _event_to_chart_dict(event, channel)
                if preview_row is None or chart_row is None:
                    continue

                rows.append(preview_row)
                channels_used.add(channel)
                file_had_session_rows = True
                events_collected += 1

                if chart_row.get("_event_type") == "candle":
                    candle_events.append(chart_row)
                else:
                    market_events.append(chart_row)

        if file_had_session_rows and str(file_path) not in files_read:
            files_read.append(str(file_path))

    rows.sort(key=lambda item: int(item.get("timestamp_ms") or 0))
    market_events.sort(
        key=lambda item: int(item.get("received_timestamp_ms") or item.get("exchange_timestamp_ms") or 0)
    )
    candle_events.sort(
        key=lambda item: int(item.get("close_time_ms") or item.get("received_timestamp_ms") or 0)
    )

    if len(rows) > max_events:
        rows = rows[-max_events:]
    if len(market_events) > max_events:
        market_events = market_events[-max_events:]
    if len(candle_events) > max_events:
        candle_events = candle_events[-max_events:]

    return {
        "available": bool(rows),
        "session_id": resolved_session,
        "rows": rows,
        "market_events": market_events,
        "candle_events": candle_events,
        "files_read": files_read,
        "channels_used": sorted(channels_used),
        "parse_warnings": parse_warnings,
        "row_count": len(rows),
    }


@dataclass
class _CoinLiveState:
    coin: str
    latest_price: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    best_bid_size: float | None = None
    best_ask_size: float | None = None
    spread: float | None = None
    spread_bps: float | None = None
    midpoint: float | None = None
    bbo_imbalance: float | None = None
    last_update_ms: int | None = None
    price_history: PriceHistory = field(default_factory=dict)
    trade_flow: list[dict[str, Any]] = field(default_factory=list)
    bbo_imbalance_history: list[tuple[float, float]] = field(default_factory=list)
    second_buckets: dict[int, dict[str, Any]] = field(default_factory=dict)


def _bbo_imbalance(bid_size: float | None, ask_size: float | None) -> float | None:
    if bid_size is None or ask_size is None:
        return None
    total = bid_size + ask_size
    if total <= 0:
        return None
    return (bid_size - ask_size) / total


def _trade_flow_window(history: list[dict[str, Any]], now_s: float, window: int) -> dict[str, Any]:
    cutoff = now_s - window
    recent = [item for item in history if item["timestamp_s"] >= cutoff]
    buy_count = sum(1 for item in recent if item["side"] == "buy")
    sell_count = sum(1 for item in recent if item["side"] == "sell")
    buy_vol = sum(item["size"] for item in recent if item["side"] == "buy")
    sell_vol = sum(item["size"] for item in recent if item["side"] == "sell")
    total = buy_vol + sell_vol
    imbalance = (buy_vol - sell_vol) / total if total else 0.0
    return {
        "buy_count": buy_count,
        "sell_count": sell_count,
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
        "total_volume": total,
        "imbalance": imbalance,
    }


def _imbalance_change(history: list[tuple[float, float]], now_s: float, window: int) -> float | None:
    if not history:
        return None
    current = history[-1][1]
    past = price_at_or_before(history, now_s - window)
    if past is None:
        return None
    return current - past[1]


def ingest_market_event(state: _CoinLiveState, event: dict[str, Any]) -> None:
    """Update per-coin live state from one normalized trade or BBO row."""
    event_type = event.get("_event_type")
    ts_ms = int(event.get("received_timestamp_ms") or event.get("exchange_timestamp_ms") or 0)
    now_s = ts_ms / 1000.0
    bucket = int(now_s)

    if event_type == "bbo":
        state.best_bid = _safe_float(event.get("best_bid_price"))
        state.best_ask = _safe_float(event.get("best_ask_price"))
        state.best_bid_size = _safe_float(event.get("best_bid_size"))
        state.best_ask_size = _safe_float(event.get("best_ask_size"))
        state.spread = _safe_float(event.get("spread"))
        state.spread_bps = _safe_float(event.get("spread_bps"))
        if state.best_bid is not None and state.best_ask is not None:
            state.midpoint = (state.best_bid + state.best_ask) / 2.0
        state.bbo_imbalance = _bbo_imbalance(state.best_bid_size, state.best_ask_size)
        if state.bbo_imbalance is not None:
            state.bbo_imbalance_history.append((now_s, state.bbo_imbalance))
            state.bbo_imbalance_history = state.bbo_imbalance_history[-200:]
        state.last_update_ms = ts_ms
        return

    if event_type == "trade":
        price = _safe_float(event.get("price"))
        size = _safe_float(event.get("size")) or 0.0
        side_raw = str(event.get("side", "")).upper()
        side = "buy" if side_raw in ("B", "BUY") else "sell"

        if price is not None:
            state.latest_price = price
            if state.coin not in state.price_history:
                state.price_history[state.coin] = []
            state.price_history[state.coin].append((now_s, price))
            state.price_history[state.coin] = state.price_history[state.coin][-500:]

        state.trade_flow.append({"timestamp_s": now_s, "side": side, "size": size, "price": price})
        state.trade_flow = state.trade_flow[-500:]
        state.last_update_ms = ts_ms

        bucket_row = state.second_buckets.setdefault(
            bucket,
            {"trades": 0, "volume": 0.0},
        )
        bucket_row["trades"] += 1
        bucket_row["volume"] += size


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_per_second_market_rows(
    events: list[dict[str, Any]],
    coins: tuple[str, ...] = MARKET.watchlist,
    *,
    reference_time_s: float | None = None,
) -> list[dict[str, Any]]:
    """Aggregate event-level WebSocket data into latest 1-second summaries per coin."""
    states = {coin: _CoinLiveState(coin=coin) for coin in coins}

    for event in events:
        coin = event.get("coin")
        if coin not in states:
            continue
        ingest_market_event(states[coin], event)

    rows: list[dict[str, Any]] = []
    if reference_time_s is None and events:
        reference_time_s = max(
            int(event.get("received_timestamp_ms") or event.get("exchange_timestamp_ms") or 0) / 1000.0
            for event in events
        )
    now_s = reference_time_s if reference_time_s is not None else time.time()

    for coin in coins:
        state = states[coin]
        current_bucket = int(now_s)
        bucket_stats = state.second_buckets.get(current_bucket, {"trades": 0, "volume": 0.0})
        history = state.price_history.get(coin, [])
        price = state.latest_price
        flow_10 = _trade_flow_window(state.trade_flow, now_s, 10)

        rows.append(
            {
                "coin": coin,
                "latest_price": price,
                "best_bid": state.best_bid,
                "best_ask": state.best_ask,
                "spread": state.spread,
                "spread_bps": state.spread_bps,
                "trades_per_second": bucket_stats["trades"],
                "volume_per_second": round(bucket_stats["volume"], 6),
                "latest_midpoint": state.midpoint,
                "bbo_imbalance": state.bbo_imbalance,
                "trade_imbalance_10s": flow_10["imbalance"],
                "return_5s": rolling_percent_change(history, price, now_s, 5),
                "return_10s": rolling_percent_change(history, price, now_s, 10),
                "return_30s": rolling_percent_change(history, price, now_s, 30),
                "last_update_time": format_last_message_time(state.last_update_ms),
            }
        )

    return rows


def build_feature_signal_rows(per_second_rows: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build research feature indicators from aggregated live market state."""
    states = {coin: _CoinLiveState(coin=coin) for coin in MARKET.watchlist}
    for event in events:
        coin = event.get("coin")
        if coin in states:
            ingest_market_event(states[coin], event)

    feature_rows: list[dict[str, Any]] = []
    per_coin = {row["coin"]: row for row in per_second_rows}
    now_s = time.time()

    for coin in MARKET.watchlist:
        state = states[coin]
        summary = per_coin.get(coin, {})
        flow_10 = _trade_flow_window(state.trade_flow, now_s, 10)
        flow_30 = _trade_flow_window(state.trade_flow, now_s, 30)

        feature_rows.append(
            {
                "coin": coin,
                "bbo_imbalance": summary.get("bbo_imbalance"),
                "bbo_imbalance_change_10s": _imbalance_change(state.bbo_imbalance_history, now_s, 10),
                "trade_imbalance_10s": flow_10["imbalance"],
                "buy_trade_count_10s": flow_10["buy_count"],
                "sell_trade_count_10s": flow_10["sell_count"],
                "volume_burst_30s": flow_30["total_volume"],
                "return_5s": summary.get("return_5s"),
                "return_10s": summary.get("return_10s"),
                "return_30s": summary.get("return_30s"),
                "spread_bps": summary.get("spread_bps"),
                "research_only": True,
            }
        )

    return feature_rows


def _filter_status(label: str, blocked: bool) -> str:
    return "BLOCKED" if blocked else label


def risk_cooldown_status(risk_status: str) -> str:
    """Combine risk, cooldown, and session-limit checks for the decision panel."""
    if not risk_status:
        return "OK"
    lower = risk_status.lower()
    if "cooldown" in lower:
        return f"COOLDOWN — {risk_status}"
    if "trade limit" in lower or "session trade" in lower:
        return f"LIMIT — {risk_status}"
    return risk_status


def build_live_bot_decisions(
    recommendations: dict | None,
    per_second_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map recommendations.json decisions to paper-only live bot panel rows."""
    if recommendations is None:
        return []

    per_coin_market = {row["coin"]: row for row in per_second_rows}
    paper = recommendations.get("paper_trader") or {}
    profile_name = paper.get("profile_id") or recommendations.get("strategy_profile") or "default"
    strategy_name = profile_name or "contrarian_v1"
    strategy_version = VERSION

    rows: list[dict[str, Any]] = []
    for decision in recommendations.get("decisions") or []:
        coin = decision.get("coin")
        market = per_coin_market.get(coin, {})
        spread_bps = market.get("spread_bps")
        spread_blocked = spread_bps is not None and spread_bps > SPREAD_FILTER_BPS

        risk_status = decision.get("risk_status") or ""
        recommendation = decision.get("recommendation", "HOLD")
        mapped = map_recommendation_to_decision(recommendation)

        rows.append(
            {
                "coin": coin,
                "decision": mapped,
                "reason": decision.get("primary_reason", "—"),
                "signal_strength": decision.get("signal_strength", "—"),
                "strategy_name": strategy_name,
                "strategy_version": strategy_version,
                "profile_name": profile_name,
                "spread_filter_status": _filter_status(
                    "OK" if spread_bps is None else f"{spread_bps:.2f} bps",
                    spread_blocked and mapped in (DECISION_PAPER_BUY, DECISION_PAPER_SELL),
                ),
                "bbo_imbalance_status": (
                    f"{market.get('bbo_imbalance'):+.3f}"
                    if market.get("bbo_imbalance") is not None
                    else "—"
                ),
                "risk_cooldown_status": risk_cooldown_status(risk_status),
                "risk_filter_status": risk_status,
                "cooldown_status": _filter_status(
                    "OK",
                    "cooldown" in risk_status.lower(),
                ),
                "trade_limit_status": _filter_status(
                    "OK",
                    "trade limit" in risk_status.lower() or "session trade" in risk_status.lower(),
                ),
                "paper_only": True,
                "execution_enabled": False,
            }
        )

    return rows


def action_taken_for_decision(decision: str, recommendation: str) -> str:
    if decision == DECISION_BLOCKED:
        return "BLOCKED"
    if decision in (DECISION_PAPER_BUY, DECISION_PAPER_SELL):
        return "SIGNAL_ONLY"
    return "NONE"


def append_decision_timeline_from_snapshot(
    snapshot_dict: dict[str, Any],
    *,
    per_second_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Append paper-only decision rows to the local timeline (read by dashboard)."""
    DATA.data_dir.mkdir(parents=True, exist_ok=True)
    path = decision_timeline_path()
    per_coin_market = {row["coin"]: row for row in (per_second_rows or [])}
    timestamp = snapshot_dict.get("timestamp") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    entries: list[dict[str, Any]] = []
    for decision in snapshot_dict.get("decisions") or []:
        coin = decision.get("coin")
        market = per_coin_market.get(coin, {})
        recommendation = decision.get("recommendation", "HOLD")
        mapped = map_recommendation_to_decision(recommendation)
        entries.append(
            {
                "timestamp": timestamp,
                "coin": coin,
                "decision": mapped,
                "reason": decision.get("primary_reason"),
                "price": decision.get("current_price"),
                "best_bid": market.get("best_bid"),
                "best_ask": market.get("best_ask"),
                "spread_bps": market.get("spread_bps"),
                "bbo_imbalance": market.get("bbo_imbalance"),
                "trade_imbalance": market.get("trade_imbalance_10s"),
                "action_taken": action_taken_for_decision(mapped, recommendation),
                "paper_only": True,
            }
        )

    if not entries:
        return

    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, default=str) + "\n")

    _trim_decision_timeline(path, DECISION_TIMELINE_LIMIT * 20)


def _trim_decision_timeline(path: Path, max_lines: int) -> None:
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    if len(lines) <= max_lines:
        return
    trimmed = lines[-max_lines:]
    path.write_text("\n".join(trimmed) + "\n", encoding="utf-8")


def load_decision_timeline(limit: int = DECISION_TIMELINE_LIMIT) -> list[dict[str, Any]]:
    """Load recent paper-only decision timeline rows."""
    path = decision_timeline_path()
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    return list(reversed(rows[-limit:]))


def build_paper_portfolio_risk(
    recommendations: dict | None,
    equity_log: Any | None = None,
) -> dict[str, Any]:
    """Summarize paper portfolio and risk for the live dashboard."""
    paper = (recommendations or {}).get("paper_trader") or {}
    summary = (recommendations or {}).get("decision_summary") or {}
    decisions = (recommendations or {}).get("decisions") or []

    blocked = [item for item in decisions if item.get("recommendation") == "BLOCKED"]
    blocked_reasons = sorted(
        {item.get("primary_reason") for item in blocked if item.get("primary_reason")}
    )

    max_drawdown_pct: float | None = None
    total_slippage = 0.0
    if equity_log is not None and not getattr(equity_log, "empty", True):
        drawdown = dashboard_data.compute_drawdown_series(equity_log)
        if not drawdown.empty:
            max_drawdown_pct = round(float(drawdown["drawdown_pct"].max()), 2)
        trades = dashboard_data.load_paper_trades()
        if trades is not None and not trades.empty and "slippage" in trades.columns:
            total_slippage = round(float(trades["slippage"].sum()), 2)

    readiness_label = "RESEARCH_ONLY"
    if paper.get("paused"):
        readiness_label = "PAUSED"
    elif summary.get("session_trades", 0) >= paper.get(
        "max_session_trades", PAPER_TRADING.max_successful_trades_per_session
    ):
        readiness_label = "SESSION_LIMIT"

    return {
        "paper_equity": paper.get("total_equity"),
        "open_paper_positions": summary.get("open_positions", 0),
        "exposure": paper.get("open_exposure"),
        "fees": paper.get("fees_paid"),
        "slippage": total_slippage,
        "max_drawdown_pct": max_drawdown_pct or summary.get("current_drawdown_pct"),
        "session_trade_count": paper.get("session_trades", summary.get("session_trades", 0)),
        "blocked_signals": len(blocked),
        "blocked_reasons": blocked_reasons,
        "capital_readiness_label": readiness_label,
        "paper_only": True,
        "paused": paper.get("paused", False),
    }


def _reference_time_from_events(events: list[dict[str, Any]], *, historical: bool) -> float | None:
    if not historical or not events:
        return None
    return max(
        int(event.get("received_timestamp_ms") or event.get("exchange_timestamp_ms") or 0) / 1000.0
        for event in events
    )


def load_live_market_dashboard_bundle() -> dict[str, Any]:
    """Load all live dashboard panels in one read-only bundle."""
    try:
        return _load_live_market_dashboard_bundle_impl()
    except Exception as exc:
        context = load_live_dashboard_context()
        fallback = _empty_live_market_dashboard_bundle(context)
        fallback["load_error"] = str(exc)
        return fallback


def _empty_live_market_dashboard_bundle(context: dict[str, Any]) -> dict[str, Any]:
    """Return a safe empty bundle when live dashboard loading fails."""
    per_second_rows = build_per_second_market_rows([], coins=MARKET.watchlist)
    recommendations = dashboard_data.load_recommendations()
    equity = dashboard_data.load_equity_log()
    return {
        "context": context,
        "websocket_status": build_websocket_status_panel(context),
        "per_second_market": per_second_rows,
        "live_bot_decisions": build_live_bot_decisions(recommendations, per_second_rows),
        "decision_timeline": load_decision_timeline(),
        "feature_signals": build_feature_signal_rows(per_second_rows, []),
        "paper_portfolio": build_paper_portfolio_risk(recommendations, equity),
        "recommendations": recommendations,
    }


def _load_live_market_dashboard_bundle_impl() -> dict[str, Any]:
    """Load all live dashboard panels in one read-only bundle."""
    context = load_live_dashboard_context()
    session_id = context.get("recorder_session_id")
    historical = context.get("data_mode") == "historical"
    events = load_recent_market_events(session_id=session_id, historical=historical)
    reference_time_s = _reference_time_from_events(events, historical=historical)
    per_second_rows = build_per_second_market_rows(events, reference_time_s=reference_time_s)
    recommendations = dashboard_data.load_recommendations()
    equity = dashboard_data.load_equity_log()

    return {
        "context": context,
        "websocket_status": build_websocket_status_panel(context),
        "per_second_market": per_second_rows,
        "live_bot_decisions": build_live_bot_decisions(recommendations, per_second_rows),
        "decision_timeline": load_decision_timeline(),
        "feature_signals": build_feature_signal_rows(per_second_rows, events),
        "paper_portfolio": build_paper_portfolio_risk(recommendations, equity),
        "recommendations": recommendations,
    }


def has_normalized_market_data(
    session_id: str | None,
    *,
    historical: bool = False,
) -> bool:
    """Return True when normalized parquet market rows are available."""
    events = load_recent_market_events(
        session_id=session_id,
        historical=historical,
        lookback_seconds=30,
        max_events=1,
    )
    return bool(events)


def latest_market_data_path(context: dict[str, Any]) -> str | None:
    """Best-effort path hint for the active or latest recorder session."""
    session_id = context.get("recorder_session_id")
    if not session_id:
        manifests = sorted(
            dashboard_data.load_recorder_manifests(),
            key=lambda item: item.get("generated_at") or item.get("started_at") or "",
            reverse=True,
        )
        if manifests:
            session_id = manifests[0].get("recorder_session_id")
    if not session_id:
        return None
    manifest_path = dashboard_data.MARKET_STREAM_DIR / "manifests" / f"{session_id}.json"
    if manifest_path.exists():
        return str(manifest_path)
    normalized_dir = dashboard_data.MARKET_STREAM_DIR / "normalized"
    if normalized_dir.exists():
        return str(normalized_dir)
    return str(dashboard_data.MARKET_STREAM_DIR)


DATA_QUALITY_EXPLANATION = (
    "Raw preview charts are for visual monitoring. Offline normalization is required "
    "before research-grade replay, feature engineering, or strategy evidence."
)

DATA_QUALITY_LABELS = {
    "raw_preview": "Raw preview: good for live visualization",
    "raw_session": "Raw session: good for preservation/recovery",
    "normalized": "Normalized: needed for research tables/replay/features",
    "replay_validated": "Replay validated: needed for backtests",
    "feature_dataset": "Feature dataset: needed for feature research",
}


def default_live_dashboard_debug_snapshot() -> dict[str, Any]:
    """Return a schema-safe debug snapshot with all expected keys present."""
    return {
        "bundle_loaded": False,
        "websocket_state": "—",
        "market_rows_count": 0,
        "candle_rows_count": 0,
        "decision_timeline_rows_count": 0,
        "historical_fallback": False,
        "latest_data_path": None,
        "load_error": None,
        "normalized_data_available": False,
        "raw_preview_available": False,
        "raw_preview_rows": 0,
        "raw_preview_session_id": None,
        "raw_preview_files_read": [],
        "raw_preview_channels_used": [],
        "raw_preview_warnings": [],
        "chart_data_source": "none",
        "chart_empty_reason": None,
        "fallback_used": False,
        "archive_available": False,
        "review_mode_loaded": None,
        "latest_session_id": None,
        "websocket_status": None,
        "auto_refresh_enabled": None,
        "refresh_interval_seconds": None,
        "selected_chart_range": None,
        "earliest_local_timestamp": None,
        "latest_local_timestamp": None,
        "requested_range_start": None,
        "requested_range_end": None,
        "local_coverage_percent": None,
        "chart_rows_loaded": None,
        "chart_rows_displayed": None,
        "normalized_rows": None,
        "archive_rows": None,
        "missing_history_warning": None,
        "normalization_queue_count": 0,
        "newest_unprocessed_session_id": None,
        "normalization_sessions_scanned": 0,
        "normalization_queue_warnings": [],
        "errors": [],
    }


def live_dashboard_debug_snapshot(
    bundle: dict[str, Any] | None,
    *,
    chart_data: dict[str, Any] | None = None,
    bundle_loaded: bool = False,
    auto_refresh_enabled: bool | None = None,
    refresh_interval_seconds: int | None = None,
) -> dict[str, Any]:
    """Build read-only debug fields for the live dashboard expander."""
    snapshot = default_live_dashboard_debug_snapshot()
    context = (bundle or {}).get("context") or {}
    ws = (bundle or {}).get("websocket_status") or {}
    timeline = (bundle or {}).get("decision_timeline") or []
    chart_data = chart_data or {}
    market_rows_count = chart_data.get("market_event_count", 0)
    candle_rows_count = chart_data.get("candle_event_count", 0)
    snapshot.update(
        {
            "bundle_loaded": bundle_loaded and bundle is not None,
            "websocket_state": ws.get("display_connection_status")
            or ws.get("connection_state")
            or ws.get("state")
            or "—",
            "market_rows_count": market_rows_count,
            "candle_rows_count": candle_rows_count,
            "decision_timeline_rows_count": len(timeline),
            "historical_fallback": context.get("data_mode") == "historical",
            "latest_data_path": latest_market_data_path(context),
            "load_error": (bundle or {}).get("load_error"),
            "normalized_data_available": chart_data.get("normalized_data_available", False),
            "raw_preview_available": chart_data.get("raw_preview_available", False),
            "raw_preview_rows": chart_data.get("raw_preview_rows", 0),
            "raw_preview_session_id": chart_data.get("raw_preview_session_id"),
            "raw_preview_files_read": chart_data.get("raw_preview_files_read", []),
            "raw_preview_channels_used": chart_data.get("raw_preview_channels_used", []),
            "raw_preview_warnings": chart_data.get("raw_preview_warnings", []),
            "chart_data_source": chart_data.get("chart_data_source", "none"),
            "chart_empty_reason": chart_data.get("chart_empty_reason"),
            "fallback_used": chart_data.get("fallback_used", False),
            "archive_available": chart_data.get("archive_available", False),
            "auto_refresh_enabled": auto_refresh_enabled,
            "refresh_interval_seconds": refresh_interval_seconds,
            "selected_chart_range": chart_data.get("selected_chart_range"),
            "earliest_local_timestamp": chart_data.get("earliest_local_timestamp_ms"),
            "latest_local_timestamp": chart_data.get("latest_local_timestamp_ms"),
            "requested_range_start": chart_data.get("requested_range_start_ms"),
            "requested_range_end": chart_data.get("requested_range_end_ms"),
            "local_coverage_percent": chart_data.get("local_coverage_percent"),
            "chart_rows_loaded": chart_data.get("chart_rows_loaded"),
            "chart_rows_displayed": chart_data.get("chart_rows_displayed"),
            "normalized_rows": chart_data.get("normalized_rows"),
            "archive_rows": chart_data.get("archive_rows"),
            "missing_history_warning": chart_data.get("missing_history_warning"),
        }
    )
    try:
        queue = build_normalization_queue(show_processed=False)
        snapshot.update(
            {
                "normalization_queue_count": queue.get("normalization_queue_count", 0),
                "newest_unprocessed_session_id": queue.get("newest_unprocessed_session_id"),
                "normalization_sessions_scanned": queue.get("normalization_sessions_scanned", 0),
                "normalization_queue_warnings": queue.get("normalization_queue_warnings") or [],
            }
        )
    except Exception as exc:
        warnings = list(snapshot.get("normalization_queue_warnings") or [])
        warnings.append(str(exc))
        snapshot["normalization_queue_warnings"] = warnings
    return snapshot


def resolve_latest_recorder_session_id(
    *,
    context: dict[str, Any] | None = None,
    archive_index: dict[str, Any] | None = None,
) -> str | None:
    """Resolve the best session id for read-only chart/reliability panels."""
    context = context or {}
    session_id = context.get("recorder_session_id")
    if session_id:
        return str(session_id)

    manifest = context.get("manifest") or {}
    manifest_session = manifest.get("recorder_session_id")
    if manifest_session:
        return str(manifest_session)

    archive_index = archive_index or {}
    sessions = archive_index.get("sessions") or []
    if sessions:
        return str(sessions[0])

    return _resolve_raw_preview_session_id(None)


def resolve_latest_recorder_manifest(
    *,
    context: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    """Return the manifest for the latest resolved recorder session."""
    context = context or {}
    manifest = context.get("manifest")
    if isinstance(manifest, dict) and manifest.get("recorder_session_id"):
        if session_id is None or manifest.get("recorder_session_id") == session_id:
            return manifest

    resolved = session_id or resolve_latest_recorder_session_id(context=context)
    if not resolved:
        return None

    manifests = sorted(
        dashboard_data.load_recorder_manifests(),
        key=lambda item: item.get("generated_at") or item.get("started_at") or "",
        reverse=True,
    )
    return _manifest_for_session(manifests, resolved)


def format_coin_chart_empty_message(coin: str, chart_data: dict[str, Any]) -> str:
    """Build a specific empty-chart message for one coin."""
    reason = chart_data.get("chart_empty_reason") or "No local chart rows matched the selected range."
    earliest_ms = chart_data.get("earliest_local_timestamp_ms")
    latest_ms = chart_data.get("latest_local_timestamp_ms")
    range_key = chart_data.get("selected_chart_range") or "selected range"
    fallback_hint = (
        " Try Live 5m, All local, or enable latest-available fallback."
        if not chart_data.get("fallback_used")
        else ""
    )

    coverage = ""
    if earliest_ms is not None and latest_ms is not None:
        coverage = (
            f" Latest local data exists from "
            f"{format_last_message_time(earliest_ms)} to {format_last_message_time(latest_ms)}."
        )

    return (
        f"No {coin} chart rows found for {range_key}. "
        f"{reason}{coverage}{fallback_hint}"
    )


def build_recorder_reliability_summary(
    ws: dict[str, Any] | None,
    *,
    context: dict[str, Any] | None = None,
    checkpoint: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    quality_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compact recorder reliability fields for the dashboard panel."""
    ws = ws or {}
    context = context or {}
    manifest = manifest or context.get("manifest") or {}
    session_id = (
        ws.get("recorder_session_id")
        or context.get("recorder_session_id")
        or manifest.get("recorder_session_id")
    )
    if session_id in (None, "—"):
        session_id = resolve_latest_recorder_session_id(context=context)

    status_state = ws.get("state") or ws.get("connection_state") or "STOPPED"
    manifest_state = manifest.get("state") or manifest.get("lifecycle_state")
    if manifest.get("manifest_finalized"):
        manifest_state = manifest_state or "FINALIZED"

    raw_events = ws.get("raw_event_count", ws.get("raw_events"))
    if raw_events is None:
        raw_events = manifest.get("raw_row_count")
    if raw_events is None and checkpoint:
        raw_events = checkpoint.get("raw_events_written")

    dropped_raw = ws.get("dropped_raw_count", ws.get("raw_dropped_count", 0))
    if not dropped_raw:
        dropped_raw = manifest.get("dropped_raw_count", 0)

    reconnects = ws.get("reconnect_count", manifest.get("reconnect_count", 0))
    gaps = ws.get("gap_count", manifest.get("gap_statistics", {}).get("total_gaps", 0))
    degraded = ws.get("degraded", manifest.get("degraded", False))
    shutdown_reason = ws.get("shutdown_reason") or manifest.get("shutdown_reason")

    consistency_warning = ws.get("consistency_warning")
    if manifest.get("manifest_finalized") and status_state == "STOPPED":
        manifest_label = "finalized"
    elif ws.get("running"):
        manifest_label = "live"
    elif consistency_warning:
        manifest_label = str(consistency_warning)
    elif manifest.get("manifest_finalized") is True:
        manifest_label = "finalized"
    elif manifest.get("manifest_finalized") is False:
        manifest_label = "open"
    elif manifest:
        manifest_label = "manifest_on_disk"
    else:
        manifest_label = "none"

    quality_label = None
    if quality_report:
        quality_label = quality_report.get("quality_label")
    elif session_id:
        report = dashboard_data.load_quality_report(str(session_id))
        if report:
            quality_label = report.get("quality_label")

    state_mismatch_warning = None
    if manifest_state and status_state and str(manifest_state).upper() != str(status_state).upper():
        if not (manifest.get("manifest_finalized") and status_state == "STOPPED"):
            state_mismatch_warning = (
                f"Status state is {status_state} but latest manifest state is {manifest_state}."
            )

    checkpoint_age = None
    if checkpoint:
        generated_at = checkpoint.get("generated_at")
        if generated_at:
            checkpoint_age = generated_at

    return {
        "session_id": session_id,
        "state": status_state,
        "manifest_state": manifest_state,
        "raw_events": raw_events if raw_events is not None else "—",
        "dropped_raw": dropped_raw,
        "reconnects": reconnects,
        "gaps": gaps,
        "degraded": degraded,
        "shutdown_reason": shutdown_reason or "—",
        "manifest_consistency": manifest_label,
        "quality_label": quality_label or "—",
        "state_mismatch_warning": state_mismatch_warning,
        "checkpoint_age": checkpoint_age,
    }


def build_data_coverage_rows(
    archive_index: dict[str, Any],
    *,
    coins: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Per-coin data coverage table rows (read-only, schema-safe)."""
    coins = coins or MARKET.watchlist
    normalized = archive_index.get("normalized") or {}
    raw = archive_index.get("raw_preview") or {}
    norm_coins = set(normalized.get("coins") or [])
    event_types = set(normalized.get("event_types") or [])
    raw_channels = set(raw.get("channels") or [])

    earliest_ms = archive_index.get("earliest_timestamp_ms")
    latest_ms = archive_index.get("latest_timestamp_ms")
    raw_available = bool(raw.get("available"))
    normalized_available = bool(normalized.get("available"))

    rows: list[dict[str, Any]] = []
    for coin in coins:
        note_parts: list[str] = []
        if coin not in norm_coins and normalized_available:
            note_parts.append("no normalized partitions")
        if not raw_available and not normalized_available:
            note_parts.append("no local data")

        rows.append(
            {
                "coin": coin,
                "earliest_local_timestamp": earliest_ms,
                "latest_local_timestamp": latest_ms,
                "trades_available": "trades" in event_types or "trade" in raw_channels,
                "bbo_available": "bbo" in event_types or "bbo" in raw_channels,
                "candles_available": "candle" in event_types or "candle" in raw_channels,
                "normalized_available": coin in norm_coins,
                "raw_available": raw_available,
                "local_coverage_note": "; ".join(note_parts) if note_parts else "—",
            }
        )
    return rows


DEFAULT_NORMALIZATION_QUEUE_CHANNELS = ("trades", "bbo")
DEFAULT_NORMALIZATION_QUEUE_VISIBLE = 5


def _iter_session_manifests(stream_dir: Path) -> list[dict[str, Any]]:
    """Load recorder manifests, skipping checkpoint sidecar files."""
    manifests_dir = stream_dir / "manifests"
    if not manifests_dir.exists():
        return []
    manifests: list[dict[str, Any]] = []
    for path in manifests_dir.glob("*.json"):
        if path.name.endswith(".checkpoint.json"):
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("recorder_session_id"):
            manifests.append(payload)
    return manifests


def _session_manifest_sort_key(manifest: dict[str, Any]) -> tuple[int, str]:
    start_ms = manifest.get("start_time_ms")
    try:
        start_key = int(start_ms) if start_ms is not None else 0
    except (TypeError, ValueError):
        start_key = 0
    generated = str(manifest.get("generated_at") or manifest.get("started_at") or "")
    return (start_key, generated)


def _format_duration_seconds(duration_seconds: float | None) -> str:
    if duration_seconds is None:
        return "—"
    total = max(0, int(duration_seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _normalized_exists_for_session(
    session_id: str,
    manifest: dict[str, Any],
    stream_dir: Path,
) -> bool:
    report_path = stream_dir / "normalization_reports" / f"{session_id}.json"
    if report_path.exists() and report_path.stat().st_size > 0:
        return True
    normalized_rows = int(manifest.get("normalized_row_count") or 0)
    if manifest.get("raw_only_session"):
        return False
    if not manifest.get("live_normalization", True):
        return False
    if normalized_rows > 0:
        return True
    normalized_files = [
        path
        for path in (manifest.get("files_written") or [])
        if "normalized" in str(path) and str(path).endswith(".parquet")
    ]
    return bool(normalized_files)


def _recommended_normalization_channels(
    manifest: dict[str, Any],
    channel_counts: dict[str, int] | None = None,
) -> tuple[str, ...]:
    counts = channel_counts or {}
    if not counts:
        subscriptions = manifest.get("subscriptions") or manifest.get("active_subscriptions") or []
        if isinstance(subscriptions, (list, tuple)):
            for item in subscriptions:
                if isinstance(item, dict):
                    channel = item.get("type") or item.get("channel")
                    if channel:
                        counts[str(channel)] = counts.get(str(channel), 0) + 1
    preferred = [channel for channel in DEFAULT_NORMALIZATION_QUEUE_CHANNELS if counts.get(channel)]
    if preferred:
        return tuple(preferred)
    return DEFAULT_NORMALIZATION_QUEUE_CHANNELS


def build_normalization_recommended_command(
    session_id: str,
    *,
    coins: tuple[str, ...] | None = None,
    channels: tuple[str, ...] | None = None,
    resume: bool = False,
) -> str:
    """Display-only offline normalization command (never executed by the dashboard)."""
    coin_list = coins or MARKET.watchlist
    channel_list = channels or DEFAULT_NORMALIZATION_QUEUE_CHANNELS
    command = (
        f"python3 market_data_normalize.py --session-id {session_id} "
        f"--coins {' '.join(coin_list)} "
        f"--channels {' '.join(channel_list)} "
        f"--skip-allmids-expansion"
    )
    if resume:
        command += " --resume"
    return command


def _normalization_next_step(
    *,
    normalized_exists: bool,
    consistency_label: str | None,
    in_progress: bool,
    raw_events: int,
) -> str:
    if raw_events <= 0:
        return "No usable raw events; re-run recorder with --long-run."
    if normalized_exists:
        return "Normalized — proceed to replay validation when ready."
    if in_progress:
        return "Normalization in progress — resume with the recommended command."
    if consistency_label == "INCONSISTENT_FINALIZED_MANIFEST":
        return "Manifest inconsistent — inspect lifecycle metadata before normalizing."
    if consistency_label == "STALE_STATUS":
        return "Stale recorder status — reconcile session lifecycle before normalizing."
    if consistency_label == "RAW_USABLE_BUT_LIFECYCLE_UNKNOWN":
        return "Raw JSONL usable — run offline normalization when lifecycle metadata is acceptable."
    return "Run offline normalization with the recommended command."


def build_normalization_queue(
    *,
    show_processed: bool = False,
    stream_dir: Path | None = None,
    max_visible: int = DEFAULT_NORMALIZATION_QUEUE_VISIBLE,
) -> dict[str, Any]:
    """List raw-usable sessions awaiting offline normalization (read-only)."""
    from market_data.lifecycle import assign_consistency_label, validate_manifest
    from market_data.offline_normalizer import checkpoint_path
    from market_data.recovery import discover_session_raw_files

    root = stream_dir or dashboard_data.MARKET_STREAM_DIR
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    manifests = sorted(_iter_session_manifests(root), key=_session_manifest_sort_key, reverse=True)
    sessions_scanned = 0

    for manifest in manifests:
        session_id = str(manifest.get("recorder_session_id") or "")
        if not session_id:
            warnings.append("manifest_missing_session_id")
            continue

        sessions_scanned += 1
        raw_events = int(manifest.get("raw_row_count") or 0)
        channel_counts: dict[str, int] = {}
        if raw_events <= 0:
            _files, _sizes, channel_counts, _malformed, discovered = discover_session_raw_files(
                root, session_id
            )
            raw_events = discovered
        else:
            for channel, count in (manifest.get("channel_counts") or {}).items():
                channel_counts[str(channel)] = int(count)

        validation = validate_manifest(manifest, root)
        consistency_label = assign_consistency_label(
            manifest_data=manifest,
            validation=validation,
            raw_usable=raw_events > 0,
            status_inspection=None,
            session_id=session_id,
        ).value
        if not validation.get("valid"):
            warnings.append(f"{session_id}: {validation.get('issues', [])[:1]}")

        normalized_exists = _normalized_exists_for_session(session_id, manifest, root)
        if normalized_exists and not show_processed:
            continue
        if raw_events <= 0:
            continue

        quality_report = dashboard_data.load_quality_report(session_id)
        quality_label = "unknown"
        if quality_report:
            quality_label = str(quality_report.get("quality_label") or "unknown")

        checkpoint_data: dict[str, Any] | None = None
        ckpt_path = checkpoint_path(root, session_id)
        if ckpt_path.exists():
            try:
                with ckpt_path.open(encoding="utf-8") as handle:
                    checkpoint_data = json.load(handle)
            except (json.JSONDecodeError, OSError):
                warnings.append(f"{session_id}: unreadable_normalization_checkpoint")
        in_progress = bool(
            checkpoint_data
            and not checkpoint_data.get("completed")
            and not normalized_exists
        )
        if checkpoint_data and not checkpoint_data.get("completed") and normalized_exists:
            warnings.append(f"{session_id}: checkpoint_not_completed_but_report_exists")

        start_ms = manifest.get("start_time_ms")
        end_ms = manifest.get("end_time_ms")
        duration_seconds = None
        try:
            if start_ms is not None and end_ms is not None:
                duration_seconds = max(0.0, (int(end_ms) - int(start_ms)) / 1000.0)
        except (TypeError, ValueError):
            duration_seconds = None

        channels = _recommended_normalization_channels(manifest, channel_counts)
        recommended_command = build_normalization_recommended_command(
            session_id,
            channels=channels,
            resume=in_progress,
        )

        manifest_label = consistency_label
        if consistency_label == "INCONSISTENT_FINALIZED_MANIFEST":
            manifest_label = "INCONSISTENT_MANIFEST"

        rows.append(
            {
                "session_id": session_id,
                "start_time_ms": start_ms,
                "end_time_ms": end_ms,
                "duration_seconds": duration_seconds,
                "duration_label": _format_duration_seconds(duration_seconds),
                "raw_events": raw_events,
                "quality_label": quality_label,
                "shutdown_reason": manifest.get("shutdown_reason") or "—",
                "normalized_exists": normalized_exists,
                "recommended_channels": " ".join(channels),
                "recommended_command": recommended_command,
                "next_step": _normalization_next_step(
                    normalized_exists=normalized_exists,
                    consistency_label=consistency_label,
                    in_progress=in_progress,
                    raw_events=raw_events,
                ),
                "manifest_consistency": manifest_label,
                "in_progress": in_progress,
            }
        )

    unprocessed_rows = [row for row in rows if not row.get("normalized_exists")]
    newest_unprocessed = unprocessed_rows[0]["session_id"] if unprocessed_rows else None

    return {
        "rows": rows,
        "visible_rows": rows[:max_visible],
        "older_rows": rows[max_visible:],
        "normalization_queue_count": len(unprocessed_rows),
        "newest_unprocessed_session_id": newest_unprocessed,
        "normalization_sessions_scanned": sessions_scanned,
        "normalization_queue_warnings": warnings,
    }


def build_research_readiness(
    archive_index: dict[str, Any],
    *,
    session_id: str | None = None,
    replay_count: int = 0,
    feature_count: int = 0,
    backtest_count: int = 0,
) -> dict[str, Any]:
    """Research pipeline readiness labels and next recommended command."""
    raw = archive_index.get("raw_preview") or {}
    normalized = archive_index.get("normalized") or {}
    raw_ready = bool(raw.get("available"))
    normalized_ready = bool(normalized.get("available"))
    replay_ready = replay_count > 0
    feature_ready = feature_count > 0
    backtest_ready = backtest_count > 0

    labels = {
        "RAW_READY": raw_ready,
        "NORMALIZED_READY": normalized_ready,
        "REPLAY_READY": replay_ready,
        "FEATURE_READY": feature_ready,
        "BACKTEST_READY": backtest_ready,
    }

    sid = session_id or (archive_index.get("sessions") or [None])[0]
    next_command = None
    if raw_ready and not normalized_ready and sid:
        next_command = f"python3 market_data_normalize.py --session-id {sid}"
    elif normalized_ready and not replay_ready and sid:
        next_command = f"python3 market_data_replay.py --session-id {sid}"
    elif replay_ready and not backtest_ready:
        next_command = (
            "python3 event_backtest.py --latest-replay --coins BTC ETH SOL "
            "--strategy contrarian_bbo_confirmation_v1"
        )
    elif replay_ready and not feature_ready:
        next_command = "python3 feature_research.py --latest-replay"
    elif not raw_ready:
        next_command = (
            "python3 websocket_recorder.py --long-run --coins BTC ETH SOL"
        )

    return {
        "labels": labels,
        "next_recommended_command": next_command,
        "data_quality_labels": DATA_QUALITY_LABELS,
        "explanation": DATA_QUALITY_EXPLANATION,
    }


def meeting_header_cards(bundle: dict[str, Any]) -> dict[str, str]:
    """Top-level meeting-friendly status labels."""
    ws = bundle.get("websocket_status") or {}
    paper = bundle.get("paper_portfolio") or {}

    display_status = ws.get("display_connection_status")
    if display_status:
        websocket_label = display_status
    elif ws.get("stale"):
        websocket_label = "STALE"
    elif ws.get("running") and ws.get("connection_state") == "CONNECTED":
        websocket_label = "CONNECTED"
    else:
        websocket_label = "DISCONNECTED"

    if paper.get("paused"):
        bot_label = "PAUSED"
    elif bundle.get("recommendations"):
        bot_label = "RUNNING"
    else:
        bot_label = "PAUSED"

    return {
        "websocket": websocket_label,
        "paper_bot": bot_label,
        "safety": "PAPER ONLY",
    }
