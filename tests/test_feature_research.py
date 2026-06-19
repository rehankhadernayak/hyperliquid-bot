"""Offline tests for feature research engine v1."""

from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from feature_research.feature_analysis import (
    FeatureAnalyzer,
    _quantile_buckets,
    _safe_correlation,
    summarize_missing_data,
    top_signals,
)
from feature_research.feature_builder import FeatureBuilder
from feature_research.feature_cli import build_parser, config_from_args, run_feature_research
from feature_research.feature_config import FeatureResearchConfig
from feature_research.feature_dataset import (
    build_dataset_records,
    compute_dataset_hash,
    write_dataset,
)
from feature_research.feature_engine import FeatureResearchEngine, validate_no_leakage
from feature_research.feature_models import (
    FEATURE_COLUMN_NAMES,
    FeatureLabel,
    FeatureRow,
    FutureDirection,
    SignalCandidate,
    make_feature_row_id,
)
from feature_research.feature_config import FEATURE_RESEARCH_SCHEMA_VERSION
from feature_research.feature_report import (
    list_feature_research_reports,
    render_markdown_report,
    write_feature_research_report,
)
from feature_research.feature_safety import (
    ALLOWED_CHANNELS_V1,
    FORBIDDEN_MODULE_NAMES,
    FORBIDDEN_SOURCE_PATTERNS,
    PAPER_ONLY,
)
from feature_research.label_builder import LabelBuilder
from market_data.event_models import BboEvent, TradeEvent
from market_data.quality_models import QualityLabel
from market_data.replay_engine import ReplayEngine
from market_data.replay_events import build_replay_event
from market_data.replay_filters import ReplayFilters

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURE_RESEARCH_DIR = REPO_ROOT / "feature_research"


def _write_manifest(tmp_path: Path, session_id: str, files_written: list[str]) -> None:
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "recorder_session_id": session_id,
        "start_time_ms": 1_000_000,
        "end_time_ms": 1_100_000,
        "network": "mainnet",
        "coins": ["BTC", "ETH", "SOL"],
        "schema_version": "1.0.0",
        "files_written": files_written,
        "raw_row_count": 0,
        "normalized_row_count": 0,
        "paper_only": True,
        "execution_enabled": False,
    }
    (manifests_dir / f"{session_id}.json").write_text(json.dumps(data), encoding="utf-8")


def _write_quality(tmp_path: Path, session_id: str, label: str = QualityLabel.GOOD_FOR_RESEARCH.value) -> None:
    reports_dir = tmp_path / "quality_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": "1.0.0",
        "recorder_session_id": session_id,
        "quality_label": label,
        "data_integrity_warnings": [],
        "gap_summary": {"total_gaps": 0},
    }
    (reports_dir / f"{session_id}.json").write_text(json.dumps(data), encoding="utf-8")


def _write_parquet(tmp_path: Path, event_type: str, coin: str, rows: list[dict]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    partition = (
        tmp_path / "normalized" / f"event_type={event_type}" / "date=2026-06-18" / f"coin={coin}"
    )
    partition.mkdir(parents=True, exist_ok=True)
    path = partition / f"events-{event_type}-{coin}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), str(path))
    return path


def _trade_row(coin: str, recv_ms: int, price: float, trade_id: int = 1, side: str = "B") -> dict:
    return {
        "coin": coin,
        "side": side,
        "price": price,
        "size": 0.1,
        "trade_id": trade_id,
        "transaction_hash": "0xabc",
        "exchange_timestamp_ms": recv_ms,
        "received_timestamp_ms": recv_ms,
    }


def _bbo_row(coin: str, recv_ms: int, bid: float = 99.0, ask: float = 101.0) -> dict:
    return {
        "coin": coin,
        "best_bid_price": bid,
        "best_bid_size": 1.0,
        "best_ask_price": ask,
        "best_ask_size": 1.0,
        "spread": ask - bid,
        "spread_bps": ((ask - bid) / ((bid + ask) / 2)) * 10000,
        "exchange_timestamp_ms": recv_ms,
        "received_timestamp_ms": recv_ms,
    }


def _replay_event(event, event_type: str, recv_ms: int, seq: int, coin: str = "BTC") -> object:
    return build_replay_event(
        event=event,
        event_type=event_type,
        replay_session_id="replay1",
        source_recorder_session_id="recorder1",
        source_file="test.parquet",
        source_row_index=seq,
        replay_sequence=seq,
        primary_time_ms=recv_ms,
        primary_time_source="received_timestamp_ms",
    )


@pytest.fixture
def fixture_stream(tmp_path: Path) -> tuple[Path, str, str]:
    session_id = "fixture_recorder_session"
    replay_id = "fixture_replay_session"
    _write_quality(tmp_path, session_id)

    rows_bbo: list[dict] = []
    rows_trade: list[dict] = []
    base_ms = 1_000_000
    price = 100.0
    for i in range(60):
        t_ms = base_ms + i * 1000
        bid = price - 0.5
        ask = price + 0.5
        rows_bbo.append(_bbo_row("BTC", t_ms, bid, ask))
        rows_trade.append(_trade_row("BTC", t_ms + 100, price, trade_id=i + 1))
        price += 0.01 * (1 if i % 3 else -1)

    bbo_path = _write_parquet(tmp_path, "bbo", "BTC", rows_bbo)
    trade_path = _write_parquet(tmp_path, "trade", "BTC", rows_trade)
    _write_manifest(tmp_path, session_id, [str(bbo_path), str(trade_path)])

    replays_dir = tmp_path / "replays"
    replays_dir.mkdir(parents=True, exist_ok=True)
    replay_report = {
        "replay_session_id": replay_id,
        "source_recorder_session_id": session_id,
        "quality_label": QualityLabel.GOOD_FOR_RESEARCH.value,
        "stream_hash": "abc",
        "filters": {"coins": ["BTC"], "channels": ["trades", "bbo"]},
        "paper_only": True,
        "execution_enabled": False,
    }
    (replays_dir / f"{replay_id}.json").write_text(json.dumps(replay_report), encoding="utf-8")
    return tmp_path, session_id, replay_id


class TestCLIParsing:
    def test_build_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.coins == ["BTC", "ETH", "SOL"]
        assert args.channels == ["trades", "bbo"]
        assert args.horizons_seconds == [10, 30, 60, 300]

    def test_config_from_args(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--replay-session-id", "abc123", "--coins", "BTC", "--dry-run"]
        )
        config = config_from_args(args)
        assert config.replay_session_id == "abc123"
        assert config.coins == ("BTC",)
        assert config.dry_run is True


class TestFeatureBuilder:
    def test_trade_bbo_filtering(self, fixture_stream):
        stream_dir, _, replay_id = fixture_stream
        config = FeatureResearchConfig(
            replay_session_id=replay_id,
            coins=("BTC",),
            channels=("trades", "bbo"),
            market_stream_dir=stream_dir,
            test_mode=True,
        )
        engine = ReplayEngine(market_stream_dir=stream_dir, test_mode=True)
        session = engine.load_session("fixture_recorder_session")
        events, _, _ = engine.prepare_events(
            session,
            ReplayFilters(coins=["BTC"], channels=["trades", "bbo"]),
            replay_session_id=replay_id,
        )
        builder = FeatureBuilder(config)
        for event in events:
            builder.process_event(event)
        assert builder.stats.trade_events > 0
        assert builder.stats.bbo_events > 0
        assert builder.stats.rows_created > 0

    def test_missing_bbo_skipped(self):
        config = FeatureResearchConfig(coins=("BTC",), test_mode=True)
        builder = FeatureBuilder(config)
        trade = TradeEvent(
            coin="BTC", side="B", price=100.0, size=0.1, trade_id=1,
            transaction_hash="0x", exchange_timestamp_ms=1000, received_timestamp_ms=1000,
        )
        event = _replay_event(trade, "trade", 1000, 1)
        row = builder.process_event(event)
        assert row is None
        assert builder.stats.skipped_no_bbo == 1

    def test_stale_bbo_skipped(self):
        config = FeatureResearchConfig(coins=("BTC",), bbo_stale_ms=1000, test_mode=True)
        builder = FeatureBuilder(config)
        bbo = BboEvent(
            coin="BTC", best_bid_price=99, best_bid_size=1, best_ask_price=101,
            best_ask_size=1, spread=2, spread_bps=20,
            exchange_timestamp_ms=1000, received_timestamp_ms=1000,
        )
        builder.process_event(_replay_event(bbo, "bbo", 1000, 1))
        trade = TradeEvent(
            coin="BTC", side="B", price=100.0, size=0.1, trade_id=1,
            transaction_hash="0x", exchange_timestamp_ms=5000, received_timestamp_ms=5000,
        )
        row = builder.process_event(_replay_event(trade, "trade", 5000, 2))
        assert row is None
        assert builder.stats.skipped_stale_bbo == 1

    def test_return_features(self):
        config = FeatureResearchConfig(coins=("BTC",), test_mode=True)
        builder = FeatureBuilder(config)
        for i, t_ms in enumerate([1000, 2000, 3000, 4000, 5000, 6000]):
            bbo = BboEvent(
                coin="BTC", best_bid_price=99 + i, best_bid_size=1, best_ask_price=101 + i,
                best_ask_size=1, spread=2, spread_bps=20,
                exchange_timestamp_ms=t_ms, received_timestamp_ms=t_ms,
            )
            builder.process_event(_replay_event(bbo, "bbo", t_ms, i * 2 + 1))
            trade = TradeEvent(
                coin="BTC", side="B", price=100.0 + i, size=0.1, trade_id=i + 1,
                transaction_hash="0x", exchange_timestamp_ms=t_ms + 50,
                received_timestamp_ms=t_ms + 50,
            )
            builder.process_event(_replay_event(trade, "trade", t_ms + 50, i * 2 + 2))
        rows = builder.rows
        assert rows
        assert "return_10s_pct" in rows[-1].features

    def test_feature_row_fields(self):
        config = FeatureResearchConfig(coins=("BTC",), test_mode=True)
        builder = FeatureBuilder(config)
        for i, t_ms in enumerate([1000, 2000, 3000]):
            bbo = BboEvent(
                coin="BTC", best_bid_price=99 + i, best_bid_size=2, best_ask_price=101 + i,
                best_ask_size=1, spread=2, spread_bps=20,
                exchange_timestamp_ms=t_ms, received_timestamp_ms=t_ms,
            )
            builder.process_event(_replay_event(bbo, "bbo", t_ms, i * 2 + 1))
            trade = TradeEvent(
                coin="BTC", side="B", price=100.0 + i, size=0.1, trade_id=i + 1,
                transaction_hash="0x", exchange_timestamp_ms=t_ms + 50,
                received_timestamp_ms=t_ms + 50,
            )
            builder.process_event(_replay_event(trade, "trade", t_ms + 50, i * 2 + 2))
        rows = builder.rows
        assert rows
        row = rows[-1]
        assert row.feature_row_id == make_feature_row_id("replay1", "BTC", row.source_replay_sequence)
        assert row.best_bid == 101
        assert row.spread_bps == 20


class TestLabelBuilder:
    def test_future_label_calculation(self):
        config = FeatureResearchConfig(
            coins=("BTC",), horizons_seconds=(10, 30), test_mode=True,
        )
        label_builder = LabelBuilder(config)
        base = 1_000_000
        for i in range(40):
            t_ms = base + i * 1000
            price = 100.0 + i * 0.1
            bbo = BboEvent(
                coin="BTC", best_bid_price=price - 0.5, best_bid_size=1,
                best_ask_price=price + 0.5, best_ask_size=1, spread=1, spread_bps=10,
                exchange_timestamp_ms=t_ms, received_timestamp_ms=t_ms,
            )
            label_builder.ingest_event(_replay_event(bbo, "bbo", t_ms, i))

        row = FeatureRow(
            feature_row_id="test1",
            replay_session_id="r1",
            source_recorder_session_id="s1",
            coin="BTC",
            event_timestamp_ms=base + 5000,
            received_timestamp_ms=base + 5000,
            latest_trade_price=100.5,
            latest_midpoint=100.5,
            best_bid=100.0,
            best_ask=101.0,
            spread=1.0,
            spread_bps=10.0,
            recent_trade_count=1,
            recent_volume=0.1,
            current_event_type="trade",
            source_replay_sequence=5,
            feature_config_hash="abc",
        )
        labels = label_builder.build_labels((row,))
        assert len(labels) == 1
        assert labels[0].labels["future_return_10s_pct"] is not None

    def test_insufficient_future_data(self):
        config = FeatureResearchConfig(coins=("BTC",), horizons_seconds=(300,), test_mode=True)
        label_builder = LabelBuilder(config)
        row = FeatureRow(
            feature_row_id="test1",
            replay_session_id="r1",
            source_recorder_session_id="s1",
            coin="BTC",
            event_timestamp_ms=1_000_000,
            received_timestamp_ms=1_000_000,
            latest_trade_price=100.0,
            latest_midpoint=100.0,
            best_bid=99.0,
            best_ask=101.0,
            spread=2.0,
            spread_bps=20.0,
            recent_trade_count=1,
            recent_volume=0.1,
            current_event_type="trade",
            source_replay_sequence=1,
            feature_config_hash="abc",
        )
        labels = label_builder.build_labels((row,))
        assert labels[0].labels["future_direction_300s"] == FutureDirection.INSUFFICIENT_FUTURE_DATA.value

    def test_cost_aware_labels(self):
        config = FeatureResearchConfig(
            coins=("BTC",), horizons_seconds=(10,), trading_fee_pct=0.05, slippage_pct=0.02,
            test_mode=True,
        )
        label_builder = LabelBuilder(config)
        base = 1_000_000
        for i in range(20):
            t_ms = base + i * 1000
            price = 100.0 + (2.0 if i >= 6 else 0.0)
            bbo = BboEvent(
                coin="BTC", best_bid_price=price - 0.5, best_bid_size=1,
                best_ask_price=price + 0.5, best_ask_size=1, spread=1, spread_bps=10,
                exchange_timestamp_ms=t_ms, received_timestamp_ms=t_ms,
            )
            label_builder.ingest_event(_replay_event(bbo, "bbo", t_ms, i))
        row = FeatureRow(
            feature_row_id="test1",
            replay_session_id="r1",
            source_recorder_session_id="s1",
            coin="BTC",
            event_timestamp_ms=base + 5000,
            received_timestamp_ms=base + 5000,
            latest_trade_price=100.0,
            latest_midpoint=100.0,
            best_bid=99.5,
            best_ask=100.5,
            spread=1.0,
            spread_bps=10.0,
            recent_trade_count=1,
            recent_volume=0.1,
            current_event_type="trade",
            source_replay_sequence=5,
            feature_config_hash="abc",
        )
        labels = label_builder.build_labels((row,))
        lbl = labels[0].labels
        assert "future_up_after_costs_10s" in lbl
        assert lbl.get("future_direction_10s") in (
            FutureDirection.UP.value,
            FutureDirection.DOWN.value,
            FutureDirection.FLAT.value,
        )


class TestNoFutureLeakage:
    def test_future_changes_affect_labels_not_features(self):
        config = FeatureResearchConfig(coins=("BTC",), horizons_seconds=(10,), test_mode=True)
        builder = FeatureBuilder(config)
        label_builder = LabelBuilder(config)
        base = 1_000_000
        events = []
        for i in range(30):
            t_ms = base + i * 1000
            price = 100.0 + i * 0.05
            bbo = BboEvent(
                coin="BTC", best_bid_price=price - 0.5, best_bid_size=1,
                best_ask_price=price + 0.5, best_ask_size=1, spread=1, spread_bps=10,
                exchange_timestamp_ms=t_ms, received_timestamp_ms=t_ms,
            )
            trade = TradeEvent(
                coin="BTC", side="B", price=price, size=0.1, trade_id=i + 1,
                transaction_hash="0x", exchange_timestamp_ms=t_ms + 50,
                received_timestamp_ms=t_ms + 50,
            )
            events.append(_replay_event(bbo, "bbo", t_ms, i * 2 + 1))
            events.append(_replay_event(trade, "trade", t_ms + 50, i * 2 + 2))

        for event in events:
            label_builder.ingest_event(event)
        for event in events:
            builder.process_event(event)

        frozen = builder.freeze_rows()
        assert frozen, "expected feature rows from stream"
        target_row = frozen[0]
        features_before = copy.deepcopy(target_row.to_dict())
        labels_before = label_builder.build_labels((target_row,))

        mutated_label_builder = LabelBuilder(config)
        for event in events:
            if event.primary_time_ms > target_row.event_timestamp_ms:
                if event.original_event_type == "bbo":
                    bbo = event.event
                    mutated_bbo = BboEvent(
                        coin=bbo.coin,
                        best_bid_price=bbo.best_bid_price + 500,
                        best_bid_size=bbo.best_bid_size,
                        best_ask_price=bbo.best_ask_price + 500,
                        best_ask_size=bbo.best_ask_size,
                        spread=bbo.spread,
                        spread_bps=bbo.spread_bps,
                        exchange_timestamp_ms=bbo.exchange_timestamp_ms,
                        received_timestamp_ms=bbo.received_timestamp_ms,
                    )
                    mutated_label_builder.ingest_event(
                        _replay_event(
                            mutated_bbo, "bbo", event.primary_time_ms, event.replay_sequence
                        )
                    )
                elif event.original_event_type == "trade":
                    trade = event.event
                    mutated_trade = TradeEvent(
                        coin=trade.coin,
                        side=trade.side,
                        price=trade.price + 500,
                        size=trade.size,
                        trade_id=trade.trade_id,
                        transaction_hash=trade.transaction_hash,
                        exchange_timestamp_ms=trade.exchange_timestamp_ms,
                        received_timestamp_ms=trade.received_timestamp_ms,
                    )
                    mutated_label_builder.ingest_event(
                        _replay_event(
                            mutated_trade, "trade", event.primary_time_ms, event.replay_sequence
                        )
                    )
            else:
                mutated_label_builder.ingest_event(event)

        labels_after = mutated_label_builder.build_labels((target_row,))
        features_after = target_row.to_dict()

        assert features_before == features_after
        assert labels_before[0].labels["future_return_10s_pct"] != labels_after[0].labels["future_return_10s_pct"]

    def test_immutable_feature_rows_before_labeling(self):
        config = FeatureResearchConfig(coins=("BTC",), test_mode=True)
        builder = FeatureBuilder(config)
        for i, t_ms in enumerate([1000, 2000, 3000]):
            bbo = BboEvent(
                coin="BTC", best_bid_price=99 + i, best_bid_size=1, best_ask_price=101 + i,
                best_ask_size=1, spread=2, spread_bps=20,
                exchange_timestamp_ms=t_ms, received_timestamp_ms=t_ms,
            )
            builder.process_event(_replay_event(bbo, "bbo", t_ms, i * 2 + 1))
            trade = TradeEvent(
                coin="BTC", side="B", price=100.0 + i, size=0.1, trade_id=i + 1,
                transaction_hash="0x", exchange_timestamp_ms=t_ms + 50,
                received_timestamp_ms=t_ms + 50,
            )
            builder.process_event(_replay_event(trade, "trade", t_ms + 50, i * 2 + 2))
        frozen = builder.freeze_rows()
        result = validate_no_leakage(frozen)
        assert result.passed


class TestFeatureAnalysis:
    def test_correlation_calculation(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert _safe_correlation(xs, ys) == pytest.approx(1.0)

    def test_quantile_bucketing(self):
        pairs = [(float(i), float(i), True, 1.0, -1.0) for i in range(25)]
        buckets, _ = _quantile_buckets(pairs, 5)
        assert len(buckets) == 5

    def test_small_sample_warning(self):
        config = FeatureResearchConfig(coins=("BTC",), min_rows_for_signal=100, test_mode=True)
        rows = tuple(
            FeatureRow(
                feature_row_id=f"r{i}",
                replay_session_id="r1",
                source_recorder_session_id="s1",
                coin="BTC",
                event_timestamp_ms=1000 + i,
                received_timestamp_ms=1000 + i,
                latest_trade_price=100.0,
                latest_midpoint=100.0,
                best_bid=99.0,
                best_ask=101.0,
                spread=2.0,
                spread_bps=20.0,
                recent_trade_count=1,
                recent_volume=0.1,
                current_event_type="trade",
                source_replay_sequence=i,
                feature_config_hash="abc",
                features={"return_10s_pct": float(i % 3)},
            )
            for i in range(10)
        )
        labels = [
            FeatureLabel(
                feature_row_id=f"r{i}",
                labels={"future_return_10s_pct": float(i % 5)},
            )
            for i in range(10)
        ]
        results = FeatureAnalyzer(config).analyze(rows, labels)
        assert all(r.candidate_signal == SignalCandidate.NO_SIGNAL.value for r in results)


class TestDeterministicHash:
    def test_repeat_run_determinism(self, fixture_stream, tmp_path: Path):
        stream_dir, _, replay_id = fixture_stream
        out_dir = tmp_path / "feature_research"
        config = FeatureResearchConfig(
            replay_session_id=replay_id,
            coins=("BTC",),
            channels=("trades", "bbo"),
            horizons_seconds=(10, 30),
            output_dir=out_dir,
            market_stream_dir=stream_dir,
            test_mode=True,
        )
        r1 = FeatureResearchEngine(config).run(feature_research_id="fixed_id_1")
        r2 = FeatureResearchEngine(config).run(feature_research_id="fixed_id_2")
        assert r1.deterministic_dataset_hash == r2.deterministic_dataset_hash
        assert r1.deterministic_dataset_hash != ""


class TestReportOutput:
    def test_json_and_markdown_output(self, tmp_path: Path):
        report = {
            "schema_version": FEATURE_RESEARCH_SCHEMA_VERSION,
            "feature_research_id": "test123",
            "replay_session_id": "replay1",
            "source_recorder_session_id": "rec1",
            "quality_label": "GOOD_FOR_RESEARCH",
            "row_count": 10,
            "feature_count": len(FEATURE_COLUMN_NAMES),
            "label_count": 10,
            "horizons": [10, 30],
            "coins": ["BTC"],
            "channels": ["trades", "bbo"],
            "missing_data_summary": {"total_rows": 10, "features": {}},
            "leakage_check": {"passed": True, "checks": [], "warnings": []},
            "top_possible_signals": [],
            "rejected_signals": [],
            "warnings": [],
            "limitations": [],
            "recommendations": [],
            "deterministic_dataset_hash": "abc123",
            "paper_only": True,
            "execution_enabled": False,
            "safety_statement": "test",
        }
        paths = write_feature_research_report(report, tmp_path)
        assert paths["json"].exists()
        assert paths["markdown"].exists()
        md = paths["markdown"].read_text(encoding="utf-8")
        assert "Feature Research Report" in md
        listed = list_feature_research_reports(tmp_path)
        assert len(listed) == 1


class TestDatasetOutput:
    def test_dataset_parquet_or_jsonl(self, tmp_path: Path):
        row = FeatureRow(
            feature_row_id="r1",
            replay_session_id="replay1",
            source_recorder_session_id="rec1",
            coin="BTC",
            event_timestamp_ms=1000,
            received_timestamp_ms=1000,
            latest_trade_price=100.0,
            latest_midpoint=100.0,
            best_bid=99.0,
            best_ask=101.0,
            spread=2.0,
            spread_bps=20.0,
            recent_trade_count=1,
            recent_volume=0.1,
            current_event_type="trade",
            source_replay_sequence=1,
            feature_config_hash="abc",
            features={"return_10s_pct": 0.5},
        )
        label = FeatureLabel(feature_row_id="r1", labels={"future_return_10s_pct": 0.1})
        records = build_dataset_records((row,), [label])
        path, fmt = write_dataset(records, tmp_path / "datasets" / "test.parquet")
        assert path.exists()
        assert fmt in ("parquet", "jsonl")


class TestEngineIntegration:
    def test_full_run(self, fixture_stream, tmp_path: Path):
        stream_dir, _, replay_id = fixture_stream
        config = FeatureResearchConfig(
            replay_session_id=replay_id,
            coins=("BTC",),
            channels=("trades", "bbo"),
            horizons_seconds=(10, 30),
            output_dir=tmp_path / "feature_research",
            market_stream_dir=stream_dir,
            test_mode=True,
        )
        result = FeatureResearchEngine(config).run()
        assert result.report["row_count"] > 0
        assert result.report_paths["json"].exists()
        assert result.report["paper_only"] is True
        assert result.report["execution_enabled"] is False

    def test_dry_run(self, fixture_stream, tmp_path: Path):
        stream_dir, _, replay_id = fixture_stream
        config = FeatureResearchConfig(
            replay_session_id=replay_id,
            coins=("BTC",),
            output_dir=tmp_path / "feature_research",
            market_stream_dir=stream_dir,
            dry_run=True,
            test_mode=True,
        )
        result = FeatureResearchEngine(config).run()
        assert result.report["dry_run"] is True
        assert not result.report_paths

    def test_zero_row_handling(self, tmp_path: Path):
        session_id = "empty_session"
        _write_quality(tmp_path, session_id)
        _write_manifest(tmp_path, session_id, [])
        replay_id = "empty_replay"
        replays_dir = tmp_path / "replays"
        replays_dir.mkdir(parents=True, exist_ok=True)
        (replays_dir / f"{replay_id}.json").write_text(
            json.dumps(
                {
                    "replay_session_id": replay_id,
                    "source_recorder_session_id": session_id,
                    "filters": {},
                }
            ),
            encoding="utf-8",
        )
        config = FeatureResearchConfig(
            replay_session_id=replay_id,
            output_dir=tmp_path / "feature_research",
            market_stream_dir=tmp_path,
            test_mode=True,
        )
        result = FeatureResearchEngine(config).run()
        assert result.report["row_count"] == 0


class TestDashboardLoader:
    def test_load_feature_research_reports(self, tmp_path: Path, monkeypatch):
        out_dir = tmp_path / "feature_research"
        report = {
            "feature_research_id": "dash_test",
            "replay_session_id": "replay1",
            "source_recorder_session_id": "rec1",
            "row_count": 5,
            "generated_at": "2026-06-19T00:00:00Z",
            "paper_only": True,
            "execution_enabled": False,
        }
        write_feature_research_report(report, out_dir)
        import dashboard_data

        monkeypatch.setattr(dashboard_data, "FEATURE_RESEARCH_DIR", out_dir)
        reports = dashboard_data.load_feature_research_reports()
        assert len(reports) == 1
        assert reports[0]["feature_research_id"] == "dash_test"


class TestSafety:
    def test_no_forbidden_imports(self):
        for path in FEATURE_RESEARCH_DIR.rglob("*.py"):
            if path.name == "feature_safety.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert alias.name.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert node.module.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES

    def test_no_forbidden_patterns(self):
        for path in FEATURE_RESEARCH_DIR.rglob("*.py"):
            if path.name == "feature_safety.py":
                continue
            source = path.read_text(encoding="utf-8").lower()
            for pattern in FORBIDDEN_SOURCE_PATTERNS:
                assert pattern not in source, f"{pattern} found in {path}"

    def test_paper_only_constants(self):
        assert PAPER_ONLY is True
        assert ALLOWED_CHANNELS_V1 == frozenset({"trades", "bbo"})

    def test_no_network_in_engine(self, fixture_stream, tmp_path: Path):
        stream_dir, _, replay_id = fixture_stream
        config = FeatureResearchConfig(
            replay_session_id=replay_id,
            coins=("BTC",),
            output_dir=tmp_path / "feature_research",
            market_stream_dir=stream_dir,
            test_mode=True,
        )

        def block_network(*args, **kwargs):
            raise RuntimeError("Network call attempted during feature research")

        with patch("urllib.request.urlopen", side_effect=block_network):
            with patch("requests.get", side_effect=block_network):
                with patch("requests.post", side_effect=block_network):
                    result = FeatureResearchEngine(config).run()
        assert result.report["row_count"] >= 0

    def test_existing_paper_trader_unchanged(self):
        import paper_trader

        assert hasattr(paper_trader, "PaperTrader")

    def test_existing_event_backtester_unchanged(self):
        from event_backtester.event_backtest_engine import EventBacktestEngine

        assert EventBacktestEngine is not None


class TestCLI:
    def test_cli_dry_run(self, fixture_stream, tmp_path: Path, capsys):
        stream_dir, _, replay_id = fixture_stream
        out_dir = tmp_path / "feature_research"
        code = run_feature_research(
            [
                "--replay-session-id",
                replay_id,
                "--coins",
                "BTC",
                "--output-dir",
                str(out_dir),
                "--market-stream-dir",
                str(stream_dir),
                "--dry-run",
            ]
        )
        assert code == 0
        captured = capsys.readouterr()
        assert "Feature research complete" in captured.out
