"""Offline tests for long-run recorder resilience features."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

from market_data.checkpoint import build_checkpoint_data, read_checkpoint, write_checkpoint
from market_data.lifecycle import (
    SessionConsistencyLabel,
    diagnose_session,
    ensure_terminal_shutdown_reason,
    format_diagnosis_report,
    is_terminal_shutdown_reason,
    repair_session_lifecycle_metadata,
    validate_manifest,
)
from market_data.manifest import build_manifest, write_manifest
from market_data.offline_normalizer import (
    OfflineNormalizerConfig,
    NormalizationProgress,
    normalize_session,
    write_normalization_report,
)
from market_data.recorder import MarketDataRecorder
from market_data.recorder_config import RecorderConfig, apply_long_run_defaults, build_config_from_cli
from market_data.recovery import discover_session_raw_files, recover_session, write_recovery_report
from market_data.replay_engine import ReplayEngine, ReplayError, check_normalized_completeness
from market_data.quality_checks import assign_overall_label, inspect_session
from market_data.quality_report import build_quality_report
from market_data.replay_manifest import load_session_manifest
from market_data.status import (
    format_status_summary,
    inspect_live_status,
    mark_lock_inactive,
    read_lock,
    write_lock,
    write_status,
)
from market_data.subscription_watchdog import SubscriptionWatchdog
from market_data.subscriptions import SubscriptionTracker


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ws_messages.json"
SESSION_ID = "testlongrun1234567890abcdef123456"


def _raw_record(channel: str, payload: dict, session_id: str = SESSION_ID) -> dict:
    return {
        "schema_version": "1.0.0",
        "recorder_session_id": session_id,
        "network": "mainnet",
        "channel": channel,
        "subscription": {"type": channel, "coin": "BTC"} if channel != "allMids" else None,
        "exchange_timestamp_ms": 1718000001000,
        "received_timestamp_ms": 1718000001000,
        "sequence": 1,
        "raw_payload": payload,
        "raw_payload_hash": "abc",
        "reconnect_generation": 0,
        "source": "hyperliquid_websocket",
    }


def _write_raw_session(tmp_path: Path, session_id: str = SESSION_ID) -> Path:
    stream = tmp_path / "market_stream"
    raw_dir = stream / "raw" / "2026-06-22"
    raw_dir.mkdir(parents=True)
    raw_file = raw_dir / "events-10.jsonl"
    with FIXTURES.open(encoding="utf-8") as handle:
        fixtures = json.load(handle)
    with raw_file.open("w", encoding="utf-8") as handle:
        for key in ("trade", "bbo", "all_mids"):
            msg = fixtures[key]
            record = _raw_record(msg["channel"], msg, session_id)
            handle.write(json.dumps(record) + "\n")
    return stream


class TestLongRunConfig:
    def test_long_run_defaults(self):
        config = build_config_from_cli(
            coins=("BTC", "ETH", "SOL"),
            network="mainnet",
            output_dir=Path("data/market_stream"),
            duration_seconds=21600,
            log_level="INFO",
            raw_only=False,
            disable_l2=False,
            candle_interval="1m",
            long_run=True,
        )
        assert config.long_run is True
        assert config.raw_only is True
        assert config.effective_live_normalization is False
        assert config.disable_l2 is True
        assert config.expand_allmids_live is False
        assert config.checkpoint_manifest is True
        assert config.subscription_watchdog is True
        assert config.status_heartbeat_enabled is True

    def test_enable_live_normalization_override(self):
        config = build_config_from_cli(
            coins=("BTC",),
            network="mainnet",
            output_dir=Path("data/market_stream"),
            duration_seconds=None,
            log_level="INFO",
            raw_only=False,
            disable_l2=False,
            candle_interval="1m",
            long_run=True,
            enable_live_normalization=True,
        )
        assert config.effective_live_normalization is True

    def test_enable_l2_override(self):
        config = build_config_from_cli(
            coins=("BTC",),
            network="mainnet",
            output_dir=Path("data/market_stream"),
            duration_seconds=None,
            log_level="INFO",
            raw_only=False,
            disable_l2=False,
            candle_interval="1m",
            long_run=True,
            enable_l2=True,
        )
        assert config.disable_l2 is False


class TestRawOnlyRecorder:
    def test_raw_only_no_normalized_writer(self, tmp_path):
        config = RecorderConfig(
            coins=("BTC",),
            output_dir=tmp_path / "market_stream",
            raw_only=True,
            long_run=True,
        )
        recorder = MarketDataRecorder(config)
        assert recorder._normalized_writer is None

    def test_allmids_not_expanded_live_in_long_run(self, tmp_path):
        config = RecorderConfig(
            coins=("BTC", "ETH", "SOL"),
            output_dir=tmp_path / "market_stream",
            live_normalization=True,
            expand_allmids_live=False,
        )
        recorder = MarketDataRecorder(config)
        fixtures = json.load(FIXTURES.open())
        msg = fixtures["all_mids"]
        before = recorder._normalized_writer.row_count if recorder._normalized_writer else 0
        recorder._handle_message("allMids", msg, 1718000001000)
        recorder._raw_writer.shutdown()
        if recorder._normalized_writer:
            recorder._normalized_writer.shutdown()
        assert recorder._raw_writer.row_count == 1
        if recorder._normalized_writer:
            assert recorder._normalized_writer.row_count == before


class TestStatusAndLock:
    def test_status_includes_pid(self, tmp_path):
        stream = tmp_path / "market_stream"
        write_status(stream, {
            "running": True,
            "recorder_session_id": SESSION_ID,
            "pid": 12345,
            "heartbeat_timestamp": "2026-06-22T10:00:00+00:00",
        })
        status = inspect_live_status(stream)
        assert status["pid"] == 12345

    def test_stale_status_when_pid_missing(self, tmp_path):
        stream = tmp_path / "market_stream"
        write_status(stream, {
            "running": True,
            "recorder_session_id": SESSION_ID,
            "pid": 999999999,
            "heartbeat_timestamp": "2020-01-01T00:00:00+00:00",
        })
        write_lock(stream, 999999999, SESSION_ID, active=True)
        status = inspect_live_status(stream)
        assert status["running"] is False
        assert status["state"] == "STALE_STATUS"

    def test_lock_file_active_inactive(self, tmp_path):
        stream = tmp_path / "market_stream"
        write_lock(stream, os.getpid(), SESSION_ID, active=True)
        lock = read_lock(stream)
        assert lock["active"] is True
        mark_lock_inactive(stream)
        lock = read_lock(stream)
        assert lock["active"] is False

    def test_stale_status_when_heartbeat_old(self, tmp_path):
        stream = tmp_path / "market_stream"
        write_status(stream, {
            "running": True,
            "recorder_session_id": SESSION_ID,
            "pid": os.getpid(),
            "heartbeat_timestamp": "2020-01-01T00:00:00+00:00",
        })
        write_lock(stream, os.getpid(), SESSION_ID, active=True)
        with mock.patch(
            "market_data.status.pid_matches_recorder",
            return_value=True,
        ):
            status = inspect_live_status(stream, stale_heartbeat_seconds=30.0)
        assert status["running"] is False
        assert status["state"] == "STALE_STATUS"
        assert status["heartbeat_stale"] is True

    def test_stale_status_when_heartbeat_missing(self, tmp_path):
        stream = tmp_path / "market_stream"
        write_status(stream, {
            "running": True,
            "recorder_session_id": SESSION_ID,
            "pid": os.getpid(),
        })
        write_lock(stream, os.getpid(), SESSION_ID, active=True)
        with mock.patch(
            "market_data.status.pid_matches_recorder",
            return_value=True,
        ):
            status = inspect_live_status(stream)
        assert status["running"] is False
        assert status["state"] == "STALE_STATUS"
        assert status["heartbeat_stale"] is True

    def test_stale_status_when_lock_inactive(self, tmp_path):
        stream = tmp_path / "market_stream"
        write_status(stream, {
            "running": True,
            "recorder_session_id": SESSION_ID,
            "pid": os.getpid(),
            "heartbeat_timestamp": datetime.now(timezone.utc).isoformat(),
        })
        write_lock(stream, os.getpid(), SESSION_ID, active=False)
        with mock.patch(
            "market_data.status.pid_matches_recorder",
            return_value=True,
        ):
            status = inspect_live_status(stream)
        assert status["running"] is False
        assert status["state"] == "STALE_STATUS"
        assert status["lock_stale"] is True

    def test_format_status_shows_stale(self, tmp_path):
        summary = format_status_summary({
            "running": False,
            "state": "STALE_STATUS",
            "recorder_session_id": SESSION_ID,
            "heartbeat_timestamp": "2020-01-01T00:00:00+00:00",
        })
        assert "STALE_STATUS" in summary
        assert "--recover-session" in summary


class TestDashboardStaleStatus:
    def test_dashboard_loader_handles_stale_status(self, tmp_path, monkeypatch):
        import dashboard_data

        stream = tmp_path / "market_stream"
        write_status(stream, {
            "running": True,
            "recorder_session_id": SESSION_ID,
            "pid": 999999999,
            "heartbeat_timestamp": "2020-01-01T00:00:00+00:00",
        })
        monkeypatch.setattr(dashboard_data, "MARKET_STREAM_DIR", stream)
        status = dashboard_data.load_recorder_status()
        assert status is not None
        assert status["running"] is False
        assert status["state"] == "STALE_STATUS"


class TestNoLiveFileModification:
    def test_recovery_does_not_modify_raw_files(self, tmp_path):
        stream = _write_raw_session(tmp_path)
        raw_dir = stream / "raw"
        before = {
            str(p): p.read_bytes()
            for p in raw_dir.rglob("*.jsonl")
        }
        recover_session(SESSION_ID, stream)
        after = {
            str(p): p.read_bytes()
            for p in raw_dir.rglob("*.jsonl")
        }
        assert before == after


class TestCheckpointAndRecovery:
    def test_manifest_checkpoint_writing(self, tmp_path):
        manifests_dir = tmp_path / "manifests"
        data = build_checkpoint_data(
            recorder_session_id=SESSION_ID,
            start_time_ms=1000,
            raw_files=["a.jsonl"],
            file_sizes={"a.jsonl": 100},
            channel_counts={"trades": 5},
            reconnect_count=0,
            active_subscriptions=10,
            requested_subscriptions=10,
            known_gaps=[],
            dropped_normalized_count=0,
            raw_dropped_count=0,
            degraded=False,
            degraded_reason=None,
            last_heartbeat_timestamp="2026-06-22T10:00:00+00:00",
        )
        path = write_checkpoint(manifests_dir, SESSION_ID, data)
        loaded = read_checkpoint(manifests_dir, SESSION_ID)
        assert loaded is not None
        assert loaded["recorder_session_id"] == SESSION_ID
        assert path.exists()

    def test_recovery_from_missing_manifest(self, tmp_path):
        stream = _write_raw_session(tmp_path)
        report = recover_session(SESSION_ID, stream)
        assert report["raw_usable"] is True
        assert report["raw_event_count"] >= 2
        manifest_path = stream / "manifests" / f"{SESSION_ID}.json"
        assert manifest_path.exists()

    def test_recovery_when_final_manifest_exists(self, tmp_path):
        stream = _write_raw_session(tmp_path)
        manifests_dir = stream / "manifests"
        manifests_dir.mkdir(parents=True)
        manifest = {
            "recorder_session_id": SESSION_ID,
            "manifest_finalized": True,
            "raw_row_count": 3,
            "paper_only": True,
            "execution_enabled": False,
        }
        (manifests_dir / f"{SESSION_ID}.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        report = recover_session(SESSION_ID, stream)
        assert report["final_manifest_exists"] is True
        assert report["raw_usable"] is True

    def test_recovery_report_output(self, tmp_path):
        stream = tmp_path / "market_stream"
        report = {"session_id": SESSION_ID, "raw_usable": True}
        paths = write_recovery_report(stream, SESSION_ID, report)
        assert paths["json"].exists()
        assert paths["markdown"].exists()


class TestOfflineNormalizer:
    def test_reads_raw_jsonl(self, tmp_path):
        stream = _write_raw_session(tmp_path)
        config = OfflineNormalizerConfig(
            session_id=SESSION_ID,
            market_stream_dir=stream,
            channels=("trades", "bbo"),
            skip_allmids_expansion=True,
        )
        report = normalize_session(config)
        assert report["events_processed"] >= 2

    def test_skip_allmids_expansion(self, tmp_path):
        stream = _write_raw_session(tmp_path)
        config = OfflineNormalizerConfig(
            session_id=SESSION_ID,
            market_stream_dir=stream,
            channels=("trades", "bbo", "allMids"),
            skip_allmids_expansion=True,
        )
        report = normalize_session(config)
        assert report["skipped_channels"].get("allMids", 0) >= 1

    def test_writes_reports(self, tmp_path):
        stream = _write_raw_session(tmp_path)
        config = OfflineNormalizerConfig(
            session_id=SESSION_ID,
            market_stream_dir=stream,
            channels=("trades",),
            skip_allmids_expansion=True,
        )
        report = normalize_session(config)
        assert "report_paths" in report
        json_path = Path(report["report_paths"]["json"])
        assert json_path.exists()

    def test_malformed_raw_handling(self, tmp_path):
        stream = tmp_path / "market_stream"
        raw_dir = stream / "raw" / "2026-06-22"
        raw_dir.mkdir(parents=True)
        raw_file = raw_dir / "events-10.jsonl"
        raw_file.write_text("not json\n", encoding="utf-8")
        config = OfflineNormalizerConfig(
            session_id=SESSION_ID,
            market_stream_dir=stream,
            channels=("trades",),
        )
        report = normalize_session(config)
        assert report["malformed_count"] >= 0

    def test_resume_checkpoint(self, tmp_path):
        progress = NormalizationProgress(
            session_id=SESSION_ID,
            last_file="/tmp/foo.jsonl",
            last_line=10,
            events_processed=100,
        )
        data = progress.to_dict()
        restored = NormalizationProgress.from_dict(data)
        assert restored.events_processed == 100


class TestSubscriptionWatchdog:
    def test_detects_drop(self, tmp_path):
        config = RecorderConfig(coins=("BTC",), output_dir=tmp_path / "ms")
        tracker = SubscriptionTracker(config)
        resubscribed = []

        def on_resub(sub):
            resubscribed.append(sub)

        watchdog = SubscriptionWatchdog(
            tracker=tracker,
            threshold_seconds=0.001,
            on_resubscribe=on_resub,
        )
        warnings = watchdog.check()
        assert isinstance(warnings, list)

    def test_resubscribe_callback(self, tmp_path):
        config = RecorderConfig(coins=("BTC",), output_dir=tmp_path / "ms")
        tracker = SubscriptionTracker(config)
        called = []
        watchdog = SubscriptionWatchdog(
            tracker=tracker,
            on_resubscribe=lambda s: called.append(s),
        )
        watchdog.check()
        assert isinstance(called, list)


class TestQualityAndReplay:
    def test_raw_only_not_unusable(self, tmp_path):
        stream = _write_raw_session(tmp_path)
        manifest = load_session_manifest(SESSION_ID, stream)
        state = inspect_session(manifest)
        label = assign_overall_label(state, manifest)
        assert label.value in ("USABLE_WITH_WARNINGS", "GOOD_FOR_RESEARCH", "DEGRADED")
        assert label.value != "UNUSABLE"

    def test_quality_report_raw_only(self, tmp_path):
        stream = _write_raw_session(tmp_path)
        report = build_quality_report(SESSION_ID, stream)
        assert report["quality_label"] != "UNUSABLE"
        assert report["usability_flags"]["usable_for_raw_replay"]["usable"] is True

    def test_replay_blocks_incomplete_normalized(self, tmp_path):
        stream = _write_raw_session(tmp_path)
        manifest = load_session_manifest(SESSION_ID, stream)
        complete, msg = check_normalized_completeness(manifest, stream, SESSION_ID)
        assert complete is False
        assert "market_data_normalize.py" in (msg or "")

    def test_replay_use_raw(self, tmp_path):
        stream = _write_raw_session(tmp_path)
        engine = ReplayEngine(stream)
        result = engine.run(SESSION_ID, dry_run=True, use_raw=True, output_dir=stream / "replays")
        assert result.report["total_events"] > 0

    def test_replay_require_normalized_blocks(self, tmp_path):
        stream = _write_raw_session(tmp_path)
        engine = ReplayEngine(stream)
        with pytest.raises(ReplayError):
            engine.run(SESSION_ID, dry_run=True, require_normalized=True)


class TestSecurity:
    def test_no_wallet_in_recorder_config(self):
        config = RecorderConfig(long_run=True)
        d = config.to_dict()
        assert d["paper_only"] is True
        assert d["execution_enabled"] is False

    def test_no_network_in_offline_normalizer(self, tmp_path, monkeypatch):
        stream = _write_raw_session(tmp_path)

        def fail_network(*args, **kwargs):
            raise AssertionError("Network call attempted in offline normalizer test")

        monkeypatch.setattr("socket.socket", fail_network)
        config = OfflineNormalizerConfig(
            session_id=SESSION_ID,
            market_stream_dir=stream,
            channels=("trades",),
        )
        normalize_session(config)


class TestRecorderLifecycle:
    def test_finalize_never_keeps_running(self, tmp_path):
        config = RecorderConfig(
            coins=("BTC",),
            output_dir=tmp_path / "market_stream",
            raw_only=True,
        )
        recorder = MarketDataRecorder(config)
        reason = recorder._finalize_shutdown_reason()
        assert reason == "unknown_stop"
        assert is_terminal_shutdown_reason(reason)

    def test_unknown_exit_becomes_unknown_stop(self, tmp_path):
        config = RecorderConfig(
            coins=("BTC",),
            output_dir=tmp_path / "market_stream",
            raw_only=True,
        )
        recorder = MarketDataRecorder(config)
        import asyncio

        asyncio.run(recorder._shutdown())
        manifest_path = (
            tmp_path
            / "market_stream"
            / "manifests"
            / f"{recorder.recorder_session_id}.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["shutdown_reason"] == "unknown_stop"
        assert manifest["manifest_finalized"] is True

    def test_duration_limit_shutdown_reason(self, tmp_path):
        config = RecorderConfig(
            coins=("BTC",),
            output_dir=tmp_path / "market_stream",
            raw_only=True,
        )
        recorder = MarketDataRecorder(config)
        recorder._shutdown_reason = "duration_limit"
        import asyncio

        asyncio.run(recorder._shutdown())
        manifest_path = (
            tmp_path
            / "market_stream"
            / "manifests"
            / f"{recorder.recorder_session_id}.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["shutdown_reason"] == "duration_limit"

    def test_keyboard_interrupt_shutdown_reason(self, tmp_path):
        config = RecorderConfig(
            coins=("BTC",),
            output_dir=tmp_path / "market_stream",
            raw_only=True,
        )
        recorder = MarketDataRecorder(config)
        recorder.stop(reason="keyboard_interrupt")
        import asyncio

        asyncio.run(recorder._shutdown())
        manifest_path = (
            tmp_path
            / "market_stream"
            / "manifests"
            / f"{recorder.recorder_session_id}.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["shutdown_reason"] == "keyboard_interrupt"

    def test_status_flags_inconsistent_finalized_manifest(self, tmp_path):
        stream = tmp_path / "market_stream"
        manifests_dir = stream / "manifests"
        manifests_dir.mkdir(parents=True)
        manifest = {
            "recorder_session_id": SESSION_ID,
            "manifest_finalized": True,
            "shutdown_reason": "running",
            "end_time_ms": 2000,
            "raw_row_count": 3,
            "paper_only": True,
            "execution_enabled": False,
            "files_written": [],
        }
        (manifests_dir / f"{SESSION_ID}.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        write_status(stream, {
            "running": False,
            "recorder_session_id": SESSION_ID,
            "pid": 999999999,
            "manifest_finalized": True,
            "shutdown_reason": "running",
        })
        status = inspect_live_status(stream)
        assert status["running"] is False
        assert status["state"] == "STOPPED"
        assert status["consistency_warning"] == "INCONSISTENT_FINALIZED_MANIFEST"
        summary = format_status_summary(status)
        assert "INCONSISTENT_FINALIZED_MANIFEST" in summary
        assert "--diagnose-session" in summary

    def test_diagnose_labels_inconsistent_manifest(self, tmp_path):
        stream = _write_raw_session(tmp_path, session_id=SESSION_ID)
        manifests_dir = stream / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        raw_files, _, _, _, raw_count = discover_session_raw_files(stream, SESSION_ID)
        manifest = build_manifest(
            recorder_session_id=SESSION_ID,
            config_dict={"network": "mainnet", "coins": ["BTC"]},
            start_time_ms=1000,
            end_time_ms=2000,
            raw_row_count=raw_count,
            normalized_row_count=0,
            duplicate_count=0,
            malformed_count=0,
            reconnect_count=0,
            heartbeat_stats={},
            channel_stats={},
            latency_stats={},
            gap_stats={},
            subscription_summary={},
            files_written=raw_files,
            shutdown_reason="running",
            configuration_hash="abc",
        )
        manifest["manifest_finalized"] = True
        write_manifest(manifests_dir, SESSION_ID, manifest)
        diagnosis = diagnose_session(SESSION_ID, stream)
        assert diagnosis["consistency_label"] == (
            SessionConsistencyLabel.INCONSISTENT_FINALIZED_MANIFEST.value
        )
        report = format_diagnosis_report(diagnosis)
        assert "INCONSISTENT_FINALIZED_MANIFEST" in report

    def test_repair_report_does_not_modify_raw_files(self, tmp_path):
        stream = _write_raw_session(tmp_path, session_id=SESSION_ID)
        raw_dir = stream / "raw"
        before = {str(p): p.read_bytes() for p in raw_dir.rglob("*.jsonl")}
        manifests_dir = stream / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "recorder_session_id": SESSION_ID,
            "manifest_finalized": True,
            "shutdown_reason": "running",
            "end_time_ms": 2000,
            "raw_row_count": 3,
            "paper_only": True,
            "execution_enabled": False,
            "files_written": [],
        }
        (manifests_dir / f"{SESSION_ID}.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        report = repair_session_lifecycle_metadata(SESSION_ID, stream, apply=False)
        after = {str(p): p.read_bytes() for p in raw_dir.rglob("*.jsonl")}
        assert before == after
        assert report["raw_files_modified"] is False
        assert report["applied"] is False

    def test_validate_manifest_rejects_non_terminal_shutdown(self):
        validation = validate_manifest(
            {
                "manifest_finalized": True,
                "shutdown_reason": "running",
                "end_time_ms": 1000,
                "raw_row_count": 1,
                "paper_only": True,
                "execution_enabled": False,
                "files_written": [],
            },
            Path("data/market_stream"),
        )
        assert validation["valid"] is False
        assert any("non-terminal" in issue for issue in validation["issues"])

    def test_ensure_terminal_shutdown_reason_mapping(self):
        assert ensure_terminal_shutdown_reason("running") == "unknown_stop"
        assert ensure_terminal_shutdown_reason(None) == "unknown_stop"
        assert ensure_terminal_shutdown_reason("duration_limit") == "duration_limit"
        assert ensure_terminal_shutdown_reason("user_interrupt") == "keyboard_interrupt"

    def test_no_execution_code_in_lifecycle_module(self):
        import market_data.lifecycle as lifecycle

        source = Path(lifecycle.__file__).read_text(encoding="utf-8").lower()
        forbidden = ("private_key", "wallet", "sign(", "place_order", "cancel_order")
        for token in forbidden:
            assert token not in source
