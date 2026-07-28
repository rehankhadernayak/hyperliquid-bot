"""Tests for local archive integrity inspection and quarantine recovery."""

from __future__ import annotations

import json
from pathlib import Path

import lz4.frame

from historical_l2_archive_integrity import (
    ensure_local_archive_integrity,
    inspect_jsonl_bytes,
    inspect_local_archive_pair,
    quarantine_local_archive_files,
)
from tests.test_historical_l2_importer import _l2_archive_line


def _make_lz4(payload: bytes) -> bytes:
    return lz4.frame.compress(payload)


class TestInspectJsonlBytes:
    def test_null_bytes_fail(self) -> None:
        payload = b'{"ok": true}\n' + b"\x00" * 32
        report = inspect_jsonl_bytes(payload)
        assert report.ok is False
        assert report.null_byte_count > 0

    def test_truncated_tail_fails(self) -> None:
        payload = b'{"channel":"l2Book","data":{"coin":"BTC","time":1,"levels":[[],[]]}}\n{"trunc'
        report = inspect_jsonl_bytes(payload)
        assert report.ok is False
        assert report.truncated_tail is True

    def test_valid_payload_passes(self) -> None:
        line = _l2_archive_line(coin="BTC", time_ms=1_700_000_000_000)
        report = inspect_jsonl_bytes((line + "\n").encode("utf-8"))
        assert report.ok is True


class TestInspectLocalArchivePair:
    def test_corrupt_lz4_stream(self, tmp_path: Path) -> None:
        lz4_path = tmp_path / "BTC.lz4"
        jsonl_path = tmp_path / "BTC.jsonl"
        lz4_path.write_bytes(b"not-lz4")
        jsonl_path.write_text(_l2_archive_line(coin="BTC") + "\n")

        report = inspect_local_archive_pair(lz4_path=lz4_path, jsonl_path=jsonl_path)
        assert report.ok is False
        assert report.lz4_decompress_ok is False

    def test_valid_existing_file_untouched(self, tmp_path: Path) -> None:
        line = _l2_archive_line(coin="BTC")
        payload = (line + "\n").encode("utf-8")
        lz4_path = tmp_path / "BTC.lz4"
        jsonl_path = tmp_path / "BTC.jsonl"
        lz4_path.write_bytes(_make_lz4(payload))
        jsonl_path.write_bytes(payload)

        before = jsonl_path.read_bytes()
        report = inspect_local_archive_pair(lz4_path=lz4_path, jsonl_path=jsonl_path)
        assert report.ok is True
        assert jsonl_path.read_bytes() == before


class TestQuarantineRecovery:
    def test_corrupt_jsonl_quarantined_and_eligible_for_redownload(self, tmp_path: Path) -> None:
        quarantine_root = tmp_path / "quarantine"
        downloads = tmp_path / "downloads" / "20230929" / "9"
        downloads.mkdir(parents=True)
        lz4_path = downloads / "BTC.lz4"
        jsonl_path = downloads / "BTC.jsonl"
        good_line = _l2_archive_line(coin="BTC")
        good_payload = (good_line + "\n").encode("utf-8")
        lz4_path.write_bytes(_make_lz4(good_payload))
        jsonl_path.write_bytes(good_payload + b"\x00" * 40)

        entry = {
            "coin": "BTC",
            "date": "20230929",
            "hour": "9",
            "local_lz4_path": str(lz4_path),
            "local_jsonl_path": str(jsonl_path),
        }
        result = ensure_local_archive_integrity(entry, quarantine_root=quarantine_root)
        assert result["integrity_ok"] is False
        assert result["quarantine"]["quarantined"] is True
        assert not jsonl_path.is_file()
        assert lz4_path.is_file()

    def test_no_destructive_deletion_of_unrelated_files(self, tmp_path: Path) -> None:
        neighbor = tmp_path / "neighbor.jsonl"
        neighbor.write_text("keep\n")
        corrupt = tmp_path / "BTC.jsonl"
        corrupt.write_bytes(b"\x00")

        quarantine_local_archive_files(
            lz4_path=None,
            jsonl_path=corrupt,
            reasons=["null bytes"],
            quarantine_root=tmp_path / "quarantine",
            paths_to_move=[corrupt],
        )
        assert neighbor.read_text() == "keep\n"
        assert not corrupt.is_file()

    def test_invalid_integer_token_in_jsonl_line(self) -> None:
        bad_line = json.dumps(
            {
                "channel": "l2Book",
                "data": {
                    "coin": "BTC",
                    "time": "not-an-int",
                    "levels": [[{"px": "1", "sz": "1"}], [{"px": "2", "sz": "1"}]],
                },
            }
        )
        report = inspect_jsonl_bytes((bad_line + "\n").encode("utf-8"), max_lines_to_scan=5)
        assert report.ok is False
        assert report.malformed_line_count >= 1

    def test_invalid_floating_point_token_in_jsonl_line(self) -> None:
        bad_line = json.dumps(
            {
                "channel": "l2Book",
                "data": {
                    "coin": "ETH",
                    "time": "0.40000000000009095",
                    "levels": [[{"px": "1", "sz": "1"}], [{"px": "2", "sz": "1"}]],
                },
            }
        )
        report = inspect_jsonl_bytes((bad_line + "\n").encode("utf-8"), max_lines_to_scan=5)
        assert report.ok is False
