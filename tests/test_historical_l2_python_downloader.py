"""Offline tests for the Python-native historical L2 sample downloader."""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import lz4.frame
import pytest

from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY
from historical_l2_python_downloader import (
    DEFAULT_MAX_FILES,
    HARD_CAP_MAX_FILES,
    HARD_CAP_MAX_FILES_ALLOW_MORE,
    classify_download_error,
    create_unsigned_s3_client,
    decompress_one_file,
    discover_latest_command_plan,
    download_one_file,
    load_plan,
    main,
    parse_s3_path,
    resolve_max_files,
    run_sample_download,
    select_plan_files,
    write_report_json,
    write_report_markdown,
)

MODULE_FILE = Path(__file__).resolve().parent.parent / "historical_l2_python_downloader.py"


def _sample_plan(*, paper_only: bool = True, execution_enabled: bool = False) -> dict:
    return {
        "generated_at": "2026-01-01T00:00:00Z",
        "paper_only": paper_only,
        "execution_enabled": execution_enabled,
        "files": [
            {
                "date": "20230916",
                "hour": "0",
                "coin": "BTC",
                "s3_path": "s3://hyperliquid-archive/market_data/20230916/0/l2Book/BTC.lz4",
                "local_lz4_path": "downloads/hyperliquid/20230916/0/BTC.lz4",
                "local_jsonl_path": "downloads/hyperliquid/20230916/0/BTC.jsonl",
                "status": "planned",
            },
            {
                "date": "20230916",
                "hour": "0",
                "coin": "ETH",
                "s3_path": "s3://hyperliquid-archive/market_data/20230916/0/l2Book/ETH.lz4",
                "local_lz4_path": "downloads/hyperliquid/20230916/0/ETH.lz4",
                "local_jsonl_path": "downloads/hyperliquid/20230916/0/ETH.jsonl",
                "status": "planned",
            },
            {
                "date": "20230916",
                "hour": "0",
                "coin": "SOL",
                "s3_path": "s3://hyperliquid-archive/market_data/20230916/0/l2Book/SOL.lz4",
                "local_lz4_path": "downloads/hyperliquid/20230916/0/SOL.lz4",
                "local_jsonl_path": "downloads/hyperliquid/20230916/0/SOL.jsonl",
                "status": "planned",
            },
            {
                "date": "20230916",
                "hour": "1",
                "coin": "BTC",
                "s3_path": "s3://hyperliquid-archive/market_data/20230916/1/l2Book/BTC.lz4",
                "local_lz4_path": "downloads/hyperliquid/20230916/1/BTC.lz4",
                "local_jsonl_path": "downloads/hyperliquid/20230916/1/BTC.jsonl",
                "status": "planned",
            },
            {
                "date": "20230916",
                "hour": "9",
                "coin": "SOL",
                "s3_path": "s3://hyperliquid-archive/market_data/20230916/9/l2Book/SOL.lz4",
                "local_lz4_path": "downloads/hyperliquid/20230916/9/SOL.lz4",
                "local_jsonl_path": "downloads/hyperliquid/20230916/9/SOL.jsonl",
                "status": "planned",
            },
        ],
    }


def _write_plan(tmp_path: Path, plan: dict, name: str = "plan.json") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return path


def _make_fake_lz4(content: bytes = b'{"test": true}\n') -> bytes:
    return lz4.frame.compress(content)


def _make_fake_get_object_response(payload: bytes) -> dict:
    body = MagicMock()
    body.iter_chunks.return_value = [payload]
    body.close = MagicMock()
    return {"Body": body}


class TestParseS3Path:
    def test_valid_path(self) -> None:
        bucket, key = parse_s3_path(
            "s3://hyperliquid-archive/market_data/20230916/0/l2Book/BTC.lz4"
        )
        assert bucket == "hyperliquid-archive"
        assert key == "market_data/20230916/0/l2Book/BTC.lz4"

    def test_reject_non_s3(self) -> None:
        with pytest.raises(ValueError, match="must start with s3://"):
            parse_s3_path("https://example.com/file.lz4")

    def test_reject_missing_key(self) -> None:
        with pytest.raises(ValueError, match="missing key"):
            parse_s3_path("s3://bucket-only")


class TestUnsignedClient:
    @patch("historical_l2_python_downloader.boto3.client")
    def test_create_unsigned_s3_client(self, mock_boto_client: MagicMock) -> None:
        create_unsigned_s3_client()
        mock_boto_client.assert_called_once()
        call_kwargs = mock_boto_client.call_args
        assert call_kwargs[0][0] == "s3"
        assert call_kwargs[1]["config"] is not None


class TestLoadPlan:
    def test_load_plan_validates_and_loads(self, tmp_path: Path) -> None:
        path = _write_plan(tmp_path, _sample_plan())
        loaded = load_plan(path)
        assert loaded["paper_only"] is True
        assert loaded["execution_enabled"] is False
        assert len(loaded["files"]) == 4
        assert loaded["_source_plan_path"] == str(path.resolve())

    def test_reject_non_paper_plan(self, tmp_path: Path) -> None:
        path = _write_plan(tmp_path, _sample_plan(paper_only=False))
        with pytest.raises(ValueError, match="paper_only must be true"):
            load_plan(path)

    def test_reject_execution_enabled_plan(self, tmp_path: Path) -> None:
        path = _write_plan(tmp_path, _sample_plan(execution_enabled=True))
        with pytest.raises(ValueError, match="execution_enabled must be false"):
            load_plan(path)

    def test_reject_missing_files(self, tmp_path: Path) -> None:
        plan = _sample_plan()
        plan["files"] = []
        path = _write_plan(tmp_path, plan)
        with pytest.raises(ValueError, match="files must be a non-empty list"):
            load_plan(path)

    def test_reject_missing_required_field(self, tmp_path: Path) -> None:
        plan = _sample_plan()
        del plan["files"][0]["coin"]
        path = _write_plan(tmp_path, plan)
        with pytest.raises(ValueError, match="missing required field: coin"):
            load_plan(path)


class TestDiscoverLatestCommandPlan:
    def test_discover_latest_by_mtime(self, tmp_path: Path) -> None:
        older = _write_plan(tmp_path, _sample_plan(), "older.json")
        time.sleep(0.01)
        newer = _write_plan(tmp_path, _sample_plan(), "newer.json")
        assert discover_latest_command_plan(command_plan_dir=tmp_path) == newer

    def test_discover_latest_missing_dir(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing"
        with pytest.raises(FileNotFoundError, match="no command or backfill plan"):
            discover_latest_command_plan(command_plan_dir=missing)


class TestMaxFilesCaps:
    def test_default_max_files_is_one(self) -> None:
        assert DEFAULT_MAX_FILES == 1
        assert resolve_max_files(None) == 1

    def test_hard_cap_three_without_allow_more(self) -> None:
        assert resolve_max_files(1) == 1
        assert resolve_max_files(3) == 3
        with pytest.raises(ValueError, match="exceeds hard cap of 3"):
            resolve_max_files(4)

    def test_hard_cap_twenty_four_with_allow_more(self) -> None:
        assert resolve_max_files(10, allow_more=True) == 10
        assert resolve_max_files(HARD_CAP_MAX_FILES_ALLOW_MORE, allow_more=True) == 24
        with pytest.raises(ValueError, match="exceeds hard cap of 24"):
            resolve_max_files(25, allow_more=True)

    def test_reject_zero_max_files(self) -> None:
        with pytest.raises(ValueError, match="must be at least 1"):
            resolve_max_files(0)


class TestSelectPlanFiles:
    def test_filter_by_coin_date_hour(self) -> None:
        files = _sample_plan()["files"]
        selected = select_plan_files(files, coin="SOL", date="20230916", hour="9")
        assert len(selected) == 1
        assert selected[0]["coin"] == "SOL"
        assert selected[0]["hour"] == "9"

    def test_filter_by_coin_only(self) -> None:
        files = _sample_plan()["files"]
        selected = select_plan_files(files, coin="BTC")
        assert len(selected) == 2
        assert all(entry["coin"] == "BTC" for entry in selected)


class TestClassifyDownloadError:
    def test_403_client_error_is_forbidden(self) -> None:
        from botocore.exceptions import ClientError

        exc = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}, "ResponseMetadata": {"HTTPStatusCode": 403}},
            "GetObject",
        )
        assert classify_download_error(exc) == "forbidden"

    def test_404_client_error_is_not_found(self) -> None:
        from botocore.exceptions import ClientError

        exc = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}, "ResponseMetadata": {"HTTPStatusCode": 404}},
            "GetObject",
        )
        assert classify_download_error(exc) == "not_found"

    def test_connection_error_is_network_error(self) -> None:
        assert classify_download_error(ConnectionError("timeout")) == "network_error"

    def test_unknown_error(self) -> None:
        assert classify_download_error(RuntimeError("boom")) == "unknown_error"


class TestDownloadOneFile:
    def test_download_skip_existing(self, tmp_path: Path) -> None:
        entry = _sample_plan()["files"][0]
        entry = dict(entry)
        entry["local_lz4_path"] = str(tmp_path / "BTC.lz4")
        (tmp_path / "BTC.lz4").write_bytes(b"existing")

        mock_client = MagicMock()
        result = download_one_file(mock_client, entry, overwrite=False)

        assert result["download_status"] == "skipped_existing"
        mock_client.get_object.assert_not_called()
        mock_client.download_file.assert_not_called()

    def test_download_success_via_get_object_streaming(self, tmp_path: Path) -> None:
        entry = _sample_plan()["files"][0]
        entry = dict(entry)
        lz4_path = tmp_path / "BTC.lz4"
        entry["local_lz4_path"] = str(lz4_path)

        mock_client = MagicMock()
        mock_client.get_object.return_value = _make_fake_get_object_response(b"fake-lz4-data")
        result = download_one_file(mock_client, entry, overwrite=False)

        assert result["download_status"] == "downloaded"
        assert lz4_path.read_bytes() == b"fake-lz4-data"
        mock_client.get_object.assert_called_once_with(
            Bucket="hyperliquid-archive",
            Key="market_data/20230916/0/l2Book/BTC.lz4",
        )
        mock_client.download_file.assert_not_called()
        mock_client.download_fileobj.assert_not_called()

    def test_403_classified_as_forbidden(self, tmp_path: Path) -> None:
        from botocore.exceptions import ClientError

        entry = _sample_plan()["files"][0]
        entry = dict(entry)
        entry["local_lz4_path"] = str(tmp_path / "BTC.lz4")

        mock_client = MagicMock()
        mock_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}, "ResponseMetadata": {"HTTPStatusCode": 403}},
            "GetObject",
        )
        result = download_one_file(mock_client, entry, overwrite=False)

        assert result["download_status"] == "failed"
        assert result["download_error_type"] == "forbidden"
        assert not Path(entry["local_lz4_path"]).exists()

    def test_module_does_not_use_download_file_or_head_object(self) -> None:
        source = MODULE_FILE.read_text(encoding="utf-8")
        assert "download_file" not in source
        assert "download_fileobj" not in source
        assert "head_object" not in source.lower()


class TestDecompressOneFile:
    def test_decompress_skip_existing(self, tmp_path: Path) -> None:
        entry = _sample_plan()["files"][0]
        entry = dict(entry)
        entry["local_lz4_path"] = str(tmp_path / "BTC.lz4")
        entry["local_jsonl_path"] = str(tmp_path / "BTC.jsonl")
        (tmp_path / "BTC.jsonl").write_text('{"existing": true}\n')

        result = decompress_one_file(entry, overwrite=False)
        assert result["decompress_status"] == "skipped_existing"

    def test_decompress_success(self, tmp_path: Path) -> None:
        entry = _sample_plan()["files"][0]
        entry = dict(entry)
        lz4_path = tmp_path / "BTC.lz4"
        jsonl_path = tmp_path / "BTC.jsonl"
        entry["local_lz4_path"] = str(lz4_path)
        entry["local_jsonl_path"] = str(jsonl_path)

        payload = b'{"coin":"BTC"}\n'
        lz4_path.write_bytes(_make_fake_lz4(payload))

        result = decompress_one_file(entry, overwrite=False)
        assert result["decompress_status"] == "decompressed"
        assert jsonl_path.read_bytes() == payload


class TestRunSampleDownload:
    def test_run_sample_download_with_mock_client(self, tmp_path: Path) -> None:
        plan_path = _write_plan(tmp_path, _sample_plan())
        plan = load_plan(plan_path)

        for entry in plan["files"][:2]:
            entry = dict(entry)
            entry["local_lz4_path"] = str(tmp_path / Path(entry["local_lz4_path"]).name)
            entry["local_jsonl_path"] = str(tmp_path / Path(entry["local_jsonl_path"]).name)
        plan["files"] = [
            {
                **_sample_plan()["files"][0],
                "local_lz4_path": str(tmp_path / "BTC.lz4"),
                "local_jsonl_path": str(tmp_path / "BTC.jsonl"),
            }
        ]

        mock_client = MagicMock()

        def _fake_get_object(*, Bucket: str, Key: str) -> dict:
            payload = _make_fake_lz4(b'{"coin":"BTC"}\n')
            return _make_fake_get_object_response(payload)

        mock_client.get_object.side_effect = _fake_get_object

        report = run_sample_download(plan, max_files=1, client=mock_client)

        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["requested_files"] == 1
        assert report["downloaded_count"] == 1
        assert report["decompressed_count"] == 1
        assert report["failed_count"] == 0
        assert len(report["next_import_commands"]) == 1
        assert "historical_l2_importer.py import-local" in report["next_import_commands"][0]
        entry = report["file_results"][0]
        assert entry["s3_path"] == plan["files"][0]["s3_path"]
        assert entry["download_status"] == "downloaded"
        assert entry["decompress_status"] == "decompressed"

    def test_failed_download_skips_decompression(self, tmp_path: Path) -> None:
        from botocore.exceptions import ClientError

        plan_path = _write_plan(tmp_path, _sample_plan())
        plan = load_plan(plan_path)
        plan["files"] = [
            {
                **_sample_plan()["files"][0],
                "local_lz4_path": str(tmp_path / "BTC.lz4"),
                "local_jsonl_path": str(tmp_path / "BTC.jsonl"),
            }
        ]

        mock_client = MagicMock()
        mock_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}, "ResponseMetadata": {"HTTPStatusCode": 403}},
            "GetObject",
        )

        report = run_sample_download(plan, max_files=1, client=mock_client)

        assert report["failed_count"] == 1
        assert report["decompressed_count"] == 0
        entry = report["file_results"][0]
        assert entry["download_status"] == "failed"
        assert entry["download_error_type"] == "forbidden"
        assert entry["decompress_status"] == "skipped"
        assert entry["decompress_error"] == "download_failed"
        assert not Path(plan["files"][0]["local_jsonl_path"]).exists()

    def test_failed_count_per_file_not_per_stage(self, tmp_path: Path) -> None:
        from botocore.exceptions import ClientError

        plan_path = _write_plan(tmp_path, _sample_plan())
        plan = load_plan(plan_path)
        plan["files"] = [
            {
                **_sample_plan()["files"][0],
                "local_lz4_path": str(tmp_path / "BTC.lz4"),
                "local_jsonl_path": str(tmp_path / "BTC.jsonl"),
            }
        ]

        mock_client = MagicMock()
        mock_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}, "ResponseMetadata": {"HTTPStatusCode": 403}},
            "GetObject",
        )

        report = run_sample_download(plan, max_files=1, client=mock_client)
        assert report["failed_count"] <= 1
        assert report["failed_count"] == 1


class TestWriters:
    def test_write_report_json(self, tmp_path: Path) -> None:
        report = {
            "generated_at": "2026-01-01T00:00:00Z",
            "paper_only": True,
            "execution_enabled": False,
            "source_plan_path": "/tmp/plan.json",
            "requested_files": 1,
            "downloaded_count": 1,
            "decompressed_count": 1,
            "skipped_count": 0,
            "failed_count": 0,
            "file_results": [],
            "warnings": [],
            "next_import_commands": [],
        }
        output_dir = tmp_path / "reports"
        json_path = write_report_json(report, output_dir=output_dir)

        assert json_path.exists()
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["paper_only"] is True
        assert loaded["execution_enabled"] is False

    def test_write_report_markdown(self, tmp_path: Path) -> None:
        report = {
            "generated_at": "2026-01-01T00:00:00Z",
            "paper_only": True,
            "execution_enabled": False,
            "source_plan_path": "/tmp/plan.json",
            "requested_files": 1,
            "downloaded_count": 1,
            "decompressed_count": 1,
            "skipped_count": 0,
            "failed_count": 0,
            "file_results": [
                {
                    "coin": "BTC",
                    "date": "20230916",
                    "hour": "0",
                    "s3_path": "s3://hyperliquid-archive/market_data/20230916/0/l2Book/BTC.lz4",
                    "download_status": "downloaded",
                    "decompress_status": "decompressed",
                }
            ],
            "warnings": ["sample warning"],
            "next_import_commands": [
                "python3 historical_l2_importer.py import-local --input x --coin BTC"
            ],
        }
        output_dir = tmp_path / "reports"
        md_path = write_report_markdown(report, output_dir=output_dir)
        text = md_path.read_text(encoding="utf-8")

        assert md_path.exists()
        assert "## Summary" in text
        assert "## File results" in text
        assert "## Failures" in text
        assert "## Next import commands" in text
        assert "## Limitations" in text
        assert "## Research-only disclaimer" in text
        assert "s3_path=" in text


class TestCli:
    @patch("historical_l2_python_downloader.create_unsigned_s3_client")
    def test_main_with_plan(
        self,
        mock_create_client: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan = _sample_plan()
        plan["files"] = [
            {
                **plan["files"][0],
                "local_lz4_path": str(tmp_path / "BTC.lz4"),
                "local_jsonl_path": str(tmp_path / "BTC.jsonl"),
            }
        ]
        plan_path = _write_plan(tmp_path / "plans", plan)
        report_dir = tmp_path / "reports"

        mock_client = MagicMock()
        mock_client.get_object.return_value = _make_fake_get_object_response(
            _make_fake_lz4(b'{"coin":"BTC"}\n')
        )
        mock_create_client.return_value = mock_client

        exit_code = main(
            [
                "--plan",
                str(plan_path),
                "--max-files",
                "1",
                "--report-dir",
                str(report_dir),
            ]
        )
        captured = capsys.readouterr().out

        assert exit_code == 0
        assert "Downloaded: 1" in captured
        assert "Decompressed: 1" in captured
        assert "Failed: 0" in captured
        assert "Selected file:" in captured
        assert "coin=BTC" in captured
        assert "s3_path=" in captured
        assert "download_status=downloaded" in captured
        assert "decompress_status=decompressed" in captured
        assert list(report_dir.glob("*.json"))
        assert list(report_dir.glob("*.md"))

    @patch("historical_l2_python_downloader.create_unsigned_s3_client")
    def test_main_selector_by_coin_date_hour(
        self,
        mock_create_client: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan = _sample_plan()
        sol_entry = next(entry for entry in plan["files"] if entry["coin"] == "SOL" and entry["hour"] == "9")
        plan["files"] = [
            {
                **sol_entry,
                "local_lz4_path": str(tmp_path / "SOL.lz4"),
                "local_jsonl_path": str(tmp_path / "SOL.jsonl"),
            },
            *_sample_plan()["files"][:3],
        ]
        plan_path = _write_plan(tmp_path / "plans", plan)
        report_dir = tmp_path / "reports"

        mock_client = MagicMock()
        mock_client.get_object.return_value = _make_fake_get_object_response(
            _make_fake_lz4(b'{"coin":"SOL"}\n')
        )
        mock_create_client.return_value = mock_client

        exit_code = main(
            [
                "--plan",
                str(plan_path),
                "--coin",
                "SOL",
                "--date",
                "20230916",
                "--hour",
                "9",
                "--max-files",
                "1",
                "--report-dir",
                str(report_dir),
            ]
        )
        captured = capsys.readouterr().out

        assert exit_code == 0
        assert "coin=SOL" in captured
        assert "hour=9" in captured
        assert "market_data/20230916/9/l2Book/SOL.lz4" in captured
        mock_client.get_object.assert_called_once_with(
            Bucket="hyperliquid-archive",
            Key="market_data/20230916/9/l2Book/SOL.lz4",
        )
        report = json.loads(list(report_dir.glob("*.json"))[0].read_text(encoding="utf-8"))
        assert report["selector"] == {"coin": "SOL", "date": "20230916", "hour": "9"}
        assert report["file_results"][0]["coin"] == "SOL"

    def test_main_rejects_over_cap(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        plan_path = _write_plan(tmp_path, _sample_plan())
        exit_code = main(["--plan", str(plan_path), "--max-files", "5"])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert "exceeds hard cap" in captured.err


class TestSafety:
    def test_no_subprocess_calls(self) -> None:
        source = MODULE_FILE.read_text(encoding="utf-8").lower()
        assert "subprocess" not in source
        assert "os.system" not in source

    def test_no_private_credential_use(self) -> None:
        source = MODULE_FILE.read_text(encoding="utf-8").lower()
        for token in (
            "aws_access_key",
            "aws_secret",
            "private_key",
            "getenv(",
            "environ[",
            "dotenv",
            "credentials",
        ):
            assert token not in source

    def test_no_wallet_signing_order_execution_code(self) -> None:
        source = MODULE_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        lowered = source.lower()
        for pattern in FORBIDDEN_SOURCE_PATTERNS:
            assert pattern not in lowered
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in FORBIDDEN_MODULE_NAMES

    def test_paper_only_and_execution_disabled(self) -> None:
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False
        assert HARD_CAP_MAX_FILES == 3
        assert HARD_CAP_MAX_FILES_ALLOW_MORE == 24
