"""Offline tests for the historical L2 download command builder."""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import pytest

from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY
from historical_l2_download_commands import (
    GENERATED_ONLY_WARNING,
    build_command_plan,
    build_decompress_commands,
    build_download_commands,
    build_import_commands,
    build_mkdir_commands,
    discover_latest_manifest,
    load_manifest,
    main,
    write_command_plan_json,
    write_command_plan_markdown,
)

MODULE_FILE = Path(__file__).resolve().parent.parent / "historical_l2_download_commands.py"


def _sample_manifest(*, paper_only: bool = True, execution_enabled: bool = False) -> dict:
    return {
        "generated_at": "2026-01-01T00:00:00Z",
        "paper_only": paper_only,
        "execution_enabled": execution_enabled,
        "start_date": "20230916",
        "end_date": "20230916",
        "coins": ["BTC", "SOL"],
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
        ],
    }


def _write_manifest(tmp_path: Path, manifest: dict, name: str = "historical_l2_backfill_test.json") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


class TestDiscoverLatestManifest:
    def test_discover_latest_manifest_by_mtime(self, tmp_path: Path) -> None:
        older = _write_manifest(tmp_path, _sample_manifest(), "older.json")
        time.sleep(0.01)
        newer = _write_manifest(tmp_path, _sample_manifest(), "newer.json")

        assert discover_latest_manifest(tmp_path) == newer
        assert discover_latest_manifest(tmp_path) != older

    def test_discover_latest_manifest_missing_dir(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing"
        with pytest.raises(FileNotFoundError, match="manifest directory not found"):
            discover_latest_manifest(missing)

    def test_discover_latest_manifest_empty_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="no manifest JSON files found"):
            discover_latest_manifest(tmp_path)


class TestLoadManifest:
    def test_load_manifest_validates_and_loads(self, tmp_path: Path) -> None:
        path = _write_manifest(tmp_path, _sample_manifest())
        loaded = load_manifest(path)

        assert loaded["paper_only"] is True
        assert loaded["execution_enabled"] is False
        assert len(loaded["files"]) == 3
        assert loaded["_source_manifest_path"] == str(path.resolve())

    def test_reject_non_paper_manifest(self, tmp_path: Path) -> None:
        path = _write_manifest(tmp_path, _sample_manifest(paper_only=False))
        with pytest.raises(ValueError, match="paper_only must be true"):
            load_manifest(path)

    def test_reject_execution_enabled_manifest(self, tmp_path: Path) -> None:
        path = _write_manifest(tmp_path, _sample_manifest(execution_enabled=True))
        with pytest.raises(ValueError, match="execution_enabled must be false"):
            load_manifest(path)

    def test_reject_missing_files(self, tmp_path: Path) -> None:
        manifest = _sample_manifest()
        manifest["files"] = []
        path = _write_manifest(tmp_path, manifest)
        with pytest.raises(ValueError, match="files must be a non-empty list"):
            load_manifest(path)

    def test_reject_missing_required_field(self, tmp_path: Path) -> None:
        manifest = _sample_manifest()
        del manifest["files"][0]["coin"]
        path = _write_manifest(tmp_path, manifest)
        with pytest.raises(ValueError, match="missing required field: coin"):
            load_manifest(path)


class TestCommandBuilders:
    @pytest.fixture()
    def manifest(self, tmp_path: Path) -> dict:
        path = _write_manifest(tmp_path, _sample_manifest())
        return load_manifest(path)

    def test_mkdir_command_deduplication(self, manifest: dict) -> None:
        commands = build_mkdir_commands(manifest)
        assert commands == [
            "mkdir -p downloads/hyperliquid/20230916/0",
            "mkdir -p downloads/hyperliquid/20230916/1",
        ]

    def test_download_command_format(self, manifest: dict) -> None:
        commands = build_download_commands(manifest)
        assert commands[0] == (
            "aws s3 cp --no-sign-request "
            "s3://hyperliquid-archive/market_data/20230916/0/l2Book/BTC.lz4 "
            "downloads/hyperliquid/20230916/0/BTC.lz4"
        )

    def test_decompress_command_format(self, manifest: dict) -> None:
        commands = build_decompress_commands(manifest)
        assert commands[0] == (
            "lz4 -d downloads/hyperliquid/20230916/0/BTC.lz4 "
            "downloads/hyperliquid/20230916/0/BTC.jsonl"
        )

    def test_import_command_format(self, manifest: dict) -> None:
        commands = build_import_commands(manifest)
        assert commands[0] == (
            "python3 historical_l2_importer.py import-local "
            "--input downloads/hyperliquid/20230916/0/BTC.jsonl --coin BTC"
        )

    def test_limit_applies_correctly(self, manifest: dict) -> None:
        plan = build_command_plan(manifest, limit=2)
        assert plan["total_files"] == 2
        assert len(plan["command_sections"]["download"]) == 2
        assert len(plan["command_sections"]["decompress"]) == 2
        assert len(plan["command_sections"]["import"]) == 2
        assert plan["command_sections"]["mkdir"] == ["mkdir -p downloads/hyperliquid/20230916/0"]


class TestWriters:
    def test_write_command_plan_json(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path / "manifests", _sample_manifest())
        manifest = load_manifest(manifest_path)
        plan = build_command_plan(manifest)
        output_dir = tmp_path / "commands"
        json_path = write_command_plan_json(plan, output_dir=output_dir)

        assert json_path.exists()
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["paper_only"] is True
        assert loaded["execution_enabled"] is False
        assert loaded["total_files"] == 3
        assert "mkdir" in loaded["command_sections"]

    def test_write_command_plan_markdown(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path / "manifests", _sample_manifest())
        manifest = load_manifest(manifest_path)
        plan = build_command_plan(manifest, limit=2)
        output_dir = tmp_path / "commands"
        markdown_path = write_command_plan_markdown(plan, output_dir=output_dir)
        text = markdown_path.read_text(encoding="utf-8")

        assert markdown_path.exists()
        assert "## Summary" in text
        assert "## Important safety notes" in text
        assert "## Required tools: aws, lz4" in text
        assert "## mkdir commands" in text
        assert "## download commands" in text
        assert "## decompress commands" in text
        assert "## import commands" in text
        assert "## Suggested staged workflow" in text
        assert "## Research-only disclaimer" in text
        assert GENERATED_ONLY_WARNING in text
        assert "aws s3 cp --no-sign-request" in text
        assert "lz4 -d" in text
        assert "historical_l2_importer.py import-local" in text


class TestCli:
    def test_main_with_manifest_and_limit(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        manifest_dir = tmp_path / "manifests"
        output_dir = tmp_path / "commands"
        manifest_path = _write_manifest(manifest_dir, _sample_manifest())

        exit_code = main(
            [
                "--manifest",
                str(manifest_path),
                "--limit",
                "2",
                "--output-dir",
                str(output_dir),
            ]
        )
        captured = capsys.readouterr().out

        assert exit_code == 0
        assert "Files included: 2" in captured
        assert "Total commands:" in captured
        assert GENERATED_ONLY_WARNING in captured
        assert list(output_dir.glob("*.json"))
        assert list(output_dir.glob("*.md"))

    def test_main_with_latest(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        manifest_dir = tmp_path / "manifests"
        output_dir = tmp_path / "commands"
        _write_manifest(manifest_dir, _sample_manifest(), "plan.json")

        import historical_l2_download_commands as module

        original = module.DEFAULT_MANIFEST_DIR
        module.DEFAULT_MANIFEST_DIR = manifest_dir
        try:
            exit_code = main(["--latest", "--limit", "1", "--output-dir", str(output_dir)])
        finally:
            module.DEFAULT_MANIFEST_DIR = original

        captured = capsys.readouterr().out
        assert exit_code == 0
        assert "Files included: 1" in captured


class TestCommandPlan:
    def test_paper_only_and_execution_disabled(self, tmp_path: Path) -> None:
        manifest = load_manifest(_write_manifest(tmp_path, _sample_manifest()))
        plan = build_command_plan(manifest)
        assert plan["paper_only"] is True
        assert plan["execution_enabled"] is False
        assert plan["warning"] == GENERATED_ONLY_WARNING
        assert plan["total_commands"] == 2 + 3 + 3 + 3


class TestSafety:
    def test_no_network_calls(self) -> None:
        source = MODULE_FILE.read_text(encoding="utf-8").lower()
        for token in (
            "requests.",
            "urllib.",
            "http.client",
            "socket.",
            "websocket",
            "aiohttp",
            "httpx",
            "boto3",
            "botocore",
        ):
            assert token not in source

    def test_no_subprocess_usage(self) -> None:
        source = MODULE_FILE.read_text(encoding="utf-8").lower()
        assert "subprocess" not in source
        assert "os.system" not in source

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

    def test_constants_match_schema(self) -> None:
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False
