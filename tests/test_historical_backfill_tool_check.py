"""Offline tests for historical backfill tool availability diagnostics."""

from __future__ import annotations

import ast
import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from bid_ask_schema import FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS
from historical_backfill_tool_check import (
    ALL_TIME_BBO_NOT_CLAIMED,
    BREW_MISSING_WARNING,
    BULK_DOWNLOAD_WARNING,
    check_backfill_tools,
    check_tool_available,
    main,
    print_tool_check,
)

TOOL_CHECK_FILE = (
    Path(__file__).resolve().parent.parent / "historical_backfill_tool_check.py"
)
DOCS_FILE = Path(__file__).resolve().parent.parent / "docs" / "historical_backfill_plan.md"


def _patch_which(monkeypatch: pytest.MonkeyPatch, available: set[str]) -> None:
    def fake_which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in available else None

    monkeypatch.setattr("historical_backfill_tool_check.shutil.which", fake_which)


class TestCheckToolAvailable:
    def test_returns_true_when_tool_on_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_which(monkeypatch, {"aws"})
        assert check_tool_available("aws") is True

    def test_returns_false_when_tool_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_which(monkeypatch, set())
        assert check_tool_available("aws") is False


class TestCheckBackfillTools:
    def test_all_tools_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_which(monkeypatch, {"brew", "aws", "lz4"})
        result = check_backfill_tools()

        assert result["brew_available"] is True
        assert result["aws_available"] is True
        assert result["lz4_available"] is True
        assert result["ready_for_manual_download"] is True
        assert result["missing_tools"] == []
        assert BREW_MISSING_WARNING not in result["warnings"]
        assert BULK_DOWNLOAD_WARNING in result["warnings"]
        assert ALL_TIME_BBO_NOT_CLAIMED in result["warnings"]

    def test_aws_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_which(monkeypatch, {"brew", "lz4"})
        result = check_backfill_tools()

        assert result["aws_available"] is False
        assert result["ready_for_manual_download"] is False
        assert result["missing_tools"] == ["aws"]

    def test_lz4_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_which(monkeypatch, {"brew", "aws"})
        result = check_backfill_tools()

        assert result["lz4_available"] is False
        assert result["ready_for_manual_download"] is False
        assert result["missing_tools"] == ["lz4"]

    def test_brew_missing_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_which(monkeypatch, {"aws", "lz4"})
        result = check_backfill_tools()

        assert result["brew_available"] is False
        assert result["ready_for_manual_download"] is True
        assert BREW_MISSING_WARNING in result["warnings"]

    def test_ready_false_when_aws_and_lz4_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_which(monkeypatch, {"brew"})
        result = check_backfill_tools()

        assert result["ready_for_manual_download"] is False
        assert set(result["missing_tools"]) == {"aws", "lz4"}


class TestPrintToolCheck:
    def test_cli_prints_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_which(monkeypatch, set())
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_tool_check()
        output = buffer.getvalue()

        assert "Historical Backfill Tool Check" in output
        assert "brew: missing" in output
        assert "aws: missing" in output
        assert "lz4: missing" in output
        assert "ready_for_manual_download: false" in output
        assert "missing_tools: aws, lz4" in output
        assert "Next: Install required tools manually" in output
        assert BULK_DOWNLOAD_WARNING in output
        assert ALL_TIME_BBO_NOT_CLAIMED in output

    def test_main_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_which(monkeypatch, {"aws", "lz4"})
        assert main() == 0


class TestDocs:
    def test_docs_include_required_local_tools(self) -> None:
        docs = DOCS_FILE.read_text(encoding="utf-8")
        assert "## Required Local Tools" in docs
        assert "aws CLI" in docs
        assert "lz4" in docs
        assert "no automatic installs" in docs.lower()
        assert "no automatic downloads" in docs.lower()


class TestSafety:
    def test_no_network_calls(self) -> None:
        source = TOOL_CHECK_FILE.read_text(encoding="utf-8").lower()
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
        source = TOOL_CHECK_FILE.read_text(encoding="utf-8").lower()
        assert "subprocess" not in source
        assert "os.system" not in source

    def test_no_wallet_signing_order_execution_code(self) -> None:
        source = TOOL_CHECK_FILE.read_text(encoding="utf-8")
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
