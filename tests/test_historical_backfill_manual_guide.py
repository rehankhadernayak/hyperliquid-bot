"""Offline tests for the manual historical backfill guide."""

from __future__ import annotations

import ast
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from bid_ask_schema import FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS
from historical_backfill_manual_guide import (
    ALL_TIME_BBO_NOT_CLAIMED,
    NOTHING_EXECUTED_WARNING,
    build_backfill_step_plan,
    build_manual_decompress_command,
    build_manual_download_command,
    build_manual_import_command,
    build_manual_validation_command,
    main,
    print_backfill_step_plan,
)
from historical_l2_importer import (
    build_archive_s3_path,
    default_historical_output_path,
)

GUIDE_FILE = Path(__file__).resolve().parent.parent / "historical_backfill_manual_guide.py"
DOCS_FILE = Path(__file__).resolve().parent.parent / "docs" / "historical_backfill_plan.md"
BASE_TS = 1_718_000_000_000


def _l2_archive_line(*, time_ms: int = BASE_TS) -> str:
    payload = {
        "channel": "l2Book",
        "data": {
            "coin": "SOL",
            "time": time_ms,
            "levels": [
                [{"px": "150.0", "sz": "10.0", "n": 2}],
                [{"px": "150.5", "sz": "8.0", "n": 3}],
            ],
        },
    }
    return json.dumps(payload)


class TestManualCommands:
    def test_build_manual_download_command(self) -> None:
        command = build_manual_download_command("20230916", 9, "SOL")
        assert command == (
            "aws s3 cp --no-sign-request "
            "s3://hyperliquid-archive/market_data/20230916/9/l2Book/SOL.lz4 "
            "./downloads/hyperliquid/20230916/9/SOL.lz4"
        )

    def test_build_manual_decompress_command(self) -> None:
        command = build_manual_decompress_command("20230916", 9, "SOL")
        assert command == (
            "lz4 -d ./downloads/hyperliquid/20230916/9/SOL.lz4 "
            "./downloads/hyperliquid/20230916/9/SOL.jsonl"
        )

    def test_build_manual_import_command(self) -> None:
        command = build_manual_import_command("20230916", 9, "SOL")
        assert command == (
            "python3 historical_l2_importer.py import-local "
            "--input ./downloads/hyperliquid/20230916/9/SOL.jsonl --coin SOL"
        )

    def test_build_manual_validation_command_uses_historical_summarize(self) -> None:
        command = build_manual_validation_command("SOL")
        expected_csv = default_historical_output_path("SOL")
        assert command == f"python3 historical_l2_importer.py summarize --csv {expected_csv}"


class TestStepPlan:
    def test_step_plan_ordering(self) -> None:
        steps = build_backfill_step_plan("20230916", 9, "SOL")
        actions = [step.action for step in steps]
        assert actions == [
            "inspect archive path",
            "manually download",
            "manually decompress",
            "import local decompressed file",
            "validate generated CSV",
            "review report",
            "resample if valid",
        ]
        assert [step.order for step in steps] == list(range(1, 8))

    def test_step_plan_includes_commands_for_executable_steps(self) -> None:
        steps = build_backfill_step_plan("20230916", 9, "SOL")
        inspect_step = steps[0]
        download_step = steps[1]
        assert "inspect-path" in (inspect_step.command or "")
        assert "aws s3 cp" in (download_step.command or "")
        assert steps[5].command is None
        assert steps[6].command is None


class TestCliPrintsWithoutExecuting:
    def test_main_prints_commands_without_executing(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(["--date", "20230916", "--hour", "9", "--coin", "SOL"])
        captured = capsys.readouterr().out

        assert exit_code == 0
        assert build_archive_s3_path("20230916", 9, "SOL") in captured
        assert build_manual_download_command("20230916", 9, "SOL") in captured
        assert build_manual_decompress_command("20230916", 9, "SOL") in captured
        assert build_manual_import_command("20230916", 9, "SOL") in captured
        assert build_manual_validation_command("SOL") in captured
        assert NOTHING_EXECUTED_WARNING in captured
        assert ALL_TIME_BBO_NOT_CLAIMED in captured

    def test_print_backfill_step_plan_writes_to_stdout(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_backfill_step_plan("20230916", 9, "SOL")
        output = buffer.getvalue()
        assert "Manual download command:" in output
        assert "Validation command:" in output
        assert "Step plan:" in output


class TestDocs:
    def test_docs_include_manual_phase_2_workflow(self) -> None:
        docs = DOCS_FILE.read_text(encoding="utf-8")
        assert "## Manual Phase 2 Workflow" in docs
        assert "Why manual first" in docs
        assert "historical_backfill_manual_guide.py" in docs
        assert "historical_l2_importer.py summarize" in docs
        assert "historical_1s" in docs


class TestSafety:
    def test_no_network_calls(self) -> None:
        source = GUIDE_FILE.read_text(encoding="utf-8").lower()
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
        source = GUIDE_FILE.read_text(encoding="utf-8").lower()
        assert "subprocess" not in source
        assert "os.system" not in source

    def test_no_wallet_signing_order_execution_code(self) -> None:
        source = GUIDE_FILE.read_text(encoding="utf-8")
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
