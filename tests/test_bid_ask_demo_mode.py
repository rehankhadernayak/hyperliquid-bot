"""Tests for standalone bid/ask demo mode packet preparer."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from bid_ask_schema import (
    EXECUTION_ENABLED,
    FORBIDDEN_MODULE_NAMES,
    FORBIDDEN_SOURCE_PATTERNS,
    PAPER_ONLY,
)
from bid_ask_demo_mode import (
    DASHBOARD_LAUNCH_COMMAND,
    DEFAULT_OUTPUT_DIR,
    build_demo_packet,
    discover_artifacts,
    main,
    prepare_demo_mode,
    validate_output_dir,
    write_demo_packet_markdown,
)

DEMO_MODE_FILE = Path(__file__).resolve().parent.parent / "bid_ask_demo_mode.py"


def _write_phase3_audit(reports_root: Path) -> Path:
    path = reports_root / "bid_ask_research_scheduler" / "phase3_audit_latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2026-07-07T07:16:59Z",
        "paper_only": True,
        "execution_enabled": False,
        "downloads": {"total_files": 42},
        "imported_historical_1s": {
            "BTC": {
                "row_count": 1000,
                "last_timestamp": "2023-09-17T23:59:58Z",
                "coverage_hours": 48.0,
            },
            "ETH": {
                "row_count": 900,
                "last_timestamp": "2023-09-17T23:59:58Z",
                "coverage_hours": 48.0,
            },
        },
        "resampled_coverage": {
            "BTC": {"15s": {"row_count": 200}},
            "ETH": {"15s": {"row_count": 180}},
        },
        "candidate_freshness": {
            "filtered_paper_trades": 5,
            "cumulative_net_return_bps": 3.84,
            "status": "not_market_ready",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_daily_summary(reports_root: Path, stamp: str = "20260707T071022Z") -> Path:
    path = reports_root / "bid_ask_daily" / f"bid_ask_daily_summary_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2026-07-07T07:10:22Z",
        "overall_status": {"ready_for_review": True, "warnings": ["Research-only."]},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_walk_forward(reports_root: Path, coin: str, stamp: str) -> Path:
    path = reports_root / "bid_ask_walk_forward" / f"walk_forward_{coin}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "coins": [coin],
        "completed_windows": 3,
        "model_accuracy_mean": 0.42,
        "baseline_accuracy_mean": 0.38,
        "filtered_paper_trades": 5,
        "cumulative_net_return_bps": 3.84,
        "positive_after_costs": True,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_candidate_validation(reports_root: Path, stamp: str) -> Path:
    path = (
        reports_root
        / "bid_ask_candidate_validation"
        / f"candidate_validation_BTC_DOWN_MICRO_H20_CANDIDATE_B_{stamp}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "filtered_paper_trades": 5,
        "cumulative_net_return_bps": 3.84,
        "status": "not_market_ready",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_validation_report(data_root: Path, coin: str, interval: str, stamp: str) -> Path:
    report_dir = data_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{coin}_{interval}_validation_{stamp}.json"
    payload = {
        "coin": coin,
        "interval": interval,
        "passed": True,
        "row_count": 100,
        "missing_periods_count": 0,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _seed_fake_artifacts(reports_root: Path, data_root: Path) -> None:
    _write_phase3_audit(reports_root)
    _write_daily_summary(reports_root)
    _write_walk_forward(reports_root, "BTC", "20260706T183038")
    _write_walk_forward(reports_root, "ETH", "20260706T120000")
    _write_candidate_validation(reports_root, "20260706T181446")
    _write_validation_report(data_root, "BTC", "1s", "20260707T070956Z")
    _write_validation_report(data_root, "BTC", "15s", "20260707T071002Z")


@pytest.fixture
def fake_roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    reports_root = tmp_path / "reports"
    data_root = tmp_path / "data" / "bid_ask_csv"
    output_dir = tmp_path / "reports" / "demo_mode"
    _seed_fake_artifacts(reports_root, data_root)
    return reports_root, data_root, output_dir


class TestPacketGenerated:
    def test_prepare_writes_json_and_markdown(self, fake_roots: tuple[Path, Path, Path]) -> None:
        reports_root, data_root, output_dir = fake_roots
        packet = prepare_demo_mode(
            reports_root=reports_root,
            data_root=data_root,
            output_dir=output_dir,
        )

        json_path = output_dir / "demo_packet_latest.json"
        md_path = output_dir / "demo_packet_latest.md"
        assert json_path.is_file()
        assert md_path.is_file()
        assert packet["json_report_path"] == str(json_path)
        assert packet["markdown_report_path"] == str(md_path)

        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["paper_only"] is True
        assert loaded["execution_enabled"] is False
        assert "project_explanation_30s" in loaded
        assert "current_kpis" in loaded
        assert "candidate_b" in loaded
        assert "milestone_journey" in loaded
        assert "architecture_text" in loaded
        assert "likely_questions_and_answers" in loaded
        assert "what_not_to_claim" in loaded
        assert "next_research_plan" in loaded

    def test_discover_artifacts_reads_latest_reports(
        self, fake_roots: tuple[Path, Path, Path]
    ) -> None:
        reports_root, data_root, _ = fake_roots
        artifacts = discover_artifacts(reports_root=reports_root, data_root=data_root)
        assert artifacts["phase3_audit"] is not None
        assert artifacts["daily_summary"] is not None
        assert artifacts["candidate_validation"] is not None
        assert artifacts["walk_forward"]["BTC"] is not None

    def test_output_dir_must_be_demo_mode(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="demo_mode"):
            validate_output_dir(tmp_path / "other_dir")


class TestDemoScriptIncluded:
    def test_demo_script_included_by_default(self, fake_roots: tuple[Path, Path, Path]) -> None:
        reports_root, data_root, output_dir = fake_roots
        packet = build_demo_packet(reports_root=reports_root, data_root=data_root)
        assert packet["demo_script_5min"] is not None
        assert len(packet["demo_script_5min"]) >= 5

        md_path = write_demo_packet_markdown(packet, output_dir=output_dir)
        text = md_path.read_text(encoding="utf-8")
        assert "## 5-Minute Demo Script" in text

    def test_demo_script_can_be_disabled(self, fake_roots: tuple[Path, Path, Path]) -> None:
        reports_root, data_root, output_dir = fake_roots
        packet = build_demo_packet(
            reports_root=reports_root,
            data_root=data_root,
            include_script=False,
        )
        assert packet["demo_script_5min"] is None

        md_path = write_demo_packet_markdown(packet, output_dir=output_dir)
        text = md_path.read_text(encoding="utf-8")
        assert "## 5-Minute Demo Script" not in text

    def test_checklist_included_by_default(self, fake_roots: tuple[Path, Path, Path]) -> None:
        reports_root, data_root, output_dir = fake_roots
        packet = build_demo_packet(reports_root=reports_root, data_root=data_root)
        assert packet["demo_checklist"] is not None

        md_path = write_demo_packet_markdown(packet, output_dir=output_dir)
        assert "## Demo Checklist" in md_path.read_text(encoding="utf-8")


class TestSafetyWarningsIncluded:
    def test_safety_boundary_in_packet(self, fake_roots: tuple[Path, Path, Path]) -> None:
        reports_root, data_root, output_dir = fake_roots
        packet = prepare_demo_mode(
            reports_root=reports_root,
            data_root=data_root,
            output_dir=output_dir,
        )
        safety = packet["safety_boundary"]
        assert safety["paper_only"] is True
        assert safety["execution_enabled"] is False
        assert safety["real_execution_ready"] is False
        assert safety["market_ready"] is False
        assert "No orders" in " ".join(safety["demo_rules"])

        md_text = (output_dir / "demo_packet_latest.md").read_text(encoding="utf-8")
        assert "## Safety Boundary" in md_text
        assert "Research-only" in md_text or "research-only" in md_text
        assert "## What Not to Claim" in md_text
        assert any("profit" in item.lower() for item in packet["what_not_to_claim"])

    def test_research_only_warnings_present(self, fake_roots: tuple[Path, Path, Path]) -> None:
        reports_root, data_root, output_dir = fake_roots
        packet = prepare_demo_mode(
            reports_root=reports_root,
            data_root=data_root,
            output_dir=output_dir,
        )
        assert any("Research-only" in warning for warning in packet["warnings"])


class TestDashboardCommandIncluded:
    def test_dashboard_launch_command_in_outputs(self, fake_roots: tuple[Path, Path, Path]) -> None:
        reports_root, data_root, output_dir = fake_roots
        packet = prepare_demo_mode(
            reports_root=reports_root,
            data_root=data_root,
            output_dir=output_dir,
        )
        assert packet["dashboard_launch_command"] == DASHBOARD_LAUNCH_COMMAND
        assert "streamlit run bid_ask_research_dashboard.py" in packet["dashboard_launch_command"]

        md_text = (output_dir / "demo_packet_latest.md").read_text(encoding="utf-8")
        assert DASHBOARD_LAUNCH_COMMAND in md_text
        assert "## Dashboard Launch Command" in md_text

    def test_cli_prints_paths(
        self,
        fake_roots: tuple[Path, Path, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        reports_root, data_root, output_dir = fake_roots
        exit_code = main(
            [
                "prepare",
                "--reports-root",
                str(reports_root),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(output_dir),
            ]
        )
        captured = capsys.readouterr().out
        assert exit_code == 0
        assert "json_report_path=" in captured
        assert "markdown_report_path=" in captured
        assert "dashboard_launch_command=" in captured
        assert DASHBOARD_LAUNCH_COMMAND in captured


class TestNoSubprocessNetworkAws:
    def test_no_forbidden_imports(self) -> None:
        source = DEMO_MODE_FILE.read_text(encoding="utf-8")
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

    def test_no_network_calls_in_source(self) -> None:
        source = DEMO_MODE_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {
            "requests",
            "urllib",
            "http",
            "socket",
            "websockets",
            "httpx",
            "aiohttp",
            "boto3",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in forbidden
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in forbidden

    def test_no_subprocess_calls_in_source(self) -> None:
        source = DEMO_MODE_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] != "subprocess"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] != "subprocess"

    def test_no_runtime_network_subprocess_or_aws(
        self,
        fake_roots: tuple[Path, Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        reports_root, data_root, output_dir = fake_roots

        def _blocked(*args: object, **kwargs: object) -> None:
            raise AssertionError("network/subprocess/AWS call attempted during prepare")

        monkeypatch.setattr("urllib.request.urlopen", _blocked)
        monkeypatch.setattr("subprocess.run", _blocked)
        monkeypatch.setattr("subprocess.Popen", _blocked)
        monkeypatch.setattr("boto3.client", _blocked)

        packet = prepare_demo_mode(
            reports_root=reports_root,
            data_root=data_root,
            output_dir=output_dir,
        )
        assert packet["json_report_path"]
        assert packet["markdown_report_path"]

    def test_paper_only_constants(self) -> None:
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False
        assert DEFAULT_OUTPUT_DIR == Path("reports/demo_mode")

    def test_writes_only_under_demo_mode(self, fake_roots: tuple[Path, Path, Path]) -> None:
        reports_root, data_root, output_dir = fake_roots
        prepare_demo_mode(
            reports_root=reports_root,
            data_root=data_root,
            output_dir=output_dir,
        )
        assert output_dir.name == "demo_mode"
        files = list(output_dir.iterdir())
        assert len(files) == 2
        assert all(path.parent == output_dir for path in files)
