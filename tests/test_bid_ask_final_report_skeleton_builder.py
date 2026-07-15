"""Tests for bid_ask_final_report_skeleton_builder (offline, temp dirs, no network)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import bid_ask_final_report_skeleton_builder as builder
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY

BUILDER_FILE = Path(__file__).resolve().parent.parent / "bid_ask_final_report_skeleton_builder.py"

TEST_STAMP = "20240102T100000Z"

REQUIRED_MARKDOWN_HEADINGS = (
    "## 1. Title and Safety Disclaimer",
    "## 2. Abstract",
    "## 3. Research Question",
    "## 4. System Overview",
    "## 5. Data Pipeline",
    "## 6. Historical Backfill and Validation",
    "## 7. Feature Engineering",
    "## 8. Labeling and Evaluation",
    "## 9. Candidate Timeline",
    "## 10. Current Results Summary",
    "## 11. Failure Analysis",
    "## 12. Robustness Checks",
    "## 13. Dashboard and Demo Artifacts",
    "## 14. Limitations",
    "## 15. Next Steps",
    "## 16. Appendix: Commands and Artifact Paths",
)

PROFITABILITY_CLAIM_PHRASES = (
    "proves profitability",
    "proven profitable",
    "guaranteed returns",
    "market ready for live trading",
    "ready for live execution",
    "robust market-ready edge",
)


def _write_research_status(
    path: Path,
    *,
    ledger_row_count: int = 5,
    btc_rows: int = 100,
    eth_rows: int = 80,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2024-01-02T10:00:00Z",
        "paper_only": True,
        "execution_enabled": False,
        "real_execution_ready": False,
        "data_status": {
            "coins": {
                "BTC": {
                    "1s": {"row_count": btc_rows, "file_exists": True},
                    "15s": {"row_count": 40, "file_exists": True},
                    "1m": {"row_count": 10, "file_exists": True},
                },
                "ETH": {
                    "1s": {"row_count": eth_rows, "file_exists": True},
                    "15s": {"row_count": 30, "file_exists": True},
                    "1m": {"row_count": 8, "file_exists": True},
                },
            },
            "warnings": ["BTC 1s: 120 missing or incomplete bar(s) detected"],
        },
        "candidate_timeline": list(builder.CANDIDATE_TIMELINE)
        if hasattr(builder, "CANDIDATE_TIMELINE")
        else [],
        "ledger_summary": {
            "available": True,
            "ledger_row_count": ledger_row_count,
            "top_by_paper_score": [
                {
                    "config_id": "eth_sparse",
                    "coin": "ETH",
                    "paper_score": 12.0,
                    "trade_count": 4,
                    "positive_after_costs": True,
                    "sparse_warning": True,
                }
            ],
        },
        "accepted_trades_analysis": {
            "available": True,
            "by_coin": {
                "BTC": {
                    "available": True,
                    "trade_count": 5,
                    "cumulative_net_return_bps": 3.84,
                },
                "ETH": {
                    "available": True,
                    "trade_count": 14,
                    "cumulative_net_return_bps": 31.28,
                },
            },
        },
        "blockers": ["No robust positive candidates — positive configs have fewer than 10 trades."],
        "next_actions": ["Continue historical backfill for BTC and ETH 1s data."],
    }
    if not payload["candidate_timeline"]:
        from bid_ask_research_status_builder import CANDIDATE_TIMELINE

        payload["candidate_timeline"] = list(CANDIDATE_TIMELINE)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_candidate_ledger(path: Path, row_count: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2024-01-02T10:00:00Z",
        "paper_only": True,
        "execution_enabled": False,
        "real_execution_ready": False,
        "row_count": row_count,
        "rows": [{"candidate_name": "BTC_DOWN_B"} for _ in range(row_count)],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_dashboard_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2024-01-02T10:00:00Z",
        "paper_only": True,
        "execution_enabled": False,
        "next_recommended_action": "Continue backfill before retuning thresholds.",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_demo_packet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2024-01-02T10:00:00Z",
        "paper_only": True,
        "warnings": ["Research-only. Does not prove profitability."],
        "what_not_to_claim": ["Profitability is proven", "System is market ready"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_feature_audit(path: Path, coin: str, interval: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2024-01-02T10:00:00Z",
        "coin": coin,
        "interval": interval,
        "row_count": 50,
        "feature_families": {
            "spread_cost_regime": [],
            "candidate_c_core": [],
        },
        "paper_only": True,
        "execution_enabled": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    reports_root = tmp_path / "reports"
    data_root = tmp_path / "data" / "bid_ask_csv"
    output_dir = tmp_path / "reports" / "final_research_report"

    _write_research_status(reports_root / "research_status" / "research_status_latest.json")
    _write_candidate_ledger(reports_root / "candidate_ledger" / "candidate_ledger.json", row_count=7)
    _write_dashboard_summary(reports_root / "dashboard_artifacts" / "dashboard_summary_latest.json")
    _write_demo_packet(reports_root / "demo_mode" / "demo_packet_latest.json")
    _write_feature_audit(
        reports_root / "feature_audit" / f"feature_audit_BTC_15s_{TEST_STAMP}.json",
        "BTC",
        "15s",
    )

    return {
        "reports_root": reports_root,
        "data_root": data_root,
        "output_dir": output_dir,
    }


class TestBuildFinalReportSkeleton:
    def test_creates_markdown_and_json(self, workspace: dict[str, Path]) -> None:
        result = builder.build_and_write_final_report_skeleton(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
        )

        json_latest = workspace["output_dir"] / builder.DEFAULT_LATEST_JSON_NAME
        md_latest = workspace["output_dir"] / builder.DEFAULT_LATEST_MD_NAME

        assert json_latest.is_file()
        assert md_latest.is_file()
        assert Path(result["output_paths"]["json_stamped"]).is_file()
        assert Path(result["output_paths"]["markdown_stamped"]).is_file()

    def test_includes_all_required_sections(self, workspace: dict[str, Path]) -> None:
        result = builder.build_and_write_final_report_skeleton(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
        )
        markdown = Path(result["output_paths"]["markdown_latest"]).read_text(encoding="utf-8")

        for heading in REQUIRED_MARKDOWN_HEADINGS:
            assert heading in markdown

        for section in builder.REPORT_SECTIONS:
            assert section in result["sections"]

        assert "### Candidate C core" in markdown
        assert "### Candidate E depth-aware" in markdown
        assert "### Direction labels" in markdown
        assert "### Cost-aware labels" in markdown
        assert "### Binary tradeable-edge labels" in markdown
        assert "### Candidate B:" in markdown or "Candidate B" in markdown

    def test_uses_research_status_when_present(self, workspace: dict[str, Path]) -> None:
        result = builder.build_final_report_skeleton(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
        )

        assert result["inputs"]["research_status"]["available"] is True
        assert result["inputs"]["context_source"] == "research_status_latest"
        assert result["data_row_counts"]["BTC"]["1s"] == 100
        assert result["ledger_summary"]["ledger_row_count"] == 5
        assert result["blockers"]
        assert result["next_actions"]

        markdown = builder.write_final_report_skeleton_markdown(
            result,
            workspace["output_dir"],
            filename="research_status_check.md",
        ).read_text(encoding="utf-8")
        assert "Research status loaded from:" in markdown

    def test_handles_missing_artifacts(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "reports" / "final_research_report"
        result = builder.build_and_write_final_report_skeleton(
            reports_root=tmp_path / "missing_reports",
            data_root=tmp_path / "missing_data",
            output_dir=output_dir,
        )

        assert result["inputs"]["research_status"]["available"] is False
        assert result["inputs"]["candidate_ledger"]["row_count"] == 0
        assert result["inputs"]["dashboard_summary"]["available"] is False
        assert result["inputs"]["demo_packet"]["available"] is False
        assert result["inputs"]["feature_audits"] == []
        assert (output_dir / builder.DEFAULT_LATEST_JSON_NAME).is_file()

        markdown = Path(result["output_paths"]["markdown_latest"]).read_text(encoding="utf-8")
        assert "Draft — To be updated" in markdown
        assert "Research status not built" in markdown
        assert "Candidate ledger not available" in markdown

    def test_does_not_claim_profitability(self, workspace: dict[str, Path]) -> None:
        result = builder.build_final_report_skeleton(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
        )
        markdown = builder.write_final_report_skeleton_markdown(
            result,
            workspace["output_dir"],
            filename="profitability_check.md",
        ).read_text(encoding="utf-8")

        lowered = markdown.lower()
        for phrase in PROFITABILITY_CLAIM_PHRASES:
            assert phrase not in lowered

        assert result["does_not_claim_profitability"] is True
        assert "does not prove profitability" in lowered or "do not prove" in lowered
        assert "not proof of profitability" in lowered or "not prove profitability" in lowered

    def test_safety_fields_included(self, workspace: dict[str, Path]) -> None:
        result = builder.build_final_report_skeleton(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
        )

        assert result["paper_only"] is True
        assert result["execution_enabled"] is False
        assert result["real_execution_ready"] is False
        assert result["safety"]["paper_only"] is True
        assert result["safety_statement"]
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False

        markdown = builder.write_final_report_skeleton_markdown(
            result,
            workspace["output_dir"],
            filename="safety_check.md",
        ).read_text(encoding="utf-8")
        assert "paper_only: true" in markdown
        assert "execution_enabled: false" in markdown
        assert "real_execution_ready: false" in markdown

    def test_candidate_timeline_entries(self, workspace: dict[str, Path]) -> None:
        result = builder.build_final_report_skeleton(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
        )

        candidates = {entry["candidate"]: entry for entry in result["candidate_timeline"]}
        assert candidates["B"]["status"] == "rejected"
        assert "diagnostic" in candidates["C"]["status"] or "rejected" in candidates["C"]["status"]
        assert candidates["E"]["status"] == "interesting but too sparse"


class TestCli:
    def test_main_build_command(
        self,
        workspace: dict[str, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = builder.main(
            [
                "build",
                "--reports-root",
                str(workspace["reports_root"]),
                "--data-root",
                str(workspace["data_root"]),
                "--output-dir",
                str(workspace["output_dir"]),
            ]
        )

        captured = capsys.readouterr().out
        assert exit_code == 0
        assert "json_latest_path=" in captured
        assert "markdown_latest_path=" in captured
        assert "paper_only=true" in captured
        assert "real_execution_ready=false" in captured
        assert "does_not_claim_profitability=true" in captured


def _block_network_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network/subprocess call attempted")

    monkeypatch.setattr("subprocess.run", _blocked)
    monkeypatch.setattr("subprocess.Popen", _blocked)
    monkeypatch.setattr("boto3.client", _blocked)


class TestSafety:
    def test_no_forbidden_imports(self) -> None:
        source = BUILDER_FILE.read_text(encoding="utf-8")
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
        source = BUILDER_FILE.read_text(encoding="utf-8").lower()
        forbidden = (
            "requests.get",
            "requests.post",
            "urllib.request",
            "http.client",
            "aiohttp",
            "websocket",
            "boto3",
        )
        for pattern in forbidden:
            assert pattern not in source

    def test_no_subprocess_calls_in_source(self) -> None:
        source = BUILDER_FILE.read_text(encoding="utf-8").lower()
        forbidden = ("subprocess", "os.system", "popen")
        for pattern in forbidden:
            assert pattern not in source

    def test_output_dir_must_be_final_research_report(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="final_research_report"):
            builder.build_and_write_final_report_skeleton(
                reports_root=tmp_path / "reports",
                data_root=tmp_path / "data",
                output_dir=tmp_path / "reports" / "other_dir",
            )

    def test_no_network_subprocess_or_aws_at_runtime(
        self,
        workspace: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _block_network_subprocess(monkeypatch)
        builder.build_and_write_final_report_skeleton(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
        )
