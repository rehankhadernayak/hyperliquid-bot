"""Focused offline tests for the candidate regression ledger."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any

import pytest

from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY
from bid_ask_candidate_ledger import (
    DEFAULT_OUTPUT_DIR,
    LEDGER_COLUMNS,
    RELAXED_CANDIDATE_B_NAME,
    REPORT_TYPE_CANDIDATE_VALIDATION,
    REPORT_TYPE_ROBUSTNESS,
    REPORT_TYPE_WALK_FORWARD,
    STRICT_CANDIDATE_B_NAME,
    build_ledger_rows,
    build_ledger_summary,
    parse_candidate_validation_row,
    parse_robustness_row,
    parse_walk_forward_row,
    update_ledger,
)

LEDGER_FILE = Path(__file__).resolve().parent.parent / "bid_ask_candidate_ledger.py"


def _candidate_validation_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "candidate_name": STRICT_CANDIDATE_B_NAME,
        "run_timestamp": "2026-07-01T08:00:00Z",
        "config": {
            "candidate_name": STRICT_CANDIDATE_B_NAME,
            "coin": "BTC",
            "interval": "15s",
            "horizon": 20,
            "feature_set": "microstructure",
            "label_mode": "direction",
            "trade_classes": ["DOWN"],
            "min_confidence": 0.65,
            "min_edge_bps": 2.0,
            "cost_multiplier": 1.25,
            "max_spread_bps": 8.0,
        },
        "filtered_paper_trades": 25,
        "cumulative_net_return_bps": 12.5,
        "average_net_return_bps": 0.5,
        "win_rate_after_costs": 0.62,
        "accuracy_edge_vs_baseline": 0.05,
        "positive_after_costs": True,
        "beats_baseline_accuracy": True,
        "validation_status": {
            "paper_candidate_pass": True,
            "not_market_ready": False,
        },
        "stability_warnings": ["warning one"],
        "paper_only": True,
        "execution_enabled": False,
        "real_execution_ready": False,
    }
    base.update(overrides)
    return base


def _walk_forward_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "generated_at": "2026-07-02T09:00:00+00:00",
        "coins": ["BTC"],
        "interval": "15s",
        "horizon": 20,
        "feature_set": "microstructure",
        "label_mode": "direction",
        "trade_classes": ["DOWN"],
        "min_confidence": 0.65,
        "min_edge_bps": 2.0,
        "cost_multiplier": 1.25,
        "max_spread_bps": 8.0,
        "filtered_paper_trades": 30,
        "cumulative_net_return_bps": 15.0,
        "average_net_return_bps": 0.6,
        "win_rate_after_costs": 0.64,
        "accuracy_edge_vs_baseline": 0.04,
        "positive_after_costs": True,
        "beats_baseline_accuracy": True,
        "ready_for_more_research": True,
        "readiness": {
            "ready_for_more_research": True,
            "positive_after_costs": True,
            "beats_baseline_accuracy": True,
            "real_execution_ready": False,
        },
        "per_coin": {
            "BTC": {
                "status": "completed",
                "filtered_paper_trades": 30,
                "cumulative_net_return_bps": 15.0,
            }
        },
        "warnings": ["warn-a", "warn-b"],
        "paper_only": True,
        "execution_enabled": False,
        "real_execution_ready": False,
    }
    base.update(overrides)
    return base


def _robustness_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "candidate_name": STRICT_CANDIDATE_B_NAME,
        "run_timestamp": "2026-07-03T10:00:00Z",
        "config": {
            "candidate_name": STRICT_CANDIDATE_B_NAME,
            "coin": "BTC",
            "interval": "15s",
            "horizon": 20,
            "feature_set": "microstructure",
            "label_mode": "direction",
            "trade_classes": ["DOWN"],
            "min_confidence": 0.65,
            "min_edge_bps": 2.0,
            "cost_multiplier": 1.25,
            "max_spread_bps": 8.0,
        },
        "robustness_pass": False,
        "robustness_summary": {
            "slices_tested": 7,
            "robustness_pass": False,
        },
        "paper_only": True,
        "execution_enabled": False,
        "real_execution_ready": False,
    }
    base.update(overrides)
    return base


def _write_reports(reports_root: Path) -> None:
    validation_dir = reports_root / "bid_ask_candidate_validation"
    walk_forward_dir = reports_root / "bid_ask_walk_forward"
    validation_dir.mkdir(parents=True)
    walk_forward_dir.mkdir(parents=True)

    (validation_dir / "candidate_validation_strict.json").write_text(
        json.dumps(_candidate_validation_payload(), indent=2),
        encoding="utf-8",
    )
    (validation_dir / "candidate_validation_relaxed.json").write_text(
        json.dumps(
            _candidate_validation_payload(
                candidate_name=RELAXED_CANDIDATE_B_NAME,
                run_timestamp="2026-07-04T11:00:00Z",
                config={
                    "candidate_name": RELAXED_CANDIDATE_B_NAME,
                    "coin": "BTC",
                    "interval": "15s",
                    "horizon": 20,
                    "feature_set": "microstructure",
                    "label_mode": "direction",
                    "trade_classes": ["DOWN"],
                    "min_confidence": 0.6,
                    "min_edge_bps": 1.0,
                    "cost_multiplier": 1.25,
                    "max_spread_bps": 8.0,
                },
                filtered_paper_trades=40,
                cumulative_net_return_bps=20.0,
                validation_status={"paper_candidate_pass": False, "not_market_ready": True},
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    (walk_forward_dir / "walk_forward_BTC_test.json").write_text(
        json.dumps(_walk_forward_payload(), indent=2),
        encoding="utf-8",
    )
    (validation_dir / "robustness_strict.json").write_text(
        json.dumps(_robustness_payload(), indent=2),
        encoding="utf-8",
    )


class TestParsing:
    def test_candidate_validation_parsed(self) -> None:
        row = parse_candidate_validation_row(
            "reports/bid_ask_candidate_validation/candidate_validation_test.json",
            _candidate_validation_payload(),
        )
        assert row["report_type"] == REPORT_TYPE_CANDIDATE_VALIDATION
        assert row["candidate_name"] == STRICT_CANDIDATE_B_NAME
        assert row["coin"] == "BTC"
        assert row["interval"] == "15s"
        assert row["horizon"] == 20
        assert row["feature_set"] == "microstructure"
        assert row["label_mode"] == "direction"
        assert row["trade_classes"] == "DOWN"
        assert row["filtered_paper_trades"] == 25
        assert row["cumulative_net_return_bps"] == 12.5
        assert row["paper_candidate_pass"] is True
        assert row["status"] == "paper_candidate_pass"
        assert row["warnings_count"] == 1
        assert row["real_execution_ready"] is False

    def test_walk_forward_parsed(self) -> None:
        row = parse_walk_forward_row(
            "reports/bid_ask_walk_forward/walk_forward_BTC_test.json",
            _walk_forward_payload(),
        )
        assert row["report_type"] == REPORT_TYPE_WALK_FORWARD
        assert row["run_timestamp"] == "2026-07-02T09:00:00+00:00"
        assert row["coin"] == "BTC"
        assert row["filtered_paper_trades"] == 30
        assert row["cumulative_net_return_bps"] == 15.0
        assert row["beats_baseline"] is True
        assert row["positive_after_costs"] is True
        assert row["paper_candidate_pass"] is None
        assert row["robustness_pass"] is None
        assert row["status"] == "ready_for_more_research"
        assert row["warnings_count"] == 2
        assert row["real_execution_ready"] is False

    def test_robustness_parsed(self) -> None:
        row = parse_robustness_row(
            "reports/bid_ask_candidate_validation/robustness_test.json",
            _robustness_payload(),
        )
        assert row["report_type"] == REPORT_TYPE_ROBUSTNESS
        assert row["candidate_name"] == STRICT_CANDIDATE_B_NAME
        assert row["robustness_pass"] is False
        assert row["status"] == "robustness_fail"
        assert row["paper_candidate_pass"] is None
        assert row["filtered_paper_trades"] is None
        assert row["real_execution_ready"] is False

    def test_missing_fields_handled(self) -> None:
        row = parse_candidate_validation_row(
            "reports/bid_ask_candidate_validation/minimal.json",
            {"run_timestamp": "2026-07-01T08:00:00Z"},
        )
        assert row["candidate_name"] == ""
        assert row["coin"] == ""
        assert row["interval"] == ""
        assert row["horizon"] is None
        assert row["filtered_paper_trades"] is None
        assert row["cumulative_net_return_bps"] is None
        assert row["paper_candidate_pass"] is None
        assert row["status"] == "not_market_ready"
        assert row["warnings_count"] == 0
        assert row["real_execution_ready"] is False


class TestLedgerOutputs:
    def test_csv_json_md_written(self, tmp_path: Path) -> None:
        reports_root = tmp_path / "reports"
        output_dir = tmp_path / "reports" / "candidate_ledger"
        _write_reports(reports_root)

        result = update_ledger(reports_root=reports_root, output_dir=output_dir)

        csv_path = Path(result["csv_path"])
        json_path = Path(result["json_path"])
        markdown_path = Path(result["markdown_path"])

        assert csv_path.is_file()
        assert json_path.is_file()
        assert markdown_path.is_file()
        assert result["row_count"] == 4

        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 4
        assert set(rows[0].keys()) == set(LEDGER_COLUMNS)

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["paper_only"] is True
        assert payload["execution_enabled"] is False
        assert payload["real_execution_ready"] is False
        assert len(payload["rows"]) == 4
        assert payload["summary"]["status_distribution"]

        markdown = markdown_path.read_text(encoding="utf-8")
        assert "Best by cumulative net return" in markdown
        assert "Latest strict Candidate B" in markdown
        assert "Latest relaxed Candidate B" in markdown
        assert "Status distribution" in markdown

    def test_sorted_by_timestamp(self, tmp_path: Path) -> None:
        reports_root = tmp_path / "reports"
        _write_reports(reports_root)

        rows = build_ledger_rows(reports_root=reports_root)
        timestamps = [row["run_timestamp"] for row in rows]
        assert timestamps == sorted(timestamps)

    def test_summary_highlights(self, tmp_path: Path) -> None:
        reports_root = tmp_path / "reports"
        _write_reports(reports_root)
        rows = build_ledger_rows(reports_root=reports_root)
        summary = build_ledger_summary(rows)

        assert summary["best_by_trade_count"]["filtered_paper_trades"] == 40
        assert summary["best_by_cumulative_net_return_bps"]["cumulative_net_return_bps"] == 20.0
        assert summary["latest_strict_candidate_b"]["candidate_name"] == STRICT_CANDIDATE_B_NAME
        assert summary["latest_relaxed_candidate_b"]["candidate_name"] == RELAXED_CANDIDATE_B_NAME
        assert summary["status_distribution"]["paper_candidate_pass"] == 1


class TestSafety:
    def test_safety_flags_remain_paper_only(self, tmp_path: Path) -> None:
        reports_root = tmp_path / "reports"
        output_dir = tmp_path / "reports" / "candidate_ledger"
        _write_reports(reports_root)

        result = update_ledger(reports_root=reports_root, output_dir=output_dir)
        payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

        assert payload["paper_only"] is True
        assert payload["execution_enabled"] is False
        assert payload["real_execution_ready"] is False
        assert all(row["real_execution_ready"] is False for row in payload["rows"])

    def test_output_dir_must_be_candidate_ledger(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="candidate_ledger"):
            update_ledger(
                reports_root=tmp_path / "reports",
                output_dir=tmp_path / "reports" / "other_dir",
            )

    def test_no_forbidden_imports(self) -> None:
        source = LEDGER_FILE.read_text(encoding="utf-8")
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

    def test_no_network_calls(self) -> None:
        source = LEDGER_FILE.read_text(encoding="utf-8").lower()
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

    def test_no_subprocess_calls(self) -> None:
        source = LEDGER_FILE.read_text(encoding="utf-8").lower()
        forbidden = ("subprocess", "os.system", "popen")
        for pattern in forbidden:
            assert pattern not in source

    def test_paper_only_constants(self) -> None:
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False
        assert DEFAULT_OUTPUT_DIR == Path("reports/candidate_ledger")
