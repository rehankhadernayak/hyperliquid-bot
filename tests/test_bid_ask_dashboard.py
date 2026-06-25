"""Tests for bid/ask research dashboard helpers and safety guards."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

import bid_ask_dashboard


REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_1s_csv(path: Path, coin: str, rows: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "timestamp_ms,timestamp_iso,coin,best_bid,best_ask,mid_price,spread,spread_bps,"
        "bid_size,ask_size,bbo_imbalance,source_session_id,source_channel"
    ]
    base_ts = 1_700_000_000_000
    for index in range(rows):
        ts = base_ts + index * 1000
        bid = 100.0 + index
        ask = 100.5 + index
        mid = (bid + ask) / 2.0
        spread = ask - bid
        spread_bps = spread / mid * 10000.0
        lines.append(
            f"{ts},2024-01-01T00:00:00Z,{coin},{bid},{ask},{mid},{spread},{spread_bps},"
            f"1.0,1.0,0.0,session,bbo"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_15s_csv(path: Path, coin: str, rows: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "start_timestamp_ms,end_timestamp_ms,timestamp_iso,coin,open_mid,high_mid,low_mid,"
        "close_mid,avg_bid,avg_ask,avg_mid,avg_spread,avg_spread_bps,avg_bbo_imbalance,"
        "row_count,missing_seconds"
    ]
    base_ts = 1_700_000_000_000
    for index in range(rows):
        start = base_ts + index * 15_000
        end = start + 14_999
        mid = 100.0 + index
        lines.append(
            f"{start},{end},2024-01-01T00:00:00Z,{coin},{mid},{mid + 0.5},{mid - 0.5},"
            f"{mid},{mid - 0.25},{mid + 0.25},{mid},0.5,5.0,0.0,15,0"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_validation_report(
    reports_dir: Path,
    coin: str,
    interval: str,
    stamp: str,
    *,
    passed: bool = True,
    row_count: int = 10,
    missing_periods: int = 0,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{coin}_{interval}_validation_{stamp}.json"
    payload = {
        "coin": coin,
        "interval": interval,
        "row_count": row_count,
        "coverage_start": 1_700_000_000_000,
        "coverage_end": 1_700_000_004_000,
        "duplicate_timestamp_count": 0,
        "missing_periods_count": missing_periods,
        "invalid_bid_ask_count": 0,
        "negative_spread_count": 0,
        "extreme_spread_count": 0,
        "warnings": [],
        "passed": passed,
        "generated_at": f"2024-01-01T00:00:{stamp[-2:]}Z",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_training_report(
    reports_dir: Path,
    coin: str,
    report_id: str,
    generated_at: str,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{report_id}.json"
    payload = {
        "report_id": report_id,
        "coin": coin,
        "interval": "15s",
        "horizon": 1,
        "model_used": "majority_class",
        "generated_at": generated_at,
        "evaluation": {
            "accuracy": 0.55,
            "baseline_accuracy": 0.50,
            "train_rows": 100,
            "test_rows": 40,
            "label_counts": {
                "test": {"UP": 10, "DOWN": 12, "FLAT": 18},
            },
            "warnings": ["Research-only evaluation; does not prove profitability"],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_extraction_report(reports_dir: Path, coin: str, stamp: str) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{coin}_{stamp}.json"
    payload = {
        "coin": coin,
        "locally_collected_data_only": True,
        "row_count": 100,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "bid_ask_csv"
    _write_1s_csv(root / "1s" / "BTC_1s.csv", "BTC", rows=10)
    _write_1s_csv(root / "1s" / "ETH_1s.csv", "ETH", rows=4)
    _write_15s_csv(root / "15s" / "BTC_15s.csv", "BTC", rows=3)
    _write_15s_csv(root / "15s" / "SOL_15s.csv", "SOL", rows=2)
    _write_15s_csv(root / "1m" / "BTC_1m.csv", "BTC", rows=2)
    reports = root / "reports"
    _write_validation_report(reports, "BTC", "1s", "20240101T100000Z", row_count=10)
    _write_validation_report(
        reports, "BTC", "1s", "20240101T120000Z", row_count=10, missing_periods=5
    )
    _write_validation_report(reports, "BTC", "15s", "20240101T100000Z", passed=True)
    _write_extraction_report(reports, "BTC", "20240101T100000Z")
    return root


@pytest.fixture
def training_reports_dir(tmp_path: Path) -> Path:
    reports = tmp_path / "training"
    _write_training_report(reports, "BTC", "older_report", "2024-01-01T10:00:00Z")
    _write_training_report(reports, "BTC", "newer_report", "2024-01-02T10:00:00Z")
    _write_training_report(reports, "ETH", "eth_report", "2024-01-03T10:00:00Z")
    return reports


class TestDiscoverBidAskFiles:
    def test_finds_1s_15s_1m_csvs(self, data_root: Path, training_reports_dir: Path):
        discovery = bid_ask_dashboard.discover_bid_ask_files(data_root, training_reports_dir)
        assert discovery["counts"]["csv_1s"] == 2
        assert discovery["counts"]["csv_15s"] == 2
        assert discovery["counts"]["csv_1m"] == 1
        assert discovery["counts"]["validation_reports"] == 3
        assert discovery["locally_collected_data_only"] is True

    def test_empty_root_is_graceful(self, tmp_path: Path):
        empty_root = tmp_path / "empty"
        empty_root.mkdir()
        discovery = bid_ask_dashboard.discover_bid_ask_files(empty_root, tmp_path / "missing")
        assert discovery["counts"]["csv_1s"] == 0
        assert discovery["counts"]["validation_reports"] == 0
        assert discovery["locally_collected_data_only"] is False


class TestCoverageTable:
    def test_row_counts(self, data_root: Path):
        table = bid_ask_dashboard.build_coverage_table(data_root=data_root, coins=("BTC",))
        row = table.iloc[0]
        assert row["coin"] == "BTC"
        assert row["has_1s_csv"]
        assert row["has_15s_csv"]
        assert row["has_1m_csv"]
        assert row["row_count_1s"] == 10
        assert row["row_count_15s"] == 3
        assert row["row_count_1m"] == 2
        assert row["validation_passed_1s"]
        assert row["missing_seconds_1s"] == 5

    def test_missing_files_handled_gracefully(self, tmp_path: Path):
        table = bid_ask_dashboard.build_coverage_table(
            data_root=tmp_path / "missing",
            coins=("BTC",),
        )
        row = table.iloc[0]
        assert not row["has_1s_csv"]
        assert row["row_count_1s"] == 0
        assert row["validation_passed_1s"] is None
        assert row["missing_seconds_1s"] is None


class TestValidationReport:
    def test_latest_validation_report_selected(self, data_root: Path):
        report = bid_ask_dashboard.load_latest_validation_report("BTC", "1s", data_root)
        assert report is not None
        assert report["missing_periods_count"] == 5
        assert "20240101T120000Z" in report["report_path"]

    def test_missing_validation_report_returns_none(self, tmp_path: Path):
        report = bid_ask_dashboard.load_latest_validation_report(
            "BTC", "1s", tmp_path / "missing"
        )
        assert report is None


class TestTrainingReport:
    def test_latest_training_report_selected(self, training_reports_dir: Path):
        report = bid_ask_dashboard.load_latest_training_report("BTC", training_reports_dir)
        assert report is not None
        assert report["report_id"] == "newer_report"
        assert report["evaluation"]["accuracy"] == pytest.approx(0.55)

    def test_missing_training_report_returns_none(self, tmp_path: Path):
        report = bid_ask_dashboard.load_latest_training_report("BTC", tmp_path / "missing")
        assert report is None


class TestCsvPreview:
    def test_loads_first_rows(self, data_root: Path):
        preview = bid_ask_dashboard.load_csv_preview("BTC", "1s", data_root, n=5)
        assert preview["found"] is True
        assert len(preview["preview"]) == 5
        assert preview["total_rows"] == 10
        assert "BTC_1s.csv" in preview["path"]

    def test_missing_csv_is_graceful(self, tmp_path: Path):
        preview = bid_ask_dashboard.load_csv_preview("BTC", "1s", tmp_path / "missing")
        assert preview["found"] is False
        assert preview["preview"].empty


class TestChartDataframe:
    def test_chart_dataframe_1s(self, data_root: Path):
        chart_df = bid_ask_dashboard.build_chart_dataframe("BTC", "1s", data_root)
        assert not chart_df.empty
        assert list(chart_df.columns) == [
            "timestamp_ms",
            "best_bid",
            "best_ask",
            "mid_price",
            "spread",
            "spread_bps",
        ]
        assert len(chart_df) == 10

    def test_chart_dataframe_15s(self, data_root: Path):
        chart_df = bid_ask_dashboard.build_chart_dataframe("BTC", "15s", data_root)
        assert not chart_df.empty
        assert "avg_bid" in chart_df.columns
        assert "avg_ask" in chart_df.columns
        assert "avg_spread_bps" in chart_df.columns
        assert len(chart_df) == 3

    def test_chart_dataframe_missing_file(self, tmp_path: Path):
        chart_df = bid_ask_dashboard.build_chart_dataframe("BTC", "1m", tmp_path / "missing")
        assert chart_df.empty


class TestSuggestedCommands:
    def test_commands_are_display_only_strings(self):
        commands = bid_ask_dashboard.build_suggested_commands("BTC")
        assert "extract" in commands
        assert "validate" in commands
        assert "resample" in commands
        assert "train" in commands
        assert "run-all" in commands
        assert "python3 bid_ask_data_pipeline.py" in commands["extract"]
        assert "BTC" in commands["extract"]


class TestSafetyGuards:
    FORBIDDEN_PATTERNS = (
        "private_key",
        "place_order",
        "execute_order",
        "sign_transaction",
    )

    NETWORK_IMPORTS = (
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
        "socket",
    )

    def test_no_command_execution(self):
        source = (REPO_ROOT / "bid_ask_dashboard.py").read_text(encoding="utf-8")
        forbidden = ("subprocess", "Popen", "os.system", "bid_ask_data_pipeline.main")
        for token in forbidden:
            assert token not in source

        render_sources = [
            inspect.getsource(bid_ask_dashboard._render_next_commands),
            inspect.getsource(bid_ask_dashboard.build_suggested_commands),
        ]
        for render_source in render_sources:
            for token in forbidden:
                assert token not in render_source

    def test_no_network_calls(self):
        tree = ast.parse((REPO_ROOT / "bid_ask_dashboard.py").read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".", 1)[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        for module in self.NETWORK_IMPORTS:
            assert module not in imported

    def test_no_wallet_signing_order_execution_code(self):
        raw = (REPO_ROOT / "bid_ask_dashboard.py").read_text(encoding="utf-8")
        body = raw.split('"""', 2)[-1].lower() if raw.count('"""') >= 2 else raw.lower()
        for pattern in self.FORBIDDEN_PATTERNS:
            assert pattern not in body

        tree = ast.parse(raw)
        forbidden = {"wallet", "signing", "web3", "eth_account"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in forbidden
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in forbidden

    def test_research_disclaimer_present(self):
        assert "Research-only results" in bid_ask_dashboard.RESEARCH_DISCLAIMER
