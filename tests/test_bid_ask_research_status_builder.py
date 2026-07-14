"""Tests for bid_ask_research_status_builder (offline, temp dirs, no network)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import bid_ask_research_status_builder as builder
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY

BUILDER_FILE = Path(__file__).resolve().parent.parent / "bid_ask_research_status_builder.py"

TEST_STAMP = "20240102T100000Z"


def _write_1s_csv(
    path: Path,
    coin: str,
    rows: int = 5,
    base_ts: int = 1_700_000_000_000,
    *,
    include_sizes: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if include_sizes:
        header = (
            "timestamp_ms,timestamp_iso,coin,best_bid,best_ask,mid_price,spread,spread_bps,"
            "bid_size,ask_size,bbo_imbalance,source_session_id,source_channel"
        )
    else:
        header = (
            "timestamp_ms,timestamp_iso,coin,best_bid,best_ask,mid_price,spread,spread_bps,"
            "bbo_imbalance,source_session_id,source_channel"
        )
    lines = [header]
    for index in range(rows):
        ts = base_ts + index * 1000
        bid = 100.0 + index
        ask = 100.5 + index
        mid = (bid + ask) / 2.0
        spread = ask - bid
        spread_bps = spread / mid * 10000.0
        if include_sizes:
            lines.append(
                f"{ts},2024-01-01T00:00:00Z,{coin},{bid},{ask},{mid},{spread},{spread_bps},"
                f"1.0,1.0,0.0,session,bbo"
            )
        else:
            lines.append(
                f"{ts},2024-01-01T00:00:00Z,{coin},{bid},{ask},{mid},{spread},{spread_bps},"
                f"0.0,session,bbo"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_15s_csv(
    path: Path,
    coin: str,
    rows: int = 3,
    base_ts: int = 1_700_000_000_000,
    *,
    include_sizes: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "start_timestamp_ms,end_timestamp_ms,timestamp_iso,coin,open_mid,high_mid,low_mid,"
        "close_mid,avg_bid,avg_ask,avg_mid,avg_spread,avg_spread_bps,avg_bbo_imbalance,"
        "row_count,missing_seconds"
    )
    if include_sizes:
        header += ",bid_size,ask_size"
    lines = [header]
    for index in range(rows):
        start = base_ts + index * 15_000
        end = start + 14_999
        mid = 100.0 + index
        row = (
            f"{start},{end},2024-01-01T00:00:00Z,{coin},{mid},{mid + 0.5},{mid - 0.5},"
            f"{mid},{mid - 0.25},{mid + 0.25},{mid},0.5,5.0,0.0,15,0"
        )
        if include_sizes:
            row += ",1.0,1.0"
        lines.append(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_validation_report(
    reports_dir: Path,
    coin: str,
    interval: str,
    stamp: str,
    *,
    passed: bool = True,
    row_count: int = 10,
    warnings: list[str] | None = None,
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
        "missing_periods_count": 120,
        "warnings": warnings or ["120 missing or incomplete bar(s) detected"],
        "passed": passed,
        "generated_at": "2024-01-02T10:00:00Z",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_candidate_ledger(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "execution_enabled": False,
        "generated_at": "2024-01-02T10:00:00Z",
        "paper_only": True,
        "real_execution_ready": False,
        "row_count": len(rows),
        "rows": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_candidate_grid(path: Path, results: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "execution_enabled": False,
        "generated_at": "2024-01-02T10:00:00Z",
        "paper_only": True,
        "results": results,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_accepted_trades_analysis(
    path: Path,
    *,
    coin: str,
    trade_count: int = 14,
    cumulative_net: float = 31.28,
    suspicious_regimes: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "coin": coin,
        "generated_at": "2024-01-02T10:00:00Z",
        "trade_count": trade_count,
        "cumulative_net_return_bps": cumulative_net,
        "average_net_return_bps": cumulative_net / max(trade_count, 1),
        "win_rate": 0.64,
        "suspicious_regimes": suspicious_regimes or ["hour=3 clustered gains"],
        "paper_only": True,
        "execution_enabled": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    reports_root = tmp_path / "reports"
    data_root = tmp_path / "data" / "bid_ask_csv"
    output_dir = tmp_path / "reports" / "research_status"

    _write_1s_csv(data_root / "historical_1s" / "BTC_1s.csv", "BTC", rows=10, include_sizes=True)
    _write_1s_csv(data_root / "historical_1s" / "ETH_1s.csv", "ETH", rows=8, include_sizes=True)
    _write_15s_csv(data_root / "historical_15s" / "BTC_15s.csv", "BTC", rows=4, include_sizes=True)
    _write_15s_csv(data_root / "historical_15s" / "ETH_15s.csv", "ETH", rows=3, include_sizes=False)
    _write_1s_csv(data_root / "1m" / "BTC_1m.csv", "BTC", rows=2, include_sizes=False)

    validation_dir = data_root / "reports"
    _write_validation_report(validation_dir, "BTC", "1s", TEST_STAMP, row_count=10)
    _write_validation_report(validation_dir, "BTC", "15s", TEST_STAMP)
    _write_validation_report(validation_dir, "ETH", "1s", TEST_STAMP)

    ledger_rows = [
        {
            "candidate_name": "ETH_UP_E",
            "coin": "ETH",
            "interval": "15s",
            "horizon": 20,
            "feature_set": "candidate_e_depth_aware",
            "label_mode": "binary_tradeable_up",
            "filtered_paper_trades": 8,
            "cumulative_net_return_bps": 12.5,
            "average_net_return_bps": 1.56,
            "positive_after_costs": True,
            "source_file": "reports/bid_ask_walk_forward/walk_forward_ETH.json",
        },
        {
            "candidate_name": "BTC_DOWN_B",
            "coin": "BTC",
            "interval": "15s",
            "horizon": 20,
            "feature_set": "micro",
            "filtered_paper_trades": 25,
            "cumulative_net_return_bps": -50.0,
            "positive_after_costs": False,
            "source_file": "reports/bid_ask_walk_forward/walk_forward_BTC.json",
        },
    ]
    _write_candidate_ledger(reports_root / "candidate_ledger" / "candidate_ledger.json", ledger_rows)

    grid_results = [
        {
            "rank": 1,
            "config_id": "eth_up_sparse",
            "coin": "ETH",
            "interval": "15s",
            "horizon": 20,
            "feature_set": "candidate_e_depth_aware",
            "label_mode": "binary_tradeable_up",
            "paper_score": 42.5,
            "positive_after_costs": True,
            "filtered_paper_trades": 3,
            "status": "rejected",
            "too_few_trades": True,
        },
        {
            "rank": 2,
            "config_id": "eth_up_robust",
            "coin": "ETH",
            "interval": "15s",
            "horizon": 20,
            "feature_set": "candidate_e_depth_aware",
            "paper_score": 35.0,
            "positive_after_costs": True,
            "filtered_paper_trades": 12,
            "status": "completed",
        },
        {
            "rank": 3,
            "config_id": "btc_down_negative",
            "coin": "BTC",
            "paper_score": -100.0,
            "positive_after_costs": False,
            "filtered_paper_trades": 40,
        },
    ]
    _write_candidate_grid(
        reports_root / "candidate_grid" / f"candidate_grid_{TEST_STAMP}.json",
        grid_results,
    )

    _write_accepted_trades_analysis(
        reports_root / "accepted_trades_analysis" / f"accepted_trades_analysis_ETH_{TEST_STAMP}.json",
        coin="ETH",
        trade_count=14,
        cumulative_net=31.28,
        suspicious_regimes=["hour=3 clustered gains"],
    )
    _write_accepted_trades_analysis(
        reports_root / "accepted_trades_analysis" / f"accepted_trades_analysis_BTC_{TEST_STAMP}.json",
        coin="BTC",
        trade_count=5,
        cumulative_net=3.84,
        suspicious_regimes=[],
    )

    return {
        "reports_root": reports_root,
        "data_root": data_root,
        "output_dir": output_dir,
    }


class TestBuildResearchStatus:
    def test_report_generated(self, workspace: dict[str, Path]) -> None:
        result = builder.build_and_write_research_status(
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
        assert "# Bid/Ask Research Status and Decision Log" in md_latest.read_text(encoding="utf-8")

    def test_required_sections_present(self, workspace: dict[str, Path]) -> None:
        result = builder.build_and_write_research_status(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
        )
        markdown = Path(result["output_paths"]["markdown_latest"]).read_text(encoding="utf-8")
        for section in builder.REPORT_SECTIONS:
            assert f"## {section}" in markdown

    def test_missing_candidate_ledger_handled(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "reports" / "research_status"
        result = builder.build_and_write_research_status(
            reports_root=tmp_path / "missing_reports",
            data_root=tmp_path / "missing_data",
            output_dir=output_dir,
        )

        assert result["ledger_summary"]["ledger_row_count"] == 0
        assert result["ledger_summary"]["available"] is False
        assert (output_dir / builder.DEFAULT_LATEST_JSON_NAME).is_file()

    def test_detects_size_columns_in_csv(self, workspace: dict[str, Path]) -> None:
        result = builder.build_research_status(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
        )

        btc_1s = result["data_status"]["coins"]["BTC"]["1s"]
        eth_15s = result["data_status"]["coins"]["ETH"]["15s"]

        assert btc_1s["has_bid_size"] is True
        assert btc_1s["has_ask_size"] is True
        assert eth_15s["has_bid_size"] is False
        assert eth_15s["has_ask_size"] is False

    def test_flags_sparse_positives_under_10_trades(self, workspace: dict[str, Path]) -> None:
        result = builder.build_research_status(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
        )

        sparse_rows = [
            row
            for row in result["ledger_summary"]["top_by_paper_score"]
            if row.get("sparse_warning")
        ]
        assert sparse_rows
        assert any(row.get("trade_count", 99) < 10 for row in sparse_rows)

        ledger_positives = result["ledger_summary"]["positive_after_costs_ledger"]
        assert any(row.get("sparse_warning") for row in ledger_positives)

        markdown = builder.write_research_status_markdown(
            result,
            workspace["output_dir"],
            filename="sparse_check.md",
        ).read_text(encoding="utf-8")
        assert "SPARSE (<10 trades)" in markdown

    def test_safety_fields_included(self, workspace: dict[str, Path]) -> None:
        result = builder.build_research_status(
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

        markdown = builder.write_research_status_markdown(
            result,
            workspace["output_dir"],
            filename="safety_check.md",
        ).read_text(encoding="utf-8")
        assert "paper_only: true" in markdown
        assert "execution_enabled: false" in markdown
        assert "real_execution_ready: false" in markdown

    def test_accepted_trades_summary_included(self, workspace: dict[str, Path]) -> None:
        result = builder.build_research_status(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
        )

        eth = result["accepted_trades_analysis"]["by_coin"]["ETH"]
        assert eth["available"] is True
        assert eth["trade_count"] == 14
        assert eth["cumulative_net_return_bps"] == 31.28
        assert eth["suspicious_regime_count"] == 1

    def test_candidate_timeline_included(self, workspace: dict[str, Path]) -> None:
        result = builder.build_research_status(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
        )

        candidates = {entry["candidate"]: entry for entry in result["candidate_timeline"]}
        assert candidates["B"]["status"] == "rejected"
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

    def test_output_dir_must_be_research_status(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="research_status"):
            builder.build_and_write_research_status(
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
        builder.build_and_write_research_status(
            reports_root=workspace["reports_root"],
            data_root=workspace["data_root"],
            output_dir=workspace["output_dir"],
        )
