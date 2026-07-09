"""Tests for bid_ask_feature_audit (offline temp dirs, no network)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

import bid_ask_feature_audit as audit
from bid_ask_resampler import RESAMPLED_COLUMNS
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY

AUDIT_FILE = Path(__file__).resolve().parent.parent / "bid_ask_feature_audit.py"


def _write_resampled_csv(
    path: Path,
    *,
    rows: int = 80,
    coin: str = "BTC",
    base_ts: int = 1_700_000_000_000,
    include_imbalance: bool = True,
    include_sizes: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index in range(rows):
        start = base_ts + index * 15_000
        end = start + 14_999
        mid = 100.0 + index * 0.01
        spread_bps = 2.0 + (index % 5) * 0.2
        row: dict[str, object] = {
            "start_timestamp_ms": start,
            "end_timestamp_ms": end,
            "timestamp_iso": "2024-01-01T00:00:00Z",
            "coin": coin,
            "open_mid": mid,
            "high_mid": mid + 0.5,
            "low_mid": mid - 0.5,
            "close_mid": mid,
            "avg_bid": mid - 0.25,
            "avg_ask": mid + 0.25,
            "avg_mid": mid,
            "avg_spread": 0.5,
            "avg_spread_bps": spread_bps,
            "avg_bbo_imbalance": 0.05 * (index % 7 - 3) if include_imbalance else None,
            "row_count": 15,
            "missing_seconds": 0,
        }
        if include_sizes:
            row["bid_size"] = 1.0 + index * 0.01
            row["ask_size"] = 1.2 + index * 0.01
        records.append(row)

    pd.DataFrame(records, columns=list(RESAMPLED_COLUMNS)).to_csv(path, index=False)


def _write_accepted_trades_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "coin": "BTC",
            "window_index": 0,
            "row_index": 10,
            "timestamp_iso": "2024-01-01T08:00:00Z",
            "predicted_class": "DOWN",
            "actual_label": "DOWN",
            "confidence": 0.7,
            "future_return_bps": -5.0,
            "gross_return_bps": -3.0,
            "cost_bps": 2.0,
            "net_return_bps": -5.0,
            "spread_bps": 2.0,
        },
        {
            "coin": "BTC",
            "window_index": 1,
            "row_index": 20,
            "timestamp_iso": "2024-01-01T09:00:00Z",
            "predicted_class": "DOWN",
            "actual_label": "FLAT",
            "confidence": 0.66,
            "future_return_bps": 1.0,
            "gross_return_bps": 3.0,
            "cost_bps": 2.0,
            "net_return_bps": 1.0,
            "spread_bps": 8.0,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "bid_ask_csv"
    reports_root = tmp_path / "reports"
    output_dir = reports_root / "feature_audit"
    csv_path = data_root / "historical_15s" / "BTC_15s.csv"
    _write_resampled_csv(csv_path)
    return {
        "data_root": data_root,
        "reports_root": reports_root,
        "output_dir": output_dir,
    }


class TestSyntheticAudit:
    def test_synthetic_csv_audit_works(self, workspace: dict[str, Path]) -> None:
        report = audit.run_feature_audit(
            coin="BTC",
            interval="15s",
            source="historical",
            data_root=workspace["data_root"],
            reports_root=workspace["reports_root"],
            output_dir=workspace["output_dir"],
            generated_at="2026-07-09T10:00:00Z",
            json_name="feature_audit_test.json",
            markdown_name="feature_audit_test.md",
        )

        assert report["coin"] == "BTC"
        assert report["row_count"] == 80
        assert report["feature_stats"]
        assert "spread_bps" in report["feature_stats"]
        assert "feature_correlations" in report
        assert report["label_diagnostics"]["overall_class_balance"]

    def test_missing_optional_columns_handled(self, tmp_path: Path) -> None:
        data_root = tmp_path / "bid_ask_csv"
        csv_path = data_root / "historical_15s" / "ETH_15s.csv"
        _write_resampled_csv(csv_path, coin="ETH", include_imbalance=True, include_sizes=False)

        df, _ = audit.load_resampled_csv("ETH", "15s", data_root, source="historical")
        featured, skipped = audit.compute_candidate_features(df, interval="15s")

        assert "spread_bps" in featured.columns
        assert "microprice" not in featured.columns
        assert "bid_size" not in featured.columns
        assert "microprice" in skipped
        assert "bid_size" in skipped

        report = audit.build_feature_audit_report(
            coin="ETH",
            interval="15s",
            source="historical",
            data_root=data_root,
            reports_root=tmp_path / "reports",
            generated_at="2026-07-09T10:00:00Z",
        )
        assert "microprice" in report["skipped_features"]
        assert not any(
            column in report["feature_stats"]
            for column in audit.FEATURE_FAMILIES["liquidity_size"]
        )

    def test_optional_size_columns_enable_microprice(self, tmp_path: Path) -> None:
        data_root = tmp_path / "bid_ask_csv"
        csv_path = data_root / "historical_15s" / "BTC_15s.csv"
        _write_resampled_csv(csv_path, include_sizes=True)

        df, _ = audit.load_resampled_csv("BTC", "15s", data_root, source="historical")
        df["bid_size"] = 1.0
        df["ask_size"] = 1.0
        featured, skipped = audit.compute_candidate_features(df, interval="15s")

        assert "microprice" in featured.columns
        assert "total_top_size" in featured.columns
        assert "microprice" not in skipped
        assert "bid_size" not in skipped

    def test_output_json_and_markdown_written(self, workspace: dict[str, Path]) -> None:
        report = audit.run_feature_audit(
            coin="BTC",
            interval="15s",
            source="historical",
            data_root=workspace["data_root"],
            reports_root=workspace["reports_root"],
            output_dir=workspace["output_dir"],
            generated_at="2026-07-09T10:00:00Z",
            json_name="feature_audit_test.json",
            markdown_name="feature_audit_test.md",
        )

        json_path = Path(report["output_paths"]["json"])
        md_path = Path(report["output_paths"]["markdown"])
        assert json_path.is_file()
        assert md_path.is_file()

        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["paper_only"] is True
        assert loaded["feature_stats"]

        markdown = md_path.read_text(encoding="utf-8")
        assert "# Bid/Ask Candidate C Feature Audit" in markdown
        assert "spread_bps" in markdown

    def test_accepted_trades_diagnostics_when_csv_exists(
        self,
        workspace: dict[str, Path],
    ) -> None:
        wf_dir = workspace["reports_root"] / "bid_ask_walk_forward"
        _write_accepted_trades_csv(wf_dir / "accepted_trades_BTC_test.csv")

        report = audit.run_feature_audit(
            coin="BTC",
            interval="15s",
            source="historical",
            data_root=workspace["data_root"],
            reports_root=workspace["reports_root"],
            output_dir=workspace["output_dir"],
            generated_at="2026-07-09T10:00:00Z",
        )

        accepted = report["accepted_trades_diagnostics"]
        assert accepted is not None
        assert accepted["trade_count"] == 2
        assert accepted["net_return_summary"]["count"] == 2

    def test_cli_main_writes_reports(self, workspace: dict[str, Path]) -> None:
        exit_code = audit.main(
            [
                "audit",
                "--coin",
                "BTC",
                "--source",
                "historical",
                "--interval",
                "15s",
                "--data-root",
                str(workspace["data_root"]),
                "--reports-root",
                str(workspace["reports_root"]),
                "--output-dir",
                str(workspace["output_dir"]),
                "--json-name",
                "cli_audit.json",
                "--markdown-name",
                "cli_audit.md",
            ]
        )
        assert exit_code == 0
        assert (workspace["output_dir"] / "cli_audit.json").is_file()
        assert (workspace["output_dir"] / "cli_audit.md").is_file()


class TestSafetyFields:
    def test_safety_fields_present(self, workspace: dict[str, Path]) -> None:
        report = audit.run_feature_audit(
            coin="BTC",
            interval="15s",
            source="historical",
            data_root=workspace["data_root"],
            reports_root=workspace["reports_root"],
            output_dir=workspace["output_dir"],
            generated_at="2026-07-09T10:00:00Z",
        )

        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False
        safety = report["safety_flags"]
        assert safety["paper_only"] is True
        assert safety["execution_enabled"] is False
        assert safety["real_execution_ready"] is False
        assert safety["read_only"] is True
        assert safety["no_network_calls"] is True
        assert safety["no_subprocess_calls"] is True
        assert safety["research_only"] is True
        assert safety["safety_statement"]
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False


def _block_network_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network/subprocess call attempted")

    monkeypatch.setattr("subprocess.run", _blocked)
    monkeypatch.setattr("subprocess.Popen", _blocked)
    monkeypatch.setattr("boto3.client", _blocked)


class TestSafety:
    def test_no_forbidden_imports(self) -> None:
        source = AUDIT_FILE.read_text(encoding="utf-8")
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
        source = AUDIT_FILE.read_text(encoding="utf-8").lower()
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
        source = AUDIT_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_roots.add(alias.name.split(".", 1)[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        forbidden_imports = {"subprocess", "boto3", "requests", "urllib", "socket", "web3"}
        assert forbidden_imports.isdisjoint(imported_roots)
        lowered = source.lower()
        for pattern in ("os.system", "popen("):
            assert pattern not in lowered

    def test_no_network_subprocess_at_runtime(
        self,
        workspace: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _block_network_subprocess(monkeypatch)
        audit.run_feature_audit(
            coin="BTC",
            interval="15s",
            source="historical",
            data_root=workspace["data_root"],
            reports_root=workspace["reports_root"],
            output_dir=workspace["output_dir"],
            generated_at="2026-07-09T10:00:00Z",
        )
