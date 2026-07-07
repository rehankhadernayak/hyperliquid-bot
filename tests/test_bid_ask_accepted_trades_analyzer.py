"""Focused offline tests for accepted paper trades analysis."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from bid_ask_accepted_trades_analyzer import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORTS_DIR,
    SAFETY_NOTE,
    analyze_accepted_trades,
    compute_loss_concentration,
    discover_latest_accepted_trades_csv,
    load_accepted_trades_csv,
    main,
    resolve_accepted_trades_csv,
    run_analysis,
    write_analysis_json,
    write_analysis_markdown,
)
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY

ANALYZER_FILE = Path(__file__).resolve().parent.parent / "bid_ask_accepted_trades_analyzer.py"

FULL_COLUMNS = (
    "coin",
    "window_index",
    "row_index",
    "timestamp_iso",
    "predicted_class",
    "actual_label",
    "confidence",
    "future_return_bps",
    "gross_return_bps",
    "cost_bps",
    "net_return_bps",
    "spread_bps",
)

MINIMAL_COLUMNS = (
    "coin",
    "timestamp_iso",
    "predicted_class",
    "net_return_bps",
)


def _make_trade_row(
    *,
    timestamp_iso: str = "2023-09-16T03:14:00Z",
    predicted_class: str = "DOWN",
    net_return_bps: float = -3.0,
    confidence: float = 0.96,
    spread_bps: float = 2.5,
    cost_bps: float = 5.5,
    coin: str = "BTC",
    window_index: int = 1,
    row_index: int = 0,
) -> dict:
    gross = abs(net_return_bps) + cost_bps if net_return_bps >= 0 else cost_bps - net_return_bps
    return {
        "coin": coin,
        "window_index": window_index,
        "row_index": row_index,
        "timestamp_iso": timestamp_iso,
        "predicted_class": predicted_class,
        "actual_label": predicted_class,
        "confidence": confidence,
        "future_return_bps": net_return_bps,
        "gross_return_bps": gross,
        "cost_bps": cost_bps,
        "net_return_bps": net_return_bps,
        "spread_bps": spread_bps,
    }


def _write_accepted_trades_csv(
    path: Path,
    rows: list[dict],
    *,
    columns: tuple[str, ...] = FULL_COLUMNS,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(columns)).to_csv(path, index=False)
    return path


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    return tmp_path / "walk_forward_reports"


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "accepted_trades_analysis"


class TestDiscoverLatestAcceptedTradesCsv:
    def test_latest_csv_selected(self, reports_dir: Path) -> None:
        older = reports_dir / "accepted_trades_BTC_20260101T000000.csv"
        newer = reports_dir / "accepted_trades_BTC_20260601T120000.csv"
        _write_accepted_trades_csv(older, [_make_trade_row()])
        _write_accepted_trades_csv(newer, [_make_trade_row(net_return_bps=2.0)])

        latest = discover_latest_accepted_trades_csv("BTC", reports_dir=reports_dir)
        assert latest == newer

    def test_ignores_multi_coin_filename(self, reports_dir: Path) -> None:
        single = reports_dir / "accepted_trades_BTC_20260601T120000.csv"
        multi = reports_dir / "accepted_trades_BTC_ETH_20260702T075024.csv"
        _write_accepted_trades_csv(single, [_make_trade_row(net_return_bps=1.0)])
        _write_accepted_trades_csv(multi, [_make_trade_row(net_return_bps=99.0)])

        latest = discover_latest_accepted_trades_csv("BTC", reports_dir=reports_dir)
        assert latest == single

    def test_missing_directory_returns_none(self, tmp_path: Path) -> None:
        assert discover_latest_accepted_trades_csv("BTC", reports_dir=tmp_path / "missing") is None


class TestResolveAcceptedTradesCsv:
    def test_explicit_csv_loaded(self, reports_dir: Path) -> None:
        explicit = reports_dir / "accepted_trades_BTC_custom.csv"
        _write_accepted_trades_csv(explicit, [_make_trade_row()])
        resolved = resolve_accepted_trades_csv(
            coin="BTC",
            csv_path=explicit,
            reports_dir=reports_dir,
            use_latest=False,
        )
        assert resolved == explicit

    def test_latest_used_when_csv_not_provided(self, reports_dir: Path) -> None:
        latest = reports_dir / "accepted_trades_BTC_20260601T120000.csv"
        _write_accepted_trades_csv(latest, [_make_trade_row()])
        resolved = resolve_accepted_trades_csv(
            coin="BTC",
            csv_path=None,
            reports_dir=reports_dir,
            use_latest=True,
        )
        assert resolved == latest


class TestAnalyzeAcceptedTrades:
    def test_by_hour_aggregation(self) -> None:
        rows = [
            _make_trade_row(timestamp_iso="2023-09-16T03:14:00Z", net_return_bps=4.0),
            _make_trade_row(timestamp_iso="2023-09-16T03:30:00Z", net_return_bps=2.0),
            _make_trade_row(timestamp_iso="2023-09-16T14:11:15Z", net_return_bps=-6.0),
        ]
        df = pd.DataFrame(rows, columns=list(FULL_COLUMNS))
        analysis = analyze_accepted_trades(df)

        assert analysis["net_return_by_hour"]["03"]["trades"] == 2
        assert analysis["net_return_by_hour"]["03"]["cumulative_net_return_bps"] == pytest.approx(6.0)
        assert analysis["net_return_by_hour"]["14"]["cumulative_net_return_bps"] == pytest.approx(-6.0)

    def test_by_class_aggregation(self) -> None:
        rows = [
            _make_trade_row(predicted_class="DOWN", net_return_bps=3.0),
            _make_trade_row(predicted_class="DOWN", net_return_bps=1.0),
            _make_trade_row(predicted_class="UP", net_return_bps=-4.0),
        ]
        df = pd.DataFrame(rows, columns=list(FULL_COLUMNS))
        analysis = analyze_accepted_trades(df)

        assert analysis["net_return_by_predicted_class"]["DOWN"]["cumulative_net_return_bps"] == pytest.approx(4.0)
        assert analysis["net_return_by_predicted_class"]["UP"]["cumulative_net_return_bps"] == pytest.approx(-4.0)

    def test_best_worst_trades(self) -> None:
        rows = [
            _make_trade_row(timestamp_iso="2023-09-16T01:00:00Z", net_return_bps=-10.0),
            _make_trade_row(timestamp_iso="2023-09-16T02:00:00Z", net_return_bps=5.0),
            _make_trade_row(timestamp_iso="2023-09-16T03:00:00Z", net_return_bps=-1.0),
        ]
        df = pd.DataFrame(rows, columns=list(FULL_COLUMNS))
        analysis = analyze_accepted_trades(df)

        assert analysis["best_trades"][0]["net_return_bps"] == pytest.approx(5.0)
        assert analysis["worst_trades"][0]["net_return_bps"] == pytest.approx(-10.0)

    def test_loss_concentration(self) -> None:
        net_returns = [5.0, 2.0, -1.0, -2.0, -3.0, -4.0, -20.0]
        concentration = compute_loss_concentration(net_returns, worst_n=5)

        assert concentration["worst_trade_count"] == 5
        assert concentration["total_loss_bps"] == pytest.approx(-30.0)
        assert concentration["worst_trades_pct_of_total_loss"] == pytest.approx(1.0)
        assert concentration["concentration_warning"] is True

    def test_missing_optional_columns_handled(self) -> None:
        rows = [
            {
                "coin": "BTC",
                "timestamp_iso": "2023-09-16T03:14:00Z",
                "predicted_class": "DOWN",
                "net_return_bps": -2.0,
            },
            {
                "coin": "BTC",
                "timestamp_iso": "2023-09-16T06:00:00Z",
                "predicted_class": "UP",
                "net_return_bps": 3.0,
            },
        ]
        path = Path("ignored.csv")
        df = pd.DataFrame(rows, columns=list(MINIMAL_COLUMNS))
        loaded = load_accepted_trades_csv(
            _write_accepted_trades_csv(path, rows, columns=MINIMAL_COLUMNS)
        )
        analysis = analyze_accepted_trades(loaded)

        assert len(loaded) == 2
        assert analysis["net_return_by_confidence_bucket"] == {}
        assert analysis["net_return_by_spread_bucket"] == {}
        assert analysis["net_return_by_cost_bucket"] == {}
        assert analysis["trade_count"] == 2

    def test_safety_flags_included(self) -> None:
        df = pd.DataFrame([_make_trade_row()], columns=list(FULL_COLUMNS))
        analysis = analyze_accepted_trades(df)

        assert analysis["paper_only"] is True
        assert analysis["execution_enabled"] is False
        assert analysis["safety_note"] == SAFETY_NOTE


class TestReportWriters:
    def test_write_json_and_markdown(
        self,
        output_dir: Path,
        reports_dir: Path,
    ) -> None:
        csv_path = _write_accepted_trades_csv(
            reports_dir / "accepted_trades_BTC_test.csv",
            [_make_trade_row(), _make_trade_row(net_return_bps=-5.0)],
        )
        analysis = run_analysis(csv_path, coin="BTC", output_dir=output_dir)

        json_path = Path(analysis["json_report_path"])
        markdown_path = Path(analysis["markdown_report_path"])
        assert json_path.is_file()
        assert markdown_path.is_file()

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["paper_only"] is True
        assert payload["execution_enabled"] is False
        assert "recommended_next_filters" in payload

        markdown = markdown_path.read_text(encoding="utf-8")
        assert "## Net return by hour (UTC)" in markdown
        assert "## Recommended next filters to test" in markdown
        assert SAFETY_NOTE in markdown


class TestCli:
    def test_main_with_explicit_csv(
        self,
        reports_dir: Path,
        output_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        csv_path = _write_accepted_trades_csv(
            reports_dir / "accepted_trades_BTC_cli.csv",
            [_make_trade_row(net_return_bps=1.5)],
        )
        exit_code = main(
            [
                "analyze",
                "--csv",
                str(csv_path),
                "--output-dir",
                str(output_dir),
                "--no-latest",
            ]
        )
        captured = capsys.readouterr().out

        assert exit_code == 0
        assert "json_report_path=" in captured
        assert "markdown_report_path=" in captured
        assert "trade_count=1" in captured
        assert SAFETY_NOTE in captured

    def test_main_latest(
        self,
        reports_dir: Path,
        output_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_accepted_trades_csv(
            reports_dir / "accepted_trades_BTC_20260601T120000.csv",
            [_make_trade_row(net_return_bps=-2.0), _make_trade_row(net_return_bps=-3.0)],
        )
        exit_code = main(
            [
                "analyze",
                "--coin",
                "BTC",
                "--reports-dir",
                str(reports_dir),
                "--output-dir",
                str(output_dir),
            ]
        )
        captured = capsys.readouterr().out

        assert exit_code == 0
        assert "trade_count=2" in captured
        assert "cumulative_net_return_bps=-5.00" in captured


class TestSafety:
    def test_no_wallet_signing_order_execution_code(self) -> None:
        source = ANALYZER_FILE.read_text(encoding="utf-8")
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
        source = ANALYZER_FILE.read_text(encoding="utf-8").lower()
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
        source = ANALYZER_FILE.read_text(encoding="utf-8").lower()
        assert "subprocess" not in source
        assert "os.system" not in source
        assert "popen" not in source

    def test_paper_only_constants(self) -> None:
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False

    def test_default_output_dirs(self) -> None:
        assert DEFAULT_REPORTS_DIR == Path("reports/bid_ask_walk_forward")
        assert DEFAULT_OUTPUT_DIR == Path("reports/accepted_trades_analysis")

    def test_custom_output_names(self, output_dir: Path) -> None:
        analysis = {
            "generated_at": "2026-07-07T12:00:00+00:00",
            "coin": "BTC",
            "trade_count": 1,
            "cumulative_net_return_bps": 1.0,
            "average_net_return_bps": 1.0,
            "win_rate": 1.0,
            "best_trades": [],
            "worst_trades": [],
            "net_return_by_hour": {},
            "net_return_by_predicted_class": {},
            "net_return_by_confidence_bucket": {},
            "net_return_by_spread_bucket": {},
            "net_return_by_cost_bucket": {},
            "loss_concentration": compute_loss_concentration([1.0]),
            "positive_concentration": {
                "best_bucket_group": None,
                "best_bucket_key": None,
                "best_bucket_net_bps": 0.0,
                "total_positive_net_bps": 0.0,
                "best_bucket_pct_of_positive_net": 0.0,
                "concentration_warning": False,
            },
            "suspicious_regimes": [],
            "recommended_next_filters": [],
            "paper_only": True,
            "execution_enabled": False,
            "safety_note": SAFETY_NOTE,
        }
        json_path = write_analysis_json(
            analysis,
            output_dir=output_dir,
            json_name="custom.json",
        )
        markdown_path = write_analysis_markdown(
            analysis,
            output_dir=output_dir,
            markdown_name="custom.md",
        )
        assert json_path.name == "custom.json"
        assert markdown_path.name == "custom.md"
