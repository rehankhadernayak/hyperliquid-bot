"""Focused offline tests for Candidate F robustness analysis."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from bid_ask_candidate_f_robustness_analyzer import (
    CANDIDATE_F_FEATURE_SET,
    DEFAULT_CANDIDATE_GRID_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WALK_FORWARD_DIR,
    SAFETY_NOTE,
    VERDICT_FRAGILE,
    VERDICT_HOUR_CONCENTRATED,
    VERDICT_INSUFFICIENT_TRADES,
    VERDICT_NO_POSITIVE,
    VERDICT_SPARSE,
    analyze_accepted_trades_robustness,
    analyze_config_robustness,
    build_candidate_f_robustness_report,
    compute_robustness_verdict,
    discover_latest_candidate_f_grid,
    find_matching_accepted_trades_csv,
    identify_positive_candidate_f_configs,
    main,
    run_candidate_f_robustness_analysis,
    write_robustness_json,
    write_robustness_markdown,
)
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY

ANALYZER_FILE = Path(__file__).resolve().parent.parent / "bid_ask_candidate_f_robustness_analyzer.py"


def _candidate_f_row(
    *,
    config_id: str = "ETH_UP_H20_C0.65_E2_M1.25_S8",
    average_net_return_bps: float = 1.6912,
    accepted_trades_count: int = 0,
    filtered_paper_trades: int = 17,
    coin: str = "ETH",
    trade_class: str = "UP",
) -> dict:
    return {
        "config_id": config_id,
        "coin": coin,
        "interval": "15s",
        "horizon": 20,
        "feature_set": CANDIDATE_F_FEATURE_SET,
        "label_mode": "binary_tradeable_up",
        "trade_class": trade_class,
        "min_confidence": 0.65,
        "min_edge_bps": 2.0,
        "cost_multiplier": 1.25,
        "max_spread_bps": 8.0,
        "average_net_return_bps": average_net_return_bps,
        "cumulative_net_return_bps": average_net_return_bps * filtered_paper_trades,
        "accepted_trades_count": accepted_trades_count,
        "filtered_paper_trades": filtered_paper_trades,
        "accepted_edge_predictions": filtered_paper_trades,
        "status": "ok",
        "config": {
            "coin": coin,
            "interval": "15s",
            "horizon": 20,
            "feature_set": CANDIDATE_F_FEATURE_SET,
            "label_mode": "binary_tradeable_up",
            "trade_class": trade_class,
            "min_confidence": 0.65,
            "min_edge_bps": 2.0,
            "cost_multiplier": 1.25,
            "max_spread_bps": 8.0,
            "source": "historical",
        },
    }


def _write_grid_json(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2026-07-17T08:06:53Z",
        "paper_only": True,
        "execution_enabled": False,
        "ranked": rows,
        "results": rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_walk_forward_match(
    walk_forward_dir: Path,
    *,
    stamp: str = "20260717T120000",
    coin: str = "ETH",
    config: dict | None = None,
) -> tuple[Path, Path]:
    walk_forward_dir.mkdir(parents=True, exist_ok=True)
    config = config or _candidate_f_row()["config"]
    report_path = walk_forward_dir / f"walk_forward_{coin}_{stamp}.json"
    csv_path = walk_forward_dir / f"accepted_trades_{coin}_{stamp}.csv"
    payload = {
        "generated_at": "2026-07-17T12:00:00Z",
        "coins": [coin],
        "feature_set": config["feature_set"],
        "label_mode": config["label_mode"],
        "horizon": config["horizon"],
        "interval": config["interval"],
        "min_confidence": config["min_confidence"],
        "min_edge_bps": config["min_edge_bps"],
        "cost_multiplier": config["cost_multiplier"],
        "max_spread_bps": config["max_spread_bps"],
        "trade_classes": [config["trade_class"]],
        "accepted_trades_csv_path": str(csv_path),
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("timestamp_iso,net_return_bps\n", encoding="utf-8")
    return report_path, csv_path


def _make_trade_rows(
    count: int,
    *,
    net_values: list[float] | None = None,
    hour: str = "03",
    day: str = "2024-06-10",
    include_regime: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        net = net_values[index] if net_values and index < len(net_values) else 1.0
        row = {
            "coin": "ETH",
            "window_index": 1,
            "row_index": index,
            "timestamp_iso": f"{day}T{hour}:00:{index:02d}Z",
            "predicted_class": "UP",
            "actual_label": "1",
            "confidence": 0.8,
            "future_return_bps": net + 5.0,
            "gross_return_bps": net + 5.0,
            "cost_bps": 5.0,
            "net_return_bps": net,
            "spread_bps": 2.0,
            "label_mode": "binary_tradeable_up",
            "predicted_edge_probability": 0.8,
            "true_edge_label": 1,
            "estimated_cost_bps": 5.0,
        }
        if include_regime:
            row.update(
                {
                    "low_volatility_regime": 1.0,
                    "high_volatility_regime": 0.0,
                    "tight_spread_regime": 1.0,
                    "wide_spread_regime": 0.0,
                }
            )
        rows.append(row)
    return rows


def _write_trades_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def grid_dir(tmp_path: Path) -> Path:
    return tmp_path / "candidate_grid"


@pytest.fixture
def walk_forward_dir(tmp_path: Path) -> Path:
    return tmp_path / "walk_forward"


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "candidate_f_robustness"


class TestDiscoveryAndFiltering:
    def test_no_candidate_f_configs_handled(self, grid_dir: Path, output_dir: Path, walk_forward_dir: Path) -> None:
        _write_grid_json(
            grid_dir / "candidate_grid_20260717T080653.json",
            [
                {
                    "config_id": "BTC_MICRO",
                    "feature_set": "microstructure",
                    "average_net_return_bps": 2.0,
                    "filtered_paper_trades": 20,
                }
            ],
        )

        report = run_candidate_f_robustness_analysis(
            grid_dir=grid_dir,
            walk_forward_dir=walk_forward_dir,
            output_dir=output_dir,
            write_stamped=False,
        )

        assert report["overall_verdict"] == VERDICT_NO_POSITIVE
        assert report["positive_configs"] == []
        assert report["robust_candidate"] is False

    def test_positive_config_detected(self, grid_dir: Path) -> None:
        row = _candidate_f_row()
        _write_grid_json(grid_dir / "candidate_grid_20260717T080653.json", [row])
        positives = identify_positive_candidate_f_configs([row])
        assert len(positives) == 1
        assert positives[0]["config_id"] == "ETH_UP_H20_C0.65_E2_M1.25_S8"
        assert positives[0]["average_net_return_bps"] == pytest.approx(1.6912)

    def test_discover_latest_candidate_f_grid(self, grid_dir: Path) -> None:
        older = grid_dir / "candidate_grid_20260701T000000.json"
        newer = grid_dir / "candidate_grid_20260717T080653.json"
        _write_grid_json(older, [_candidate_f_row(config_id="OLD")])
        _write_grid_json(newer, [_candidate_f_row()])
        latest = discover_latest_candidate_f_grid(grid_dir=grid_dir)
        assert latest == newer


class TestVerdicts:
    def test_sparse_verdict_for_17_trades(
        self,
        grid_dir: Path,
        walk_forward_dir: Path,
    ) -> None:
        row = _candidate_f_row()
        config = row["config"]
        _, csv_path = _write_walk_forward_match(walk_forward_dir, config=config)
        _write_trades_csv(csv_path, _make_trade_rows(17, net_values=[1.0] * 17))

        analysis = analyze_config_robustness(
            identify_positive_candidate_f_configs([row])[0],
            walk_forward_dir=walk_forward_dir,
        )
        assert analysis["verdict"] == VERDICT_SPARSE
        assert analysis["trade_analysis"]["trade_count"] == 17

    def test_outlier_driven_verdict(
        self,
        grid_dir: Path,
        walk_forward_dir: Path,
    ) -> None:
        row = _candidate_f_row(filtered_paper_trades=30)
        config = row["config"]
        _, csv_path = _write_walk_forward_match(walk_forward_dir, config=config)
        net_values = [20.0, 15.0, 10.0] + [0.1] * 27
        _write_trades_csv(csv_path, _make_trade_rows(30, net_values=net_values))

        analysis = analyze_config_robustness(
            identify_positive_candidate_f_configs([row])[0],
            walk_forward_dir=walk_forward_dir,
        )
        assert analysis["verdict"] == VERDICT_FRAGILE
        assert analysis["trade_analysis"]["top_3_trades_contribution_pct"] > 0.70

    def test_hour_concentration_verdict(
        self,
        grid_dir: Path,
        walk_forward_dir: Path,
    ) -> None:
        row = _candidate_f_row(filtered_paper_trades=30)
        config = row["config"]
        _, csv_path = _write_walk_forward_match(walk_forward_dir, config=config)
        rows = _make_trade_rows(25, net_values=[1.0] * 25, hour="03")
        rows.extend(_make_trade_rows(5, net_values=[1.0] * 5, hour="14"))
        _write_trades_csv(csv_path, rows)

        analysis = analyze_config_robustness(
            identify_positive_candidate_f_configs([row])[0],
            walk_forward_dir=walk_forward_dir,
        )
        assert analysis["verdict"] == VERDICT_HOUR_CONCENTRATED
        assert analysis["trade_analysis"]["dominant_utc_hour_trade_fraction"] > 0.60

    def test_missing_accepted_trades_handled(
        self,
        grid_dir: Path,
        walk_forward_dir: Path,
    ) -> None:
        row = _candidate_f_row()
        analysis = analyze_config_robustness(
            identify_positive_candidate_f_configs([row])[0],
            walk_forward_dir=walk_forward_dir,
        )
        assert analysis["matching_status"] == "not_found"
        assert analysis["accepted_trades_csv_path"] is None
        assert analysis["verdict"] == VERDICT_INSUFFICIENT_TRADES

    def test_compute_robustness_verdict_without_trades(self) -> None:
        assert (
            compute_robustness_verdict(trade_analysis=None, has_matching_trades=False)
            == VERDICT_INSUFFICIENT_TRADES
        )


class TestTradeAnalysis:
    def test_regime_columns_analyzed_when_present(self) -> None:
        rows = _make_trade_rows(3, include_regime=True)
        df = pd.DataFrame(rows)
        analysis = analyze_accepted_trades_robustness(df)
        assert "low_volatility_regime" in analysis["by_regime"]
        assert analysis["by_regime"]["low_volatility_regime"]["count"] == 3

    def test_by_hour_and_day_aggregation(self) -> None:
        rows = [
            *_make_trade_rows(2, net_values=[2.0, 4.0], hour="03", day="2024-06-10"),
            *_make_trade_rows(1, net_values=[-1.0], hour="14", day="2024-06-11"),
        ]
        analysis = analyze_accepted_trades_robustness(pd.DataFrame(rows))
        assert analysis["by_utc_hour"]["03"]["count"] == 2
        assert analysis["by_utc_hour"]["03"]["sum_net_bps"] == pytest.approx(6.0)
        assert analysis["by_day"]["2024-06-10"]["count"] == 2
        assert analysis["by_day"]["2024-06-11"]["count"] == 1


class TestReportWriters:
    def test_safety_fields_included(self, output_dir: Path) -> None:
        report = build_candidate_f_robustness_report(
            grid_path=Path("grid.json"),
            grid_rows=[_candidate_f_row()],
            positive_configs=identify_positive_candidate_f_configs([_candidate_f_row()]),
            config_analyses=[],
            generated_at="2026-07-20T12:00:00Z",
        )
        assert report["paper_only"] is True
        assert report["execution_enabled"] is False
        assert report["real_execution_ready"] is False
        assert report["robust_candidate"] is False
        assert report["safety_note"] == SAFETY_NOTE

        write_robustness_json(report, output_dir=output_dir, json_name="test.json")
        write_robustness_markdown(report, output_dir=output_dir, markdown_name="test.md")
        payload = json.loads((output_dir / "test.json").read_text(encoding="utf-8"))
        markdown = (output_dir / "test.md").read_text(encoding="utf-8")

        assert payload["real_execution_ready"] is False
        assert "## Safety statement" in markdown
        assert "## Verdict" in markdown
        assert "## Next recommended action" in markdown

    def test_end_to_end_report_paths(
        self,
        grid_dir: Path,
        walk_forward_dir: Path,
        output_dir: Path,
    ) -> None:
        _write_grid_json(grid_dir / "candidate_grid_20260717T080653.json", [_candidate_f_row()])
        report = run_candidate_f_robustness_analysis(
            grid_dir=grid_dir,
            walk_forward_dir=walk_forward_dir,
            output_dir=output_dir,
            write_stamped=True,
        )
        assert Path(report["json_report_path"]).name == "candidate_f_robustness_latest.json"
        assert Path(report["markdown_report_path"]).name == "candidate_f_robustness_latest.md"
        stamped_json = list(output_dir.glob("candidate_f_robustness_202*.json"))
        stamped_md = list(output_dir.glob("candidate_f_robustness_202*.md"))
        assert len(stamped_json) == 1
        assert len(stamped_md) == 1


class TestMatching:
    def test_find_matching_accepted_trades_csv(self, walk_forward_dir: Path) -> None:
        config = _candidate_f_row()["config"]
        _, csv_path = _write_walk_forward_match(walk_forward_dir, config=config)
        matched = find_matching_accepted_trades_csv(config, walk_forward_dir=walk_forward_dir)
        assert matched == csv_path


class TestCli:
    def test_main_runs(
        self,
        grid_dir: Path,
        walk_forward_dir: Path,
        output_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_grid_json(grid_dir / "candidate_grid_20260717T080653.json", [_candidate_f_row()])
        exit_code = main(
            [
                "--grid-dir",
                str(grid_dir),
                "--walk-forward-dir",
                str(walk_forward_dir),
                "--output-dir",
                str(output_dir),
                "--no-stamped",
            ]
        )
        captured = capsys.readouterr().out
        assert exit_code == 0
        assert "overall_verdict=" in captured
        assert SAFETY_NOTE in captured


class TestSafety:
    def test_forbidden_imports_absent(self) -> None:
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

    def test_no_network_subprocess_or_shell(self) -> None:
        source = ANALYZER_FILE.read_text(encoding="utf-8").lower()
        forbidden = (
            "requests.get",
            "requests.post",
            "urllib.request",
            "http.client",
            "aiohttp",
            "websocket",
            "boto3",
            "subprocess",
            "os.system",
            "popen",
        )
        for pattern in forbidden:
            assert pattern not in source

    def test_paper_only_constants(self) -> None:
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False

    def test_default_paths(self) -> None:
        assert DEFAULT_CANDIDATE_GRID_DIR == Path("reports/candidate_grid")
        assert DEFAULT_WALK_FORWARD_DIR == Path("reports/bid_ask_walk_forward")
        assert DEFAULT_OUTPUT_DIR == Path("reports/candidate_f_robustness")
