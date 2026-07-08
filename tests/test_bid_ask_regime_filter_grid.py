"""Focused offline tests for regime filter grid runner."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from bid_ask_regime_filter_grid import (
    DEFAULT_EXCLUDE_HOURS_LIST,
    DEFAULT_MAX_COST_BPS_LIST,
    DEFAULT_MIN_TRADES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORTS_DIR,
    DEFAULT_SPREAD_BUCKET_LIST,
    EXPLORATORY_DISCLAIMER,
    SAFETY_NOTE,
    RegimeFilterSpec,
    apply_regime_filter,
    build_filter_combinations,
    compute_diagnostic_score,
    enrich_accepted_trades,
    evaluate_filter_variant,
    evaluate_regime_filter_grid,
    main,
    rank_filter_variants,
    run_regime_filter_grid,
    write_regime_filter_csv,
    write_regime_filter_json,
    write_regime_filter_markdown,
)
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY

GRID_FILE = Path(__file__).resolve().parent.parent / "bid_ask_regime_filter_grid.py"

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


def _write_accepted_trades_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(FULL_COLUMNS)).to_csv(path, index=False)
    return path


@pytest.fixture
def sample_rows() -> list[dict]:
    return [
        _make_trade_row(
            timestamp_iso="2023-09-16T03:14:00Z",
            net_return_bps=-4.0,
            cost_bps=6.0,
            spread_bps=2.0,
            row_index=0,
        ),
        _make_trade_row(
            timestamp_iso="2023-09-16T05:30:00Z",
            net_return_bps=-17.0,
            cost_bps=7.0,
            spread_bps=8.0,
            row_index=1,
        ),
        _make_trade_row(
            timestamp_iso="2023-09-16T10:00:00Z",
            net_return_bps=5.0,
            cost_bps=4.5,
            spread_bps=1.5,
            row_index=2,
        ),
        _make_trade_row(
            timestamp_iso="2023-09-16T12:00:00Z",
            net_return_bps=3.0,
            cost_bps=5.0,
            spread_bps=9.0,
            row_index=3,
        ),
        _make_trade_row(
            timestamp_iso="2023-09-16T22:45:00Z",
            net_return_bps=-7.0,
            cost_bps=6.5,
            spread_bps=7.0,
            row_index=4,
        ),
        _make_trade_row(
            timestamp_iso="2023-09-17T08:00:00Z",
            net_return_bps=2.0,
            cost_bps=4.0,
            spread_bps=1.0,
            row_index=5,
        ),
    ]


@pytest.fixture
def sample_df(sample_rows: list[dict]) -> pd.DataFrame:
    return enrich_accepted_trades(pd.DataFrame(sample_rows))


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    return tmp_path / "walk_forward_reports"


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "regime_filter_grid"


class TestBuildFilterCombinations:
    def test_default_combination_count(self) -> None:
        specs = build_filter_combinations()
        expected = (
            len(DEFAULT_EXCLUDE_HOURS_LIST)
            * len(DEFAULT_MAX_COST_BPS_LIST)
            * len(DEFAULT_SPREAD_BUCKET_LIST)
        )
        assert len(specs) == expected
        assert len(specs) == 72


class TestNoFilterBaseline:
    def test_baseline_keeps_all_trades(self, sample_df: pd.DataFrame) -> None:
        spec = RegimeFilterSpec(
            exclude_hours=frozenset(),
            max_cost_bps=None,
            spread_bucket=None,
        )
        result = evaluate_filter_variant(sample_df, spec, min_trades=1)
        assert result["trades_before"] == 6
        assert result["trades_after"] == 6
        assert result["trades_removed"] == 0
        assert result["cumulative_net_return_bps_before"] == pytest.approx(-18.0)
        assert result["cumulative_net_return_bps_after"] == pytest.approx(-18.0)


class TestExcludeHourFilter:
    def test_exclude_hour_removes_correct_rows(self, sample_df: pd.DataFrame) -> None:
        spec = RegimeFilterSpec(
            exclude_hours=frozenset({"03", "05", "22"}),
            max_cost_bps=None,
            spread_bucket=None,
        )
        filtered = apply_regime_filter(sample_df, spec)
        assert len(filtered) == 3
        assert set(filtered["_hour"].tolist()) == {"10", "12", "08"}

        result = evaluate_filter_variant(sample_df, spec, min_trades=1)
        assert result["trades_after"] == 3
        assert result["removed_loss_bps"] == pytest.approx(-28.0)
        assert result["removed_gain_bps"] == pytest.approx(0.0)


class TestMaxCostFilter:
    def test_max_cost_removes_high_cost_rows(self, sample_df: pd.DataFrame) -> None:
        spec = RegimeFilterSpec(
            exclude_hours=frozenset(),
            max_cost_bps=5.0,
            spread_bucket=None,
        )
        filtered = apply_regime_filter(sample_df, spec)
        assert len(filtered) == 3
        assert all(float(value) <= 5.0 for value in filtered["_cost_bps"].tolist())

        result = evaluate_filter_variant(sample_df, spec, min_trades=1)
        assert result["trades_after"] == 3
        assert result["trades_removed"] == 3


class TestMinTradesFilter:
    def test_ranking_excludes_variants_below_min_trades(self, sample_df: pd.DataFrame) -> None:
        variants = evaluate_regime_filter_grid(
            sample_df,
            exclude_hours_list=("", "3,5,22"),
            max_cost_bps_list=("",),
            spread_bucket_list=("",),
            min_trades=5,
        )
        ranked = rank_filter_variants(variants, min_trades=5)
        assert all(int(row["trades_after"]) >= 5 for row in ranked)
        assert len(ranked) < len(variants)


class TestDiagnosticScore:
    def test_penalizes_too_few_trades(self) -> None:
        sufficient = compute_diagnostic_score(
            trades_before=10,
            trades_after=6,
            cumulative_net_return_bps_after=4.0,
            average_net_return_bps_after=0.67,
            removed_loss_bps=-8.0,
            removed_gain_bps=2.0,
            min_trades=5,
        )
        insufficient = compute_diagnostic_score(
            trades_before=10,
            trades_after=3,
            cumulative_net_return_bps_after=4.0,
            average_net_return_bps_after=1.33,
            removed_loss_bps=-8.0,
            removed_gain_bps=2.0,
            min_trades=5,
        )
        assert sufficient["too_few_trades"] is False
        assert insufficient["too_few_trades"] is True
        assert insufficient["too_few_trades_penalty"] == 500.0
        assert sufficient["diagnostic_score"] > insufficient["diagnostic_score"]


class TestOverfitWarning:
    def test_exploratory_overfit_warning_present(self, sample_df: pd.DataFrame) -> None:
        variants = evaluate_regime_filter_grid(
            sample_df,
            exclude_hours_list=("",),
            max_cost_bps_list=("",),
            spread_bucket_list=("",),
            min_trades=1,
        )
        assert all(variant["exploratory_overfit_warning"] is True for variant in variants)


class TestOutputFiles:
    def test_output_files_written(
        self,
        reports_dir: Path,
        output_dir: Path,
        sample_rows: list[dict],
    ) -> None:
        csv_path = _write_accepted_trades_csv(
            reports_dir / "accepted_trades_BTC_20260708T100900.csv",
            sample_rows,
        )
        report = run_regime_filter_grid(
            coin="BTC",
            csv_path=csv_path,
            reports_dir=reports_dir,
            output_dir=output_dir,
            use_latest=False,
            exclude_hours_list=("", "3"),
            max_cost_bps_list=("", "5"),
            spread_bucket_list=("",),
            min_trades=1,
            top_n=5,
        )

        json_path = Path(str(report["json_report_path"]))
        csv_report_path = Path(str(report["csv_report_path"]))
        markdown_path = Path(str(report["markdown_report_path"]))

        assert json_path.is_file()
        assert csv_report_path.is_file()
        assert markdown_path.is_file()
        assert str(json_path.parent) == str(output_dir)

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["paper_only"] is True
        assert payload["execution_enabled"] is False
        assert payload["exploratory_overfit_warning"] is True

        markdown = markdown_path.read_text(encoding="utf-8")
        assert EXPLORATORY_DISCLAIMER in markdown
        assert SAFETY_NOTE in markdown

        with csv_report_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert "diagnostic_score" in rows[0]


class TestCli:
    def test_main_run(
        self,
        reports_dir: Path,
        output_dir: Path,
        sample_rows: list[dict],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        csv_path = _write_accepted_trades_csv(
            reports_dir / "accepted_trades_BTC_cli.csv",
            sample_rows,
        )
        exit_code = main(
            [
                "run",
                "--csv",
                str(csv_path),
                "--output-dir",
                str(output_dir),
                "--no-latest",
                "--exclude-hours-list",
                ";3",
                "--max-cost-bps-list",
                ";5",
                "--spread-bucket-list",
                ";",
                "--min-trades",
                "1",
            ]
        )
        captured = capsys.readouterr().out

        assert exit_code == 0
        assert "json_report_path=" in captured
        assert "csv_report_path=" in captured
        assert "markdown_report_path=" in captured
        assert EXPLORATORY_DISCLAIMER in captured


class TestSafety:
    def test_no_wallet_signing_order_execution_code(self) -> None:
        source = GRID_FILE.read_text(encoding="utf-8")
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
        source = GRID_FILE.read_text(encoding="utf-8").lower()
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
        source = GRID_FILE.read_text(encoding="utf-8").lower()
        assert "subprocess" not in source
        assert "os.system" not in source
        assert "popen" not in source

    def test_paper_only_constants(self) -> None:
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False

    def test_default_output_dirs(self) -> None:
        assert DEFAULT_REPORTS_DIR == Path("reports/bid_ask_walk_forward")
        assert DEFAULT_OUTPUT_DIR == Path("reports/regime_filter_grid")


class TestReportWriters:
    def test_custom_report_names(self, output_dir: Path) -> None:
        report = {
            "run_timestamp": "2026-07-08T10:09:00+00:00",
            "coin": "BTC",
            "ranked": [
                {
                    "rank": 1,
                    "filter_id": "exclude_hours=none|max_cost_bps=none|spread_bucket=none",
                    "exclude_hours": [],
                    "max_cost_bps": None,
                    "spread_bucket": None,
                    "diagnostic_score": 100.0,
                    "trades_before": 6,
                    "trades_after": 6,
                    "trades_removed": 0,
                    "cumulative_net_return_bps_before": -18.0,
                    "cumulative_net_return_bps_after": -18.0,
                    "average_net_return_bps_after": -3.0,
                    "win_rate_after": 0.5,
                    "removed_loss_bps": 0.0,
                    "removed_gain_bps": 0.0,
                    "positive_after_costs": False,
                    "concentration_warning": False,
                    "exploratory_overfit_warning": True,
                    "meets_min_trades": True,
                    "real_execution_ready": False,
                }
            ],
            "top_n": [],
            "baseline_variant": None,
            "best_variant": None,
            "paper_only": True,
            "execution_enabled": False,
            "safety_note": SAFETY_NOTE,
            "exploratory_disclaimer": EXPLORATORY_DISCLAIMER,
            "safety_statement": "paper-only",
        }
        json_path = write_regime_filter_json(report, output_dir=output_dir)
        csv_path = write_regime_filter_csv(report, output_dir=output_dir)
        markdown_path = write_regime_filter_markdown(report, output_dir=output_dir)
        assert json_path.name.startswith("regime_filter_grid_BTC_")
        assert csv_path.name.startswith("regime_filter_grid_BTC_")
        assert markdown_path.name.startswith("regime_filter_grid_BTC_")
