"""Focused offline tests for paper decision filter / signal gate."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from bid_ask_decision_logger import DECISION_COLUMNS
from bid_ask_decision_filter import (
    DEFAULT_DECISIONS_DIR,
    DEFAULT_OUTPUT_DIR,
    FILTER_STATUS_ACCEPTED,
    FILTER_STATUS_BLOCKED,
    SAFETY_NOTE,
    apply_decision_filter,
    default_filter_config,
    discover_latest_decision_log,
    evaluate_filtered_decisions,
    load_decisions,
    main,
    run_filter,
    write_filter_report_json,
    write_filter_report_markdown,
)
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY

FILTER_FILE = Path(__file__).resolve().parent.parent / "bid_ask_decision_filter.py"


def _make_decision_row(
    *,
    decision_id: str = "abc123",
    coin: str = "BTC",
    model_signal: str = "UP",
    baseline_signal: str = "FLAT",
    actual_label: str = "UP",
    spread_bps: float = 1.5,
    confidence: float = 0.8,
    signal_correct: bool | None = None,
    baseline_correct: bool | None = None,
    features_json: str = '{"mid_return_1": 1.0}',
) -> dict:
    if signal_correct is None:
        signal_correct = model_signal == actual_label
    if baseline_correct is None:
        baseline_correct = baseline_signal == actual_label
    return {
        "decision_id": decision_id,
        "generated_at": "2024-06-01T00:00:00+00:00",
        "coin": coin,
        "interval": "15s",
        "source_csv": "data/bid_ask_csv/15s/BTC_15s.csv",
        "source_row_timestamp": 1_718_000_040_000,
        "source_row_iso": "2024-06-10T00:00:00Z",
        "best_bid_or_avg_bid": 100.0,
        "best_ask_or_avg_ask": 101.0,
        "mid_or_close_mid": 100.5,
        "spread_or_avg_spread": 1.0,
        "spread_bps_or_avg_spread_bps": spread_bps,
        "model_signal": model_signal,
        "baseline_signal": baseline_signal,
        "confidence": confidence,
        "horizon": 1,
        "future_return_bps": 2.0,
        "actual_label": actual_label,
        "signal_correct": signal_correct,
        "baseline_correct": baseline_correct,
        "features_json": features_json,
        "paper_only": True,
        "execution_enabled": False,
        "notes": "test",
    }


def _write_decision_log(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(DECISION_COLUMNS)).to_csv(path, index=False)
    return path


def _test_config(**overrides) -> dict:
    config = default_filter_config()
    config["min_sample_size"] = 1
    config["require_model_beats_baseline"] = False
    config.update(overrides)
    return config


def _make_accepted_rows(count: int) -> list[dict]:
    return [
        _make_decision_row(
            decision_id=f"row-{index}",
            model_signal="UP",
            actual_label="UP" if index % 2 == 0 else "DOWN",
            baseline_signal="FLAT",
            signal_correct=index % 2 == 0,
            baseline_correct=False,
        )
        for index in range(count)
    ]


@pytest.fixture
def decisions_dir(tmp_path: Path) -> Path:
    return tmp_path / "reports" / "bid_ask_decisions"


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "reports" / "bid_ask_decision_filters"


class TestDiscoverLatestDecisionLog:
    def test_discover_latest_log(self, decisions_dir: Path) -> None:
        older = decisions_dir / "bid_ask_decisions_20240101T000000Z.csv"
        newer = decisions_dir / "bid_ask_decisions_20240601T120000Z.csv"
        _write_decision_log(older, [_make_decision_row(decision_id="old")])
        _write_decision_log(newer, [_make_decision_row(decision_id="new")])

        latest = discover_latest_decision_log(decisions_dir)
        assert latest == newer

    def test_missing_directory_returns_none(self, tmp_path: Path) -> None:
        assert discover_latest_decision_log(tmp_path / "missing") is None


class TestLoadDecisions:
    def test_load_decisions(self, decisions_dir: Path) -> None:
        path = _write_decision_log(
            decisions_dir / "bid_ask_decisions_test.csv",
            [_make_decision_row()],
        )
        df = load_decisions(path)
        assert len(df) == 1
        assert list(df.columns) == list(DECISION_COLUMNS)

    def test_rejects_non_paper_log(self, decisions_dir: Path) -> None:
        row = _make_decision_row()
        row["paper_only"] = False
        path = _write_decision_log(
            decisions_dir / "bid_ask_decisions_bad.csv",
            [row],
        )
        with pytest.raises(ValueError, match="paper-only"):
            load_decisions(path)


class TestDefaultFilterConfig:
    def test_default_values(self) -> None:
        config = default_filter_config()
        assert config["max_spread_bps"] == 10.0
        assert config["allowed_coins"] == ("BTC", "ETH", "SOL")
        assert config["blocked_signals"] == ("baseline_fallback",)
        assert config["min_confidence"] is None
        assert config["require_model_beats_baseline"] is True
        assert config["min_sample_size"] == 100
        assert config["block_if_missing_features"] is True


class TestApplyDecisionFilter:
    def test_spread_too_wide_block(self) -> None:
        rows = [
            _make_decision_row(decision_id="wide", spread_bps=15.0),
            _make_decision_row(decision_id="ok", spread_bps=5.0),
        ]
        df = apply_decision_filter(pd.DataFrame(rows), _test_config())
        blocked = df[df["decision_id"] == "wide"].iloc[0]
        accepted = df[df["decision_id"] == "ok"].iloc[0]
        assert blocked["filter_status"] == FILTER_STATUS_BLOCKED
        assert blocked["block_reason"] == "spread_too_wide"
        assert accepted["filter_status"] == FILTER_STATUS_ACCEPTED

    def test_coin_not_allowed_block(self) -> None:
        rows = [
            _make_decision_row(decision_id="doge", coin="DOGE"),
            _make_decision_row(decision_id="btc", coin="BTC"),
        ]
        df = apply_decision_filter(pd.DataFrame(rows), _test_config())
        assert df.loc[df["decision_id"] == "doge", "block_reason"].iloc[0] == "coin_not_allowed"
        assert df.loc[df["decision_id"] == "btc", "filter_status"].iloc[0] == FILTER_STATUS_ACCEPTED

    def test_baseline_fallback_blocked_by_default(self) -> None:
        rows = [
            _make_decision_row(decision_id="fallback", model_signal="baseline_fallback"),
            _make_decision_row(decision_id="model", model_signal="UP"),
        ]
        config = _test_config(blocked_signals=("baseline_fallback",))
        df = apply_decision_filter(pd.DataFrame(rows), config)
        assert (
            df.loc[df["decision_id"] == "fallback", "block_reason"].iloc[0]
            == "signal_blocked"
        )
        assert df.loc[df["decision_id"] == "model", "filter_status"].iloc[0] == FILTER_STATUS_ACCEPTED

    def test_missing_features_block(self) -> None:
        rows = [
            _make_decision_row(decision_id="missing", features_json="{}"),
            _make_decision_row(decision_id="present", features_json='{"a": 1}'),
        ]
        df = apply_decision_filter(pd.DataFrame(rows), _test_config())
        assert df.loc[df["decision_id"] == "missing", "block_reason"].iloc[0] == "missing_features"
        assert df.loc[df["decision_id"] == "present", "filter_status"].iloc[0] == FILTER_STATUS_ACCEPTED

    def test_sample_too_small_block(self) -> None:
        rows = [_make_decision_row(decision_id="only-one")]
        config = _test_config(min_sample_size=100)
        df = apply_decision_filter(pd.DataFrame(rows), config)
        assert df["filter_status"].eq(FILTER_STATUS_BLOCKED).all()
        assert df["block_reason"].eq("sample_too_small").all()

    def test_accepted_signal_remains_accepted(self) -> None:
        rows = [
            _make_decision_row(
                decision_id="good",
                coin="ETH",
                model_signal="DOWN",
                spread_bps=2.0,
                confidence=0.9,
            )
        ]
        df = apply_decision_filter(pd.DataFrame(rows), _test_config())
        row = df.iloc[0]
        assert row["filter_status"] == FILTER_STATUS_ACCEPTED
        assert row["block_reason"] == ""

    def test_model_does_not_beat_baseline_global_block(self) -> None:
        rows = [
            _make_decision_row(
                decision_id=f"row-{index}",
                model_signal="UP",
                baseline_signal="FLAT",
                actual_label="FLAT",
                signal_correct=False,
                baseline_correct=True,
            )
            for index in range(5)
        ]
        config = _test_config(min_sample_size=1, require_model_beats_baseline=True)
        df = apply_decision_filter(pd.DataFrame(rows), config)
        assert df["filter_status"].eq(FILTER_STATUS_BLOCKED).all()
        assert df["block_reason"].eq("model_does_not_beat_baseline").all()


class TestEvaluateFilteredDecisions:
    def test_accepted_accuracy(self) -> None:
        rows = [
            _make_decision_row(decision_id="1", model_signal="UP", actual_label="UP"),
            _make_decision_row(decision_id="2", model_signal="DOWN", actual_label="UP"),
            _make_decision_row(decision_id="3", model_signal="FLAT", actual_label="FLAT"),
        ]
        df = apply_decision_filter(pd.DataFrame(rows), _test_config())
        summary = evaluate_filtered_decisions(df)
        accepted = df[df["filter_status"] == FILTER_STATUS_ACCEPTED]
        expected = float(accepted["signal_correct"].astype(bool).sum()) / len(accepted)
        assert summary["accepted_accuracy"] == pytest.approx(expected)

    def test_baseline_comparison_on_accepted(self) -> None:
        rows = [
            _make_decision_row(
                decision_id="1",
                model_signal="UP",
                baseline_signal="UP",
                actual_label="UP",
                signal_correct=True,
                baseline_correct=True,
            ),
            _make_decision_row(
                decision_id="2",
                model_signal="DOWN",
                baseline_signal="FLAT",
                actual_label="UP",
                signal_correct=False,
                baseline_correct=False,
            ),
        ]
        df = apply_decision_filter(pd.DataFrame(rows), _test_config())
        summary = evaluate_filtered_decisions(df)
        accepted = df[df["filter_status"] == FILTER_STATUS_ACCEPTED]
        baseline_on_accepted = float(accepted["baseline_correct"].astype(bool).sum()) / len(
            accepted
        )
        assert summary["baseline_accuracy_on_accepted"] == pytest.approx(baseline_on_accepted)
        assert summary["accepted_edge_vs_baseline"] == pytest.approx(
            summary["accepted_accuracy"] - baseline_on_accepted
        )

    def test_block_reason_counts(self) -> None:
        rows = [
            _make_decision_row(decision_id="1", spread_bps=20.0),
            _make_decision_row(decision_id="2", spread_bps=25.0),
            _make_decision_row(decision_id="3", coin="DOGE"),
        ]
        df = apply_decision_filter(pd.DataFrame(rows), _test_config())
        summary = evaluate_filtered_decisions(df)
        assert summary["block_reason_counts"]["spread_too_wide"] == 2
        assert summary["block_reason_counts"]["coin_not_allowed"] == 1
        assert summary["blocked_count"] == 3


class TestReportWriters:
    def test_json_report_created(self, output_dir: Path) -> None:
        report = {
            "generated_at": "2024-06-01T00:00:00+00:00",
            "total_decisions": 2,
            "accepted_count": 1,
            "blocked_count": 1,
            "accepted_accuracy": 1.0,
            "blocked_accuracy": 0.0,
            "baseline_accuracy_on_accepted": 0.0,
            "accepted_edge_vs_baseline": 1.0,
            "unfiltered_accuracy": 0.5,
            "unfiltered_baseline_accuracy": 0.5,
            "accepted_by_coin": {"BTC": 1},
            "accepted_by_signal": {"UP": 1},
            "block_reason_counts": {"spread_too_wide": 1},
            "warnings": [SAFETY_NOTE],
            "paper_only": True,
            "execution_enabled": False,
        }
        path = write_filter_report_json(report, output_dir=output_dir)
        assert path.is_file()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["accepted_count"] == 1

    def test_markdown_report_created(self, output_dir: Path) -> None:
        report = {
            "generated_at": "2024-06-01T00:00:00+00:00",
            "source_log_path": "/tmp/test.csv",
            "total_decisions": 2,
            "accepted_count": 1,
            "blocked_count": 1,
            "accepted_accuracy": 1.0,
            "blocked_accuracy": 0.0,
            "baseline_accuracy_on_accepted": 0.0,
            "accepted_edge_vs_baseline": 1.0,
            "unfiltered_accuracy": 0.5,
            "unfiltered_baseline_accuracy": 0.5,
            "accepted_by_coin": {"BTC": 1},
            "accepted_by_signal": {"UP": 1},
            "block_reason_counts": {"signal_blocked": 1},
            "filter_config": default_filter_config(),
            "warnings": [SAFETY_NOTE],
            "paper_only": True,
            "execution_enabled": False,
        }
        path = write_filter_report_markdown(report, output_dir=output_dir)
        text = path.read_text(encoding="utf-8")
        assert path.is_file()
        assert "## Overall filter result" in text
        assert "## Accepted vs blocked counts" in text
        assert "## Accuracy after filtering" in text
        assert "## Baseline comparison" in text
        assert "## Block reasons" in text
        assert "## Coin breakdown (accepted)" in text
        assert "## Limitations" in text
        assert "## Paper-only disclaimer" in text
        assert SAFETY_NOTE in text


class TestRunFilter:
    def test_run_filter_writes_reports(
        self,
        decisions_dir: Path,
        output_dir: Path,
    ) -> None:
        path = _write_decision_log(
            decisions_dir / "bid_ask_decisions_20240601T000000Z.csv",
            _make_accepted_rows(3),
        )
        report = run_filter(
            path,
            config=_test_config(),
            output_dir=output_dir,
        )
        assert Path(report["json_report_path"]).is_file()
        assert Path(report["markdown_report_path"]).is_file()


class TestCli:
    def test_main_latest(
        self,
        decisions_dir: Path,
        output_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_decision_log(
            decisions_dir / "bid_ask_decisions_20240601T000000Z.csv",
            _make_accepted_rows(3),
        )
        exit_code = main(
            [
                "--latest",
                "--decisions-dir",
                str(decisions_dir),
                "--output-dir",
                str(output_dir),
                "--min-sample-size",
                "1",
                "--no-require-model-beats-baseline",
            ]
        )
        captured = capsys.readouterr().out
        assert exit_code == 0
        assert "json_report_path=" in captured
        assert "markdown_report_path=" in captured
        assert "accepted_count=" in captured
        assert "blocked_count=" in captured
        assert "accepted_accuracy=" in captured
        assert "accepted_edge_vs_baseline=" in captured
        assert SAFETY_NOTE in captured


class TestSafety:
    def test_no_wallet_signing_order_execution_code(self) -> None:
        source = FILTER_FILE.read_text(encoding="utf-8")
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
        source = FILTER_FILE.read_text(encoding="utf-8").lower()
        forbidden = (
            "requests.get",
            "requests.post",
            "urllib.request",
            "http.client",
            "aiohttp",
            "websocket",
        )
        for pattern in forbidden:
            assert pattern not in source

    def test_no_subprocess_calls(self) -> None:
        source = FILTER_FILE.read_text(encoding="utf-8").lower()
        assert "subprocess" not in source
        assert "os.system" not in source
        assert "popen" not in source

    def test_paper_only_constants(self) -> None:
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False

    def test_default_output_dirs(self) -> None:
        assert DEFAULT_DECISIONS_DIR == Path("reports/bid_ask_decisions")
        assert DEFAULT_OUTPUT_DIR == Path("reports/bid_ask_decision_filters")
