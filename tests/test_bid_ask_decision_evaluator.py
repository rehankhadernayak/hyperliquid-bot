"""Focused offline tests for paper decision evaluation board."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from bid_ask_decision_logger import DECISION_COLUMNS
from bid_ask_decision_evaluator import (
    CLASS_IMBALANCE_RATIO,
    DEFAULT_DECISIONS_DIR,
    DEFAULT_OUTPUT_DIR,
    MIN_SAMPLE_SIZE,
    SAFETY_NOTE,
    discover_decision_logs,
    evaluate_decisions,
    load_decision_log,
    main,
    run_evaluation,
    spread_bucket,
    suggest_blocking_rules,
    write_evaluation_json,
    write_evaluation_markdown,
)
from bid_ask_schema import EXECUTION_ENABLED, FORBIDDEN_MODULE_NAMES, FORBIDDEN_SOURCE_PATTERNS, PAPER_ONLY

EVALUATOR_FILE = Path(__file__).resolve().parent.parent / "bid_ask_decision_evaluator.py"


def _make_decision_row(
    *,
    decision_id: str = "abc123",
    coin: str = "BTC",
    model_signal: str = "UP",
    baseline_signal: str = "FLAT",
    actual_label: str = "UP",
    spread_bps: float = 1.5,
    signal_correct: bool | None = None,
    baseline_correct: bool | None = None,
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
        "confidence": 0.8,
        "horizon": 1,
        "future_return_bps": 2.0,
        "actual_label": actual_label,
        "signal_correct": signal_correct,
        "baseline_correct": baseline_correct,
        "features_json": "{}",
        "paper_only": True,
        "execution_enabled": False,
        "notes": "test",
    }


def _write_decision_log(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(DECISION_COLUMNS)).to_csv(path, index=False)
    return path


@pytest.fixture
def decisions_dir(tmp_path: Path) -> Path:
    return tmp_path / "reports" / "bid_ask_decisions"


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "reports" / "bid_ask_decision_evaluation"


class TestDiscoverDecisionLogs:
    def test_discover_latest_log(self, decisions_dir: Path) -> None:
        older = decisions_dir / "bid_ask_decisions_20240101T000000Z.csv"
        newer = decisions_dir / "bid_ask_decisions_20240601T120000Z.csv"
        _write_decision_log(older, [_make_decision_row(decision_id="old")])
        _write_decision_log(newer, [_make_decision_row(decision_id="new")])

        logs = discover_decision_logs(decisions_dir)
        assert logs[0] == newer
        assert logs[1] == older

    def test_missing_directory_returns_empty(self, tmp_path: Path) -> None:
        assert discover_decision_logs(tmp_path / "missing") == []


class TestLoadDecisionLog:
    def test_load_decision_log(self, decisions_dir: Path) -> None:
        path = _write_decision_log(
            decisions_dir / "bid_ask_decisions_test.csv",
            [_make_decision_row()],
        )
        df = load_decision_log(path)
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
            load_decision_log(path)


class TestSpreadBucket:
    @pytest.mark.parametrize(
        ("spread_bps", "expected"),
        [
            (1.0, "tight"),
            (2.0, "tight"),
            (2.1, "normal"),
            (10.0, "normal"),
            (10.1, "wide"),
            (25.0, "wide"),
            (25.1, "too_wide"),
        ],
    )
    def test_spread_bucket(self, spread_bps: float, expected: str) -> None:
        assert spread_bucket(spread_bps) == expected


class TestEvaluateDecisions:
    def test_overall_accuracy(self) -> None:
        rows = [
            _make_decision_row(decision_id="1", model_signal="UP", actual_label="UP"),
            _make_decision_row(decision_id="2", model_signal="DOWN", actual_label="UP"),
            _make_decision_row(decision_id="3", model_signal="FLAT", actual_label="FLAT"),
            _make_decision_row(decision_id="4", model_signal="UP", actual_label="DOWN"),
        ]
        df = pd.DataFrame(rows, columns=list(DECISION_COLUMNS))
        evaluation = evaluate_decisions(df)

        assert evaluation["total_decisions"] == 4
        assert evaluation["model_accuracy"] == pytest.approx(0.5)
        assert evaluation["model_correct_count"] == 2

    def test_baseline_accuracy(self) -> None:
        rows = [
            _make_decision_row(
                decision_id="1",
                baseline_signal="UP",
                actual_label="UP",
                baseline_correct=True,
            ),
            _make_decision_row(
                decision_id="2",
                baseline_signal="FLAT",
                actual_label="UP",
                baseline_correct=False,
            ),
            _make_decision_row(
                decision_id="3",
                baseline_signal="FLAT",
                actual_label="FLAT",
                baseline_correct=True,
            ),
        ]
        df = pd.DataFrame(rows, columns=list(DECISION_COLUMNS))
        evaluation = evaluate_decisions(df)

        assert evaluation["baseline_accuracy"] == pytest.approx(2 / 3)
        assert evaluation["model_edge_vs_baseline"] == pytest.approx(
            evaluation["model_accuracy"] - evaluation["baseline_accuracy"]
        )

    def test_accuracy_by_coin(self) -> None:
        rows = [
            _make_decision_row(decision_id="1", coin="BTC", model_signal="UP", actual_label="UP"),
            _make_decision_row(decision_id="2", coin="BTC", model_signal="DOWN", actual_label="UP"),
            _make_decision_row(decision_id="3", coin="ETH", model_signal="UP", actual_label="UP"),
        ]
        df = pd.DataFrame(rows, columns=list(DECISION_COLUMNS))
        evaluation = evaluate_decisions(df)

        assert evaluation["accuracy_by_coin"]["BTC"]["accuracy"] == pytest.approx(0.5)
        assert evaluation["accuracy_by_coin"]["ETH"]["accuracy"] == pytest.approx(1.0)

    def test_accuracy_by_signal(self) -> None:
        rows = [
            _make_decision_row(decision_id="1", model_signal="UP", actual_label="UP"),
            _make_decision_row(decision_id="2", model_signal="UP", actual_label="DOWN"),
            _make_decision_row(decision_id="3", model_signal="DOWN", actual_label="DOWN"),
        ]
        df = pd.DataFrame(rows, columns=list(DECISION_COLUMNS))
        evaluation = evaluate_decisions(df)

        assert evaluation["accuracy_by_signal"]["UP"]["accuracy"] == pytest.approx(0.5)
        assert evaluation["accuracy_by_signal"]["DOWN"]["accuracy"] == pytest.approx(1.0)

    def test_accuracy_by_spread_bucket(self) -> None:
        rows = [
            _make_decision_row(decision_id="1", spread_bps=1.0, model_signal="UP", actual_label="UP"),
            _make_decision_row(decision_id="2", spread_bps=5.0, model_signal="UP", actual_label="DOWN"),
            _make_decision_row(decision_id="3", spread_bps=30.0, model_signal="DOWN", actual_label="DOWN"),
        ]
        df = pd.DataFrame(rows, columns=list(DECISION_COLUMNS))
        evaluation = evaluate_decisions(df)

        assert evaluation["accuracy_by_spread_bucket"]["tight"]["count"] == 1
        assert evaluation["accuracy_by_spread_bucket"]["normal"]["count"] == 1
        assert evaluation["accuracy_by_spread_bucket"]["too_wide"]["count"] == 1

    def test_label_counts(self) -> None:
        rows = [
            _make_decision_row(decision_id="1", actual_label="UP"),
            _make_decision_row(decision_id="2", actual_label="UP"),
            _make_decision_row(decision_id="3", actual_label="DOWN"),
            _make_decision_row(decision_id="4", actual_label="FLAT"),
        ]
        df = pd.DataFrame(rows, columns=list(DECISION_COLUMNS))
        evaluation = evaluate_decisions(df)

        assert evaluation["label_counts"] == {"UP": 2, "DOWN": 1, "FLAT": 1}

    def test_incorrect_examples(self) -> None:
        rows = [
            _make_decision_row(decision_id="correct", model_signal="UP", actual_label="UP"),
            _make_decision_row(decision_id="wrong", model_signal="UP", actual_label="DOWN"),
        ]
        df = pd.DataFrame(rows, columns=list(DECISION_COLUMNS))
        evaluation = evaluate_decisions(df)

        assert len(evaluation["incorrect_examples"]) == 1
        assert evaluation["incorrect_examples"][0]["decision_id"] == "wrong"


class TestWarnings:
    def _make_rows(self, count: int, **kwargs) -> list[dict]:
        return [
            _make_decision_row(decision_id=f"row-{index}", **kwargs)
            for index in range(count)
        ]

    def test_small_sample_warning(self) -> None:
        rows = self._make_rows(MIN_SAMPLE_SIZE - 1)
        evaluation = evaluate_decisions(pd.DataFrame(rows, columns=list(DECISION_COLUMNS)))
        assert any("Small sample size" in warning for warning in evaluation["warnings"])

    def test_model_below_baseline_warning(self) -> None:
        rows = [
            _make_decision_row(
                decision_id=f"row-{index}",
                model_signal="UP",
                baseline_signal="FLAT",
                actual_label="FLAT",
                signal_correct=False,
                baseline_correct=True,
            )
            for index in range(MIN_SAMPLE_SIZE)
        ]
        evaluation = evaluate_decisions(pd.DataFrame(rows, columns=list(DECISION_COLUMNS)))
        assert evaluation["model_accuracy"] < evaluation["baseline_accuracy"]
        assert any("does not beat baseline" in warning for warning in evaluation["warnings"])

    def test_class_imbalance_warning(self) -> None:
        dominant = int(MIN_SAMPLE_SIZE * CLASS_IMBALANCE_RATIO) + 5
        minority = MIN_SAMPLE_SIZE - dominant
        rows = [
            _make_decision_row(decision_id=f"up-{index}", actual_label="UP")
            for index in range(dominant)
        ] + [
            _make_decision_row(decision_id=f"down-{index}", actual_label="DOWN")
            for index in range(minority)
        ]
        evaluation = evaluate_decisions(pd.DataFrame(rows, columns=list(DECISION_COLUMNS)))
        assert any("class imbalance" in warning.lower() for warning in evaluation["warnings"])

    def test_spread_too_wide_warning(self) -> None:
        rows = [
            _make_decision_row(decision_id=f"tight-{index}", spread_bps=1.0)
            for index in range(27)
        ] + [
            _make_decision_row(decision_id=f"wide-{index}", spread_bps=30.0)
            for index in range(3)
        ]
        evaluation = evaluate_decisions(pd.DataFrame(rows, columns=list(DECISION_COLUMNS)))
        assert any("Spread too wide" in warning for warning in evaluation["warnings"])


class TestSuggestBlockingRules:
    def test_block_model_when_below_baseline(self) -> None:
        evaluation = {
            "total_decisions": MIN_SAMPLE_SIZE,
            "model_accuracy": 0.4,
            "baseline_accuracy": 0.6,
            "accuracy_by_coin": {},
            "accuracy_by_signal": {},
            "accuracy_by_spread_bucket": {},
            "warnings": [],
        }
        rules = suggest_blocking_rules(evaluation)
        assert any(rule["rule_type"] == "block_model" for rule in rules)

    def test_block_coin_when_sample_small(self) -> None:
        evaluation = {
            "total_decisions": 10,
            "model_accuracy": 0.8,
            "baseline_accuracy": 0.5,
            "accuracy_by_coin": {"BTC": {"count": 5, "correct": 4, "accuracy": 0.8}},
            "accuracy_by_signal": {},
            "accuracy_by_spread_bucket": {},
            "warnings": [],
        }
        rules = suggest_blocking_rules(evaluation)
        assert any(
            rule["rule_type"] == "block_coin" and rule["coin"] == "BTC" for rule in rules
        )

    def test_block_signal_when_below_baseline(self) -> None:
        evaluation = {
            "total_decisions": MIN_SAMPLE_SIZE,
            "model_accuracy": 0.6,
            "baseline_accuracy": 0.7,
            "accuracy_by_coin": {},
            "accuracy_by_signal": {
                "UP": {"count": 10, "correct": 4, "accuracy": 0.4},
            },
            "accuracy_by_spread_bucket": {},
            "warnings": [],
        }
        rules = suggest_blocking_rules(evaluation)
        assert any(
            rule["rule_type"] == "block_signal" and rule["signal"] == "UP" for rule in rules
        )


class TestReportWriters:
    def test_json_report_created(self, output_dir: Path) -> None:
        evaluation = {
            "generated_at": "2024-06-01T00:00:00+00:00",
            "total_decisions": 1,
            "model_accuracy": 1.0,
            "baseline_accuracy": 0.0,
            "model_edge_vs_baseline": 1.0,
            "accuracy_by_coin": {},
            "accuracy_by_signal": {},
            "accuracy_by_spread_bucket": {},
            "label_counts": {"UP": 1, "DOWN": 0, "FLAT": 0},
            "incorrect_examples": [],
            "warnings": [SAFETY_NOTE],
            "paper_only": True,
            "execution_enabled": False,
            "suggested_blocking_rules": [],
        }
        path = write_evaluation_json(evaluation, output_dir=output_dir)
        assert path.is_file()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["total_decisions"] == 1

    def test_markdown_report_created(self, output_dir: Path) -> None:
        evaluation = {
            "generated_at": "2024-06-01T00:00:00+00:00",
            "source_log_path": "/tmp/test.csv",
            "total_decisions": 2,
            "model_accuracy": 0.5,
            "baseline_accuracy": 0.5,
            "model_edge_vs_baseline": 0.0,
            "model_correct_count": 1,
            "baseline_correct_count": 1,
            "accuracy_by_coin": {"BTC": {"count": 2, "correct": 1, "accuracy": 0.5}},
            "accuracy_by_signal": {"UP": {"count": 2, "correct": 1, "accuracy": 0.5}},
            "accuracy_by_spread_bucket": {
                "tight": {"count": 2, "correct": 1, "accuracy": 0.5},
            },
            "label_counts": {"UP": 1, "DOWN": 1, "FLAT": 0},
            "incorrect_examples": [
                {
                    "decision_id": "bad",
                    "coin": "BTC",
                    "model_signal": "UP",
                    "baseline_signal": "FLAT",
                    "actual_label": "DOWN",
                    "spread_bps": 1.0,
                    "spread_bucket": "tight",
                    "confidence": 0.5,
                    "future_return_bps": -1.0,
                }
            ],
            "warnings": [SAFETY_NOTE],
            "paper_only": True,
            "execution_enabled": False,
            "suggested_blocking_rules": [
                {
                    "rule_type": "block_model",
                    "action": "Skip model",
                    "reason": "test",
                    "paper_only": True,
                }
            ],
        }
        path = write_evaluation_markdown(evaluation, output_dir=output_dir)
        text = path.read_text(encoding="utf-8")
        assert path.is_file()
        assert "## Overall result" in text
        assert "## Accuracy vs baseline" in text
        assert "## Coin breakdown" in text
        assert "## Signal breakdown" in text
        assert "## Spread bucket breakdown" in text
        assert "## Incorrect examples" in text
        assert "## Suggested paper-only blocking rules" in text
        assert "## Limitations" in text
        assert "## Research-only disclaimer" in text
        assert SAFETY_NOTE in text


class TestRunEvaluation:
    def test_run_evaluation_writes_reports(
        self,
        decisions_dir: Path,
        output_dir: Path,
    ) -> None:
        path = _write_decision_log(
            decisions_dir / "bid_ask_decisions_20240601T000000Z.csv",
            [_make_decision_row()],
        )
        evaluation = run_evaluation(path, output_dir=output_dir)
        assert Path(evaluation["json_report_path"]).is_file()
        assert Path(evaluation["markdown_report_path"]).is_file()
        assert evaluation["suggested_blocking_rules"] is not None


class TestCli:
    def test_main_latest(
        self,
        decisions_dir: Path,
        output_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_decision_log(
            decisions_dir / "bid_ask_decisions_20240601T000000Z.csv",
            [_make_decision_row()],
        )
        exit_code = main(
            [
                "--latest",
                "--decisions-dir",
                str(decisions_dir),
                "--output-dir",
                str(output_dir),
            ]
        )
        captured = capsys.readouterr().out
        assert exit_code == 0
        assert "json_report_path=" in captured
        assert "markdown_report_path=" in captured
        assert "model_accuracy=" in captured
        assert "baseline_accuracy=" in captured
        assert SAFETY_NOTE in captured


class TestSafety:
    def test_no_wallet_signing_order_execution_code(self) -> None:
        source = EVALUATOR_FILE.read_text(encoding="utf-8")
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
        source = EVALUATOR_FILE.read_text(encoding="utf-8").lower()
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
        source = EVALUATOR_FILE.read_text(encoding="utf-8").lower()
        assert "subprocess" not in source
        assert "os.system" not in source
        assert "popen" not in source

    def test_paper_only_constants(self) -> None:
        assert PAPER_ONLY is True
        assert EXECUTION_ENABLED is False

    def test_default_output_dirs(self) -> None:
        assert DEFAULT_DECISIONS_DIR == Path("reports/bid_ask_decisions")
        assert DEFAULT_OUTPUT_DIR == Path("reports/bid_ask_decision_evaluation")
