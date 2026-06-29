"""Focused offline tests for bid/ask CSV validation."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from bid_ask_schema import (
    BID_ASK_CSV_COLUMNS,
    BEST_ASK,
    BEST_BID,
    FORBIDDEN_MODULE_NAMES,
    FORBIDDEN_SOURCE_PATTERNS,
    SPREAD,
    SPREAD_BPS,
    TIMESTAMP_MS,
    build_bid_ask_row,
)
from bid_ask_resampler import RESAMPLED_COLUMNS, resample_bid_ask, resolve_bid_ask_data_dir
from bid_ask_validator import (
    ValidationReport,
    bid_ask_csv_path,
    load_bid_ask_csv_for_validation,
    reports_dir,
    save_validation_report,
    validate_bid_ask_dataframe,
    validate_coin_interval,
)

VALIDATOR_FILE = Path(__file__).resolve().parent.parent / "bid_ask_validator.py"

BASE_MS = 1_718_000_040_000


def _write_1s_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(BID_ASK_CSV_COLUMNS)).to_csv(path, index=False)


def _write_resampled_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _make_rows(
    *,
    coin: str = "BTC",
    start_ms: int = BASE_MS,
    count: int,
    step_ms: int = 1000,
    bid_offset: float = 0.0,
    ask_spread: float = 1.0,
) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        best_bid = 100.0 + bid_offset + index * 0.1
        best_ask = best_bid + ask_spread
        rows.append(
            build_bid_ask_row(
                timestamp_ms=start_ms + index * step_ms,
                coin=coin,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_size=2.0,
                ask_size=1.0,
                source_session_id="test-session",
                source_channel="bbo",
            )
        )
    return rows


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "bid_ask_csv"


class TestValid1sCsv:
    def test_valid_1s_csv_passes(self, data_root: Path) -> None:
        path = bid_ask_csv_path("BTC", "1s", data_root)
        _write_1s_csv(path, _make_rows(count=10))
        df = load_bid_ask_csv_for_validation("BTC", "1s", data_root)
        report = validate_bid_ask_dataframe(df, "1s", coin="BTC")

        assert report.passed is True
        assert report.row_count == 10
        assert report.invalid_bid_ask_count == 0
        assert report.negative_spread_count == 0
        assert report.duplicate_timestamp_count == 0
        assert report.coverage_start == BASE_MS
        assert report.coverage_end == BASE_MS + 9_000

    def test_historical_1s_csv_path(self, data_root: Path) -> None:
        path = bid_ask_csv_path("ETH", "1s", data_root, source="historical")
        assert path == data_root / "historical_1s" / "ETH_1s.csv"
        _write_1s_csv(path, _make_rows(coin="ETH", count=5))
        report = validate_coin_interval("ETH", "1s", data_root=data_root, source="historical")
        assert report.passed is True
        assert report.input_path == str(path)
        assert resolve_bid_ask_data_dir("1s", "historical", data_root) == data_root / "historical_1s"


class Test1sFailures:
    def test_ask_below_bid_fails(self, data_root: Path) -> None:
        rows = _make_rows(count=3)
        rows[1][BEST_ASK] = rows[1][BEST_BID] - 0.5
        _write_1s_csv(bid_ask_csv_path("BTC", "1s", data_root), rows)
        df = load_bid_ask_csv_for_validation("BTC", "1s", data_root)
        report = validate_bid_ask_dataframe(df, "1s", coin="BTC")

        assert report.passed is False
        assert report.invalid_bid_ask_count == 1

    def test_negative_spread_fails(self, data_root: Path) -> None:
        rows = _make_rows(count=3)
        rows[0][SPREAD] = -0.25
        rows[0][SPREAD_BPS] = -25.0
        _write_1s_csv(bid_ask_csv_path("BTC", "1s", data_root), rows)
        df = load_bid_ask_csv_for_validation("BTC", "1s", data_root)
        report = validate_bid_ask_dataframe(df, "1s", coin="BTC")

        assert report.passed is False
        assert report.negative_spread_count >= 1

    def test_duplicate_timestamp_detected(self, data_root: Path) -> None:
        rows = _make_rows(count=4)
        rows[2][TIMESTAMP_MS] = rows[1][TIMESTAMP_MS]
        _write_1s_csv(bid_ask_csv_path("BTC", "1s", data_root), rows)
        df = load_bid_ask_csv_for_validation("BTC", "1s", data_root)
        report = validate_bid_ask_dataframe(df, "1s", coin="BTC")

        assert report.passed is False
        assert report.duplicate_timestamp_count == 1

    def test_missing_seconds_detected(self, data_root: Path) -> None:
        rows = _make_rows(count=3, start_ms=BASE_MS, step_ms=1000)
        rows.append(
            build_bid_ask_row(
                timestamp_ms=BASE_MS + 5_000,
                coin="BTC",
                best_bid=100.0,
                best_ask=101.0,
            )
        )
        _write_1s_csv(bid_ask_csv_path("BTC", "1s", data_root), rows)
        df = load_bid_ask_csv_for_validation("BTC", "1s", data_root)
        report = validate_bid_ask_dataframe(df, "1s", coin="BTC")

        assert report.missing_periods_count == 2
        assert any("missing second" in warning for warning in report.warnings)

    def test_non_monotonic_timestamp_detected(self, data_root: Path) -> None:
        rows = _make_rows(count=4)
        rows[3][TIMESTAMP_MS] = rows[1][TIMESTAMP_MS] - 500
        _write_1s_csv(bid_ask_csv_path("BTC", "1s", data_root), rows)
        df = load_bid_ask_csv_for_validation("BTC", "1s", data_root)
        report = validate_bid_ask_dataframe(df, "1s", coin="BTC")

        assert report.passed is False
        assert any("non-monotonic" in warning for warning in report.warnings)


class TestExtremeSpread:
    def test_extreme_spread_warning(self, data_root: Path) -> None:
        rows = _make_rows(count=2, ask_spread=1.0)
        rows[0][SPREAD_BPS] = 500.0
        _write_1s_csv(bid_ask_csv_path("BTC", "1s", data_root), rows)
        df = load_bid_ask_csv_for_validation("BTC", "1s", data_root)
        report = validate_bid_ask_dataframe(
            df,
            "1s",
            coin="BTC",
            extreme_spread_bps_threshold=100.0,
        )

        assert report.extreme_spread_count == 1
        assert report.passed is True
        assert any("extreme spread" in warning for warning in report.warnings)


class TestResampledValidation:
    def _write_resampled_from_1s(
        self,
        data_root: Path,
        interval: str,
        *,
        count: int = 30,
    ) -> pd.DataFrame:
        path_1s = bid_ask_csv_path("BTC", "1s", data_root)
        _write_1s_csv(path_1s, _make_rows(count=count))
        source = load_bid_ask_csv_for_validation("BTC", "1s", data_root)
        resampled = resample_bid_ask(source, interval)
        output_path = bid_ask_csv_path("BTC", interval, data_root)
        _write_resampled_csv(output_path, resampled)
        return resampled

    def test_valid_15s_csv_passes(self, data_root: Path) -> None:
        self._write_resampled_from_1s(data_root, "15s")
        df = load_bid_ask_csv_for_validation("BTC", "15s", data_root)
        report = validate_bid_ask_dataframe(df, "15s", coin="BTC")

        assert report.passed is True
        assert report.row_count == 2
        assert list(df.columns) == list(RESAMPLED_COLUMNS)

    def test_valid_1m_csv_passes(self, data_root: Path) -> None:
        self._write_resampled_from_1s(data_root, "1m", count=60)
        df = load_bid_ask_csv_for_validation("BTC", "1m", data_root)
        report = validate_bid_ask_dataframe(df, "1m", coin="BTC")

        assert report.passed is True
        assert report.row_count == 1

    def test_invalid_resampled_ohlc_high_low_detected(self, data_root: Path) -> None:
        resampled = self._write_resampled_from_1s(data_root, "15s")
        resampled.loc[0, "high_mid"] = 90.0
        resampled.loc[0, "low_mid"] = 110.0
        _write_resampled_csv(bid_ask_csv_path("BTC", "15s", data_root), resampled)

        df = load_bid_ask_csv_for_validation("BTC", "15s", data_root)
        report = validate_bid_ask_dataframe(df, "15s", coin="BTC")

        assert report.passed is False
        assert report.invalid_bid_ask_count >= 1
        assert any("high_mid < low_mid" in warning for warning in report.warnings)


class TestReportPersistence:
    def test_report_created(self, data_root: Path) -> None:
        _write_1s_csv(bid_ask_csv_path("BTC", "1s", data_root), _make_rows(count=5))
        report = validate_coin_interval("BTC", "1s", data_root=data_root)

        assert report.passed is True
        assert report.input_path.endswith("BTC_1s.csv")
        assert report.report_path
        report_file = Path(report.report_path)
        assert report_file.is_file()
        assert report_file.parent == reports_dir(data_root)

        payload = json.loads(report_file.read_text(encoding="utf-8"))
        assert payload["coin"] == "BTC"
        assert payload["interval"] == "1s"
        assert payload["row_count"] == 5
        assert payload["passed"] is True
        assert "coverage_start" in payload
        assert "coverage_end" in payload
        assert "warnings" in payload

    def test_save_validation_report_fields(self, data_root: Path) -> None:
        report = ValidationReport(
            coin="ETH",
            interval="15s",
            row_count=3,
            coverage_start=BASE_MS,
            coverage_end=BASE_MS + 30_000,
            duplicate_timestamp_count=0,
            missing_periods_count=0,
            invalid_bid_ask_count=0,
            negative_spread_count=0,
            extreme_spread_count=0,
            warnings=[],
            passed=True,
            input_path=str(bid_ask_csv_path("ETH", "15s", data_root)),
        )
        path = save_validation_report(report, data_root=data_root)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for field in (
            "coin",
            "interval",
            "row_count",
            "coverage_start",
            "coverage_end",
            "duplicate_timestamp_count",
            "missing_periods_count",
            "invalid_bid_ask_count",
            "negative_spread_count",
            "extreme_spread_count",
            "warnings",
            "passed",
            "input_path",
            "report_path",
        ):
            assert field in payload


class TestInputFileSafety:
    def test_input_csv_file_not_modified(self, data_root: Path) -> None:
        path = bid_ask_csv_path("BTC", "1s", data_root)
        _write_1s_csv(path, _make_rows(count=8))
        before_text = path.read_text(encoding="utf-8")
        before_hash = hashlib.sha256(before_text.encode("utf-8")).hexdigest()
        before_mtime = path.stat().st_mtime

        validate_coin_interval("BTC", "1s", data_root=data_root)

        after_text = path.read_text(encoding="utf-8")
        after_hash = hashlib.sha256(after_text.encode("utf-8")).hexdigest()
        after_mtime = path.stat().st_mtime

        assert after_hash == before_hash
        assert after_mtime == before_mtime


class TestSafety:
    def test_no_wallet_signing_order_execution_code(self) -> None:
        source = VALIDATOR_FILE.read_text(encoding="utf-8")
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
        source = VALIDATOR_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {
            "requests",
            "urllib",
            "http",
            "socket",
            "websockets",
            "httpx",
            "aiohttp",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in forbidden
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in forbidden
