"""Focused offline tests for bid/ask CSV resampling."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from bid_ask_schema import (
    BID_ASK_CSV_COLUMNS,
    FORBIDDEN_MODULE_NAMES,
    FORBIDDEN_SOURCE_PATTERNS,
    TIMESTAMP_MS,
    build_bid_ask_row,
)
from bid_ask_resampler import (
    RESAMPLED_COLUMNS,
    bid_ask_1s_path,
    load_1s_csv,
    reports_dir,
    resample_bid_ask,
    resample_coin,
    resampled_csv_path,
    write_resampled_csv,
    write_resampling_report,
)

RESAMPLER_FILE = Path(__file__).resolve().parent.parent / "bid_ask_resampler.py"

BASE_MS = 1_718_000_040_000  # aligned to 15s and 1m bucket boundaries


def _write_1s_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(BID_ASK_CSV_COLUMNS)).to_csv(path, index=False)


def _make_rows(
    *,
    coin: str = "BTC",
    start_ms: int = BASE_MS,
    count: int,
    step_ms: int = 1000,
    bid_offset: float = 0.0,
) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        best_bid = 100.0 + bid_offset + index * 0.1
        best_ask = best_bid + 1.0
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


@pytest.fixture
def btc_1s_path(data_root: Path) -> Path:
    path = bid_ask_1s_path("BTC", data_root)
    _write_1s_csv(path, _make_rows(count=30))
    return path


class TestLoad1sCsv:
    def test_load_1s_csv(self, data_root: Path, btc_1s_path: Path) -> None:
        df = load_1s_csv("BTC", data_root)
        assert len(df) == 30
        assert list(df.columns) == list(BID_ASK_CSV_COLUMNS)
        assert df[TIMESTAMP_MS].is_monotonic_increasing


class TestResample15s:
    def test_15_second_resampling(self, data_root: Path, btc_1s_path: Path) -> None:
        df = load_1s_csv("BTC", data_root)
        resampled = resample_bid_ask(df, "15s")

        assert list(resampled.columns) == list(RESAMPLED_COLUMNS)
        assert len(resampled) == 2

        first = resampled.iloc[0]
        assert first["start_timestamp_ms"] == BASE_MS
        assert first["end_timestamp_ms"] == BASE_MS + 15_000 - 1
        assert first["row_count"] == 15
        assert first["missing_seconds"] == 0

    def test_open_high_low_close_mid_15s(
        self, data_root: Path, btc_1s_path: Path
    ) -> None:
        df = load_1s_csv("BTC", data_root)
        first = resample_bid_ask(df, "15s").iloc[0]

        assert first["open_mid"] == pytest.approx(100.5)
        assert first["close_mid"] == pytest.approx(100.5 + 14 * 0.1)
        assert first["high_mid"] == pytest.approx(100.5 + 14 * 0.1)
        assert first["low_mid"] == pytest.approx(100.5)

    def test_avg_bid_ask_spread_15s(self, data_root: Path, btc_1s_path: Path) -> None:
        df = load_1s_csv("BTC", data_root)
        first = resample_bid_ask(df, "15s").iloc[0]

        expected_avg_bid = sum(100.0 + i * 0.1 for i in range(15)) / 15
        expected_avg_spread_bps = sum(
            1.0 / (100.0 + i * 0.1 + 0.5) * 10000.0 for i in range(15)
        ) / 15
        assert first["avg_bid"] == pytest.approx(expected_avg_bid)
        assert first["avg_ask"] == pytest.approx(expected_avg_bid + 1.0)
        assert first["avg_spread"] == pytest.approx(1.0)
        assert first["avg_spread_bps"] == pytest.approx(expected_avg_spread_bps)
        assert first["avg_bbo_imbalance"] == pytest.approx(1.0 / 3.0)


class TestResample1m:
    def test_1_minute_resampling(self, data_root: Path) -> None:
        path = bid_ask_1s_path("BTC", data_root)
        _write_1s_csv(path, _make_rows(count=90))
        df = load_1s_csv("BTC", data_root)
        resampled = resample_bid_ask(df, "1m")

        assert len(resampled) == 2
        assert resampled.iloc[0]["row_count"] == 60
        assert resampled.iloc[0]["missing_seconds"] == 0
        assert resampled.iloc[1]["row_count"] == 30
        assert resampled.iloc[1]["missing_seconds"] == 30


class TestMissingSeconds:
    def test_missing_seconds(self, data_root: Path) -> None:
        path = bid_ask_1s_path("BTC", data_root)
        _write_1s_csv(path, _make_rows(count=10))
        resampled = resample_bid_ask(load_1s_csv("BTC", data_root), "15s")

        assert len(resampled) == 1
        assert resampled.iloc[0]["row_count"] == 10
        assert resampled.iloc[0]["missing_seconds"] == 5


class TestOutputPaths:
    def test_output_paths(self, data_root: Path, btc_1s_path: Path) -> None:
        df = load_1s_csv("BTC", data_root)
        path_15s = write_resampled_csv(resample_bid_ask(df, "15s"), "BTC", "15s", data_root)
        path_1m = write_resampled_csv(resample_bid_ask(df, "1m"), "BTC", "1m", data_root)

        assert path_15s == data_root / "15s" / "BTC_15s.csv"
        assert path_1m == data_root / "1m" / "BTC_1m.csv"
        assert path_15s.is_file()
        assert path_1m.is_file()

    def test_resampling_report_path(self, data_root: Path, btc_1s_path: Path) -> None:
        source_path = bid_ask_1s_path("BTC", data_root)
        output_path = resampled_csv_path("BTC", "15s", data_root)
        report_path = write_resampling_report(
            coin="BTC",
            interval="15s",
            source_path=source_path,
            output_path=output_path,
            input_rows=30,
            output_rows=2,
            total_missing_seconds=0,
            data_root=data_root,
        )

        assert report_path.parent == reports_dir(data_root)
        assert report_path.name.startswith("BTC_15s_resample_")
        assert report_path.suffix == ".json"


class TestInvalidInterval:
    def test_invalid_interval_rejected(self, data_root: Path, btc_1s_path: Path) -> None:
        df = load_1s_csv("BTC", data_root)
        with pytest.raises(ValueError, match="Unsupported interval"):
            resample_bid_ask(df, "5s")
        with pytest.raises(ValueError, match="Unsupported interval"):
            resampled_csv_path("BTC", "5s", data_root)


class TestRawFileSafety:
    def test_no_raw_file_modifications(self, data_root: Path) -> None:
        source_path = bid_ask_1s_path("BTC", data_root)
        _write_1s_csv(source_path, _make_rows(count=20))
        before_text = source_path.read_text(encoding="utf-8")
        before_hash = hashlib.sha256(before_text.encode("utf-8")).hexdigest()
        before_mtime = source_path.stat().st_mtime

        resample_coin("BTC", intervals=("15s",), data_root=data_root)

        after_text = source_path.read_text(encoding="utf-8")
        after_hash = hashlib.sha256(after_text.encode("utf-8")).hexdigest()
        after_mtime = source_path.stat().st_mtime

        assert after_hash == before_hash
        assert after_mtime == before_mtime


class TestSafety:
    def test_no_wallet_signing_order_execution_code(self) -> None:
        source = RESAMPLER_FILE.read_text(encoding="utf-8")
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

    def test_no_network_imports(self) -> None:
        source = RESAMPLER_FILE.read_text(encoding="utf-8")
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