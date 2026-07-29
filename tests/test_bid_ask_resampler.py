"""Focused offline tests for bid/ask CSV resampling."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from bid_ask_schema import (
    ASK_SIZE,
    BID_ASK_CSV_COLUMNS,
    BID_SIZE,
    FORBIDDEN_MODULE_NAMES,
    FORBIDDEN_SOURCE_PATTERNS,
    SOURCE_CHANNEL,
    SOURCE_SESSION_ID,
    TIMESTAMP_MS,
    build_bid_ask_row,
)
from bid_ask_resampler import (
    RESAMPLED_COLUMNS,
    assert_resampled_aligned_with_source,
    bid_ask_1s_path,
    last_resampled_bucket_start,
    load_1s_csv,
    reports_dir,
    resolve_bid_ask_data_dir,
    resample_bid_ask,
    resample_coin,
    resampled_csv_path,
    sanitize_1s_dataframe,
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
    bid_size: float | None = 2.0,
    ask_size: float | None = 1.0,
    source_session_id: str = "test-session",
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
                bid_size=bid_size if bid_size is not None else None,
                ask_size=ask_size if ask_size is not None else None,
                source_session_id=source_session_id,
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


class TestTopOfBookSizes:
    def test_preserves_last_valid_sizes_in_15s(self, data_root: Path) -> None:
        path = bid_ask_1s_path("BTC", data_root)
        rows = _make_rows(count=15)
        for index, row in enumerate(rows):
            row[BID_SIZE] = 1.0 + index * 0.1
            row[ASK_SIZE] = 2.0 + index * 0.05
        _write_1s_csv(path, rows)

        first = resample_bid_ask(load_1s_csv("BTC", data_root), "15s").iloc[0]
        assert first[BID_SIZE] == pytest.approx(1.0 + 14 * 0.1)
        assert first[ASK_SIZE] == pytest.approx(2.0 + 14 * 0.05)

    def test_last_valid_skips_null_sizes(self, data_root: Path) -> None:
        path = bid_ask_1s_path("BTC", data_root)
        rows = _make_rows(count=15)
        rows[13][BID_SIZE] = None
        rows[13][ASK_SIZE] = None
        rows[14][BID_SIZE] = 9.9
        rows[14][ASK_SIZE] = 8.8
        _write_1s_csv(path, rows)

        first = resample_bid_ask(load_1s_csv("BTC", data_root), "15s").iloc[0]
        assert first[BID_SIZE] == pytest.approx(9.9)
        assert first[ASK_SIZE] == pytest.approx(8.8)

    def test_all_null_sizes_in_bucket(self, data_root: Path) -> None:
        path = bid_ask_1s_path("BTC", data_root)
        rows = _make_rows(count=15, bid_size=None, ask_size=None)
        _write_1s_csv(path, rows)

        first = resample_bid_ask(load_1s_csv("BTC", data_root), "15s").iloc[0]
        assert pd.isna(first[BID_SIZE])
        assert pd.isna(first[ASK_SIZE])

    def test_sizes_carried_to_1m_bars(self, data_root: Path) -> None:
        path = bid_ask_1s_path("BTC", data_root)
        rows = _make_rows(count=60)
        for index, row in enumerate(rows):
            row[BID_SIZE] = float(index)
            row[ASK_SIZE] = float(60 - index)
        _write_1s_csv(path, rows)

        first = resample_bid_ask(load_1s_csv("BTC", data_root), "1m").iloc[0]
        assert first[BID_SIZE] == pytest.approx(59.0)
        assert first[ASK_SIZE] == pytest.approx(1.0)

    def test_no_future_leakage_sizes_use_in_bucket_only(self, data_root: Path) -> None:
        """Each bar's sizes must come from ticks inside that bar, not the next bucket."""
        path = bid_ask_1s_path("BTC", data_root)
        rows = _make_rows(count=30)
        for index, row in enumerate(rows):
            bucket = index // 15
            row[BID_SIZE] = 100.0 + bucket * 1000.0 + index
            row[ASK_SIZE] = 200.0 + bucket * 1000.0 + index
        _write_1s_csv(path, rows)

        resampled = resample_bid_ask(load_1s_csv("BTC", data_root), "15s")
        assert resampled.iloc[0][BID_SIZE] == pytest.approx(100.0 + 14)
        assert resampled.iloc[1][BID_SIZE] == pytest.approx(1100.0 + 29)

    def test_avg_bbo_imbalance_preserved(self, data_root: Path, btc_1s_path: Path) -> None:
        df = load_1s_csv("BTC", data_root)
        first = resample_bid_ask(df, "15s").iloc[0]
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


class TestSourceBoundsAndStaleTail:
    def test_last_bucket_matches_final_1s_observation(self) -> None:
        final_1s_ms = 1_696_085_999_604
        assert last_resampled_bucket_start(final_1s_ms, "15s") == 1_696_085_985_000

    def test_shortening_1s_removes_stale_resampled_tail(self, data_root: Path) -> None:
        source_path = bid_ask_1s_path("ETH", data_root, source="historical")
        resampled_path = resampled_csv_path("ETH", "15s", data_root, source="historical")

        _write_1s_csv(source_path, _make_rows(coin="ETH", count=60))
        resample_coin("ETH", intervals=("15s",), data_root=data_root, source="historical")
        long_resampled = pd.read_csv(resampled_path)
        long_rows = len(long_resampled)

        _write_1s_csv(source_path, _make_rows(coin="ETH", count=20))
        resample_coin("ETH", intervals=("15s",), data_root=data_root, source="historical")
        short_resampled = pd.read_csv(resampled_path)

        assert len(short_resampled) < long_rows
        source_max_ms = int(load_1s_csv("ETH", data_root, source="historical")[TIMESTAMP_MS].max())
        expected_last_start = last_resampled_bucket_start(source_max_ms, "15s")
        assert int(short_resampled.iloc[-1]["start_timestamp_ms"]) == expected_last_start
        assert_resampled_aligned_with_source(
            short_resampled,
            source_min_ms=int(
                load_1s_csv("ETH", data_root, source="historical")[TIMESTAMP_MS].min()
            ),
            source_max_ms=source_max_ms,
            interval="15s",
        )

    def test_atomic_write_replaces_longer_existing_file(self, data_root: Path) -> None:
        path = resampled_csv_path("BTC", "15s", data_root)
        long_df = resample_bid_ask(
            pd.DataFrame(_make_rows(count=30), columns=list(BID_ASK_CSV_COLUMNS)),
            "15s",
        )
        write_resampled_csv(long_df, "BTC", "15s", data_root)
        long_line_count = len(path.read_text(encoding="utf-8").splitlines())

        short_df = resample_bid_ask(
            pd.DataFrame(_make_rows(count=10), columns=list(BID_ASK_CSV_COLUMNS)),
            "15s",
        )
        write_resampled_csv(short_df, "BTC", "15s", data_root)
        written = pd.read_csv(path)

        assert len(written) == len(short_df)
        assert len(path.read_text(encoding="utf-8").splitlines()) < long_line_count


class TestOutputPaths:
    def test_output_paths(self, data_root: Path, btc_1s_path: Path) -> None:
        df = load_1s_csv("BTC", data_root)
        path_15s = write_resampled_csv(resample_bid_ask(df, "15s"), "BTC", "15s", data_root)
        path_1m = write_resampled_csv(resample_bid_ask(df, "1m"), "BTC", "1m", data_root)

        assert path_15s == data_root / "15s" / "BTC_15s.csv"
        assert path_1m == data_root / "1m" / "BTC_1m.csv"
        assert path_15s.is_file()
        assert path_1m.is_file()

    def test_historical_output_paths(self, data_root: Path) -> None:
        source_path = bid_ask_1s_path("SOL", data_root, source="historical")
        _write_1s_csv(source_path, _make_rows(coin="SOL", count=30))
        results = resample_coin("SOL", intervals=("15s", "1m"), data_root=data_root, source="historical")

        path_15s = results["15s"]["output_path"]
        path_1m = results["1m"]["output_path"]
        assert path_15s == data_root / "historical_15s" / "SOL_15s.csv"
        assert path_1m == data_root / "historical_1m" / "SOL_1m.csv"
        assert path_15s.is_file()
        assert path_1m.is_file()


class TestResolveBidAskDataDir:
    def test_local_source_paths(self, data_root: Path) -> None:
        assert resolve_bid_ask_data_dir("1s", "local", data_root) == data_root / "1s"
        assert resolve_bid_ask_data_dir("15s", "local", data_root) == data_root / "15s"
        assert resolve_bid_ask_data_dir("1m", "local", data_root) == data_root / "1m"

    def test_historical_source_paths(self, data_root: Path) -> None:
        assert resolve_bid_ask_data_dir("1s", "historical", data_root) == data_root / "historical_1s"
        assert resolve_bid_ask_data_dir("15s", "historical", data_root) == data_root / "historical_15s"
        assert resolve_bid_ask_data_dir("1m", "historical", data_root) == data_root / "historical_1m"

    def test_local_source_remains_default(self, data_root: Path) -> None:
        assert resolve_bid_ask_data_dir("1s", data_root=data_root) == data_root / "1s"
        assert resampled_csv_path("BTC", "15s", data_root) == data_root / "15s" / "BTC_15s.csv"

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


class TestSanitizeInput:
    def test_historical_archive_metadata_preserved(self, data_root: Path) -> None:
        path = bid_ask_1s_path("ETH", data_root)
        rows = _make_rows(coin="ETH", count=15, source_session_id="historical_archive")
        for row in rows:
            row[SOURCE_CHANNEL] = "historical_l2Book"
        _write_1s_csv(path, rows)

        df = load_1s_csv("ETH", data_root)
        assert (df[SOURCE_SESSION_ID] == "historical_archive").all()
        assert (df[SOURCE_CHANNEL] == "historical_l2Book").all()
        resampled = resample_bid_ask(df, "15s")
        assert len(resampled) == 1

    def test_mixed_numeric_strings_dropped(self, data_root: Path) -> None:
        path = bid_ask_1s_path("BTC", data_root)
        rows = _make_rows(count=5)
        rows[2]["mid_price"] = "not-a-price"
        rows[3]["spread"] = "bad"
        _write_1s_csv(path, rows)

        df = load_1s_csv("BTC", data_root)
        assert len(df) == 3

    def test_nan_and_infinity_dropped(self) -> None:
        rows = _make_rows(count=4)
        rows[1]["best_bid"] = float("nan")
        rows[2]["best_ask"] = float("inf")
        rows[3]["spread_bps"] = float("-inf")
        df = pd.DataFrame(rows, columns=list(BID_ASK_CSV_COLUMNS))
        cleaned, removed = sanitize_1s_dataframe(df)
        assert removed == 3
        assert len(cleaned) == 1

    def test_non_positive_timestamp_dropped(self) -> None:
        rows = _make_rows(count=3)
        rows[1][TIMESTAMP_MS] = 0
        rows[2][TIMESTAMP_MS] = -100
        df = pd.DataFrame(rows, columns=list(BID_ASK_CSV_COLUMNS))
        cleaned, removed = sanitize_1s_dataframe(df)
        assert removed == 2
        assert len(cleaned) == 1

    def test_duplicate_timestamps_still_resample(self, data_root: Path) -> None:
        path = bid_ask_1s_path("BTC", data_root)
        rows = _make_rows(count=15)
        rows[14][TIMESTAMP_MS] = rows[13][TIMESTAMP_MS]
        _write_1s_csv(path, rows)

        resampled = resample_bid_ask(load_1s_csv("BTC", data_root), "15s")
        assert len(resampled) == 1
        assert resampled.iloc[0]["row_count"] == 15

    def test_shifted_row_with_historical_archive_in_price_column_dropped(self) -> None:
        """Malformed CSV row must not crash resampling when metadata strings appear in price cols."""
        good = build_bid_ask_row(
            timestamp_ms=BASE_MS,
            coin="ETH",
            best_bid=100.0,
            best_ask=101.0,
            source_session_id="historical_archive",
            source_channel="historical_l2Book",
        )
        bad = {column: None for column in BID_ASK_CSV_COLUMNS}
        bad.update(
            {
                "timestamp_ms": 0.4,
                "timestamp_iso": "2023-09-16T00:00:05.332000Z",
                "coin": "ETH",
                "best_bid": 8.9177,
                "best_ask": 0.060568,
                "mid_price": "historical_archive",
                "spread": "historical_l2Book",
                SOURCE_SESSION_ID: None,
                SOURCE_CHANNEL: None,
            }
        )
        df = pd.DataFrame([good, bad], columns=list(BID_ASK_CSV_COLUMNS))
        cleaned, removed = sanitize_1s_dataframe(df)
        assert removed == 1
        resampled = resample_bid_ask(cleaned, "15s")
        assert len(resampled) == 1

    def test_metadata_not_numerically_coerced(self) -> None:
        rows = _make_rows(count=2, source_session_id="historical_archive")
        rows[0][SOURCE_CHANNEL] = "historical_l2Book"
        df = pd.DataFrame(rows, columns=list(BID_ASK_CSV_COLUMNS))
        cleaned, _ = sanitize_1s_dataframe(df)
        assert cleaned[SOURCE_SESSION_ID].dtype == object or str(cleaned[SOURCE_SESSION_ID].dtype).startswith("str")
        assert cleaned.iloc[0][SOURCE_SESSION_ID] == "historical_archive"

    def test_historical_source_resample_end_to_end(self, data_root: Path) -> None:
        path = bid_ask_1s_path("ETH", data_root, source="historical")
        rows = _make_rows(coin="ETH", count=60, source_session_id="historical_archive")
        for row in rows:
            row[SOURCE_CHANNEL] = "historical_l2Book"
        _write_1s_csv(path, rows)

        results = resample_coin("ETH", intervals=("15s", "1m"), data_root=data_root, source="historical")
        assert results["15s"]["output_path"].is_file()
        assert results["1m"]["output_path"].is_file()


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