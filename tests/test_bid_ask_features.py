"""Tests for shared bid/ask feature engineering (Candidate F regime-depth)."""

from __future__ import annotations

import pandas as pd
import pytest

from bid_ask_features import (
    CANDIDATE_E_DEPTH_AWARE_COLUMNS,
    CANDIDATE_F_REGIME_DEPTH_COLUMNS,
    CANDIDATE_F_REGIME_DEPTH_EXTRA_COLUMNS,
    assert_no_future_feature_columns,
    create_candidate_f_regime_depth_features,
    parse_feature_set,
)
from bid_ask_resampler import RESAMPLED_COLUMNS

BASE_MS = 1_718_000_040_000


def _make_resampled_rows(
    *,
    count: int,
    start_ms: int = BASE_MS,
    step_ms: int = 15_000,
    mid_start: float = 100.0,
    mid_step: float = 0.5,
    spread_bps: float = 2.0,
    imbalance: float = 0.2,
    bid_size: float = 2.0,
    ask_size: float = 1.0,
) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        start = start_ms + index * step_ms
        end = start + step_ms - 1
        mid = mid_start + index * mid_step
        rows.append(
            {
                "start_timestamp_ms": start,
                "end_timestamp_ms": end,
                "timestamp_iso": f"2024-06-10T00:00:{index % 60:02d}Z",
                "coin": "BTC",
                "open_mid": mid,
                "high_mid": mid + 0.1,
                "low_mid": mid - 0.1,
                "close_mid": mid,
                "avg_bid": mid - 0.5,
                "avg_ask": mid + 0.5,
                "avg_mid": mid,
                "avg_spread": 1.0,
                "avg_spread_bps": spread_bps + (index % 5) * 0.2,
                "avg_bbo_imbalance": imbalance + index * 0.01,
                "row_count": 15,
                "missing_seconds": 0,
                "bid_size": bid_size + index * 0.01,
                "ask_size": ask_size + index * 0.01,
            }
        )
    return rows


class TestCandidateFRegimeDepthFeatures:
    def test_candidate_f_includes_all_candidate_e_and_regime_columns(self) -> None:
        raw_df = pd.DataFrame(_make_resampled_rows(count=300), columns=list(RESAMPLED_COLUMNS))
        featured, _ = create_candidate_f_regime_depth_features(raw_df, interval="15s")

        for column in CANDIDATE_E_DEPTH_AWARE_COLUMNS:
            assert column in featured.columns
        for column in CANDIDATE_F_REGIME_DEPTH_EXTRA_COLUMNS:
            assert column in featured.columns
        assert list(CANDIDATE_F_REGIME_DEPTH_COLUMNS) == list(CANDIDATE_E_DEPTH_AWARE_COLUMNS) + list(
            CANDIDATE_F_REGIME_DEPTH_EXTRA_COLUMNS
        )

    def test_rolling_features_use_only_past_and_current_data(self) -> None:
        raw_df = pd.DataFrame(
            _make_resampled_rows(count=300, mid_start=100.0, mid_step=1.0),
            columns=list(RESAMPLED_COLUMNS),
        )
        featured, _ = create_candidate_f_regime_depth_features(raw_df, interval="15s")

        altered = raw_df.copy()
        altered.loc[altered.index[-1], "close_mid"] = altered.loc[altered.index[-1], "close_mid"] * 2
        altered_featured, _ = create_candidate_f_regime_depth_features(altered, interval="15s")

        for column in CANDIDATE_F_REGIME_DEPTH_COLUMNS:
            pd.testing.assert_series_equal(
                featured[column].iloc[:-1],
                altered_featured[column].iloc[:-1],
                check_names=False,
            )

    def test_missing_depth_columns_fail_clearly(self) -> None:
        rows = _make_resampled_rows(count=40)
        for row in rows:
            row.pop("bid_size", None)
            row.pop("ask_size", None)
        raw_df = pd.DataFrame(rows, columns=list(RESAMPLED_COLUMNS))

        with pytest.raises(ValueError, match="bid_size/ask_size"):
            create_candidate_f_regime_depth_features(raw_df, interval="15s")

    def test_missing_spread_column_fails_clearly(self) -> None:
        raw_df = pd.DataFrame(_make_resampled_rows(count=40), columns=list(RESAMPLED_COLUMNS))
        raw_df = raw_df.drop(columns=["avg_spread_bps"])

        with pytest.raises(ValueError, match="spread"):
            create_candidate_f_regime_depth_features(raw_df, interval="15s")

    def test_no_future_return_columns_in_features(self) -> None:
        raw_df = pd.DataFrame(_make_resampled_rows(count=80), columns=list(RESAMPLED_COLUMNS))
        featured, _ = create_candidate_f_regime_depth_features(raw_df, interval="15s")
        assert_no_future_feature_columns(CANDIDATE_F_REGIME_DEPTH_COLUMNS)
        assert "future_return_bps" not in featured.columns
        assert "label" not in featured.columns

    def test_parse_feature_set_recognizes_candidate_f_regime_depth(self) -> None:
        assert parse_feature_set("candidate_f_regime_depth") == "candidate_f_regime_depth"
        assert parse_feature_set("CANDIDATE_F_REGIME_DEPTH") == "candidate_f_regime_depth"

    def test_regime_flags_are_binary(self) -> None:
        raw_df = pd.DataFrame(_make_resampled_rows(count=300), columns=list(RESAMPLED_COLUMNS))
        featured, _ = create_candidate_f_regime_depth_features(raw_df, interval="15s")

        for column in (
            "low_volatility_regime",
            "high_volatility_regime",
            "tight_spread_regime",
            "wide_spread_regime",
        ):
            values = featured[column].dropna().unique()
            assert set(values).issubset({0.0, 1.0})
