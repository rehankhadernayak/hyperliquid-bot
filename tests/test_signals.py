"""Tests for signal threshold logic."""

import pytest

from config import SIGNAL
from signals import apply_signal_thresholds, signal_from_1m_change


class TestProductionSignalThresholds:
    """Signal labels when production thresholds are active."""

    @pytest.fixture(autouse=True)
    def use_production_thresholds(self):
        apply_signal_thresholds(test_mode=False)

    def test_none_change_returns_not_available(self):
        assert signal_from_1m_change(None) == "N/A"

    def test_flat_when_change_is_between_down_and_up_thresholds(self):
        assert signal_from_1m_change(0.0) == "FLAT"
        assert signal_from_1m_change(0.24) == "FLAT"
        assert signal_from_1m_change(-0.24) == "FLAT"

    def test_up_boundary_just_above_up_threshold(self):
        assert signal_from_1m_change(0.26) == "UP"

    def test_up_boundary_exactly_at_up_threshold_stays_flat(self):
        assert signal_from_1m_change(SIGNAL.production.up_pct) == "FLAT"

    def test_up_boundary_exactly_at_strong_up_threshold_is_up_not_strong(self):
        assert signal_from_1m_change(SIGNAL.production.strong_up_pct) == "UP"

    def test_strong_up_boundary_just_above_strong_up_threshold(self):
        assert signal_from_1m_change(0.51) == "STRONG_UP"

    def test_down_boundary_just_below_down_threshold(self):
        assert signal_from_1m_change(-0.26) == "DOWN"

    def test_down_boundary_exactly_at_down_threshold_stays_flat(self):
        assert signal_from_1m_change(SIGNAL.production.down_pct) == "FLAT"

    def test_down_boundary_exactly_at_strong_down_threshold_is_down_not_strong(self):
        assert signal_from_1m_change(SIGNAL.production.strong_down_pct) == "DOWN"

    def test_strong_down_boundary_just_below_strong_down_threshold(self):
        assert signal_from_1m_change(-0.51) == "STRONG_DOWN"


class TestTestModeSignalThresholds:
    """Signal labels when TEST_MODE thresholds are active."""

    @pytest.fixture(autouse=True)
    def use_test_mode_thresholds(self):
        apply_signal_thresholds(test_mode=True)

    def test_flat_when_change_is_between_test_mode_thresholds(self):
        assert signal_from_1m_change(0.04) == "FLAT"
        assert signal_from_1m_change(-0.04) == "FLAT"

    def test_up_uses_lower_test_mode_threshold(self):
        assert signal_from_1m_change(0.06) == "UP"

    def test_strong_up_uses_lower_test_mode_threshold(self):
        assert signal_from_1m_change(0.11) == "STRONG_UP"

    def test_down_uses_lower_test_mode_threshold(self):
        assert signal_from_1m_change(-0.06) == "DOWN"

    def test_strong_down_uses_lower_test_mode_threshold(self):
        assert signal_from_1m_change(-0.11) == "STRONG_DOWN"
