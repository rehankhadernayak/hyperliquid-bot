"""
Signal logic based on rolling 1-minute price changes.
"""

from config import SIGNAL, SignalThresholds

# Active thresholds (updated by apply_signal_thresholds)
SIGNAL_STRONG_UP_PCT = SIGNAL.production.strong_up_pct
SIGNAL_UP_PCT = SIGNAL.production.up_pct
SIGNAL_STRONG_DOWN_PCT = SIGNAL.production.strong_down_pct
SIGNAL_DOWN_PCT = SIGNAL.production.down_pct


def apply_signal_thresholds(test_mode: bool) -> None:
    """Switch between normal and test signal thresholds."""
    global SIGNAL_STRONG_UP_PCT, SIGNAL_UP_PCT, SIGNAL_STRONG_DOWN_PCT, SIGNAL_DOWN_PCT

    thresholds = SIGNAL.test_mode if test_mode else SIGNAL.production
    SIGNAL_STRONG_UP_PCT = thresholds.strong_up_pct
    SIGNAL_UP_PCT = thresholds.up_pct
    SIGNAL_STRONG_DOWN_PCT = thresholds.strong_down_pct
    SIGNAL_DOWN_PCT = thresholds.down_pct


def signal_from_change(
    change_1m: float | None,
    thresholds: SignalThresholds,
) -> str:
    """
    Return a signal based on approximate 1-minute percentage change.

    Uses explicit thresholds so strategies and experiments do not depend on
    module-level global state.
    """
    if change_1m is None:
        return "N/A"

    if change_1m > thresholds.strong_up_pct:
        return "STRONG_UP"
    if change_1m > thresholds.up_pct:
        return "UP"
    if change_1m < thresholds.strong_down_pct:
        return "STRONG_DOWN"
    if change_1m < thresholds.down_pct:
        return "DOWN"
    return "FLAT"


def signal_from_1m_change(change_1m: float | None) -> str:
    """
    Return a signal based on approximate 1-minute percentage change.

    STRONG_UP   if 1-minute change > +0.50%
    UP          if 1-minute change > +0.25%
    STRONG_DOWN if 1-minute change < -0.50%
    DOWN        if 1-minute change < -0.25%
    FLAT        otherwise
    N/A         when there is not enough history yet
    """
    active = SignalThresholds(
        strong_up_pct=SIGNAL_STRONG_UP_PCT,
        up_pct=SIGNAL_UP_PCT,
        strong_down_pct=SIGNAL_STRONG_DOWN_PCT,
        down_pct=SIGNAL_DOWN_PCT,
    )
    return signal_from_change(change_1m, active)
