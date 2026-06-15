"""
Signal logic based on rolling 1-minute price changes.
"""

# Normal thresholds (TEST_MODE = False)
NORMAL_STRONG_UP_PCT = 0.50
NORMAL_UP_PCT = 0.25
NORMAL_STRONG_DOWN_PCT = -0.50
NORMAL_DOWN_PCT = -0.25

# Lower thresholds for paper-trading signal testing (TEST_MODE = True)
TEST_STRONG_UP_PCT = 0.10
TEST_UP_PCT = 0.05
TEST_STRONG_DOWN_PCT = -0.10
TEST_DOWN_PCT = -0.05

# Active thresholds (updated by apply_signal_thresholds)
SIGNAL_STRONG_UP_PCT = NORMAL_STRONG_UP_PCT
SIGNAL_UP_PCT = NORMAL_UP_PCT
SIGNAL_STRONG_DOWN_PCT = NORMAL_STRONG_DOWN_PCT
SIGNAL_DOWN_PCT = NORMAL_DOWN_PCT


def apply_signal_thresholds(test_mode: bool) -> None:
    """Switch between normal and test signal thresholds."""
    global SIGNAL_STRONG_UP_PCT, SIGNAL_UP_PCT, SIGNAL_STRONG_DOWN_PCT, SIGNAL_DOWN_PCT

    if test_mode:
        SIGNAL_STRONG_UP_PCT = TEST_STRONG_UP_PCT
        SIGNAL_UP_PCT = TEST_UP_PCT
        SIGNAL_STRONG_DOWN_PCT = TEST_STRONG_DOWN_PCT
        SIGNAL_DOWN_PCT = TEST_DOWN_PCT
    else:
        SIGNAL_STRONG_UP_PCT = NORMAL_STRONG_UP_PCT
        SIGNAL_UP_PCT = NORMAL_UP_PCT
        SIGNAL_STRONG_DOWN_PCT = NORMAL_STRONG_DOWN_PCT
        SIGNAL_DOWN_PCT = NORMAL_DOWN_PCT


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
    if change_1m is None:
        return "N/A"

    if change_1m > SIGNAL_STRONG_UP_PCT:
        return "STRONG_UP"
    if change_1m > SIGNAL_UP_PCT:
        return "UP"
    if change_1m < SIGNAL_STRONG_DOWN_PCT:
        return "STRONG_DOWN"
    if change_1m < SIGNAL_DOWN_PCT:
        return "DOWN"
    return "FLAT"
