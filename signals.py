"""
Signal logic based on rolling 1-minute price changes.
"""

# Signal thresholds based on approximate 1-minute percentage change
SIGNAL_STRONG_UP_PCT = 0.50
SIGNAL_UP_PCT = 0.25
SIGNAL_STRONG_DOWN_PCT = -0.50
SIGNAL_DOWN_PCT = -0.25


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
