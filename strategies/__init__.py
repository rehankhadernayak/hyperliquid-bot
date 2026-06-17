"""Named strategy implementations for offline experiments."""

from strategies.contrarian_v1 import ContrarianV1Strategy
from strategies.hold_baseline import HoldBaselineStrategy

STRATEGY_REGISTRY: dict[str, type] = {
    "contrarian_v1": ContrarianV1Strategy,
    "hold_baseline": HoldBaselineStrategy,
}


def get_strategy(strategy_name: str):
    """Instantiate a registered strategy by name."""
    if strategy_name not in STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy: {strategy_name!r}")
    return STRATEGY_REGISTRY[strategy_name]()
