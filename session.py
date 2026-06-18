"""Session identity and configuration snapshot helpers."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict

from config import (
    DEFAULT_BACKTEST,
    MARKET,
    PAPER_TRADING,
    SIGNAL,
    TEST_MODE,
    VERSION,
)

LIVE_STRATEGY_PROFILE = "contrarian_v1"


def generate_session_id() -> str:
    """Return a unique session identifier for one analyzer run."""
    return uuid.uuid4().hex


def mode_label(test_mode: bool) -> str:
    """Return a stable mode label for logs and reports."""
    return "TEST_MODE" if test_mode else "production"


def session_config_snapshot(*, test_mode: bool = TEST_MODE) -> dict:
    """Capture configuration values active for this session."""
    return {
        "version": VERSION,
        "test_mode": test_mode,
        "mode": mode_label(test_mode),
        "strategy_profile": LIVE_STRATEGY_PROFILE,
        "market": asdict(MARKET),
        "signal_thresholds": asdict(SIGNAL.test_mode if test_mode else SIGNAL.production),
        "paper_trading": asdict(PAPER_TRADING),
        "costs": {
            "trading_fee_pct": PAPER_TRADING.trading_fee_pct,
            "slippage_pct": PAPER_TRADING.slippage_pct,
        },
        "backtest_defaults": asdict(DEFAULT_BACKTEST),
    }


def configuration_hash(config_snapshot: dict) -> str:
    """Return a stable SHA-256 hash of the session configuration snapshot."""
    payload = json.dumps(config_snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_session_config(session_id: str, config_snapshot: dict, output_dir) -> None:
    """Write the session configuration snapshot to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{session_id}.json"

    import json

    with path.open("w", encoding="utf-8") as handle:
        json.dump(config_snapshot, handle, indent=2)
