"""
Central configuration for the read-only paper trading bot.

All tunable settings live here. Change values in this file before running
main.py (live analyzer), backtester.py, or the dashboard.

No wallets, signing, or real orders — paper trading only.
"""

from dataclasses import dataclass
from pathlib import Path

VERSION = "8.2.0-paper"


@dataclass(frozen=True)
class SignalThresholds:
    """Percentage change thresholds for UP / DOWN signals."""

    strong_up_pct: float
    up_pct: float
    strong_down_pct: float
    down_pct: float


@dataclass(frozen=True)
class MarketConfig:
    """Public market data settings (read-only Info API)."""

    watchlist: tuple[str, ...]
    polling_interval_seconds: int
    api_url: str
    request_timeout_seconds: int


@dataclass(frozen=True)
class SignalConfig:
    """Signal thresholds for test mode vs production mode."""

    test_mode: SignalThresholds
    production: SignalThresholds


@dataclass(frozen=True)
class PaperTradingConfig:
    """Fake account and session risk limits for live paper trading."""

    starting_balance_usdc: float
    trade_size_usdc: float
    max_allocation_per_coin_usdc: float
    max_entries_per_coin: int
    min_seconds_between_buys: int
    max_total_open_exposure_usdc: float
    max_successful_trades_per_session: int
    min_equity_before_pause_usdc: float
    trading_fee_pct: float
    slippage_pct: float


@dataclass
class BacktestConfig:
    """
    Settings used only during backtests.

    Does not change live paper trading defaults (10 session trades).
    Backtests default to 1,000 session trades unless you override this.
    """

    max_session_trades: int = 1000
    trading_fee_pct: float = 0.05
    slippage_pct: float = 0.02
    close_positions_at_end: bool = True


@dataclass(frozen=True)
class DataPaths:
    """Local CSV and JSON file locations."""

    data_dir: Path
    price_log_csv: Path
    paper_trades_csv: Path
    backtest_results_json: Path
    recommendations_json: Path
    equity_log_csv: Path
    session_configs_dir: Path
    session_reports_dir: Path
    experiments_dir: Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Defaults for offline strategy experiments."""

    development_pct: float = 0.70
    walk_forward_min_development_timestamps: int = 10
    walk_forward_evaluation_window: int = 5
    close_positions_at_end: bool = True


@dataclass(frozen=True)
class EvidenceRules:
    """
    Rule thresholds for evidence assessment labels.

    Labels are descriptive only — never mark a strategy proven or profitable.
    See README for the full decision tree.
    """

    min_completed_trades: int = 5
    min_completed_trades_promising: int = 10
    max_drawdown_reject_pct: float = 25.0
    max_drawdown_weak_pct: float = 15.0
    min_out_of_sample_return_pct: float = 0.0
    min_profit_factor: float = 1.1
    max_dev_oos_return_gap_pct: float = 10.0
    max_single_trade_profit_share_pct: float = 60.0
    max_single_coin_profit_share_pct: float = 70.0
    max_fees_of_gross_profit_pct: float = 50.0
    max_parameter_combinations: int = 20


# Evidence assessment labels (descriptive only).
EVIDENCE_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
EVIDENCE_WEAK = "WEAK"
EVIDENCE_PROMISING = "PROMISING"
EVIDENCE_UNSTABLE = "UNSTABLE"
EVIDENCE_REJECT = "REJECT"


# --- Market ---

MARKET = MarketConfig(
    watchlist=("BTC", "ETH", "SOL"),
    polling_interval_seconds=10,
    api_url="https://api.hyperliquid.xyz/info",
    request_timeout_seconds=10,
)


# --- Signals ---

SIGNAL = SignalConfig(
    test_mode=SignalThresholds(
        strong_up_pct=0.10,
        up_pct=0.05,
        strong_down_pct=-0.10,
        down_pct=-0.05,
    ),
    production=SignalThresholds(
        strong_up_pct=0.50,
        up_pct=0.25,
        strong_down_pct=-0.50,
        down_pct=-0.25,
    ),
)


# --- Paper trading (live analyzer) ---

PAPER_TRADING = PaperTradingConfig(
    starting_balance_usdc=1000.0,
    trade_size_usdc=100.0,
    max_allocation_per_coin_usdc=300.0,
    max_entries_per_coin=3,
    min_seconds_between_buys=60,
    max_total_open_exposure_usdc=600.0,
    max_successful_trades_per_session=10,
    min_equity_before_pause_usdc=980.0,
    trading_fee_pct=0.05,
    slippage_pct=0.02,
)


# --- Backtest defaults ---

DEFAULT_BACKTEST = BacktestConfig()


# --- Data files ---

_DATA_DIR = Path("data")

DATA = DataPaths(
    data_dir=_DATA_DIR,
    price_log_csv=_DATA_DIR / "price_log.csv",
    paper_trades_csv=_DATA_DIR / "paper_trades.csv",
    backtest_results_json=_DATA_DIR / "backtest_results.json",
    recommendations_json=_DATA_DIR / "recommendations.json",
    equity_log_csv=_DATA_DIR / "equity_log.csv",
    session_configs_dir=_DATA_DIR / "session_configs",
    session_reports_dir=_DATA_DIR / "session_reports",
    experiments_dir=_DATA_DIR / "experiments",
)


# --- Experiments ---

DEFAULT_EXPERIMENT = ExperimentConfig()
EVIDENCE = EvidenceRules()


# --- Live analyzer ---

# True = use SIGNAL.test_mode thresholds (more frequent paper trades).
# False = use SIGNAL.production thresholds.
TEST_MODE = True
