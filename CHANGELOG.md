# Changelog

All notable changes to this project are documented here.

## [8.2.0-paper] — 2026-06-17

Evidence-analysis and strategy-experiment framework. Still read-only market data and simulated trades only — no wallets, signing, or live order execution.

### Added

- **Strategy interface** (`strategy_types.py`, `strategies/`) — reusable protocol with structured `StrategySignalResult`; `contrarian_v1` preserves existing percentage-change behavior exactly.
- **Strategy profiles** (`strategy_profiles.py`) — immutable named profiles: `contrarian_v1_test`, `contrarian_v1_production`, `hold_baseline`; full profile saved with every experiment result.
- **Experiment runner** (`experiment_runner.py`) — replays historical prices for multiple profiles on identical timestamps; isolated traders; chronological 70/30 development/out-of-sample split; optional walk-forward; results under `data/experiments/`.
- **Experiment metrics** (`experiment_metrics.py`) — net return, fees, slippage, drawdown, exposure time, holding time, per-coin breakdown, and safe zero-trade handling.
- **Evidence assessment** (`experiment_evidence.py`, `config.EVIDENCE`) — descriptive labels (`INSUFFICIENT_DATA`, `WEAK`, `PROMISING`, `UNSTABLE`, `REJECT`); rules documented in README and config.
- **Overfitting warnings** (`experiment_overfitting.py`) — trade count, dev/OOS gap, single-trade/coin dominance, fee drag, and data-leakage checks.
- **Dashboard Experiment Analysis** — experiment selector, profile comparison, dev/OOS metrics, equity and drawdown curves, evidence labels, warnings, and full profile config.
- **Tests** — comprehensive offline coverage in `tests/test_experiments.py` (99 tests total).

### Changed

- **Version** — `VERSION = "8.2.0-paper"`.
- **`signals.py`** — extracted `signal_from_change()` for profile-aware strategies; `signal_from_1m_change()` behavior unchanged.

### Safety

- No wallet, signing, exchange execution, deposits, withdrawals, or leverage added.
- Experiments never write to live `data/paper_trades.csv`, `data/price_log.csv`, or `data/equity_log.csv`.
- v8.1 session analytics and existing data formats preserved.

## [8.1.0-paper] — 2026-06-17

Forward-testing analytics release. Still read-only market data and simulated trades only — no wallets, signing, or live order execution.

### Added

- **Session tracking** — unique `session_id` per analyzer run, recorded in paper trades and equity snapshots; mode label (`TEST_MODE` / `production`); configuration snapshots in `data/session_configs/`.
- **Live paper-trading costs** — `PAPER_TRADING.trading_fee_pct` and `slippage_pct` applied to live paper fills using the same rules as `BacktestPaperTrader`.
- **Equity logging** — append-only `data/equity_log.csv` with one row per analyzer cycle.
- **Extended paper trades CSV** — `session_id`, `mode`, `signal`, `entry_count`, `fee`, `slippage`, `trade_pnl`, `cumulative_realized_pnl`; legacy CSV rows are normalized on read and migrated on next write.
- **Session reports** (`session_report.py`) — per-session metrics saved to `data/session_reports/<session_id>.json`; auto-generated on analyzer shutdown.
- **Dashboard analytics** — session/date filters, current-session vs all-time views, equity curve, drawdown, cumulative realized PnL, PnL by coin, fees, win rate, and profit factor.
- **`dashboard_data.py`** — shared read-only loading and analytics helpers for the dashboard and session reports.
- **Tests** — coverage for session IDs, fees/slippage, equity logging, session reports, legacy CSV compatibility, and dashboard helpers.

### Changed

- **Version** — `VERSION = "8.1.0-paper"`.
- **Paper trading summary** — terminal and snapshot output now show session ID, mode, and fees paid.
- **`recommendations.json`** — includes `session_id` and expanded paper-trader fields (`fees_paid`, `mode`).

### Safety

- No wallet, signing, exchange execution, deposits, withdrawals, or leverage added.
- Tests fail if live `data/paper_trades.csv`, `data/price_log.csv`, or `data/equity_log.csv` are modified during the run.

## [8.0.0-paper] — 2026-06-16

Paper-trading release. Read-only market data and simulated trades only — no wallets, signing, or live order execution.

### Added

- **Decision engine** (`decision_engine.py`) — per-coin recommendations (PAPER BUY / PAPER SELL / HOLD / BLOCKED) with confidence labels and risk-aware reasons.
- **Shared risk rules** (`risk_rules.py`) — single source of truth for buy/sell eligibility used by both the decision engine and paper trader.
- **Market snapshot layer** (`market_snapshot.py`) — structured snapshots, terminal output, and `recommendations.json` for the dashboard.
- **Backtest comparison** — replays `data/price_log.csv` under TEST_MODE and production thresholds; writes `data/backtest_results.json`.
- **Streamlit dashboard** — read-only views for recommendations, session state, price history, paper trades, and backtest summaries.
- **Health check** (`health_check.py`) — verifies data dirs, config, imports, tests, and absence of wallet/execution modules.
- **Test suite** — offline pytest coverage for signals, paper trader, decision engine, risk alignment, backtester, and config.
- **Version constant** — `VERSION = "8.0.0-paper"` in `config.py`.

### Safety

- Hyperliquid **Info API only** (public mid prices); no authenticated exchange endpoints.
- Paper trades logged to local CSV; backtests use an isolated `BacktestPaperTrader` that never writes to live paper trade files.
- Tests include fixtures that fail if `data/paper_trades.csv` or `data/price_log.csv` are modified during the run.

### Configuration

- Central `config.py` for market watchlist, signal thresholds, paper trading limits, backtest fees/slippage, and data paths.
- `TEST_MODE` toggles lower signal thresholds for development; production thresholds are stricter and used by default after each test.

### Known limitations

- Signals use rolling 1-minute percentage change only; no order book, funding, or on-chain data.
- Backtests assume fixed fees and slippage; results do not guarantee future performance.
- Dashboard and analyzer require locally logged data; the analyzer needs network access to Hyperliquid's public API.
