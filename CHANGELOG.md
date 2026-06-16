# Changelog

All notable changes to this project are documented here.

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
