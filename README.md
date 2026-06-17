# Hyperliquid Paper Trading Bot

**Version:** `8.2.0-paper` (see `config.VERSION`)

A read-only Hyperliquid market analyzer with simulated paper trading, backtesting, strategy experiments, and a local dashboard. This release fetches **public mid prices only** — no wallets, private keys, signing, or real orders.

---

## Purpose

Build and evaluate a contrarian signal strategy on Hyperliquid perpetual markets in a safe, offline-friendly workflow:

1. Poll public prices and compute rolling signals.
2. Apply shared risk rules and produce read-only recommendations.
3. Simulate trades against a fake USDC balance.
4. Log results to CSV/JSON for review, backtesting, and offline strategy experiments.

Use this project to **observe**, **paper trade**, and **backtest** — not to execute live trades.

---

## Features

- **Live market analyzer** — polls Hyperliquid Info API every 10 seconds for BTC, ETH, and SOL mid prices.
- **Signal engine** — 1-minute and 5-minute rolling percentage changes with UP / DOWN / STRONG / FLAT labels.
- **Shared risk rules** — per-coin allocation, cooldowns, session exposure caps, and equity pause limits.
- **Decision engine** — PAPER BUY / PAPER SELL / HOLD / BLOCKED recommendations with reasons and confidence labels.
- **Paper trader** — simulated buys on DOWN signals, sells on UP signals when a position is open.
- **Snapshot layer** — structured terminal output plus `data/recommendations.json`.
- **CSV/JSON logging** — `data/price_log.csv`, `data/paper_trades.csv`, `data/equity_log.csv`, `data/backtest_results.json`.
- **Session tracking** — unique `session_id`, mode label, and configuration snapshot per analyzer run.
- **Forward-testing costs** — live paper trades apply the same fee and slippage rules as the backtester.
- **Session reports** — `session_report.py` writes per-session analytics to `data/session_reports/`.
- **Backtester** — replays historical prices under TEST_MODE and production thresholds side by side.
- **Strategy experiments** — compare immutable strategy profiles on identical historical data with development/out-of-sample splits, evidence labels, and overfitting warnings (`experiment_runner.py`).
- **Streamlit dashboard** — read-only visualization with session/date filters, equity curve, drawdown, PnL analytics, and experiment analysis.
- **Health check** — `python3 health_check.py` verifies setup, imports, tests, and safety constraints.

---

## Architecture

```text
Hyperliquid Info API
        │
        ▼
  market analyzer          (price history, rolling % changes)
        │
        ▼
   signal engine           (1m change → UP/DOWN/FLAT labels)
        │
        ▼
  shared risk rules        (allocation, cooldown, session limits)
        │
        ▼
  decision engine          (read-only recommendations)
        │
        ▼
   paper trader            (simulated fills, no real orders)
        │
        ▼
  snapshot layer           (terminal + recommendations.json)
        │
        ├──────────────────┐
        ▼                  ▼
   terminal            dashboard (Streamlit)
        │
        ▼
  CSV / JSON logs          (price_log, paper_trades, equity_log, backtest_results)
        │
        ├──────────────────┐
        ▼                  ▼
   session reports     dashboard (Streamlit)
        │
        ▼
    backtester             (replay price_log.csv offline)
        │
        ▼
  experiment runner        (multi-profile offline experiments)
```

| Module | Role |
|--------|------|
| `hyperliquid_client.py` | Read-only Info API client (`allMids`) |
| `market_analyzer.py` | Price history and rolling percent changes |
| `signals.py` | Threshold-based signal labels |
| `strategies/` | Named strategy implementations (`contrarian_v1`, `hold_baseline`) |
| `strategy_profiles.py` | Immutable experiment profiles |
| `experiment_runner.py` | Offline multi-profile experiment runner |
| `risk_rules.py` | Shared buy/sell eligibility checks |
| `decision_engine.py` | Recommendations without execution |
| `paper_trader.py` | Simulated account and trade log |
| `market_snapshot.py` | Snapshot assembly and display |
| `data_logger.py` | Append-only price CSV logging |
| `dashboard_data.py` | Dashboard data loading and analytics helpers |
| `session.py` | Session ID and configuration snapshots |
| `equity_logger.py` | Append-only equity CSV logging |
| `session_report.py` | Per-session forward-testing reports |
| `main.py` | Live analyzer loop |
| `backtester.py` | Historical replay and comparison |
| `dashboard.py` | Streamlit read-only UI |
| `config.py` | All tunable settings, evidence rules, and `VERSION` |

---

## Setup

**Requirements:** Python 3.10+ recommended.

1. Clone the repository and enter the project directory.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the data directory exists (created automatically on first run):

   ```bash
   mkdir -p data
   ```

4. (Optional) Run the release health check:

   ```bash
   python3 health_check.py
   ```

---

## How to Run

### Live analyzer

Polls Hyperliquid public prices, prints snapshots, runs paper trading, and appends to `data/price_log.csv`, `data/paper_trades.csv`, and `data/equity_log.csv`. Each run gets a unique session ID; configuration is saved under `data/session_configs/`.

```bash
python3 main.py
```

Stop with `Ctrl+C`. Requires network access to `https://api.hyperliquid.xyz/info`.

### Dashboard

Read-only Streamlit UI for logged data:

```bash
streamlit run dashboard.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`). Use session and date filters to compare the current session against all-time filtered results, including equity curve, drawdown, cumulative realized PnL, and PnL by coin.

### Session report

Generate a read-only JSON report for one session (from logged trades and equity):

```bash
python3 session_report.py <session_id>
```

Reports are saved to `data/session_reports/<session_id>.json`. The analyzer also writes a report on graceful shutdown (`Ctrl+C`).

### Backtester

Replays `data/price_log.csv` offline (no API calls). Writes comparison results to `data/backtest_results.json`:

```bash
python3 backtester.py
```

The backtester uses an isolated `BacktestPaperTrader` and **does not** write to `data/paper_trades.csv`.

### Strategy experiments

Compare multiple immutable strategy profiles on the same historical data (no API calls). Results are saved under `data/experiments/<experiment_id>.json`:

```bash
python3 experiment_runner.py --profiles contrarian_v1_test contrarian_v1_production hold_baseline
```

Optional flags:

- `--development-pct 0.70` — chronological split (default: first 70% development, final 30% out-of-sample)
- `--walk-forward` — additional expanding-window walk-forward folds
- `--price-log path/to/price_log.csv` — alternate input file

Each experiment saves configuration, input-data hash, development/out-of-sample/combined metrics, equity curves, trade logs, evidence assessment, and overfitting warnings. Profiles are never modified during a run.

**Historical and paper results do not guarantee future performance.**

### Tests

Offline test suite (no Hyperliquid API calls). Tests fail if live CSV files are modified:

```bash
python3 -m pytest
```

### Health check

Full release verification (data dirs, config, imports, forbidden-module scan, pytest):

```bash
python3 health_check.py
```

---

## Configuration

All settings live in **`config.py`**. Edit that file before running the analyzer, backtester, or dashboard.

| Setting | Variable | Description |
|---------|----------|-------------|
| Release version | `VERSION` | Current release tag (`8.2.0-paper`) |
| Watchlist, API URL, poll interval | `MARKET` | Public market data settings |
| Signal thresholds | `SIGNAL` | Separate test vs production percentage thresholds |
| Live paper limits | `PAPER_TRADING` | Starting balance, trade size, exposure caps, session limits, fees, slippage |
| Backtest-only settings | `DEFAULT_BACKTEST` | Max session trades (1,000), fees, slippage, forced close |
| Experiment defaults | `DEFAULT_EXPERIMENT` | Development split ratio, walk-forward window sizes |
| Evidence rules | `EVIDENCE` | Thresholds for evidence assessment labels (see below) |
| File paths | `DATA` | CSV and JSON locations under `data/` (includes `experiments_dir`) |
| Live analyzer mode | `TEST_MODE` | `True` = test thresholds; `False` = production thresholds |

### Strategy profiles

Named immutable profiles in `strategy_profiles.py`:

| Profile | Strategy | Thresholds | Purpose |
|---------|----------|------------|---------|
| `contrarian_v1_test` | `contrarian_v1` | `SIGNAL.test_mode` | More frequent signals for development |
| `contrarian_v1_production` | `contrarian_v1` | `SIGNAL.production` | Stricter production-style thresholds |
| `hold_baseline` | `hold_baseline` | n/a (always FLAT) | No-trade baseline for comparison |

Each profile defines thresholds, lookback period, cooldown, trade size, fee, slippage, allocation limits, and session limits. The full profile is saved with every experiment result.

### Evidence assessment rules

Evidence labels are **descriptive only** — they do not mark a strategy as proven or profitable, and no winner is chosen solely by highest return.

Configured in `config.EVIDENCE`:

| Label | When applied |
|-------|----------------|
| `INSUFFICIENT_DATA` | Fewer than `min_completed_trades` (default 5) completed trades in out-of-sample period |
| `REJECT` | Out-of-sample max drawdown ≥ `max_drawdown_reject_pct` (25%), or negative OOS return with profit factor &lt; 1 |
| `UNSTABLE` | Development vs OOS return gap ≥ `max_dev_oos_return_gap_pct` (10%), or &gt;60% of profit from one trade, or &gt;70% from one coin |
| `WEAK` | Fees ≥ `max_fees_of_gross_profit_pct` (50%) of gross profit, or middling results that fail PROMISING criteria |
| `PROMISING` | ≥ `min_completed_trades_promising` (10) OOS trades, OOS return ≥ 0%, drawdown &lt; `max_drawdown_weak_pct` (15%), profit factor ≥ `min_profit_factor` (1.1), and stable dev/OOS gap |

Overfitting warnings (separate from labels) cover low trade counts, dev/OOS gaps, single-trade/coin dominance, fee drag, too many profiles compared, identical split periods, and look-ahead detection.

### TEST_MODE vs production thresholds

Signal labels depend on the **1-minute rolling percentage change** crossing configured thresholds:

| Mode | When used | Strong UP / UP | Strong DOWN / DOWN | Typical effect |
|------|-----------|----------------|--------------------|----------------|
| **TEST_MODE** | `TEST_MODE = True` in `config.py`; live analyzer at startup; backtest `test_mode` arm | ±0.10% / ±0.05% | ±0.10% / ±0.05% | More frequent signals — useful for development and demos |
| **Production** | `TEST_MODE = False`; backtest `production_mode` arm | ±0.50% / ±0.25% | ±0.50% / ±0.25% | Stricter — fewer, larger-move signals |

The live analyzer calls `apply_signal_thresholds(TEST_MODE)` once at startup. The backtester runs **both** modes on the same price log so you can compare behavior. Pytest restores production thresholds after each test to avoid leaking state.

Live paper trading defaults to **10** successful trades per session. Backtests default to **1,000** session trades via `DEFAULT_BACKTEST.max_session_trades`.

---

## Safety Guarantees

This v8.2 paper release is designed to stay read-only and simulation-only:

- **No wallet integration** — no private keys, mnemonics, or signing libraries.
- **No exchange execution** — no order placement, cancellation, or authenticated trading endpoints.
- **Info API only** — single public POST to `allMids`; no API secrets required.
- **Isolated backtests and experiments** — `BacktestPaperTrader` and `ExperimentPaperTrader` do not append to live CSV files.
- **Test guards** — pytest fixtures assert live CSV files are unchanged after the test run.
- **Health check** — scans for forbidden module names and execution-related patterns.

If you add wallet or live-trading code, the health check is expected to fail until those modules are removed or the check is intentionally updated for a future release.

---

## Known Limitations

- **Single signal input** — 1-minute rolling change only; no order book depth, funding rates, or multi-timeframe fusion.
- **Fixed watchlist** — BTC, ETH, SOL by default (configurable in `MARKET.watchlist`).
- **Simplified fills** — live paper trading and backtests apply configurable slippage and fees using the same rules; neither models full market microstructure.
- **Local data dependency** — backtests, experiments, and dashboard need prior price logging (run the live analyzer or supply `data/price_log.csv`).
- **Experiment scope** — one contrarian strategy family plus a hold baseline; no hyperparameter search or autonomous execution.
- **Session-scoped in-memory state** — restarting the analyzer resets start prices and in-memory paper state; CSV/JSON logs and session reports persist across runs.

---

## Performance Disclaimer

**Historical and paper results do not guarantee future performance.**

Historical replay and live paper trading use public mids with simplified fees and slippage. Neither models latency, partial fills, liquidation, or changing market conditions. Past simulated P&amp;L is for research and tooling validation only — not investment advice.

---

## Project Layout

```text
config.py              # VERSION and all settings
main.py                # Live analyzer entry point
session.py             # Session ID and config snapshots
equity_logger.py       # Equity CSV appends
session_report.py      # Per-session analytics reports
dashboard_data.py      # Dashboard loading and analytics helpers
backtester.py          # Offline replay and comparison
experiment_runner.py   # Multi-profile strategy experiments
strategy_profiles.py   # Immutable strategy profiles
strategies/            # Strategy implementations
dashboard.py           # Streamlit dashboard
health_check.py        # Release verification script
hyperliquid_client.py  # Read-only Info API
market_analyzer.py     # Price history math
signals.py             # Signal thresholds and labels
risk_rules.py          # Shared risk eligibility
decision_engine.py     # Recommendations layer
paper_trader.py        # Simulated trading
market_snapshot.py     # Snapshots and terminal output
data_logger.py         # Price CSV appends
data/                  # Local logs (CSV/JSON/experiments)
tests/                 # Offline pytest suite
CHANGELOG.md           # Release notes
```

See [CHANGELOG.md](CHANGELOG.md) for the v8.2.0-paper release summary.
