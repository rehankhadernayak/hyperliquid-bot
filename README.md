# Hyperliquid Paper Trading Bot

**Version:** `8.0.0-paper` (see `config.VERSION`)

A read-only Hyperliquid market analyzer with simulated paper trading, backtesting, and a local dashboard. This release fetches **public mid prices only** — no wallets, private keys, signing, or real orders.

---

## Purpose

Build and evaluate a contrarian signal strategy on Hyperliquid perpetual markets in a safe, offline-friendly workflow:

1. Poll public prices and compute rolling signals.
2. Apply shared risk rules and produce read-only recommendations.
3. Simulate trades against a fake USDC balance.
4. Log results to CSV/JSON for review and backtesting.

Use this project to **observe**, **paper trade**, and **backtest** — not to execute live trades.

---

## Features

- **Live market analyzer** — polls Hyperliquid Info API every 10 seconds for BTC, ETH, and SOL mid prices.
- **Signal engine** — 1-minute and 5-minute rolling percentage changes with UP / DOWN / STRONG / FLAT labels.
- **Shared risk rules** — per-coin allocation, cooldowns, session exposure caps, and equity pause limits.
- **Decision engine** — PAPER BUY / PAPER SELL / HOLD / BLOCKED recommendations with reasons and confidence labels.
- **Paper trader** — simulated buys on DOWN signals, sells on UP signals when a position is open.
- **Snapshot layer** — structured terminal output plus `data/recommendations.json`.
- **CSV/JSON logging** — `data/price_log.csv`, `data/paper_trades.csv`, `data/backtest_results.json`.
- **Backtester** — replays historical prices under TEST_MODE and production thresholds side by side.
- **Streamlit dashboard** — read-only visualization of prices, trades, recommendations, and backtest summaries.
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
  CSV / JSON logs          (price_log, paper_trades, backtest_results)
        │
        ▼
    backtester             (replay price_log.csv offline)
```

| Module | Role |
|--------|------|
| `hyperliquid_client.py` | Read-only Info API client (`allMids`) |
| `market_analyzer.py` | Price history and rolling percent changes |
| `signals.py` | Threshold-based signal labels |
| `risk_rules.py` | Shared buy/sell eligibility checks |
| `decision_engine.py` | Recommendations without execution |
| `paper_trader.py` | Simulated account and trade log |
| `market_snapshot.py` | Snapshot assembly and display |
| `data_logger.py` | Append-only price CSV logging |
| `main.py` | Live analyzer loop |
| `backtester.py` | Historical replay and comparison |
| `dashboard.py` | Streamlit read-only UI |
| `config.py` | All tunable settings and `VERSION` |

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

Polls Hyperliquid public prices, prints snapshots, runs paper trading, and appends to `data/price_log.csv` and `data/paper_trades.csv`.

```bash
python3 main.py
```

Stop with `Ctrl+C`. Requires network access to `https://api.hyperliquid.xyz/info`.

### Dashboard

Read-only Streamlit UI for logged data:

```bash
streamlit run dashboard.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

### Backtester

Replays `data/price_log.csv` offline (no API calls). Writes comparison results to `data/backtest_results.json`:

```bash
python3 backtester.py
```

The backtester uses an isolated `BacktestPaperTrader` and **does not** write to `data/paper_trades.csv`.

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
| Release version | `VERSION` | Current release tag (`8.0.0-paper`) |
| Watchlist, API URL, poll interval | `MARKET` | Public market data settings |
| Signal thresholds | `SIGNAL` | Separate test vs production percentage thresholds |
| Live paper limits | `PAPER_TRADING` | Starting balance, trade size, exposure caps, session limits |
| Backtest-only settings | `DEFAULT_BACKTEST` | Max session trades (1,000), fees, slippage, forced close |
| File paths | `DATA` | CSV and JSON locations under `data/` |
| Live analyzer mode | `TEST_MODE` | `True` = test thresholds; `False` = production thresholds |

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

This v8.0 paper release is designed to stay read-only and simulation-only:

- **No wallet integration** — no private keys, mnemonics, or signing libraries.
- **No exchange execution** — no order placement, cancellation, or authenticated trading endpoints.
- **Info API only** — single public POST to `allMids`; no API secrets required.
- **Isolated backtests** — `BacktestPaperTrader` does not append to live `data/paper_trades.csv`.
- **Test guards** — pytest fixtures assert live CSV files are unchanged after the test run.
- **Health check** — scans for forbidden module names and execution-related patterns.

If you add wallet or live-trading code, the health check is expected to fail until those modules are removed or the check is intentionally updated for a future release.

---

## Known Limitations

- **Single signal input** — 1-minute rolling change only; no order book depth, funding rates, or multi-timeframe fusion.
- **Fixed watchlist** — BTC, ETH, SOL by default (configurable in `MARKET.watchlist`).
- **Simplified fills** — paper trades use mid price; backtests apply configurable slippage and fees but not full market microstructure.
- **Local data dependency** — backtests and dashboard need prior price logging (run the live analyzer or supply `data/price_log.csv`).
- **No persistence of in-memory state** — restarting the analyzer resets start prices and paper session state (CSV logs remain).

---

## Performance Disclaimer

**Backtest and paper-trading results do not guarantee future performance.**

Historical replay uses past mid prices with simplified fees and slippage. Live paper trading uses real-time public mids but still does not model latency, partial fills, liquidation, or changing market conditions. Past simulated P&amp;L is for research and tooling validation only — not investment advice.

---

## Project Layout

```text
config.py              # VERSION and all settings
main.py                # Live analyzer entry point
backtester.py          # Offline replay and comparison
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
data/                  # Local logs (CSV/JSON)
tests/                 # Offline pytest suite
CHANGELOG.md           # Release notes
```

See [CHANGELOG.md](CHANGELOG.md) for the v8.0.0-paper release summary.
