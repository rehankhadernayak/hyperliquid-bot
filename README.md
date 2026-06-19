# Hyperliquid Paper Trading Bot

**Version:** `9.0.0-review` (see `config.VERSION`)

A read-only Hyperliquid market analyzer with simulated paper trading, backtesting, strategy experiments, adviser review packaging, and a local dashboard. This release fetches **public mid prices only** — no wallets, private keys, signing, or real orders.

**This release cannot place real trades and is intended only for supervised technical and risk review.**

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
- **Streamlit dashboard** — read-only visualization with session/date filters, equity curve, drawdown, PnL analytics, experiment analysis, and **Review Mode** for adviser meetings.
- **Review package generator** — `python3 review_package.py` creates a timestamped adviser review bundle under `review_packages/`.
- **Capital readiness assessment** — conservative descriptive assessment via `python3 capital_readiness.py` (never authorizes real trading).
- **Release review check** — `python3 review_check.py` verifies version, safety, tests, package generation, and checksums.
- **Health check** — `python3 health_check.py` verifies setup, imports, tests, and safety constraints.
- **WebSocket market data recorder** — read-only public WebSocket capture to JSONL (raw) and Parquet (normalized) under `data/market_stream/`.
- **Market data quality reports** — `python3 market_data_quality_report.py` validates recorder sessions for research usability.
- **Market data replay** — `python3 market_data_replay.py` deterministically replays recorded WebSocket sessions (read-only, paper-only).
- **Event-driven backtester** — `python3 event_backtest.py` simulates strategy decisions on replayed trades and BBO with bid/ask-aware fills (read-only, paper-only, research-only).

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
| `websocket_recorder.py` | Read-only WebSocket market data recorder CLI |
| `market_data/` | Recorder, normalizer, gap/latency tracking, quality reports |
| `market_data_quality_report.py` | Session data-quality validator (read-only) |
| `market_data_replay.py` | Deterministic WebSocket session replay (read-only) |

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

Read-only Streamlit UI for logged data. Includes a **Review Mode** tab for adviser meetings (no export or execution from the UI):

```bash
streamlit run dashboard.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`). Use session and date filters to compare the current session against all-time filtered results, including equity curve, drawdown, cumulative realized PnL, and PnL by coin.

### Review package (adviser meeting)

Generate a timestamped review bundle for supervised technical and risk review:

```bash
python3 review_package.py
```

Creates `review_packages/<package_id>/` with executive summary, architecture docs, strategy documentation, evidence CSVs, manifest, and checksums. Export is CLI-only — not available from the dashboard.

### Capital readiness assessment

Conservative descriptive assessment (never authorizes real trading):

```bash
python3 capital_readiness.py
```

Optional JSON output: `python3 capital_readiness.py --json`

### Release review check

Full v9 release verification:

```bash
python3 review_check.py
```

### Session report

Generate a read-only JSON report for one session (from logged trades and equity):

```bash
python3 session_report.py <session_id>
```

Reports are saved to `data/session_reports/<session_id>.json`. The analyzer also writes a report on graceful shutdown (`Ctrl+C`).

### WebSocket market data recorder

Collect read-only public market data over WebSocket. No wallets, signing, or order execution.

```bash
python3 websocket_recorder.py --coins BTC ETH SOL
python3 websocket_recorder.py --duration-seconds 300
python3 websocket_recorder.py --status
```

Output layout:

- `data/market_stream/raw/YYYY-MM-DD/events-HH.jsonl` — append-only raw JSONL (source of truth)
- `data/market_stream/normalized/...` — partitioned Parquet (derived, for analysis)
- `data/market_stream/manifests/<session_id>.json` — session manifest

Channels: `allMids`, `trades`, `bbo`, `l2Book`, `candle`, `activeAssetCtx`.

### Market data quality report

Validate whether a recorder session is reliable enough for research and future backtesting. Reads manifest and listed files only; writes separate reports under `data/market_stream/quality_reports/`. Does **not** modify raw, normalized, or session files.

```bash
python3 market_data_quality_report.py --session-id <RECORDER_SESSION_ID>
python3 market_data_quality_report.py --latest
python3 market_data_quality_report.py --session-id <ID> --format json
python3 market_data_quality_report.py --session-id <ID> --format markdown
```

#### Quality labels

| Label | Meaning |
|-------|---------|
| `GOOD_FOR_RESEARCH` | Manifest exists, raw files readable, adequate channel coverage, acceptable reconnect/gap profile |
| `USABLE_WITH_WARNINGS` | Minor gaps, duplicate timestamps, latency warnings, or partial channel completeness — raw data preserved |
| `DEGRADED` | Missing normalized files, high malformed rate, stale channels, timestamp issues, or count mismatches |
| `UNUSABLE` | Missing manifest/raw files, unreadable records, corrupt data, or no usable market data |

Thresholds are defined in `market_data/quality_models.py`.

#### Raw data is the source of truth

Quality reports always inspect raw JSONL first. Normalized Parquet is derived (deduplicated, expanded for `allMids` and batched trades). Count mismatches between raw and normalized are expected when one raw message expands into many rows.

#### Latency by channel

Latency (`received_timestamp_ms - exchange_timestamp_ms`) is computed separately per channel:

- **trades, bbo, l2Book** — suitable for exchange-to-receive latency research
- **allMids** — latency unavailable (no exchange send timestamp)
- **candle** — informational only (timestamps are candle open/close, not message send time)
- **activeAssetCtx** — contextual; not comparable to trade timestamps

Negative latency values are **not** clamped; they indicate clock-quality warnings.

#### Important disclaimer

Quality reports assess **data integrity only**. They do **not** prove strategy profitability or authorize live trading.

### Market data replay

Deterministically replay a recorded WebSocket session from normalized Parquet files. Reads manifest and quality report; writes separate replay reports under `data/market_stream/replays/`. Does **not** modify raw, normalized, or session files. Does **not** connect to paper trading or execution.

```bash
python3 market_data_replay.py --session-id <RECORDER_SESSION_ID>
python3 market_data_replay.py --latest
python3 market_data_replay.py --session-id <ID> --dry-run
python3 market_data_replay.py --session-id <ID> --coins BTC ETH SOL
python3 market_data_replay.py --session-id <ID> --channels trades bbo l2Book candle activeAssetCtx allMids
python3 market_data_replay.py --session-id <ID> --speed 1.0
python3 market_data_replay.py --session-id <ID> --speed 0
python3 market_data_replay.py --session-id <ID> --max-events 1000
python3 market_data_replay.py --session-id <ID> --output-dir data/market_stream/replays
```

#### Dry-run (default)

Dry-run validates the session, applies filters, sorts deterministically, computes counts and a stream hash, and writes a replay report — without sleeping or emitting events to consumers. This is the safest validation mode.

#### Speed control

- `--speed 1.0` — replay using recorded timing between events
- `--speed 10.0` — replay 10× faster
- `--speed 0` — emit as fast as possible (no sleeping)
- Use `--emit` to stream events to a consumer (disables dry-run)

#### Filtering

Filter by coin (`--coins`), channel (`--channels`), time range (`--start-timestamp-ms`, `--end-timestamp-ms`), or event count (`--max-events`). Filters do not modify source data.

#### Deterministic ordering

Events are sorted by: (1) exchange timestamp when available and safe, otherwise received timestamp; (2) received timestamp tie-breaker; (3) original local sequence; (4) channel priority; (5) coin alphabetical order; (6) source file path; (7) source row index. Running replay twice with the same data and filters produces the same stream hash.

#### Important disclaimers

- Replay is **read-only** and **paper-only** — no wallets, signing, orders, or execution.
- Replay is **separate from trading execution** — output is not connected to `paper_trader.py`.
- Replay does **not** prove strategy profitability — it only reconstructs historical market data in deterministic order.

### Event-driven backtesting

Event-driven backtesting answers: *“Can the strategy be tested fairly on recorded event data with bid/ask-aware fills, costs, risk limits, and no future leakage?”*

It differs from **row-based experiments** (`experiment_runner.py`):

| | Row-based experiments | Event-driven backtests |
| --- | --- | --- |
| Input | `data/price_log.csv` timestamp rows | Recorded WebSocket replay (trades + BBO) |
| Timing | One row per polling timestamp | One event at a time from replay cursor |
| Fill model | Midpoint/last price + slippage | Best bid (sell) / best ask (buy) + slippage |
| Output | `data/experiments/` | `data/event_backtests/` |

Pipeline:

```text
recorded WebSocket data → quality report → deterministic replay → event-driven backtester
```

```bash
python3 event_backtest.py --replay-session-id <REPLAY_ID>
python3 event_backtest.py --source-recorder-session-id <RECORDER_ID>
python3 event_backtest.py --latest-replay
python3 event_backtest.py --replay-session-id <ID> --coins BTC ETH SOL
python3 event_backtest.py --replay-session-id <ID> --channels trades bbo
python3 event_backtest.py --strategy contrarian_v1_medium_no_trend_filter
python3 event_backtest.py --starting-equity 1000
python3 event_backtest.py --output-dir data/event_backtests
python3 event_backtest.py --max-events 10000
python3 event_backtest.py --replay-session-id <ID> --dry-run
```

#### Why bid/ask fills matter

Row-based backtests fill at midpoint or last trade price. Event backtests fill at the **best ask** for buys and **best bid** for sells (plus configured slippage and fees). This is more realistic for contrarian strategies that react to trades but would execute against the current book.

#### Limitations

- v1 supports **trades** and **bbo** channels only (not l2Book, candle, activeAssetCtx, allMids).
- A **~300-second smoke dataset** produces few completed trades — expect `INSUFFICIENT_EVENT_DATA` evidence labels.
- Results between row-based and event-based frameworks are **not expected to match exactly**.
- Neither framework is claimed more correct without further evidence.

#### Safety disclaimer

Event backtesting is **paper-only research**. It does not authorize live trading, connect to execution, or modify recorder/replay/experiment files. Reports are written only under `data/event_backtests/`.

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
| Release version | `VERSION` | Current release tag (`9.0.0-review`) |
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

This v9.0.0-review release is designed to stay read-only and simulation-only:

- **No wallet integration** — no private keys, mnemonics, or signing libraries.
- **No exchange execution** — no order placement, cancellation, or authenticated trading endpoints.
- **Info API only** — single public POST to `allMids`; no API secrets required.
- **Isolated backtests and experiments** — `BacktestPaperTrader` and `ExperimentPaperTrader` do not append to live CSV files.
- **Test guards** — pytest fixtures assert live CSV files are unchanged after the test run.
- **Review package CLI-only** — dashboard Review Mode is read-only; no export or execution actions.
- **Capital readiness is descriptive** — highest assessment tier means "present to adviser," not "approved to trade."

If you add wallet or live-trading code, the health check is expected to fail until those modules are removed or the check is intentionally updated for a future release.

---

## Known Limitations

- **Single signal input** — 1-minute rolling change only; no order book depth, funding rates, or multi-timeframe fusion.
- **Fixed watchlist** — BTC, ETH, SOL by default (configurable in `MARKET.watchlist`).
- **Simplified fills** — live paper trading and backtests apply configurable slippage and fees using the same rules; neither models full market microstructure.
- **Local data dependency** — backtests, experiments, and dashboard need prior price logging (run the live analyzer or supply `data/price_log.csv`).
- **Experiment scope** — one contrarian strategy family plus a hold baseline; no hyperparameter search or autonomous execution.
- **Review package scope** — excludes raw trading history by default; evidence summaries only.
- **Capital readiness thresholds** — conservative rules block progression on insufficient evidence, unexplained trades, or risk violations.

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
review_package.py      # Adviser review package generator
capital_readiness.py   # Descriptive capital-readiness assessment
review_check.py        # v9 release verification script
review_data.py         # Review data loading and sanitization
review_documents.py  # Review markdown generators
dashboard_review.py    # Dashboard Review Mode helpers
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

See [CHANGELOG.md](CHANGELOG.md) for the v9.0.0-review release summary.
