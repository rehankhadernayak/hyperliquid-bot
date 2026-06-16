# Hyperliquid Bot Project

## Goal

Build a safe, staged Hyperliquid trading system.

## Stages

1. Market scanner
2. Signal engine
3. Paper trader
4. Account monitor
5. Real order execution

## Current Stage

Stage 1: Market scanner

## Safety Rules

- No wallet yet
- No private keys
- No real trading
- No API secrets
- Read-only public market data only

## Configuration

All tunable settings live in **`config.py`** — edit that file before running the analyzer, backtester, or dashboard.

| Setting | Where in `config.py` |
|---------|----------------------|
| Coins to track, API URL, polling interval | `MARKET` |
| Signal thresholds (test vs production) | `SIGNAL` |
| Live paper trading limits (cash, trade size, exposure) | `PAPER_TRADING` |
| Backtest-only limits (max trades, fees, slippage) | `DEFAULT_BACKTEST` / `BacktestConfig` |
| CSV and JSON file paths | `DATA` |
| Live analyzer test mode on/off | `TEST_MODE` |

Live paper trading defaults to **10** session trades. Backtesting defaults to **1,000** session trades. Set `TEST_MODE = True` in `config.py` to use lower signal thresholds during live runs.

## Tests

Run the automated test suite (offline; no Hyperliquid API calls):

```bash
python3 -m pytest
```

## Dashboard

View logged prices and paper trade results in a simple read-only dashboard.

1. Install dependencies (if you have not already):

   ```bash
   pip install -r requirements.txt
   ```

2. Run the dashboard:

   ```bash
   streamlit run dashboard.py
   ```

The dashboard reads `data/price_log.csv` and `data/paper_trades.csv` (if it exists).
No wallet, signing, or real orders are used.