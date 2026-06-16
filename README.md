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